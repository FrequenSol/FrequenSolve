"""SLURM curvature execution: remote staging, compute-node launch and fetch.

The fake cluster keeps its "remote" filesystem in a local directory. Login
commands run in a real shell that also prints Lmod-style notices on stderr;
``sbatch`` executes the generated script with fake MPI launchers (named like
the real ones) and a fake ``FS_seismic``, then (for end-to-end tests) runs the
SDK's oracle on the request the solver received, as Sauce would on a compute
node.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from frequensolve.imaging import BFGSHistory, NativeCurvature
from frequensolve.orchestrator.sites.hpc import SlurmRunConfig, SlurmSite
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.curvature import SlurmCurvatureRunner
from frequensolve.orchestrator.sites.hpc.site import SlurmSiteConfig
from tests.statistical_mesh_fixture import write_mesh
from tests.test_imaging_curvature import checkpoint
from tests.test_slurm_site_refactor import (
    DummyCredentials,
    DummyRawClient,
    DummySSHClientClass,
)

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

_SCHEDULER = ("sbatch", "squeue", "sacct", "scontrol", "scancel")
_NOTICE = b"Lmod: the following modules were reloaded\n"
_RECORDED = (
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "PRTE_MCA_hwloc_default_binding_policy",
)
_VERSIONS = {
    "srun": "slurm 23.11.1",
    "ibrun": "TACC ibrun",
    "mpirun": "mpirun (Open MPI) 5.0.8",
    "mpiexec": "HYDRA build details:",
}


class _Channel:
    def __init__(self, status):
        self.status = status

    def recv_exit_status(self):
        return self.status


class _Stream:
    def __init__(self, data, status):
        self.data = data if isinstance(data, bytes) else data.encode()
        self.channel = _Channel(status)

    def read(self):
        return self.data


def _streams(stdout, stderr, status):
    return None, _Stream(stdout, status), _Stream(stderr, status)


class FakeCluster:
    """Login node, scheduler and compute nodes backed by local directories."""

    def __init__(self, tmp_path, *, oracle=None, linux=False):
        self.tmp = tmp_path
        self.bin = tmp_path / "cluster-bin"
        self.install = tmp_path / "install"
        self.calls = tmp_path / "sauce-calls.jsonl"
        self.launches = tmp_path / "launches.txt"
        self.oracle = oracle
        self.login, self.compute, self.puts, self.gets = [], [], [], []
        self.jobs, self.directives, self.allocations = {}, {}, set()
        self.sauce_status = 0
        self.sauce_message = "ERROR: covariance factors are invalid"
        self.processed = 0
        self.bin.mkdir()
        self.install.mkdir()
        scripts = {"task_affinity": 'exec "$@"\n'}
        if linux:
            scripts["uname"] = "echo Linux\n"
        for name, version in _VERSIONS.items():
            # Each launcher records its command line, then starts the program.
            scripts[name] = (
                'if [ "${1:-}" = --version ]; then '
                f"echo {shlex.quote(version)}; exit 0; fi\n"
                f"printf '%s\\n' \"${{0##*/}} $*\" >> "
                f"{shlex.quote(str(self.launches))}\n"
                'while [ "$#" -gt 0 ]; do case "$1" in\n'
                "    -n|-o|--ntasks|--cpus-per-task|--map-by|--bind-to) shift 2 ;;\n"
                "    -*) shift ;;\n"
                "    *) break ;;\n"
                "esac; done\n"
                'exec "$@"\n'
            )
        for name, body in scripts.items():
            path = self.bin / name
            path.write_text("#!/bin/bash\n" + body)
            path.chmod(0o755)
        solver = self.install / "FS_seismic"
        solver.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "path = args[args.index('--curvature') + 1]\n"
            "request = json.load(open(path))\n"
            f"environment = {{name: os.environ.get(name) for name in {_RECORDED!r}}}\n"
            f"with open({str(self.calls)!r}, 'a') as calls:\n"
            "    record = dict(argv=args, path=path, request=request,"
            " environment=environment)\n"
            "    calls.write(json.dumps(record) + '\\n')\n"
            "print('fake Sauce', request['method'])\n"
            "status = int(os.environ.get('FAKE_SAUCE_STATUS', '0'))\n"
            "if status:\n"
            "    print(os.environ['FAKE_SAUCE_MESSAGE'], file=sys.stderr)\n"
            "    sys.exit(status)\n"
            "if os.environ.get('FAKE_SAUCE_WRITE'):\n"
            "    open(request['output'], 'w').write('output')\n"
        )
        solver.chmod(0o755)

    # -- shell ---------------------------------------------------------------

    def _environment(self, **extra):
        inherited = {
            name: value for name, value in os.environ.items() if name not in _RECORDED
        }
        return dict(
            inherited,
            PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            FAKE_SAUCE_STATUS=str(self.sauce_status),
            FAKE_SAUCE_MESSAGE=self.sauce_message,
            **extra,
        )

    def _shell(self, command, **extra):
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            env=self._environment(**extra),
        )
        return result.stdout, _NOTICE + result.stderr, result.returncode

    def run_login_cmd(self, command, *, timeout=None):
        words = shlex.split(command)
        if words[0] not in _SCHEDULER:
            self.login.append(command)
            return _streams(*self._shell(command))
        if words[0] == "sbatch":
            return _streams(f"Submitted batch job {self._sbatch(words)}\n", _NOTICE, 0)
        if words[0] == "squeue":
            job = words[words.index("-j") + 1]
            running = job in self.allocations
            return _streams("R\n" if running else "", _NOTICE, 0)
        if words[0] == "sacct":
            job = words[words.index("-j") + 1]
            state = "COMPLETED" if self.jobs.get(job) == 0 else "FAILED"
            return _streams(f"{state}\n", _NOTICE, 0)
        return _streams("", _NOTICE, 0)

    def _sbatch(self, words):
        job = str(100 + len(self.jobs))
        script = Path(words[-1])
        text = script.read_text()
        directives = dict(
            re.findall(r"^#SBATCH (-\w|--[\w-]+)[ =](.+)$", text, flags=re.MULTILINE)
        )
        self.directives[job] = dict(directives, slurm_args=words[1:-1])
        log = Path(directives["-o"].replace("%j", job))
        with open(log, "w") as stream:
            result = subprocess.run(
                ["bash", str(script)],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=self._environment(),
            )
        self.jobs[job] = result.returncode
        self._run_oracle()
        return job

    def _run_oracle(self):
        records = self.sauce_calls()
        for record in records[self.processed :]:
            if self.oracle is not None and not self.sauce_status:
                self.oracle(Path(record["path"]))
        self.processed = len(records)

    def run_compute_cmd(self, command):
        self.compute.append(command)
        return _streams(*self._shell(command, FAKE_SAUCE_WRITE="1"))

    # -- transfers -----------------------------------------------------------

    def put(self, local, remote, *, compress=True):
        self.puts.append((Path(local), Path(str(remote)), compress))
        Path(str(remote)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, str(remote))

    def get(self, remote, local, overwrite=False, *, compress=True):
        self.gets.append((Path(str(remote)), Path(local), compress))
        shutil.copyfile(str(remote), local)

    # -- inspection ----------------------------------------------------------

    def sauce_calls(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def launch_lines(self):
        return [line.split() for line in self.launches.read_text().splitlines()]

    def uploaded(self, pattern):
        return [remote for _, remote, _ in self.puts if re.search(pattern, str(remote))]


class FakeSlurmSite(SlurmSite):
    site_name = "FakeSlurm"
    credentials_cls = DummyCredentials
    default_queue = "debug"

    def authenticate(self, host=None):
        return DummyRawClient()


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    def build(
        oracle=None, launcher="srun", solver=None, cores=8, linux=False, **options
    ):
        fake = FakeCluster(tmp_path, oracle=oracle, linux=linux)
        config = SlurmSiteConfig(
            hostname="login.example.edu",
            queue="debug",
            mpi_wrapper=(
                launcher if os.path.isabs(launcher) else str(fake.bin / launcher)
            ),
            poll_interval=0,
            max_nodes=4,
            cores_per_node=cores,
            sockets_per_node=2,
        )
        site = FakeSlurmSite(
            config=config,
            solver=solver or str(fake.install / "FS_seismic"),
            work_dir=str(tmp_path / "remote" / "work"),
            solver_policy="off",
            run_config=SlurmRunConfig(poll_interval=0),
            **options,
        )
        for name in ("run_login_cmd", "run_compute_cmd", "put", "get"):
            setattr(site, name, getattr(fake, name))
        return site, fake

    return build


def _history(size=6, pairs=2):
    rng = np.random.default_rng(7)
    factor = rng.normal(size=(size, size))
    hessian = factor @ factor.T + size * np.eye(size)
    history = BFGSHistory(rng.uniform(0.5, 2.0, size), state="s", coordinates="c")
    steps = rng.normal(size=(pairs, size))
    history(
        checkpoint(
            [dict(step=s, difference=hessian @ s) for s in steps],
            pairs,
        )
    )
    return history


def _oracle(path):
    from tests.test_imaging_statistical_refinement import _mesh_runner

    _mesh_runner(path)


def test_history_operations_stage_once_and_reuse_remote_factors(cluster, tmp_path):
    site, fake = cluster(oracle=_oracle)
    native = NativeCurvature(
        workdir=tmp_path / "local" / "curvature", runner=site.run_curvature
    )
    history = _history()

    factors = native.bfgs_uncertainty(history, prior_std=0.5)
    root = Path(site.work_dir) / "curvature"
    (remote_root,) = root.iterdir()
    histories = fake.uploaded(r"/histories/")
    assert histories == [remote_root / "histories" / f"{history.digest}.h5"]
    operation = factors.path.parent.name
    remote_operation = remote_root / operation
    # Relative external links resolve inside the mirrored remote layout.
    with h5py.File(remote_operation / "input.h5", "r") as h5:
        np.testing.assert_array_equal(h5["steps"][()], history.steps)
    sent = json.loads((remote_operation / "request.json").read_text())
    assert sent["input"] == str(remote_operation / "input.h5")
    assert sent["output"] == str(remote_operation / "pending.h5")
    assert all(
        compress is False for local, _, compress in fake.puts if local.suffix == ".h5"
    )
    assert (factors.path.parent / "solver.log").read_text().startswith("fake Sauce")
    np.testing.assert_array_equal(
        factors.read("variance"),
        h5py.File(remote_operation / "pending.h5", "r")["variance"][()],
    )

    uploads = len(fake.puts)
    projection = csr_matrix(np.eye(6)[:2])
    projected = native.covariance(factors, projection=projection)
    again = native.bfgs_uncertainty(history, prior_std=0.5)

    later = [remote for _, remote, _ in fake.puts[uploads:]]
    assert not any(
        "histories" in str(path) or path.name == "result.h5" for path in later
    )
    remote_factors = json.loads(
        (remote_root / projected.path.parent.name / "request.json").read_text()
    )["factors"]
    assert remote_factors == str(remote_operation / "pending.h5")
    assert projected.read("variance").shape == (2,)
    assert again.metadata["history_pairs"] == 2

    # Every Sauce process ran inside a batch job, never in a login command.
    assert len(fake.jobs) == len(fake.sauce_calls()) == 3
    assert not any("--curvature" in command for command in fake.login)
    assert not any(str(fake.install) in command for command in fake.login)
    directives = fake.directives["100"]
    assert (directives["-N"], directives["-n"], directives["-p"]) == ("1", "2", "debug")
    assert fake.launch_lines()[0][:7] == [
        "srun",
        "--ntasks",
        "2",
        "--cpus-per-task",
        "4",
        str(fake.install / "FS_seismic"),
        "-nthreads",
    ]


def test_remote_covariance_reads_linked_warm_start_modes(cluster, tmp_path):
    from tests.test_imaging_curvature_transfer_wrappers import _linking_runner

    site, fake = cluster(oracle=_linking_runner([]))
    local = tmp_path / "local"
    native = NativeCurvature(workdir=local / "curvature", runner=site.run_curvature)
    identities = dict(state="s", coordinates="c")
    base, modes = [2.0, 3.0], [[1.0, 0.0], [0.0, 1.0]]
    refreshed = native.refresh_curvature(
        base, modes, [[4.0, 0.0], [0.0, 5.0]], **identities
    )
    warm = native.warm_start_curvature(base, modes, [0.5, 0.25], **identities)
    # Sauce linked the warm start's modes into its input, which stays local.
    assert sorted(p.parent for p in local.rglob("input.h5")) == [warm.path.parent]
    assert not (refreshed.path.parent / "input.h5").exists()

    uploads = len(fake.puts)
    applied = native.covariance(warm, vectors=[1.0, 2.0])
    # The factors and their linked input are reused remotely, not uploaded again.
    later = [remote for _, remote, _ in fake.puts[uploads:]]
    assert not any(path.parent.name == warm.path.parent.name for path in later)
    assert (
        applied.metadata["factors_digests"]["/modes"] == warm.output_digests["/modes"]
    )
    assert sorted(p.parent for p in local.rglob("input.h5")) == [warm.path.parent]


@pytest.mark.parametrize(
    ("launcher", "flags", "allocates", "placed"),
    [
        ("srun", ["--ntasks", "2", "--cpus-per-task", "4"], True, True),
        ("ibrun", ["-n", "2", "-o", "0", "task_affinity"], False, True),
        (
            "mpirun",
            ["-n", "2", "--map-by", "slot:PE=4", "--bind-to", "core"],
            True,
            True,
        ),
        ("mpiexec", ["-n", "2"], True, False),
    ],
)
def test_launchers_give_every_rank_its_own_cores(
    cluster, tmp_path, launcher, flags, allocates, placed
):
    """Hybrid ranks get T cores each; thread places only where ranks are bound."""

    site, fake = cluster(oracle=_write_output, launcher=launcher, linux=True)

    site.run_curvature(_manual_request(tmp_path))

    (line,) = fake.launch_lines()
    assert line[: len(flags) + 2] == [
        launcher,
        *flags,
        str(fake.install / "FS_seismic"),
    ]
    assert line[len(flags) + 2 : len(flags) + 4] == ["-nthreads", "4"]
    directives = fake.directives["100"]
    assert (directives["-N"], directives["-n"]) == ("1", "2")
    if allocates:
        assert directives["--ntasks-per-node"] == "2"
        assert directives["--cpus-per-task"] == "4"
    else:  # TACC's ibrun allocates whole nodes and places ranks itself.
        assert "--cpus-per-task" not in directives
    (record,) = fake.sauce_calls()
    environment = record["environment"]
    assert environment["OMP_NUM_THREADS"] == "4"
    if placed:
        assert (environment["OMP_PLACES"], environment["OMP_PROC_BIND"]) == (
            "cores",
            "close",
        )
    else:  # Unbound ranks must not pin threads onto the same cores.
        assert environment["OMP_PLACES"] is None
        assert environment["PRTE_MCA_hwloc_default_binding_policy"] == "none"


def test_open_mpi_ranks_stay_unbound_where_binding_is_unsupported(cluster, tmp_path):
    site, fake = cluster(oracle=_write_output, launcher="mpirun")

    site.run_curvature(_manual_request(tmp_path))

    (line,) = fake.launch_lines()
    assert line[:3] == ["mpirun", "-n", "2"] and "--bind-to" not in line
    environment = fake.sauce_calls()[0]["environment"]
    assert environment["OMP_PLACES"] is None
    assert environment["PRTE_MCA_hwloc_default_binding_policy"] == "none"


def test_explicit_thread_places_are_kept(cluster, tmp_path):
    site, fake = cluster(oracle=_write_output, environment={"OMP_PROC_BIND": "spread"})

    site.run_curvature(_manual_request(tmp_path))

    environment = fake.sauce_calls()[0]["environment"]
    assert (environment["OMP_PLACES"], environment["OMP_PROC_BIND"]) == (
        "cores",
        "spread",
    )


def test_partial_remote_history_is_uploaded_again(cluster, tmp_path):
    site, fake = cluster(oracle=_oracle)
    workdir = (tmp_path / "local" / "curvature").resolve()
    native = NativeCurvature(workdir=workdir, runner=site.run_curvature)
    history = _history()
    remote = Path(str(SlurmCurvatureRunner(site).remote_root(workdir)))
    truncated = remote / "histories" / f"{history.digest}.h5"
    truncated.parent.mkdir(parents=True)
    truncated.write_bytes(b"partial")

    local = workdir / "histories" / f"{history.digest}.h5"
    with native.retain(history):
        native.bfgs_uncertainty(history, prior_std=0.5)
        assert truncated.read_bytes() == local.read_bytes()
    assert fake.uploaded(r"/histories/") == [truncated]
    # The local copy is deleted after use; the complete remote mirror stays.
    assert not local.exists() and truncated.stat().st_size > len(b"partial")


def test_single_rank_mesh_methods_use_one_rank_and_stage_meshes(cluster, tmp_path):
    site, fake = cluster(oracle=_oracle)
    native = NativeCurvature(
        workdir=tmp_path / "local" / "curvature", runner=site.run_curvature
    )
    (tmp_path / "meshes").mkdir()
    coarse, _ = write_mesh(tmp_path / "meshes" / "coarse.h5")
    fine, _ = write_mesh(tmp_path / "meshes" / "fine.h5", refined=True)

    lifted = native.mesh_prior(coarse, fine, np.ones(4), np.ones(4))
    native.mesh_prior(coarse, fine, np.ones(4), np.ones(4))

    assert lifted.read("prior_std").shape == (6,)
    meshes = fake.uploaded(r"/external/")
    assert sorted(path.name for path in meshes) == ["coarse.h5", "fine.h5"]
    sent = fake.sauce_calls()[-1]["request"]
    assert {sent["source_mesh"], sent["target_mesh"]} == {str(path) for path in meshes}
    assert sent["dimension"] == 2
    for job in ("100", "101"):
        directives = fake.directives[job]
        assert (directives["-N"], directives["-n"]) == ("1", "1")
        assert directives["--cpus-per-task"] == "8"
    assert fake.sauce_calls()[0]["argv"][:2] == ["-nthreads", "8"]
    assert fake.launch_lines()[0][:5] == [
        "srun",
        "--ntasks",
        "1",
        "--cpus-per-task",
        "8",
    ]


def _manual_request(tmp_path, method="bfgs_rsvd"):
    operation = tmp_path / "local" / "curvature" / f"{method}-op"
    operation.mkdir(parents=True)
    with h5py.File(operation / "input.h5", "w") as h5:
        h5["metadata"] = np.bytes_("{}")
    request = operation / "request.json"
    request.write_text(
        json.dumps(
            dict(
                schema="fs-curvature-request-1",
                method=method,
                input=str(operation / "input.h5"),
                output=str(operation / "pending.h5"),
                state="s",
                coordinates="c",
            )
        )
    )
    return request


def _write_output(path):
    Path(json.loads(path.read_text())["output"]).write_text("ok")


def test_batch_resources_follow_overrides_and_curvature_run_config(cluster, tmp_path):
    site, fake = cluster(
        oracle=_write_output,
        curvature_run_config={"queue": "debug", "nodes": 2, "poll_interval": 0},
    )
    request = _manual_request(tmp_path)

    site.run_curvature(request)
    site.run_curvature(request, nodes=1, ranks_per_node=4, slurm_args=["--exclusive"])

    first, second = fake.directives["100"], fake.directives["101"]
    assert (first["-N"], first["-n"], first["--ntasks-per-node"]) == ("2", "4", "2")
    assert (second["-N"], second["-n"], second["slurm_args"]) == (
        "1",
        "4",
        ["--exclusive"],
    )
    assert (first["--cpus-per-task"], second["--cpus-per-task"]) == ("4", "2")
    threads = [record["argv"][1] for record in fake.sauce_calls()]
    assert threads == ["4", "2"]
    assert (request.parent / "pending.h5").read_text() == "ok"
    with pytest.raises(TypeError, match="Unexpected curvature option"):
        site.run_curvature(request, ranks=4)
    with pytest.raises(ValueError, match="Maximum number of nodes"):
        site.run_curvature(request, nodes=8)


def test_failure_raises_with_remote_log_tail_and_fetches_log(cluster, tmp_path):
    site, fake = cluster()
    fake.sauce_status = 3
    request = _manual_request(tmp_path, "covariance_action")

    with pytest.raises(RuntimeError) as error:
        site.run_curvature(request)

    message = str(error.value)
    assert "exit status 3" in message
    assert "ERROR: covariance factors are invalid" in message
    assert "solver.log" in message
    assert "predates" not in message
    assert "invalid" in (request.parent / "solver.log").read_text()
    assert not (request.parent / "pending.h5").exists()


def test_dispatcher_without_curvature_routing_gets_a_hint(cluster, tmp_path):
    site, fake = cluster()
    fake.sauce_status, fake.sauce_message = 1, "Job file not provided."

    with pytest.raises(RuntimeError, match="predates --curvature routing"):
        site.run_curvature(_manual_request(tmp_path))
    assert fake.jobs == {"100": 1}


def test_attached_allocation_runs_on_its_batch_host(cluster, tmp_path):
    site, fake = cluster()
    distributed = _manual_request(tmp_path, "rickett")
    single = _manual_request(tmp_path, "mesh_sample")
    with pytest.raises(RuntimeError, match="No active compute allocation"):
        site.run_curvature(distributed, mode="attached")
    site.pool.id = "77"
    fake.allocations.add("77")

    def attach():
        site._compute_client = DummySSHClientClass(DummyRawClient())
        site.pool.nhost, site.pool.nproc, site.pool.ncore = 2, 4, 16

    site._attach_compute_client = attach

    site.run_curvature(distributed)
    site.run_curvature(single)

    assert fake.jobs == {}
    assert all("nohup bash" in command for command in fake.compute)
    assert [record["argv"][1] for record in fake.sauce_calls()] == ["4", "8"]
    assert [line[:5] for line in fake.launch_lines()] == [
        ["srun", "--ntasks", "4", "--cpus-per-task", "4"],
        ["srun", "--ntasks", "1", "--cpus-per-task", "8"],
    ]
    assert (distributed.parent / "pending.h5").read_text() == "output"
    with pytest.raises(ValueError, match="pass mode='batch'"):
        site.run_curvature(distributed, nodes=2)


@pytest.mark.integration
def test_real_sauce_and_mpirun_behind_the_fake_cluster(cluster, tmp_path, monkeypatch):
    """The generated batch script runs real Sauce ranks on mirrored remote files."""

    solver = os.environ.get("FS_CURVATURE_SOLVER")
    launcher = shutil.which("mpirun")
    if not solver or launcher is None:
        pytest.skip("Set FS_CURVATURE_SOLVER and provide mpirun")
    # An inherited FS_SOLVER_PATH would redirect FS_seismic to another install.
    monkeypatch.delenv("FS_SOLVER_PATH", raising=False)
    site, fake = cluster(launcher=launcher, solver=solver, cores=4)
    remote = NativeCurvature(
        workdir=tmp_path / "local" / "curvature", runner=site.run_curvature
    )
    local = NativeCurvature(solver, workdir=tmp_path / "reference")
    history = _history(size=401, pairs=5)
    (tmp_path / "meshes").mkdir()
    coarse, _ = write_mesh(tmp_path / "meshes" / "coarse.h5")
    fine, _ = write_mesh(tmp_path / "meshes" / "fine.h5", refined=True)
    projection = csr_matrix(np.eye(401)[::40])

    factors = remote.bfgs_uncertainty(history, prior_std=0.5)
    projected = remote.covariance(factors, projection=projection)
    lifted = remote.mesh_prior(coarse, fine, np.ones(4), np.full(4, 0.5))
    expected = local.bfgs_uncertainty(history, prior_std=0.5)

    assert (factors.metadata["ranks"], factors.metadata["threads"]) == (2, 2)
    assert projected.metadata["ranks"] == 2
    assert (lifted.metadata["ranks"], lifted.metadata["threads"]) == (1, 4)
    assert not any(path.name == "result.h5" for _, path, _ in fake.puts)
    assert len(fake.uploaded(r"/histories/")) == 1
    np.testing.assert_allclose(
        factors.read("variance"), expected.read("variance"), rtol=1e-12
    )
    np.testing.assert_allclose(
        projected.read("variance"),
        local.covariance(expected, projection=projection).read("variance"),
        rtol=1e-12,
    )
