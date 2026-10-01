"""Hybrid MPI/OpenMP launches give every rank its own cores.

Open MPI binds each rank to one core by default on Linux, which squashes all of
a rank's OpenMP threads onto that core. Local launches leave ranks unbound;
SLURM sweeps map each rank onto its threads' cores (or let srun/ibrun do so).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frequensolve.orchestrator.sites.curvature import OPEN_MPI_UNBOUND
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.local import site as local_site
from tests.test_site_curvature_slurm import FakeCluster, FakeSlurmSite
from tests.test_slurm_site_refactor import DummySSHClientClass

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

_BINDING = (
    "PRTE_MCA_hwloc_default_binding_policy",
    "OMPI_MCA_hwloc_base_binding_policy",
)
_RECORDED = ("OMP_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND", *_BINDING)


# -- LocalSite.run_task ----------------------------------------------------------


class _Process:
    def wait(self):
        return 0


@pytest.fixture
def launches(monkeypatch):
    calls = []

    def popen(args, **kwargs):
        calls.append((list(args), dict(kwargs["env"])))
        return _Process()

    monkeypatch.setattr(local_site.subprocess, "Popen", popen)
    return calls


def _job_file(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"result_path": str(tmp_path / "results")}))
    return str(path)


def test_local_multi_rank_task_leaves_open_mpi_ranks_unbound(tmp_path, launches):
    env = {"PATH": "/usr/bin", "OMP_NUM_THREADS": "1"}

    local_site.run_task(
        _job_file(tmp_path), 0, "/opt/fs3d_s", env, n_ranks=2, n_threads=8
    )

    ((args, launched_env),) = launches
    assert args == [
        "mpirun",
        "-np",
        "2",
        "/opt/fs3d_s",
        "-nthreads",
        "4",
        "--job",
        str(tmp_path / "job.json"),
        "--task",
        "1",
    ]
    assert {name: launched_env[name] for name in _BINDING} == OPEN_MPI_UNBOUND
    assert env == {"PATH": "/usr/bin", "OMP_NUM_THREADS": "1"}  # Not mutated.


def test_local_launch_keeps_an_explicit_binding_policy(tmp_path, launches):
    env = {"OMPI_MCA_hwloc_base_binding_policy": "core"}

    local_site.run_task(
        _job_file(tmp_path), 0, "/opt/fs3d_s", env, n_ranks=4, n_threads=8
    )

    ((_, launched_env),) = launches
    assert launched_env["OMPI_MCA_hwloc_base_binding_policy"] == "core"
    assert launched_env["PRTE_MCA_hwloc_default_binding_policy"] == "none"


def test_local_single_rank_and_frequency_group_launches(tmp_path, launches):
    job = _job_file(tmp_path)

    local_site.run_task(job, 0, "/opt/fs3d_s", {}, n_ranks=1, n_threads=8)
    local_site.run_task(
        job, 0, "/opt/fs3d_s", {}, n_ranks=4, n_threads=8, frequency_groups=2
    )

    (single, single_env), (group, group_env) = launches
    assert single[:3] == ["/opt/fs3d_s", "-nthreads", "8"]
    assert not set(_BINDING) & set(single_env)
    assert group[:6] == ["mpirun", "-np", "4", "/opt/fs3d_s", "-nthreads", "2"]
    assert group[-2:] == ["--frequency-groups", "2"]
    assert {name: group_env[name] for name in _BINDING} == OPEN_MPI_UNBOUND


# -- SLURM adaptive sweep ---------------------------------------------------------

_SWEEP_SOLVER = """\
#!{python}
import json, os, sys
args = sys.argv[1:]
record = dict(argv=args, env={{name: os.environ.get(name) for name in {recorded!r}}})
with open(os.environ["FAKE_SWEEP_CALLS"], "a") as calls:
    calls.write(json.dumps(record) + "\\n")
if "--sizing" in args:
    tasks = int(os.environ["FAKE_SWEEP_TASKS"])
    with open(args[args.index("--sizing") + 1], "w") as sizing:
        json.dump({{"schema": "fs-sizing-2", "task": [{{"memory": "1 GB"}}] * tasks}}, sizing)
"""


def _run_sweep(
    tmp_path,
    monkeypatch,
    launcher,
    *,
    linux=True,
    launcher_args=(),
    n_tasks=2,
    task_ranks=None,
    bash="bash",
    environment=None,
):
    """Render and run a batch sweep with fake launchers; return its launches."""

    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    fake = FakeCluster(tmp_path, linux=linux)
    if not linux:
        (fake.bin / "uname").write_text("#!/bin/bash\necho Darwin\n")
        (fake.bin / "uname").chmod(0o755)
    solver = fake.install / "fs-solver"
    solver.write_text(_SWEEP_SOLVER.format(python=sys.executable, recorded=_RECORDED))
    solver.chmod(0o755)
    site = FakeSlurmSite(
        config=hpc.SlurmSiteConfig(
            hostname="login.example.edu",
            queue="debug",
            mpi_wrapper=str(fake.bin / launcher),
            launcher_args=tuple(launcher_args),
            max_nodes=1,
            cores_per_node=8,
            memory_per_node=4096,
        ),
        solver=str(solver),
        work_dir=str(tmp_path / "work"),
        solver_policy="off",
        environment=environment,
    )
    scheduler = (
        Path(hpc.__file__).parent / "templates" / "sweep" / "adaptive_scheduler.py"
    )
    monkeypatch.setattr(site, "_adaptive_scheduler_remote_path", lambda: scheduler)
    output = tmp_path / "logs"
    script = site._sweep_SLURM_script(
        n_tasks=n_tasks,
        n_job_tasks=n_tasks,
        task_indices=list(range(1, n_tasks + 1)),
        n_nodes=1,
        ranks_per_node=4,
        stdout=str(output),
        duration="00-00:10:00",
        run_path=str(tmp_path),
        postprocess_job=True,
        pack=False,
        launch_delay_seconds=0,
        **({"task_ranks": task_ranks} if task_ranks else {}),
    )
    sweep = tmp_path / "sweep.sh"
    sweep.write_text(script)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"project_path": str(tmp_path), "result_path": "r"}))
    calls = tmp_path / "solver-calls.jsonl"
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _RECORDED and not name.startswith(("OMPI_MCA_", "PRTE_MCA_"))
    }
    env.update(
        PATH=f"{fake.bin}{os.pathsep}{env['PATH']}",
        FAKE_SWEEP_CALLS=str(calls),
        FAKE_SWEEP_TASKS=str(n_tasks),
    )
    completed = subprocess.run(
        [bash, str(sweep), str(job)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    return script, fake.launch_lines(), records


def _step(records, flag):
    return [record for record in records if flag in record["argv"]]


def test_open_mpi_sweep_on_linux_maps_every_rank_onto_its_threads_cores(
    tmp_path, monkeypatch
):
    script, lines, records = _run_sweep(tmp_path, monkeypatch, "mpirun")

    placement = ["--map-by", "slot:PE=2", "--bind-to", "core"]
    init, *tasks, smooth = lines
    assert init[:3] == ["mpirun", "-n", "4"] and init[3:7] == placement
    assert smooth[:3] == ["mpirun", "-n", "4"] and smooth[3:7] == placement
    assert [line[:7] for line in tasks] == [["mpirun", "-n", "4", *placement]] * 2
    for record in records:
        assert record["env"]["OMP_NUM_THREADS"] == "2"
        assert (record["env"]["OMP_PLACES"], record["env"]["OMP_PROC_BIND"]) == (
            "cores",
            "close",
        )
        assert record["env"]["PRTE_MCA_hwloc_default_binding_policy"] is None
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --ntasks-per-node=4" in script


@pytest.mark.parametrize(("launcher", "linux"), [("mpirun", False), ("mpiexec", True)])
def test_other_mpirun_launchers_leave_ranks_unbound_without_thread_places(
    tmp_path, monkeypatch, launcher, linux
):
    _, lines, records = _run_sweep(tmp_path, monkeypatch, launcher, linux=linux)

    assert all("--bind-to" not in line and "--map-by" not in line for line in lines)
    for record in records:
        assert record["env"]["OMP_PLACES"] is None
        assert {name: record["env"][name] for name in _BINDING} == OPEN_MPI_UNBOUND


@pytest.mark.parametrize(
    ("launcher_args", "environment"),
    [
        (("--bind-to", "socket"), None),
        ((), {"OMPI_MCA_hwloc_base_binding_policy": "l3cache"}),
    ],
)
def test_explicit_mpirun_binding_settings_win(
    tmp_path, monkeypatch, launcher_args, environment
):
    _, lines, records = _run_sweep(
        tmp_path,
        monkeypatch,
        "mpirun",
        launcher_args=launcher_args,
        environment=environment,
    )

    assert all("--map-by" not in line for line in lines)
    for record in records:
        assert record["env"]["OMP_PLACES"] is None
        assert record["env"]["PRTE_MCA_hwloc_default_binding_policy"] is None


def test_srun_sweep_gives_every_step_its_threads_cores(tmp_path, monkeypatch):
    script, lines, _ = _run_sweep(tmp_path, monkeypatch, "srun")

    init, *tasks, smooth = lines
    assert init[:5] == ["srun", "-n", "4", "--cpus-per-task", "2"]
    assert smooth[:5] == ["srun", "-n", "4", "--cpus-per-task", "2"]
    assert all(line[:2] == ["srun", "--exclusive"] for line in tasks)
    assert all(line[line.index("--cpus-per-task") + 1] == "2" for line in tasks)
    assert "#SBATCH --cpus-per-task=2" in script


def test_ibrun_sweep_keeps_tacc_placement(tmp_path, monkeypatch):
    script, lines, _ = _run_sweep(tmp_path, monkeypatch, "ibrun")

    init, *tasks, smooth = lines
    assert init[:3] == ["ibrun", "-n", "4"] and "--cpus-per-task" not in init
    assert all("task_affinity" in line for line in tasks)
    assert "--cpus-per-task" not in script.split("set -euo pipefail")[0]


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="needs /bin/bash")
def test_sweep_runs_with_empty_launcher_arguments_under_set_u(tmp_path, monkeypatch):
    """bash < 4.4 (macOS /bin/bash) rejects "${empty[@]}" under ``set -u``."""

    _, lines, records = _run_sweep(
        tmp_path, monkeypatch, "srun", launcher_args=(), bash="/bin/bash"
    )

    assert len(lines) == 4 and len(_step(records, "--task")) == 2


def test_pinned_sweep_tasks_run_on_their_producer_ranks(tmp_path, monkeypatch):
    script, lines, records = _run_sweep(
        tmp_path, monkeypatch, "srun", n_tasks=3, task_ranks={1: 3, 2: 1, 3: 2}
    )

    assert '"skip_sizing": true' in script
    assert _step(records, "--init-no-size") and not _step(records, "--sizing")
    ranks = {
        int(line[-1]): int(line[line.index("--ntasks") + 1]) for line in lines[1:-1]
    }
    assert ranks == {1: 3, 2: 1, 3: 2}
    status = json.loads((tmp_path / "logs" / "scheduler_status.json").read_text())
    assert status["state"] == "complete"
    assert status["task_ranks"] == {"1": 3, "2": 1, "3": 2}


def test_sweep_drops_pins_that_do_not_fit_the_allocation(tmp_path, monkeypatch, capsys):
    _, lines, records = _run_sweep(
        tmp_path, monkeypatch, "srun", n_tasks=2, task_ranks={1: 9, 2: 2}
    )

    config = json.loads((tmp_path / "logs" / "scheduler_config.json").read_text())
    assert config["task_ranks"] == {"2": 2} and config["skip_sizing"] is False
    assert _step(records, "--sizing")
    assert "written on 9 MPI ranks" in capsys.readouterr().out
