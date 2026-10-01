"""Tasks reading partitioned solver artifacts run on their producer's MPI ranks.

Sauce records the rank count of the task that saved an objective state,
objective vector, receiver state or background checkpoint, and rejects (or,
for backgrounds, recomputes) a consumer task on any other count. Sites pin
those tasks to the recorded count and size every other task as before.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.local.site import LocalSite
from frequensolve.orchestrator.sites.partition import (
    TaskPartition,
    partitioned_inputs,
    task_partitions,
)

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

_STATE = "fs-objective-linearization-3"


def _manifest(path, ranks=None, *, schema=_STATE, **extra):
    """Write one manifest as Sauce publishes it."""

    payload = {"schema": schema, **extra}
    if schema == "fs-background-state-1":
        payload["ranks"] = ranks
    elif ranks is not None:
        payload["partition"] = {
            "n_ranks": ranks,
            "compatibility": "same_mesh_partition",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class _Consumer:
    """The inputs of an ``FWIOperatorJob`` that Sauce partitions by rank."""

    workflow = "fwi_operator"
    name = "imaging_normal_0003"
    frequency_groups = 1
    max_ranks_per_task = None
    state = background = objective_vector = receiver_state = receiver_vector = None

    def __init__(self, project, action="normal", n_tasks=3, **inputs):
        self.project = Path(project)
        self.action = action
        self.n_tasks = n_tasks
        for name, relative in inputs.items():
            setattr(self, name, self.project / relative)

    def _project_path(self):
        return self.project

    def _remote_path(self, work_dir):
        return Path(work_dir) / "jobs" / "sim" / self.name


def _task_files(project, stem, ranks, **options):
    for task, count in ranks.items():
        _manifest(project / f"{stem}_{task}.json", count, **options)


def test_every_saved_linearization_input_pins_its_tasks(tmp_path):
    job = _Consumer(
        tmp_path,
        action="vjp",
        state="lin/state.json",
        objective_vector="ops/vjp.json",
        background="lin/background.json",
    )
    _task_files(tmp_path, "lin/state", {1: 4, 2: 2})
    _task_files(tmp_path, "ops/vjp", {1: 4, 2: 2}, schema="fs-objective-vector-3")
    _task_files(tmp_path, "lin/background", {3: 6}, schema="fs-background-state-1")

    assert task_partitions(job) == {
        1: TaskPartition(4, True, ("state", "objective_vector")),
        2: TaskPartition(2, True, ("state", "objective_vector")),
        3: TaskPartition(6, False, ("background",)),
    }
    assert task_partitions(job, [2]) == {
        2: TaskPartition(2, True, ("state", "objective_vector"))
    }


def test_writers_canonical_artifacts_and_other_jobs_impose_nothing(tmp_path):
    _task_files(tmp_path, "lin/state", {1: 4})
    _manifest(tmp_path / "canonical.json", schema="fs-objective-linearization-4")

    linearize = _Consumer(tmp_path, action="linearize", state="lin/state.json")
    canonical = _Consumer(tmp_path, n_tasks=1, state="canonical.json")
    other = _Consumer(tmp_path, state="lin/state.json")
    other.workflow = "forward"

    assert partitioned_inputs(linearize) == [] and task_partitions(linearize) == {}
    assert task_partitions(canonical) == {}
    assert partitioned_inputs(other) == []


def test_tasks_fall_back_to_the_named_file_like_sauce(tmp_path):
    """A multi-task job reads ``<stem>_<task>`` when it exists, else the stem."""

    _manifest(tmp_path / "shared.json", 3)
    _manifest(tmp_path / "shared_2.json", 5)
    job = _Consumer(tmp_path, n_tasks=2, state="shared.json")
    single = _Consumer(tmp_path, n_tasks=1, state="shared_2.json")

    assert {task: pin.ranks for task, pin in task_partitions(job).items()} == {
        1: 3,
        2: 5,
    }
    assert task_partitions(single)[1].ranks == 5


def test_receiver_bundles_pin_through_their_first_group(tmp_path):
    member = _manifest(
        tmp_path / "rcv" / "state_1_group_a.json", 3, schema="fs-receiver-state-1"
    )
    bundle = {
        "schema": "fs-receiver-state-bundle-1",
        # Sauce names members by the absolute path it wrote them to.
        "groups": {"0": {"file": f"/remote/elsewhere/{member.name}"}},
    }
    (tmp_path / "rcv" / "state_1.json").write_text(json.dumps(bundle))
    job = _Consumer(
        tmp_path, action="receiver_vjp", n_tasks=1, receiver_state="rcv/state_1.json"
    )

    assert task_partitions(job) == {1: TaskPartition(3, True, ("receiver_state",))}


# -- SLURM ---------------------------------------------------------------------------


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    from tests.test_site_curvature_slurm import FakeCluster, FakeSlurmSite
    from tests.test_slurm_site_refactor import DummySSHClientClass

    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    fake = FakeCluster(tmp_path)
    site = FakeSlurmSite(
        config=hpc.SlurmSiteConfig(
            hostname="login.example.edu",
            queue="debug",
            mpi_wrapper=str(fake.bin / "srun"),
            max_nodes=4,
            cores_per_node=8,
            memory_per_node=4096,
        ),
        solver=str(fake.install / "FS_seismic"),
        work_dir=str(tmp_path / "remote" / "work"),
        solver_policy="off",
    )
    site.run_login_cmd = fake.run_login_cmd
    return site, fake


def test_slurm_site_reads_unfetched_manifests_from_the_remote_project(slurm, tmp_path):
    site, fake = slurm
    project = tmp_path / "project"
    remote = Path(site.work_dir)
    job = _Consumer(
        project,
        action="vjp",
        state="lin/state.json",
        objective_vector="ops/vjp.json",
    )
    _task_files(project, "lin/state", {1: 4})  # Fetched with the linearize job.
    _task_files(remote, "lin/state", {2: 2})
    _manifest(remote / "lin" / "state.json", 6)  # Task 3 falls back to the stem.
    _task_files(remote, "ops/vjp", {1: 4, 2: 2, 3: 6}, schema="fs-objective-vector-3")

    pins = site._task_partitions(job, [1, 2, 3])

    assert {task: pin.ranks for task, pin in pins.items()} == {1: 4, 2: 2, 3: 6}
    assert len(fake.login) == 1  # One command for every unfetched manifest.


def test_slurm_site_resolves_remote_receiver_bundles(slurm, tmp_path):
    site, fake = slurm
    project = tmp_path / "project"
    remote = Path(site.work_dir)
    member = _manifest(
        remote / "rcv" / "state_1_a.json", 2, schema="fs-receiver-state-1"
    )
    _manifest(
        remote / "rcv" / "state_1.json",
        schema="fs-receiver-state-bundle-1",
        groups={"0": {"file": member.name}},  # Neither file was fetched.
    )
    job = _Consumer(
        project, action="receiver_jvp", n_tasks=2, receiver_state="rcv/state.json"
    )

    pins = site._task_partitions(job, [1])

    assert pins == {1: TaskPartition(2, True, ("receiver_state",))}
    assert len(fake.login) == 2  # The bundle, then its group file.


def _batch(site, monkeypatch):
    seen = {}

    def script(**kwargs):
        seen.update(kwargs)
        return "#!/bin/bash\n"

    monkeypatch.setattr(site, "_sweep_SLURM_script", script)
    monkeypatch.setattr(site, "_transfer_SLURM_job", lambda s, j: ("a.sh", "b.json"))
    monkeypatch.setattr(
        site, "_submit_sbatch", lambda command: seen.setdefault("sbatch", "42")
    )
    return seen


def test_slurm_batch_pins_every_linearization_consumer(slurm, tmp_path, monkeypatch):
    site, _ = slurm
    job = _Consumer(tmp_path / "project", state="lin/state.json")
    _task_files(job.project, "lin/state", {1: 4, 2: 2, 3: 1})
    seen = _batch(site, monkeypatch)

    site._submit_slurm_batch(
        job,
        hpc.SlurmRunConfig(nodes=1, ranks_per_node=4),
        task_plan={"pending_indices": [1, 2]},
    )

    assert seen["task_indices"] == [2, 3]
    assert {task: pin.ranks for task, pin in seen["task_ranks"].items()} == {2: 2, 3: 1}


def test_required_pins_that_do_not_fit_fail_before_submission(
    slurm, tmp_path, monkeypatch
):
    site, _ = slurm
    job = _Consumer(tmp_path / "project", n_tasks=2, state="lin/state.json")
    _task_files(job.project, "lin/state", {1: 2, 2: 16})

    with pytest.raises(ValueError, match="Task 2 reads state written on 16 MPI ranks"):
        site._submit_slurm_batch(
            job,
            hpc.SlurmRunConfig(nodes=1, ranks_per_node=4),
            task_plan={"pending_indices": [0, 1]},
        )
    # A background-only pin is merely dropped: Sauce recomputes the fields.
    assert site._fitting_task_ranks(
        {2: TaskPartition(16, False, ("background",)), 1: 2}, [1, 2], limit=4
    ) == {1: 2}


def test_attached_sweeps_need_one_producer_rank_count(slurm, tmp_path, capsys):
    site, _ = slurm
    site.pool.nproc = 8
    job = _Consumer(tmp_path / "project", n_tasks=2, state="lin/state.json")

    assert site._attached_task_ranks(job, 2) == 4  # nproc // n_tasks.
    _task_files(job.project, "lin/state", {1: 3, 2: 3})
    assert site._attached_task_ranks(job, 2) == 3
    _task_files(job.project, "lin/state", {2: 2})
    with pytest.raises(ValueError, match="mode='batch'"):
        site._attached_task_ranks(job, 2)

    reuse = _Consumer(tmp_path / "other", n_tasks=2, background="lin/background.json")
    _task_files(
        reuse.project, "lin/background", {1: 3, 2: 2}, schema="fs-background-state-1"
    )
    assert site._attached_task_ranks(reuse, 2) == 4
    assert "keep background reuse" in capsys.readouterr().out


# -- LocalSite -------------------------------------------------------------------------


class _Future:
    def result(self):
        return {"status": "success"}

    def release(self):
        pass


def _local_submissions(tmp_path, monkeypatch, job, **options):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/opt/fs2d_s")
    site = LocalSite()
    job._file = tmp_path / "job.json"
    job._file.write_text("{}")
    job._stdout_path = tmp_path / "logs"
    job.task_run_plan = lambda **_: {"pending_indices": list(range(job.n_tasks))}
    submitted = []

    class Client:
        def submit(self, function, job_file, task, *args, **kwargs):
            submitted.append((task, kwargs["n_ranks"]))
            return _Future()

    monkeypatch.setattr(site, "_ensure_dask_for_tasks", lambda count: None)
    monkeypatch.setattr(site, "_dask_client_or_raise", lambda: Client())
    monkeypatch.setattr(site, "_current_threads_per_worker", lambda: 8)
    site._submit_local_tasks(job, **options)
    return submitted


def test_local_site_runs_linearization_consumers_on_their_producer_ranks(
    tmp_path, monkeypatch
):
    job = _Consumer(tmp_path, state="lin/state.json", background="lin/background.json")
    _task_files(tmp_path, "lin/state", {1: 4, 2: 1})

    submitted = _local_submissions(tmp_path, monkeypatch, job, procs_per_job=2)

    # The mesh step, then tasks 3, 2, 1 (task 3 has no saved state).
    assert submitted == [(-1, 1), (2, 2), (1, 1), (0, 4)]

    job.max_ranks_per_task = 2
    with pytest.raises(ValueError, match="written on 4 MPI ranks"):
        _local_submissions(tmp_path, monkeypatch, job, procs_per_job=2)


def test_local_frequency_group_launch_uses_the_producer_rank_count(
    tmp_path, monkeypatch
):
    job = _Consumer(tmp_path, action="solve", n_tasks=2, state="lin/state.json")
    job.frequency_groups = 2
    _task_files(tmp_path, "lin/state", {1: 3, 2: 3})

    submitted = _local_submissions(tmp_path, monkeypatch, job, procs_per_job=1)

    assert submitted == [(-1, 1), (0, 6)]  # Two groups of three ranks.
    _task_files(tmp_path, "lin/state", {2: 2})
    with pytest.raises(ValueError, match="different MPI rank counts"):
        _local_submissions(tmp_path, monkeypatch, job, procs_per_job=1)


# -- end to end through the adaptive sweep ----------------------------------------------

_LAUNCHER = """\
#!/bin/bash
size=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--ntasks) size=$2; shift 2 ;;
        --cpus-per-task) shift 2 ;;
        -*) shift ;;
        *) break ;;
    esac
done
FAKE_WORLD_SIZE=$size exec "$@"
"""

# Emulates Sauce: linearize saves a partitioned state; later actions reject
# a task whose MPI rank count differs from the one that saved its state.
_SOLVER = """\
#!{python}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
job = json.loads(Path(args[args.index("--job") + 1]).read_text())
results = Path(job["project_path"]) / "lin"
if "--sizing" in args:
    tasks = [dict(memory=f"{{gib}} GB") for gib in job["memory_gib"]]
    with open(args[args.index("--sizing") + 1], "w") as sizing:
        json.dump(dict(schema="fs-sizing-2", task=tasks), sizing)
if "--task" not in args:
    sys.exit(0)
task, ranks = int(args[args.index("--task") + 1]), int(os.environ["FAKE_WORLD_SIZE"])
state = results / f"state_{{task}}.json"
if job["action"] == "linearize":
    results.mkdir(exist_ok=True)
    partition = dict(n_ranks=ranks, compatibility="same_mesh_partition")
    state.write_text(json.dumps(dict(schema="{schema}", partition=partition)))
elif json.loads(state.read_text())["partition"]["n_ranks"] != ranks:
    print("Error: Objective state requires the saved mesh partition and MPI rank count")
    sys.exit(1)
"""


def test_adaptive_sweep_breaks_unpinned_consumers_and_pinning_fixes_them(
    tmp_path, monkeypatch
):
    """Sizing gives a consumer other ranks than its linearize task; pins match."""

    from tests.test_site_curvature_slurm import FakeSlurmSite
    from tests.test_slurm_site_refactor import DummySSHClientClass

    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "srun").write_text(_LAUNCHER)
    (bin_dir / "srun").chmod(0o755)
    solver = tmp_path / "fs-solver"
    solver.write_text(_SOLVER.format(python=sys.executable, schema=_STATE))
    solver.chmod(0o755)
    site = FakeSlurmSite(
        config=hpc.SlurmSiteConfig(
            hostname="login.example.edu",
            queue="debug",
            mpi_wrapper=str(bin_dir / "srun"),
            max_nodes=1,
            cores_per_node=8,
            memory_per_node=4096,  # 1 GiB per rank with four ranks per node.
        ),
        solver=str(solver),
        work_dir=str(tmp_path / "work"),
        solver_policy="off",
    )
    scheduler = (
        Path(hpc.__file__).parent / "templates" / "sweep" / "adaptive_scheduler.py"
    )
    monkeypatch.setattr(site, "_adaptive_scheduler_remote_path", lambda: scheduler)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def sweep(label, action, memory_gib, task_ranks=None):
        job = tmp_path / f"{label}.json"
        job.write_text(
            json.dumps(
                dict(
                    project_path=str(tmp_path),
                    result_path="r",
                    action=action,
                    memory_gib=memory_gib,
                )
            )
        )
        script = tmp_path / f"{label}.sh"
        script.write_text(
            site._sweep_SLURM_script(
                n_tasks=2,
                n_job_tasks=2,
                task_indices=[1, 2],
                n_nodes=1,
                ranks_per_node=4,
                stdout=str(tmp_path / label),
                duration="00-00:10:00",
                run_path=str(tmp_path),
                pack=False,
                launch_delay_seconds=0,
                **({"task_ranks": task_ranks} if task_ranks else {}),
            )
        )
        subprocess.run(
            ["bash", str(script), str(job)],
            cwd=tmp_path,
            env=env,
            check=True,
            timeout=120,
        )
        return json.loads((tmp_path / label / "scheduler_status.json").read_text())

    produced = sweep("linearize", "linearize", [1, 1])
    unpinned = sweep("normal", "normal", [3, 3])
    consumer = _Consumer(tmp_path, n_tasks=2, state="lin/state.json")
    pinned = sweep("pinned", "normal", [3, 3], site._task_partitions(consumer, [1, 2]))

    assert produced["state"] == "complete" and produced["task_ranks"] == {
        "1": 2,
        "2": 2,
    }
    assert unpinned["task_ranks"] == {"1": 4, "2": 4}
    assert unpinned["failed_tasks"] and all(
        "saved mesh partition" in reason
        for reason in unpinned["failed_reasons"].values()
    )
    assert pinned["state"] == "complete" and pinned["task_ranks"] == {"1": 2, "2": 2}
