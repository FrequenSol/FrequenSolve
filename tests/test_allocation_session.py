"""Persistent sessions exercised with a local, asynchronous fake cluster."""

import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from frequensolve.orchestrator.sites.base import RunFailedError
from frequensolve.orchestrator.sites.hpc import AdaptiveWorkers, SlurmRunConfig
from frequensolve.orchestrator.sites.hpc import session as session_module
from frequensolve.orchestrator.sites.hpc import site as hpc
from tests.test_slurm_site_refactor import DummySlurmSite, DummySSHClientClass

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic, pytest.mark.timeout(30)]


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out waiting for fake cluster")


def published_request(session, run_id):
    """Read a request whether or not the controller archived it yet."""
    pending = session.root / "requests" / f"{run_id}.json"
    processed = session.root / "requests" / "processed" / f"{run_id}.json"

    def read():
        for path in (processed, pending, processed):
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                continue
        return None

    return wait_until(read)


class Job:
    supports_trace_packing = True
    max_ranks_per_task = None
    postprocess_only = False

    def __init__(self, root, name, tasks=2, *, smooth=False, **solver):
        self.name = name
        self.n_tasks = tasks
        self.root = root / "jobs" / name
        self.root.mkdir(parents=True)
        self.job_file = self.root / "job.json"
        self.job_file.write_text(
            json.dumps(dict(tasks=tasks, events=str(root / "events.jsonl"), **solver))
        )
        self.smooth = smooth
        self._job_id = None

    def requires_postprocess(self):
        return self.smooth

    def task_run_plan(self, **kwargs):
        return {"pending_indices": list(range(self.n_tasks))}

    def save_for_remote(self, *args):
        return self.job_file, self.job_file

    def write_run_state(self, **kwargs):
        self.last_state = kwargs

    def latest_run(self, site=None):
        return getattr(self, "record", None)


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mpi = bin_dir / "ibrun"
    mpi.write_text(
        f"""#!{sys.executable}
import os, sys
# Like TACC's ibrun, re-split arguments on whitespace: shell code breaks.
args = ' '.join(sys.argv[1:]).split()
ranks, offset = '1', '0'
while args and args[0].startswith('-'):
    option = args.pop(0)
    if option in ('-n', '-o', '--nodes', '--ntasks', '--cpus-per-task'):
        value = args.pop(0)
        if option in ('-n', '--ntasks'): ranks = value
        if option == '-o': offset = value
if args[0] == 'task_affinity': args.pop(0)
os.environ['FAKE_RANKS'] = ranks
os.environ['FAKE_OFFSET'] = offset
os.execv(args[0], args)
"""
    )
    mpi.chmod(0o755)
    solver = bin_dir / "solver"
    solver.write_text(
        f"""#!{sys.executable}
import json, os, sys, time
args = sys.argv[1:]
if '--curvature' in args:
    request = json.load(open(args[args.index('--curvature')+1]))
    print('curvature')
    open(request['output'], 'w').write('curvature output')
    sys.exit(0)
job = json.load(open(args[args.index('--job')+1]))
step = next((s for s in ('init', 'init-no-size', 'smooth', 'pack') if '--'+s in args), None)
if '--task' in args: step = int(args[args.index('--task')+1])
if '--sizing' in args:
    json.dump({{'schema':'fs-sizing-2','task':[{{'memory':job.get('memory','32 MB')}}]*job['tasks']}}, open(args[args.index('--sizing')+1],'w'))
row = dict(step=step, pid=os.getpid(), ranks=int(os.environ['FAKE_RANKS']), offset=int(os.environ['FAKE_OFFSET']), gpu=os.environ.get('FS_GPU_DEVICE'), job=args[args.index('--job')+1], event='start', at=time.time())
with open(job['events'], 'a') as f: f.write(json.dumps(row)+'\\n')
time.sleep(job.get('sleep', 0.03))
row.update(event='end', at=time.time())
with open(job['events'], 'a') as f: f.write(json.dumps(row)+'\\n')
if step == job.get('fail'):
    print('Error: requested solver failure')
    sys.exit(3)
print('solver success')
"""
    )
    solver.chmod(0o755)
    site = DummySlurmSite(
        work_dir=tmp_path,
        solver=solver,
        modules=[],
        run_config=SlurmRunConfig(nodes=1, ranks_per_node=4, poll_interval=0),
    )
    site.config.mpi_wrapper = str(mpi)
    processes, cancellations, uploads = [], [], []

    def put(local, remote, **kwargs):
        remote = Path(remote)
        remote.parent.mkdir(parents=True, exist_ok=True)
        if Path(local) != remote:
            shutil.copyfile(local, remote)
        uploads.append(remote)

    def read(path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    def login(command, **kwargs):
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    def submit(command):
        script = Path(shlex.split(command)[-1])
        process = subprocess.Popen(
            ["bash", str(script)],
            stdout=(script.parent / "test-controller.log").open("w"),
            stderr=subprocess.STDOUT,
            env=dict(os.environ, SLURM_JOB_ID=str(len(processes) + 100)),
            start_new_session=True,
        )
        processes.append(process)
        return str(len(processes) + 99)

    def cancel(job_id):
        cancellations.append(job_id)
        process = processes[int(job_id) - 100]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def get(remote, local, **kwargs):
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote, local)

    monkeypatch.setattr(site, "put", put)
    monkeypatch.setattr(site, "get", get)
    monkeypatch.setattr(site, "_read_remote_json", read)
    monkeypatch.setattr(site, "_run_session_command", login)
    monkeypatch.setattr(site, "_submit_sbatch", submit)
    monkeypatch.setattr(site, "cancel_job", cancel)
    # squeue: this user's live session allocations.
    monkeypatch.setattr(
        site,
        "_session_allocations",
        lambda: {
            str(index + 100): {"state": "running", "time_left": 3600.0}
            for index, process in enumerate(processes)
            if process.poll() is None
        },
    )
    monkeypatch.setattr(
        site,
        "update_status",
        lambda job_id: (
            "running" if processes[int(job_id) - 100].poll() is None else "complete"
        ),
    )
    monkeypatch.setattr(site, "check_solver_compatibility", lambda **kwargs: None)
    monkeypatch.setattr(site, "prepare_job", lambda job, **kwargs: None)
    monkeypatch.setattr(site, "is_run_current", lambda job: False)
    monkeypatch.setattr(site, "_transfer_remote_simulation_inputs", lambda job: None)
    monkeypatch.setattr(site, "_task_partitions", lambda job, tasks: {})
    monkeypatch.setattr(site, "_record_site_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(site, "_finalize_run_record", lambda *args: None)
    value = SimpleNamespace(
        site=site,
        root=tmp_path,
        processes=processes,
        cancellations=cancellations,
        uploads=uploads,
    )
    yield value
    for index, process in enumerate(processes):
        if process.poll() is None:
            cancel(str(index + 100))


def test_one_allocation_runs_multiple_jobs_and_postprocessing(cluster):
    with cluster.site.session(workers=AdaptiveWorkers(max_ranks_per_task=2)) as session:
        for i in range(2):
            result = session.run(
                Job(cluster.root, f"gradient-{i}", smooth=True), check=True
            )
            assert result.successful
            steps = [row["step"] for row in result.status.raw["steps"]]
            assert steps[0] == "init"
            assert sorted(step for step in steps if isinstance(step, int)) == [1, 2]
            assert steps[-2:] == ["smooth", "pack"]
        smooth = Job(cluster.root, "smooth-only", smooth=True)
        result = session.submit(smooth, postprocess_only=True).wait(check=True)
        assert [row["step"] for row in result.status.raw["steps"]] == ["smooth", "pack"]
        assert session.status()["state"] == "ready"
        assert len(cluster.processes) == 1
    # The drained controller ends its batch job; no scancel is needed.
    assert cluster.processes[0].poll() == 0
    assert cluster.cancellations == []
    session.close()
    assert cluster.cancellations == []


def test_parallel_jobs_share_rank_offsets(cluster):
    with cluster.site.session() as session:
        handles = [
            session.submit(Job(cluster.root, f"normal-{i}", sleep=0.1))
            for i in range(3)
        ]
        assert all(
            result.successful for result in session.wait_all(handles, check=True)
        )
        events = [
            json.loads(line)
            for line in (cluster.root / "events.jsonl").read_text().splitlines()
        ]
        active = {}
        concurrent_tasks = False
        for row in sorted(events, key=lambda row: row["at"]):
            if row["event"] == "end":
                active.pop(row["pid"])
            else:
                slots = set(range(row["offset"], row["offset"] + row["ranks"]))
                assert not any(slots & other for other in active.values())
                active[row["pid"]] = slots
                concurrent_tasks |= len(active) > 1
        assert concurrent_tasks
        assert not active
        assert len(cluster.processes) == 1


def test_cancel_one_run_keeps_allocation_and_other_run(cluster):
    with cluster.site.session() as session:
        first = session.submit(Job(cluster.root, "cancel-me", sleep=5))
        wait_until(lambda: first.status().is_running)
        first.cancel()
        assert first.wait(check=False).status.state == "cancelled"
        assert session.run(Job(cluster.root, "survivor"), check=True).successful
        assert not cluster.cancellations


def test_failure_is_reported_and_owned_allocation_is_released(cluster):
    with pytest.raises(RunFailedError, match="requested solver failure"):
        with cluster.site.session() as session:
            session.run(Job(cluster.root, "failure", fail=1), check=True)
    # The cancelled controller ended its batch job before close() returned.
    assert cluster.processes[0].poll() is not None


def test_borrowed_session_reconnect_does_not_release_allocation(cluster):
    with cluster.site.session() as owner:
        first = owner.submit(Job(cluster.root, "first"))
        with cluster.site.attach_session(owner.session_id) as borrowed:
            assert not borrowed.owned
            assert borrowed.handle(first.job, first.id).wait(check=True).successful
        assert not cluster.cancellations
        assert owner.run(Job(cluster.root, "second")).successful


def test_requests_are_idempotent_and_conflicting_ids_fail(cluster):
    with cluster.site.session() as session:
        job = Job(cluster.root, "dedup")
        first = session.submit(job, request_id="a" * 32)
        payload = published_request(session, "a" * 32)
        # Repeat publication as would happen after losing an acknowledgment.
        session._publish(
            session.root / "requests" / ("a" * 32 + ".json"), payload, immutable=True
        )
        assert first.wait(check=True).successful
        payload["pack"] = False
        with pytest.raises(ValueError, match="different work"):
            session._publish(
                session.root / "requests" / ("a" * 32 + ".json"),
                payload,
                immutable=True,
            )
        events = [
            json.loads(line)
            for line in (cluster.root / "events.jsonl").read_text().splitlines()
        ]
        assert sum(row["step"] == 1 and row["event"] == "start" for row in events) == 1
        # Republishing after the controller archived the request is still
        # idempotent and still rejects different work.
        wait_until(
            lambda: (
                session.root / "requests" / "processed" / ("a" * 32 + ".json")
            ).is_file()
        )
        payload["pack"] = True
        session._publish(
            session.root / "requests" / ("a" * 32 + ".json"), payload, immutable=True
        )
        payload["pack"] = False
        with pytest.raises(ValueError, match="different work"):
            session._publish(
                session.root / "requests" / ("a" * 32 + ".json"),
                payload,
                immutable=True,
            )
        assert not (session.root / "requests" / ("a" * 32 + ".json")).exists()


def test_curvature_uses_the_session(cluster):
    directory = cluster.root / "local" / "curvature" / "op"
    directory.mkdir(parents=True)
    source = directory / "input.h5"
    with h5py.File(source, "w") as h5:
        h5["x"] = [1.0]
    request = directory / "request.json"
    output = directory / "result.h5"
    request.write_text(
        json.dumps(dict(method="mesh_sample", input=str(source), output=str(output)))
    )
    with cluster.site.session() as session:
        session.run_curvature(request)
        assert output.read_text() == "curvature output"
        assert (directory / "solver.log").read_text() == "curvature\n"
        assert len(cluster.processes) == 1


def test_explicit_session_rejects_batch_fallback_and_resource_changes(cluster):
    with cluster.site.session() as session:
        job = Job(cluster.root, "invalid")
        with pytest.raises(ValueError, match="batch"):
            session.submit(job, mode="batch")
        with pytest.raises(ValueError, match="nodes"):
            session.submit(job, nodes=2)
        assert len(cluster.processes) == 1


def test_controller_loss_fails_handles_without_new_batch_job(cluster):
    with cluster.site.session(cleanup_timeout=1) as session:
        handle = session.submit(Job(cluster.root, "never", sleep=5))
        cluster.processes[0].kill()
        cluster.processes[0].wait()
        # Slurm's terminal state is authoritative regardless of the heartbeat.
        assert handle.wait(check=False).status.state == "failed"
        assert len(cluster.processes) == 1


def test_dead_client_lease_closes_idle_allocation(cluster):
    session = cluster.site.session(lease_timeout=0.5)
    session.wait_ready()
    session._stop.set()  # Simulate client disappearance without a close command.
    wait_until(lambda: cluster.processes[0].poll() is not None)
    assert session.status()["reason"] == "client lease expired"
    session.close()


def test_walltime_signal_marks_work_timed_out(cluster):
    import signal

    with cluster.site.session() as session:
        handle = session.submit(Job(cluster.root, "walltime", sleep=5))
        wait_until(lambda: handle.status().is_running)
        cluster.processes[0].send_signal(signal.SIGUSR1)
        result = handle.wait(check=False)
        assert result.status.state == "timeout"
        wait_until(lambda: cluster.processes[0].poll() is not None)
        assert session.status()["reason"] == "walltime"


def test_startup_timeout_releases_pending_allocation(cluster, monkeypatch):
    monkeypatch.setattr(cluster.site, "update_status", lambda job_id: "pending")
    # Simulate an allocation that never starts its controller.
    original = cluster.site._submit_sbatch

    def pending(command):
        result = original(command)
        cluster.processes[-1].kill()
        cluster.processes[-1].wait()
        (Path(shlex.split(command)[-1]).parent / "status.json").unlink(missing_ok=True)
        return result

    monkeypatch.setattr(cluster.site, "_submit_sbatch", pending)
    with pytest.raises(TimeoutError, match="readiness"):
        with cluster.site.session(startup_timeout=0.1, cleanup_timeout=0.1):
            pytest.fail("controller never became ready")
    assert cluster.cancellations == ["100"]


def test_invalid_generation_fails_request_without_poisoning_controller(cluster):
    with cluster.site.session() as session:
        run_id = "b" * 32
        session._publish(
            session.root / "requests" / f"{run_id}.json",
            dict(schema="fs-session-request-1", generation="stale"),
            immutable=True,
        )
        status_file = session.root / "runs" / run_id / "status.json"
        status = wait_until(lambda: cluster.site._read_remote_json(status_file))
        assert status["state"] == "failed"
        assert "generation mismatch" in status["reason"]
        assert session.run(Job(cluster.root, "valid")).successful


def test_oversized_task_fails_with_actionable_error(cluster):
    with cluster.site.session(workers=AdaptiveWorkers(max_ranks_per_task=1)) as session:
        with pytest.raises(
            RunFailedError, match="increase allocation or worker rank limit"
        ):
            session.run(Job(cluster.root, "oversized", memory="2 GB"), check=True)
        assert session.status()["state"] == "ready"


def test_legacy_rank_pins_are_preserved(cluster, monkeypatch):
    from frequensolve.orchestrator.sites.partition import TaskPartition

    monkeypatch.setattr(
        cluster.site,
        "_task_partitions",
        lambda job, tasks: {
            1: TaskPartition(2, True, ("state",)),
            2: TaskPartition(1, True, ("state",)),
        },
    )
    with cluster.site.session(workers=AdaptiveWorkers(max_ranks_per_task=2)) as session:
        result = session.run(Job(cluster.root, "pinned"), check=True)
        assert result.status.raw["task_ranks"] == {"1": 2, "2": 1}
        assert len(cluster.processes) == 1


def test_same_output_directory_cannot_run_twice_concurrently(cluster):
    with cluster.site.session() as session:
        job = Job(cluster.root, "same-job", sleep=1)
        first = session.submit(job)
        wait_until(lambda: first.status().is_running)
        payload = published_request(session, first.id)
        second = session._submit_request(
            {
                key: value
                for key, value in payload.items()
                if key not in {"schema", "generation"}
            },
            job,
        )
        with pytest.raises(RunFailedError, match="another active run"):
            second.wait(check=True)
        first.cancel()
        assert first.wait(check=False).status.state == "cancelled"


def test_run_record_keeps_allocation_and_session_ids(cluster, monkeypatch):
    from frequensolve.orchestrator.sites.hpc import SlurmSite
    from frequensolve.project import Project
    from frequensolve.simulation.jobs import FrequencyDomainJob

    project = Project(name="record", path=cluster.root / "project")
    sim = project.new_simulation(name="acoustic", physics="acoustic", dimension=2)
    job = FrequencyDomainJob(name="forward", simulation=sim, f_list=[10.0])
    monkeypatch.setattr(
        cluster.site,
        "_record_site_run",
        SlurmSite._record_site_run.__get__(cluster.site),
    )
    monkeypatch.setattr(
        cluster.site,
        "_finalize_run_record",
        SlurmSite._finalize_run_record.__get__(cluster.site),
    )
    monkeypatch.setattr(cluster.site, "_store_remote_run_records", lambda *args: None)
    monkeypatch.setattr(cluster.site, "prepare_job", lambda job, **kwargs: job.save())
    monkeypatch.setattr(job, "task_run_plan", lambda **kwargs: {"pending_indices": [0]})
    # Do not launch this real contract against the fake solver; test record and
    # handle identity using a durable completed status as the controller emits.
    with cluster.site.session() as session:
        handle = session.submit(job, pack=False)
        record = job.latest_run(site=cluster.site.site_name)
        assert record.scheduler_id == session.allocation_id
        assert record.metadata["session_run_id"] == handle.id
        assert record.metadata["session_id"] == session.session_id
        assert record.metadata["generation"] == session.generation
        session.cancel_job(handle.id)
        assert handle.wait(check=False).status.state == "cancelled"
        assert job.latest_run(site=cluster.site.site_name).status == "cancelled"


def test_attach_preserves_worker_policy(cluster):
    workers = AdaptiveWorkers(min_ranks=2, max_ranks_per_task=3, mem_cushion=2)
    with cluster.site.session(workers=workers) as session:
        with cluster.site.attach_session(session.session_id) as attached:
            assert attached.workers == workers


def test_session_fetches_remote_logs_and_wavefields(cluster, monkeypatch):
    job = Job(cluster.root, "fetch")
    remote_log = cluster.root / "remote.log"
    remote_log.write_text("remote solver log")
    local_log = cluster.root / "fetched.log"

    def fetch_logs(job, *, local_dir=None):
        shutil.copyfile(remote_log, local_log)
        return local_log

    wavefields = object()
    monkeypatch.setattr(cluster.site, "fetch_logs", fetch_logs)
    monkeypatch.setattr(
        cluster.site, "fetch_wavefields", lambda job, **kwargs: wavefields
    )
    with cluster.site.session() as session:
        assert session.fetch_logs(job, local_dir=cluster.root) == local_log
        assert local_log.read_text() == "remote solver log"
        assert session.fetch_wavefields(job) is wavefields


def test_single_rank_preparation_limit_applies_to_postprocess(cluster):
    job = Job(cluster.root, "single-rank-prepare", smooth=True)
    job.max_ranks_per_task = 1
    job.postprocess_only = True
    with cluster.site.session() as session:
        result = session.run(job, check=True, pack=False)
        assert result.status.raw["steps"][0]["step"] == "smooth"
        assert result.status.raw["steps"][0]["ranks"] == 1


# -- liveness, lease and teardown -------------------------------------------


class Clock:
    """Controllable monotonic and wall clocks for the session module."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return 1.7e9 + self.now

    def sleep(self, seconds):
        self.now += seconds


class Heartbeats:
    """A fake site whose controller heartbeat and Slurm state are scripted."""

    def __init__(self, run_config=None):
        self.work_dir = Path("/remote/work")
        self.run_config = run_config or SlurmRunConfig()
        self.payload = None
        self.slurm = "running"
        self.cancelled = []

    def _read_remote_json(self, path):
        return None if self.payload is None else dict(self.payload)

    def update_status(self, job_id):
        return self.slurm

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)


def test_gpu_partition_session_exports_environment_and_places_devices(
    cluster, monkeypatch
):
    shape = hpc.SlurmSiteConfig(
        hostname="login.example.edu",
        queue="debug",
        mpi_wrapper=cluster.site.config.mpi_wrapper,
        partitions={
            "debug": {
                "max_nodes": 4,
                "cores_per_node": 8,
                "memory_per_node": 65536,
                "gpus_per_node": 4,
                "gpu_memory_per_gpu": 2048,
                "environment": {"I_MPI_OFFLOAD": "1"},
            }
        },
    )
    monkeypatch.setattr(cluster.site, "config", shape)
    monkeypatch.setattr(cluster.site, "config_for_partition", lambda queue: shape)

    with cluster.site.session(workers=AdaptiveWorkers(max_ranks_per_task=2)) as session:
        config = json.loads((session.root / "config.json").read_text())
        script = (session.root / "allocation.sh").read_text().splitlines()
        handles = [
            session.submit(Job(cluster.root, f"gpu-{i}", sleep=0.2)) for i in range(2)
        ]
        assert all(
            result.successful for result in session.wait_all(handles, check=True)
        )

    # GPU partitions default to one rank per device; device memory bounds ranks.
    assert config["ranks_per_node"] == 4 and config["gpus_per_node"] == 4
    assert config["scheduler"]["mem_per_rank_gib"] == pytest.approx(2.0)
    assert "export I_MPI_OFFLOAD=1" in script
    starts = [
        json.loads(line)
        for line in (cluster.root / "events.jsonl").read_text().splitlines()
        if '"start"' in line
    ]
    # Rank 0 of each step selects the device under its ibrun -o slot.
    assert starts and all(row["gpu"] == str(row["offset"]) for row in starts)
    assert len({row["gpu"] for row in starts if isinstance(row["step"], int)}) > 1


def test_controller_places_gpu_ranks_only_for_ibrun_slots(monkeypatch, tmp_path):
    controller = load_controller(monkeypatch)

    def prefix(mpi, gpus):
        config = {
            "generation": "g",
            "total_ranks": 16,
            "nodes": 2,
            "ranks_per_node": 8,
            "gpus_per_node": gpus,
            "scheduler": {"mpi": mpi, "mpi_args": [], "omp_threads": 2},
        }
        return controller.SessionController(tmp_path, config)._prefix(4, 6)

    place = str(tmp_path / "fs-gpu-place")
    assert prefix("/opt/ibrun", 8)[6:] == [place, "6", "8", "8"]
    assert os.access(place, os.X_OK)
    assert place not in prefix("/opt/ibrun", 0)
    # Session srun steps let Slurm choose nodes, so a slot names no device.
    assert place not in prefix("/opt/srun", 8)


def runtime_warning(caught):
    return next(str(w.message) for w in caught if w.category is RuntimeWarning)


def scripted_session(monkeypatch, **options):
    clock = Clock()
    monkeypatch.setattr(session_module, "time", clock)
    site = Heartbeats(options.pop("run_config", None))
    session = session_module.AllocationSession(site, **options)
    session.allocation_id = "100"
    session.generation = "g"
    return session, site, clock


def beat(site, sequence, *, updated_at, state="ready"):
    site.payload = dict(
        generation="g", state=state, sequence=sequence, updated_at=updated_at
    )


def test_heartbeat_window_defaults_and_overrides(monkeypatch, caplog):
    session, _, _ = scripted_session(monkeypatch)
    assert SlurmRunConfig().scheduler_heartbeat_timeout == 300
    assert session.heartbeat_timeout == session_module.DEFAULT_HEARTBEAT_TIMEOUT
    assert session.cleanup_timeout is None
    # An explicitly configured site window is honoured as-is.
    for configured in (60, 900):
        run_config = SlurmRunConfig(scheduler_heartbeat_timeout=configured)
        window = scripted_session(monkeypatch, run_config=run_config)[0]
        assert window.heartbeat_timeout == configured
    disabled = SlurmRunConfig(scheduler_heartbeat_timeout=None)
    assert (
        scripted_session(monkeypatch, run_config=disabled)[0].heartbeat_timeout is None
    )
    # A direct heartbeat_timeout overrides the site's window; None disables it.
    short = SlurmRunConfig(scheduler_heartbeat_timeout=60)
    explicit = scripted_session(monkeypatch, run_config=short, heartbeat_timeout=900)[0]
    assert explicit.heartbeat_timeout == 900
    off = scripted_session(monkeypatch, run_config=short, heartbeat_timeout=None)[0]
    assert off.heartbeat_timeout is None
    assert (
        scripted_session(monkeypatch, heartbeat_timeout=45)[0].heartbeat_timeout == 45
    )
    with pytest.raises(ValueError, match="heartbeat_timeout"):
        scripted_session(monkeypatch, heartbeat_timeout=0)
    with pytest.raises(ValueError, match="heartbeat_timeout"):
        scripted_session(monkeypatch, heartbeat_timeout="never")


def test_starter_site_config_writes_the_300_second_heartbeat_window():
    from frequensolve.orchestrator.sites.config_file import (
        STARTER_SITE_CONFIG,
        tomllib,
    )

    config = tomllib.loads(STARTER_SITE_CONFIG)
    windows = [
        table["run_config"]["scheduler_heartbeat_timeout"]
        for table in config["sites"].values()
        if "scheduler_heartbeat_timeout" in table.get("run_config", {})
    ]
    assert windows == [300]


def short_window_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.name == session_module.__name__
        and "heartbeat window" in record.getMessage()
    ]


def test_configured_60_second_window_is_honoured_with_one_warning(
    cluster, monkeypatch, caplog
):
    monkeypatch.setattr(
        cluster.site,
        "run_config",
        SlurmRunConfig(
            nodes=1, ranks_per_node=4, poll_interval=0, scheduler_heartbeat_timeout=60
        ),
    )
    with caplog.at_level("WARNING", logger=session_module.__name__):
        with cluster.site.session() as session:
            assert session.heartbeat_timeout == 60
            config = json.loads((session.root / "config.json").read_text())
            assert config["heartbeat_timeout"] == 60
            session.start()  # starting again does not repeat the warning
            assert session.run(Job(cluster.root, "short-window")).successful
    (warning,) = short_window_warnings(caplog)
    assert "60 s" in warning.getMessage()
    assert "heartbeat_timeout" in warning.getMessage()


def test_direct_heartbeat_timeout_overrides_site_window(cluster, monkeypatch, caplog):
    monkeypatch.setattr(
        cluster.site,
        "run_config",
        SlurmRunConfig(
            nodes=1, ranks_per_node=4, poll_interval=0, scheduler_heartbeat_timeout=60
        ),
    )
    with caplog.at_level("WARNING", logger=session_module.__name__):
        with cluster.site.session(heartbeat_timeout=600) as session:
            assert session.heartbeat_timeout == 600
            config = json.loads((session.root / "config.json").read_text())
            assert config["heartbeat_timeout"] == 600
    assert not short_window_warnings(caplog)


@pytest.mark.parametrize("skew", [600.0, -600.0])
def test_skewed_controller_clock_counts_as_fresh_while_advancing(monkeypatch, skew):
    session, site, clock = scripted_session(monkeypatch)
    for sequence in range(1, 8):
        # The controller's wall clock is ten minutes ahead of or behind ours.
        beat(site, sequence, updated_at=clock.time() + skew)
        assert session.status()["state"] == "ready"
        clock.now += 250  # long client-side gaps between polls are harmless


def test_heartbeat_gaps_of_61_to_120_seconds_do_not_fail(monkeypatch):
    session, site, clock = scripted_session(monkeypatch)
    beat(site, 1, updated_at=clock.time())
    assert session.status()["state"] == "ready"
    for _ in range(60):  # one unchanged heartbeat observed for 120 s
        clock.now += 2
        assert session.status()["state"] == "ready"
    beat(site, 2, updated_at=clock.time() - 120)
    for _ in range(60):
        clock.now += 2
        assert session.status()["state"] == "ready"


def test_unchanged_heartbeat_with_running_allocation_means_controller_lost(
    monkeypatch,
):
    session, site, clock = scripted_session(monkeypatch, heartbeat_timeout=100)
    beat(site, 7, updated_at=clock.time() + 3600)
    assert session.status()["state"] == "ready"
    clock.now += 99
    assert session.status()["state"] == "ready"
    clock.now += 2
    # One stale reading is not enough; a second one a poll interval later is.
    assert session.status()["state"] == "ready"
    assert session.status()["state"] == "ready"  # no time has passed
    clock.now += 0.2
    lost = session.status()
    assert lost["state"] == "failed" and lost["heartbeat_expired"]
    assert "heartbeat expired" in lost["reason"] and "100" in lost["reason"]
    with pytest.raises(RuntimeError, match="heartbeat expired"):
        session._ready = True
        session._require_ready()
    # A new heartbeat revives it.
    beat(site, 8, updated_at=0)
    assert session.status()["state"] == "ready"


def test_stale_heartbeat_requires_running_allocation_and_successful_reads(
    monkeypatch,
):
    session, site, clock = scripted_session(monkeypatch, heartbeat_timeout=10)
    beat(site, 1, updated_at=clock.time())
    assert session.status()["state"] == "ready"
    payload = site.payload
    site.payload = None  # transport failures are not evidence of loss
    for _ in range(5):
        clock.now += 6
        assert session.status()["state"] == "ready"
    site.payload = payload
    site.slurm = "unknown"  # nor is a heartbeat Slurm cannot corroborate
    for _ in range(3):
        clock.now += 6
        assert session.status()["state"] == "ready"
    site.slurm = "running"
    clock.now += 6  # the cached Slurm state is refreshed every 5 s
    session.status()
    clock.now += 1
    assert session.status()["state"] == "failed"


def test_disabled_heartbeat_timeout_never_expires(monkeypatch):
    session, site, clock = scripted_session(monkeypatch, heartbeat_timeout=None)
    beat(site, 1, updated_at=0)
    for _ in range(5):
        clock.now += 10_000
        assert session.status()["state"] == "ready"


def test_wait_ready_applies_the_same_liveness_rule(monkeypatch):
    session, site, clock = scripted_session(monkeypatch, heartbeat_timeout=5)
    monkeypatch.setattr(session, "start", lambda: session)
    beat(site, 1, updated_at=clock.time(), state="starting")
    with pytest.raises(RuntimeError, match="heartbeat expired"):
        session.wait_ready(timeout=60)
    beat(site, 2, updated_at=clock.time(), state="ready")
    assert session.wait_ready(timeout=60) is session


def load_controller(monkeypatch):
    directory = Path(hpc.__file__).parent / "templates" / "sweep"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("session_controller")


def test_controller_heartbeat_is_independent_of_slow_run_status_writes(
    tmp_path, monkeypatch
):
    controller_module = load_controller(monkeypatch)
    controller = controller_module.SessionController(
        tmp_path,
        dict(generation="g", total_ranks=1, lease_timeout=5, heartbeat_interval=0.02),
    )
    original = controller_module._write_json_atomic
    run_writes = []

    def slow(path, payload):
        if "runs" in Path(path).parts:
            run_writes.append(path)
            time.sleep(0.5)  # a stalled shared filesystem
        original(path, payload)

    monkeypatch.setattr(controller_module, "_write_json_atomic", slow)
    controller._controller_status("ready")
    heartbeat = controller_module.threading.Thread(target=controller._beat)
    heartbeat.start()
    try:
        run = dict(id="a" * 32, state="running", rows=[])
        (tmp_path / "runs" / run["id"]).mkdir()
        before = json.loads((tmp_path / "status.json").read_text())["sequence"]
        controller._status(run)
        after = json.loads((tmp_path / "status.json").read_text())["sequence"]
        assert after - before >= 5
        # Unchanged run status is not rewritten every tick.
        controller._status(run)
        controller._status(run)
        assert len(run_writes) == 1
        controller._status(run, "complete")
        assert len(run_writes) == 2
    finally:
        controller._heartbeat_stop.set()
        heartbeat.join()


def test_controller_lease_ignores_clock_skew_and_honors_revocation(
    tmp_path, monkeypatch
):
    controller_module = load_controller(monkeypatch)
    controller = controller_module.SessionController(
        tmp_path, dict(generation="g", total_ranks=1, lease_timeout=30)
    )
    now = [500.0]
    clock = SimpleNamespace(monotonic=lambda: now[0], time=time.time)
    monkeypatch.setattr(controller_module, "time", clock)
    lease = tmp_path / "leases" / "client.json"

    def renew(sequence, **extra):
        payload = dict(generation="g", sequence=sequence, timeout=30)
        # The client's clock is a day behind: its expiry is long past here.
        payload["expires_at"] = time.time() - 86400
        lease.write_text(json.dumps(dict(payload, **extra)))

    for sequence in range(1, 5):
        renew(sequence)
        assert controller._lease_alive()
        now[0] += 25
    now[0] += 6  # 31 s without a renewal
    assert not controller._lease_alive()
    renew(9)
    assert controller._lease_alive()
    renew(10, revoked=True, expires_at=0)
    assert not controller._lease_alive()


def test_controller_archives_processed_requests_and_commands(cluster):
    with cluster.site.session() as session:
        handle = session.submit(Job(cluster.root, "archived"))
        assert handle.wait(check=True).successful
        handle.cancel()  # a no-op command for a finished run
        processed = session.root / "commands" / "processed"
        wait_until(lambda: any(processed.glob("*.json")))
        assert not list((session.root / "requests").glob("*.json"))
        assert not list((session.root / "commands").glob("*.json"))
        assert (session.root / "requests" / "processed" / f"{handle.id}.json").is_file()
        heartbeat = cluster.site._read_remote_json(session.root / "status.json")
        assert heartbeat["sequence"] >= 1


def test_drain_waits_for_in_flight_work_without_scancel(cluster, monkeypatch):
    states_at_cancel = []
    cancel = cluster.site.cancel_job

    def observed_cancel(job_id):
        states_at_cancel.append(
            cluster.site._read_remote_json(session.root / "status.json")["state"]
        )
        cancel(job_id)

    monkeypatch.setattr(cluster.site, "cancel_job", observed_cancel)
    with cluster.site.session() as session:
        handle = session.submit(Job(cluster.root, "in-flight", tasks=1, sleep=0.6))
        wait_until(lambda: handle.status().is_running)
    # The context exited while the run was in flight: it finished first, and
    # the allocation was never cancelled while the controller was draining.
    status = json.loads((session.root / "runs" / handle.id / "status.json").read_text())
    assert status["state"] == "complete"
    assert all(state == "closed" for state in states_at_cancel)
    assert cluster.processes[0].poll() == 0


def test_drain_cap_warns_and_leaves_allocation_running(cluster):
    with pytest.warns(RuntimeWarning, match="cleanup_timeout") as caught:
        with cluster.site.session(cleanup_timeout=0.3) as session:
            handle = session.submit(Job(cluster.root, "slow-drain", tasks=1, sleep=1))
            wait_until(lambda: handle.status().is_running)
    message = runtime_warning(caught)
    assert session.session_id in message and session.allocation_id in message
    assert not cluster.cancellations
    # The controller finishes the drain by itself instead of being killed.
    wait_until(lambda: cluster.processes[0].poll() is not None)
    status = json.loads((session.root / "runs" / handle.id / "status.json").read_text())
    assert status["state"] == "complete"


def test_stale_heartbeat_during_drain_warns_instead_of_cancelling(cluster, monkeypatch):
    with pytest.warns(RuntimeWarning, match="lost its controller heartbeat"):
        with cluster.site.session(heartbeat_timeout=0.5) as session:
            handle = session.submit(Job(cluster.root, "hung-drain", sleep=3))
            wait_until(lambda: handle.status().is_running)
            frozen = cluster.site._read_remote_json(session.root / "status.json")
            original = cluster.site._read_remote_json
            monkeypatch.setattr(
                cluster.site,
                "_read_remote_json",
                lambda path: (
                    dict(frozen)
                    if Path(path) == session.root / "status.json"
                    else original(path)
                ),
            )
    assert not cluster.cancellations


def test_close_failure_after_successful_work_warns_and_keeps_results(
    cluster, monkeypatch
):
    login = cluster.site._run_session_command

    def failing_close(command, **kwargs):
        if "/commands/" in command and "ln " in command:
            raise RuntimeError("ssh connection reset")
        return login(command, **kwargs)

    with pytest.warns(RuntimeWarning, match="attach_session") as caught:
        with cluster.site.session() as session:
            result = session.run(Job(cluster.root, "kept"), check=True)
            monkeypatch.setattr(cluster.site, "_run_session_command", failing_close)
    assert result.successful
    assert "ssh connection reset" in runtime_warning(caught)
    assert session.session_id in runtime_warning(caught)
    assert not cluster.cancellations  # an unconfirmed drain is not cancelled


def test_failed_body_still_cancels_and_reports_original_error(cluster, monkeypatch):
    monkeypatch.setattr(
        cluster.site,
        "cancel_job",
        lambda job_id: (_ for _ in ()).throw(RuntimeError("scancel failed")),
    )
    with pytest.raises(KeyError, match="original"):
        with cluster.site.session():
            raise KeyError("original")


def test_enter_reports_readiness_error_not_cleanup_error(cluster, monkeypatch):
    monkeypatch.setattr(cluster.site, "update_status", lambda job_id: "pending")
    original = cluster.site._submit_sbatch

    def pending(command):
        result = original(command)
        cluster.processes[-1].kill()
        cluster.processes[-1].wait()
        (Path(shlex.split(command)[-1]).parent / "status.json").unlink(missing_ok=True)
        return result

    def broken_cancel(job_id):
        raise RuntimeError("scancel unavailable")

    monkeypatch.setattr(cluster.site, "_submit_sbatch", pending)
    monkeypatch.setattr(cluster.site, "cancel_job", broken_cancel)
    with pytest.raises(TimeoutError, match="readiness"):
        with cluster.site.session(startup_timeout=0.1, cleanup_timeout=0.1):
            pytest.fail("controller never became ready")


def test_lease_renews_while_submit_uploads(cluster, monkeypatch):
    session = cluster.site.session(lease_timeout=1.0)
    lease = None
    sequences = []

    def slow_upload(job):
        # Longer than the lease: renewal must not wait for the submit lock.
        for _ in range(12):
            sequences.append(json.loads(lease.read_text())["sequence"])
            time.sleep(0.25)

    monkeypatch.setattr(cluster.site, "_transfer_remote_simulation_inputs", slow_upload)
    with session:
        lease = session.root / "leases" / f"{session._client_id}.json"
        result = session.run(Job(cluster.root, "slow-upload"), check=True)
        assert result.successful
        assert sequences[-1] > sequences[0] + 3
        assert session.status()["state"] == "ready"
    assert cluster.processes[0].poll() == 0


def test_session_command_timeout_defaults_and_overrides(monkeypatch):
    assert SlurmRunConfig().session_command_timeout == 120.0
    session, _, _ = scripted_session(monkeypatch)
    assert session.command_timeout == session_module.DEFAULT_SESSION_COMMAND_TIMEOUT
    configured = SlurmRunConfig(session_command_timeout=300)
    assert (
        scripted_session(monkeypatch, run_config=configured)[0].command_timeout == 300
    )
    explicit = scripted_session(
        monkeypatch, run_config=configured, session_command_timeout=45
    )[0]
    assert explicit.command_timeout == 45
    with pytest.raises(ValueError, match="session_command_timeout"):
        scripted_session(monkeypatch, session_command_timeout=0)
    with pytest.raises(ValueError, match="session_command_timeout"):
        SlurmRunConfig(session_command_timeout=float("inf"))
    # The flat site.toml key reaches the run config.
    assert (
        SlurmRunConfig().merged(session_command_timeout=90).session_command_timeout
        == 90
    )


def test_slow_session_mkdir_names_command_path_and_timeout(cluster, monkeypatch):
    site = cluster.site
    monkeypatch.setattr(
        site,
        "_run_session_command",
        hpc.SlurmSite._run_session_command.__get__(site),
    )
    timeouts = []

    def slow_login(command, *, timeout=None):
        timeouts.append(timeout)
        raise TimeoutError("SSH command to login timed out")

    monkeypatch.setattr(site, "run_login_cmd", slow_login)
    session = site.session(session_command_timeout=200)
    with pytest.raises(TimeoutError) as caught:
        session.start()
    message = str(caught.value)
    assert "mkdir" in message
    assert str(session.root) in message
    assert "200 s" in message and "session_command_timeout" in message
    assert "Enterprise HPC" not in message
    # Session commands use their own bound, not the 15 s identity-query bound.
    assert timeouts == [200]
    assert cluster.processes == []  # nothing was submitted


def test_failed_session_publication_names_command_and_path(cluster, monkeypatch):
    site = cluster.site
    helper = hpc.SlurmSite._run_session_command.__get__(site)

    def failing_login(command, *, timeout=None):
        return (
            None,
            SimpleNamespace(
                read=lambda: b"",
                channel=SimpleNamespace(recv_exit_status=lambda: 1),
            ),
            SimpleNamespace(read=lambda: b"mv: Input/output error"),
        )

    monkeypatch.setattr(site, "run_login_cmd", failing_login)
    target = Path("/remote/work/.fs_sessions/x/leases/client.json")
    with pytest.raises(RuntimeError) as caught:
        helper("mv -f -- a b", timeout=120.0, action="publication (mv)", path=target)
    message = str(caught.value)
    assert "publication (mv)" in message and str(target) in message
    assert "Input/output error" in message and "exit status 1" in message


def test_lost_controller_reports_its_remote_log(monkeypatch):
    session, site, clock = scripted_session(monkeypatch, heartbeat_timeout=10)
    log = str(Path("/remote/work/.fs_sessions") / session.session_id / "controller.log")
    assert str(session.controller_log) == log
    beat(site, 1, updated_at=0)
    session.status()
    clock.now += 11
    session.status()
    clock.now += 1
    lost = session.status()
    assert lost["heartbeat_expired"] and lost["controller_log"] == log
    assert log in lost["reason"]
    # A controller that published "failed", or a job Slurm ended, names it too.
    beat(site, 2, updated_at=0, state="failed")
    assert session.status()["controller_log"] == log
    site.payload, site.slurm = None, "failed"
    session._controller_state = None
    clock.now += 6
    assert session.status() == {"state": "failed", "controller_log": log}


def test_controller_logs_its_failure_before_a_failing_status_write(
    tmp_path, monkeypatch, capsys
):
    controller_module = load_controller(monkeypatch)
    controller = controller_module.SessionController(
        tmp_path,
        dict(generation="g", total_ranks=1, lease_timeout=30, heartbeat_interval=60),
    )
    publish = controller._controller_status

    def status(state=None):
        if state == "failed":
            raise OSError("work filesystem unavailable")
        publish(state)

    def tick():
        raise RuntimeError("original failure")

    run = dict(id="b" * 32, state="running", rows=[])
    controller.runs[run["id"]] = run
    monkeypatch.setattr(controller, "_controller_status", status)
    monkeypatch.setattr(controller, "tick", tick)
    monkeypatch.setattr(
        controller,
        "_cancel",
        lambda *args: (_ for _ in ()).throw(OSError("cannot write run status")),
    )
    with pytest.raises(RuntimeError, match="original failure"):
        controller.run()
    log = capsys.readouterr().out
    assert log.index("Controller failed: RuntimeError('original failure')") < (
        log.index("Could not publish the failed controller status")
    )
    assert "Traceback" in log and "cannot write run status" in log
    assert controller.stop_reason == "controller failed: RuntimeError"


def test_controller_enables_faulthandler_before_anything_else(monkeypatch):
    controller_module = load_controller(monkeypatch)
    enabled = []
    monkeypatch.setattr(
        controller_module.faulthandler,
        "enable",
        lambda **kwargs: enabled.append(kwargs),
    )
    monkeypatch.setattr(controller_module.sys, "argv", ["session_controller.py"])
    with pytest.raises(SystemExit):
        controller_module.main()  # --root is missing
    assert enabled == [{"all_threads": True}]


# -- reuse and linger ---------------------------------------------------------


def events(cluster):
    return [
        json.loads(line)
        for line in (cluster.root / "events.jsonl").read_text().splitlines()
    ]


def test_lingering_session_is_reused_by_the_next_client(cluster):
    with cluster.site.session(reuse="study", linger="1m") as first:
        assert not first.reused
        assert first.run(Job(cluster.root, "first"), check=True).successful
    # A normal exit only detaches; the allocation stays up for reuse.
    assert cluster.processes[0].poll() is None and not cluster.cancellations
    wait_until(lambda: first.status().get("clients") == 0)
    [info] = cluster.site.sessions(reuse="study")
    assert (info.session_id, info.allocation_id) == (first.session_id, "100")
    assert (info.state, info.linger, info.reuse) == ("ready", 60.0, "study")
    assert info.idle_seconds is not None
    with cluster.site.session(reuse="study", linger="1m") as second:
        assert second.reused and second.session_id == first.session_id
        assert second.run(Job(cluster.root, "second"), check=True).successful
    assert len(cluster.processes) == 1
    second.release()
    # release() ends it even after the context detached; no scancel needed.
    assert cluster.processes[0].poll() == 0
    assert cluster.site.sessions() == [] and not cluster.cancellations


def test_closed_session_still_completing_in_slurm_is_not_listed(cluster, monkeypatch):
    with cluster.site.session(reuse="done") as session:
        pass
    monkeypatch.setattr(
        cluster.site,
        "_session_allocations",
        lambda: {"100": {"state": "running", "time_left": 60.0}},  # Slurm CG
    )
    assert session.status()["state"] == "closed"
    assert cluster.site.sessions() == []


def test_default_session_neither_lingers_nor_is_reused(cluster):
    with cluster.site.session(reuse="plain") as first:
        pass
    assert cluster.processes[0].poll() is not None
    with cluster.site.session(linger="1m") as unscoped:
        pass
    with cluster.site.session() as other:
        # Sessions without reuse never adopt one, even a lingering session.
        assert not other.reused
    assert len(cluster.processes) == 3
    assert cluster.processes[1].poll() is None
    unscoped.release()
    assert not first.reused


def test_reuse_needs_the_same_scope_resources_and_walltime(cluster, monkeypatch):
    with cluster.site.session(reuse="a", linger="1m") as first:
        pass
    with cluster.site.session(reuse="b") as other_scope:
        assert not other_scope.reused
    with cluster.site.session(reuse="a", ranks_per_node=2) as other_shape:
        assert not other_shape.reused
    with monkeypatch.context() as patch:
        patch.setattr(
            cluster.site,
            "_session_allocations",
            lambda: {"100": {"state": "running", "time_left": 60.0}},
        )
        with cluster.site.session(reuse="a") as late:
            assert not late.reused
    assert len(cluster.processes) == 4
    with cluster.site.session(reuse="a", min_time_left=0) as again:
        assert again.reused and again.session_id == first.session_id
    with pytest.raises(RuntimeError, match="No running or queued"):
        cluster.site.attach_session(reuse="missing").start()
    assert len(cluster.processes) == 4


def test_project_scopes_differ_by_path_and_name(tmp_path):
    def project(name, path):
        return SimpleNamespace(name=name, path=tmp_path / path)

    key, label = session_module.reuse_key(project("fwi", "a"))
    assert label == "fwi" and key.startswith("project:fwi:")
    assert key == session_module.reuse_key(project("fwi", "a"))[0]
    assert key != session_module.reuse_key(project("fwi", "b"))[0]
    assert key != session_module.reuse_key(project("rtm", "a"))[0]
    assert session_module.reuse_key("fwi") == ("key:fwi", "fwi")
    with pytest.raises(TypeError):
        session_module.reuse_key(3)


def test_reuse_options_are_validated(cluster):
    with pytest.raises(ValueError, match="share=True requires reuse"):
        cluster.site.session(share=True)
    with pytest.raises(ValueError, match="borrows"):
        cluster.site.attach_session("a" * 32, linger=60)
    with pytest.raises(ValueError, match="linger"):
        cluster.site.session(linger="soon")
    with pytest.raises(TypeError, match="exactly one"):
        cluster.site.attach_session()
    assert session_module._duration("1h30m", "x") == 5400
    assert session_module._duration("01:30:00", "x") == 5400
    assert session_module._duration(90, "x") == 90


def test_attached_session_is_joined_only_with_share(cluster):
    with cluster.site.session(reuse="s") as owner:
        with cluster.site.session(reuse="s") as other:
            assert not other.reused
        with cluster.site.session(reuse="s", share=True) as joined:
            assert joined.reused and joined.session_id == owner.session_id
            assert joined.run(Job(cluster.root, "joined"), check=True).successful
        # Closing a shared session detaches; the owner keeps working.
        assert cluster.processes[0].poll() is None
        assert owner.run(Job(cluster.root, "owner"), check=True).successful
    wait_until(lambda: cluster.processes[0].poll() is not None)
    assert len(cluster.processes) == 2


def test_disappeared_client_leaves_its_run_to_the_next_client(cluster):
    first = cluster.site.session(reuse="crash", linger="1m", lease_timeout=1.0)
    first.wait_ready()
    job = Job(cluster.root, "long", tasks=1, sleep=2)
    run = first.submit(job)
    job.record = SimpleNamespace(
        metadata={
            "session_id": first.session_id,
            "session_run_id": run.id,
            "generation": first.generation,
        }
    )
    first._stop.set()  # the client vanishes without detaching
    first._lease_thread.join()
    wait_until(lambda: first.status().get("clients") == 0)
    with cluster.site.session(reuse="crash", linger="1m") as second:
        assert second.reused
        resumed = second.submit(job)
        assert resumed.id == run.id
        assert resumed.wait(check=True).successful
    starts = [row for row in events(cluster) if row["event"] == "start"]
    assert [row["step"] for row in starts if row["step"] == 1] == [1]
    assert len(cluster.processes) == 1
    second.release()


def test_lease_of_an_exited_local_process_does_not_block_reuse(cluster):
    first = cluster.site.session(reuse="pid", linger="1m")
    first.wait_ready()
    first._stop.set()
    first._lease_thread.join()
    # The controller still counts the lease; its process is gone.
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    lease = first.root / "leases" / f"{first._client_id}.json"
    lease.write_text(json.dumps(dict(json.loads(lease.read_text()), pid=exited.pid)))
    assert first.status()["clients"] == 1
    with cluster.site.session(reuse="pid", linger="1m") as second:
        assert second.reused
    second.release()


def test_queued_allocation_is_adopted_after_an_interrupt(cluster, monkeypatch):
    scripts = []
    start = cluster.site._submit_sbatch
    monkeypatch.setattr(
        cluster.site, "_submit_sbatch", lambda command: scripts.append(command) or "100"
    )
    monkeypatch.setattr(
        cluster.site,
        "_session_allocations",
        lambda: {"100": {"state": "pending", "time_left": 3600.0}},
    )

    def interrupted(self, timeout=None):
        self.start()
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(session_module.AllocationSession, "wait_ready", interrupted)
        with pytest.raises(KeyboardInterrupt):
            with cluster.site.session(reuse="queued", linger="1m"):
                pass
    # A lingering allocation keeps its place in the queue for the next client.
    assert not cluster.cancellations
    second = cluster.site.session(reuse="queued", linger="1m")
    second.start()
    assert second.reused and len(scripts) == 1
    start(scripts[0])  # Slurm starts the queued job
    with second:
        assert second.run(Job(cluster.root, "queued"), check=True).successful
    second.release()
    assert cluster.processes[0].poll() == 0


def test_release_sessions_keeps_attached_ones(cluster):
    with cluster.site.session(reuse="idle", linger="1m") as idle:
        pass
    wait_until(lambda: idle.status().get("clients") == 0)
    with cluster.site.session(reuse="busy", linger="1m") as busy:
        wait_until(lambda: busy.status().get("clients") == 1)
        assert cluster.site.release_sessions("busy") == []
        released = cluster.site.release_sessions()
        assert [info.session_id for info in released] == [idle.session_id]
        wait_until(lambda: cluster.processes[0].poll() is not None)
        assert busy.run(Job(cluster.root, "busy"), check=True).successful
    busy.release()


def test_controller_lingers_for_orphaned_runs_then_idles(tmp_path, monkeypatch):
    controller_module = load_controller(monkeypatch)
    controller = controller_module.SessionController(
        tmp_path, dict(generation="g", total_ranks=1, lease_timeout=30, linger=100)
    )
    now = [1000.0]
    clock = SimpleNamespace(monotonic=lambda: now[0], time=time.time)
    monkeypatch.setattr(controller_module, "time", clock)
    controller.started = now[0]
    lease = tmp_path / "leases" / "client.json"
    lease.write_text(json.dumps(dict(generation="g", sequence=1, timeout=30)))
    controller.active_runs = 1
    controller._check_clients()
    assert controller.clients == ["client"]
    now[0] += 31  # the client stopped renewing
    controller._check_clients()
    # An orphaned run keeps the allocation; nothing counts as idle yet.
    assert controller.stop_reason is None and controller.idle_since is None
    controller.active_runs = 0
    controller._check_clients()
    controller._controller_status()
    status = json.loads((tmp_path / "status.json").read_text())
    assert (status["clients"], status["client_ids"], status["linger"]) == (0, [], 100)
    now[0] += 99
    controller._check_clients()
    assert controller.stop_reason is None
    now[0] += 1
    controller._check_clients()
    assert "linger" in controller.stop_reason


def test_controller_lingers_as_long_as_the_last_clients_ask(tmp_path, monkeypatch):
    controller_module = load_controller(monkeypatch)
    controller = controller_module.SessionController(
        tmp_path, dict(generation="g", total_ranks=1, lease_timeout=30, linger=100)
    )
    leases = tmp_path / "leases"

    def lease(name, linger, **extra):
        payload = dict(generation="g", sequence=1, timeout=30, linger=linger)
        (leases / f"{name}.json").write_text(json.dumps(dict(payload, **extra)))

    lease("owner", 600)
    lease("sharer", 0)
    lease("borrower", None)
    controller._check_clients()
    assert controller.clients == ["borrower", "owner", "sharer"]
    assert controller.linger == 600
    lease("owner", 600, revoked=True)
    controller._check_clients()
    assert controller.linger == 0  # the sharer is the last owner attached
    lease("sharer", 0, revoked=True)
    lease("borrower", None, revoked=True)
    controller.active_runs = 1
    controller._check_clients()
    # Without linger, the original rule: cancel outstanding work and end.
    assert controller.stop_reason == "client lease expired"


def test_slurm_time_left_parsing():
    from frequensolve.orchestrator.sites.hpc.slurm_helpers import slurm_time_seconds

    assert slurm_time_seconds("1-02:03:04") == 93784
    assert slurm_time_seconds("2:03:04") == 7384
    assert slurm_time_seconds("3:04") == 184
    assert slurm_time_seconds("5") == 300
    assert slurm_time_seconds("2-03") == 183600
    assert slurm_time_seconds("UNLIMITED") == float("inf")
    assert slurm_time_seconds("INVALID") is None
