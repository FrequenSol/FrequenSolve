"""Persistent sessions exercised with a local, asynchronous fake cluster."""

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


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mpi = bin_dir / "ibrun"
    mpi.write_text(
        f"""#!{sys.executable}
import os, sys
args = sys.argv[1:]
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
row = dict(step=step, pid=os.getpid(), ranks=int(os.environ['FAKE_RANKS']), offset=int(os.environ['FAKE_OFFSET']), job=args[args.index('--job')+1], event='start', at=time.time())
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
    monkeypatch.setattr(site, "_run_login_checked", login)
    monkeypatch.setattr(site, "_submit_sbatch", submit)
    monkeypatch.setattr(site, "cancel_job", cancel)
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
    assert cluster.processes[0].poll() == 0
    assert cluster.cancellations == ["100"]
    session.close()
    assert cluster.cancellations == ["100"]


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
    assert cluster.processes[0].poll() is not None
    assert cluster.cancellations == ["100"]


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
        payload = json.loads(
            (session.root / "requests" / ("a" * 32 + ".json")).read_text()
        )
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
        # A stale heartbeat forces the Slurm liveness check.
        status_file = session.root / "status.json"
        payload = json.loads(status_file.read_text())
        payload["updated_at"] = 0
        status_file.write_text(json.dumps(payload))
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
        payload = json.loads(
            (session.root / "requests" / f"{first.id}.json").read_text()
        )
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


def test_transient_status_read_keeps_fresh_heartbeat_but_expires(cluster, monkeypatch):
    with cluster.site.session() as session:
        assert session.status()["state"] == "ready"
        original = cluster.site._read_remote_json
        monkeypatch.setattr(
            cluster.site,
            "_read_remote_json",
            lambda path: (
                None if path == session.root / "status.json" else original(path)
            ),
        )
        session._require_ready()
        session._controller_state = dict(
            session._controller_state, updated_at=time.time() - 61
        )
        with pytest.raises(RuntimeError, match="heartbeat expired"):
            session._require_ready()
        monkeypatch.setattr(cluster.site, "_read_remote_json", original)


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
