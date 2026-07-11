import io
import json
import logging
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.base import JobStatus, RunHandle
from frequensolve.orchestrator.sites.local import LocalSite, run_task
from frequensolve.orchestrator.sites.local import site as local_module
from frequensolve.orchestrator.sites.local.dask_logging import (
    configure_dependency_logging,
)
from frequensolve.orchestrator.utils.progress import status_table_html


class DummyFuture:
    def __init__(self, result=None):
        self._result = result or {"status": "success"}
        self.status = "finished"
        self.cancelled = False
        self.released = False

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result

    def cancel(self):
        self.cancelled = True
        self.status = "cancelled"

    def release(self):
        self.released = True


class OneShotFuture(DummyFuture):
    def __init__(self, result=None):
        super().__init__(result)
        self.result_calls = 0

    def result(self):
        self.result_calls += 1
        if self.result_calls > 1:
            raise RuntimeError("future result requested after terminal poll")
        return self._result


class DummyJob:
    name = "local-job"
    trace_manifest = None
    _stdout_path = None
    run_metadata = None

    def __init__(self):
        self.states = []
        self.removed_packed = False
        self.trace_outputs_ok = True

    def write_run_state(self, status="completed", **extra):
        self.states.append((status, extra))

    def remove_packed_trace_products(self):
        self.removed_packed = True
        return True

    def trace_outputs_exist(self):
        return self.trace_outputs_ok


def make_site(monkeypatch):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    site = LocalSite()
    closed = []
    site.close = lambda **kwargs: closed.append(kwargs)
    return site, closed


def make_run(site, job, futures, shutdown_on_completion=True):
    run = RunHandle(site=site, job=job, id="local:test")
    run.backend["futures"] = futures
    run.backend["shutdown_on_completion"] = shutdown_on_completion
    site._futures.extend(futures)
    return run


def test_local_solver_uses_configured_path(tmp_path):
    configured_solver = tmp_path / "configured-solver"
    configured_solver.touch()
    site = object.__new__(LocalSite)
    site.solver = configured_solver

    assert LocalSite._get_solver_path(site) == str(configured_solver)


def test_local_site_uses_profile_environment_without_forwarding_credentials(
    monkeypatch,
):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    monkeypatch.setenv("HPC_PASSWORD", "secret")

    site = LocalSite(environment={"OMP_NUM_THREADS": 2})

    assert site.env["OMP_NUM_THREADS"] == "2"
    assert site.env["MKL_DYNAMIC"] == "FALSE"
    assert site.env["MKL_NUM_THREADS"] == "1"
    assert site.env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert "HPC_PASSWORD" not in site.env


def test_local_site_rejects_credentials_in_profile_environment(monkeypatch):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")

    with pytest.raises(ValueError, match="Credential variable"):
        LocalSite(environment={"HPC_PASSWORD": "secret"})


def test_local_site_profile_can_override_numeric_runtime_defaults(monkeypatch):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")

    site = LocalSite(environment={"MKL_NUM_THREADS": 2})

    assert site.env["MKL_NUM_THREADS"] == "2"


def test_local_task_status_counts_submitted_futures_as_running():
    assert local_module._local_task_status(["pending", "lost", "finished"]) == {
        "successful": 1,
        "failed": 0,
        "running": 2,
        "pending": 0,
        "total": 3,
    }


def test_local_poll_reports_dask_pending_futures_as_running(monkeypatch):
    site, _closed = make_site(monkeypatch)
    job = DummyJob()
    futures = [DummyFuture(), DummyFuture()]
    for future in futures:
        future.status = "pending"
    run = make_run(site, job, futures, shutdown_on_completion=False)

    status = site._poll_local_run(run)
    panel = status_table_html([run], {0: status})

    assert status.state == "running"
    assert status.message == (
        "tasks: 0 successful, 0 failed, 2 running, 0 pending, 2 total"
    )
    assert status.raw["task_status"] == {
        "successful": 0,
        "failed": 0,
        "running": 2,
        "pending": 0,
        "total": 2,
    }
    assert "running" in panel
    assert "pending" not in panel


def test_local_poll_includes_already_current_tasks_in_progress(monkeypatch):
    site, _closed = make_site(monkeypatch)
    job = DummyJob()
    job.n_tasks = 90
    futures = [DummyFuture(), *[DummyFuture() for _ in range(70)]]
    for future in futures[1:]:
        future.status = "pending"
    run = make_run(site, job, futures, shutdown_on_completion=False)
    run.backend["task_plan"] = {
        "current_tasks": list(range(1, 20)),
        "pending_indices": list(range(19, 90)),
        "reused_tasks": [],
    }

    status = site._poll_local_run(run)

    assert status.message == (
        "tasks: 20 successful, 0 failed, 70 running, 0 pending, 90 total"
    )
    assert status.raw["task_status"] == {
        "successful": 20,
        "succeeded": 20,
        "failed": 0,
        "running": 70,
        "pending": 0,
        "total": 90,
        "current": 19,
        "submitted_total": 71,
        "includes_current_tasks": True,
    }


def test_local_poll_reports_finished_futures_without_failed_count(monkeypatch):
    site, _closed = make_site(monkeypatch)
    job = DummyJob()
    run = make_run(site, job, [DummyFuture(), DummyFuture()])

    status = site._poll_local_run(run)

    assert status.state == "completed"
    assert status.raw["task_status"] == {
        "successful": 2,
        "failed": 0,
        "running": 0,
        "pending": 0,
        "total": 2,
    }


def test_local_wait_releases_futures_and_closes_by_default(monkeypatch, capsys):
    site, closed = make_site(monkeypatch)
    job = DummyJob()
    futures = [DummyFuture(), DummyFuture()]
    run = make_run(site, job, futures)

    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )

    result = site._wait_local_run(run)

    assert result.successful
    assert job.states[-1][0] == "completed"
    assert all(future.released for future in futures)
    assert site._futures == []
    assert closed == [{"wait": True, "retire": True}]
    captured = capsys.readouterr()
    assert captured.out == ""


def test_wait_all_uses_custom_wait_finalizer(monkeypatch):
    site, _ = make_site(monkeypatch)
    job = DummyJob()
    calls = []

    def wait_fn(run, timeout=None, poll_interval=None):
        calls.append((timeout, poll_interval))
        return run._make_result(
            JobStatus(state="completed", return_code=0, job_id=run.id)
        )

    run = RunHandle(
        site=site,
        job=job,
        id="local:custom",
        _status_fn=lambda run: JobStatus(
            state="completed",
            return_code=0,
            job_id=run.id,
        ),
        _wait_fn=wait_fn,
    )

    [result] = site.wait_all([run], poll_interval=0.0)

    assert result.successful
    assert calls == [(0, 0)]


def test_run_task_supports_solver_pack_mode(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        def wait(self):
            return 0

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        kwargs["stdout"].write("solver output\n")
        return FakeProcess()

    job_file = tmp_path / "job.json"
    job_file.write_text("{}")
    monkeypatch.setattr(local_module.subprocess, "Popen", fake_popen)

    result = run_task(
        str(job_file),
        local_module.PACK_TASK_ID,
        "/solver",
        {},
        n_threads=2,
        stdout_dir=str(tmp_path / "logs"),
    )

    assert result["status"] == "success"
    assert result["stdout"].endswith("pack.log")
    assert captured["args"] == [
        "/solver",
        "-nthreads",
        "2",
        "--job",
        str(job_file),
        "--pack",
    ]
    assert (tmp_path / "logs" / "pack.log").read_text() == (
        f"[INFO] {local_module.logger.name}: Executing: "
        f"/solver -nthreads 2 --job {job_file} --pack\n"
        "solver output\n"
    )


def test_run_task_adds_fresh_flag(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        def wait(self):
            return 0

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return FakeProcess()

    job_file = tmp_path / "job.json"
    job_file.write_text("{}")
    monkeypatch.setattr(local_module.subprocess, "Popen", fake_popen)

    result = run_task(str(job_file), 0, "/solver", {}, stdout_dir=None, fresh=True)

    assert result["status"] == "success"
    assert captured["args"] == [
        "/solver",
        "-nthreads",
        "1",
        "--job",
        str(job_file),
        "--fresh",
        "--task",
        "1",
    ]


def test_run_task_reports_solver_convergence_failure(monkeypatch, tmp_path):
    class FakeProcess:
        def wait(self):
            return 0

    monkeypatch.setattr(
        local_module.subprocess,
        "Popen",
        lambda args, **kwargs: FakeProcess(),
    )
    job_file = tmp_path / "job.json"
    job_file.write_text(
        json.dumps(
            {
                "project_path": str(tmp_path),
                "result_path": "results",
            }
        )
    )
    manifest = tmp_path / "results/_fs_run/tasks/task_000001/run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "solver": {
                    "convergence": {
                        "converged": True,
                        "status": "converged",
                        "solve_count": 1,
                        "failure_count": 0,
                        "worst_code": 0,
                        "solves": [
                            {
                                "context": "forward",
                                "converged": True,
                                "iterations": 16,
                                "residual": 2.2e-3,
                                "solver": "FS_MG",
                                "status": "converged",
                            }
                        ],
                    }
                }
            }
        )
    )

    result = run_task(str(job_file), 0, "/solver", {}, stdout_dir=None)

    assert result["status"] == "error"
    assert result["complete"] is True
    assert result["run_manifest"] == str(manifest)
    convergence = result["solver"]["convergence"]
    assert convergence["failed"] is True
    assert convergence["iterations"] == 16
    assert convergence["residual"] == 0.0022


def test_submit_local_tasks_captures_init_log(monkeypatch, tmp_path):
    site, _closed = make_site(monkeypatch)
    site.executable = "/solver"
    site.threads_per_worker = 2
    submissions = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append(
                {
                    "func": func,
                    "job_file": job_file,
                    "task_id": task_id,
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            return DummyFuture(
                {
                    "task_id": task_id,
                    "status": "success",
                    "stdout": str(
                        tmp_path / "logs" / local_module._task_log_name(task_id)
                    ),
                }
            )

    class FakeJob(DummyJob):
        def __init__(self):
            super().__init__()
            self._file = tmp_path / "job.json"
            self._file.write_text("{}")
            self._stdout_path = tmp_path / "logs"

        def save(self):
            return self._file

        def task_run_plan(self, reuse=True, force=False):
            return {
                "pending_indices": [0],
                "current_tasks": [],
                "reused_tasks": [],
            }

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "init.log").write_text("stale init log")
    site._dask_client = FakeClient()

    submission = site._submit_local_tasks(FakeJob())

    assert isinstance(submission, local_module.LocalTaskSubmission)
    assert submissions[0]["task_id"] == local_module.MESH_TASK_ID
    assert submissions[0]["kwargs"]["n_threads"] == 2
    assert submissions[0]["kwargs"]["resources"] == {"CPU": 2}
    assert submissions[0]["kwargs"]["stdout_dir"] == str(tmp_path / "logs")
    assert submissions[1]["task_id"] == 0
    assert submissions[1]["kwargs"]["stdout_dir"] == str(tmp_path / "logs")
    assert not (tmp_path / "logs" / "init.log").exists()


def test_submit_local_tasks_runs_init_with_all_auto_threads(monkeypatch, tmp_path):
    site, _closed = make_site(monkeypatch)
    site.executable = "/solver"
    site.config.cores = 8
    site.config.memory = 8000
    submissions = []
    initialized = []
    closed = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append({"task_id": task_id, "kwargs": kwargs})
            return DummyFuture({"task_id": task_id, "status": "success"})

    def fake_initialize(n_workers=None):
        initialized.append(n_workers)
        (
            site._active_n_workers,
            site._active_threads_per_worker,
            site._active_memory_per_worker,
        ) = site._cluster_settings(n_workers)
        site._dask_client = FakeClient()

    def fake_close(**kwargs):
        closed.append(kwargs)
        site._dask_client = None
        site._active_n_workers = None
        site._active_threads_per_worker = None
        site._active_memory_per_worker = None

    class FakeJob(DummyJob):
        def __init__(self):
            super().__init__()
            self._file = tmp_path / "job.json"
            self._file.write_text("{}")
            self._stdout_path = tmp_path / "logs"

        def save(self):
            return self._file

        def task_run_plan(self, reuse=True, force=False):
            return {
                "pending_indices": [0, 1, 2, 3],
                "current_tasks": [],
                "reused_tasks": [],
            }

    site.close = fake_close
    monkeypatch.setattr(site, "_initialize_dask", fake_initialize)

    site._submit_local_tasks(FakeJob())

    assert initialized == [1, 4]
    assert closed == [{"wait": True, "retire": True}]
    assert submissions[0]["task_id"] == local_module.MESH_TASK_ID
    assert submissions[0]["kwargs"]["n_threads"] == 8
    assert submissions[0]["kwargs"]["resources"] == {"CPU": 8}
    assert [item["kwargs"]["n_threads"] for item in submissions[1:]] == [2, 2, 2, 2]


def test_submit_local_tasks_preserves_explicit_n_workers(monkeypatch, tmp_path):
    site, _closed = make_site(monkeypatch)
    site.executable = "/solver"
    site.n_workers = 4
    site.config.cores = 8
    submissions = []
    initialized = []
    closed = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append({"task_id": task_id, "kwargs": kwargs})
            return DummyFuture({"task_id": task_id, "status": "success"})

    def fake_initialize(n_workers=None):
        initialized.append(n_workers)
        (
            site._active_n_workers,
            site._active_threads_per_worker,
            site._active_memory_per_worker,
        ) = site._cluster_settings(n_workers)
        site._dask_client = FakeClient()

    def fake_close(**kwargs):
        closed.append(kwargs)
        site._dask_client = None
        site._active_n_workers = None
        site._active_threads_per_worker = None
        site._active_memory_per_worker = None

    class FakeJob(DummyJob):
        def __init__(self):
            super().__init__()
            self._file = tmp_path / "job.json"
            self._file.write_text("{}")
            self._stdout_path = tmp_path / "logs"

        def save(self):
            return self._file

        def task_run_plan(self, reuse=True, force=False):
            return {
                "pending_indices": [0, 1],
                "current_tasks": [],
                "reused_tasks": [],
            }

    site.close = fake_close
    monkeypatch.setattr(site, "_initialize_dask", fake_initialize)

    site._submit_local_tasks(FakeJob())

    assert initialized == [4]
    assert closed == []
    assert site._active_n_workers == 4
    assert submissions[0]["task_id"] == local_module.MESH_TASK_ID
    assert submissions[0]["kwargs"]["n_threads"] == 2
    assert [item["kwargs"]["n_threads"] for item in submissions[1:]] == [2, 2]


def test_submit_local_tasks_force_run_disables_reuse_and_passes_fresh(
    monkeypatch, tmp_path
):
    site, _closed = make_site(monkeypatch)
    site.executable = "/solver"
    site.threads_per_worker = 2
    submissions = []
    plan_calls = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append({"task_id": task_id, "kwargs": kwargs})
            return DummyFuture({"task_id": task_id, "status": "success"})

    class FakeJob(DummyJob):
        def __init__(self):
            super().__init__()
            self._file = tmp_path / "job.json"
            self._file.write_text("{}")
            self._stdout_path = tmp_path / "logs"

        def save(self):
            return self._file

        def task_run_plan(self, reuse=True, force=False):
            plan_calls.append({"reuse": reuse, "force": force})
            return {
                "pending_indices": [0],
                "current_tasks": [],
                "reused_tasks": [],
            }

    site._dask_client = FakeClient()

    site._submit_local_tasks(FakeJob(), force_run=True)

    assert plan_calls == [{"reuse": False, "force": True}]
    assert [item["task_id"] for item in submissions] == [local_module.MESH_TASK_ID, 0]
    assert all(item["kwargs"]["fresh"] is True for item in submissions)


def test_auto_dask_sizing_refreshes_for_larger_later_job(monkeypatch):
    site, _closed = make_site(monkeypatch)
    site.config.cores = 16
    site.config.memory = 16000
    site._dask_client = object()
    site._active_n_workers = 1
    site._active_threads_per_worker = 16
    site._active_memory_per_worker = 14400
    closed = []
    initialized = []

    def fake_close(**kwargs):
        closed.append(kwargs)
        site._dask_client = None
        site._active_n_workers = None
        site._active_threads_per_worker = None
        site._active_memory_per_worker = None

    def fake_initialize(n_workers=None):
        initialized.append(n_workers)
        (
            site._active_n_workers,
            site._active_threads_per_worker,
            site._active_memory_per_worker,
        ) = site._cluster_settings(n_workers)
        site._dask_client = object()

    site.close = fake_close
    monkeypatch.setattr(site, "_initialize_dask", fake_initialize)

    site._ensure_dask_for_tasks(16)

    assert closed == [{"wait": True, "retire": True}]
    assert initialized == [16]
    assert site.n_workers is None
    assert site.threads_per_worker is None
    assert site._active_n_workers == 16
    assert site._active_threads_per_worker == 1


def test_initialize_dask_preserves_auto_requested_settings(monkeypatch):
    captured = {}

    class FakeCluster:
        dashboard_link = "http://127.0.0.1:12345/status"

        def __init__(self, **kwargs):
            captured["cluster"] = kwargs

        def scale(self, _workers):
            pass

        def close(self, timeout=30.0, fast=False):
            captured["cluster_closed"] = {"timeout": timeout, "fast": fast}

    class FakeClient:
        def __init__(self, cluster, timeout):
            captured["client"] = {"cluster": cluster, "timeout": timeout}

        def close(self, timeout=30.0):
            captured["client_closed"] = {"timeout": timeout}

    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    monkeypatch.setattr(local_module, "LocalCluster", FakeCluster)
    monkeypatch.setattr(local_module, "Client", FakeClient)

    site = LocalSite()
    site.config.cores = 16
    site.config.memory = 16000

    site._initialize_dask(1)

    assert site.n_workers is None
    assert site.threads_per_worker is None
    assert site.memory_per_worker is None
    assert site._active_n_workers == 1
    assert site._active_threads_per_worker == 16
    assert site._active_memory_per_worker == 14400
    assert captured["cluster"]["n_workers"] == 1
    assert captured["cluster"]["threads_per_worker"] == 16
    assert captured["cluster"]["memory_limit"] == "14400MB"


def test_close_uses_cluster_close_without_retiring_workers(monkeypatch):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    site = LocalSite()
    calls = []

    class FakeClient:
        def cancel(self, futures, force=False):
            calls.append(("cancel", list(futures), force))

        def retire_workers(self, *args, **kwargs):
            raise AssertionError("close should not explicitly retire local workers")

        def close(self, timeout=30.0):
            calls.append(("client.close", timeout))

    class FakeCluster:
        def scale(self, workers):
            raise AssertionError("close should not scale local cluster to zero")

        def close(self, timeout=30.0, fast=False):
            calls.append(("cluster.close", timeout, fast))

    future = DummyFuture()
    site._dask_client = FakeClient()
    site._dask_cluster = FakeCluster()
    site._futures = [future]
    site._closed = False
    site._active_n_workers = 1
    site._active_threads_per_worker = 16
    site._active_memory_per_worker = 14400

    site.close(wait=True, retire=True, timeout=7.0)

    assert calls == [
        ("cancel", [future], True),
        ("client.close", 7.0),
        ("cluster.close", 7.0, False),
    ]
    assert site._dask_client is None
    assert site._dask_cluster is None
    assert site._futures == []
    assert site._active_n_workers is None


def test_local_submit_force_run_bypasses_current_skip(monkeypatch):
    site, _closed = make_site(monkeypatch)
    seen = {}

    class CurrentJob(DummyJob):
        name = "current-job"

        def is_run_current(self):
            return True

    def fake_submit(job, **kwargs):
        seen["kwargs"] = kwargs
        return local_module.LocalTaskSubmission(
            futures=[DummyFuture()], task_plan={"pending_indices": [0]}
        )

    monkeypatch.setattr(site, "_submit_local_tasks", fake_submit)

    run = site.submit(CurrentJob(), force_run=True)

    assert run.mode == "local"
    assert seen["kwargs"]["force_run"] is True
    assert run.backend["fresh"] is True


def test_submit_local_tasks_reports_init_log_on_failure(monkeypatch, tmp_path):
    site, _closed = make_site(monkeypatch)
    site.executable = "/solver"
    site.threads_per_worker = 2
    init_log = tmp_path / "logs" / "init.log"

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            return DummyFuture(
                {
                    "task_id": task_id,
                    "status": "error",
                    "error": "bad mesh",
                    "stdout": str(init_log),
                }
            )

    class FakeJob(DummyJob):
        def __init__(self):
            super().__init__()
            self._file = tmp_path / "job.json"
            self._file.write_text("{}")
            self._stdout_path = tmp_path / "logs"

        def save(self):
            return self._file

        def task_run_plan(self, reuse=True, force=False):
            return {
                "pending_indices": [0],
                "current_tasks": [],
                "reused_tasks": [],
            }

    job = FakeJob()
    site._dask_client = FakeClient()

    try:
        site._submit_local_tasks(job)
    except RuntimeError as exc:
        assert "Mesh task failed: bad mesh" in str(exc)
        assert str(init_log) in str(exc)
    else:
        raise AssertionError("Expected mesh failure")
    assert job.states[-1][0] == "failed"
    assert job.states[-1][1]["mesh"]["stdout"] == str(init_log)


def test_local_wait_runs_pack_after_frequency_tasks(monkeypatch, tmp_path):
    site, closed = make_site(monkeypatch)
    site.threads_per_worker = 2
    submissions = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append(
                {
                    "func": func,
                    "job_file": job_file,
                    "task_id": task_id,
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            return DummyFuture({"task_id": task_id, "status": "success"})

    site._dask_client = FakeClient()
    job = DummyJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [DummyFuture({"task_id": 0, "status": "success"})]
    run = make_run(site, job, futures)
    run.backend["pack_after_tasks"] = True

    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )

    result = site._wait_local_run(run)

    assert result.successful
    assert submissions[-1]["func"] is run_task
    assert submissions[-1]["task_id"] == local_module.PACK_TASK_ID
    assert submissions[-1]["kwargs"]["stdout_dir"] == str(job._stdout_path)
    assert job.removed_packed
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["pack"]["task_id"] == local_module.PACK_TASK_ID
    assert closed == [{"wait": True, "retire": True}]


def test_local_wait_smooth_only_runs_imaging_postprocess(monkeypatch, tmp_path):
    site, closed = make_site(monkeypatch)
    submissions = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append({"task_id": task_id, "kwargs": kwargs})
            return DummyFuture({"task_id": task_id, "status": "success"})

    def fake_ensure(task_count):
        site._active_threads_per_worker = 3
        site._dask_client = FakeClient()

    class FakeJob(DummyJob):
        def __init__(self):
            super().__init__()
            self._file = tmp_path / "job.json"
            self._file.write_text("{}")
            self._stdout_path = tmp_path / "logs"

    monkeypatch.setattr(site, "_ensure_dask_for_tasks", fake_ensure)
    job = FakeJob()
    run = make_run(site, job, futures=[])
    run.backend["smooth_only"] = True
    run.backend["task_plan"] = {
        "skipped_task_records": [{"task": 1, "status": "succeeded"}]
    }

    result = site._wait_local_run(run)

    assert result.status.state == "completed"
    assert submissions[0]["task_id"] == local_module.SMOOTH_TASK_ID
    assert submissions[0]["kwargs"]["n_threads"] == 3
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["smooth"]["status"] == "success"
    assert closed == [{"wait": True, "retire": True}]


def test_local_watch_reuses_terminal_smooth_only_result(monkeypatch, tmp_path):
    site, _closed = make_site(monkeypatch)
    smooth_future = OneShotFuture(
        {"task_id": local_module.SMOOTH_TASK_ID, "status": "success"}
    )

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            return smooth_future

    def fake_ensure(task_count):
        site._active_threads_per_worker = 3
        site._dask_client = FakeClient()

    class FakeJob(DummyJob):
        def __init__(self):
            super().__init__()
            self._file = tmp_path / "job.json"
            self._stdout_path = tmp_path / "logs"

    monkeypatch.setattr(site, "_ensure_dask_for_tasks", fake_ensure)
    job = FakeJob()
    run = RunHandle(
        site=site,
        job=job,
        id="local:test",
        poll_interval=0.0,
        _status_fn=site._poll_local_run,
        _wait_fn=site._wait_local_run,
        _finalize_fn=site._finalize_local_run,
    )
    run.backend["futures"] = []
    run.backend["smooth_only"] = True
    run.backend["task_plan"] = {
        "skipped_task_records": [{"task": 1, "status": "succeeded"}]
    }
    run.backend["shutdown_on_completion"] = False

    statuses = list(run.watch(timeout=1.0, poll_interval=0.0))

    assert [status.state for status in statuses] == ["completed"]
    assert smooth_future.result_calls == 1
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["tasks"] == [{"task": 1, "status": "succeeded"}]
    assert job.states[-1][1]["smooth"] == {
        "task_id": local_module.SMOOTH_TASK_ID,
        "status": "success",
    }


def test_local_wait_runs_smooth_after_imaging_frequency_tasks(monkeypatch, tmp_path):
    site, closed = make_site(monkeypatch)
    site.threads_per_worker = 2
    submissions = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append(
                {
                    "func": func,
                    "job_file": job_file,
                    "task_id": task_id,
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            return DummyFuture({"task_id": task_id, "status": "success"})

    class FakeImagingJob(DummyJob):
        pass

    monkeypatch.setattr(local_module, "ImagingJob", FakeImagingJob)
    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )
    site._dask_client = FakeClient()
    job = FakeImagingJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [DummyFuture({"task_id": 0, "status": "success"})]
    run = make_run(site, job, futures)
    run.backend["pack_after_tasks"] = False

    result = site._wait_local_run(run)

    assert result.successful
    assert submissions[-1]["func"] is run_task
    assert submissions[-1]["task_id"] == local_module.SMOOTH_TASK_ID
    assert submissions[-1]["kwargs"]["stdout_dir"] == str(job._stdout_path)
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["smooth"]["task_id"] == local_module.SMOOTH_TASK_ID
    assert result.status.raw["smooth"]["task_id"] == local_module.SMOOTH_TASK_ID
    assert closed == [{"wait": True, "retire": True}]


def test_local_watch_runs_smooth_before_yielding_imaging_completion(
    monkeypatch, tmp_path
):
    site, _closed = make_site(monkeypatch)
    site.threads_per_worker = 2
    submissions = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append({"func": func, "task_id": task_id, "kwargs": kwargs})
            return DummyFuture({"task_id": task_id, "status": "success"})

    class FakeImagingJob(DummyJob):
        pass

    monkeypatch.setattr(local_module, "ImagingJob", FakeImagingJob)
    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )
    site._dask_client = FakeClient()
    job = FakeImagingJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [DummyFuture({"task_id": 0, "status": "success"})]
    run = RunHandle(
        site=site,
        job=job,
        id="local:test",
        poll_interval=0.0,
        _status_fn=site._poll_local_run,
        _wait_fn=site._wait_local_run,
        _finalize_fn=site._finalize_local_run,
    )
    run.backend["futures"] = futures
    run.backend["pack_after_tasks"] = False
    run.backend["shutdown_on_completion"] = False
    site._futures.extend(futures)

    statuses = list(run.watch(timeout=1.0, poll_interval=0.0))

    assert [status.state for status in statuses] == ["completed"]
    assert [item["task_id"] for item in submissions] == [local_module.SMOOTH_TASK_ID]
    assert statuses[-1].raw["smooth"]["task_id"] == local_module.SMOOTH_TASK_ID
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["smooth"]["task_id"] == local_module.SMOOTH_TASK_ID


def test_local_watch_reuses_terminal_poll_results_when_finalizing_imaging(
    monkeypatch, tmp_path
):
    site, _closed = make_site(monkeypatch)
    site.threads_per_worker = 2
    submissions = []
    smooth_future = OneShotFuture(
        {"task_id": local_module.SMOOTH_TASK_ID, "status": "success"}
    )

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append({"func": func, "task_id": task_id, "kwargs": kwargs})
            return smooth_future

    class FakeImagingJob(DummyJob):
        pass

    monkeypatch.setattr(local_module, "ImagingJob", FakeImagingJob)
    site._dask_client = FakeClient()
    job = FakeImagingJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    frequency_future = OneShotFuture({"task_id": 0, "status": "success"})
    run = RunHandle(
        site=site,
        job=job,
        id="local:test",
        poll_interval=0.0,
        _status_fn=site._poll_local_run,
        _wait_fn=site._wait_local_run,
        _finalize_fn=site._finalize_local_run,
    )
    run.backend["futures"] = [frequency_future]
    run.backend["pack_after_tasks"] = False
    run.backend["shutdown_on_completion"] = False
    site._futures.append(frequency_future)

    statuses = list(run.watch(timeout=1.0, poll_interval=0.0))

    assert [status.state for status in statuses] == ["completed"]
    assert frequency_future.result_calls == 1
    assert smooth_future.result_calls == 1
    assert [item["task_id"] for item in submissions] == [local_module.SMOOTH_TASK_ID]
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["tasks"] == [{"task_id": 0, "status": "success"}]
    assert job.states[-1][1]["smooth"] == {
        "task_id": local_module.SMOOTH_TASK_ID,
        "status": "success",
    }


def test_local_watch_runs_pack_before_yielding_completed_status(monkeypatch, tmp_path):
    site, _closed = make_site(monkeypatch)
    site.threads_per_worker = 2
    submissions = []

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            submissions.append({"func": func, "task_id": task_id, "kwargs": kwargs})
            return DummyFuture({"task_id": task_id, "status": "success"})

    site._dask_client = FakeClient()
    job = DummyJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [DummyFuture({"task_id": 0, "status": "success"})]
    run = RunHandle(
        site=site,
        job=job,
        id="local:test",
        poll_interval=0.0,
        _status_fn=site._poll_local_run,
        _wait_fn=site._wait_local_run,
        _finalize_fn=site._finalize_local_run,
    )
    run.backend["futures"] = futures
    run.backend["pack_after_tasks"] = True
    run.backend["shutdown_on_completion"] = False
    site._futures.extend(futures)

    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )

    statuses = list(run.watch(timeout=1.0, poll_interval=0.0))

    assert [status.state for status in statuses] == ["completed"]
    assert submissions[-1]["func"] is run_task
    assert submissions[-1]["task_id"] == local_module.PACK_TASK_ID
    assert job.removed_packed
    assert statuses[-1].raw["pack"]["task_id"] == local_module.PACK_TASK_ID
    assert job.states[-1][0] == "completed"


def test_local_finalize_does_not_zero_timeout_dask_wait(monkeypatch):
    site, _closed = make_site(monkeypatch)
    job = DummyJob()
    run = make_run(site, job, [DummyFuture()])
    seen = []

    def fake_wait(futures, timeout=None):
        seen.append(timeout)
        return SimpleNamespace(not_done=[])

    monkeypatch.setattr(local_module, "wait", fake_wait)

    result = site._finalize_local_run(
        run,
        JobStatus(state="completed", return_code=0, job_id=run.id),
    )

    assert result.successful
    assert seen == [None]


def test_local_wait_reports_failed_frequency_tasks_without_failing_run(
    monkeypatch, tmp_path
):
    site, closed = make_site(monkeypatch)
    job = DummyJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [
        DummyFuture({"task_id": 0, "status": "success", "complete": True}),
        DummyFuture({"task_id": 1, "status": "error", "complete": True}),
    ]
    run = make_run(site, job, futures)
    run.backend["pack_after_tasks"] = False

    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )

    result = site._wait_local_run(run)

    assert result.successful
    assert result.status.state == "completed"
    assert result.status.message == "tasks: 1 succeeded, 1 failed, 2 complete, 2 total"
    assert result.status.raw["task_summary"] == {
        "total": 2,
        "complete": 2,
        "succeeded": 1,
        "failed": 1,
        "not_run": 0,
    }
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["tasks"] == [
        {"task_id": 0, "status": "success", "complete": True},
        {"task_id": 1, "status": "error", "complete": True},
    ]
    assert job.states[-1][1]["errors"] == [
        {"task_id": 1, "status": "error", "complete": True}
    ]
    assert closed == [{"wait": True, "retire": True}]


def test_local_wait_fails_when_frequency_failures_exceed_tolerance(
    monkeypatch, tmp_path
):
    site, _closed = make_site(monkeypatch)
    job = DummyJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [
        DummyFuture({"task_id": 0, "status": "success", "complete": True}),
        DummyFuture({"task_id": 1, "status": "error", "complete": True}),
    ]
    run = make_run(site, job, futures)
    run.backend["pack_after_tasks"] = False
    run.backend["tolerate_failures"] = 0

    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )

    result = site._wait_local_run(run)

    assert result.status.state == "failed"
    assert "failure tolerance exceeded" in result.status.message
    assert result.status.raw["tolerate_failures"] == 0
    assert job.states[-1][0] == "failed"


def test_local_wait_keeps_successful_solve_completed_when_pack_fails(
    monkeypatch, tmp_path
):
    site, _closed = make_site(monkeypatch)
    site.threads_per_worker = 2

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            return DummyFuture({"task_id": task_id, "status": "error"})

    site._dask_client = FakeClient()
    job = DummyJob()
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [DummyFuture({"task_id": 0, "status": "success"})]
    run = make_run(site, job, futures)
    run.backend["pack_after_tasks"] = True

    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )

    result = site._wait_local_run(run)

    assert result.status.state == "completed"
    assert "packing failed" in result.status.message
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["pack_error"]["task_id"] == local_module.PACK_TASK_ID


def test_local_wait_marks_pack_failure_as_run_failure_when_outputs_are_missing(
    monkeypatch, tmp_path
):
    site, _closed = make_site(monkeypatch)
    site.threads_per_worker = 2

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            return DummyFuture({"task_id": task_id, "status": "error"})

    site._dask_client = FakeClient()
    job = DummyJob()
    job.trace_outputs_ok = False
    job._file = tmp_path / "job.json"
    job._stdout_path = tmp_path / "logs"
    futures = [DummyFuture({"task_id": 0, "status": "success"})]
    run = make_run(site, job, futures)
    run.backend["pack_after_tasks"] = True

    monkeypatch.setattr(
        local_module, "wait", lambda futures, timeout=None: SimpleNamespace(not_done=[])
    )

    result = site._wait_local_run(run)

    assert result.status.state == "failed"
    assert result.status.message == "Packing task failed"
    assert job.states[-1][0] == "failed"
    assert job.states[-1][1]["pack"]["task_id"] == local_module.PACK_TASK_ID


def test_local_wait_timeout_cancels_and_releases_futures(monkeypatch):
    site, closed = make_site(monkeypatch)
    job = DummyJob()
    future = DummyFuture()
    future.status = "pending"
    run = make_run(site, job, [future])

    monkeypatch.setattr(
        local_module,
        "wait",
        lambda futures, timeout=None: SimpleNamespace(not_done=list(futures)),
    )

    result = site._wait_local_run(run, timeout=0.01)

    assert result.status.state == "timeout"
    assert job.states[-1][0] == "timeout"
    assert future.cancelled
    assert future.released
    assert site._futures == []
    assert closed == [{"wait": True, "retire": True}]


def test_local_dask_dashboard_uses_loopback_without_wildcard_origin(monkeypatch):
    captured = {}

    class FakeCluster:
        dashboard_link = "http://127.0.0.1:12345/status"

        def __init__(self, **kwargs):
            captured["cluster"] = kwargs

    class FakeClient:
        def __init__(self, cluster, timeout):
            captured["client"] = {"cluster": cluster, "timeout": timeout}

    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    monkeypatch.setattr(local_module, "LocalCluster", FakeCluster)
    monkeypatch.setattr(local_module, "Client", FakeClient)
    monkeypatch.setattr(
        logging.getLogger("distributed"),
        "_frequensolve_dependency_level",
        logging.WARNING,
        raising=False,
    )

    def fake_update(current, config, priority="new"):
        captured["config"] = config
        captured["priority"] = priority
        return current

    monkeypatch.setattr(local_module.dask_config, "update", fake_update)

    site = LocalSite(n_workers=1, threads_per_worker=1)
    site.config.cores = 1
    site.config.memory = 4096

    site._initialize_dask()

    assert captured["cluster"]["dashboard_address"] == "localhost:0"
    assert captured["cluster"]["host"] == "localhost"
    assert captured["cluster"]["silence_logs"] == logging.WARNING
    assert local_module.DASK_LOGGING_PRELOAD in captured["cluster"]["preload"]
    assert local_module.DASK_LOGGING_PRELOAD in captured["cluster"]["preload_nanny"]
    assert (
        local_module.DASK_LOGGING_PRELOAD
        in captured["cluster"]["scheduler_kwargs"]["preload"]
    )
    assert captured["priority"] == "new"
    assert (
        captured["config"]["distributed"]["scheduler"]["dashboard"][
            "bokeh-application"
        ]["allow_websocket_origin"]
        == []
    )
    assert captured["config"]["distributed"]["logging"]["distributed"] == "WARNING"
    assert captured["config"]["distributed"]["logging"]["distributed.core"] == "WARNING"
    assert (
        local_module.DASK_LOGGING_PRELOAD
        in captured["config"]["distributed"]["worker"]["preload"]
    )
    assert (
        local_module.DASK_LOGGING_PRELOAD
        in captured["config"]["distributed"]["scheduler"]["preload"]
    )
    assert (
        local_module.DASK_LOGGING_PRELOAD
        in captured["config"]["distributed"]["nanny"]["preload"]
    )
    assert site.dashboard_url == "http://localhost:12345/status"


def test_dask_logging_preload_quiets_distributed_core_connection_noise():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("distributed.core")
    old_level = logger.level
    old_handlers = list(logger.handlers)
    old_propagate = logger.propagate
    try:
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        configure_dependency_logging(logging.WARNING)
        logger.info("Connection to tcp://127.0.0.1:12345 has been closed.")
        logger.warning("warning still visible")

        output = stream.getvalue()
        assert "Connection to tcp://127.0.0.1:12345 has been closed." not in output
        assert "warning still visible" in output
    finally:
        logger.handlers = old_handlers
        logger.propagate = old_propagate
        logger.setLevel(old_level)
