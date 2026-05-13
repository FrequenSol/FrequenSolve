import io
import logging
from types import SimpleNamespace

from frequensolve.orchestrator.sites import local as local_module
from frequensolve.orchestrator.sites.base import RunHandle
from frequensolve.orchestrator.sites.dask_logging import configure_dependency_logging
from frequensolve.orchestrator.sites.local import LocalSite, run_task


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


class DummyJob:
    name = "local-job"
    trace_manifest = None
    _stdout_path = None
    run_metadata = None

    def __init__(self):
        self.states = []

    def write_run_state(self, status="completed", **extra):
        self.states.append((status, extra))


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
    assert (
        "\033[38;5;244mLocalSite local:test: \033[38;5;28mrunning\033[0m"
        in captured.out
    )
    assert (
        "\033[38;5;244mLocalSite local:test: \033[38;5;40mcompleted\033[0m"
        in captured.out
    )


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
        "-j",
        str(job_file),
        "--pack",
    ]
    assert (tmp_path / "logs" / "pack.log").read_text() == (
        f"[INFO] {local_module.logger.name}: Executing: "
        f"/solver -nthreads 2 -j {job_file} --pack\n"
        "solver output\n"
    )


def test_submit_local_tasks_captures_mesh_log(monkeypatch, tmp_path):
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

        def task_run_plan(self, reuse=True):
            return {
                "pending_indices": [0],
                "current_tasks": [],
                "reused_tasks": [],
            }

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "mesh.log").write_text("stale mesh log")
    site._dask_client = FakeClient()

    submission = site._submit_local_tasks(FakeJob())

    assert isinstance(submission, local_module.LocalTaskSubmission)
    assert submissions[0]["task_id"] == local_module.MESH_TASK_ID
    assert submissions[0]["kwargs"]["stdout_dir"] == str(tmp_path / "logs")
    assert submissions[1]["task_id"] == 0
    assert submissions[1]["kwargs"]["stdout_dir"] == str(tmp_path / "logs")
    assert not (tmp_path / "logs" / "mesh.log").exists()


def test_submit_local_tasks_reports_mesh_log_on_failure(monkeypatch, tmp_path):
    site, _closed = make_site(monkeypatch)
    site.executable = "/solver"
    site.threads_per_worker = 2
    mesh_log = tmp_path / "logs" / "mesh.log"

    class FakeClient:
        def submit(self, func, job_file, task_id, *args, **kwargs):
            return DummyFuture(
                {
                    "task_id": task_id,
                    "status": "error",
                    "error": "bad mesh",
                    "stdout": str(mesh_log),
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

        def task_run_plan(self, reuse=True):
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
        assert str(mesh_log) in str(exc)
    else:
        raise AssertionError("Expected mesh failure")
    assert job.states[-1][0] == "failed"
    assert job.states[-1][1]["mesh"]["stdout"] == str(mesh_log)


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
    assert job.states[-1][0] == "completed"
    assert job.states[-1][1]["pack"]["task_id"] == local_module.PACK_TASK_ID
    assert closed == [{"wait": True, "retire": True}]


def test_local_wait_marks_pack_failure_as_run_failure(monkeypatch, tmp_path):
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
