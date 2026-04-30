from types import SimpleNamespace

from frequensolve.orchestrator.sites import local as local_module
from frequensolve.orchestrator.sites.base import RunHandle
from frequensolve.orchestrator.sites.local import LocalSite


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


def test_local_wait_releases_futures_and_closes_by_default(monkeypatch):
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
