import asyncio

import pytest

from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunFailedError,
    RunHandle,
    RunResult,
)
from frequensolve.orchestrator.utils.progress import wait_all


class DummyJob:
    name = "failed-job"
    trace_manifest = None
    _stdout_path = None
    run_metadata = None


class DummySite(BaseSite):
    def __init__(self):
        super().__init__()
        self.fetch_traces_called = False

    def submit(self, job, *, check=False, **kwargs):
        return RunHandle(
            site=self,
            job=job,
            id="run-1",
            check=check,
            poll_interval=0.0,
            _status_fn=lambda run: JobStatus(
                state="failed",
                return_code=1,
                job_id="run-1",
                message="solver failed",
            ),
        )

    def fetch_traces(self, job, upscale=1):
        self.fetch_traces_called = True
        return "traces"


def failed_run(site=None):
    site = site or DummySite()
    job = DummyJob()
    return RunHandle(
        site=site,
        job=job,
        id="run-1",
        poll_interval=0.0,
        _status_fn=lambda run: JobStatus(
            state="failed",
            return_code=1,
            job_id="run-1",
            message="solver failed",
        ),
    )


def successful_run(site=None):
    site = site or DummySite()
    job = DummyJob()
    return RunHandle(
        site=site,
        job=job,
        id="run-1",
        poll_interval=0.0,
        _status_fn=lambda run: JobStatus(
            state="completed",
            return_code=0,
            job_id="run-1",
        ),
    )


def test_run_handle_wait_honors_submit_time_fetch_after_success():
    fetch_calls = []
    run = successful_run()
    run._pending_fetch_fn = lambda run: fetch_calls.append(run.id)

    result = run.wait()

    assert result.successful
    assert fetch_calls == ["run-1"]
    assert run.wait() is result
    assert fetch_calls == ["run-1"]


@pytest.mark.parametrize("precompleted", [False, True])
def test_watch_fetches_before_terminal_yield_and_only_once(precompleted):
    run = (
        RunHandle.skipped(DummySite(), DummyJob()) if precompleted else successful_run()
    )
    fetch_calls = []
    run._pending_fetch_fn = lambda handle: fetch_calls.append(handle.id)
    if not precompleted:
        states = iter(["running", "completed"])
        run._status_fn = lambda handle: JobStatus(
            state=next(states), job_id=handle.id, return_code=0
        )

    statuses = []
    for status in run.watch():
        statuses.append(status.state)
        if status.is_complete:
            assert fetch_calls == [run.id]
            break
        assert fetch_calls == []

    assert statuses == (["skipped"] if precompleted else ["running", "completed"])
    assert len(list(run.watch())) == 1
    assert run.wait().successful
    assert fetch_calls == [run.id]


@pytest.mark.parametrize("precompleted", [False, True])
@pytest.mark.parametrize("retry_method", ["watch", "wait"])
def test_watch_download_failure_preserves_fetch_for_retry(precompleted, retry_method):
    run = (
        RunHandle.skipped(DummySite(), DummyJob()) if precompleted else successful_run()
    )
    fetch_calls = []

    def fetch(handle):
        fetch_calls.append(handle.id)
        if len(fetch_calls) == 1:
            raise RuntimeError("temporary download failure")

    run._pending_fetch_fn = fetch
    with pytest.raises(RuntimeError, match="temporary download failure"):
        list(run.watch())

    assert run._result.successful
    assert run._pending_fetch_fn is fetch
    if retry_method == "watch":
        assert len(list(run.watch())) == 1
    else:
        assert run.wait().successful
    assert run._pending_fetch_fn is None
    assert run.wait().successful
    assert fetch_calls == [run.id, run.id]


@pytest.mark.parametrize("state", ["failed", "cancelled", "timeout"])
def test_watch_does_not_fetch_unsuccessful_terminal_outputs(state):
    run = failed_run()
    run._status_fn = lambda handle: JobStatus(
        state=state, job_id=handle.id, return_code=1
    )
    fetch_calls = []
    run._pending_fetch_fn = lambda handle: fetch_calls.append(handle.id)

    assert [status.state for status in run.watch()] == [state]
    assert fetch_calls == []
    with pytest.raises(RunFailedError):
        run.wait()
    assert fetch_calls == []


def test_watch_local_timeout_does_not_fetch(monkeypatch):
    from frequensolve.orchestrator.sites import base

    times = iter([0.0, 2.0])
    monkeypatch.setattr(base.time, "monotonic", lambda: next(times))
    run = successful_run()
    run._status_fn = lambda handle: JobStatus(state="running", job_id=handle.id)
    fetch_calls = []
    run._pending_fetch_fn = lambda handle: fetch_calls.append(handle.id)

    assert [status.state for status in run.watch(timeout=1)] == ["running", "timeout"]
    assert fetch_calls == []
    assert run._result.status.state == "timeout"


def test_precompleted_fetch_intent_retries_after_fetch_failure():
    fetch_calls = []
    run = RunHandle.skipped(DummySite(), DummyJob())

    def fetch(run):
        fetch_calls.append(run.id)
        if len(fetch_calls) == 1:
            raise RuntimeError("temporary download failure")

    run._pending_fetch_fn = fetch

    with pytest.raises(RuntimeError, match="temporary download failure"):
        run.wait()

    assert run.wait().successful
    assert fetch_calls == [None, None]
    assert run.wait().successful
    assert fetch_calls == [None, None]


def test_wait_all_fetches_precompleted_submit_outputs_once():
    fetch_calls = []
    run = RunHandle.skipped(DummySite(), DummyJob())
    run._pending_fetch_fn = lambda run: fetch_calls.append(run.id)

    [result] = wait_all([run], poll_interval=0.0)

    assert result.successful
    assert fetch_calls == [None]
    assert wait_all([run], poll_interval=0.0) == [result]
    assert fetch_calls == [None]


def test_run_handle_wait_raises_by_default_for_failed_run():
    run = failed_run()

    with pytest.raises(
        RunFailedError,
        match=(
            "FrequenSolve run failed: job=failed-job; state=failed; "
            "job_id=run-1; solver failed"
        ),
    ) as exc_info:
        run.wait()

    assert exc_info.value.result.status.state == "failed"


def test_run_handle_wait_check_false_returns_failed_result():
    run = failed_run()

    result = run.wait(check=False)

    assert not result.successful
    assert result.status.state == "failed"


def test_site_submit_defaults_to_non_strict_wait_for_failed_run():
    run = DummySite().submit(DummyJob())

    result = run.wait()

    assert not result.successful
    assert result.status.state == "failed"


def test_site_submit_check_true_restores_strict_wait_for_failed_run():
    run = DummySite().submit(DummyJob(), check=True)

    with pytest.raises(
        RunFailedError,
        match=(
            "FrequenSolve run failed: job=failed-job; state=failed; "
            "job_id=run-1; solver failed"
        ),
    ):
        run.wait()


def test_run_handle_wait_async_raises_by_default_for_failed_run():
    run = failed_run()

    async def wait_for_run():
        return await run.wait_async()

    with pytest.raises(
        RunFailedError,
        match=(
            "FrequenSolve run failed: job=failed-job; state=failed; "
            "job_id=run-1; solver failed"
        ),
    ):
        asyncio.run(wait_for_run())


def test_wait_all_check_false_returns_failed_result():
    run = failed_run()

    [result] = wait_all([run], check=False, poll_interval=0.0)

    assert not result.successful
    assert result.status.state == "failed"


def test_wait_all_check_true_does_not_fetch_failed_outputs():
    fetch_calls = []
    run = failed_run()
    run._pending_fetch_fn = lambda run: fetch_calls.append(run.id)

    with pytest.raises(
        RunFailedError,
        match=(
            "FrequenSolve run failed: job=failed-job; state=failed; "
            "job_id=run-1; solver failed"
        ),
    ):
        wait_all([run], fetch=True, poll_interval=0.0)

    assert fetch_calls == []


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_submit_time_fetch_skips_unsuccessful_terminal_runs(state):
    fetch_calls = []
    run = RunHandle(
        site=DummySite(),
        job=DummyJob(),
        id="run-1",
        poll_interval=0.0,
        _status_fn=lambda run: JobStatus(
            state=state,
            return_code=1,
            job_id="run-1",
        ),
        _pending_fetch_fn=lambda run: fetch_calls.append(run.id),
    )

    with pytest.raises(RunFailedError):
        run.wait()

    assert fetch_calls == []


def test_submit_time_fetch_skips_timed_out_run():
    fetch_calls = []
    run = RunHandle(
        site=DummySite(),
        job=DummyJob(),
        id="run-1",
        poll_interval=0.0,
        _status_fn=lambda run: JobStatus(
            state="running",
            return_code=-1,
            job_id="run-1",
        ),
        _pending_fetch_fn=lambda run: fetch_calls.append(run.id),
    )

    with pytest.raises(RunFailedError):
        run.wait(timeout=0.0)

    assert fetch_calls == []


def test_explicit_non_strict_wait_all_preserves_failed_output_fetch():
    fetch_calls = []
    run = failed_run()
    run._pending_fetch_fn = lambda run: fetch_calls.append(run.id)

    [result] = wait_all([run], fetch=True, check=False, poll_interval=0.0)

    assert not result.successful
    assert fetch_calls == ["run-1"]


def test_explicit_non_strict_wait_all_fetches_precompleted_failure():
    fetch_calls = []
    run = failed_run()
    run._result = run._make_result(run.status())
    run._pending_fetch_fn = lambda run: fetch_calls.append(run.id)

    [result] = wait_all([run], fetch=True, check=False, poll_interval=0.0)

    assert not result.successful
    assert fetch_calls == ["run-1"]
    assert run._pending_fetch_fn is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_site_specific_wait_consumes_submit_time_fetch(asynchronous):
    fetch_calls = []
    site = DummySite()
    job = DummyJob()
    result = RunResult(
        job=job,
        site=site,
        status=JobStatus(state="completed", return_code=0, job_id="run-1"),
    )

    async def wait_async(run, timeout, poll_interval):
        return result

    run = RunHandle(
        site=site,
        job=job,
        id="run-1",
        poll_interval=0.0,
        _generic_wait=False,
        _wait_fn=lambda run, timeout, poll_interval: result,
        _wait_async_fn=wait_async,
        _pending_fetch_fn=lambda run: fetch_calls.append(run.id),
    )

    waited = asyncio.run(run.wait_async()) if asynchronous else run.wait()

    assert waited is result
    assert fetch_calls == ["run-1"]


def test_failed_run_result_traces_raise_before_fetching_outputs():
    site = DummySite()
    result = RunResult(
        job=DummyJob(),
        status=JobStatus(
            state="failed",
            return_code=1,
            job_id="run-1",
            message="solver failed",
        ),
        site=site,
    )

    with pytest.raises(
        RunFailedError,
        match=(
            "FrequenSolve run failed: job=failed-job; state=failed; "
            "job_id=run-1; solver failed"
        ),
    ):
        result.traces(upscale=4)

    assert site.fetch_traces_called is False


def polling_run(outcomes):
    run = successful_run()
    values = iter(outcomes)
    calls = []

    def poll(handle):
        calls.append(handle.id)
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return JobStatus(state=value, job_id=handle.id, return_code=0)

    run._status_fn = poll
    return run, calls


def test_wait_recovers_transient_reads_and_resets_consecutive_failure_bound():
    from frequensolve.orchestrator.utils.status_errors import TransientStatusReadError

    transient = TransientStatusReadError("observe existing run-1; do not resubmit")
    run, calls = polling_run(
        ["running", transient, transient, "running", transient, transient, "completed"]
    )
    fetched = []
    run._pending_fetch_fn = lambda handle: fetched.append(handle.id)
    assert run.wait().successful
    assert calls == ["run-1"] * 7
    assert fetched == ["run-1"]


def test_wait_exhaustion_preserves_handle_for_later_recovery():
    from frequensolve.orchestrator.utils.status_errors import TransientStatusReadError

    transient = TransientStatusReadError("observe existing run-1; do not resubmit")
    run, calls = polling_run([transient] * 3 + ["completed"])
    with pytest.raises(TransientStatusReadError, match="existing run-1"):
        run.wait()
    assert len(calls) == 3
    assert run._result is None
    assert run.wait().successful
    assert len(calls) == 4


def test_transient_read_does_not_restart_timeout_or_poll_after_deadline(monkeypatch):
    from frequensolve.orchestrator.utils import progress
    from frequensolve.orchestrator.utils.status_errors import TransientStatusReadError

    now = [0.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        progress.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    run, calls = polling_run([TransientStatusReadError("temporary"), "completed"])
    run.poll_interval = 2
    result = run.wait(timeout=1, check=False)
    assert result.status.state == "timeout"
    assert len(calls) == 1


def test_transient_read_does_not_block_other_runs():
    from frequensolve.orchestrator.utils.status_errors import TransientStatusReadError

    first, first_calls = polling_run(
        [TransientStatusReadError("temporary"), "completed"]
    )
    second, second_calls = polling_run(["completed"])
    assert all(result.successful for result in wait_all([first, second]))
    assert len(first_calls) == 2
    assert len(second_calls) == 1


@pytest.mark.parametrize(
    "error", [RuntimeError("access denied"), ValueError("malformed")]
)
def test_wait_does_not_retry_permanent_status_errors(error):
    run, calls = polling_run([error, "completed"])
    with pytest.raises(type(error), match=str(error)):
        run.wait()
    assert len(calls) == 1


def test_terminal_failure_after_transient_read_still_raises():
    from frequensolve.orchestrator.utils.status_errors import TransientStatusReadError

    run, calls = polling_run([TransientStatusReadError("temporary"), "failed"])
    with pytest.raises(RunFailedError):
        run.wait()
    assert len(calls) == 2
