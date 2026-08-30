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
