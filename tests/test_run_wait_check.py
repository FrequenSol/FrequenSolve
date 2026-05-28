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

    def submit(self, job, **kwargs):
        return RunHandle(
            site=self,
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
    run._fetch_fn = lambda run: fetch_calls.append(run.id)

    with pytest.raises(
        RunFailedError,
        match=(
            "FrequenSolve run failed: job=failed-job; state=failed; "
            "job_id=run-1; solver failed"
        ),
    ):
        wait_all([run], fetch=True, poll_interval=0.0)

    assert fetch_calls == []


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
