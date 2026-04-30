import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Union

TERMINAL_STATES = {
    "completed",
    "complete",
    "failed",
    "cancelled",
    "timeout",
    "skipped",
}

SUCCESS_STATES = {
    "completed",
    "complete",
    "skipped",
}

__all__ = [
    "BaseSite",
    "JobStatus",
    "RunHandle",
    "RunResult",
    "_check_if_notebook",
]


def _wait_for_path(
    path: Union[str, Path], timeout: float = 5.0, poll_interval: float = 0.2
) -> bool:
    """Wait for the given path to exist."""
    waited = 0.0
    path = Path(path)
    while not path.exists() and waited < timeout:
        time.sleep(poll_interval)
        waited += poll_interval
    return path.exists()


def _check_if_notebook() -> bool:
    """Check if we're running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False  # Probably standard Python interpreter


@dataclass
class JobStatus:
    """Status and result information for a command execution.

    This class tracks both immediate execution results (return code, output)
    and ongoing job status information for batch/queued jobs.

    Attributes:
       state (str):
          Current status of the execution:
             "pending":   Job is queued/waiting to start
             "running":   Job is currently executing
             "completed": Job finished successfully
             "failed":    Job failed or was cancelled
             "unknown":   Status cannot be determined
       return_code (int):
          Exit code from the command (0 typically indicates success)
       stdout (str):
          Standard output captured from the command
       stderr (str):
          Standard error output from the command
       job_id (Optional[str]):
          Job identifier for batch/queued jobs
       start_time (Optional[float]):
          Unix timestamp when job started
    """

    state: str = "unknown"
    return_code: int = -1
    stdout: str = ""
    stderr: str = ""
    job_id: Optional[str] = None
    hostname: Optional[str] = None
    message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[float] = None
    start_time: Optional[Union[float, datetime]] = None
    end_time: Optional[Union[float, datetime]] = None

    @property
    def is_queued(self) -> bool:
        return self.state == "pending"

    @property
    def is_running(self) -> bool:
        return self.state == "running"

    @property
    def is_complete(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_successful(self) -> bool:
        return self.state in SUCCESS_STATES and self.return_code in {0, -1}


@dataclass
class RunResult:
    """Final result returned by a completed run handle."""

    job: Any
    status: JobStatus
    site: Optional["BaseSite"] = None
    artifacts: Any = None
    trace_manifest: Any = None
    logs_path: Optional[Path] = None
    run_metadata: Any = None

    @property
    def successful(self) -> bool:
        return self.status.is_successful

    def traces(self, upscale: int = 1):
        if self.site is not None:
            return self.site.fetch_traces(self.job, upscale=upscale)
        from frequensolve.seismic.traces import TraceDataset

        return TraceDataset.from_job(self.job, upscale=upscale)


@dataclass
class RunHandle:
    """Awaitable handle for a submitted or skipped run."""

    site: "BaseSite"
    job: Any
    id: Optional[str] = None
    mode: str = "unknown"
    poll_interval: float = 5.0
    backend: Dict[str, Any] = field(default_factory=dict)
    _status_fn: Optional[Callable[["RunHandle"], JobStatus]] = None
    _wait_fn: Optional[
        Callable[["RunHandle", Optional[float], Optional[float]], RunResult]
    ] = None
    _wait_async_fn: Optional[
        Callable[["RunHandle", Optional[float], Optional[float]], Awaitable[RunResult]]
    ] = None
    _cancel_fn: Optional[Callable[["RunHandle"], None]] = None
    _fetch_fn: Optional[Callable[["RunHandle"], Any]] = None
    _result: Optional[RunResult] = None
    _last_status: JobStatus = field(default_factory=JobStatus)

    @classmethod
    def skipped(
        cls, site: "BaseSite", job: Any, message: str = "Run is current"
    ) -> "RunHandle":
        status = JobStatus(
            state="skipped",
            return_code=0,
            job_id=getattr(job, "_job_id", None),
            message=message,
            end_time=datetime.now(),
        )
        handle = cls(
            site=site, job=job, id=status.job_id, mode="skipped", poll_interval=0.0
        )
        handle._last_status = status
        handle._result = handle._make_result(status)
        return handle

    def __await__(self):
        return self.wait_async().__await__()

    def _make_result(self, status: JobStatus) -> RunResult:
        return RunResult(
            job=self.job,
            status=status,
            site=self.site,
            trace_manifest=getattr(self.job, "trace_manifest", None),
            logs_path=getattr(self.job, "_stdout_path", None),
            run_metadata=getattr(self.job, "run_metadata", None),
        )

    def status(self) -> JobStatus:
        if self._result is not None:
            return self._result.status
        if self._status_fn is not None:
            self._last_status = self._status_fn(self)
        return self._last_status

    def wait(
        self,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        if self._result is not None:
            return self._result
        if self._wait_fn is not None:
            self._result = self._wait_fn(self, timeout, poll_interval)
            return self._result

        interval = self.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()
        last_state = object()
        while True:
            status = self.status()
            if status.state != last_state and hasattr(self.site, "_emit_status"):
                self.site._emit_status(status)
                last_state = status.state
            if status.is_complete:
                self._result = self._make_result(status)
                return self._result
            if timeout is not None and time.monotonic() - start > timeout:
                timeout_status = JobStatus(
                    state="timeout",
                    job_id=self.id,
                    message=f"Timed out waiting for run after {timeout} seconds",
                )
                self._last_status = timeout_status
                self._result = self._make_result(timeout_status)
                return self._result
            time.sleep(interval)

    async def wait_async(
        self,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        if self._result is not None:
            return self._result
        if self._wait_async_fn is not None:
            self._result = await self._wait_async_fn(self, timeout, poll_interval)
            return self._result
        if self._wait_fn is not None:
            return await asyncio.to_thread(self.wait, timeout, poll_interval)

        interval = self.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()
        last_state = object()
        while True:
            status = await asyncio.to_thread(self.status)
            if status.state != last_state and hasattr(self.site, "_emit_status"):
                self.site._emit_status(status)
                last_state = status.state
            if status.is_complete:
                self._result = self._make_result(status)
                return self._result
            if timeout is not None and time.monotonic() - start > timeout:
                timeout_status = JobStatus(
                    state="timeout",
                    job_id=self.id,
                    message=f"Timed out waiting for run after {timeout} seconds",
                )
                self._last_status = timeout_status
                self._result = self._make_result(timeout_status)
                return self._result
            await asyncio.sleep(interval)

    def watch(
        self,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> Iterable[JobStatus]:
        interval = self.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()
        last_state = object()
        while True:
            status = self.status()
            if status.state != last_state:
                yield status
                last_state = status.state
            if status.is_complete:
                self._result = self._make_result(status)
                return
            if timeout is not None and time.monotonic() - start > timeout:
                timeout_status = JobStatus(
                    state="timeout",
                    job_id=self.id,
                    message=f"Timed out waiting for run after {timeout} seconds",
                )
                self._last_status = timeout_status
                yield timeout_status
                self._result = self._make_result(timeout_status)
                return
            time.sleep(interval)

    def cancel(self) -> None:
        if self._cancel_fn is not None:
            self._cancel_fn(self)
            return
        if self.id is None:
            return
        self.site.cancel_job(self.id)

    def fetch(self):
        if self._fetch_fn is not None:
            return self._fetch_fn(self)
        if hasattr(self.site, "fetch_outputs"):
            return self.site.fetch_outputs(self.job)
        return None

    def traces(self, upscale: int = 1):
        self.fetch()
        return self.site.fetch_traces(self.job, upscale=upscale)

    def logs(self, **kwargs):
        if hasattr(self.site, "fetch_logs"):
            return self.site.fetch_logs(self.job, **kwargs)
        if hasattr(self.job, "_stdout_path"):
            return self.job._stdout_path
        return None


@dataclass(kw_only=True)
class BaseSite:
    """Base class for site configuration."""

    _is_notebook: bool = field(default_factory=_check_if_notebook)
    verbose: bool = False

    def _emit(self, message: str, level: int = logging.INFO) -> None:
        """Log a site message and optionally print it for interactive users."""

        logging.getLogger(self.__class__.__module__).log(level, message)
        if getattr(self, "verbose", False):
            print(message)

    def _emit_status(self, status: JobStatus) -> None:
        """Emit a normalized run status update."""

        job = f" {status.job_id}" if status.job_id else ""
        message = f"{self.__class__.__name__}{job}: {status.state}"
        if status.message:
            message = f"{message} - {status.message}"
        self._emit(message)

    def prepare_job(self, job, *, sync_project: bool = False):
        """Persist local job inputs before a run is submitted.

        Site implementations call this before checking run fingerprints or
        transferring files.  It keeps the user-facing lifecycle simple:
        creating a job and calling ``site.submit(job)`` is enough.
        """

        if hasattr(job, "save"):
            job.save()
        project = getattr(getattr(job, "simulation", None), "_project", None)
        if sync_project and project is not None and hasattr(self, "sync"):
            self.sync(project)
        return job

    def submit(self, job, **kwargs) -> RunHandle:
        """Submit a job and return an awaitable run handle."""
        raise NotImplementedError

    def run(
        self,
        job,
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        **submit_kwargs,
    ) -> RunResult:
        """Submit a job and block until completion."""
        return self.submit(job, **submit_kwargs).wait(timeout, poll_interval)

    def handle(
        self, job, job_id: Optional[str] = None, mode: str = "attached"
    ) -> RunHandle:
        """Create a run handle for an existing submitted job."""
        job_id = job_id or getattr(job, "_job_id", None)
        if job_id is None:
            raise ValueError("Cannot create a run handle without a job id")
        poll = getattr(self, "_poll_run", None)
        if poll is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} cannot poll existing submitted jobs"
            )
        return RunHandle(
            site=self,
            job=job,
            id=str(job_id),
            mode=mode,
            poll_interval=getattr(getattr(self, "config", None), "poll_interval", 5),
            _status_fn=poll,
            _cancel_fn=lambda run: self.cancel_job(str(run.id)),
        )

    def cancel_job(self, job_id: str) -> bool | None:
        """Cancel a running job.

        Args:
            job_id: The ID of the job to cancel
        """
        raise NotImplementedError
