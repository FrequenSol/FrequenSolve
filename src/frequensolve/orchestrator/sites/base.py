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

STATUS_COLORS = {
    "pending": "\033[38;5;27m",
    "running": "\033[38;5;28m",
    "complete": "\033[38;5;40m",
    "completed": "\033[38;5;40m",
    "timeout": "\033[38;5;202m",
    "failed": "\033[38;5;160m",
    "cancelled": "\033[38;5;160m",
    "canceled": "\033[38;5;160m",
    "skipped": "\033[38;5;244m",
    "unknown": "\033[38;5;244m",
}
STATUS_PREFIX_COLOR = "\033[38;5;244m"
STATUS_RESET = "\033[0m"

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

    def wavefields(self, upscale: int = 1):
        if self.site is not None:
            return self.site.fetch_wavefields(self.job, upscale=upscale)
        return self.job.wavefields.open(upscale=upscale)

    def output_files(
        self,
        *,
        kind: Optional[str] = None,
        suffix: Optional[Union[str, tuple[str, ...]]] = None,
        base: Optional[Union[str, Path]] = None,
        existing: bool = False,
    ) -> list[Path]:
        """Return output files reported by the completed run."""

        metadata = self.run_metadata or getattr(self.job, "run_metadata", None)
        if metadata is None:
            return []
        return metadata.output_files(
            kind=kind,
            suffix=suffix,
            base=base,
            existing=existing,
        )

    def logs(self, **kwargs):
        if self.site is not None:
            return self.site.fetch_logs(self.job, **kwargs)
        if hasattr(self.job, "_stdout_path"):
            return self.job._stdout_path
        return None


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
    _finalize_fn: Optional[Callable[["RunHandle", JobStatus], RunResult]] = None
    _timeout_fn: Optional[Callable[["RunHandle", JobStatus], RunResult]] = None
    _generic_wait: bool = True
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
        if hasattr(site, "_emit_status"):
            site._emit_status(status)
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

    def _complete_from_status(self, status: JobStatus) -> RunResult:
        if self._result is not None:
            return self._result
        if self._finalize_fn is not None:
            self._result = self._finalize_fn(self, status)
        elif (finalize := self._run_record_finalizer()) is not None:
            finalize(self, status)
            self._result = self._make_result(status)
        elif self._wait_fn is not None:
            self._result = self._wait_fn(self, 0, 0)
        else:
            self._result = self._make_result(status)
        return self._result

    def _run_record_finalizer(self):
        if self.mode != "batch":
            return None
        finalize = getattr(self.site, "_finalize_run_record", None)
        return finalize if callable(finalize) else None

    def wait(
        self,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        if self._result is not None:
            return self._result
        if not self._generic_wait and self._wait_fn is not None:
            self._result = self._wait_fn(self, timeout, poll_interval)
            return self._result
        from frequensolve.orchestrator.progress import wait

        self._result = wait(self, timeout=timeout, poll_interval=poll_interval)
        return self._result

    async def wait_async(
        self,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        if self._result is not None:
            return self._result
        if not self._generic_wait and self._wait_async_fn is not None:
            self._result = await self._wait_async_fn(self, timeout, poll_interval)
            return self._result
        return await asyncio.to_thread(self.wait, timeout, poll_interval)

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
            if status.is_complete:
                status = self._complete_from_status(status).status
            if status.state != last_state:
                yield status
                last_state = status.state
            if status.is_complete:
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

    def wavefields(self, upscale: int = 1):
        self.fetch()
        return self.site.fetch_wavefields(self.job, upscale=upscale)

    def logs(self, **kwargs):
        if self.site is not None:
            return self.site.fetch_logs(self.job, **kwargs)
        if hasattr(self.job, "_stdout_path"):
            return self.job._stdout_path
        return None


@dataclass(kw_only=True)
class BaseSite:
    """Base class for site configuration."""

    _is_notebook: bool = field(default_factory=_check_if_notebook)
    verbose: bool = False

    def _emit(
        self,
        message: str,
        level: int = logging.INFO,
        *,
        force: bool = False,
    ) -> None:
        """Log a site message and optionally print it for interactive users."""

        logging.getLogger(self.__class__.__module__).log(level, message)
        if force or getattr(self, "verbose", False):
            print(message)

    def _emit_status(self, status: JobStatus, *, force: bool = True) -> None:
        """Emit a normalized run status update."""

        job = f" {status.job_id}" if status.job_id else ""
        state = str(status.state)
        color = STATUS_COLORS.get(state.lower(), STATUS_COLORS["unknown"])
        message = (
            f"{STATUS_PREFIX_COLOR}{self.__class__.__name__}{job}: "
            f"{color}{state}{STATUS_RESET}"
        )
        if status.message:
            message = f"{message} - {status.message}"
        self._emit(message, force=force)

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

    @staticmethod
    def _as_jobs(job: Any) -> tuple[list[Any], bool]:
        if isinstance(job, (list, tuple)):
            return list(job), False
        return [job], True

    @staticmethod
    def _frequency_task(job: Any, frequency: Any) -> int:
        try:
            target = complex(frequency)
        except (TypeError, ValueError):
            target = complex(float(frequency))

        for index, value in enumerate(getattr(job, "f_list", []), start=1):
            try:
                candidate = complex(value)
            except (TypeError, ValueError):
                continue
            if abs(candidate - target) <= 1e-9 * max(1.0, abs(target)):
                return index
        raise ValueError(f"Frequency {frequency!r} is not part of job {job.name!r}")

    @staticmethod
    def _task_log_candidates(log_dir: Union[str, Path], task: int) -> list[Path]:
        log_dir = Path(log_dir)
        return [
            log_dir / f"task_{task}.log",
            log_dir / f"task_{task}.txt",
            log_dir / f"task_{task}.out",
        ]

    @classmethod
    def _select_log_path(
        cls,
        job: Any,
        log_dir: Union[str, Path],
        *,
        task: Optional[int] = None,
        frequency: Optional[Any] = None,
    ) -> Path:
        if task is not None and frequency is not None:
            raise ValueError("Pass either `task` or `frequency`, not both")
        if frequency is not None:
            task = cls._frequency_task(job, frequency)
        log_dir = Path(log_dir)
        if task is None:
            return log_dir
        if task < 1:
            raise ValueError("Log task numbers are one-based and must be >= 1")
        for path in cls._task_log_candidates(log_dir, int(task)):
            if path.exists():
                return path
        candidates = ", ".join(
            str(path) for path in cls._task_log_candidates(log_dir, int(task))
        )
        raise FileNotFoundError(
            f"No log file found for task {task}; checked {candidates}"
        )

    @staticmethod
    def _latest_task_log_path(log_dir: Union[str, Path]) -> Optional[Path]:
        log_path = Path(log_dir)
        if not log_path.is_dir():
            return None
        best_index = -1
        best_path = None
        for pattern in ("task_*.log", "task_*.txt", "task_*.out"):
            for file in log_path.glob(pattern):
                try:
                    index = int(file.stem.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    continue
                if index > best_index:
                    best_index = index
                    best_path = file
        return best_path

    @staticmethod
    def _print_file_with_header(path: Union[str, Path], header: str) -> None:
        path = Path(path)
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            print(f"--- {header} (could not read: {exc}) ---")
            return
        print(f"\n{'=' * 60}\n{header}\n{path}\n{'=' * 60}\n{text}\n")

    def _show_logs(
        self,
        selection: Path,
        *,
        job_name: Optional[str] = None,
        label: str = "log",
    ) -> None:
        if selection.is_file():
            self._print_file_with_header(selection, f"[{job_name or 'job'}] {label}")
            return
        latest = self._latest_task_log_path(selection)
        if latest is not None:
            self._print_file_with_header(
                latest,
                f"[{job_name or 'job'}] latest task log",
            )

    def fetch_wavefields(
        self,
        job: Any,
        *,
        upscale: int = 1,
        path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
        **_: Any,
    ) -> Any:
        """Return local wavefield outputs for one job or a mapping for many jobs."""

        jobs, single = self._as_jobs(job)
        mapped_project = project_path if project_path is not None else path
        mapped_project = Path(mapped_project).resolve() if mapped_project else None
        out = {
            item.name: item.wavefields.open(
                upscale=upscale,
                project_path=mapped_project,
            )
            for item in jobs
        }
        if single:
            return out[jobs[0].name]
        return out

    def fetch_logs(
        self,
        job: Any,
        *,
        local_dir: Optional[Union[str, Path]] = None,
        task: Optional[int] = None,
        frequency: Optional[Any] = None,
        show: bool = False,
        **_: Any,
    ) -> Union[Path, Dict[str, Path]]:
        """Return local logs for one job or a mapping for multiple jobs.

        ``task`` is one-based. ``frequency`` selects the matching frequency in
        ``job.f_list`` and returns that task's log file.
        """

        jobs, single = self._as_jobs(job)
        requested = Path(local_dir) if local_dir is not None else None
        out: Dict[str, Path] = {}
        for item in jobs:
            if requested is None:
                log_dir = Path(getattr(item, "_stdout_path"))
            elif single:
                log_dir = requested
            else:
                log_dir = requested / item.name
            selected = self._select_log_path(
                item,
                log_dir,
                task=task,
                frequency=frequency,
            )
            if show:
                self._show_logs(selected, job_name=getattr(item, "name", None))
            out[item.name] = selected
        if single:
            return out[jobs[0].name]
        return out

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

    def wait(
        self,
        runs: Iterable[RunHandle],
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        fetch: bool = False,
    ) -> list[RunResult]:
        """Wait for multiple run handles while rendering one combined status."""

        return self.wait_all(
            runs,
            timeout=timeout,
            poll_interval=poll_interval,
            fetch=fetch,
        )

    def wait_all(
        self,
        runs: Iterable[RunHandle],
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        fetch: bool = False,
    ) -> list[RunResult]:
        """Wait for many submitted runs and return results in input order."""

        from frequensolve.orchestrator.progress import wait_all

        return wait_all(
            runs,
            timeout=timeout,
            poll_interval=poll_interval,
            fetch=fetch,
        )

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
