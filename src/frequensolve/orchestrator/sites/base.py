"""Shared run-handle and site abstractions for local, cloud, and HPC execution."""

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
    "RunFailedError",
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
        state: Current lifecycle state, such as ``"pending"``, ``"running"``,
            ``"completed"``, ``"failed"``, or ``"unknown"``.
        return_code: Exit code from the command. ``0`` typically indicates
            success.
        stdout: Standard output captured from the command.
        stderr: Standard error output captured from the command.
        job_id: Identifier for batch or queued jobs.
        start_time: Timestamp when the job started, when the site reports it.
    """

    #: Current lifecycle state for the run.
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
        """Return whether the run is waiting to start."""

        return self.state == "pending"

    @property
    def is_running(self) -> bool:
        """Return whether the run is currently executing."""

        return self.state == "running"

    @property
    def is_complete(self) -> bool:
        """Return whether the run has reached a terminal state."""

        return self.state in TERMINAL_STATES

    @property
    def is_successful(self) -> bool:
        """Return whether the terminal state and return code indicate success."""

        return self.state in SUCCESS_STATES and self.return_code in {0, -1}


@dataclass
class RunResult:
    """Final result returned by a completed run handle.

    Args:
        job: Job object associated with the run.
        status: Final job status.
        site: Site that submitted or attached to the run.
        artifacts: Optional site-specific artifact bundle.
        trace_manifest: Optional resolved trace manifest.
        logs_path: Optional path to run logs.
        run_metadata: Optional persisted run metadata.
    """

    job: Any
    status: JobStatus
    site: Optional["BaseSite"] = None
    artifacts: Any = None
    trace_manifest: Any = None
    logs_path: Optional[Path] = None
    run_metadata: Any = None

    @property
    def successful(self) -> bool:
        """Return whether the run completed successfully."""

        return self.status.is_successful

    def raise_for_status(self) -> None:
        """Raise ``RunFailedError`` if the run did not succeed."""

        if not self.successful:
            raise RunFailedError(self)

    def traces(self, upscale: int = 1):
        """Open receiver trace outputs for this run.

        Args:
            upscale: Optional time/frequency upscaling factor for trace reads.

        Returns:
            ``TraceDataset`` for local results or the site's fetched trace
            dataset for remote results.
        """

        self.raise_for_status()
        if self.site is not None:
            return self.site.fetch_traces(self.job, upscale=upscale)
        from frequensolve.seismic.traces import TraceDataset

        return TraceDataset.from_job(self.job, upscale=upscale)

    def wavefields(self, upscale: int = 1):
        """Open wavefield outputs for this run.

        Args:
            upscale: Optional upscaling factor used by trace readers.

        Returns:
            Wavefield trace dataset from the site or job artifact handle.
        """

        self.raise_for_status()
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
        fetch_missing: bool = True,
    ) -> list[Path]:
        """Return output files reported by the completed run.

        Args:
            kind: Optional artifact kind filter.
            suffix: Optional filename suffix or suffixes to include.
            base: Optional base directory used to resolve relative artifact
                paths.
            existing: If ``True``, return only files that currently exist.
            fetch_missing: If ``True`` and this run is remote-backed, ask the
                site to fetch filesystem outputs when no matching files are
                already available locally.

        Returns:
            List of matching output paths.
        """

        metadata = self.run_metadata or getattr(self.job, "run_metadata", None)
        if metadata is None:
            return []
        files = metadata.output_files(
            kind=kind,
            suffix=suffix,
            base=base,
            existing=existing,
        )
        if files or self.site is None or not fetch_missing:
            return files

        fetch_output_files = getattr(self.site, "fetch_output_files", None)
        if not callable(fetch_output_files):
            return files

        fetch_output_files(self.job)
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
        """Return or fetch run logs.

        Args:
            **kwargs: Site-specific log selection options such as ``task`` or
                ``frequency``.

        Returns:
            Path to logs, fetched log mapping, or ``None`` when unavailable.
        """

        if self.site is not None:
            return self.site.fetch_logs(self.job, **kwargs)
        if hasattr(self.job, "_stdout_path"):
            return self.job._stdout_path
        return None


class RunFailedError(RuntimeError):
    """Raised when a run reaches an unsuccessful terminal status."""

    def __init__(self, result: RunResult):
        """Create an error that keeps the failed run result attached."""

        self.result = result
        super().__init__(self._message(result))

    @staticmethod
    def _message(result: RunResult) -> str:
        job_name = getattr(result.job, "name", "<unknown>")
        status = result.status
        parts = [
            f"FrequenSolve run failed: job={job_name}",
            f"state={status.state}",
        ]
        if status.job_id:
            parts.append(f"job_id={status.job_id}")
        if status.message:
            parts.append(status.message)
        parts.append("The failed RunResult is attached to this exception.")
        return "; ".join(parts)


@dataclass
class RunHandle:
    """Awaitable handle for a submitted or skipped run.

    Args:
        site: Site that owns the run.
        job: Submitted job object.
        id: Site-specific run id.
        mode: Submission mode such as ``"batch"``, ``"local"``, or
            ``"skipped"``.
        poll_interval: Default polling interval in seconds.
        check: Whether ``wait()`` and ``wait_async()`` raise when this run
            reaches an unsuccessful terminal status and the caller does not
            pass an explicit ``check`` value.
        backend: Site-specific run metadata.
    """

    site: "BaseSite"
    job: Any
    id: Optional[str] = None
    mode: str = "unknown"
    poll_interval: float = 5.0
    check: bool = True
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
        """Create a completed handle for a run skipped as already current.

        Args:
            site: Site that would have submitted the job.
            job: Job object associated with the skipped run.
            message: Human-readable reason the run was skipped.

        Returns:
            ``RunHandle`` whose result is already available.
        """

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
        """Poll and return the latest run status."""

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
        *,
        check: Optional[bool] = None,
    ) -> RunResult:
        """Block until the run reaches a terminal state.

        Args:
            timeout: Optional maximum wait time in seconds.
            poll_interval: Optional polling interval in seconds.
            check: Whether to raise ``RunFailedError`` for failed, cancelled,
                or timed-out runs. Defaults to the handle's submit-time
                ``check`` value.

        Returns:
            Final ``RunResult``.
        """

        effective_check = self.check if check is None else check
        if self._result is not None:
            if effective_check:
                self._result.raise_for_status()
            return self._result
        if not self._generic_wait and self._wait_fn is not None:
            self._result = self._wait_fn(self, timeout, poll_interval)
            if effective_check:
                self._result.raise_for_status()
            return self._result
        from frequensolve.orchestrator.utils.progress import wait

        self._result = wait(
            self,
            timeout=timeout,
            poll_interval=poll_interval,
            check=effective_check,
        )
        return self._result

    async def wait_async(
        self,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        *,
        check: Optional[bool] = None,
    ) -> RunResult:
        """Asynchronously wait for the run to finish.

        Args:
            timeout: Optional maximum wait time in seconds.
            poll_interval: Optional polling interval in seconds.
            check: Whether to raise ``RunFailedError`` for failed, cancelled,
                or timed-out runs. Defaults to the handle's submit-time
                ``check`` value.

        Returns:
            Final ``RunResult``.
        """

        effective_check = self.check if check is None else check
        if self._result is not None:
            if effective_check:
                self._result.raise_for_status()
            return self._result
        if not self._generic_wait and self._wait_async_fn is not None:
            self._result = await self._wait_async_fn(self, timeout, poll_interval)
            if effective_check:
                self._result.raise_for_status()
            return self._result
        return await asyncio.to_thread(
            self.wait,
            timeout,
            poll_interval,
            check=effective_check,
        )

    def watch(
        self,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> Iterable[JobStatus]:
        """Yield status changes until the run finishes or times out.

        Args:
            timeout: Optional maximum watch time in seconds.
            poll_interval: Optional polling interval in seconds.

        Yields:
            ``JobStatus`` objects whenever the public state changes.
        """

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
        """Cancel the run using the site-specific cancel operation when possible."""

        if self._cancel_fn is not None:
            self._cancel_fn(self)
            return
        if self.id is None:
            return
        self.site.cancel_job(self.id)

    def fetch(self):
        """Fetch completed run outputs using the site implementation.

        Returns:
            Site-specific fetch result, or ``None`` when no fetch operation is
            available.
        """

        if self._fetch_fn is not None:
            return self._fetch_fn(self)
        if hasattr(self.site, "fetch_outputs"):
            return self.site.fetch_outputs(self.job)
        return None

    def traces(self, upscale: int = 1):
        """Fetch outputs if needed and open receiver traces.

        Args:
            upscale: Optional time/frequency upscaling factor for trace reads.

        Returns:
            Trace dataset returned by the site.
        """

        self.fetch()
        return self.site.fetch_traces(self.job, upscale=upscale)

    def wavefields(self, upscale: int = 1):
        """Fetch outputs if needed and open wavefield traces.

        Args:
            upscale: Optional upscaling factor for wavefield trace reads.

        Returns:
            Wavefield dataset returned by the site.
        """

        self.fetch()
        return self.site.fetch_wavefields(self.job, upscale=upscale)

    def logs(self, **kwargs):
        """Return or fetch logs for this run.

        Args:
            **kwargs: Site-specific log selection options.

        Returns:
            Path to logs, fetched log mapping, or ``None`` when unavailable.
        """

        if self.site is not None:
            return self.site.fetch_logs(self.job, **kwargs)
        if hasattr(self.job, "_stdout_path"):
            return self.job._stdout_path
        return None


@dataclass(kw_only=True)
class BaseSite:
    """Base class for execution-site implementations.

    Args:
        verbose: Whether site methods should print status messages in addition
            to logging them.
    """

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

        should_print = force or getattr(self, "verbose", False)
        log_level = logging.DEBUG if should_print and level <= logging.INFO else level
        logging.getLogger(self.__class__.__module__).log(log_level, message)
        if should_print:
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

    def prepare_job(
        self,
        job,
        *,
        sync_project: bool = False,
        validate: bool = True,
    ):
        """Persist local job inputs before a run is submitted.

        Site implementations call this before checking run fingerprints or
        transferring files.  It keeps the user-facing lifecycle simple:
        creating a job and calling ``site.submit(job)`` is enough.

        Args:
            job: Job object with an optional ``save`` method.
            sync_project: Whether to synchronize the owning project after the
                job is saved.
            validate: Whether to run job validation before submission.

        Returns:
            The same job object, for fluent site implementations.
        """

        if hasattr(job, "save"):
            job.save()
        if validate and hasattr(job, "validate"):
            job.validate(raise_errors=True)
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
        """Return local wavefield outputs for one job or many jobs.

        Args:
            job: Single job or a sequence of jobs.
            upscale: Optional upscaling factor for wavefield trace reads.
            path: Optional project path alias used to resolve relative outputs.
            project_path: Optional project path used to resolve relative outputs.
            **_: Ignored compatibility keyword arguments.

        Returns:
            Wavefield dataset for a single job, or a mapping keyed by job name.
        """

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

        Args:
            job: Single job or sequence of jobs.
            local_dir: Optional log directory or destination directory.
            task: Optional one-based task number.
            frequency: Optional frequency used to select a task log.
            show: Whether to print the selected log contents.
            **_: Ignored compatibility keyword arguments.

        Returns:
            Log path for a single job, or a mapping keyed by job name.
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

    def submit(self, job, *, check: bool = False, **kwargs) -> RunHandle:
        """Submit a job and return an awaitable run handle.

        Args:
            job: Job object to submit.
            check: Whether the returned handle raises by default when waited
                and the run reaches an unsuccessful terminal status.
            **kwargs: Site-specific submission options.

        Returns:
            ``RunHandle`` for the submitted run.
        """
        raise NotImplementedError

    def run(
        self,
        job,
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        check: bool = False,
        **submit_kwargs,
    ) -> RunResult:
        """Submit a job and block until completion.

        Args:
            job: Job object to submit.
            timeout: Optional maximum wait time in seconds.
            poll_interval: Optional polling interval in seconds.
            check: Whether to raise ``RunFailedError`` for failed, cancelled,
                or timed-out runs.
            **submit_kwargs: Site-specific submission options.

        Returns:
            Final ``RunResult``.
        """
        return self.submit(job, **submit_kwargs).wait(
            timeout,
            poll_interval,
            check=check,
        )

    def wait(
        self,
        runs: Iterable[RunHandle],
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        fetch: bool = False,
        check: bool = True,
    ) -> list[RunResult]:
        """Wait for multiple run handles while rendering one combined status.

        Args:
            runs: Run handles to wait on.
            timeout: Optional maximum wait time in seconds.
            poll_interval: Optional polling interval in seconds.
            fetch: Whether to fetch outputs after each successful run.

        Returns:
            Final results in input order.
        """

        return self.wait_all(
            runs,
            timeout=timeout,
            poll_interval=poll_interval,
            fetch=fetch,
            check=check,
        )

    def wait_all(
        self,
        runs: Iterable[RunHandle],
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        fetch: bool = False,
        check: bool = True,
    ) -> list[RunResult]:
        """Wait for many submitted runs and return results in input order.

        Args:
            runs: Run handles to wait on.
            timeout: Optional maximum wait time in seconds.
            poll_interval: Optional polling interval in seconds.
            fetch: Whether to fetch outputs after each successful run.

        Returns:
            Final results in input order.
        """

        from frequensolve.orchestrator.utils.progress import wait_all

        return wait_all(
            runs,
            timeout=timeout,
            poll_interval=poll_interval,
            fetch=fetch,
            check=check,
        )

    def handle(
        self, job, job_id: Optional[str] = None, mode: str = "attached"
    ) -> RunHandle:
        """Create a run handle for an existing submitted job.

        Args:
            job: Job object associated with the existing run.
            job_id: Site-specific run id. Defaults to ``job._job_id``.
            mode: Handle mode label.

        Returns:
            ``RunHandle`` attached to the existing run.

        Raises:
            ValueError: If no job id is available.
            NotImplementedError: If the site cannot poll existing jobs.
        """
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
