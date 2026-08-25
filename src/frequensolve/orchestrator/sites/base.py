"""Shared run-handle and site abstractions for local, cloud, and HPC execution."""

import asyncio
import builtins
import html
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Union,
    cast,
)

from frequensolve.simulation.jobs.run_state import SkipPolicy

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


class _TraceFetchingSite(Protocol):
    """Optional receiver-trace capability implemented by concrete sites."""

    def fetch_traces(self, job: Any, upscale: int = 1) -> Any: ...


__all__ = [
    "BaseSite",
    "JobStatus",
    "RunFailedError",
    "RunHandle",
    "RunResult",
    "SubmitPlan",
    "_check_if_notebook",
]


def _merge_task_status_with_plan(
    payload: Mapping[str, Any],
    task_plan: Optional[Mapping[str, Any]],
    *,
    job: Any = None,
) -> Dict[str, Any]:
    """Fold already-current task-plan rows into submitted-task status counts."""

    out = dict(payload)
    if not isinstance(task_plan, Mapping) or out.get("includes_current_tasks"):
        return out

    current_tasks = _task_plan_current_tasks(task_plan)
    if not current_tasks:
        return out

    submitted_total = _int_count(out.get("total"))
    job_total = _int_count(getattr(job, "n_tasks", None))
    if job_total and submitted_total >= job_total:
        return out

    succeeded = _int_count(out.get("successful", out.get("succeeded")))
    failed = _int_count(out.get("failed"))
    running = _int_count(out.get("running"))
    pending = _int_count(out.get("pending"))
    current_count = len(current_tasks)
    total = job_total or submitted_total + current_count
    missing = max(0, total - (submitted_total + current_count))

    out["successful"] = succeeded + current_count
    out["succeeded"] = out["successful"]
    out["failed"] = failed
    out["running"] = running
    out["pending"] = pending + missing
    out["total"] = total
    out["current"] = current_count
    out["submitted_total"] = submitted_total
    out["includes_current_tasks"] = True
    return out


def _task_plan_current_tasks(task_plan: Mapping[str, Any]) -> set[int]:
    current = set()
    for value in task_plan.get("current_tasks", []) or []:
        try:
            current.add(int(value))
        except (TypeError, ValueError):
            continue
    for record in task_plan.get("reused_tasks", []) or []:
        if not isinstance(record, Mapping):
            continue
        task = record.get("task")
        if task is None and "task_id" in record:
            try:
                task = int(record["task_id"]) + 1
            except (TypeError, ValueError):
                task = None
        if task is None:
            continue
        try:
            current.add(int(task))
        except (TypeError, ValueError):
            continue
    return current


def _int_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _task_numbers(values: Iterable[Any], total: int) -> tuple[int, ...]:
    tasks = []
    for value in values:
        try:
            task = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= task <= total:
            tasks.append(task)
    return tuple(sorted(set(tasks)))


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
    get_ipython = getattr(builtins, "get_ipython", None)
    if not callable(get_ipython):
        return False
    return get_ipython().__class__.__name__ == "ZMQInteractiveShell"


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


@dataclass(frozen=True)
class SubmitPlan:
    """Read-only preview of the frequency tasks a submit call would run.

    Args:
        job: Job being inspected.
        site: Site used to build the preview.
        total_tasks: Total number of frequency tasks in the job.
        pending_tasks: One-based tasks that would be submitted.
        current_tasks: One-based tasks already current locally.
        reused_tasks: One-based tasks whose prior outputs can be reused.
        accepted_tasks: One-based tasks accepted by compatibility policy.
        accepted_failed_tasks: One-based failed tasks accepted by policy.
        failed_existing_tasks: One-based tasks with traces that exist but are
            marked failed and would rerun.
        skip_policy: Policy used to classify existing outputs.
        force: Whether reusable/current outputs are being ignored.
    """

    job: Any
    site: Optional["BaseSite"]
    total_tasks: int
    pending_tasks: tuple[int, ...]
    current_tasks: tuple[int, ...]
    reused_tasks: tuple[int, ...] = ()
    accepted_tasks: tuple[int, ...] = ()
    accepted_failed_tasks: tuple[int, ...] = ()
    failed_existing_tasks: tuple[int, ...] = ()
    skip_policy: Any = None
    force: bool = False

    @property
    def n_tasks_to_run(self) -> int:
        """Return the number of tasks that would run."""

        return len(self.pending_tasks)

    @property
    def n_current_tasks(self) -> int:
        """Return the number of tasks already current."""

        return len(self.current_tasks)

    @property
    def n_tasks_to_skip(self) -> int:
        """Return the number of tasks that would be skipped as current."""

        return len(self.skipped_tasks)

    @property
    def n_reused_tasks(self) -> int:
        """Return the number of tasks reusable from earlier outputs."""

        return len(self.reused_tasks)

    @property
    def n_accepted_tasks(self) -> int:
        """Return the number of tasks skipped by compatibility policy."""

        return len(self.accepted_tasks)

    @property
    def n_accepted_failed_tasks(self) -> int:
        """Return the number of failed tasks accepted by policy."""

        return len(self.accepted_failed_tasks)

    @property
    def n_failed_existing_tasks(self) -> int:
        """Return the number of existing trace outputs marked failed."""

        return len(self.failed_existing_tasks)

    @property
    def skipped_tasks(self) -> tuple[int, ...]:
        """Return one-based tasks that would be skipped."""

        return tuple(
            sorted(
                {
                    *self.current_tasks,
                    *self.reused_tasks,
                    *self.accepted_tasks,
                    *self.accepted_failed_tasks,
                }
            )
        )

    @property
    def pending_indices(self) -> tuple[int, ...]:
        """Return zero-based solver task indices that would be submitted."""

        return tuple(task - 1 for task in self.pending_tasks)

    @property
    def pending_frequencies(self) -> tuple[Any, ...]:
        """Return frequencies for the tasks that would be submitted."""

        f_list = list(getattr(self.job, "f_list", []) or [])
        frequencies = []
        for task in self.pending_tasks:
            try:
                frequencies.append(f_list[task - 1])
            except IndexError:
                continue
        return tuple(frequencies)

    @property
    def all_current(self) -> bool:
        """Return whether no frequency tasks need to run."""

        return self.n_tasks_to_run == 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the plan."""

        return {
            "job": getattr(self.job, "name", None),
            "site": self.site.__class__.__name__ if self.site is not None else None,
            "total_tasks": self.total_tasks,
            "n_tasks_to_run": self.n_tasks_to_run,
            "n_tasks_to_skip": self.n_tasks_to_skip,
            "pending_tasks": list(self.pending_tasks),
            "pending_indices": list(self.pending_indices),
            "pending_frequencies": list(self.pending_frequencies),
            "skipped_tasks": list(self.skipped_tasks),
            "current_tasks": list(self.current_tasks),
            "reused_tasks": list(self.reused_tasks),
            "accepted_tasks": list(self.accepted_tasks),
            "accepted_failed_tasks": list(self.accepted_failed_tasks),
            "failed_existing_tasks": list(self.failed_existing_tasks),
            "skip_policy": getattr(self.skip_policy, "mode", self.skip_policy),
            "force": self.force,
        }

    def _task_ranges(self, tasks: Iterable[int]) -> str:
        values = sorted({int(task) for task in tasks})
        if not values:
            return "-"
        ranges = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append((start, previous))
            start = previous = value
        ranges.append((start, previous))
        return ", ".join(
            str(start) if start == end else f"{start}-{end}" for start, end in ranges
        )

    def summary(self) -> str:
        """Return a concise human-readable plan summary."""

        name = getattr(self.job, "name", "job")
        message = (
            f"{name}: {self.n_tasks_to_run} / {self.total_tasks} "
            "frequency tasks would run"
        )
        if self.force:
            message += " (skip=False; current outputs ignored)"
        else:
            categories = []
            if self.n_current_tasks:
                categories.append(f"{self.n_current_tasks} current")
            if self.n_reused_tasks:
                categories.append(f"{self.n_reused_tasks} reusable")
            if self.n_accepted_tasks:
                categories.append(f"{self.n_accepted_tasks} compatible")
            if self.n_accepted_failed_tasks:
                categories.append(f"{self.n_accepted_failed_tasks} accepted failed")
            category_text = ", ".join(categories) if categories else "0 current"
            message += f"; {self.n_tasks_to_skip} would skip " f"({category_text})"
        if self.pending_tasks:
            message += f"\nPending tasks: {self._task_ranges(self.pending_tasks)}"
        if self.skipped_tasks:
            message += f"\nSkipped tasks: {self._task_ranges(self.skipped_tasks)}"
        if self.reused_tasks:
            message += f"\nReusable tasks: {self._task_ranges(self.reused_tasks)}"
        if self.accepted_tasks:
            message += (
                "\nCompatible tasks accepted by policy: "
                f"{self._task_ranges(self.accepted_tasks)}"
            )
        if self.accepted_failed_tasks:
            message += (
                "\nFailed tasks accepted by policy: "
                f"{self._task_ranges(self.accepted_failed_tasks)}"
            )
        if self.failed_existing_tasks:
            message += (
                "\nExisting traces marked failed; would rerun: "
                f"{self._task_ranges(self.failed_existing_tasks)}"
            )
        return message

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        name = html.escape(str(getattr(self.job, "name", "job")))
        site = (
            html.escape(self.site.__class__.__name__) if self.site is not None else "-"
        )
        pending = html.escape(self._task_ranges(self.pending_tasks))
        current = html.escape(self._task_ranges(self.current_tasks))
        reused = html.escape(self._task_ranges(self.reused_tasks))
        accepted = html.escape(self._task_ranges(self.accepted_tasks))
        accepted_failed = html.escape(self._task_ranges(self.accepted_failed_tasks))
        failed_existing = html.escape(self._task_ranges(self.failed_existing_tasks))
        skip_policy = html.escape(str(getattr(self.skip_policy, "mode", "-") or "-"))
        force = "yes" if self.force else "no"
        rows = [
            ("Job", name),
            ("Site", site),
            ("Skip policy", skip_policy),
            ("Tasks that would run", f"{self.n_tasks_to_run} / {self.total_tasks}"),
            ("Tasks that would skip", f"{self.n_tasks_to_skip} / {self.total_tasks}"),
            ("Pending tasks", pending),
            ("Current tasks", current),
            ("Reusable tasks", reused),
            ("Compatible tasks", accepted),
            ("Accepted failed tasks", accepted_failed),
            ("Existing failed traces", failed_existing),
            ("Force", force),
        ]
        body = "".join(
            "<tr>"
            f"<th style='text-align:left;padding:2px 10px 2px 0'>{label}</th>"
            f"<td style='text-align:left;padding:2px 0'>{value}</td>"
            "</tr>"
            for label, value in rows
        )
        return f"<table>{body}</table>"


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

    def traces(self, upscale: int = 1) -> Any:
        """Open receiver trace outputs for this run.

        Args:
            upscale: Optional time/frequency upscaling factor for trace reads.

        Returns:
            ``TraceDataset`` for local results or the site's fetched trace
            dataset for remote results.
        """

        self.raise_for_status()
        if self.site is not None:
            site = cast(_TraceFetchingSite, self.site)
            return site.fetch_traces(self.job, upscale=upscale)
        from frequensolve.seismic.traces import TraceDataset

        return TraceDataset.from_job(self.job, upscale=upscale)

    def wavefields(self, upscale: int = 1) -> Any:
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

        fetch_output_files(self.job, kind=kind, suffix=suffix)
        metadata = self.run_metadata or getattr(self.job, "run_metadata", None)
        if metadata is None:
            return []
        return metadata.output_files(
            kind=kind,
            suffix=suffix,
            base=base,
            existing=existing,
        )

    def logs(self, **kwargs: Any) -> Any:
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
        _fetch_on_complete: Whether waits fetch outputs after terminal success.
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
    _fetch_on_complete: bool = False
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

    def __await__(self) -> Generator[Any, None, RunResult]:
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

    def _run_record_finalizer(
        self,
    ) -> Optional[Callable[["RunHandle", JobStatus], Any]]:
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
            if self._fetch_on_complete and (
                self._result.successful or not effective_check
            ):
                self.fetch()
            if effective_check:
                self._result.raise_for_status()
            return self._result
        from frequensolve.orchestrator.utils.progress import wait

        self._result = wait(
            self,
            timeout=timeout,
            poll_interval=poll_interval,
            fetch=self._fetch_on_complete,
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
            if self._fetch_on_complete and (
                self._result.successful or not effective_check
            ):
                self.fetch()
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
                result = self._complete_from_status(status)
                if self._fetch_on_complete and (result.successful or not self.check):
                    self.fetch()
                status = result.status
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

    def fetch(self) -> Any:
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

    def traces(self, upscale: int = 1) -> Any:
        """Fetch outputs if needed and open receiver traces.

        Args:
            upscale: Optional time/frequency upscaling factor for trace reads.

        Returns:
            Trace dataset returned by the site.
        """

        self.fetch()
        site = cast(_TraceFetchingSite, self.site)
        return site.fetch_traces(self.job, upscale=upscale)

    def wavefields(self, upscale: int = 1) -> Any:
        """Fetch outputs if needed and open wavefield traces.

        Args:
            upscale: Optional upscaling factor for wavefield trace reads.

        Returns:
            Wavefield dataset returned by the site.
        """

        self.fetch()
        return self.site.fetch_wavefields(self.job, upscale=upscale)

    def logs(self, **kwargs: Any) -> Any:
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
        job: Any,
        *,
        sync_project: bool = False,
        validate: bool = True,
    ) -> Any:
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

        if validate and hasattr(job, "validate"):
            report = job.validate(
                raise_errors=True,
                **self._job_validation_options(job),
            )
            self._emit_validation_warnings(job, report)
        if hasattr(job, "save"):
            job.save()
        project = getattr(getattr(job, "simulation", None), "_project", None)
        if sync_project and project is not None and hasattr(self, "sync"):
            self.sync(project)
        return job

    def _job_validation_options(self, job: Any) -> Dict[str, Any]:
        """Return execution-site-specific preflight options for one job."""

        return {}

    def _emit_validation_warnings(self, _job: Any, report: Any) -> None:
        """Log each warning produced by this validation pass."""

        for issue in getattr(report, "warnings", []) or []:
            self._emit(str(issue), level=logging.WARNING)

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

    def dry_run(
        self,
        job: Any,
        *,
        skip: Optional[Any] = None,
        skip_policy: Optional[Any] = None,
        residual: Optional[float] = None,
        ignore_solver_options: Optional[bool] = None,
        force: bool = False,
        rerun: bool = False,
        reuse: bool = True,
        validate: bool = True,
        refresh_metadata: bool = False,
    ) -> SubmitPlan:
        """Preview which frequency tasks would be submitted.

        This is a lightweight counterpart to :meth:`submit`: it persists and
        optionally validates local job inputs so fingerprints match a real
        submit call, but it does not sync, submit, write run state, call
        ``task_run_plan``, copy reusable outputs, or remove stale outputs.

        Args:
            job: Job object to inspect.
            skip: Skip policy name or object. Supported names include
                ``"strict"``, ``"compatible"``, ``"tolerant"``, ``"none"``,
                and ``"false"``. Boolean ``False`` also reruns every task.
            skip_policy: Alias for ``skip``.
            residual: Residual threshold for tolerant failed-task acceptance.
            ignore_solver_options: Ignore solver-only simulation settings when
                comparing task compatibility.
            force: Ignore reusable/current outputs and count all tasks.
            rerun: Alias for ``force``.
            reuse: Include prior matching frequency outputs that submit would
                reuse. Defaults to ``True`` to mirror normal submission.
            validate: Whether to run job validation before checking outputs.
            refresh_metadata: If ``True`` and the site supports it, fetch remote
                ``_fs_run`` metadata before counting current tasks.

        Returns:
            ``SubmitPlan`` with one-based pending/current task numbers and
            zero-based ``pending_indices`` for solver submission.
        """

        policy_value = skip if skip is not None else skip_policy
        normalized_policy = SkipPolicy.from_value(
            policy_value,
            residual=residual,
            ignore_solver_options=ignore_solver_options,
            reuse=reuse,
        )
        fresh_run = bool(force or rerun or normalized_policy.force)
        self.prepare_job(job, validate=validate)
        if refresh_metadata:
            fetch_metadata = getattr(self, "fetch_run_metadata", None)
            if not callable(fetch_metadata):
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not support refresh_metadata"
                )
            fetch_metadata(job)

        total = _int_count(getattr(job, "n_tasks", None))
        if total == 0:
            try:
                total = len(getattr(job, "f_list", []) or [])
            except TypeError:
                total = 0

        if hasattr(job, "plan_tasks"):
            task_plan = job.plan_tasks(
                skip_policy=policy_value,
                reuse=reuse,
                force=fresh_run,
                apply=False,
                residual=residual,
                ignore_solver_options=ignore_solver_options,
            )
            pending = tuple(int(index) + 1 for index in task_plan["pending_indices"])
            planned_current = _task_numbers(
                task_plan.get("strict_current_tasks", ()), total
            )
            planned_reused = _task_numbers(
                (
                    record.get("task")
                    for record in task_plan.get("reused_tasks", [])
                    if isinstance(record, Mapping)
                ),
                total,
            )
            accepted = _task_numbers(task_plan.get("accepted_tasks", ()), total)
            accepted_failed = _task_numbers(
                task_plan.get("accepted_failed_tasks", ()),
                total,
            )
            skipped = {
                *planned_current,
                *planned_reused,
                *accepted,
                *accepted_failed,
            }
            failed_existing = (
                () if fresh_run else self._failed_existing_tasks(job, total, skipped)
            )
            return SubmitPlan(
                job=job,
                site=self,
                total_tasks=total,
                pending_tasks=pending,
                current_tasks=planned_current,
                reused_tasks=planned_reused,
                accepted_tasks=accepted,
                accepted_failed_tasks=accepted_failed,
                failed_existing_tasks=failed_existing,
                skip_policy=task_plan.get("skip_policy"),
                force=fresh_run,
            )

        if fresh_run:
            current: tuple[int, ...] = ()
        elif hasattr(job, "current_tasks"):
            current_values = []
            for task in job.current_tasks():
                try:
                    task_number = int(task)
                except (TypeError, ValueError):
                    continue
                if 1 <= task_number <= total:
                    current_values.append(task_number)
            current = tuple(sorted(set(current_values)))
        elif hasattr(job, "is_run_current") and job.is_run_current():
            current = tuple(range(1, total + 1))
        else:
            current = ()

        reused: tuple[int, ...] = ()
        if not fresh_run and reuse and hasattr(job, "reusable_task_outputs"):
            reused_values = []
            current_set = set(current)
            for record in job.reusable_task_outputs():
                if not isinstance(record, Mapping):
                    continue
                raw_task = record.get("task")
                if raw_task is None:
                    continue
                try:
                    task_number = int(raw_task)
                except (TypeError, ValueError):
                    continue
                if 1 <= task_number <= total and task_number not in current_set:
                    reused_values.append(task_number)
            reused = tuple(sorted(set(reused_values)))

        skipped = {*current, *reused}
        pending = tuple(
            task for task in range(1, total + 1) if fresh_run or task not in skipped
        )
        failed_existing = (
            () if fresh_run else self._failed_existing_tasks(job, total, skipped)
        )
        return SubmitPlan(
            job=job,
            site=self,
            total_tasks=total,
            pending_tasks=pending,
            current_tasks=current,
            reused_tasks=reused,
            failed_existing_tasks=failed_existing,
            skip_policy=policy_value,
            force=fresh_run,
        )

    @staticmethod
    def _failed_existing_tasks(
        job: Any,
        total: int,
        skipped: Iterable[int] = (),
    ) -> tuple[int, ...]:
        if not hasattr(job, "frequency_status"):
            return ()
        skipped_set = {int(task) for task in skipped}
        failed_values = []
        for row in job.frequency_status():
            if not isinstance(row, Mapping):
                continue
            raw_task = row.get("task")
            if raw_task is None:
                continue
            try:
                task_number = int(raw_task)
            except (TypeError, ValueError):
                continue
            if (
                1 <= task_number <= total
                and task_number not in skipped_set
                and row.get("status") == "failed"
                and bool(row.get("trace_exists"))
            ):
                failed_values.append(task_number)
        return tuple(sorted(set(failed_values)))

    def submit(self, job: Any, *, check: bool = False, **kwargs: Any) -> RunHandle:
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
        job: Any,
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        check: bool = False,
        **submit_kwargs: Any,
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
        self, job: Any, job_id: Optional[str] = None, mode: str = "attached"
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

        def cancel(run: RunHandle) -> None:
            self.cancel_job(str(run.id))

        return RunHandle(
            site=self,
            job=job,
            id=str(job_id),
            mode=mode,
            poll_interval=getattr(getattr(self, "config", None), "poll_interval", 5),
            _status_fn=poll,
            _cancel_fn=cancel,
        )

    def cancel_job(self, job_id: str) -> bool | None:
        """Cancel a running job.

        Args:
            job_id: The ID of the job to cancel
        """
        raise NotImplementedError
