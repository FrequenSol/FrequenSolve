"""Site-neutral progress monitoring for submitted runs."""

from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from frequensolve.orchestrator.sites.base import (
    JobStatus,
    RunHandle,
    RunResult,
    _check_if_notebook,
)

__all__ = ["RunMonitor", "status_table_html", "status_text", "wait", "wait_all"]


def wait(
    run: RunHandle,
    *,
    timeout: Optional[float] = None,
    poll_interval: Optional[float] = None,
    fetch: bool = False,
    check: bool = True,
) -> RunResult:
    """Wait for one run using the generic progress monitor.

    Args:
        check: Raise ``RunFailedError`` for unsuccessful terminal statuses.
    """

    return wait_all(
        [run],
        timeout=timeout,
        poll_interval=poll_interval,
        fetch=fetch,
        check=check,
    )[0]


def wait_all(
    runs: Iterable[RunHandle],
    *,
    timeout: Optional[float] = None,
    poll_interval: Optional[float] = None,
    fetch: bool = False,
    check: bool = True,
) -> list[RunResult]:
    """Wait for many runs, possibly from different sites, in input order.

    Args:
        check: Raise ``RunFailedError`` for unsuccessful terminal statuses.
    """

    return RunMonitor(runs).wait(
        timeout=timeout,
        poll_interval=poll_interval,
        fetch=fetch,
        check=check,
    )


@dataclass
class RunMonitor:
    """Poll and render progress for one or more submitted runs."""

    runs: Iterable[RunHandle]
    is_notebook: bool = field(default_factory=_check_if_notebook)
    _signature: Optional[tuple] = field(default=None, init=False)
    _widget: Any = field(default=None, init=False)

    def wait(
        self,
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        fetch: bool = False,
        check: bool = True,
    ) -> list[RunResult]:
        handles = list(self.runs)
        if not handles:
            return []

        interval = self._wait_interval(handles, poll_interval)
        start = time.monotonic()
        completed: set[int] = set()
        statuses: Dict[int, JobStatus] = {}

        while len(completed) < len(handles):
            for index, run in enumerate(handles):
                if index in completed:
                    statuses[index] = run._result.status
                    continue
                if run._result is not None:
                    statuses[index] = run._result.status
                    completed.add(index)
                    continue

                status = self._poll(run)
                run._last_status = status
                statuses[index] = status

                if status.is_complete:
                    self._complete(run, status, fetch=fetch, check=check)
                    completed.add(index)
                    statuses[index] = run._result.status

            self._emit(handles, statuses)
            if len(completed) == len(handles):
                break

            if timeout is not None and time.monotonic() - start > timeout:
                for index, run in enumerate(handles):
                    if index in completed:
                        continue
                    status = JobStatus(
                        state="timeout",
                        job_id=run.id,
                        message=f"Timed out waiting for runs after {timeout} seconds",
                    )
                    run._last_status = status
                    statuses[index] = status
                    self._timeout(run, status, fetch=False)
                    completed.add(index)
                    statuses[index] = run._result.status
                self._emit(handles, statuses)
                break

            time.sleep(interval)

        results = [run._result for run in handles]
        if check:
            for result in results:
                result.raise_for_status()
        return results

    @staticmethod
    def _wait_interval(
        runs: Iterable[RunHandle],
        poll_interval: Optional[float],
    ) -> float:
        if poll_interval is not None:
            return poll_interval
        intervals = [run.poll_interval for run in runs if run.poll_interval >= 0]
        return min(intervals) if intervals else 1.0

    @staticmethod
    def _poll(run: RunHandle) -> JobStatus:
        if run._result is not None:
            return run._result.status
        return run.status()

    @staticmethod
    def _complete(
        run: RunHandle,
        status: JobStatus,
        *,
        fetch: bool,
        check: bool,
    ) -> None:
        run._complete_from_status(status)
        if fetch and (run._result.successful or not check):
            run.fetch()

    @staticmethod
    def _timeout(run: RunHandle, status: JobStatus, *, fetch: bool) -> None:
        if run._timeout_fn is not None:
            run._result = run._timeout_fn(run, status)
        else:
            run._result = run._make_result(status)
        if fetch:
            run.fetch()

    def _emit(self, runs: list[RunHandle], statuses: Dict[int, JobStatus]) -> None:
        signature = _status_signature(runs, statuses)
        if signature == self._signature:
            return
        self._signature = signature

        if self.is_notebook:
            try:
                from IPython.display import HTML, display
            except Exception:
                self._emit_terminal(runs, statuses)
                return
            payload = status_table_html(runs, statuses)
            if self._widget is None:
                self._widget = display(HTML(payload), display_id=True)
            else:
                self._widget.update(HTML(payload))
            return

        self._emit_terminal(runs, statuses)

    @staticmethod
    def _emit_terminal(
        runs: list[RunHandle],
        statuses: Dict[int, JobStatus],
    ) -> None:
        if len(runs) == 1:
            run = runs[0]
            status = statuses.get(0, run._last_status)
            emit_status = getattr(run.site, "_emit_status", None)
            if callable(emit_status):
                emit_status(status, force=True)
                return
        print(status_text(runs, statuses))


def status_text(runs: list[RunHandle], statuses: Dict[int, JobStatus]) -> str:
    """Return terminal-friendly progress text for many runs."""

    lines = []
    for index, run in enumerate(runs):
        status = statuses.get(index, run._last_status)
        name = getattr(run.job, "name", f"run-{index + 1}")
        site = _site_label(run)
        summary = _status_summary(status)
        job_id = f" [{status.job_id or run.id}]" if (status.job_id or run.id) else ""
        lines.append(f"{site} {name}{job_id}: {status.state}{summary}")
    return "\n".join(lines)


def status_table_html(runs: list[RunHandle], statuses: Dict[int, JobStatus]) -> str:
    """Render the generic run progress table as HTML."""

    rows = []
    complete = 0
    for index, run in enumerate(runs):
        status = statuses.get(index, run._last_status)
        complete += int(status.is_complete)
        site = html.escape(_site_label(run))
        name = html.escape(str(getattr(run.job, "name", f"run-{index + 1}")))
        job_id = html.escape(str(status.job_id or run.id or ""))
        state = html.escape(str(status.state))
        counts = _status_counts(status)
        color = _status_html_color(status.state)
        rows.append(
            "<tr style='border-bottom:1px solid #d8dee4'>"
            f"<td style='padding:7px 10px; color:#57606a'>{site}</td>"
            f"<td style='padding:7px 10px; font-weight:650; color:#24292f'>"
            f"{name}</td>"
            f"<td style='padding:7px 10px; color:#57606a; font-variant-numeric:"
            f"tabular-nums'>{job_id}</td>"
            f"<td style='padding:7px 10px'><span style='background:{color}; "
            "color:white; border-radius:999px; padding:2px 9px; "
            "font-size:12px; font-weight:700; line-height:1.6'>"
            f"{state}</span></td>"
            f"{_count_cell(counts['succeeded'], '#1a7f37', border_left=True)}"
            f"{_count_cell(counts['failed'], '#cf222e')}"
            f"{_count_cell(counts['running'], '#8250df')}"
            f"{_count_cell(counts['pending'], '#9a6700')}"
            f"{_count_cell(counts['total'], '#24292f')}"
            "</tr>"
        )
    return (
        "<div style='background:#ffffff; border:1px solid #d0d7de; "
        "border-radius:8px; padding:12px; color:#24292f; "
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        "max-width:1040px; box-shadow:0 1px 2px rgba(31,35,40,0.08)'>"
        "<div style='display:flex; justify-content:space-between; align-items:"
        "center; gap:16px; margin-bottom:10px'>"
        "<strong style='color:#24292f; font-size:14px'>FrequenSolve runs</strong>"
        f"<span style='color:#57606a; font-size:12px; font-weight:700'>"
        f"{complete}/{len(runs)} complete</span>"
        "</div>"
        "<table style='border-collapse:collapse; width:100%; font-size:13px; "
        "color:#24292f'>"
        "<thead><tr style='text-align:left; color:#57606a; background:#f6f8fa; "
        "border-bottom:1px solid #d8dee4'>"
        f"{_header_cell('Site')}"
        f"{_header_cell('Job')}"
        f"{_header_cell('Job ID')}"
        f"{_header_cell('State')}"
        f"{_header_cell('Succeeded', align='right', border_left=True)}"
        f"{_header_cell('Failed', align='right')}"
        f"{_header_cell('Running', align='right')}"
        f"{_header_cell('Pending', align='right')}"
        f"{_header_cell('Total', align='right')}"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _status_signature(
    runs: list[RunHandle],
    statuses: Dict[int, JobStatus],
) -> tuple:
    values = []
    for index, run in enumerate(runs):
        status = statuses.get(index, run._last_status)
        payload = (
            status.raw.get("task_status") if isinstance(status.raw, dict) else None
        )
        values.append(
            (
                _site_label(run),
                run.id,
                status.state,
                status.message,
                payload.get("successful") if payload else None,
                payload.get("succeeded") if payload else None,
                payload.get("failed") if payload else None,
                payload.get("running") if payload else None,
                payload.get("pending") if payload else None,
                payload.get("total") if payload else None,
                payload.get("updated_at") if payload else None,
            )
        )
    return tuple(values)


def _status_counts(status: JobStatus) -> Dict[str, Any]:
    payload = status.raw.get("task_status") if isinstance(status.raw, dict) else None
    if not isinstance(payload, dict):
        return {
            "succeeded": "-",
            "failed": "-",
            "running": "-",
            "pending": "-",
            "total": "-",
        }
    return {
        "succeeded": payload.get("successful", payload.get("succeeded", 0)),
        "failed": payload.get("failed", 0),
        "running": payload.get("running", 0),
        "pending": payload.get("pending", 0),
        "total": payload.get("total", 0),
    }


def _status_summary(status: JobStatus) -> str:
    payload = status.raw.get("task_status") if isinstance(status.raw, dict) else None
    if isinstance(payload, dict):
        counts = _status_counts(status)
        return (
            f" - {counts['succeeded']} successful, {counts['failed']} failed, "
            f"{counts['running']} running, {counts['pending']} pending, "
            f"{counts['total']} total"
        )
    if status.message:
        return f" - {status.message}"
    return ""


def _status_html_color(state: str) -> str:
    state = str(state).lower()
    if state in {"complete", "completed", "success", "successful", "skipped"}:
        return "#16a34a"
    if state in {"failed", "cancelled", "canceled", "timeout"}:
        return "#dc2626"
    if state == "running":
        return "#8250df"
    if state == "pending":
        return "#9a6700"
    return "#6b7280"


def _header_cell(
    label: str,
    *,
    align: str = "left",
    border_left: bool = False,
) -> str:
    label = html.escape(label)
    align = "right" if align == "right" else "left"
    divider = (
        "border-left:1px solid #d0d7de; padding-left:16px; " if border_left else ""
    )
    return (
        "<th style='padding:6px 10px; font-size:11px; font-weight:800; "
        f"letter-spacing:0.02em; text-transform:uppercase; {divider}"
        f"text-align:{align}'>{label}</th>"
    )


def _count_cell(value: Any, color: str, *, border_left: bool = False) -> str:
    value = html.escape(str(value))
    divider = (
        "border-left:1px solid #d0d7de; padding-left:16px; " if border_left else ""
    )
    return (
        "<td style='padding:7px 10px; text-align:right; "
        f"font-variant-numeric:tabular-nums; {divider}"
        f"color:{color}; font-weight:750'>{value}</td>"
    )


def _site_label(run: RunHandle) -> str:
    site = getattr(run, "site", None)
    if site is None:
        return "site"
    return str(getattr(site, "site_name", None) or site.__class__.__name__)
