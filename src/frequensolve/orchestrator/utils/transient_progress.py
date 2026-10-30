"""Advisory accepted-step progress; scheduler status remains authoritative."""

from __future__ import annotations

import json
import math
from datetime import datetime


def read_local_progress(job):
    """Read fixed operation/task paths without scanning simulation output."""
    payloads = {}
    for task in range(1, job.n_tasks + 1):
        path = job.task_manifest_file(task).with_name("progress.json")
        try:
            payloads[task] = json.loads(path.read_text())
        except (OSError, ValueError):
            payloads[task] = None
    return payloads


def _parse_timestamp(value):
    """Accept Sauce's compact timezone offset on supported Python versions."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")


def _completed_steps(payload, total, not_before):
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != "fs-transient-progress-1":
        return None
    completed = payload.get("completed_steps")
    if (
        type(completed) is not int
        or type(payload.get("total_steps")) is not int
        or payload["total_steps"] != total
        or not 0 <= completed <= total
    ):
        return None
    if not_before is not None:
        try:
            started = _parse_timestamp(payload["started_at"])
            # Sauce's run clock records whole seconds. A solve can start in
            # the same second as a fractional client submission timestamp.
            if started.tzinfo is None or started.timestamp() < math.floor(not_before):
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
    return completed


def _estimated_completion(payload, completed, total):
    """Estimate integration finish from this attempt's measured advances only."""
    initial = payload.get("initial_step")
    elapsed = payload.get("elapsed_seconds")
    if type(initial) is not int or not 0 <= initial < completed:
        return None
    if type(elapsed) not in (int, float):
        return None
    try:
        if not math.isfinite(elapsed) or elapsed <= 0:
            return None
        updated = _parse_timestamp(payload["updated_at"])
        if updated.tzinfo is None:
            return None
        estimate = updated.timestamp() + elapsed * (total - completed) / (
            completed - initial
        )
        # Validate datetime range before formatting an untrusted status payload.
        return (
            datetime.fromtimestamp(estimate).astimezone().isoformat(timespec="seconds")
        )
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None


def update_progress(run, status):
    """Attach step counts to a transient status without changing its lifecycle."""
    job = run.job
    if getattr(job, "workflow", None) != "transient":
        return
    total_per_task = job.time.steps
    total = total_per_task * job.n_tasks
    completed = None
    estimated_completion = None
    if status.is_successful:
        completed = total
    elif status.is_queued:
        completed = 0
    else:
        reader = getattr(run.site, "_read_transient_progress", None)
        payloads = reader(run) if callable(reader) else {}
        current = set((run.backend.get("task_plan") or {}).get("current_tasks", []))
        cached = run.backend.setdefault("transient_completed_steps", {})
        estimates = run.backend.setdefault("transient_completion_estimates", {})
        for task in range(1, job.n_tasks + 1):
            if task in current:
                cached[task] = total_per_task
                continue
            value = _completed_steps(
                payloads.get(task),
                total_per_task,
                run.backend.get("progress_not_before"),
            )
            if value is not None:
                cached[task] = value
                estimates[task] = _estimated_completion(
                    payloads[task], value, total_per_task
                )
        if cached:
            completed = sum(cached.values())
        unfinished = [
            task
            for task in range(1, job.n_tasks + 1)
            if cached.get(task) != total_per_task
        ]
        if (
            status.is_running
            and unfinished
            and all(estimates.get(task) for task in unfinished)
        ):
            estimated_completion = max(
                (estimates[task] for task in unfinished),
                key=lambda stamp: datetime.fromisoformat(stamp).timestamp(),
            )
    status.raw["time_progress"] = {"completed_steps": completed, "total_steps": total}
    if estimated_completion is not None:
        status.raw["time_progress"]["estimated_completion"] = estimated_completion
    status.progress = None if completed is None else completed / total
    summary = progress_summary(status.raw["time_progress"])
    # Preserve diagnostics on failure; task counts remain available in raw status.
    if status.is_complete and not status.is_successful and status.message:
        if not status.message.endswith(summary):
            status.message = f"{status.message}; {summary}"
    else:
        status.message = summary


def progress_summary(payload):
    completed = payload["completed_steps"]
    total = payload["total_steps"]
    if completed is None:
        return f"waiting for solver progress ({total} time steps)"
    summary = f"{completed}/{total} time steps completed"
    if payload.get("estimated_completion"):
        finish = datetime.fromisoformat(payload["estimated_completion"]).astimezone()
        summary += f"; estimated finish ~{finish:%Y-%m-%d %H:%M:%S %Z}"
    return summary
