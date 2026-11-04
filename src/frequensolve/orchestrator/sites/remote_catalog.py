"""Explain why a remote task catalog entry cannot be reused for a staged job.

A remote fetch accepts a task only when its committed record (the fixed
``result.json`` or a ``tasks.h5`` row) exists, carries the current staged
fingerprints, describes the expected partition, and reports success. Each
refusal raises its own ``RemoteTaskError`` subclass, so a task that ran for the
current staging and failed is never reported as stale staging.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Union

from frequensolve.simulation.artifact_contract import ArtifactCatalog, TaskResult
from frequensolve.simulation.task_index import TaskIndex, TaskIndexEntry

__all__ = [
    "RemoteTaskError",
    "RemoteTaskFailedError",
    "RemoteTaskMissingError",
    "RemoteTaskPartitionError",
    "RemoteTaskStaleError",
    "validate_remote_task_catalog",
]

TaskRecord = Union[TaskResult, TaskIndexEntry]


class RemoteTaskError(RuntimeError):
    """A remote task record cannot be reused for the current staged job."""

    def __init__(self, task: int, message: str):
        super().__init__(message)
        self.task = int(task)


class RemoteTaskMissingError(RemoteTaskError):
    """The remote catalog holds no committed record for the task."""


class RemoteTaskStaleError(RemoteTaskError):
    """The record was committed by another staged generation of the job."""


class RemoteTaskPartitionError(RemoteTaskError):
    """The record describes another frequency or angular partition."""


class RemoteTaskFailedError(RemoteTaskError):
    """The task ran for the current staged job and failed."""


def validate_remote_task_catalog(
    job: Any,
    catalog: Union[ArtifactCatalog, TaskIndex],
    tasks: Iterable[int],
    *,
    site_name: str,
    remote_root: Optional[Union[str, PurePosixPath]] = None,
) -> None:
    """Require exact task partitions and current staged provenance.

    ``remote_root`` is the remote result directory; when given, refusals name
    the remote ``result.json`` and the task log beside it in ``../logs``.
    """

    expected = getattr(
        job, "staged_task_fingerprints", job.staged_artifact_fingerprints
    )(site_name)
    if expected is None:
        raise RuntimeError(
            "Remote staged provenance is missing; restage the job before fetching"
        )
    angular = getattr(job, "n_modes", None) is not None
    indexed = isinstance(catalog, TaskIndex)
    for task in tasks:
        if angular:
            if indexed:
                raise RemoteTaskError(
                    task,
                    f"Remote angular task {task} cannot be validated from the "
                    "task index; its result.json is required",
                )
            result = catalog.results.get(task)
            partition = None
            if result is not None and (
                result.partition.task != task
                or result.partition.task_count != job.n_tasks
                or result.partition.azimuthal_mode != task - 1
            ):
                partition = (
                    f"reports azimuthal mode {result.partition.azimuthal_mode} of "
                    f"{result.partition.task_count} tasks instead of mode "
                    f"{task - 1} of {job.n_tasks}"
                )
            error = _refusal(
                task,
                result,
                expected,
                indexed=False,
                label=f"angular task {task} (azimuthal mode {task - 1})",
                partition=partition,
                remote_root=remote_root,
            )
            if error is not None:
                raise error
            continue
        frequency = complex(job.f_list[task - 1])
        record: Optional[TaskRecord]
        if indexed:
            if catalog.is_task_current(
                task, frequency=frequency, fingerprints=expected
            ):
                continue
            record = catalog.tasks.get(task)
        else:
            record = catalog.results.get(task)
        error = _refusal(
            task,
            record,
            expected,
            indexed=indexed,
            label=f"task {task} ({_frequency_label(frequency)})",
            partition=_frequency_mismatch(record, frequency),
            remote_root=remote_root,
        )
        if error is not None:
            raise error
        if indexed:
            raise RemoteTaskError(
                task, f"Remote task {task} does not match the current staged job"
            )


def _frequency_mismatch(
    record: Optional[TaskRecord], frequency: complex
) -> Optional[str]:
    """Describe a record whose partition is not the expected frequency."""

    if record is None:
        return None
    if isinstance(record, TaskIndexEntry):
        found = record.frequency
    else:
        found = record.partition.frequency
    if found == frequency:
        return None
    if found is None:
        return f"reports azimuthal mode {record.partition.azimuthal_mode} instead"
    return f"reports frequency {_frequency_label(found)} instead"


def _refusal(
    task: int,
    record: Optional[TaskRecord],
    expected: Mapping[str, str],
    *,
    indexed: bool,
    label: str,
    partition: Optional[str],
    remote_root: Optional[Union[str, PurePosixPath]],
) -> Optional[RemoteTaskError]:
    """Return the first reason ``record`` is not current, or ``None``."""

    relative = f"_fs_run/tasks/task_{task:06d}/result.json"
    if remote_root is None:
        result_path = f"{relative} in the remote result directory"
        log = f"Task log: site.fetch_logs(job, task={task}) downloads it."
    else:
        root = PurePosixPath(remote_root)
        result_path = str(root / relative)
        log = (
            f"Task log: {root.parent / 'logs' / f'task_{task}.log'} "
            f"(site.fetch_logs(job, task={task}) downloads it)."
        )
    if record is None:
        where = (
            "the remote task index has no row for it"
            if indexed
            else f"no result at {result_path}"
        )
        return RemoteTaskMissingError(
            task,
            f"Remote {label} has not committed a result ({where}); the task "
            "is still running, stopped before publishing, or never started. " + log,
        )
    differing = [
        name
        for name, digest in expected.items()
        if record.fingerprints.get(name) != digest
    ]
    if differing:
        details = ", ".join(
            f"{name} {_digest(record.fingerprints.get(name))} != staged "
            f"{_digest(expected[name])}"
            for name in differing
        )
        return RemoteTaskStaleError(
            task,
            f"Remote {label} does not match the current staged job: "
            f"{'its task index row' if indexed else result_path} was written by "
            f"another staged generation ({details}; that run reported "
            f"{_status(record)}). Restage and resubmit the job, or check "
            "whether another client restaged it.",
        )
    if partition is not None:
        return RemoteTaskPartitionError(
            task, f"Remote {label} has the current fingerprints but {partition}."
        )
    if record.successful:
        return None
    lines = [
        f"Remote {label} ran for the current staged job and failed: "
        f"{_status(record)}."
    ]
    if indexed:
        if record.iterations is not None or record.residual is not None:
            lines.append(
                f"Task index summary: {_value(record.iterations)} iterations, "
                f"residual {_value(record.residual)}."
            )
        lines.append(f"Per-solve contexts and tolerances: {result_path}")
    else:
        lines.extend(_solve_failures(record.solver))
        lines.append(f"Task result: {result_path}")
    lines.append(log)
    return RemoteTaskFailedError(task, "\n".join(lines))


def _solve_failures(solver: Mapping[str, Any]) -> list[str]:
    """Describe the unconverged linear solves Sauce recorded for one task."""

    convergence = solver.get("convergence")
    if not isinstance(convergence, Mapping):
        return ["The task published no solver convergence record."]
    solves = convergence.get("solves")
    solves = solves if isinstance(solves, list) else []
    failures = [
        (index, solve)
        for index, solve in enumerate(solves)
        if isinstance(solve, Mapping)
        and not solve.get("converged", solve.get("status") == "converged")
    ]
    if not failures:
        return [
            f"Solver convergence status {convergence.get('status')!r} records no "
            "failed solve; the failure happened outside the linear solves."
        ]
    lines = [f"Failed solves ({len(failures)} of {len(solves)}):"]
    for index, solve in failures:
        lines.append(
            f"  solves[{index}] {solve.get('context', '?')}: "
            f"{solve.get('status', 'not converged')} after "
            f"{_value(solve.get('iterations'))} iterations, residual "
            f"{_value(solve.get('residual'))}, tolerance "
            f"{_value(solve.get('tolerance'))} ({solve.get('solver', '?')} "
            f"grid {_value(solve.get('grid'))}, code {_value(solve.get('code'))})"
        )
    return lines


def _status(record: TaskRecord) -> str:
    if isinstance(record, TaskIndexEntry):
        return f"task index status {record.status!r}"
    return f"status state {record.state!r}, code {record.code}"


def _frequency_label(value: complex) -> str:
    value = complex(value)
    if value.imag == 0.0:
        return f"{value.real:.6g} Hz"
    return f"{value.real:.6g}{value.imag:+.6g}j Hz"


def _digest(value: Optional[str]) -> str:
    return "missing" if value is None else value[:19]


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "?" if value is None else str(value)
