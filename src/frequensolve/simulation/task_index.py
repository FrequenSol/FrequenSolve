"""Fast consolidated index for committed Sauce task artifacts.

The index is a cache over atomically published task ``result.json`` records.
It is never used to infer artifact names: every payload path comes from the
producer-authored index, and fallback reads only fixed task-result paths.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from typing import Dict, Iterable, List, Mapping, Optional, Sized, Union

import h5py
import numpy as np

from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    ArtifactContractError,
    ArtifactRecord,
    ArtifactRequest,
    task_result_path,
)

__all__ = [
    "TASK_INDEX_VERSION",
    "TaskIndex",
    "TaskIndexEntry",
    "load_task_catalog",
    "task_index_path",
]

TASK_INDEX_VERSION = "fs-task-index-1"
_FINGERPRINT_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_TASK_STATUS = frozenset({"success", "failed", "skipped"})
_TIMING_DATASETS = (
    "timing_mesh",
    "timing_setup",
    "timing_assembly",
    "timing_solve_forward",
    "timing_solve_adjoint",
    "timing_imaging",
)


def task_index_path(result_path: Path | str) -> Path:
    """Return the single fixed path of the consolidated task index."""

    return Path(result_path) / "_fs_run" / "tasks.h5"


def _dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    value = group.get(name)
    if not isinstance(value, h5py.Dataset):
        raise ArtifactContractError(
            f"task index is missing dataset {group.name}/{name}"
        )
    if value.ndim != 1:
        raise ArtifactContractError(
            f"task index dataset {value.name} must be a flat array"
        )
    return value


def _integer_array(group: h5py.Group, name: str) -> np.ndarray:
    value = _dataset(group, name)
    if value.dtype.kind != "i" or value.dtype.itemsize != 8:
        raise ArtifactContractError(f"task index dataset {value.name} must be int64")
    return np.asarray(value[...], dtype=np.int64)


def _float_array(group: h5py.Group, name: str) -> np.ndarray:
    value = _dataset(group, name)
    if value.dtype.kind != "f" or value.dtype.itemsize != 8:
        raise ArtifactContractError(f"task index dataset {value.name} must be float64")
    return np.asarray(value[...], dtype=np.float64)


def _string_array(group: h5py.Group, name: str) -> tuple[str, ...]:
    value = _dataset(group, name)
    string_info = h5py.check_string_dtype(value.dtype)
    if string_info is None or string_info.encoding != "utf-8":
        raise ArtifactContractError(f"task index dataset {value.name} must be UTF-8")
    try:
        values = value.asstr()[...]
    except UnicodeError as exc:
        raise ArtifactContractError(
            f"task index dataset {value.name} contains invalid UTF-8"
        ) from exc
    return tuple(str(item) for item in values)


def _same_length(name: str, values: Sized, expected: int) -> None:
    if len(values) != expected:
        raise ArtifactContractError(
            f"task index column {name} has {len(values)} rows; expected {expected}"
        )


def _fingerprint(value: str, name: str) -> str:
    if not _FINGERPRINT_PATTERN.fullmatch(value):
        raise ArtifactContractError(f"task index {name} must be a SHA-256 fingerprint")
    return value


def _optional_finite(value: float, name: str, *, nonnegative: bool) -> Optional[float]:
    if math.isnan(value):
        return None
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        qualifier = "nonnegative " if nonnegative else ""
        raise ArtifactContractError(
            f"task index {name} must be a {qualifier}finite number or NaN"
        )
    return float(value)


@dataclass(frozen=True)
class TaskIndexEntry:
    """One task row from a consolidated artifact index."""

    task_id: int
    status: str
    frequency: complex
    fingerprints: Mapping[str, str]
    artifact_offset: int
    artifact_count: int
    iterations: Optional[int] = None
    residual: Optional[float] = None
    timings: Mapping[str, float] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        """Return whether this row describes reusable task output."""

        return self.status in {"success", "skipped"}


@dataclass(frozen=True)
class TaskIndex:
    """In-memory task and artifact catalog loaded from ``tasks.h5`` once."""

    result_path: Path
    path: Path
    tasks: Mapping[int, TaskIndexEntry]
    artifacts: tuple[ArtifactRecord, ...]

    @classmethod
    def read(cls, result_path: Path | str) -> "TaskIndex":
        """Read and strictly validate the fixed consolidated index."""

        root = Path(result_path).expanduser().resolve(strict=False)
        path = task_index_path(root)
        with h5py.File(path, "r") as h5:
            schema = h5.attrs.get("schema")
            if isinstance(schema, bytes):
                try:
                    schema = schema.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ArtifactContractError(
                        "task index schema attribute contains invalid UTF-8"
                    ) from exc
            if schema != TASK_INDEX_VERSION:
                raise ArtifactContractError(
                    f"unsupported task index schema {schema!r}; "
                    f"expected {TASK_INDEX_VERSION!r}"
                )
            task_group = h5.get("tasks")
            artifact_group = h5.get("artifacts")
            dependency_group = h5.get("dependencies")
            if not isinstance(task_group, h5py.Group):
                raise ArtifactContractError("task index is missing group /tasks")
            if not isinstance(artifact_group, h5py.Group):
                raise ArtifactContractError("task index is missing group /artifacts")
            if not isinstance(dependency_group, h5py.Group):
                raise ArtifactContractError("task index is missing group /dependencies")

            task_ids = _integer_array(task_group, "task_id")
            task_count = len(task_ids)
            statuses = _string_array(task_group, "status")
            frequency_real = _float_array(task_group, "frequency_real")
            frequency_imag = _float_array(task_group, "frequency_imag")
            fingerprint_job = _string_array(task_group, "fingerprint_job")
            fingerprint_simulation = _string_array(task_group, "fingerprint_simulation")
            fingerprint_outputs = _string_array(task_group, "fingerprint_outputs")
            artifact_offsets = _integer_array(task_group, "artifact_offset")
            artifact_counts = _integer_array(task_group, "artifact_count")
            iterations = _integer_array(task_group, "iterations")
            residuals = _float_array(task_group, "residual")
            timing_columns = {
                name: _float_array(task_group, name) for name in _TIMING_DATASETS
            }

            fingerprint_compatibility = (
                _string_array(task_group, "fingerprint_compatibility")
                if "fingerprint_compatibility" in task_group
                else ("",) * task_count
            )
            task_columns = {
                "fingerprint_compatibility": fingerprint_compatibility,
                "status": statuses,
                "frequency_real": frequency_real,
                "frequency_imag": frequency_imag,
                "fingerprint_job": fingerprint_job,
                "fingerprint_simulation": fingerprint_simulation,
                "fingerprint_outputs": fingerprint_outputs,
                "artifact_offset": artifact_offsets,
                "artifact_count": artifact_counts,
                "iterations": iterations,
                "residual": residuals,
                **timing_columns,
            }
            for name, values in task_columns.items():
                _same_length(name, values, task_count)

            artifact_ids = _string_array(artifact_group, "id")
            artifact_count = len(artifact_ids)
            artifact_columns = {
                "role": _string_array(artifact_group, "role"),
                "schema": _string_array(artifact_group, "schema"),
                "representation": _string_array(artifact_group, "representation"),
                "path": _string_array(artifact_group, "path"),
                "retention": _string_array(artifact_group, "retention"),
                "generation": _string_array(artifact_group, "generation"),
                "bytes": _integer_array(artifact_group, "bytes"),
                "dependency_offset": _integer_array(
                    artifact_group, "dependency_offset"
                ),
                "dependency_count": _integer_array(artifact_group, "dependency_count"),
            }
            artifact_columns["dataset_number"] = (
                _integer_array(artifact_group, "dataset_number")
                if "dataset_number" in artifact_group
                else np.zeros(artifact_count, dtype=np.int64)
            )
            for name, values in artifact_columns.items():
                _same_length(name, values, artifact_count)
            dependency_ids = _string_array(dependency_group, "id")

        if np.any(task_ids < 1) or (
            task_count > 1 and np.any(task_ids[1:] <= task_ids[:-1])
        ):
            raise ArtifactContractError(
                "task index task_id rows must be unique, positive, and sorted"
            )
        if np.any(~np.isfinite(frequency_real)) or np.any(~np.isfinite(frequency_imag)):
            raise ArtifactContractError("task index frequencies must be finite")
        if np.any(artifact_offsets < 0) or np.any(artifact_counts < 0):
            raise ArtifactContractError(
                "task index artifact offsets and counts must be nonnegative"
            )
        expected_offset = 0
        for offset, count in zip(artifact_offsets, artifact_counts):
            if int(offset) != expected_offset:
                raise ArtifactContractError(
                    "task index artifact slices must be contiguous and ordered"
                )
            expected_offset += int(count)
            if expected_offset > artifact_count:
                raise ArtifactContractError(
                    "task index artifact slice exceeds the artifact table"
                )
        if expected_offset != artifact_count:
            raise ArtifactContractError(
                "task index artifact slices do not cover the artifact table"
            )
        dependency_offsets = artifact_columns["dependency_offset"]
        dependency_counts = artifact_columns["dependency_count"]
        if np.any(dependency_offsets < 0) or np.any(dependency_counts < 0):
            raise ArtifactContractError(
                "task index dependency offsets and counts must be nonnegative"
            )
        expected_offset = 0
        for offset, count in zip(dependency_offsets, dependency_counts):
            if int(offset) != expected_offset:
                raise ArtifactContractError(
                    "task index dependency slices must be contiguous and ordered"
                )
            expected_offset += int(count)
            if expected_offset > len(dependency_ids):
                raise ArtifactContractError(
                    "task index dependency slice exceeds the dependency table"
                )
        if expected_offset != len(dependency_ids):
            raise ArtifactContractError(
                "task index dependency slices do not cover the dependency table"
            )
        if np.any(iterations < -1):
            raise ArtifactContractError(
                "task index iterations must be -1 or a nonnegative integer"
            )

        artifacts = []
        for row in range(artifact_count):
            generation = artifact_columns["generation"][row]
            payload = {
                "id": artifact_ids[row],
                "role": artifact_columns["role"][row],
                "schema": artifact_columns["schema"][row],
                "representation": artifact_columns["representation"][row],
                "path": artifact_columns["path"][row],
                "retention": artifact_columns["retention"][row],
                "bytes": int(artifact_columns["bytes"][row]),
            }
            if artifact_columns["dataset_number"][row]:
                payload["dataset_number"] = int(artifact_columns["dataset_number"][row])
            dependency_start = int(dependency_offsets[row])
            dependency_stop = dependency_start + int(dependency_counts[row])
            dependencies = dependency_ids[dependency_start:dependency_stop]
            if dependencies:
                payload["dependencies"] = list(dependencies)
            if generation:
                payload["generation"] = generation
            artifacts.append(ArtifactRecord.from_fs(payload, result_path=root))

        for raw_task_id, offset, count in zip(
            task_ids, artifact_offsets, artifact_counts
        ):
            start = int(offset)
            stop = start + int(count)
            keys = [
                (artifact.id, artifact.representation)
                for artifact in artifacts[start:stop]
            ]
            if len(keys) != len(set(keys)):
                raise ArtifactContractError(
                    "task index artifact id and representation must be unique "
                    f"within task {int(raw_task_id)}"
                )
            task_artifacts = artifacts[start:stop]
            unresolved = {
                dependency
                for artifact in task_artifacts
                for dependency in artifact.dependencies
                if sum(item.id == dependency for item in task_artifacts) != 1
            }
            if unresolved:
                raise ArtifactContractError(
                    "task index contains unresolved artifact dependencies for task "
                    f"{int(raw_task_id)}: {', '.join(sorted(unresolved))}"
                )

        tasks: Dict[int, TaskIndexEntry] = {}
        for row, raw_task_id in enumerate(task_ids):
            task_id = int(raw_task_id)
            status = statuses[row]
            if status not in _TASK_STATUS:
                raise ArtifactContractError(
                    f"task index status for task {task_id} is invalid: {status!r}"
                )
            fingerprints = {
                "job": _fingerprint(fingerprint_job[row], "fingerprint_job"),
                "simulation": _fingerprint(
                    fingerprint_simulation[row], "fingerprint_simulation"
                ),
                "outputs": _fingerprint(
                    fingerprint_outputs[row], "fingerprint_outputs"
                ),
            }
            if fingerprint_compatibility[row]:
                fingerprints["compatibility"] = _fingerprint(
                    fingerprint_compatibility[row], "fingerprint_compatibility"
                )
            timings = {}
            for name, values in timing_columns.items():
                duration = _optional_finite(
                    float(values[row]),
                    f"{name} for task {task_id}",
                    nonnegative=True,
                )
                if duration is not None:
                    timings[name.removeprefix("timing_")] = duration
            residual = _optional_finite(
                float(residuals[row]),
                f"residual for task {task_id}",
                nonnegative=True,
            )
            tasks[task_id] = TaskIndexEntry(
                task_id=task_id,
                status=status,
                frequency=complex(frequency_real[row], frequency_imag[row]),
                fingerprints=fingerprints,
                artifact_offset=int(artifact_offsets[row]),
                artifact_count=int(artifact_counts[row]),
                iterations=(None if iterations[row] == -1 else int(iterations[row])),
                residual=residual,
                timings=timings,
            )

        return cls(
            result_path=root,
            path=path,
            tasks=tasks,
            artifacts=tuple(artifacts),
        )

    def artifacts_for_task(self, task: int) -> tuple[ArtifactRecord, ...]:
        """Return artifacts for one task in producer order."""

        entry = self.tasks.get(int(task))
        if entry is None:
            return ()
        start = entry.artifact_offset
        return self.artifacts[start : start + entry.artifact_count]

    def query(
        self,
        *,
        id: Optional[str] = None,
        role: Optional[str] = None,
        representation: Optional[str] = None,
        retention: Optional[str] = None,
        task: Optional[int] = None,
    ) -> List[ArtifactRecord]:
        """Query artifacts with the same interface as ``ArtifactCatalog``."""

        selected = self.tasks if task is None else (int(task),)
        return [
            artifact
            for task_id in selected
            for artifact in self.artifacts_for_task(task_id)
            if (id is None or artifact.id == id)
            and (role is None or artifact.role == role)
            and (representation is None or artifact.representation == representation)
            and (retention is None or artifact.retention == retention)
        ]

    def select(
        self,
        request: ArtifactRequest,
        *,
        task: Optional[int] = None,
    ) -> List[ArtifactRecord]:
        """Select artifacts in representation-preference order."""

        matches = self.query(
            id=request.id,
            role=request.role,
            retention=request.retention,
            task=task,
        )
        if request.representations:
            accepted = set(request.representations)
            matches = [
                artifact for artifact in matches if artifact.representation in accepted
            ]
        return sorted(matches, key=request.preference)

    def require_one(
        self,
        request: ArtifactRequest,
        *,
        task: Optional[int] = None,
    ) -> ArtifactRecord:
        """Return one preferred artifact with ``ArtifactCatalog`` semantics."""

        matches = self.select(request, task=task)
        if matches and request.representations:
            preference = request.preference(matches[0])
            matches = [
                artifact
                for artifact in matches
                if request.preference(artifact) == preference
            ]
        if len(matches) != 1:
            scope = "the index" if task is None else f"task {int(task)}"
            raise ArtifactContractError(
                f"expected one {request.role!r} artifact for {scope}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def is_task_current(
        self,
        task: int,
        *,
        frequency: Optional[complex] = None,
        fingerprints: Optional[Mapping[str, str]] = None,
        required_artifact: Optional[ArtifactRequest] = None,
    ) -> bool:
        """Check task identity and optionally stat one required artifact."""

        entry = self.tasks.get(int(task))
        if entry is None or not entry.successful:
            return False
        if frequency is not None and entry.frequency != complex(frequency):
            return False
        if fingerprints is not None and any(
            entry.fingerprints.get(name) != digest
            for name, digest in fingerprints.items()
        ):
            return False
        if required_artifact is None:
            return True
        try:
            artifact = self.require_one(required_artifact, task=task)
            metadata = artifact.path.stat()
        except (ArtifactContractError, OSError):
            return False
        return S_ISREG(metadata.st_mode) and metadata.st_size == artifact.bytes


TaskCatalog = Union[TaskIndex, ArtifactCatalog]


def load_task_catalog(
    result_path: Path | str,
    *,
    tasks: Iterable[int],
) -> TaskCatalog:
    """Load ``tasks.h5`` or fall back to fixed task-result JSON paths.

    An index is stale when a known requested task result is absent from it or
    was committed after it. No directory traversal, filename inference, or
    payload HDF5 open participates in this decision.
    """

    root = Path(result_path).expanduser().resolve(strict=False)
    requested = tuple(dict.fromkeys(int(task) for task in tasks))
    if any(task < 1 for task in requested):
        raise ArtifactContractError("task index requests use one-based task ids")
    path = task_index_path(root)
    if not path.is_file():
        return ArtifactCatalog.read_task_results(root, tasks=requested)

    try:
        index = TaskIndex.read(root)
    except (ArtifactContractError, OSError):
        return ArtifactCatalog.read_task_results(root, tasks=requested)
    index_mtime = path.stat().st_mtime_ns
    for task in requested:
        result = task_result_path(root, task)
        try:
            result_mtime = result.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        if task not in index.tasks or result_mtime > index_mtime:
            return ArtifactCatalog.read_task_results(root, tasks=requested)
    return index
