"""Typed Sauce task-result and logical-artifact contract.

The contract reader deliberately knows nothing about Sauce's physical naming
conventions.  It opens only the fixed task-result control path and resolves
producer-authored artifact paths beneath the selected result directory.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "ARTIFACT_CONTRACT_VERSION",
    "COLLECTION_CONTRACT_VERSION",
    "OPERATION_CONTRACT_VERSION",
    "OPERATION_NAMES",
    "ArtifactCatalog",
    "ArtifactContractError",
    "ArtifactRequest",
    "ArtifactRecord",
    "OperationResult",
    "TaskPartition",
    "TaskResult",
    "load_operation_result",
    "operation_result_path",
    "task_result_path",
]

ARTIFACT_CONTRACT_VERSION = "fs-task-result-2"
COLLECTION_CONTRACT_VERSION = "fs-sharded-array-1"
OPERATION_CONTRACT_VERSION = "fs-operation-result-1"
OPERATION_NAMES = frozenset(
    {"pack", "smooth", "raytrace", "size", "validate", "init", "eikonal", "transient"}
)
_HASH_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_RETENTION = frozenset({"transient", "cache", "durable"})
_STATUS = frozenset({"success", "failed", "skipped"})


class ArtifactContractError(ValueError):
    """Raised when committed Sauce artifact metadata violates the contract."""


@dataclass(frozen=True)
class ArtifactRequest:
    """Typed consumer request for one logical artifact family.

    Representations are ordered by preference. An empty preference accepts any
    representation without changing producer order.
    """

    role: str
    representations: tuple[str, ...] = ()
    retention: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        role = _nonempty_string(self.role, "artifact request role")
        object.__setattr__(self, "role", role)
        representations = tuple(
            _nonempty_string(value, "artifact request representation")
            for value in self.representations
        )
        if len(set(representations)) != len(representations):
            raise ArtifactContractError(
                "artifact request representations must be unique"
            )
        object.__setattr__(self, "representations", representations)
        if self.retention is not None:
            retention = _nonempty_string(self.retention, "artifact request retention")
            if retention not in _RETENTION:
                raise ArtifactContractError(
                    "artifact request retention must be transient, cache, or durable"
                )
            object.__setattr__(self, "retention", retention)
        if self.id is not None:
            object.__setattr__(
                self, "id", _nonempty_string(self.id, "artifact request id")
            )

    def preference(self, artifact: "ArtifactRecord") -> int:
        """Return this request's ordering rank for a matching artifact."""

        if not self.representations:
            return 0
        try:
            return self.representations.index(artifact.representation)
        except ValueError:
            return len(self.representations)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(f"{name} must be an object")
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: Iterable[str],
    name: str,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        fields = ", ".join(repr(field) for field in unknown)
        raise ArtifactContractError(f"{name} contains unsupported fields: {fields}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactContractError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactContractError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ArtifactContractError(f"{name} must be at least {minimum}")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ArtifactContractError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ArtifactContractError(f"{name} must be a finite number")
    return result


def _fingerprint(value: Any, name: str) -> str:
    result = _nonempty_string(value, name)
    if not _HASH_PATTERN.fullmatch(result):
        raise ArtifactContractError(f"{name} must be a SHA-256 fingerprint")
    return result


def _artifact_path(value: Any, result_path: Path) -> tuple[str, Path]:
    raw = _nonempty_string(value, "artifact.path")
    if "\\" in raw:
        raise ArtifactContractError("artifact.path must use portable '/' separators")
    if re.match(r"^[A-Za-z]:", raw):
        raise ArtifactContractError("artifact.path must not be drive-qualified")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != raw
    ):
        raise ArtifactContractError("artifact.path must remain beneath ResultPath")
    root = result_path.expanduser().resolve(strict=False)
    resolved = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactContractError(
            "artifact.path resolves outside ResultPath"
        ) from exc
    return relative.as_posix(), resolved


def task_result_path(result_path: Path | str, task: int) -> Path:
    """Return the fixed task-result path for one one-based solver task."""

    task = _integer(task, "task", minimum=1)
    return Path(result_path) / "_fs_run" / "tasks" / f"task_{task:06d}" / "result.json"


@dataclass(frozen=True)
class ArtifactRecord:
    """One committed logical artifact or collection produced by Sauce."""

    id: str
    role: str
    representation: str
    schema: str
    relative_path: str
    path: Path
    retention: str
    bytes: int
    generation: Optional[str] = None
    dependencies: tuple[str, ...] = ()
    dataset_number: Optional[int] = None

    @classmethod
    def from_fs(
        cls,
        value: Mapping[str, Any],
        *,
        result_path: Path | str,
    ) -> "ArtifactRecord":
        """Parse and validate one producer-authored artifact record."""

        value = _mapping(value, "artifact")
        _reject_unknown(
            value,
            {
                "id",
                "role",
                "representation",
                "schema",
                "path",
                "retention",
                "generation",
                "bytes",
                "dependencies",
                "dataset_number",
            },
            "artifact",
        )
        retention = _nonempty_string(value.get("retention"), "artifact.retention")
        if retention not in _RETENTION:
            raise ArtifactContractError(
                "artifact.retention must be transient, cache, or durable"
            )
        relative_path, path = _artifact_path(value.get("path"), Path(result_path))
        generation = value.get("generation")
        if generation is not None:
            generation = _nonempty_string(generation, "artifact.generation")
        raw_dependencies = value.get("dependencies", [])
        if not isinstance(raw_dependencies, Sequence) or isinstance(
            raw_dependencies, (str, bytes, bytearray)
        ):
            raise ArtifactContractError("artifact.dependencies must be an array")
        dependencies = tuple(
            _nonempty_string(item, "artifact.dependencies[]")
            for item in raw_dependencies
        )
        if len(set(dependencies)) != len(dependencies):
            raise ArtifactContractError("artifact.dependencies must be unique")
        if (
            value.get("representation") == "packed_trace"
            and "dataset_number" not in value
        ):
            raise ArtifactContractError("packed_trace requires dataset_number")
        return cls(
            id=_nonempty_string(value.get("id"), "artifact.id"),
            role=_nonempty_string(value.get("role"), "artifact.role"),
            representation=_nonempty_string(
                value.get("representation"), "artifact.representation"
            ),
            schema=_nonempty_string(value.get("schema"), "artifact.schema"),
            relative_path=relative_path,
            path=path,
            retention=retention,
            generation=generation,
            bytes=_integer(value.get("bytes"), "artifact.bytes", minimum=0),
            dependencies=dependencies,
            dataset_number=(
                _integer(value["dataset_number"], "artifact.dataset_number", minimum=1)
                if "dataset_number" in value
                else None
            ),
        )

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the portable artifact record without its resolved path."""

        result: Dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "representation": self.representation,
            "schema": self.schema,
            "path": self.relative_path,
            "retention": self.retention,
            "bytes": self.bytes,
        }
        if self.generation is not None:
            result["generation"] = self.generation
        if self.dataset_number is not None:
            result["dataset_number"] = self.dataset_number
        if self.dependencies:
            result["dependencies"] = list(self.dependencies)
        return result


@dataclass(frozen=True)
class TaskPartition:
    """Exact partition identity for one solver task."""

    task: int
    frequency: complex
    task_count: Optional[int] = None
    source_batch: Optional[int] = None
    rhs_batch: Optional[int] = None

    @classmethod
    def from_fs(cls, value: Mapping[str, Any]) -> "TaskPartition":
        """Parse an exact complex-frequency task partition."""

        value = _mapping(value, "partition")
        _reject_unknown(
            value,
            {"task", "task_count", "frequency", "source_batch", "rhs_batch"},
            "partition",
        )
        frequency = _mapping(value.get("frequency"), "partition.frequency")
        _reject_unknown(frequency, {"real", "imag"}, "partition.frequency")
        task = _integer(value.get("task"), "partition.task", minimum=1)
        task_count = value.get("task_count")
        if task_count is not None:
            task_count = _integer(task_count, "partition.task_count", minimum=1)
            if task > task_count:
                raise ArtifactContractError(
                    "partition.task cannot exceed partition.task_count"
                )
        source_batch = value.get("source_batch")
        if source_batch is not None:
            source_batch = _integer(source_batch, "partition.source_batch", minimum=0)
        rhs_batch = value.get("rhs_batch")
        if rhs_batch is not None:
            rhs_batch = _integer(rhs_batch, "partition.rhs_batch", minimum=0)
        return cls(
            task=task,
            task_count=task_count,
            frequency=complex(
                _finite_float(frequency.get("real"), "partition.frequency.real"),
                _finite_float(frequency.get("imag"), "partition.frequency.imag"),
            ),
            source_batch=source_batch,
            rhs_batch=rhs_batch,
        )

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the exact task partition."""

        result: Dict[str, Any] = {
            "task": self.task,
            "frequency": {
                "real": float(self.frequency.real),
                "imag": float(self.frequency.imag),
            },
        }
        if self.task_count is not None:
            result["task_count"] = self.task_count
        if self.source_batch is not None:
            result["source_batch"] = self.source_batch
        if self.rhs_batch is not None:
            result["rhs_batch"] = self.rhs_batch
        return result


@dataclass(frozen=True)
class TaskResult:
    """One atomically committed Sauce task result."""

    path: Path
    result_path: Path
    partition: TaskPartition
    fingerprints: Dict[str, str]
    state: str
    code: int
    artifacts: tuple[ArtifactRecord, ...]
    solver: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(
        cls,
        path: Path | str,
        *,
        result_path: Path | str,
    ) -> "TaskResult":
        """Read and validate one committed task result."""

        path = Path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            raise
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(f"invalid task result JSON: {path}") from exc
        value = _mapping(value, "task result")
        _reject_unknown(
            value,
            {
                "schema",
                "partition",
                "fingerprints",
                "status",
                "solver",
                "timings",
                "resources",
                "timestamps",
                "build",
                "execution",
                "workflow",
                "license",
                "misc",
                "artifacts",
            },
            "task result",
        )
        schema = value.get("schema")
        if schema != ARTIFACT_CONTRACT_VERSION:
            raise ArtifactContractError(
                f"unsupported Sauce task-result contract {schema!r}; "
                f"expected {ARTIFACT_CONTRACT_VERSION!r}"
            )
        partition = TaskPartition.from_fs(_mapping(value.get("partition"), "partition"))
        raw_fingerprints = _mapping(value.get("fingerprints"), "fingerprints")
        fingerprints = {
            name: _fingerprint(raw_fingerprints.get(name), f"fingerprints.{name}")
            for name in ("job", "simulation", "outputs")
        }
        if "compatibility" in raw_fingerprints:
            fingerprints["compatibility"] = _fingerprint(
                raw_fingerprints["compatibility"], "fingerprints.compatibility"
            )
        status = _mapping(value.get("status"), "status")
        state = _nonempty_string(status.get("state"), "status.state")
        if state not in _STATUS:
            raise ArtifactContractError(
                "status.state must be success, failed, or skipped"
            )
        raw_artifacts = value.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ArtifactContractError("artifacts must be an array")
        root = Path(result_path).expanduser().resolve(strict=False)
        artifacts = tuple(
            ArtifactRecord.from_fs(item, result_path=root) for item in raw_artifacts
        )
        keys = [(item.id, item.representation) for item in artifacts]
        if len(set(keys)) != len(keys):
            raise ArtifactContractError(
                "artifact id and representation must be unique within a task"
            )
        dependency_matches = {
            dependency: sum(item.id == dependency for item in artifacts)
            for artifact in artifacts
            for dependency in artifact.dependencies
        }
        unresolved = {
            dependency
            for dependency, matches in dependency_matches.items()
            if matches != 1
        }
        if unresolved:
            raise ArtifactContractError(
                "artifact dependencies must resolve uniquely within the task: "
                + ", ".join(sorted(unresolved))
            )
        raw_timings = _mapping(value.get("timings", {}), "timings")
        timings = {
            str(name): _finite_float(duration, f"timings.{name}")
            for name, duration in raw_timings.items()
        }
        if any(duration < 0.0 for duration in timings.values()):
            raise ArtifactContractError("task timings must be nonnegative")
        return cls(
            path=path,
            result_path=root,
            partition=partition,
            fingerprints=fingerprints,
            state=state,
            code=_integer(status.get("code"), "status.code"),
            artifacts=artifacts,
            solver=dict(_mapping(value.get("solver", {}), "solver")),
            timings=timings,
            resources=dict(_mapping(value.get("resources", {}), "resources")),
            timestamps=dict(_mapping(value.get("timestamps", {}), "timestamps")),
            metadata={
                key: dict(_mapping(value[key], key))
                for key in ("build", "execution", "workflow", "license", "misc")
                if key in value
            },
        )

    @property
    def successful(self) -> bool:
        """Return whether the task committed successful or skipped outputs."""

        return self.state in {"success", "skipped"} and self.code == 0

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this validated task result using portable artifact paths."""

        result: Dict[str, Any] = {
            "schema": ARTIFACT_CONTRACT_VERSION,
            "partition": self.partition.to_fs(),
            "fingerprints": dict(self.fingerprints),
            "status": {"state": self.state, "code": self.code},
            "artifacts": [artifact.to_fs() for artifact in self.artifacts],
        }
        for name, value in (
            ("solver", self.solver),
            ("timings", self.timings),
            ("resources", self.resources),
            ("timestamps", self.timestamps),
        ):
            if value:
                result[name] = dict(value)
        result.update(self.metadata)
        return result


def _operation_name(value: Any, name: str = "operation name") -> str:
    result = _nonempty_string(value, name)
    if result not in OPERATION_NAMES:
        known = ", ".join(sorted(OPERATION_NAMES))
        raise ArtifactContractError(
            f"{name} must be one of the canonical operations: {known}"
        )
    return result


def operation_result_path(
    result_path: Path | str,
    workflow: str,
) -> Path:
    """Return one fixed operation-result path without filesystem discovery."""

    name = _operation_name(workflow, "requested operation")
    return Path(result_path) / "_fs_run" / "operations" / name / "result.json"


@dataclass(frozen=True)
class OperationResult:
    """One atomically committed non-frequency Sauce operation result."""

    path: Path
    result_path: Path
    name: str
    generation: str
    fingerprints: Dict[str, str]
    state: str
    code: int
    artifacts: tuple[ArtifactRecord, ...]
    solver: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(
        cls,
        path: Path | str,
        *,
        result_path: Path | str,
        workflow: str,
    ) -> "OperationResult":
        """Read and strictly validate one requested fixed operation result."""

        requested = _operation_name(workflow, "requested operation")
        path = Path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            raise
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(
                f"invalid operation result JSON: {path}"
            ) from exc
        value = _mapping(value, "operation result")
        _reject_unknown(
            value,
            {
                "schema",
                "operation",
                "fingerprints",
                "status",
                "solver",
                "timings",
                "resources",
                "timestamps",
                "build",
                "execution",
                "workflow",
                "license",
                "misc",
                "artifacts",
            },
            "operation result",
        )
        if value.get("schema") != OPERATION_CONTRACT_VERSION:
            raise ArtifactContractError(
                f"unsupported Sauce operation-result contract "
                f"{value.get('schema')!r}; expected {OPERATION_CONTRACT_VERSION!r}"
            )
        operation = _mapping(value.get("operation"), "operation")
        _reject_unknown(operation, {"name", "generation"}, "operation")
        name = _operation_name(operation.get("name"))
        if name != requested:
            raise ArtifactContractError(
                f"operation result reports {name!r}; requested {requested!r}"
            )
        generation = _nonempty_string(
            operation.get("generation"), "operation.generation"
        )
        raw_fingerprints = _mapping(value.get("fingerprints"), "fingerprints")
        _reject_unknown(
            raw_fingerprints,
            {"job", "simulation", "outputs"},
            "fingerprints",
        )
        fingerprints = {
            field_name: _fingerprint(
                raw_fingerprints.get(field_name),
                f"fingerprints.{field_name}",
            )
            for field_name in ("job", "simulation", "outputs")
        }
        status = _mapping(value.get("status"), "status")
        _reject_unknown(status, {"state", "code"}, "status")
        state = _nonempty_string(status.get("state"), "status.state")
        if state not in _STATUS:
            raise ArtifactContractError(
                "status.state must be success, failed, or skipped"
            )
        raw_artifacts = value.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ArtifactContractError("artifacts must be an array")
        root = Path(result_path).expanduser().resolve(strict=False)
        artifacts = tuple(
            ArtifactRecord.from_fs(item, result_path=root) for item in raw_artifacts
        )
        keys = [(item.id, item.representation) for item in artifacts]
        if len(keys) != len(set(keys)):
            raise ArtifactContractError(
                "artifact id and representation must be unique within an operation"
            )
        unresolved = {
            dependency
            for artifact in artifacts
            for dependency in artifact.dependencies
            if sum(item.id == dependency for item in artifacts) != 1
        }
        if unresolved:
            raise ArtifactContractError(
                "artifact dependencies must resolve uniquely within the operation: "
                + ", ".join(sorted(unresolved))
            )
        raw_timings = _mapping(value.get("timings", {}), "timings")
        timings = {
            str(field_name): _finite_float(duration, f"timings.{field_name}")
            for field_name, duration in raw_timings.items()
        }
        if any(duration < 0.0 for duration in timings.values()):
            raise ArtifactContractError("operation timings must be nonnegative")
        return cls(
            path=path,
            result_path=root,
            name=name,
            generation=generation,
            fingerprints=fingerprints,
            state=state,
            code=_integer(status.get("code"), "status.code"),
            artifacts=artifacts,
            solver=dict(_mapping(value.get("solver", {}), "solver")),
            timings=timings,
            resources=dict(_mapping(value.get("resources", {}), "resources")),
            timestamps=dict(_mapping(value.get("timestamps", {}), "timestamps")),
            metadata={
                key: dict(_mapping(value[key], key))
                for key in ("build", "execution", "workflow", "license", "misc")
                if key in value
            },
        )

    @property
    def successful(self) -> bool:
        """Return whether the operation committed reusable output."""

        return self.state in {"success", "skipped"} and self.code == 0

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this validated operation result with portable paths."""

        result: Dict[str, Any] = {
            "schema": OPERATION_CONTRACT_VERSION,
            "operation": {"name": self.name, "generation": self.generation},
            "fingerprints": dict(self.fingerprints),
            "status": {"state": self.state, "code": self.code},
            "artifacts": [artifact.to_fs() for artifact in self.artifacts],
        }
        for name, value in (
            ("solver", self.solver),
            ("timings", self.timings),
            ("resources", self.resources),
            ("timestamps", self.timestamps),
        ):
            if value:
                result[name] = dict(value)
        result.update(self.metadata)
        return result


def load_operation_result(
    result_path: Path | str,
    workflow: str,
) -> OperationResult:
    """Load one explicitly requested operation from its fixed known path."""

    path = operation_result_path(result_path, workflow)
    return OperationResult.read(
        path,
        result_path=result_path,
        workflow=workflow,
    )


@dataclass(frozen=True)
class ArtifactCatalog:
    """Task-indexed logical artifacts loaded without filesystem discovery."""

    result_path: Path
    results: Dict[int, TaskResult]

    @classmethod
    def read_task_results(
        cls,
        result_path: Path | str,
        *,
        tasks: Iterable[int],
    ) -> "ArtifactCatalog":
        """Load only the known task-result paths for an expected task set."""

        root = Path(result_path).expanduser().resolve(strict=False)
        results: Dict[int, TaskResult] = {}
        for task in dict.fromkeys(int(item) for item in tasks):
            path = task_result_path(root, task)
            if not path.is_file():
                continue
            result = TaskResult.read(path, result_path=root)
            if result.partition.task != task:
                raise ArtifactContractError(
                    f"task result {path} reports task {result.partition.task}, "
                    f"expected {task}"
                )
            results[task] = result
        return cls(result_path=root, results=results)

    def artifacts_for_task(self, task: int) -> tuple[ArtifactRecord, ...]:
        """Return committed artifacts for one one-based task."""

        result = self.results.get(int(task))
        return () if result is None else result.artifacts

    def query(
        self,
        *,
        id: Optional[str] = None,
        role: Optional[str] = None,
        representation: Optional[str] = None,
        retention: Optional[str] = None,
        task: Optional[int] = None,
    ) -> List[ArtifactRecord]:
        """Query loaded artifacts using stable logical metadata."""

        if task is None:
            selected: Iterable[TaskResult] = self.results.values()
        else:
            result = self.results.get(int(task))
            selected = () if result is None else (result,)
        return [
            artifact
            for result in selected
            for artifact in result.artifacts
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
        """Select matching artifacts in representation-preference order."""

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
        """Return exactly one preferred artifact or raise a contract error."""

        matches = self.select(request, task=task)
        if matches and request.representations:
            preferred = request.preference(matches[0])
            matches = [
                artifact
                for artifact in matches
                if request.preference(artifact) == preferred
            ]
        if len(matches) != 1:
            scope = "the catalog" if task is None else f"task {int(task)}"
            raise ArtifactContractError(
                f"expected one {request.role!r} artifact for {scope}, "
                f"found {len(matches)}"
            )
        return matches[0]
