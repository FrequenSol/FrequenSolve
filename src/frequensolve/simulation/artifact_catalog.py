"""Unified logical artifacts from explicitly known task and operation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from frequensolve.simulation.artifact_contract import (
    ArtifactContractError,
    ArtifactRecord,
    ArtifactRequest,
    OperationResult,
    load_operation_result,
    operation_result_path,
)
from frequensolve.simulation.task_index import TaskCatalog, load_task_catalog

__all__ = ["CombinedArtifactCatalog", "load_artifact_catalog"]


def _task_ids(catalog: TaskCatalog) -> Iterable[int]:
    tasks = getattr(catalog, "tasks", None)
    if tasks is None:
        tasks = getattr(catalog, "results", None)
    if not isinstance(tasks, Mapping):
        raise ArtifactContractError("task catalog does not expose task identities")
    return tasks


@dataclass(frozen=True)
class CombinedArtifactCatalog:
    """Task artifacts plus only the non-frequency operations requested by name."""

    result_path: Path
    task_catalog: TaskCatalog
    operations: Mapping[str, OperationResult]

    def artifacts_for_task(self, task: int) -> tuple[ArtifactRecord, ...]:
        """Return task-local records in producer order."""

        return self.task_catalog.artifacts_for_task(task)

    def artifacts_for_operation(self, operation: str) -> tuple[ArtifactRecord, ...]:
        """Return records for one explicitly loaded operation."""

        result = self.operations.get(operation)
        return () if result is None else result.artifacts

    def artifact_scopes(
        self,
    ) -> Mapping[tuple[str, object], tuple[ArtifactRecord, ...]]:
        """Return dependency-local record groups without filesystem discovery."""

        scopes: dict[tuple[str, object], tuple[ArtifactRecord, ...]] = {
            ("task", int(task)): tuple(self.task_catalog.query(task=int(task)))
            for task in _task_ids(self.task_catalog)
        }
        scopes.update(
            {
                ("operation", name): tuple(result.artifacts)
                for name, result in self.operations.items()
            }
        )
        return scopes

    def query(
        self,
        *,
        id: Optional[str] = None,
        role: Optional[str] = None,
        representation: Optional[str] = None,
        retention: Optional[str] = None,
        task: Optional[int] = None,
        operation: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        """Query task and explicitly loaded operation records."""

        if task is not None and operation is not None:
            raise ArtifactContractError("artifact query cannot mix task and operation")
        if operation is not None:
            selected = self.artifacts_for_operation(operation)
        elif task is not None:
            selected = self.artifacts_for_task(task)
        else:
            selected = tuple(self.task_catalog.query()) + tuple(
                artifact
                for result in self.operations.values()
                for artifact in result.artifacts
            )
        return [
            artifact
            for artifact in selected
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
        operation: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        """Select records in representation-preference order."""

        matches = self.query(
            id=request.id,
            role=request.role,
            retention=request.retention,
            task=task,
            operation=operation,
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
        operation: Optional[str] = None,
    ) -> ArtifactRecord:
        """Return exactly one preferred record in the requested scope."""

        matches = self.select(request, task=task, operation=operation)
        if matches and request.representations:
            preference = request.preference(matches[0])
            matches = [
                artifact
                for artifact in matches
                if request.preference(artifact) == preference
            ]
        if len(matches) != 1:
            if task is not None:
                scope = f"task {int(task)}"
            elif operation is not None:
                scope = f"operation {operation!r}"
            else:
                scope = "the catalog"
            raise ArtifactContractError(
                f"expected one {request.role!r} artifact for {scope}, "
                f"found {len(matches)}"
            )
        return matches[0]


def load_artifact_catalog(
    result_path: Path | str,
    *,
    tasks: Iterable[int],
    operations: Iterable[str] = (),
) -> CombinedArtifactCatalog:
    """Load task metadata and explicitly named fixed operation-result paths.

    Missing requested operations simply have no committed result yet. Malformed
    committed results are contract errors. This function never scans the result
    tree or infers an artifact name.
    """

    root = Path(result_path).expanduser().resolve(strict=False)
    task_catalog = load_task_catalog(root, tasks=tasks)
    loaded = {}
    for operation in dict.fromkeys(operations):
        path = operation_result_path(root, operation)
        if path.is_file():
            loaded[operation] = load_operation_result(root, operation)
    return CombinedArtifactCatalog(
        result_path=root,
        task_catalog=task_catalog,
        operations=loaded,
    )
