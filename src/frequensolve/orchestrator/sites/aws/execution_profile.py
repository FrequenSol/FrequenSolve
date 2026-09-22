"""Closed profile contract for the managed Slurm execution site."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

MANAGED_EXECUTION_PROFILE_FIELDS = frozenset(
    {"execution_site_id", "execution_resources"}
)


class ManagedExecutionProfileError(ValueError):
    """The managed site profile has an unsupported execution shape."""


@dataclass(frozen=True)
class ManagedExecutionProfile:
    """Execution settings sourced from one named site profile."""

    execution_site_id: str = "managed-slurm"
    execution_resources: dict[str, int] = field(
        default_factory=lambda: {
            "nodes": 1,
            "mpi_ranks": 1,
            "wall_time_seconds": 3600,
        }
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ManagedExecutionProfile":
        if not isinstance(values, Mapping):
            raise ManagedExecutionProfileError("Execution profile must be a mapping")
        if values.keys() - MANAGED_EXECUTION_PROFILE_FIELDS:
            raise ManagedExecutionProfileError(
                "Use execution_site_id and execution_resources for managed execution"
            )
        site_id = values.get("execution_site_id", "managed-slurm")
        if site_id != "managed-slurm":
            raise ManagedExecutionProfileError(
                "execution_site_id must be managed-slurm"
            )
        resources = values.get("execution_resources")
        if resources is None:
            resources = {"nodes": 1, "mpi_ranks": 1, "wall_time_seconds": 3600}
        if not isinstance(resources, Mapping):
            raise ManagedExecutionProfileError("execution_resources must be a table")
        required = {"nodes", "mpi_ranks", "wall_time_seconds"}
        limits = {
            "nodes": (1, 2),
            "mpi_ranks": (1, 8),
            "wall_time_seconds": (60, 7200),
            "cpu": (1, 128),
            "memory_mib": (1, 262144),
            "planner_memory_mib": (1, 124518),
        }
        if not required <= resources.keys() or resources.keys() - limits.keys():
            raise ManagedExecutionProfileError(
                "execution_resources needs nodes, mpi_ranks, wall_time_seconds; optional cpu and memory_mib"
            )
        validated = {
            key: _bounded_integer(value, key, *limits[key])
            for key, value in resources.items()
        }
        nodes, ranks = validated["nodes"], validated["mpi_ranks"]
        if ranks % nodes or ranks // nodes > 4 or (nodes == 1 and ranks != 1):
            raise ManagedExecutionProfileError(
                "Unsupported node/MPI-rank combination at managed-slurm"
            )
        return cls(execution_site_id=site_id, execution_resources=validated)

    def graphql_arguments(self) -> dict[str, Any]:
        """Return the hosted submitJob arguments for this profile."""
        names = {
            "mpi_ranks": "mpiRanks",
            "wall_time_seconds": "wallTimeSeconds",
            "memory_mib": "memoryMiB",
            "planner_memory_mib": "plannerMemoryMiB",
        }
        return {
            "execution_site_id": self.execution_site_id,
            "execution_resources": {
                names.get(k, k): v for k, v in self.execution_resources.items()
            },
        }


def _bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ManagedExecutionProfileError(
            f"unsupported Slurm resource shape: {name} must be from {minimum} through {maximum}"
        )
    return value
