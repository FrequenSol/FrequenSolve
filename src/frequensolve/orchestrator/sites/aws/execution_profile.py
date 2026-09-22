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
    execution_resources: dict[str, Any] = field(
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
        if resources.get("mode") == "adaptive-allocation.v1":
            from frequensolve.adaptive import AdaptivePool

            if set(resources) != {
                "mode",
                "nodes",
                "mpi_ranks",
                "wall_time_seconds",
                "pool",
            }:
                raise ManagedExecutionProfileError(
                    "Adaptive resources require a pool without planner or frequency ceilings"
                )
            nodes = _bounded_integer(resources["nodes"], "nodes", 1, 2)
            ranks = _bounded_integer(resources["mpi_ranks"], "mpi_ranks", 1, 8)
            seconds = _bounded_integer(
                resources["wall_time_seconds"], "wall_time_seconds", 60, 7200
            )
            if ranks % nodes or not isinstance(resources["pool"], Mapping):
                raise ManagedExecutionProfileError(
                    "Adaptive pool requires uniform ranks per node"
                )
            try:
                pool = AdaptivePool.from_mapping(
                    {
                        "nodes": nodes,
                        "ranks_per_node": ranks // nodes,
                        "wall_time_seconds": seconds,
                        **resources["pool"],
                    }
                )
            except (ValueError, TypeError) as error:
                raise ManagedExecutionProfileError(str(error)) from error
            if set(resources["pool"]) != {
                "threads_per_rank",
                "memory_mib_per_node",
                "partition",
            } or (
                pool.threads_per_rank > 64
                or ranks * pool.threads_per_rank > 128
                or pool.memory_mib_per_node
                % (pool.ranks_per_node * pool.threads_per_rank)
                != 0
                or pool.memory_mib_per_node % (pool.ranks_per_node * 256) != 0
                or pool.memory_mib_per_node > 124518
                or pool.partition not in {"cpu-single", "cpu-efa"}
                or (nodes > 1 and pool.partition != "cpu-efa")
            ):
                raise ManagedExecutionProfileError("Unsupported managed adaptive pool")
            return cls(site_id, {**resources, "pool": dict(resources["pool"])})
        if (
            resources.get("mode", "independent-frequency.v1")
            != "independent-frequency.v1"
        ):
            raise ManagedExecutionProfileError("Unsupported execution mode")
        resources = {k: v for k, v in resources.items() if k != "mode"}
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
                names.get(k, k): (
                    {
                        "threadsPerRank": v["threads_per_rank"],
                        "memoryMiBPerNode": v["memory_mib_per_node"],
                        "partition": v["partition"],
                    }
                    if k == "pool"
                    else v
                )
                for k, v in self.execution_resources.items()
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
