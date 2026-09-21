"""Closed profile contract for FrequenSol-managed Cloud execution backends."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

MANAGED_EXECUTION_PROFILE_FIELDS = frozenset(
    {
        "execution_backend",
        "compute_mode",
        "slurm_partition",
        "slurm_nodes",
        "slurm_ranks_per_node",
        "slurm_wall_time",
    }
)

_WALL_TIME = re.compile(
    r"^(?P<days>\d{2})-(?P<hours>[0-2]\d):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)$"
)
_BATCH_MODES = frozenset({"auto", "spot_only", "on_demand_only"})


class ManagedExecutionProfileError(ValueError):
    """A named Cloud profile has an unsafe or unsupported execution shape."""


@dataclass(frozen=True)
class ManagedExecutionProfile:
    """Validated execution settings sourced only from one named site profile."""

    backend: str = "batch"
    compute_mode: str | None = None
    slurm_partition: str | None = None
    slurm_nodes: int | None = None
    slurm_ranks_per_node: int | None = None
    slurm_wall_time: str | None = None
    slurm_wall_time_seconds: int | None = None
    emit_batch_fields: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ManagedExecutionProfile":
        backend = values.get("execution_backend", "batch")
        if not isinstance(backend, str) or backend not in {"batch", "slurm"}:
            raise ManagedExecutionProfileError(
                "execution_backend must be exactly 'batch' or 'slurm'"
            )

        compute_mode = values.get("compute_mode")
        if compute_mode is not None:
            if (
                not isinstance(compute_mode, str)
                or compute_mode.lower() not in _BATCH_MODES
            ):
                raise ManagedExecutionProfileError(
                    "compute_mode must be auto, spot_only, or on_demand_only"
                )
            compute_mode = compute_mode.lower()

        slurm_values = {
            name: values.get(name)
            for name in (
                "slurm_partition",
                "slurm_nodes",
                "slurm_ranks_per_node",
                "slurm_wall_time",
            )
        }
        supplied_slurm = [
            name for name, value in slurm_values.items() if value is not None
        ]

        if backend == "batch":
            if supplied_slurm:
                raise ManagedExecutionProfileError(
                    "Batch profiles cannot set Slurm fields: "
                    + ", ".join(supplied_slurm)
                )
            return cls(
                backend="batch",
                compute_mode=compute_mode or "auto",
                emit_batch_fields=(
                    "execution_backend" in values or "compute_mode" in values
                ),
            )

        if compute_mode is not None:
            raise ManagedExecutionProfileError("Slurm profiles cannot set compute_mode")

        missing = [name for name, value in slurm_values.items() if value is None]
        if missing:
            raise ManagedExecutionProfileError(
                "Slurm profiles must set: " + ", ".join(missing)
            )
        partition = slurm_values["slurm_partition"]
        if not isinstance(partition, str) or partition not in {
            "cpu-single",
            "cpu-efa",
        }:
            raise ManagedExecutionProfileError(
                "unsupported Slurm resource shape: slurm_partition must be "
                "'cpu-single' or 'cpu-efa'"
            )
        nodes = _bounded_integer(slurm_values["slurm_nodes"], "slurm_nodes", 1, 2)
        ranks = _bounded_integer(
            slurm_values["slurm_ranks_per_node"], "slurm_ranks_per_node", 1, 4
        )
        if partition == "cpu-single" and (nodes != 1 or ranks != 1):
            raise ManagedExecutionProfileError(
                "unsupported Slurm resource shape: cpu-single requires exactly "
                "one node and one rank"
            )
        if partition == "cpu-efa" and nodes < 2:
            raise ManagedExecutionProfileError(
                "unsupported Slurm resource shape: cpu-efa requires an explicitly "
                "distributed plan with at least two nodes"
            )
        if nodes * ranks > 8:  # Defensive if the per-field bounds change later.
            raise ManagedExecutionProfileError(
                "unsupported Slurm resource shape: total MPI ranks cannot exceed 8"
            )
        wall_time = slurm_values["slurm_wall_time"]
        wall_seconds = _wall_time_seconds(wall_time)
        if wall_seconds < 60 or wall_seconds > 2 * 60 * 60:
            raise ManagedExecutionProfileError(
                "unsupported Slurm resource shape: wall time must be between one minute and two hours"
            )
        return cls(
            backend="slurm",
            slurm_partition=partition,
            slurm_nodes=nodes,
            slurm_ranks_per_node=ranks,
            slurm_wall_time=wall_time,
            slurm_wall_time_seconds=wall_seconds,
        )

    def graphql_arguments(self) -> dict[str, Any]:
        """Return the optional submitJob arguments for this profile."""

        if self.backend == "batch":
            if not self.emit_batch_fields:
                return {}
            return {
                "execution_backend": "batch",
                "compute_mode": self.compute_mode,
            }
        return {
            "execution_backend": "slurm",
            "slurm_partition": self.slurm_partition,
            "slurm_nodes": self.slurm_nodes,
            "slurm_ranks_per_node": self.slurm_ranks_per_node,
            "slurm_wall_time_seconds": self.slurm_wall_time_seconds,
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


def _wall_time_seconds(value: Any) -> int:
    if not isinstance(value, str):
        raise ManagedExecutionProfileError(
            "slurm_wall_time must use DD-HH:MM:SS format"
        )
    match = _WALL_TIME.fullmatch(value)
    if match is None:
        raise ManagedExecutionProfileError(
            "slurm_wall_time must use DD-HH:MM:SS format"
        )
    parts = {name: int(component) for name, component in match.groupdict().items()}
    if parts["hours"] > 23:
        raise ManagedExecutionProfileError(
            "slurm_wall_time must use DD-HH:MM:SS format"
        )
    return (
        parts["days"] * 86_400
        + parts["hours"] * 3_600
        + parts["minutes"] * 60
        + parts["seconds"]
    )
