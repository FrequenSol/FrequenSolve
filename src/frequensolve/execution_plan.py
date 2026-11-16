"""Provider-neutral, versioned resource plans for independent frequency jobs.

Sizing observations are produced by trusted native sizing operations. This module
checks their arithmetic; it does not establish their provenance or authorization.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

PLAN_VERSION = "frequency-execution-plan.v1"
SIZING_VERSION = "frequency-layout-sizing.v1"
POLICY_VERSION = "memory-fit-first.v1"
MIB = 1024 * 1024
MAX_INTEGER = 2**53 - 1


class PlanError(ValueError):
    """A resource plan is invalid or lacks matching sizing evidence."""


def positive(value: Any, name: str, maximum: int = MAX_INTEGER) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise PlanError(f"{name} must be a positive integer <= {maximum}")
    return value


def digest(value: Any) -> str:
    """Hash JSON with cross-language canonical ordering and integer arithmetic."""

    def check(item: Any) -> None:
        if isinstance(item, float):
            raise PlanError(
                "Plan digests use integer units and decimal frequency strings"
            )
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise PlanError("Plan JSON keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def layout(value: Mapping[str, Any]) -> dict[str, Any]:
    names = {"partition", "node_type", "nodes", "ranks_per_node", "cpus_per_rank"}
    if not isinstance(value, Mapping) or set(value) != names:
        raise PlanError(
            "Layout needs partition, node_type, nodes, ranks_per_node, cpus_per_rank"
        )
    result = dict(value)
    for key in ("partition", "node_type"):
        if (
            not isinstance(result[key], str)
            or not result[key]
            or len(result[key]) > 128
        ):
            raise PlanError(f"Invalid {key}")
    for key in ("nodes", "ranks_per_node", "cpus_per_rank"):
        positive(result[key], key)
    positive(
        result["nodes"] * result["ranks_per_node"] * result["cpus_per_rank"],
        "total_cpus",
    )
    return result


def allocation(value: Mapping[str, Any]) -> dict[str, Any]:
    names = {"layout", "memory_mib_per_node", "wall_time_seconds"}
    if not isinstance(value, Mapping) or set(value) != names:
        raise PlanError(
            "Allocation needs layout, memory_mib_per_node, wall_time_seconds"
        )
    result = dict(value)
    result["layout"] = layout(result["layout"])
    positive(result["memory_mib_per_node"], "memory_mib_per_node")
    positive(result["wall_time_seconds"], "wall_time_seconds")
    positive(
        result["memory_mib_per_node"] * result["layout"]["nodes"], "total_memory_mib"
    )
    return result


def totals(request: Mapping[str, Any]) -> dict[str, int]:
    request = allocation(request)
    shape = request["layout"]
    ranks = shape["nodes"] * shape["ranks_per_node"]
    return {
        "mpi_ranks": ranks,
        "cpus": ranks * shape["cpus_per_rank"],
        "memory_mib": shape["nodes"] * request["memory_mib_per_node"],
    }


def memory_request(
    observation: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    worker_overhead_mib: int,
    safety_numerator: int = 1,
    safety_denominator: int = 1,
    minimum_memory_mib: int = 0,
) -> int:
    """Bound any uniform rank placement using the largest rank-local estimates.

    Native estimates already contain their native safety factor. The rational
    multiplier here is a separately recorded runtime calibration; default one
    avoids applying native safety twice. Never divide an aggregate by ranks.
    """
    shape = layout(shape)
    if (
        observation.get("version") != SIZING_VERSION
        or observation.get("layout") != shape
        or observation.get("complete") is not True
        or observation.get("memory_scope") != "rank-local-resident-estimate"
        or observation.get("native_safety_applied") is not True
    ):
        raise PlanError("Complete layout-specific native sizing evidence is required")
    rank_bytes = observation.get("rank_memory_bytes")
    if (
        not isinstance(rank_bytes, list)
        or len(rank_bytes) != shape["nodes"] * shape["ranks_per_node"]
    ):
        raise PlanError("Sizing rank count differs from the layout")
    for value in rank_bytes:
        positive(value, "rank_memory_bytes")
    positive(sum(rank_bytes), "total_rank_memory_bytes")
    if type(worker_overhead_mib) is not int or worker_overhead_mib < 0:
        raise PlanError("Invalid worker overhead")
    positive(safety_numerator, "safety_numerator")
    positive(safety_denominator, "safety_denominator")
    if safety_numerator < safety_denominator:
        raise PlanError("Runtime calibration cannot reduce native estimates")
    if (
        type(minimum_memory_mib) is not int
        or not 0 <= minimum_memory_mib <= MAX_INTEGER
    ):
        raise PlanError("Invalid minimum memory reservation")
    worst_node = sum(sorted(rank_bytes, reverse=True)[: shape["ranks_per_node"]])
    denominator = safety_denominator * MIB
    return positive(
        max(
            minimum_memory_mib,
            (worst_node * safety_numerator + denominator - 1) // denominator
            + worker_overhead_mib,
        ),
        "requested_memory_mib_per_node",
    )


def validate_request(
    request: Mapping[str, Any],
    observation: Mapping[str, Any],
    hardware: Mapping[str, Any],
    *,
    worker_overhead_mib: int,
    max_nodes: int,
    max_cpus: int,
    max_wall_time_seconds: int,
    safety_numerator: int = 1,
    safety_denominator: int = 1,
    minimum_memory_mib: int = 0,
) -> dict[str, Any]:
    request = allocation(request)
    shape = request["layout"]
    count = totals(request)
    if request["wall_time_seconds"] < 60 or request["wall_time_seconds"] % 60:
        raise PlanError("Slurm time limits must use whole-minute increments")
    for name, value in (
        ("max_nodes", max_nodes),
        ("max_cpus", max_cpus),
        ("max_wall_time_seconds", max_wall_time_seconds),
    ):
        positive(value, name)
    if (
        hardware.get("partition") != shape["partition"]
        or hardware.get("node_type") != shape["node_type"]
        or shape not in hardware.get("qualified_layouts", [])
    ):
        raise PlanError("Layout is not qualified for the selected hardware")
    for key in ("schedulable_cpus", "schedulable_memory_mib", "max_nodes"):
        positive(hardware.get(key), key)
    if (
        shape["nodes"] > min(max_nodes, hardware["max_nodes"])
        or count["cpus"] > max_cpus
        or shape["ranks_per_node"] * shape["cpus_per_rank"]
        > hardware["schedulable_cpus"]
        or request["memory_mib_per_node"] > hardware["schedulable_memory_mib"]
        or request["wall_time_seconds"] > max_wall_time_seconds
    ):
        raise PlanError("Allocation exceeds hardware or submission limits")
    if shape["nodes"] > 1 and hardware.get("network_fabric") != "efa":
        raise PlanError("Distributed layout requires qualified EFA networking")
    minimum = memory_request(
        observation,
        shape,
        worker_overhead_mib=worker_overhead_mib,
        safety_numerator=safety_numerator,
        safety_denominator=safety_denominator,
        minimum_memory_mib=minimum_memory_mib,
    )
    if request["memory_mib_per_node"] < minimum:
        raise PlanError(
            f"Memory reservation is below the validated requirement ({minimum} MiB/node)"
        )
    return {
        "status": "valid",
        "minimum_memory_mib_per_node": minimum,
        "sizing_digest": digest(observation),
        "allocation_digest": digest(request),
    }


def choose_allocation(
    observations: list[dict[str, Any]],
    hardware: list[dict[str, Any]],
    *,
    max_nodes: int,
    max_cpus: int,
    wall_time_seconds: int,
    worker_overhead_mib: int,
    memory_policy: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Choose deterministically from trusted observations of qualified layouts."""
    calibration = {
        "safety_numerator": 1,
        "safety_denominator": 1,
        "minimum_memory_mib": 0,
    }
    if memory_policy is not None:
        if (
            set(memory_policy)
            != {
                "calibration_numerator",
                "calibration_denominator",
                "worker_overhead_mib",
                "minimum_memory_mib",
            }
            or memory_policy["worker_overhead_mib"] != worker_overhead_mib
        ):
            raise PlanError("Memory calibration differs from the frozen policy")
        calibration = {
            "safety_numerator": memory_policy["calibration_numerator"],
            "safety_denominator": memory_policy["calibration_denominator"],
            "minimum_memory_mib": memory_policy["minimum_memory_mib"],
        }
    candidates = []
    for observation in observations:
        shape = layout(observation["layout"])
        for node in hardware:
            if (
                node.get("node_type") != shape["node_type"]
                or node.get("partition") != shape["partition"]
            ):
                continue
            if shape not in node.get("recommended_layouts", []):
                continue
            try:
                memory = memory_request(
                    observation,
                    shape,
                    worker_overhead_mib=worker_overhead_mib,
                    **calibration,
                )
                request = {
                    "layout": shape,
                    "memory_mib_per_node": memory,
                    "wall_time_seconds": wall_time_seconds,
                }
                validation = validate_request(
                    request,
                    observation,
                    node,
                    worker_overhead_mib=worker_overhead_mib,
                    max_nodes=max_nodes,
                    max_cpus=max_cpus,
                    max_wall_time_seconds=wall_time_seconds,
                    **calibration,
                )
                key = (
                    shape["nodes"],
                    shape["nodes"] * shape["ranks_per_node"],
                    totals(request)["cpus"],
                    shape["node_type"],
                    digest(request),
                )
                candidates.append((key, request, validation))
            except PlanError:
                continue
    if not candidates:
        raise PlanError(
            "No evaluated, qualified layout fits this frequency; extend planning or change limits"
        )
    _, request, validation = min(candidates, key=lambda candidate: candidate[0])
    return {
        "requested": request,
        "validation": validation,
        "reason": "Fewest feasible nodes, then ranks, within the qualified CPU profile",
        "estimated_runtime_seconds": None,
    }


def revise_rows(
    rows: list[dict[str, Any]], edits: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Apply allocation-only edits; authority and scientific metadata are immutable."""
    result = deepcopy(rows)
    if set(edits) - {row["frequency_index"] for row in rows}:
        raise PlanError("Edit refers to an unknown frequency")
    allowed = {
        "nodes",
        "mpi_ranks",
        "cpus_per_rank",
        "memory_mib_per_node",
        "wall_time_seconds",
        "reset",
    }
    for row in result:
        edit = edits.get(row["frequency_index"])
        if edit is None:
            continue
        if set(edit) - allowed or not edit:
            raise PlanError("Unsupported or empty plan edit")
        if "reset" in edit:
            if edit != {"reset": True}:
                raise PlanError("Reset cannot be combined with overrides")
            row["requested"] = deepcopy(row["automatic"])
            row["overrides"] = {}
        else:
            if row["requested"] is None:
                raise PlanError(
                    "Extend planning before editing a frequency without sizing evidence"
                )
            request = deepcopy(row["requested"])
            shape = request["layout"]
            nodes = edit.get("nodes", shape["nodes"])
            ranks = edit.get("mpi_ranks", shape["nodes"] * shape["ranks_per_node"])
            positive(nodes, "nodes")
            positive(ranks, "mpi_ranks")
            if ranks % nodes:
                raise PlanError("MPI ranks must divide uniformly across nodes")
            shape.update(
                nodes=nodes,
                ranks_per_node=ranks // nodes,
                cpus_per_rank=edit.get("cpus_per_rank", shape["cpus_per_rank"]),
            )
            for key in ("memory_mib_per_node", "wall_time_seconds"):
                if key in edit:
                    request[key] = edit[key]
            row["requested"] = allocation(request)
            row["overrides"] = {**row.get("overrides", {}), **edit}
        row["validation"] = {"status": "needs_validation"}
    return result
