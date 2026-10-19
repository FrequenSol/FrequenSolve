"""Offline presentation of saved plans; never validation or compute authority."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

from .execution_plan import PLAN_VERSION, PlanError, allocation, positive, totals


def _text(value: Any) -> str:
    """Keep provider/user labels from becoming terminal control sequences."""
    return "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in str(value)
    )[:512]


def _snapshot(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(data, Mapping) or not isinstance(data.get("head"), Mapping):
        raise PlanError("Review requires a saved-plan response or snapshot")
    head = dict(data["head"])
    if (
        not isinstance(head.get("plan_id"), str)
        or not head["plan_id"]
        or type(head.get("revision")) is not int
        or not 0 <= head["revision"] <= 2**53 - 1
        or not isinstance(head.get("owner"), Mapping)
        or not isinstance(head["owner"].get("project_key"), str)
    ):
        raise PlanError("Saved-plan identity or revision is unavailable")
    plan = data.get("plan")
    if plan is None:
        return head, {}
    if (
        not isinstance(plan, Mapping)
        or plan.get("version") != PLAN_VERSION
        or plan.get("plan_id") != head["plan_id"]
        or not isinstance(plan.get("rows"), list)
        or len(plan["rows"]) > 96
    ):
        raise PlanError("Saved-plan contents are incompatible with this review client")
    return head, dict(plan)


def _request(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PlanError("Plan contains an invalid allocation")
    request = allocation(value)
    shape = request["layout"]
    count = totals(request)
    return {
        **shape,
        "mpi_ranks": count["mpi_ranks"],
        "total_cpus": count["cpus"],
        "memory_mib_per_node": request["memory_mib_per_node"],
        "total_memory_mib": count["memory_mib"],
        "wall_time_seconds": request["wall_time_seconds"],
    }


def _credits(value: Any) -> str:
    try:
        if (
            not isinstance(value, str)
            or value != value.strip()
            or not Decimal(value).is_finite()
            or Decimal(value) < 0
        ):
            raise ValueError()
    except (ValueError, InvalidOperation):
        raise PlanError("Plan has an invalid recorded Credit amount") from None
    return value


def review_plan(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached display model, not trusted validation evidence.

    Missing sizing/quotes remain unavailable. All arithmetic is in the plan's
    integer units; distributed memory is never treated as shared node memory.
    """
    head, plan = _snapshot(data)
    rows = []
    seen: set[int] = set()
    for source in plan.get("rows", []):
        if not isinstance(source, Mapping):
            raise PlanError("Plan contains an invalid frequency row")
        index = source.get("frequency_index")
        if type(index) is not int or not 0 <= index <= 2**53 - 1 or index in seen:
            raise PlanError("Plan has duplicate or invalid frequency identities")
        seen.add(index)
        frequency = source.get("frequency")
        try:
            if not isinstance(frequency, str) or not Decimal(frequency).is_finite():
                raise ValueError()
        except (ValueError, InvalidOperation):
            raise PlanError("Plan has an invalid frequency value") from None
        validation = source.get("validation", {})
        if not isinstance(validation, Mapping):
            raise PlanError("Plan has invalid validation information")
        status = validation.get("status", "unavailable")
        if status not in {"valid", "needs_validation", "invalid", "unavailable"}:
            raise PlanError("Plan has an unsupported validation status")
        minimum = validation.get("minimum_memory_mib_per_node")
        if minimum is not None:
            positive(minimum, "validated memory per node")
        overrides = source.get("overrides", {})
        if not isinstance(overrides, Mapping) or any(
            not isinstance(key, str) or type(value) is not int or value <= 0
            for key, value in overrides.items()
        ):
            raise PlanError("Plan has invalid override information")
        rows.append(
            {
                "frequency_index": index,
                "frequency": frequency,
                "requested": _request(source.get("requested")),
                "automatic": _request(source.get("automatic")),
                "overrides": deepcopy(dict(overrides)),
                "validation": status,
                "validation_reason": validation.get("reason"),
                "minimum_memory_mib_per_node": minimum,
                "selection_reason": source.get("reason"),
            }
        )
    quote = data.get("quote")
    recorded_quote = None
    if quote is not None:
        try:
            amount = _credits(quote["maximum_credits"])
            purpose = quote.get("purpose", "unavailable")
            if purpose not in {
                "planning",
                "validation",
                "execution",
                "automatic",
                "unavailable",
            }:
                raise ValueError()
            recorded_quote = {"purpose": purpose, "maximum_credits": amount}
            if "fingerprint" in quote:
                if (
                    not isinstance(quote["fingerprint"], str)
                    or not quote["fingerprint"]
                ):
                    raise ValueError()
                recorded_quote["fingerprint"] = quote["fingerprint"]
            for key in ("planning_maximum_credits", "execution_maximum_credits"):
                if key in quote:
                    recorded_quote[key] = _credits(quote[key])
        except (KeyError, TypeError, AttributeError, ValueError, InvalidOperation):
            raise PlanError("Plan has an invalid recorded Credit quote") from None
    financials = {}
    for name in ("planning", "execution"):
        section = data.get(name)
        if section is None:
            continue
        if not isinstance(section, Mapping) or not isinstance(
            section.get("phase"), str
        ):
            raise PlanError("Plan has invalid recorded Credit settlement information")
        detail = {"phase": section["phase"], "purpose": section.get("purpose", name)}
        for key in ("maximum_credits", "captured_credits"):
            if key in section:
                detail[key] = _credits(section[key])
        if "captured_credits" in detail and detail["phase"] != "settled":
            raise PlanError("Captured Credits require a recorded settled operation")
        financials[name] = detail
    limits = plan.get("limits", {})
    if not isinstance(limits, Mapping):
        raise PlanError("Plan has invalid concurrency limits")
    for name, value in limits.items():
        positive(value, str(name))
    return {
        "plan_id": head["plan_id"],
        "revision": head["revision"],
        "allocation_revision": plan.get("revision"),
        "status": head.get("status", "unavailable"),
        "updated_at": head.get("updated_at"),
        "historical": bool(data.get("historical"))
        or head.get("status") == "historical",
        "owner": deepcopy(dict(head["owner"])),
        "rows": rows,
        "limits": deepcopy(dict(limits)),
        "recorded_quote": recorded_quote,
        "financials": financials,
        "requires_compute": (
            data.get("requires_compute")
            if type(data.get("requires_compute")) is bool
            else None
        ),
        "execution_id": head.get("execution_id"),
        "context": deepcopy(plan.get("context", {})),
    }


def _memory(value: int) -> str:
    return f"{value} MiB ({Decimal(value) / 1024:g} GiB)"


def format_review(data: Mapping[str, Any], *, details: bool = False) -> str:
    """Format a saved response without fetching, authorizing or changing it."""
    view = review_plan(data)
    lines = [
        f"Plan {_text(view['plan_id'])} | revision {view['revision']} | {_text(view['status'])}"
    ]
    if view["historical"]:
        lines.append("Historical revision: read-only.")
    if view["updated_at"]:
        lines.append("Server last update: " + _text(view["updated_at"]))
    if (
        view["allocation_revision"] is not None
        and view["allocation_revision"] != view["revision"]
    ):
        lines.append(
            f"Allocation manifest revision: {_text(view['allocation_revision'])}"
        )
    headers = [
        "Index",
        "Frequency (Hz)",
        "Nodes",
        "MPI ranks",
        "CPUs/rank",
        "Total CPUs",
        "Memory/node",
        "Total memory",
        "Time limit (s)",
        "Validation",
    ]
    table = [headers]
    for row in view["rows"]:
        request = row["requested"]
        cells = [str(row["frequency_index"]), _text(row["frequency"])]
        cells += (
            [
                str(request[k])
                for k in ("nodes", "mpi_ranks", "cpus_per_rank", "total_cpus")
            ]
            + [
                _memory(request["memory_mib_per_node"]),
                _memory(request["total_memory_mib"]),
                str(request["wall_time_seconds"]),
            ]
            if request
            else ["unavailable"] * 7
        )
        table.append(cells + [row["validation"]])
    if view["rows"]:
        widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
        lines.extend(
            " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            for row in table
        )
    else:
        lines.append(
            "Frequency allocations: unavailable; planning may not have completed."
        )
    for row in view["rows"] if details else []:
        row_details = []
        if row["requested"]:
            row_details.append(
                f"{_text(row['requested']['partition'])} / {_text(row['requested']['node_type'])}"
            )
        automatic = row["automatic"]
        if automatic and automatic != row["requested"]:
            row_details.append(
                f"Automatic: {automatic['nodes']} nodes, {automatic['mpi_ranks']} ranks, "
                f"{automatic['total_cpus']} CPUs, {_memory(automatic['memory_mib_per_node'])}/node, "
                f"{automatic['wall_time_seconds']} s"
            )
        if row["selection_reason"]:
            row_details.append(_text(row["selection_reason"]))
        if row["overrides"]:
            row_details.append(
                "Overrides: "
                + ", ".join(
                    f"{_text(k)}={_text(v)}"
                    for k, v in sorted(row["overrides"].items())
                )
            )
        if row["minimum_memory_mib_per_node"] is not None:
            row_details.append(
                "Validated minimum/node: " + _memory(row["minimum_memory_mib_per_node"])
            )
        if row["validation_reason"]:
            row_details.append(_text(row["validation_reason"]))
        if row_details:
            lines.append(f"[{row['frequency_index']}] " + "; ".join(row_details))
    limits = view["limits"]
    lines.append(
        "Concurrency ceilings (not summed job requests): "
        + ", ".join(f"{_text(k)}={v}" for k, v in sorted(limits.items()))
        if limits
        else "Concurrency ceilings: unavailable."
    )
    lines.append(
        "Total memory is distributed across nodes, not available on each node. Runtime predictions and measured peak memory: unavailable."
    )
    quote = view["recorded_quote"]
    lines.append(
        f"Recorded {_text(quote['purpose'])} quote: up to {quote['maximum_credits']} Credits (not a charge or fresh authorization)."
        if quote
        else "Recorded Credit quote: unavailable (not zero)."
    )
    if quote:
        for key in ("planning_maximum_credits", "execution_maximum_credits"):
            if key in quote:
                lines.append(
                    f"  {key}: {quote[key]} Credits (part of the recorded automatic maximum)"
                )
    for name, detail in view["financials"].items():
        captured = detail.get("captured_credits")
        lines.append(
            f"Latest recorded {_text(detail['purpose'])} operation: {_text(detail['phase'])}; reported captured Credits: {captured if captured is not None else 'unavailable'}. This is not a lifetime total."
        )
    if view["requires_compute"] is not None:
        lines.append(
            "Server validation response requires compute: "
            + str(view["requires_compute"]).lower()
        )
    if view["execution_id"]:
        lines.append("Accepted execution: " + _text(view["execution_id"]))
    lines.append(
        "Informational snapshot only. Cloud rechecks validation, access, pricing and the exact revision before execution."
    )
    return "\n".join(lines)


def compare_plans(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare matching saved identities; never conclude that an edit is valid."""
    before, after = review_plan(previous), review_plan(current)
    if before["plan_id"] != after["plan_id"] or before["owner"] != after["owner"]:
        raise PlanError("Compare revisions of the same plan and exact owner only")
    if before["revision"] > after["revision"]:
        raise PlanError("The comparison snapshot is newer; reopen the current plan")
    left = {row["frequency_index"]: row for row in before["rows"]}
    right = {row["frequency_index"]: row for row in after["rows"]}
    changes = []
    for index in sorted(left.keys() | right.keys()):
        old, new = left.get(index), right.get(index)
        if old and new and old["frequency"] != new["frequency"]:
            raise PlanError(
                "Frequency identity changed; do not compare unrelated scientific inputs"
            )
        fields = {}
        for key in (
            "requested",
            "automatic",
            "overrides",
            "validation",
            "validation_reason",
            "minimum_memory_mib_per_node",
        ):
            a, z = old.get(key) if old else None, new.get(key) if new else None
            if a != z:
                fields[key] = {"before": a, "after": z}
        if fields or old is None or new is None:
            identity = new if new is not None else old
            assert identity is not None
            changes.append(
                {
                    "frequency_index": index,
                    "frequency": identity["frequency"],
                    "fields": fields,
                }
            )
    return {
        "from_revision": before["revision"],
        "to_revision": after["revision"],
        "changes": changes,
        "before": before,
        "after": after,
    }


def format_diff(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    comparison = compare_plans(previous, current)
    lines = [
        f"Revision {comparison['from_revision']} -> {comparison['to_revision']} (local comparison)"
    ]
    for row in comparison["changes"]:
        lines.append(f"[{row['frequency_index']}] {_text(row['frequency'])} Hz:")
        for field, values in row["fields"].items():
            if (
                field in {"requested", "automatic"}
                and values["before"]
                and values["after"]
            ):
                for key in sorted(values["before"].keys() | values["after"].keys()):
                    a, z = values["before"].get(key), values["after"].get(key)
                    if a != z:
                        lines.append(f"  {field}.{key}: {_text(a)} -> {_text(z)}")
            else:
                lines.append(
                    f"  {field}: {_text(values['before'])} -> {_text(values['after'])}"
                )
        fields = row["fields"]
        if "requested" in fields:
            a, z = fields["requested"]["before"], fields["requested"]["after"]
            layout_keys = (
                "partition",
                "node_type",
                "nodes",
                "ranks_per_node",
                "cpus_per_rank",
            )
            layout_changed = (
                a is None or z is None or any(a[k] != z[k] for k in layout_keys)
            )
            lines.append(
                "  Layout changed: matching evidence or paid sizing may be needed; ask plan.validate() without a Credit maximum first."
                if layout_changed
                else "  Reservation-only change: server structural/evidence checks are still required; ask plan.validate() without a Credit maximum first."
            )
    for key in (
        "status",
        "limits",
        "context",
        "recorded_quote",
        "financials",
        "requires_compute",
        "execution_id",
    ):
        a, z = comparison["before"][key], comparison["after"][key]
        if a != z:
            if isinstance(a, dict) and isinstance(z, dict):
                for field in sorted(a.keys() | z.keys()):
                    if a.get(field) != z.get(field):
                        lines.append(
                            f"{key}.{_text(field)}: {_text(a.get(field))} -> {_text(z.get(field))}"
                        )
            else:
                lines.append(f"{key}: {_text(a)} -> {_text(z)}")
    if not comparison["changes"]:
        lines.append("No frequency allocation, override or validation changes.")
    context_before, context_after = (
        comparison["before"]["context"],
        comparison["after"]["context"],
    )
    if context_before != context_after:
        lines.append(
            "Context changed: scientific inputs, artifacts, solver or runtime changes require replanning; capability/policy changes need server revalidation."
        )
    lines.append(
        "This comparison authorizes nothing. Server validation determines paid sizing needs; execution uses a fresh quote and exact revision."
    )
    return "\n".join(lines)


def recovery_guidance(data: Mapping[str, Any]) -> str:
    """Explain recovery without refreshing a handle with pending authorization."""
    head, _ = _snapshot(data)
    lines = [
        "Reopen the retained identity without starting work:",
        f"  plan = site.open_plan({head['plan_id']!r}, project_key={head['owner']['project_key']!r})",
        "  print(plan.summary())",
        "Keep a plan.snapshot() before edits; compare with plan.diff(snapshot). For a stale revision, reopen and review changes before reapplying an intended edit.",
        "After a timeout or lost mutation response, keep the original handle, exact request and idempotency key. Inspect the retained plan/execution before any retry; do not create a new key or submit a second simulation to guess whether the first was accepted.",
        "A fresh open_plan() reads only. An old handle's refresh()/wait() can honor a planning Credit maximum you supplied earlier; summary(), snapshot(), diff() and recovery() never call them.",
    ]
    if data.get("historical"):
        lines.append(
            "This historical revision is read-only; omit revision= when reopening the latest plan for edits."
        )
    if head.get("status") in {"failed", "canceled"}:
        lines.append(
            "Preserve failed/canceled work and its receipts. No solver retry is automatic; an intentional new run needs a new identity and authorization."
        )
    lines.append(
        "These review helpers change no server state. Edits, validation, authorization, execution and cancellation remain explicit workflow operations. Missing usage or quotes does not mean free or unaccepted work."
    )
    return "\n".join(lines)
