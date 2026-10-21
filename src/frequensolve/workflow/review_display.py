"""Deterministic customer review text; never interprets or authorizes a proposal."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .storage import canonical

MAX_DISPLAY_BYTES = 128_000
_LABELS = {
    "cpus_per_rank": "CPUs per rank",
    "memory_mib_per_node": "Memory per node (MiB)",
    "wall_time_seconds": "Time limit (seconds)",
    "ranks_per_node": "MPI ranks per node",
    "node_type": "Node type",
    "execution_site_id": "Cluster",
    "minimum_credits": "Minimum charge (Credits)",
    "maximum_credits": "Maximum authorized charge (Credits)",
    "scientific_inputs": "Scientific settings",
    "compute_review": "Sizing or validation allocations",
    "requested": "Requested resources",
    "automatic": "Automatic sizing",
    "input_digest": "Input fingerprint",
    "fingerprint": "Fingerprint",
    "revision": "Plan revision",
    "vp": "Acoustic velocity (vp)",
    "rho": "Density (rho)",
}


def escaped(value: str) -> str:
    """Keep all label characters visible as data, including control characters."""
    value = "".join(
        (
            json.dumps(character, ensure_ascii=True)[1:-1]
            if unicodedata.category(character) in ("Cc", "Cf", "Cs", "Zl", "Zp")
            else character
        )
        for character in value
    )
    for character in "\\`*_{}[]<>()|#~":
        value = value.replace(character, "\\" + character)
    return value


def scalar(value: Any) -> str:
    if value is None:
        return "Not provided (null)"
    if value is True:
        return "Yes (true)"
    if value is False:
        return "No (false)"
    if value == {}:
        return "Empty object"
    if value == []:
        return "Empty list"
    if isinstance(value, (str, int, float)):
        return escaped(str(value))
    raise ValueError("Unsupported proposal value")


def leaves(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if len(path) > 24:
        raise ValueError("Proposal details exceed the display depth limit")
    if isinstance(value, dict) and value:
        result = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("Proposal keys must be text")
            result.extend(leaves(value[key], path + (key,)))
    elif isinstance(value, list) and value:
        result = []
        for index, child in enumerate(value):
            result.extend(leaves(child, path + (str(index),)))
    else:
        return [(path, value)]
    if len(result) > 4096:
        raise ValueError("Proposal exceeds the display field limit")
    return result


def pointer(path: tuple[str, ...]) -> str:
    return "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path)


def table(headers: tuple[str, ...], rows: list[list[Any]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(scalar(cell) for cell in row) + " |" for row in rows]
    )


def allocations(snapshot: dict, stage: str) -> str:
    if stage == "execution":
        rows = (snapshot.get("plan") or {}).get("rows", [])
        title = "Resources for this run"
        values = [
            (
                row.get("frequency"),
                row.get("requested", {}),
                row.get("validation", {}).get("status"),
            )
            for row in rows
        ]
        headers = (
            "Frequency (Hz)",
            "Node type / partition",
            "Nodes",
            "Ranks/node",
            "CPUs/rank",
            "MiB/node",
            "Limit (s)",
            "Validation",
        )
    else:
        rows = (snapshot.get("compute_review") or {}).get("planning_evaluations", [])
        title = "Allocations for " + (
            "automatic sizing" if stage == "planning" else "paid validation"
        )
        values = [(index + 1, row, "Not supplied") for index, row in enumerate(rows)]
        headers = (
            "Evaluation",
            "Node type / partition",
            "Nodes",
            "Ranks/node",
            "CPUs/rank",
            "MiB/node",
            "Limit (s)",
            "Validation",
        )
    output = []
    for label, allocation, validation in values:
        layout = allocation.get("layout", {})
        output.append(
            [
                label,
                str(layout.get("node_type", "Unavailable"))
                + " / "
                + str(layout.get("partition", "Unavailable")),
                layout.get("nodes"),
                layout.get("ranks_per_node"),
                layout.get("cpus_per_rank"),
                allocation.get("memory_mib_per_node"),
                allocation.get("wall_time_seconds"),
                validation,
            ]
        )
    return (
        "### "
        + title
        + "\n\n"
        + (
            table(headers, output)
            if output
            else "The compact allocation table is unavailable. Inspect the exact resource settings below."
        )
    )


def differences(snapshot: dict) -> str:
    changed = []
    for row in (snapshot.get("plan") or {}).get("rows", []):
        if not isinstance(row.get("automatic"), dict) or not isinstance(
            row.get("requested"), dict
        ):
            continue
        before, after = dict(leaves(row["automatic"])), dict(leaves(row["requested"]))
        for path in sorted(set(before) | set(after)):
            if (
                path not in before
                or path not in after
                or canonical(before[path]) != canonical(after[path])
            ):
                changed.append(
                    [
                        row.get("frequency"),
                        " / ".join(_LABELS.get(p, p.replace("_", " ")) for p in path),
                        before.get(path),
                        after.get(path),
                    ]
                )
    return "### Changes from automatic sizing\n\n" + (
        table(("Frequency (Hz)", "Setting", "Automatic", "Requested"), changed)
        if changed
        else "No differences were found in the supplied automatic/requested resource pairs. Exact overrides and validation remain in the details below."
    )


def render(snapshot: dict, fingerprint: str, *, paid: bool) -> str:
    """Represent every field, including unknown fields, without a raw JSON dump."""
    if len(canonical(snapshot)) > 64000 or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", fingerprint
    ):
        raise ValueError("A bounded exact proposal and fingerprint are required")
    quote = snapshot.get("quote", {})
    stage = snapshot.get("stage", quote.get("purpose"))
    if stage not in ("planning", "validation", "execution"):
        raise ValueError("Unknown paid stage")
    maximum = quote.get("maximum_credits")
    if not isinstance(maximum, str) or not re.fullmatch(
        r"(0|[1-9]\d{0,17})(\.\d{1,6})?", maximum
    ):
        raise ValueError("An exact Credit maximum is required")
    science = snapshot.get("scientific_inputs", {})
    plan = snapshot.get("plan") or {}
    compute = snapshot.get("compute_review") or {}
    target = (plan.get("context", {}) if stage == "execution" else compute).get(
        "execution_site_id"
    )
    meaning = {
        "planning": "Size this study. This approval does not run the solver campaign. Review execution separately after sizing.",
        "validation": "Validate these resource changes. This approval does not run the solver campaign. Review execution separately.",
        "execution": "Run this study using the exact saved resource revision below.",
    }[stage]
    lines = [
        f"PAID {stage.upper()} APPROVAL" if paid else "NO-SPEND REVIEW REHEARSAL",
        (
            "This can start paid compute."
            if paid
            else "This prompt cannot authorize or launch paid work. Confirmation records only a local rehearsal response."
        ),
        "### Your decision",
        (
            meaning
            if paid
            else "Practice reviewing the following "
            + stage
            + " proposal without approving it."
        ),
        f"**Maximum: {maximum} Credits** · Plan revision: {scalar(snapshot.get('revision'))} · Cluster: {scalar(target)}",
        (
            (
                "Choose Approve this stage only if you accept this proposal. The next prompt asks for exactly "
                + maximum
                + " Credits; enter the number only. Choose Request changes to describe an edit, or Cancel to stop."
            )
            if paid
            else "Choose Confirm rehearsal, Request changes, or Cancel. A note is requested only when you choose Request changes."
        ),
        "### Scientific assumptions",
        "Physics: "
        + scalar(science.get("physics"))
        + " · Dimensions: "
        + scalar(science.get("dimension"))
        + " · Precision: "
        + scalar(science.get("precision"))
        + ". All scientific settings, units, coordinates and solver assumptions appear in the exact details below. This proposal does not establish scientific accuracy or convergence.",
        allocations(snapshot, stage),
    ]
    if stage == "execution":
        lines.append(differences(snapshot))
    pricing = quote.get("pricing") or {}
    policy = pricing.get("failure_policy")
    failure_text = (
        "Allocated failed or canceled sizing/validation work may consume Credits."
        if stage != "execution"
        else (
            "The quoted policy waives charges when the execution outcome is FAILED. Cancellation has its own quoted policy."
            if policy == "WAIVE_FAILED"
            else (
                "Failed execution is billed for measured allocations and may require billing review. Canceled work can also consume Credits."
                if policy == "BILL_MEASURED_REVIEW"
                else "Failure and cancellation charging follow the quoted policies below; unavailable terms remain unknown."
            )
        )
    )
    policy_labels = {
        "allocated_resources": "Allocated resources",
        "BILL_MEASURED_REVIEW": "Measured allocation charges, with failure review",
        "WAIVE_FAILED": "Waive failed execution",
        "measured_allocations_with_minimum": "Measured allocations, subject to the minimum",
    }

    def policy_label(value: Any) -> Any:
        return policy_labels.get(value, value) if isinstance(value, str) else value

    lines.extend(
        [
            "### Credits and cancellation",
            failure_text
            + " Successful scientific results are not a condition of charging. Cancellation does not imply immediate termination or refund.",
            table(
                ("Quoted term", "Value"),
                [
                    ["Maximum (Credits)", maximum],
                    ["Minimum (Credits)", pricing.get("minimum_credits")],
                    [
                        "Charge basis",
                        policy_label(pricing.get("time_basis")),
                    ],
                    ["Failure policy", policy_label(policy)],
                    [
                        "Cancellation policy",
                        policy_label(pricing.get("cancellation_policy")),
                    ],
                ],
            ),
            "Missing quote terms remain unavailable; they are not promises of a zero charge or refund.",
            "### Exact proposal details",
            "Every field from the authoritative proposal is shown below. Paths identify exact saved settings; list indices start at zero. Null and empty values remain explicit. Unknown fields are included rather than silently dropped.",
        ]
    )
    # Sections keep scientific settings, financial terms and identity evidence
    # together. The machine snapshot and fingerprint remain untouched.
    for key in sorted(
        snapshot,
        key=lambda k: (
            {"scientific_inputs": 0, "plan": 1, "compute_review": 2, "quote": 3}.get(
                k, 4
            ),
            k,
        ),
    ):
        heading = _LABELS.get(key, key.replace("_", " ").title())
        if key == "compute_review" and stage == "execution":
            heading = "Stored sizing context (not an additional approval)"
        lines.append("#### " + escaped(heading))
        rows = [
            [
                pointer(path),
                json.dumps(
                    value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ),
            ]
            for path, value in leaves(snapshot[key], (key,))
        ]
        lines.append(table(("Exact setting path", "Proposed value"), rows))
    lines.extend(
        [
            "Cloud proposal fingerprint: " + fingerprint,
            "The client reports your response; FrequenSol does not independently verify that a human answered. "
            + (
                "Only the backend's accepted receipt establishes paid authorization."
                if paid
                else "This rehearsal grants no paid authority or service authorization."
            ),
        ]
    )
    result = "\n\n".join(lines)
    if len(result.encode()) > MAX_DISPLAY_BYTES:
        raise ValueError(
            "The complete proposal exceeds the readable review limit; use Cloud review"
        )
    return result
