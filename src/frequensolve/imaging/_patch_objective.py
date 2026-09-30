# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Restrict independent waveform terms with their full-parent normalization."""

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from frequensolve.imaging._objective import ObjectiveState


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Parent objective {label} must be positive and finite")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Parent objective {label} must be positive and finite")
    return value


def _resolved(
    term: Mapping[str, Any], configs: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Require one replicated normalization of this configured parent term."""
    fields = (
        "receiver_group",
        "objective",
        "comparison",
        "weight",
        "reduction",
        "effective_weight_mass",
        "scale",
    )
    try:
        resolved = [{key: config[key] for key in fields} for config in configs]
    except KeyError as exc:
        raise ValueError("Parent objective lacks resolved normalization") from exc
    if not resolved or any(config != resolved[0] for config in resolved[1:]):
        raise ValueError("Parent objective normalization disagrees between ranks")
    config = resolved[0]
    if config["receiver_group"] != term["receiver_group"]:
        raise ValueError("Parent objective receiver group changed")
    for key in ("objective", "comparison"):
        if any(config[key].get(k) != v for k, v in term[key].items()):
            raise ValueError(f"Parent objective {key} changed")
    if (
        config["weight"] != term["weight"]
        or config["reduction"] != term["normalization"]["reduction"]
        or config["scale"]["policy"] != term["normalization"]["scale"]["kind"]
    ):
        raise ValueError("Parent objective weight or normalization policy changed")
    if config["reduction"] == "sum" and config["effective_weight_mass"] != 1:
        raise ValueError("Parent sum objective must have unit weight mass")
    return config


def validate_patch_misfit(payload: Mapping[str, Any]) -> None:
    """Reject transformations whose restricted actions are not implemented."""
    hooks = payload.get("preprocess", {})
    if hooks.get("include_defaults") or hooks.get("hooks"):
        raise ValueError("Patch objectives do not yet support preprocessing")
    for group in payload["receiver_groups"]:
        if group.get("preprocess") or group.get("projection", {"kind": "identity"}) != {
            "kind": "identity"
        }:
            raise ValueError("Patch objectives do not yet support receiver transforms")
    for term in payload["objective_terms"]:
        if term["comparison"].get("kind") != "waveform" or term.get("preprocess"):
            raise ValueError("Patch objectives require untransformed waveform terms")


def restrict_patch_misfit(
    payload: Mapping[str, Any],
    parent_state: ObjectiveState | str | Path,
    *,
    receiver_groups: Iterable[str],
) -> dict[str, Any]:
    """Freeze parent scales and reduction before selecting child receiver rows.

    The caller retains the original observed trace references and acquisition
    IDs. Sparse child surveys select the actual rows. An explicit scale and a
    sum reduction with the parent's weight/mass reproduce its row contributions
    for values, scores, JVPs and normal actions. Frequency weights remain on the
    enclosing problem. Transforms that need separate restriction are rejected.
    """
    state = ObjectiveState(
        parent_state.path if isinstance(parent_state, ObjectiveState) else parent_state
    )
    result = deepcopy(dict(payload))
    validate_patch_misfit(result)
    selected = tuple(receiver_groups)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Patch objective requires unique nonempty receiver groups")
    groups = {group["name"]: group for group in result["receiver_groups"]}
    if not set(selected) <= groups.keys():
        raise ValueError("Patch objective names an unknown parent receiver group")
    terms = []
    for term in result["objective_terms"]:
        if term["receiver_group"] not in selected:
            continue
        config = _resolved(term, state.term_configs.get(term["id"], ()))
        scale = config["scale"]
        components, values, units = (
            scale["components"],
            scale["values"],
            scale["units"],
        )
        if (
            not components
            or len(set(components)) != len(components)
            or len(components) != len(values)
            or len(components) != len(units)
            or any(not isinstance(unit, str) or not unit.strip() for unit in units)
        ):
            raise ValueError("Parent objective has incomplete component scales")
        term["normalization"] = {
            "scale": {
                "kind": "explicit",
                "components": {
                    component: {
                        "value": _positive(value, "component scale"),
                        "units": unit,
                    }
                    for component, value, unit in zip(components, values, units)
                },
            },
            "reduction": "sum",
        }
        term["weight"] = _positive(config["weight"], "weight") / _positive(
            config["effective_weight_mass"], "weight mass"
        )
        _positive(term["weight"], "restricted weight")
        terms.append(term)
    if not terms:
        raise ValueError("Patch has no retained objective terms")
    result["objective_terms"] = terms
    result["receiver_groups"] = [groups[name] for name in selected]
    return result


def patch_objective_keys(
    saved_state: ObjectiveState | str | Path, acquisition: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Map saved sparse trace keys to original point-receiver observation keys.

    Native sparse keys name trace IDs, while a dense parent's keys name receiver
    IDs. Inline patch surveys carry the exact bridge. Return canonical keys in
    each term's existing row order; never change native persisted coordinates.
    """
    state = ObjectiveState(
        saved_state.path if isinstance(saved_state, ObjectiveState) else saved_state
    )
    groups = {group["name"]: group for group in acquisition["receiver_groups"]}
    surveys = {survey["name"]: survey for survey in acquisition.get("surveys", [])}
    result = {}
    for name, term in state.terms.items():
        group = groups[term["receiver_group"]]
        keys = term["keys"]
        sampling = group.get("sampling", {"_type": "Dense"})
        if sampling["_type"] == "Dense":
            canonical = keys.copy()
        elif sampling["_type"] == "Sparse":
            survey = surveys[sampling["survey"]]
            if survey["_type"] != "Sparse" or "traces" not in survey:
                raise ValueError("Patch objective rows require an inline point survey")
            mapping = {}
            trace_ids = set()
            for trace in survey["traces"]:
                if not trace.get("active", True):
                    continue
                receiver = trace["receiver_id"]
                if (
                    trace["point_first"] != receiver
                    or trace["point_last"] != receiver
                    or trace["receiver_position_id"] != receiver
                ):
                    raise ValueError(
                        "Patch objective rows require original point receivers"
                    )
                trace_id = trace["trace_id"]
                if trace_id in trace_ids:
                    raise ValueError("Patch objective survey repeats a trace ID")
                trace_ids.add(trace_id)
                native = (trace["source_id"], trace_id, trace["component"])
                physical = (trace["source_id"], receiver, trace["component"])
                mapping[native] = physical
            if set(map(tuple, keys)) != mapping.keys():
                raise ValueError("Patch objective rows disagree with the active survey")
            canonical = np.asarray(
                [mapping[tuple(key)] for key in keys], dtype=int
            ).reshape(-1, 3)
        else:
            raise ValueError(
                "Patch objective rows require dense or sparse point sampling"
            )
        if len(set(map(tuple, canonical))) != len(canonical):
            raise ValueError("Patch objective repeats an original observation row")
        result[name] = canonical
    return result
