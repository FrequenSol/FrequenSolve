"""Exact local notebook context; never a service or current-environment attestation."""

from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy

from .storage import canonical

VERSION = "study-notebook-binding.v1"
FIELDS = {
    "version",
    "project_id",
    "project_key",
    "plan_id",
    "plan_revision",
    "execution_id",
    "input_digest",
    "plan_digest",
    "result_digest",
    "frequency_indices",
    "frequency_coordinates",
    "study_digest",
}


def context_binding(
    view: dict, study: dict, indices: list[int], *, result_digest: str | None
) -> dict:
    head = view["head"]
    owner = head["owner"]
    if (
        not indices
        or len(indices) > 2
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or i not in (0, 1) for i in indices)
    ):
        raise ValueError("Choose distinct supported notebook frequencies")
    value = {
        "version": VERSION,
        "project_id": owner["project_id"],
        "project_key": owner["project_key"],
        "plan_id": head["plan_id"],
        "plan_revision": head["revision"],
        "execution_id": head["execution_id"],
        "input_digest": view["input_digest"],
        "plan_digest": view["plan"]["digest"],
        "result_digest": result_digest,
        "frequency_indices": list(indices),
        "frequency_coordinates": [
            view["scientific_inputs"]["frequencies"][i] for i in indices
        ],
        "study_digest": "sha256:"
        + hashlib.sha256(
            canonical({k: study[k] for k in ("title", "question", "success_criterion")})
        ).hexdigest(),
    }
    validate(value, allow_unknown_result=True)
    return deepcopy(value)


def validate(value: dict, *, allow_unknown_result: bool = False) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != FIELDS
        or value.get("version") != VERSION
        or len(canonical(value)) > 4096
    ):
        raise ValueError("Invalid notebook binding")
    for key in ("project_id", "project_key", "plan_id", "execution_id"):
        if not isinstance(value[key], str) or not 1 <= len(value[key]) <= 512:
            raise ValueError("Incomplete notebook identity")
    if type(value["plan_revision"]) is not int or value["plan_revision"] < 0:
        raise ValueError("Invalid notebook revision")
    for key in ("input_digest", "plan_digest", "result_digest", "study_digest"):
        if key == "result_digest" and allow_unknown_result and value[key] is None:
            continue
        if not isinstance(value[key], str) or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", value[key]
        ):
            raise ValueError("Incomplete notebook fingerprint")
    indices, coordinates = value["frequency_indices"], value["frequency_coordinates"]
    if (
        not isinstance(indices, list)
        or not 1 <= len(indices) <= 2
        or any(type(i) is not int or i not in (0, 1) for i in indices)
        or len(set(indices)) != len(indices)
        or not isinstance(coordinates, list)
        or len(coordinates) != len(indices)
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in coordinates)
    ):
        raise ValueError("Invalid notebook frequency context")
    if any(
        isinstance(v, bool)
        or not isinstance(v, (str, int, float))
        or not math.isfinite(float(v))
        for pair in coordinates
        for v in pair
    ):
        raise ValueError("Invalid notebook frequency coordinates")


def match(recorded: dict | None, expected: dict | None) -> dict:
    if recorded is None:
        return {"status": "unbound", "mismatch_fields": []}
    try:
        validate(recorded)
    except (ValueError, TypeError):
        return {"status": "unbound", "mismatch_fields": []}
    if expected is None:
        return {"status": "unconfirmed", "mismatch_fields": []}
    validate(expected, allow_unknown_result=True)
    changed = sorted(
        key
        for key in FIELDS
        if not (key == "result_digest" and expected[key] is None)
        and canonical(recorded[key]) != canonical(expected[key])
    )
    return {
        "status": (
            "mismatch"
            if changed
            else "unconfirmed" if expected["result_digest"] is None else "match"
        ),
        "mismatch_fields": changed,
    }


def current_result_digest(response: dict, view: dict) -> str | None:
    result, head = response.get("results") or {}, view["head"]
    expected = {
        "plan_id": head["plan_id"],
        "execution_id": head["execution_id"],
        "plan_revision": head["revision"],
        "plan_digest": view["plan"]["digest"],
    }
    digest = result.get("digest")
    if (
        any(result.get(k) != v for k, v in expected.items())
        or not isinstance(digest, str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest)
    ):
        return None
    return digest
