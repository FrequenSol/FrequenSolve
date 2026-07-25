"""Lightweight access to the packaged validation-code registry.

The versioned simulation-knowledge JSON is the single authoritative source for
stable validation-code metadata.  Package validators load only this small
registry view, lazily, so importing :mod:`frequensolve.validation` does not load
the full knowledge model or introduce a validation/knowledge import cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Mapping

_KNOWLEDGE_PACKAGE_DIRECTORY = "knowledge"
_CATALOG_RESOURCE = "simulation_knowledge_v1.json"


@dataclass(frozen=True)
class ValidationCodeSpec:
    """Registry fields needed to enforce package-validator diagnostics."""

    severity: str


@lru_cache(maxsize=1)
def load_validation_code_registry() -> Mapping[str, ValidationCodeSpec]:
    """Return immutable validation-code metadata from the packaged catalog."""

    resource = (
        files("frequensolve")
        .joinpath(_KNOWLEDGE_PACKAGE_DIRECTORY)
        .joinpath(_CATALOG_RESOURCE)
    )
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
        entries = payload["validation_codes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Packaged simulation knowledge has an invalid validation-code registry"
        ) from exc
    if not isinstance(entries, list):
        raise RuntimeError(
            "Packaged simulation knowledge validation_codes must be an array"
        )

    registry: dict[str, ValidationCodeSpec] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Packaged validation_codes[{index}] must be an object")
        code = entry.get("code")
        severity = entry.get("severity")
        if not isinstance(code, str) or not code:
            raise RuntimeError(
                f"Packaged validation_codes[{index}].code must be non-empty"
            )
        if severity not in {"error", "warning"}:
            raise RuntimeError(
                f"Packaged validation code {code!r} has invalid severity "
                f"{severity!r}"
            )
        if code in registry:
            raise RuntimeError(f"Packaged validation code {code!r} is duplicated")
        registry[code] = ValidationCodeSpec(severity=severity)
    return MappingProxyType(registry)
