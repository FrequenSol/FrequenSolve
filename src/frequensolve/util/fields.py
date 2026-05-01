"""Canonical field names shared across acquisition, outputs, and physics."""

from __future__ import annotations

from typing import Iterable, List

FIELD_ALIASES = {}

FIELD_PASSTHROUGH = {"all", "primary", "secondary"}

__all__ = ["canonical_field", "canonical_fields"]


def canonical_field(field: str) -> str:
    """Return the canonical API/export name for a field."""

    return FIELD_ALIASES.get(field, field)


def canonical_fields(fields: Iterable[str]) -> List[str]:
    """Canonicalize an ordered field list while preserving output selectors."""

    return [
        field if field in FIELD_PASSTHROUGH else canonical_field(field)
        for field in fields
    ]
