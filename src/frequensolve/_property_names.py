"""Dependency-free material-property name normalization."""

from typing import Any

PROPERTY_ALIASES = {
    "vp": "vp",
    "vs": "vs",
    "rho": "rho",
    "density": "rho",
    "qp": "qp",
    "qs": "qs",
    "vadapt": "vadapt",
    "epsilon": "epsilon",
    "gamma": "gamma",
    "delta": "delta",
    "phi": "phi",
    "theta": "theta",
}


def canonical_property_name(name: Any) -> str:
    """Return the canonical lowercase name for a material-property alias."""

    key = str(name).strip()
    normalized = key.lower()
    return PROPERTY_ALIASES.get(normalized, normalized)
