"""Shared physics and dimension normalization helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "canonical_physics",
    "canonical_dimension",
    "model_dimension",
    "normalize_simulation_physics",
]


_PHYSICS_ALIASES = {
    "acoustic": "acoustic",
    "acoustic_axisym": "acoustic_axisym",
    "elastic": "elastic",
    "elastic_axisym": "elastic_axisym",
    "elastic_axisym_torsion": "elastic_axisym_torsion",
    "coupled": "coupled",
    "coupled_aep": "coupled_aep",
    "coupled-aep": "coupled_aep",
    "coupledaep": "coupled_aep",
    "coupled_axisym": "coupled_axisym",
    "coupled_axisym_torsion": "coupled_axisym_torsion",
    "poro": "poroelastic",
    "poroelastic": "poroelastic",
    "poro-elastic": "poroelastic",
    "poro_elastic": "poroelastic",
    "biot": "poroelastic",
    "em": "em",
    "electromagnetic": "em",
    "electro-magnetic": "em",
    "electro_magnetic": "em",
    "maxwell": "em",
}

_AXISYMMETRIC_PHYSICS = {
    "acoustic_axisym",
    "elastic_axisym",
    "elastic_axisym_torsion",
    "coupled_axisym",
    "coupled_axisym_torsion",
}

_AXISYMMETRIC_DEFAULTS = {
    "acoustic": "acoustic_axisym",
    "elastic": "elastic_axisym",
    "coupled": "coupled_axisym",
}


def canonical_physics(physics: str) -> str:
    """Return the solver-facing physics name for friendly user input.

    Args:
        physics: User-facing physics name or alias.

    Returns:
        Canonical solver physics name.

    Raises:
        ValueError: If ``physics`` is not recognized.
    """

    key = str(physics).strip()
    normalized = key.lower().replace(" ", "")
    if normalized in _PHYSICS_ALIASES:
        return _PHYSICS_ALIASES[normalized]
    allowed = ", ".join(sorted(dict.fromkeys(_PHYSICS_ALIASES.values())))
    raise ValueError(f"Unknown physics {physics!r}. Expected one of: {allowed}.")


def normalize_simulation_physics(
    physics: str,
    *,
    axisymmetric: bool = False,
    dimension: Any | None = None,
) -> tuple[str, bool]:
    """Normalize physics and axisymmetry into the solver formulation key.

    Args:
        physics: User-facing physics name or alias.
        axisymmetric: Whether to request the axisymmetric formulation.
        dimension: Optional simulation dimension used to validate
            axisymmetric requests.

    Returns:
        Tuple ``(canonical_physics, axisymmetric)``.

    Raises:
        ValueError: If the requested axisymmetric formulation is unsupported or
            used with a non-2D simulation dimension.
    """

    canonical = canonical_physics(physics)
    explicit_axisym = canonical in _AXISYMMETRIC_PHYSICS
    axisymmetric = bool(axisymmetric or explicit_axisym)

    if axisymmetric and not explicit_axisym:
        try:
            canonical = _AXISYMMETRIC_DEFAULTS[canonical]
        except KeyError as exc:
            raise ValueError(
                "axisymmetric simulations are currently supported for acoustic, "
                "elastic, and coupled elastic physics only"
            ) from exc

    if axisymmetric and dimension is not None and canonical_dimension(dimension) != 2:
        raise ValueError("axisymmetric simulations require dimension=2")

    return canonical, axisymmetric


def canonical_dimension(dimension: Any) -> int | float:
    """Normalize 2D, 2.5D, and 3D user input for simulation JSON.

    Args:
        dimension: Numeric or string dimension specifier.

    Returns:
        ``2``, ``2.5``, or ``3``.

    Raises:
        ValueError: If the dimension cannot be normalized.
    """

    if isinstance(dimension, str):
        key = dimension.strip().lower().replace(" ", "")
        key = key.replace("_", ".").replace("-", ".")
        aliases = {
            "2": 2,
            "2d": 2,
            "2.0": 2,
            "2.0d": 2,
            "2.5": 2.5,
            "2.5d": 2.5,
            "2p5": 2.5,
            "2p5d": 2.5,
            "3": 3,
            "3d": 3,
            "3.0": 3,
            "3.0d": 3,
        }
        if key in aliases:
            return aliases[key]
    elif isinstance(dimension, (int, float)):
        value = float(dimension)
        if value == 2.0:
            return 2
        if value == 2.5:
            return 2.5
        if value == 3.0:
            return 3

    raise ValueError("dimension must be 2, 2.5, 3, '2D', '2.5D', or '3D'")


def model_dimension(dimension: Any) -> int:
    """Return the spatial model/mesh dimension for a simulation dimension.

    Args:
        dimension: Numeric or string dimension specifier.

    Returns:
        Model dimension. ``2.5D`` simulations use a 2D model/mesh.
    """

    value = canonical_dimension(dimension)
    return 2 if value == 2.5 else int(value)
