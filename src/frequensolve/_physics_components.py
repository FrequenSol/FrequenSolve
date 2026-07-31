"""Dependency-free physics component identities used by package metadata."""

_FAMILY_COMPONENTS = {
    "acoustic": (("pressure", "velocity"), ()),
    "electromagnetic": (("electric", "magnetic"), ()),
    "elastic": (("velocity", "stress"), ("strain", "pressure")),
    "poroelastic": (
        ("velocity", "fluid_flux", "stress", "pressure"),
        ("strain", "displacement", "fluid_displacement"),
    ),
    "coupled-aep": (
        ("pressure", "velocity", "fluid_flux", "stress"),
        ("strain", "displacement", "fluid_displacement"),
    ),
}

_PHYSICS_FAMILIES = {
    "acoustic": "acoustic",
    "acoustic_axisym": "acoustic",
    "elastic": "elastic",
    "elastic_axisym": "elastic",
    "elastic_axisym_torsion": "elastic",
    "coupled": "elastic",
    "coupled_aep": "coupled-aep",
    "coupled_axisym": "elastic",
    "coupled_axisym_torsion": "elastic",
    "poroelastic": "poroelastic",
    "em": "electromagnetic",
}


def component_family(physics: str) -> str:
    """Return the component family for one canonical physics name."""

    try:
        return _PHYSICS_FAMILIES[physics]
    except KeyError:
        raise ValueError(f"Unsupported physics '{physics}'") from None


def family_components(family: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return primary and secondary component identities for a family."""

    try:
        return _FAMILY_COMPONENTS[family]
    except KeyError:
        raise ValueError(f"Unsupported component family '{family}'") from None


def allowed_components_for_physics(physics: str) -> tuple[str, ...]:
    """Return the stable allowed component names for canonical physics."""

    primary, secondary = family_components(component_family(physics))
    return tuple(dict.fromkeys((*primary, *secondary)))
