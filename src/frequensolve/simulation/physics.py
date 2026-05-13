"""Defines canonical physics and field/component names."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Iterable, List, Mapping

from frequensolve.util.fields import FIELD_ALIASES
from frequensolve.util.physics import (
    canonical_dimension,
    canonical_physics,
    model_dimension,
    normalize_simulation_physics,
)

__all__ = [
    "AcousticComponents",
    "ElasticComponents",
    "PoroelasticComponents",
    "EMComponents",
    "canonical_physics",
    "canonical_dimension",
    "model_dimension",
    "normalize_simulation_physics",
    "components_for_physics",
]


class ValidComponents(ABC):
    """Valid components for each physics type.

    Attributes:
       primary (List[str]): Simulated variables.
       secondary (List[str]): Derived variables.
    """

    primary: ClassVar[List[str]] = []
    secondary: ClassVar[List[str]] = []
    aliases: ClassVar[Mapping[str, str]] = FIELD_ALIASES

    @classmethod
    def allowed_components(cls) -> List[str]:
        return list(dict.fromkeys([*cls.primary, *cls.secondary]))

    @classmethod
    def check_components(cls, components: Iterable[str]) -> List[str]:
        canonical = [cls.aliases.get(component, component) for component in components]
        allowed = set(cls.allowed_components())
        unknown = sorted(
            {component for component in canonical if component not in allowed}
        )
        if unknown:
            raise ValueError(
                f"Unknown {cls.__name__} component(s): {unknown}. "
                f"Allowed components are {sorted(allowed)}."
            )
        return canonical


class AcousticComponents(ValidComponents):
    primary = ["pressure", "velocity"]


class EMComponents(ValidComponents):
    primary = ["electric", "magnetic"]


class ElasticComponents(ValidComponents):
    primary = ["velocity", "stress"]
    secondary = ["strain", "pressure"]


class PoroelasticComponents(ValidComponents):
    primary = ["velocity", "fluid_flux", "stress", "pressure"]
    secondary = ["strain", "displacement", "fluid_displacement"]


_COMPONENTS_BY_PHYSICS = {
    "acoustic": AcousticComponents,
    "acoustic_axisym": AcousticComponents,
    "elastic": ElasticComponents,
    "elastic_axisym": ElasticComponents,
    "elastic_axisym_torsion": ElasticComponents,
    "coupled": ElasticComponents,
    "coupled_axisym": ElasticComponents,
    "coupled_axisym_torsion": ElasticComponents,
    "poroelastic": PoroelasticComponents,
    "EM": EMComponents,
}


def components_for_physics(physics: str) -> type[ValidComponents]:
    """Return the component registry for a physics name or alias."""

    return _COMPONENTS_BY_PHYSICS[canonical_physics(physics)]
