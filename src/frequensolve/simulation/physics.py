"""Defines canonical physics and field/component names."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Iterable, List, Mapping

from frequensolve._physics_components import family_components
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
    "CoupledAEPComponents",
    "PoroelasticComponents",
    "EMComponents",
    "canonical_physics",
    "canonical_dimension",
    "model_dimension",
    "normalize_simulation_physics",
    "components_for_physics",
]


class ValidComponents(ABC):
    """Component registry for one solver physics family.

    Attributes:
        primary: Fields solved directly by the physics formulation.
        secondary: Derived fields that can be requested from solver output.
        aliases: Mapping from public aliases to canonical component names.
    """

    primary: ClassVar[List[str]] = []
    secondary: ClassVar[List[str]] = []
    aliases: ClassVar[Mapping[str, str]] = FIELD_ALIASES

    @classmethod
    def allowed_components(cls) -> List[str]:
        """Return all canonical component names accepted by this registry."""

        return list(dict.fromkeys([*cls.primary, *cls.secondary]))

    @classmethod
    def check_components(cls, components: Iterable[str]) -> List[str]:
        """Normalize and validate component names.

        Args:
            components: Component names or aliases requested by the user.

        Returns:
            Canonical component names in input order.

        Raises:
            ValueError: If any component is not supported by this physics
                family.
        """

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
    """Valid output components for acoustic simulations."""

    primary = list(family_components("acoustic")[0])


class EMComponents(ValidComponents):
    """Valid output components for electromagnetic simulations."""

    primary = list(family_components("electromagnetic")[0])


class ElasticComponents(ValidComponents):
    """Valid output components for elastic simulations."""

    primary = list(family_components("elastic")[0])
    secondary = list(family_components("elastic")[1])


class PoroelasticComponents(ValidComponents):
    """Valid output components for poroelastic simulations."""

    primary = list(family_components("poroelastic")[0])
    secondary = list(family_components("poroelastic")[1])


class CoupledAEPComponents(ValidComponents):
    """Valid output components for coupled acoustic-elastic-poroelastic runs."""

    primary = list(family_components("coupled-aep")[0])
    secondary = list(family_components("coupled-aep")[1])


_COMPONENTS_BY_PHYSICS = {
    "acoustic": AcousticComponents,
    "acoustic_axisym": AcousticComponents,
    "elastic": ElasticComponents,
    "elastic_axisym": ElasticComponents,
    "elastic_axisym_torsion": ElasticComponents,
    "coupled": ElasticComponents,
    "coupled_aep": CoupledAEPComponents,
    "coupled_axisym": ElasticComponents,
    "coupled_axisym_torsion": ElasticComponents,
    "poroelastic": PoroelasticComponents,
    "em": EMComponents,
}


def components_for_physics(physics: str) -> type[ValidComponents]:
    """Return the component registry for a physics name or alias.

    Args:
        physics: Physics name or supported alias.

    Returns:
        ``ValidComponents`` subclass for the canonical physics family.

    Raises:
        KeyError: If no component registry is defined for the canonical physics.
        ValueError: If ``physics`` is not a supported physics name.
    """

    return _COMPONENTS_BY_PHYSICS[canonical_physics(physics)]
