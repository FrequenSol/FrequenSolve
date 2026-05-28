"""Frequency-dependent material-property dispersion models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.mixins import TypeTaggedMixin

__all__ = [
    "DispersionRelation",
    "DispersionScaling",
    "PowerLawDispersion",
    "TablulatedDispersion",
]


@register_class
class DispersionRelation(TypeTaggedMixin, ABC):
    """Base class for frequency-dependent property scaling."""

    @classmethod
    def from_fs(cls, data: Dict) -> "DispersionRelation":
        """Deserialize a registered dispersion relation from a solver payload."""

        return cls.dispatch_from_fs(data, class_registry)

    @abstractmethod
    def to_fs(self, ctx=None) -> Dict:
        """
        Serialize the dispersion relation to a dictionary.
        """
        pass

    def __rmul__(self, other: Any) -> "DispersionScaling":
        """
        Allow multiplication of a data source by a dispersion relation.

        Args:
            other: The data source (float, str, Path, or xarray.DataArray).

        Returns:
            DispersionScaling: The combined data and dispersion relation.
        """
        return DispersionScaling(property=other, dispersion=self)

    def __lmul__(self, other: Any) -> "DispersionScaling":
        """
        Same as __rmul__.
        """
        return DispersionScaling(property=other, dispersion=self)


@register_class
class PowerLawDispersion(DispersionRelation):
    """Power-law dispersion relation defined by a reference frequency and exponent."""

    def __init__(self, f0: float, alpha: float):
        """Create a power-law dispersion model.

        Args:
            f0: Reference frequency.
            alpha: Power-law exponent.
        """

        self.f0 = f0
        self.alpha = alpha

    @classmethod
    def from_fs(cls, data: Dict) -> "PowerLawDispersion":
        """Deserialize a power-law dispersion relation."""

        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this relation to the solver payload."""

        return {"_type": self.__class__.__name__, "f0": self.f0, "alpha": self.alpha}


@register_class
class TablulatedDispersion(DispersionRelation):
    """Table-based dispersion relation with configurable interpolation.

    The class name preserves the current public API spelling.
    """

    def __init__(
        self,
        frequencies: List[float],
        values: List[float],
        interpolation: str = "linear",
        extrapolation: str = "nearest",
    ):
        """Create a table-based dispersion relation.

        Args:
            frequencies: Frequency samples for the table.
            values: Dispersion scale values at ``frequencies``.
            interpolation: Interpolation method between samples.
            extrapolation: Extrapolation policy outside the sampled range.
        """

        self.frequencies = frequencies
        self.values = values
        self.interpolation = interpolation
        self.extrapolation = extrapolation

    @classmethod
    def from_fs(cls, data: Dict) -> "TablulatedDispersion":
        """Deserialize a table-based dispersion relation."""

        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this relation to the solver payload."""

        return {
            "_type": self.__class__.__name__,
            "frequencies": self.frequencies,
            "values": self.values,
            "extrapolation": self.extrapolation,
        }


@register_class
class DispersionScaling:
    """Pair a property data source with a dispersion relation.

    Users normally create this by multiplying a constant, file, or xarray value
    by a :class:`DispersionRelation`.
    """

    def __init__(self, property: Any, dispersion: DispersionRelation):
        """Create a property/dispersion pair."""

        self.property = property
        self.dispersion = dispersion
