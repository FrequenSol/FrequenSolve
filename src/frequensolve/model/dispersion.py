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
    """Abstract base class for frequency-dependent material scaling."""

    @classmethod
    def from_fs(cls, data: Dict) -> "DispersionRelation":
        """Deserialize a registered dispersion relation payload.

        Args:
            data: Serialized dispersion mapping containing ``_type``.

        Returns:
            Concrete dispersion relation instance.
        """

        return cls.dispatch_from_fs(data, class_registry)

    @abstractmethod
    def to_fs(self, ctx=None) -> Dict:
        """Serialize the dispersion relation for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible dispersion payload.
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
        """Return a scaled property using left-multiplication syntax."""

        return DispersionScaling(property=other, dispersion=self)


@register_class
class PowerLawDispersion(DispersionRelation):
    """Power-law frequency-dependent dispersion relation.

    Args:
        f0: Reference frequency in hertz.
        alpha: Power-law exponent.
    """

    def __init__(self, f0: float, alpha: float):
        self.f0 = f0
        self.alpha = alpha

    @classmethod
    def from_fs(cls, data: Dict) -> "PowerLawDispersion":
        """Deserialize a power-law dispersion payload.

        Args:
            data: Serialized dispersion mapping.

        Returns:
            ``PowerLawDispersion`` instance.
        """

        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this power-law dispersion relation.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible power-law dispersion payload.
        """

        return {"_type": self.__class__.__name__, "f0": self.f0, "alpha": self.alpha}


@register_class
class TablulatedDispersion(DispersionRelation):
    """Table-based dispersion relation with interpolation controls.

    Args:
        frequencies: Tabulated frequencies in hertz.
        values: Scaling values corresponding to ``frequencies``.
        interpolation: Interpolation method between tabulated frequencies.
        extrapolation: Extrapolation method outside the tabulated range.
    """

    def __init__(
        self,
        frequencies: List[float],
        values: List[float],
        interpolation: str = "linear",
        extrapolation: str = "nearest",
    ):
        self.frequencies = frequencies
        self.values = values
        self.interpolation = interpolation
        self.extrapolation = extrapolation

    @classmethod
    def from_fs(cls, data: Dict) -> "TablulatedDispersion":
        """Deserialize a tabulated dispersion payload.

        Args:
            data: Serialized dispersion mapping.

        Returns:
            ``TablulatedDispersion`` instance.
        """

        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this tabulated dispersion relation.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible tabulated dispersion payload.
        """

        return {
            "_type": self.__class__.__name__,
            "frequencies": self.frequencies,
            "values": self.values,
            "extrapolation": self.extrapolation,
        }


@register_class
class DispersionScaling:
    """Material property paired with a dispersion relation.

    Args:
        property: Constant, file path, xarray object, or other material property
            source to scale.
        dispersion: Dispersion relation applied to the property.
    """

    def __init__(self, property: Any, dispersion: DispersionRelation):
        self.property = property
        self.dispersion = dispersion
