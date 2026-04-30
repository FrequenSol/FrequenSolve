from abc import ABC, abstractmethod
from typing import Any, Dict, List

from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.mixins import TypeTaggedMixin


@register_class
class DispersionRelation(TypeTaggedMixin, ABC):
    """
    Abstract base class for dispersion relations.
    """

    @classmethod
    def from_fs(cls, data: Dict) -> "DispersionRelation":
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
    """
    Power-law dispersion relation.
    """

    def __init__(self, f0: float, alpha: float):
        self.f0 = f0
        self.alpha = alpha

    @classmethod
    def from_fs(cls, data: Dict) -> "PowerLawDispersion":
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        return {"_type": self.__class__.__name__, "f0": self.f0, "alpha": self.alpha}


@register_class
class TablulatedDispersion(DispersionRelation):
    """
    Linear table-based dispersion relation.
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
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        return {
            "_type": self.__class__.__name__,
            "frequencies": self.frequencies,
            "values": self.values,
            "extrapolation": self.extrapolation,
        }


@register_class
class DispersionScaling:
    """
    Represents a data source (constant, file, or xarray) scaled by a dispersion relation.
    """

    def __init__(self, property: Any, dispersion: DispersionRelation):
        self.property = property
        self.dispersion = dispersion
