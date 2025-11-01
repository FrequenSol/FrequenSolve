from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.util.class_registry import class_registry, register_class


@register_class
class DispersionRelation(ABC):
    """
    Abstract base class for dispersion relations.
    """

    @classmethod
    def from_dict(cls, data: Dict) -> "DispersionRelation":
        class_name = data.pop("_type")
        if class_name in class_registry:
            source_class = class_registry[class_name]
            return source_class.from_dict(data)
        else:
            raise ValueError(f"Unknown source class: {class_name}")

    @abstractmethod
    def to_dict(self) -> Dict:
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

    # For backward compatibility with your code
    def __dict__(self) -> Dict:
        return self.to_dict()


@register_class
class PowerLawDispersion(DispersionRelation):
    """
    Power-law dispersion relation.
    """

    def __init__(self, f0: float, alpha: float):
        self.f0 = f0
        self.alpha = alpha

    @classmethod
    def from_dict(cls, data: Dict) -> "PowerLawDispersion":
        return cls(**data)

    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict) -> "TablulatedDispersion":
        return cls(**data)

    def to_dict(self) -> Dict:
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
