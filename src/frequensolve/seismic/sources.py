import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from numpy import array as NPArray

from frequensolve.geometry.frame import (
    CoordinateValue,
    Direction,
    coordinate_value_to_fs,
    direction_to_fs,
)
from frequensolve.util.class_registry import class_registry, register_class

__all__ = ["SourceGroup", "Source", "RuptureSource", "PointSource", "CompoundSource"]


@register_class
@dataclass
class Source(ABC):

    @classmethod
    def from_dict(cls, data: Dict) -> "Source":
        data = copy.deepcopy(data)
        class_name = data.pop("_type")
        if class_name in class_registry:
            source_class = class_registry[class_name]
            return source_class.from_dict(data)
        else:
            raise ValueError(f"Unknown source class: {class_name}")

    @abstractmethod
    def __dict__(self) -> Dict:
        pass


@register_class
@dataclass
class RuptureSource(Source):
    """Source defined from Standard Rupture Format file.

    Args:
       srf_file: Path to SRF file containing rupture definition.
       name: Optional name for the source.

    Attributes:
       srf_file (str):   Path to SRF file.
       name (str):       Name of the source.
    """

    srf_file: str
    name: str = "rupture"

    @classmethod
    def from_dict(cls, data: Dict) -> "RuptureSource":
        return cls(**data)

    def __dict__(self) -> Dict:
        return self.to_fs()

    def to_fs(self, ctx=None) -> Dict:
        return {"_type": self.__class__.__name__, **asdict(self)}


@register_class
@dataclass
class PointSource(Source):
    kind: Literal["scalar", "vector", "moment", "monopole", "dipole"]
    frame: Literal["physical", "reference"] = "physical"
    coordinates: Any = field(default_factory=list)
    direction: Optional[Any] = None
    domain: Optional[int] = None
    name: str = "point"

    @classmethod
    def from_dict(cls, data: Dict):
        data = copy.deepcopy(data)
        if "coordinates" in data:
            data["coordinates"] = CoordinateValue.from_fs(data["coordinates"])
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "kind": self.kind,
            "frame": self.frame,
            "coordinates": coordinate_value_to_fs(self.coordinates),
            **(
                {"direction": direction_to_fs(self.direction)}
                if self.direction is not None
                else {}
            ),
            **({"domain": self.domain} if self.domain is not None else {}),
        }

    def __dict__(self) -> Dict:
        return self.to_fs()


@register_class
@dataclass
class CompoundSource(Source):
    kind: Literal["scalar", "vector"]
    frame: Literal["physical", "reference"] = "physical"
    coordinates: Any = field(default_factory=list)
    direction: Any = field(default_factory=list)
    domain: Optional[int] = None
    name: str = "compound"

    @classmethod
    def from_dict(cls, data: Dict):
        data = copy.deepcopy(data)
        data.pop("n_points", None)
        if "coordinates" in data:
            data["coordinates"] = CoordinateValue.from_fs(data["coordinates"])
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "kind": self.kind,
            "frame": self.frame,
            "n_points": len(self.coordinates),
            "coordinates": coordinate_value_to_fs(self.coordinates),
            **(
                {"direction": direction_to_fs(self.direction)}
                if self.direction is not None
                else {}
            ),
            **({"domain": self.domain} if self.domain is not None else {}),
        }

    def __dict__(self) -> Dict:
        return self.to_fs()


@dataclass
class SourceGroup:
    """A group of sources (to simulate simultaneously)

    Attributes:
       sources (List[PointSource]):     List of source objects
    """

    source: Source = field(default_factory=Source)
    _proj_path: Path = None
    _rel_path: Path = None

    @classmethod
    def from_dict(cls, data: Dict):
        source = Source.from_dict(copy.deepcopy(data.get("source", {})))
        return cls(source=source)

    def to_fs(self, ctx=None) -> Dict:
        return {
            "source": (
                self.source.to_fs(ctx)
                if hasattr(self.source, "to_fs")
                else self.source.__dict__()
            )
        }

    def __dict__(self) -> Dict:
        return self.to_fs()

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

    def get_coordinates(self) -> NPArray:
        coords = self.source.coordinates
        if isinstance(coords, CoordinateValue):
            coords = coords.value
        coords = NPArray(coords)
        if coords.ndim == 1:
            return coords.reshape(1, -1)
        return coords

    # TODO: fix this, point source will need to make 2D array
    def coordinates(self) -> NPArray:
        return self.get_coordinates()

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
