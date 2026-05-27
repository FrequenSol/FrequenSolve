import copy
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from numpy import array as NPArray

from frequensolve.geometry.frame import (
    CoordinateValue,
    Direction,
    coordinate_value_to_fs,
    direction_to_fs,
)
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.mixins import TypeTaggedMixin, warn_deprecated_path_api

__all__ = ["SourceGroup", "Source", "RuptureSource", "PointSource", "CompoundSource"]


@register_class
@dataclass
class Source(TypeTaggedMixin, ABC):
    """Abstract base class for solver source definitions."""

    @classmethod
    def from_fs(cls, data: Dict) -> "Source":
        """Deserialize a registered source payload.

        Args:
            data: Serialized source mapping containing ``_type``.

        Returns:
            Concrete source instance.
        """

        return cls.dispatch_from_fs(data, class_registry)

    @abstractmethod
    def to_fs(self, ctx=None) -> Dict:
        """Serialize the source for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible source payload.
        """

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
    def from_fs(cls, data: Dict) -> "RuptureSource":
        """Deserialize a rupture source payload."""

        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this rupture source for solver input."""

        return {"_type": self.__class__.__name__, **asdict(self)}


@register_class
@dataclass
class PointSource(Source):
    """Point source located at one physical coordinate.

    Args:
        kind: Source kind understood by the solver.
        coordinates: Source coordinate values, optionally with units/system
            metadata through ``CoordinateValue``.
        direction: Optional vector or moment direction.
        domain: Optional model domain id where the source is evaluated.
        name: Source name.
    """

    kind: Literal["scalar", "vector", "moment", "monopole", "dipole"]
    coordinates: Any = field(default_factory=list)
    direction: Optional[Any] = None
    domain: Optional[int] = None
    name: str = "point"

    @classmethod
    def from_fs(cls, data: Dict):
        """Deserialize a point source payload.

        Args:
            data: Serialized point-source mapping.

        Returns:
            ``PointSource`` instance.
        """

        data = copy.deepcopy(data)
        data.pop("frame", None)
        if "coordinates" in data:
            data["coordinates"] = CoordinateValue.from_fs(data["coordinates"])
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this point source for solver input."""

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "kind": self.kind,
            "coordinates": coordinate_value_to_fs(self.coordinates),
            **(
                {"direction": direction_to_fs(self.direction)}
                if self.direction is not None
                else {}
            ),
            **({"domain": self.domain} if self.domain is not None else {}),
        }


@register_class
@dataclass
class CompoundSource(Source):
    """Compound source made from multiple weighted source points.

    Args:
        kind: Source kind understood by the solver.
        coordinates: Source point coordinates.
        direction: Direction/weight vectors for each source point.
        domain: Optional model domain id where the source is evaluated.
        name: Source name.
    """

    kind: Literal["scalar", "vector"]
    coordinates: Any = field(default_factory=list)
    direction: Any = field(default_factory=list)
    domain: Optional[int] = None
    name: str = "compound"

    @classmethod
    def from_fs(cls, data: Dict):
        """Deserialize a compound source payload."""

        data = copy.deepcopy(data)
        data.pop("n_points", None)
        data.pop("frame", None)
        if "coordinates" in data:
            data["coordinates"] = CoordinateValue.from_fs(data["coordinates"])
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this compound source for solver input."""

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "kind": self.kind,
            "n_points": len(self.coordinates),
            "coordinates": coordinate_value_to_fs(self.coordinates),
            **(
                {"direction": direction_to_fs(self.direction)}
                if self.direction is not None
                else {}
            ),
            **({"domain": self.domain} if self.domain is not None else {}),
        }


@dataclass
class SourceGroup:
    """A source group simulated as one shot.

    Args:
        source: Source definition for this shot.
    """

    source: Source = field(default_factory=Source)
    _proj_path: Path = None
    _rel_path: Path = None

    @classmethod
    def from_fs(cls, data: Dict):
        """Deserialize a source-group payload."""

        source = Source.from_fs(copy.deepcopy(data.get("source", {})))
        return cls(source=source)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize the source group for solver input."""

        return {"source": self.source.to_fs(ctx)}

    def _set_path(self, proj_path: Path, rel_path: Path):
        warn_deprecated_path_api(f"{self.__class__.__name__}._set_path")
        self._proj_path = proj_path
        self._rel_path = rel_path

    def get_coordinates(self) -> NPArray:
        """Return source coordinates as a two-dimensional NumPy array."""

        coords = self.source.coordinates
        if isinstance(coords, CoordinateValue):
            coords = coords.value
        coords = NPArray(coords)
        if coords.ndim == 1:
            return coords.reshape(1, -1)
        return coords

    # TODO: fix this, point source will need to make 2D array
    def coordinates(self) -> NPArray:
        """Compatibility alias for ``get_coordinates``."""

        return self.get_coordinates()

    @property
    def _path(self) -> Path:
        warn_deprecated_path_api(f"{self.__class__.__name__}._path")
        return self._proj_path / self._rel_path
