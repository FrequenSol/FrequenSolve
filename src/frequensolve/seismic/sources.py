"""Source objects for seismic acquisition definitions."""

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
from frequensolve.util.mixins import TypeTaggedMixin

__all__ = ["SourceGroup", "Source", "RuptureSource", "PointSource", "CompoundSource"]


@register_class
@dataclass
class Source(TypeTaggedMixin, ABC):
    """Base class for all solver-facing source definitions."""

    @classmethod
    def from_fs(cls, data: Dict) -> "Source":
        """Deserialize a registered source subclass from a solver payload."""

        return cls.dispatch_from_fs(data, class_registry)

    @abstractmethod
    def to_fs(self, ctx=None) -> Dict:
        """Serialize this source to the solver input contract."""

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
        """Deserialize a rupture source from a solver payload."""

        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this rupture source to the solver input contract."""

        return {"_type": self.__class__.__name__, **asdict(self)}


@register_class
@dataclass
class PointSource(Source):
    """Single point source with optional direction and domain targeting.

    ``coordinates`` may be raw coordinates or a :class:`CoordinateValue` that
    names the coordinate system. Vector, moment, and dipole sources should also
    provide ``direction``.
    """

    kind: Literal["scalar", "vector", "moment", "monopole", "dipole"]
    coordinates: Any = field(default_factory=list)
    direction: Optional[Any] = None
    domain: Optional[int] = None
    name: str = "point"

    @classmethod
    def from_fs(cls, data: Dict):
        """Deserialize a point source from a solver payload."""

        data = copy.deepcopy(data)
        data.pop("frame", None)
        if "coordinates" in data:
            data["coordinates"] = CoordinateValue.from_fs(data["coordinates"])
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this point source to the solver input contract."""

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
    """Multi-point source whose coordinates share one source kind and direction."""

    kind: Literal["scalar", "vector"]
    coordinates: Any = field(default_factory=list)
    direction: Any = field(default_factory=list)
    domain: Optional[int] = None
    name: str = "compound"

    @classmethod
    def from_fs(cls, data: Dict):
        """Deserialize a compound source from a solver payload."""

        data = copy.deepcopy(data)
        data.pop("n_points", None)
        data.pop("frame", None)
        if "coordinates" in data:
            data["coordinates"] = CoordinateValue.from_fs(data["coordinates"])
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(**data)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this compound source to the solver input contract."""

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
    """A source group that is simulated as one shot.

    Attributes:
        source: Source object that defines the emitted field.
    """

    source: Source = field(default_factory=Source)
    _proj_path: Path = None
    _rel_path: Path = None

    @classmethod
    def from_fs(cls, data: Dict):
        """Deserialize a source group from a solver payload."""

        source = Source.from_fs(copy.deepcopy(data.get("source", {})))
        return cls(source=source)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this source group to the solver input contract."""

        return {"source": self.source.to_fs(ctx)}

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

    def get_coordinates(self) -> NPArray:
        """Return source coordinates as a two-dimensional numpy array."""

        coords = self.source.coordinates
        if isinstance(coords, CoordinateValue):
            coords = coords.value
        coords = NPArray(coords)
        if coords.ndim == 1:
            return coords.reshape(1, -1)
        return coords

    # TODO: fix this, point source will need to make 2D array
    def coordinates(self) -> NPArray:
        """Compatibility alias for :meth:`get_coordinates`."""

        return self.get_coordinates()

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
