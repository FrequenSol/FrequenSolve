"""Deprecated source types retained for public import compatibility."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from frequensolve.geometry.frame import (
    CoordinateValue,
    Direction,
    coordinate_value_to_fs,
    direction_to_fs,
)
from frequensolve.units import is_quantity
from frequensolve.util.mixins import ExportContext, warn_deprecated_path_api


class Source:
    """Compatibility dispatcher for legacy source-group payloads."""

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> Any:
        from frequensolve.seismic.sources import PointSource

        payload = copy.deepcopy(dict(data))
        source_type = payload.pop("_type", "PointSource")
        if source_type == "PointSource":
            return PointSource.from_fs(payload)
        if source_type == "CompoundSource":
            return CompoundSource.from_fs(payload)
        if source_type == "RuptureSource":
            return RuptureSource.from_fs(payload)
        raise ValueError(f"Unsupported legacy source type {source_type!r}")


@dataclass
class RuptureSource(Source):
    """Deprecated legacy SRF source retained for input compatibility."""

    srf_file: str
    name: str = "rupture"

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "RuptureSource":
        return cls(**copy.deepcopy(dict(data)))

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        return {
            "_type": "RuptureSource",
            "srf_file": self.srf_file,
            "name": self.name,
        }


@dataclass
class CompoundSource(Source):
    """Deprecated weighted-point source retained as an adapter input."""

    kind: str
    coordinates: Any = field(default_factory=list)
    direction: Any = field(default_factory=list)
    domain: Optional[int] = None
    name: str = "compound"

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "CompoundSource":
        payload = copy.deepcopy(dict(data))
        payload.pop("n_points", None)
        payload.pop("frame", None)
        if "coordinates" in payload:
            payload["coordinates"] = CoordinateValue.from_fs(payload["coordinates"])
        if "direction" in payload:
            payload["direction"] = Direction.from_fs(payload["direction"])
        return cls(**payload)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        return {
            "_type": "CompoundSource",
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
    """Deprecated logical-source view used by pre-v2 callers."""

    source: Any
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SourceGroup":
        return cls(source=Source.from_fs(copy.deepcopy(data.get("source", {}))))

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        from frequensolve.seismic.sources import PointSource

        source = self.source.to_fs(ctx)
        if isinstance(self.source, PointSource):
            source = {"_type": "PointSource", **source}
        return {"source": source}

    def _set_path(self, proj_path: Path, rel_path: Path) -> None:
        warn_deprecated_path_api(f"{self.__class__.__name__}._set_path")
        self._proj_path = Path(proj_path)
        self._rel_path = Path(rel_path)

    def get_coordinates(self) -> np.ndarray:
        """Return source coordinates as a two-dimensional array."""

        coords = self.source.coordinates
        if isinstance(coords, CoordinateValue):
            coords = coords.value
        if is_quantity(coords):
            coords = coords.magnitude
        values = np.asarray(coords, dtype=float)
        if values.ndim == 1:
            return values.reshape(1, -1)
        return values

    def coordinates(self) -> np.ndarray:
        """Compatibility alias for :meth:`get_coordinates`."""

        return self.get_coordinates()

    @property
    def _path(self) -> Path:
        warn_deprecated_path_api(f"{self.__class__.__name__}._path")
        if self._proj_path is None or self._rel_path is None:
            raise ValueError("Source path requires a project and relative path")
        return self._proj_path / self._rel_path
