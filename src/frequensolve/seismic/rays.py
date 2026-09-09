"""Readers for Sauce's authoritative ``fs-rays-1`` HDF5 product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union

import h5py
import numpy as np

__all__ = ["RayPath", "RayResults"]

_REQUIRED_GROUPS = (
    "metadata",
    "sources",
    "receivers",
    "rays",
    "points",
    "events",
    "receiver_hits",
    "status_codes",
)
_TABLE_IDS = {
    "sources": "source_id",
    "receivers": "receiver_id",
    "rays": "ray_id",
    "points": "ray_id",
    "events": "event_id",
    "receiver_hits": "hit_id",
}


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
        return np.asarray(
            [_decode(item) for item in value.reshape(-1)], dtype=object
        ).reshape(value.shape)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _record_value(value: np.ndarray, row: int) -> Any:
    selected = value[row]
    if isinstance(selected, np.ndarray):
        return selected.copy()
    return _decode(selected)


@dataclass(frozen=True)
class RayPath:
    """One ray descriptor and its contiguous retained path points."""

    ray_id: int
    ray: Mapping[str, Any]
    points: Mapping[str, np.ndarray]

    @property
    def position(self) -> np.ndarray:
        """Physical point coordinates with shape ``(point, dimension)``."""

        return np.asarray(self.points["position"])

    @property
    def travel_time(self) -> np.ndarray:
        """Cumulative travel time at retained path points."""

        return np.asarray(self.points["tau"])

    @property
    def arc_length(self) -> np.ndarray:
        """Cumulative physical path length at retained path points."""

        return np.asarray(self.points["arc_length"])


class RayResults:
    """Typed, self-validating access to one ``fs-rays-1`` result.

    ``path`` may name the HDF5 file, its sibling ``manifest.json``, or the
    directory containing the manifest. Numerical arrays are loaded lazily and
    cached on first access.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        result_root: Optional[Union[str, Path]] = None,
        validate: bool = True,
    ):
        supplied = Path(path).expanduser().resolve()
        self.result_root = (
            Path(result_root).expanduser().resolve()
            if result_root is not None
            else None
        )
        self.manifest_file: Optional[Path]
        if supplied.is_dir():
            self.manifest_file = supplied / "manifest.json"
            if not self.manifest_file.is_file():
                raise FileNotFoundError(f"No fs-rays-1 manifest found in {supplied}")
            self.hdf5_file = self._hdf5_from_manifest(self.manifest_file)
        elif supplied.name == "manifest.json" or supplied.suffix.lower() == ".json":
            self.manifest_file = supplied
            if not supplied.is_file():
                raise FileNotFoundError(supplied)
            self.hdf5_file = self._hdf5_from_manifest(supplied)
        else:
            self.hdf5_file = supplied
            sibling = supplied.parent / "manifest.json"
            self.manifest_file = sibling if sibling.is_file() else None
        if not self.hdf5_file.is_file():
            raise FileNotFoundError(self.hdf5_file)
        if validate:
            self.validate()

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        *,
        validate: bool = True,
    ) -> "RayResults":
        """Open a ray directory, manifest, or HDF5 product."""

        return cls(path, validate=validate)

    @classmethod
    def from_job(cls, job: Any, *, validate: bool = True) -> "RayResults":
        """Open the result paths declared by a ``RayTracingJob``."""

        return cls(
            job.ray_manifest_file,
            result_root=job._result_path,
            validate=validate,
        )

    @cached_property
    def manifest(self) -> Dict[str, Any]:
        """Parsed ``fs-rays-1`` manifest, or an empty mapping for bare HDF5."""

        if self.manifest_file is None:
            return {}
        try:
            return json.loads(self.manifest_file.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid ray manifest {self.manifest_file}: {exc}"
            ) from exc

    def _hdf5_from_manifest(self, manifest_file: Path) -> Path:
        try:
            manifest = json.loads(manifest_file.read_text())
            relative = Path(manifest["hdf5"]["relative_path"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid fs-rays-1 manifest {manifest_file}") from exc
        candidates = []
        if self.result_root is not None:
            candidates.append(self.result_root / relative)
        candidates.extend(
            [
                manifest_file.parent / relative,
                manifest_file.parent.parent / relative,
                manifest_file.parent / relative.name,
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve()

    @cached_property
    def metadata(self) -> Dict[str, Any]:
        """Scalar and JSON metadata stored under ``/metadata``."""

        return self._read_group("metadata", normalize_vectors=False)

    @property
    def dimension(self) -> int:
        """Physical coordinate dimension, either two or three."""

        value = self.manifest.get("dimension", self.metadata.get("dimension"))
        return int(value)

    @property
    def status(self) -> Optional[str]:
        """Manifest completion state when a manifest is available."""

        value = self.manifest.get("status")
        return str(value) if value is not None else None

    @property
    def counts(self) -> Dict[str, int]:
        """Logical table and failure counts."""

        if self.manifest:
            return {
                str(name): int(value)
                for name, value in self.manifest.get("counts", {}).items()
            }
        return {name: self._table_length(name) for name in _TABLE_IDS}

    @cached_property
    def sources(self) -> Dict[str, np.ndarray]:
        """Resolved physical source table."""

        return self._read_group("sources")

    @cached_property
    def receivers(self) -> Dict[str, np.ndarray]:
        """Resolved receiver-aperture table."""

        return self._read_group("receivers")

    @cached_property
    def rays(self) -> Dict[str, np.ndarray]:
        """Root and spawned ray descriptor table."""

        return self._read_group("rays")

    @cached_property
    def points(self) -> Dict[str, np.ndarray]:
        """Contiguous retained path-point table."""

        return self._read_group("points")

    @cached_property
    def events(self) -> Dict[str, np.ndarray]:
        """Material, boundary, PML, and terminal event table."""

        return self._read_group("events")

    @cached_property
    def receiver_hits(self) -> Dict[str, np.ndarray]:
        """Receiver proximity-hit table."""

        return self._read_group("receiver_hits")

    @cached_property
    def status_codes(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Self-describing numeric code registries keyed by registry name."""

        result: Dict[str, Dict[str, np.ndarray]] = {}
        with h5py.File(self.hdf5_file, "r") as h5:
            for name, group in h5["status_codes"].items():
                if isinstance(group, h5py.Group):
                    result[name] = self._read_h5_group(group, normalize_vectors=False)
        return result

    def table(self, name: str) -> Dict[str, np.ndarray]:
        """Return one public record table by group name."""

        if name not in _TABLE_IDS:
            raise KeyError(f"Unknown ray table: {name}")
        return getattr(self, name)

    def ray(self, ray_id: int) -> Dict[str, Any]:
        """Return the descriptor row for one stable one-based ray id."""

        row = self._ray_row(ray_id)
        return {
            name: _record_value(values, row)
            for name, values in self.rays.items()
            if values.ndim > 0 and values.shape[0] == len(self.rays["ray_id"])
        }

    def path(self, ray_id: int) -> RayPath:
        """Return one ray descriptor and its contiguous retained points."""

        row = self._ray_row(ray_id)
        if "point_offsets" in self.rays:
            offsets = np.asarray(self.rays["point_offsets"], dtype=np.int64)
            start, stop = int(offsets[row]), int(offsets[row + 1])
        else:
            start = int(self.rays["point_offset"][row])
            stop = start + int(self.rays["point_count"][row])
        points = {
            name: values[start:stop].copy()
            for name, values in self.points.items()
            if values.ndim > 0
        }
        return RayPath(int(ray_id), self.ray(ray_id), points)

    def iter_paths(
        self,
        *,
        source_id: Optional[int] = None,
        admitted_only: bool = True,
    ) -> Iterator[RayPath]:
        """Iterate retained paths, optionally filtered by source id."""

        ids = np.asarray(self.rays["ray_id"])
        for row, ray_id in enumerate(ids):
            if source_id is not None and int(self.rays["source_id"][row]) != source_id:
                continue
            if (
                admitted_only
                and "admitted" in self.rays
                and not self.rays["admitted"][row]
            ):
                continue
            yield self.path(int(ray_id))

    def events_for_ray(self, ray_id: int) -> Dict[str, np.ndarray]:
        """Return the contiguous event span for one ray."""

        return self._ray_span(ray_id, "event", self.events)

    def receiver_hits_for_ray(self, ray_id: int) -> Dict[str, np.ndarray]:
        """Return the contiguous receiver-hit span for one ray."""

        return self._ray_span(ray_id, "receiver_hit", self.receiver_hits)

    def code_name(self, registry: str, code: int) -> str:
        """Resolve a numeric code through the file's public registry."""

        table = self.status_codes[registry]
        matches = np.flatnonzero(np.asarray(table["code"]) == code)
        if matches.size == 0:
            raise KeyError(f"Unknown {registry} code: {code}")
        return str(_decode(table["name"][int(matches[0])]))

    def units(self, group: str, dataset: str) -> Optional[str]:
        """Return a physical dataset's declared units attribute."""

        with h5py.File(self.hdf5_file, "r") as h5:
            value = h5[f"{group}/{dataset}"].attrs.get("units")
        return None if value is None else str(_decode(value))

    def plot(self, **kwargs):
        """Plot retained ray paths with the public Matplotlib helper."""

        from frequensolve.plotting.rays import plot_rays

        return plot_rays(self, **kwargs)

    def validate(self) -> None:
        """Validate the manifest, required groups, counts, and CSR offsets."""

        if self.manifest:
            if self.manifest.get("schema") != "fs-rays-1":
                raise ValueError("Ray manifest schema must be 'fs-rays-1'")
            if self.manifest.get("status") not in {"complete", "partial", "failed"}:
                raise ValueError("Ray manifest has an invalid status")
            if self.manifest.get("physics") != "acoustic":
                raise ValueError("fs-rays-1 supports acoustic physics only")
            if self.manifest.get("dimension") not in {2, 3}:
                raise ValueError("Ray manifest dimension must be 2 or 3")
        with h5py.File(self.hdf5_file, "r") as h5:
            missing = [name for name in _REQUIRED_GROUPS if name not in h5]
            if missing:
                raise ValueError(
                    f"Ray HDF5 is missing required group(s): {', '.join(missing)}"
                )
            metadata = h5["metadata"]
            schema = _decode(metadata["schema_version"][()])
            layout = _decode(metadata["layout_kind"][()])
            if schema != "fs-rays-1" or layout != "indexed_rays_v1":
                raise ValueError("Ray HDF5 is not an fs-rays-1 indexed_rays_v1 product")
            dimension = int(metadata["dimension"][()])
            if dimension not in {2, 3}:
                raise ValueError("Ray HDF5 dimension must be 2 or 3")
            for name, id_name in _TABLE_IDS.items():
                if id_name not in h5[name]:
                    raise ValueError(f"Ray HDF5 is missing /{name}/{id_name}")
                observed = len(h5[name][id_name])
                expected = self.manifest.get("counts", {}).get(name)
                if expected is not None and observed != int(expected):
                    raise ValueError(
                        f"Ray manifest count for {name} is {expected}, HDF5 has {observed}"
                    )
            n_rays = len(h5["rays/ray_id"])
            for stem, table in (
                ("point", "points"),
                ("event", "events"),
                ("receiver_hit", "receiver_hits"),
            ):
                offsets = np.asarray(h5[f"rays/{stem}_offsets"][:], dtype=np.int64)
                expected_end = len(h5[f"{table}/{_TABLE_IDS[table]}"])
                if (
                    offsets.shape != (n_rays + 1,)
                    or offsets[0] != 0
                    or np.any(np.diff(offsets) < 0)
                    or offsets[-1] != expected_end
                ):
                    raise ValueError(f"Invalid /rays/{stem}_offsets CSR vector")

    def _ray_row(self, ray_id: int) -> int:
        matches = np.flatnonzero(np.asarray(self.rays["ray_id"]) == ray_id)
        if matches.size == 0:
            raise KeyError(f"Unknown ray id: {ray_id}")
        return int(matches[0])

    def _ray_span(
        self,
        ray_id: int,
        stem: str,
        table: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        row = self._ray_row(ray_id)
        offsets = np.asarray(self.rays[f"{stem}_offsets"], dtype=np.int64)
        start, stop = int(offsets[row]), int(offsets[row + 1])
        return {
            name: values[start:stop].copy()
            for name, values in table.items()
            if values.ndim > 0
        }

    def _table_length(self, name: str) -> int:
        with h5py.File(self.hdf5_file, "r") as h5:
            return len(h5[f"{name}/{_TABLE_IDS[name]}"])

    def _read_group(
        self,
        name: str,
        *,
        normalize_vectors: bool = True,
    ) -> Dict[str, Any]:
        with h5py.File(self.hdf5_file, "r") as h5:
            return self._read_h5_group(
                h5[name],
                normalize_vectors=normalize_vectors,
            )

    def _read_h5_group(
        self,
        group: h5py.Group,
        *,
        normalize_vectors: bool,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, item in group.items():
            if not isinstance(item, h5py.Dataset):
                continue
            value = _decode(item[()])
            if (
                normalize_vectors
                and isinstance(value, np.ndarray)
                and value.ndim == 2
                and self.dimension in value.shape
            ):
                logical_axes = str(_decode(item.attrs.get("logical_axes", "")))
                if (
                    logical_axes == "record,dimension"
                    and value.shape[1] != self.dimension
                ):
                    value = value.T
                elif (
                    value.shape[0] == self.dimension
                    and value.shape[1] != self.dimension
                ):
                    value = value.T
            result[name] = value
        return result
