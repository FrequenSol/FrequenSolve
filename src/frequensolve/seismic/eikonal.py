"""Readers for Sauce's ``fs-eikonal-output-1`` first-arrival product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import h5py
import numpy as np

__all__ = ["EikonalCharacteristic", "EikonalField", "EikonalResults"]


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O", "U"}:
        return np.asarray(
            [_decode(item) for item in value.reshape(-1)], dtype=object
        ).reshape(value.shape)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar(group: h5py.Group, name: str) -> Any:
    if name not in group:
        raise ValueError(f"Eikonal HDF5 is missing {group.name}/{name}")
    return _decode(group[name][()])


@dataclass(frozen=True)
class EikonalField:
    """One source's canonical native-vertex travel-time field."""

    source_id: int
    source_name: str
    position: np.ndarray
    travel_time: np.ndarray

    @property
    def reachable(self) -> np.ndarray:
        """Mask of finite, non-sentinel first-arrival values."""

        values = np.asarray(self.travel_time, dtype=float)
        return (
            np.isfinite(values)
            & (values >= 0.0)
            & (values < np.sqrt(np.finfo(float).max))
        )


@dataclass(frozen=True)
class EikonalCharacteristic:
    """One source/receiver winning-stencil backtrack."""

    characteristic_id: int
    source_id: int
    receiver_id: int
    status: int
    ambiguous: bool
    position: np.ndarray
    travel_time: np.ndarray
    vertex_id: np.ndarray
    stencil_id: np.ndarray
    owner_cell: np.ndarray

    @property
    def successful(self) -> bool:
        """Whether Sauce reported a complete characteristic."""

        return self.status == 0 and len(self.position) >= 2


class EikonalResults:
    """Typed access to an ``fs-eikonal-output-1`` manifest and HDF5 file.

    ``path`` may name the HDF5 file, its sibling ``manifest.json``, or the
    directory containing the manifest. Vector and source-major table layouts
    are normalized to ``(record, dimension)`` and ``(source, record)``.
    """

    schema = "fs-eikonal-output-1"

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
                raise FileNotFoundError(
                    f"No fs-eikonal-output-1 manifest found in {supplied}"
                )
            self.hdf5_file = self._hdf5_from_manifest(self.manifest_file)
        elif supplied.suffix.lower() == ".json":
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
    def open(cls, path: Union[str, Path], *, validate: bool = True) -> "EikonalResults":
        """Open an Eikonal output directory, manifest, or HDF5 product."""

        return cls(path, validate=validate)

    @classmethod
    def from_job(cls, job: Any, *, validate: bool = True) -> "EikonalResults":
        """Open the result paths declared by an ``EikonalJob``."""

        return cls(
            job.eikonal_manifest_file,
            result_root=job._result_path,
            validate=validate,
        )

    @cached_property
    def manifest(self) -> Dict[str, Any]:
        """Parsed output manifest, or an empty mapping for bare HDF5."""

        if self.manifest_file is None:
            return {}
        try:
            value = json.loads(self.manifest_file.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid Eikonal manifest {self.manifest_file}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Invalid Eikonal manifest {self.manifest_file}")
        return value

    def _hdf5_from_manifest(self, manifest_file: Path) -> Path:
        try:
            manifest = json.loads(manifest_file.read_text())
            relative = Path(manifest["hdf5_file"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Eikonal manifest {manifest_file}") from exc
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
        """Scalar metadata including units and logical counts."""

        return self._read_group("metadata")

    @property
    def dimension(self) -> int:
        """Physical coordinate dimension, either two or three."""

        return int(self.metadata["dimension"])

    @property
    def status(self) -> Optional[str]:
        """Manifest completion status when a manifest is available."""

        value = self.manifest.get("status")
        return None if value is None else str(value)

    @property
    def source_count(self) -> int:
        """Number of independent source solves."""

        return int(self.metadata["source_count"])

    @property
    def receiver_count(self) -> int:
        """Number of receiver query points."""

        return int(self.metadata["receiver_count"])

    @property
    def vertex_count(self) -> int:
        """Number of canonical native travel vertices."""

        return int(self.metadata["vertex_count"])

    @property
    def characteristic_count(self) -> int:
        """Number of source/receiver characteristic rows."""

        return int(self.metadata.get("characteristic_count", 0))

    @property
    def counts(self) -> Dict[str, int]:
        """Logical source, receiver, vertex, and characteristic counts."""

        return {
            "sources": self.source_count,
            "receivers": self.receiver_count,
            "vertices": self.vertex_count,
            "characteristics": self.characteristic_count,
            "characteristic_points": int(
                self.metadata.get("characteristic_point_count", 0)
            ),
        }

    @cached_property
    def preparation(self) -> Dict[str, Any]:
        """Native-complex preparation diagnostics."""

        return self._read_group("preparation")

    @cached_property
    def sources(self) -> Dict[str, Any]:
        """Resolved source catalog with normalized positions."""

        result = self._read_group("sources")
        result["position"] = self._record_vectors(
            result["position"], self.source_count, "/sources/position"
        )
        return result

    @cached_property
    def diagnostics(self) -> Dict[str, Any]:
        """Per-source solve diagnostics."""

        return self._read_group("diagnostics")

    @cached_property
    def receivers(self) -> Dict[str, Any]:
        """Resolved receiver catalog, empty when receivers were disabled."""

        if self.receiver_count == 0:
            return {}
        result = self._read_group("receivers")
        result["position"] = self._record_vectors(
            result["position"], self.receiver_count, "/receivers/position"
        )
        return result

    @cached_property
    def receiver_times(self) -> np.ndarray:
        """Receiver first arrivals with shape ``(source, receiver)``."""

        if self.receiver_count == 0:
            return np.empty((self.source_count, 0), dtype=float)
        data = self._read_group("receiver_times")["travel_time"]
        return self._source_major(
            data, self.receiver_count, "/receiver_times/travel_time"
        )

    @cached_property
    def receiver_status(self) -> np.ndarray:
        """Receiver query status with shape ``(source, receiver)``."""

        if self.receiver_count == 0:
            return np.empty((self.source_count, 0), dtype=int)
        data = self._read_group("receiver_times")["status"]
        return self._source_major(data, self.receiver_count, "/receiver_times/status")

    @cached_property
    def receiver_ambiguous(self) -> np.ndarray:
        """Receiver tie flags with shape ``(source, receiver)``."""

        if self.receiver_count == 0:
            return np.empty((self.source_count, 0), dtype=bool)
        data = self._read_group("receiver_times")["ambiguous"]
        return self._source_major(
            data, self.receiver_count, "/receiver_times/ambiguous"
        ).astype(bool)

    @property
    def has_field(self) -> bool:
        """Whether canonical-vertex fields were retained."""

        with h5py.File(self.hdf5_file, "r") as h5:
            return "field" in h5

    @cached_property
    def field_positions(self) -> np.ndarray:
        """Canonical vertex coordinates with shape ``(vertex, dimension)``."""

        if not self.has_field:
            raise ValueError("Eikonal product does not retain vertex fields")
        data = self._read_group("field")["position"]
        return self._record_vectors(data, self.vertex_count, "/field/position")

    @cached_property
    def field_times(self) -> np.ndarray:
        """Vertex travel times with shape ``(source, vertex)``."""

        if not self.has_field:
            raise ValueError("Eikonal product does not retain vertex fields")
        data = self._read_group("field")["travel_time"]
        return self._source_major(data, self.vertex_count, "/field/travel_time")

    @property
    def has_characteristics(self) -> bool:
        """Whether winning-stencil characteristic paths were retained."""

        with h5py.File(self.hdf5_file, "r") as h5:
            return "characteristics" in h5

    @cached_property
    def characteristics(self) -> Dict[str, Any]:
        """Raw ragged characteristic tables with normalized point positions."""

        if not self.has_characteristics:
            return {}
        result = self._read_group("characteristics")
        point_count = int(self.metadata.get("characteristic_point_count", 0))
        if "position" in result:
            result["position"] = self._record_vectors(
                result["position"], point_count, "/characteristics/position"
            )
        return result

    def source_index(self, source: Union[int, str, None] = None) -> int:
        """Resolve a source id or name to a zero-based result row."""

        ids = np.asarray(self.sources["id"], dtype=int)
        if source is None:
            if len(ids) != 1:
                raise ValueError(
                    "Select a source id or name for this multi-source result"
                )
            return 0
        if isinstance(source, str):
            names = np.asarray(self.sources["name"], dtype=object)
            matches = np.flatnonzero(names == source)
        else:
            matches = np.flatnonzero(ids == int(source))
        if matches.size == 0:
            raise KeyError(f"Unknown Eikonal source: {source!r}")
        return int(matches[0])

    def field(self, source: Union[int, str, None] = None) -> EikonalField:
        """Return one source's native-vertex first-arrival field."""

        row = self.source_index(source)
        return EikonalField(
            source_id=int(self.sources["id"][row]),
            source_name=str(self.sources["name"][row]),
            position=self.field_positions.copy(),
            travel_time=self.field_times[row].copy(),
        )

    def receiver_times_for_source(
        self, source: Union[int, str, None] = None
    ) -> np.ndarray:
        """Return receiver first arrivals for one source."""

        return self.receiver_times[self.source_index(source)].copy()

    def characteristic(self, characteristic_id: int) -> EikonalCharacteristic:
        """Return one one-based characteristic and its contiguous points."""

        row = int(characteristic_id) - 1
        if row < 0 or row >= self.characteristic_count:
            raise KeyError(f"Unknown characteristic id: {characteristic_id}")
        table = self.characteristics
        offsets = np.asarray(table["offset"], dtype=np.int64)
        start, stop = int(offsets[row]), int(offsets[row + 1])

        def points(name: str, dtype: Any = None) -> np.ndarray:
            values = np.asarray(table.get(name, []), dtype=dtype)
            return values[start:stop].copy()

        return EikonalCharacteristic(
            characteristic_id=row + 1,
            source_id=int(table["source_id"][row]),
            receiver_id=int(table["receiver_id"][row]),
            status=int(table["status"][row]),
            ambiguous=bool(table["ambiguous"][row]),
            position=points("position").reshape(-1, self.dimension),
            travel_time=points("travel_time", float),
            vertex_id=points("vertex_id", int),
            stencil_id=points("stencil_id", int),
            owner_cell=points("owner_cell", int),
        )

    def iter_characteristics(
        self,
        *,
        source_id: Optional[int] = None,
        receiver_id: Optional[int] = None,
        successful_only: bool = True,
    ) -> Iterator[EikonalCharacteristic]:
        """Iterate retained characteristic paths with optional id filters."""

        if not self.has_characteristics:
            return
        table = self.characteristics
        for row in range(self.characteristic_count):
            if source_id is not None and int(table["source_id"][row]) != source_id:
                continue
            if (
                receiver_id is not None
                and int(table["receiver_id"][row]) != receiver_id
            ):
                continue
            path = self.characteristic(row + 1)
            if successful_only and not path.successful:
                continue
            yield path

    def plot(self, **kwargs: Any) -> Any:
        """Plot fields and characteristics with the public Matplotlib helper."""

        from frequensolve.plotting.eikonal import plot_eikonal

        return plot_eikonal(self, **kwargs)

    def validate(self) -> None:
        """Validate schemas, logical counts, shapes, and ragged offsets."""

        if self.manifest:
            if self.manifest.get("schema") != self.schema:
                raise ValueError(f"Eikonal manifest schema must be {self.schema!r}")
            if self.manifest.get("status") != "complete":
                raise ValueError("Eikonal manifest status must be 'complete'")
            required_manifest = (
                "hdf5_file",
                "source_count",
                "receiver_count",
                "vertex_count",
            )
            missing = [name for name in required_manifest if name not in self.manifest]
            if missing:
                raise ValueError(
                    "Eikonal manifest is missing required field(s): "
                    + ", ".join(missing)
                )
        with h5py.File(self.hdf5_file, "r") as h5:
            required = ("metadata", "preparation", "sources", "diagnostics")
            missing = [name for name in required if name not in h5]
            if missing:
                raise ValueError(
                    f"Eikonal HDF5 is missing required group(s): {', '.join(missing)}"
                )
            schema = (
                _decode(h5["schema"][()])
                if "schema" in h5
                else _scalar(h5["metadata"], "schema")
            )
            if schema != self.schema:
                raise ValueError(f"Eikonal HDF5 schema must be {self.schema!r}")
            metadata = h5["metadata"]
            self._require_datasets(
                metadata,
                (
                    "schema",
                    "workflow",
                    "physics",
                    "geometry_policy",
                    "coordinate_units",
                    "travel_time_units",
                    "submitted_config_json",
                    "dimension",
                    "source_count",
                    "receiver_count",
                    "vertex_count",
                    "characteristic_count",
                    "characteristic_point_count",
                ),
            )
            if _scalar(metadata, "workflow") != "eikonal":
                raise ValueError("Eikonal HDF5 workflow must be 'eikonal'")
            if _scalar(metadata, "physics") != "acoustic":
                raise ValueError("Eikonal HDF5 physics must be 'acoustic'")
            dimension = int(_scalar(metadata, "dimension"))
            if dimension not in {2, 3}:
                raise ValueError("Eikonal HDF5 dimension must be 2 or 3")
            source_count = int(_scalar(metadata, "source_count"))
            receiver_count = int(_scalar(metadata, "receiver_count"))
            vertex_count = int(_scalar(metadata, "vertex_count"))
            characteristic_count = int(_scalar(metadata, "characteristic_count"))
            point_count = int(_scalar(metadata, "characteristic_point_count"))
            if source_count < 1 or receiver_count < 0 or vertex_count < 1:
                raise ValueError("Eikonal HDF5 contains invalid logical counts")
            for key, expected in (
                ("source_count", source_count),
                ("receiver_count", receiver_count),
                ("vertex_count", vertex_count),
                ("characteristic_point_count", point_count),
            ):
                manifest_value = self.manifest.get(key)
                if manifest_value is not None and int(manifest_value) != expected:
                    raise ValueError(
                        f"Eikonal manifest {key} is {manifest_value}, HDF5 has {expected}"
                    )
            self._require_datasets(
                h5["sources"], ("id", "owner_cell", "position", "name")
            )
            self._require_datasets(
                h5["diagnostics"],
                (
                    "status",
                    "wave_count",
                    "update_count",
                    "active_peak",
                    "reachable_vertices",
                    "ambiguous_vertices",
                    "invalid_candidate_count",
                    "wave_vertex_scan_count",
                    "queue_visit_count",
                    "queue_admission_count",
                    "residual_max",
                ),
            )
            self._require_datasets(
                h5["preparation"],
                (
                    "trace_vertex_count",
                    "trace_patch_count",
                    "physical_cell_count",
                    "directional_sample_count",
                    "full_stencil_count",
                    "rejected_geometry_count",
                    "minimum_full_stencil_quality",
                    "minimum_full_stencil_quality_threshold",
                    "prepared_bytes",
                    "total_seconds",
                ),
            )
            for name in ("id", "owner_cell", "name"):
                self._validate_table_length(h5["sources"], name, source_count)
            for name in (
                "status",
                "wave_count",
                "update_count",
                "active_peak",
                "reachable_vertices",
                "ambiguous_vertices",
                "invalid_candidate_count",
                "wave_vertex_scan_count",
                "queue_visit_count",
                "queue_admission_count",
                "residual_max",
            ):
                self._validate_table_length(h5["diagnostics"], name, source_count)
            self._validate_vector_shape(
                h5["sources/position"].shape,
                source_count,
                dimension,
                "/sources/position",
            )
            if receiver_count:
                for group in ("receivers", "receiver_times"):
                    if group not in h5:
                        raise ValueError(f"Eikonal HDF5 is missing /{group}")
                self._require_datasets(
                    h5["receivers"],
                    ("id", "group_id", "point_id", "position", "name", "group_name"),
                )
                self._require_datasets(
                    h5["receiver_times"], ("travel_time", "status", "ambiguous")
                )
                for name in ("id", "group_id", "point_id", "name", "group_name"):
                    self._validate_table_length(h5["receivers"], name, receiver_count)
                self._validate_vector_shape(
                    h5["receivers/position"].shape,
                    receiver_count,
                    dimension,
                    "/receivers/position",
                )
                for name in ("travel_time", "status", "ambiguous"):
                    self._validate_source_major_shape(
                        h5[f"receiver_times/{name}"].shape,
                        source_count,
                        receiver_count,
                        f"/receiver_times/{name}",
                    )
            field_retained = bool(self.manifest.get("field_retained", "field" in h5))
            if "field_retained" in self.manifest and field_retained != ("field" in h5):
                raise ValueError("Eikonal manifest field_retained disagrees with HDF5")
            if field_retained:
                if "field" not in h5:
                    raise ValueError(
                        "Manifest declares a retained field but /field is absent"
                    )
                self._require_datasets(h5["field"], ("position", "travel_time"))
                self._validate_vector_shape(
                    h5["field/position"].shape,
                    vertex_count,
                    dimension,
                    "/field/position",
                )
                self._validate_source_major_shape(
                    h5["field/travel_time"].shape,
                    source_count,
                    vertex_count,
                    "/field/travel_time",
                )
            characteristics_retained = bool(
                self.manifest.get("characteristics_retained", "characteristics" in h5)
            )
            if (
                "characteristics_retained" in self.manifest
                and characteristics_retained != ("characteristics" in h5)
            ):
                raise ValueError(
                    "Eikonal manifest characteristics_retained disagrees with HDF5"
                )
            if characteristics_retained:
                if "characteristics" not in h5:
                    raise ValueError(
                        "Manifest declares characteristics but /characteristics is absent"
                    )
                table = h5["characteristics"]
                self._require_datasets(
                    table,
                    ("offset", "source_id", "receiver_id", "status", "ambiguous"),
                )
                offsets = np.asarray(table["offset"][:], dtype=np.int64)
                if (
                    offsets.shape != (characteristic_count + 1,)
                    or offsets[0] != 0
                    or np.any(np.diff(offsets) < 0)
                    or offsets[-1] != point_count
                ):
                    raise ValueError("Invalid /characteristics/offset CSR vector")
                for name in ("source_id", "receiver_id", "status", "ambiguous"):
                    self._validate_table_length(table, name, characteristic_count)
                if point_count:
                    self._validate_vector_shape(
                        table["position"].shape,
                        point_count,
                        dimension,
                        "/characteristics/position",
                    )
                    for name in (
                        "travel_time",
                        "vertex_id",
                        "stencil_id",
                        "owner_cell",
                    ):
                        self._validate_table_length(table, name, point_count)

    @staticmethod
    def _require_datasets(group: h5py.Group, names: tuple[str, ...]) -> None:
        missing = [name for name in names if name not in group]
        if missing:
            paths = ", ".join(f"{group.name}/{name}" for name in missing)
            raise ValueError(f"Eikonal HDF5 is missing required dataset(s): {paths}")

    @staticmethod
    def _validate_table_length(group: h5py.Group, name: str, expected: int) -> None:
        if name not in group or group[name].shape != (expected,):
            observed = None if name not in group else group[name].shape
            raise ValueError(
                f"Eikonal HDF5 {group.name}/{name} has shape {observed}, "
                f"expected ({expected},)"
            )

    @staticmethod
    def _validate_vector_shape(
        shape: tuple[int, ...], count: int, dimension: int, name: str
    ) -> None:
        if shape not in {(count, dimension), (dimension, count)}:
            raise ValueError(
                f"Eikonal HDF5 {name} has shape {shape}, expected "
                f"({count}, {dimension})"
            )

    @staticmethod
    def _validate_source_major_shape(
        shape: tuple[int, ...], sources: int, records: int, name: str
    ) -> None:
        if shape not in {(sources, records), (records, sources)}:
            raise ValueError(
                f"Eikonal HDF5 {name} has shape {shape}, expected "
                f"({sources}, {records})"
            )

    def _record_vectors(self, value: Any, count: int, name: str) -> np.ndarray:
        array = np.asarray(value)
        self._validate_vector_shape(array.shape, count, self.dimension, name)
        if array.shape == (self.dimension, count) and array.shape != (
            count,
            self.dimension,
        ):
            array = array.T
        return array

    def _source_major(self, value: Any, records: int, name: str) -> np.ndarray:
        array = np.asarray(value)
        self._validate_source_major_shape(array.shape, self.source_count, records, name)
        if array.shape == (records, self.source_count) and array.shape != (
            self.source_count,
            records,
        ):
            array = array.T
        return array

    def _read_group(self, name: str) -> Dict[str, Any]:
        with h5py.File(self.hdf5_file, "r") as h5:
            if name not in h5:
                raise ValueError(f"Eikonal HDF5 does not contain /{name}")
            return {
                key: _decode(item[()])
                for key, item in h5[name].items()
                if isinstance(item, h5py.Dataset)
            }
