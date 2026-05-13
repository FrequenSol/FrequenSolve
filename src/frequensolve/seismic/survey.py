"""High-level seismic survey authoring helpers."""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from frequensolve.geometry.frame import CoordinateValue
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverDevice
from frequensolve.seismic.sparse_survey import SparseSurvey
from frequensolve.units import is_quantity, value_and_units_to_fs

__all__ = ["Survey"]


_RELATION_DTYPE = np.dtype(
    [
        ("source_id", np.int64),
        ("receiver_id", np.int64),
        ("field_record", np.int64),
        ("channel_number", np.int64),
        ("relation_row", np.int64),
    ]
)

_TRACE_DTYPE = np.dtype(
    [
        ("trace_id", np.int64),
        ("source_id", np.int64),
        ("receiver_id", np.int64),
        ("recv_pos_id", np.int64),
        ("component_id", np.int64),
        ("component", np.int64),
        ("channel_number", np.int64),
        ("field_record", np.int64),
        ("active", np.bool_),
        ("weight", np.float64),
    ]
)

_TRACE_SCHEMA = "fs_seismic_trace_store_v1"
_TRACE_SPECIAL_DATASETS = {"frequency", "laplace", "task_id"}


def _coords_array(values: Any, name: str) -> np.ndarray:
    if isinstance(values, CoordinateValue):
        values = values.value
    if is_quantity(values):
        values = values.magnitude
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (n, 2) or (n, 3)")
    return array


def _id_array(values: Optional[Iterable[int]], size: int, name: str) -> np.ndarray:
    if values is None:
        return np.arange(1, size + 1, dtype=np.int64)
    array = np.asarray(list(values), dtype=np.int64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have exactly {size} entries")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} must be unique")
    return array


def _string_array(
    values: Optional[Iterable[str]], size: int, prefix: str
) -> np.ndarray:
    if values is None:
        return np.asarray([f"{prefix}_{i}" for i in range(1, size + 1)], dtype=object)
    array = np.asarray([str(value) for value in values], dtype=object)
    if array.shape != (size,):
        raise ValueError(f"{prefix} names must have exactly {size} entries")
    return array


def _field_str(line: str, i1: int, i2: int) -> str:
    if len(line) < i1:
        return ""
    return line[i1 - 1 : min(len(line), i2)].strip()


def _field_int(line: str, i1: int, i2: int, default: int) -> int:
    text = _field_str(line, i1, i2)
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _field_real(line: str, i1: int, i2: int, default: float) -> float:
    text = _field_str(line, i1, i2)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _as_ids(values: Optional[Union[int, Sequence[int], np.ndarray]]) -> Optional[set]:
    if values is None:
        return None
    if isinstance(values, (int, np.integer)):
        return {int(values)}
    return {int(value) for value in values}


def _unique_sorted(values: Iterable[Any]) -> list[int]:
    return sorted({int(value) for value in values})


def _reject_frame_kwargs(kwargs: Mapping[str, Any]) -> None:
    deprecated = {"frame", "source_frame", "receiver_frame"} & set(kwargs)
    if deprecated:
        names = ", ".join(sorted(deprecated))
        raise TypeError(
            f"Survey no longer supports {names}; source and receiver coordinates are physical"
        )


def _coordinate_value_or_array(
    values: np.ndarray, *, units: Optional[Any], system: Optional[str]
) -> Union[np.ndarray, CoordinateValue]:
    if units is None and system is None:
        return values
    return CoordinateValue(values, units=units, system=system)


def _decode_h5_strings(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [
                item.decode("utf-8", "ignore") if isinstance(item, bytes) else str(item)
                for item in values.ravel()
            ],
            dtype=object,
        ).reshape(values.shape)
    return values


def _h5_string_list(values: Any) -> list[str]:
    array = _decode_h5_strings(values)
    return [str(item) for item in np.asarray(array).ravel()]


def _h5_first_string(values: Any, default: Optional[str] = None) -> Optional[str]:
    strings = _h5_string_list(values)
    if not strings:
        return default
    text = strings[0]
    return text if text else default


def _h5_dataset_strings(group: Any, name: str, size: int, prefix: str) -> np.ndarray:
    if name not in group:
        return _string_array(None, size, prefix)
    values = _h5_string_list(group[name][()])
    return _string_array(values, size, prefix)


def _h5_dataset_ints(group: Any, name: str, default: np.ndarray) -> np.ndarray:
    if name not in group:
        return np.asarray(default, dtype=np.int64)
    return np.asarray(group[name][()], dtype=np.int64)


def _h5_dataset_float_array(group: Any, name: str, default: np.ndarray) -> np.ndarray:
    if name not in group:
        return np.asarray(default, dtype=float)
    return np.asarray(group[name][()], dtype=float)


def _h5_attr_strings(obj: Any, name: str) -> list[str]:
    if name not in obj.attrs:
        return []
    return _h5_string_list(obj.attrs[name])


def _h5_attr_ints(obj: Any, name: str) -> Optional[np.ndarray]:
    if name not in obj.attrs:
        return None
    try:
        return np.asarray(obj.attrs[name], dtype=np.int64)
    except (TypeError, ValueError):
        return None


def _catalog_indices(ids: np.ndarray, wanted: Iterable[int], n_rows: int) -> list[int]:
    id_to_index = (
        {int(value): index for index, value in enumerate(ids)}
        if len(ids) == n_rows
        else {}
    )
    indices = []
    for value in wanted:
        value = int(value)
        if value in id_to_index:
            indices.append(id_to_index[value])
            continue
        fallback = value - 1
        if 0 <= fallback < n_rows:
            indices.append(fallback)
    return indices


def _h5_dataset_string_values(group: Any, name: str) -> list[str]:
    if name not in group:
        return []
    return _h5_string_list(group[name][()])


def _clean_h5_path(path: str) -> str:
    return str(path).strip().lstrip("/")


def _group_spec_matches(spec: Mapping[str, Any], group: Optional[str]) -> bool:
    if group is None:
        return True
    group = str(group).strip().lstrip("/")
    names = {
        str(spec.get("group_name", "")).strip().lstrip("/"),
        str(spec.get("dataset", "")).strip().lstrip("/"),
        str(spec.get("dataset_path", "")).strip().lstrip("/"),
    }
    return group in names


@dataclass
class Survey:
    """Source/receiver survey geometry that can populate an Acquisition."""

    name: str
    kind: str
    sources: np.ndarray
    receivers: np.ndarray
    source_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    receiver_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    source_names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    receiver_names: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=object)
    )
    relations: np.ndarray = field(default_factory=lambda: np.empty(0, _RELATION_DTYPE))
    source_file: Optional[Union[str, Path]] = None
    receiver_file: Optional[Union[str, Path]] = None
    relation_file: Optional[Union[str, Path]] = None
    offset_domain_spec: Optional[Dict[str, Any]] = None
    source_units: Optional[Any] = None
    receiver_units: Optional[Any] = None
    source_system: Optional[str] = None
    receiver_system: Optional[str] = None
    trace_tables: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.sources, CoordinateValue):
            if self.source_units is None:
                self.source_units = self.sources.units
            if self.source_system is None:
                self.source_system = self.sources.system
            self.sources = self.sources.value
        if is_quantity(self.sources):
            if self.source_units is None:
                self.source_units = self.sources.units
            self.sources = self.sources.magnitude
        if isinstance(self.receivers, CoordinateValue):
            if self.receiver_units is None:
                self.receiver_units = self.receivers.units
            if self.receiver_system is None:
                self.receiver_system = self.receivers.system
            self.receivers = self.receivers.value
        if is_quantity(self.receivers):
            if self.receiver_units is None:
                self.receiver_units = self.receivers.units
            self.receivers = self.receivers.magnitude
        self.sources = _coords_array(self.sources, "sources")
        self.receivers = _coords_array(self.receivers, "receivers")
        if self.sources.shape[1] != self.receivers.shape[1]:
            raise ValueError("sources and receivers must have the same dimension")
        if self.source_ids.size == 0:
            self.source_ids = _id_array(None, len(self.sources), "source_ids")
        else:
            self.source_ids = _id_array(
                self.source_ids, len(self.sources), "source_ids"
            )
        if self.receiver_ids.size == 0:
            self.receiver_ids = _id_array(None, len(self.receivers), "receiver_ids")
        else:
            self.receiver_ids = _id_array(
                self.receiver_ids, len(self.receivers), "receiver_ids"
            )
        if self.source_names.size == 0:
            self.source_names = _string_array(None, len(self.sources), "source")
        else:
            self.source_names = _string_array(
                self.source_names, len(self.sources), "source"
            )
        if self.receiver_names.size == 0:
            self.receiver_names = _string_array(None, len(self.receivers), "receiver")
        else:
            self.receiver_names = _string_array(
                self.receiver_names, len(self.receivers), "receiver"
            )
        self.relations = np.asarray(self.relations, dtype=_RELATION_DTYPE)
        self.trace_tables = {
            str(name): np.asarray(table, dtype=_TRACE_DTYPE)
            for name, table in self.trace_tables.items()
        }

    @classmethod
    def dense(
        cls,
        name: str = "survey",
        *,
        sources: Sequence[Sequence[float]],
        receivers: Sequence[Sequence[float]],
        source_ids: Optional[Iterable[int]] = None,
        receiver_ids: Optional[Iterable[int]] = None,
        source_names: Optional[Iterable[str]] = None,
        receiver_names: Optional[Iterable[str]] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        source_units: Optional[Any] = None,
        receiver_units: Optional[Any] = None,
        source_system: Optional[str] = None,
        receiver_system: Optional[str] = None,
    ) -> "Survey":
        if isinstance(sources, CoordinateValue):
            if source_units is None:
                source_units = sources.units
            if source_system is None:
                source_system = sources.system
        elif is_quantity(sources) and source_units is None:
            source_units = sources.units
        if isinstance(receivers, CoordinateValue):
            if receiver_units is None:
                receiver_units = receivers.units
            if receiver_system is None:
                receiver_system = receivers.system
        elif is_quantity(receivers) and receiver_units is None:
            receiver_units = receivers.units
        source_coords = _coords_array(sources, "sources")
        receiver_coords = _coords_array(receivers, "receivers")
        return cls(
            name=name,
            kind="Dense",
            sources=source_coords,
            receivers=receiver_coords,
            source_ids=_id_array(source_ids, len(source_coords), "source_ids"),
            receiver_ids=_id_array(receiver_ids, len(receiver_coords), "receiver_ids"),
            source_names=_string_array(source_names, len(source_coords), "source"),
            receiver_names=_string_array(
                receiver_names, len(receiver_coords), "receiver"
            ),
            source_units=source_units if source_units is not None else units,
            receiver_units=receiver_units if receiver_units is not None else units,
            source_system=source_system if source_system is not None else system,
            receiver_system=receiver_system if receiver_system is not None else system,
        )

    @classmethod
    def offset_domain(
        cls,
        name: str = "survey",
        *,
        sources: Sequence[Sequence[float]],
        receivers: Sequence[Sequence[float]],
        min: Optional[float] = None,
        max: Optional[float] = None,
        metric: str = "horizontal",
        axis: Optional[Sequence[float]] = None,
        absolute: bool = True,
        source_ids: Optional[Iterable[int]] = None,
        receiver_ids: Optional[Iterable[int]] = None,
        source_names: Optional[Iterable[str]] = None,
        receiver_names: Optional[Iterable[str]] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        source_units: Optional[Any] = None,
        receiver_units: Optional[Any] = None,
        source_system: Optional[str] = None,
        receiver_system: Optional[str] = None,
    ) -> "Survey":
        if isinstance(sources, CoordinateValue):
            if source_units is None:
                source_units = sources.units
            if source_system is None:
                source_system = sources.system
        elif is_quantity(sources) and source_units is None:
            source_units = sources.units
        if isinstance(receivers, CoordinateValue):
            if receiver_units is None:
                receiver_units = receivers.units
            if receiver_system is None:
                receiver_system = receivers.system
        elif is_quantity(receivers) and receiver_units is None:
            receiver_units = receivers.units
        source_coords = _coords_array(sources, "sources")
        receiver_coords = _coords_array(receivers, "receivers")
        domain: Dict[str, Any] = {"metric": metric, "absolute": bool(absolute)}
        if min is not None:
            domain["min"] = value_and_units_to_fs(min)
        if max is not None:
            domain["max"] = value_and_units_to_fs(max)
        if axis is not None:
            domain["axis"] = list(axis)
        return cls(
            name=name,
            kind="OffsetDomain",
            sources=source_coords,
            receivers=receiver_coords,
            source_ids=_id_array(source_ids, len(source_coords), "source_ids"),
            receiver_ids=_id_array(receiver_ids, len(receiver_coords), "receiver_ids"),
            source_names=_string_array(source_names, len(source_coords), "source"),
            receiver_names=_string_array(
                receiver_names, len(receiver_coords), "receiver"
            ),
            offset_domain_spec=domain,
            source_units=source_units if source_units is not None else units,
            receiver_units=receiver_units if receiver_units is not None else units,
            source_system=source_system if source_system is not None else system,
            receiver_system=receiver_system if receiver_system is not None else system,
        )

    @classmethod
    def from_sps(
        cls,
        source_file: Union[str, Path],
        receiver_file: Union[str, Path],
        relation_file: Union[str, Path],
        *,
        name: str = "survey",
        dimension: int = 2,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        source_units: Optional[Any] = None,
        receiver_units: Optional[Any] = None,
        source_system: Optional[str] = None,
        receiver_system: Optional[str] = None,
    ) -> "Survey":
        sources, source_ids, source_names, source_keys = cls._read_sps_points(
            source_file, "S", dimension
        )
        receivers, receiver_ids, receiver_names, receiver_keys = cls._read_sps_points(
            receiver_file, "R", dimension
        )
        relations = cls._read_sps_relations(
            relation_file,
            source_keys=source_keys,
            receiver_keys=receiver_keys,
        )
        return cls(
            name=name,
            kind="SPSFiles",
            sources=sources,
            receivers=receivers,
            source_ids=source_ids,
            receiver_ids=receiver_ids,
            source_names=source_names,
            receiver_names=receiver_names,
            relations=relations,
            source_file=source_file,
            receiver_file=receiver_file,
            relation_file=relation_file,
            source_units=source_units if source_units is not None else units,
            receiver_units=receiver_units if receiver_units is not None else units,
            source_system=source_system if source_system is not None else system,
            receiver_system=receiver_system if receiver_system is not None else system,
        )

    @classmethod
    def load(
        cls,
        source: Any,
        receiver_file: Optional[Union[str, Path]] = None,
        relation_file: Optional[Union[str, Path]] = None,
        *,
        group: Optional[str] = None,
        name: str = "survey",
        dimension: int = 2,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        source_units: Optional[Any] = None,
        receiver_units: Optional[Any] = None,
        source_system: Optional[str] = None,
        receiver_system: Optional[str] = None,
    ) -> "Survey":
        """Load a survey from an authored object, SPS files, traces, or run result.

        ``source`` may be an existing ``Survey``, a trace HDF5 file, a result
        directory, a ``TraceDataset``/``TraceManifest``, a completed run result,
        or an SPS triplet supplied either as ``(sps, spr, spx)`` or as the first
        three positional arguments.
        """

        if isinstance(source, Survey):
            return source

        if receiver_file is not None or relation_file is not None:
            if receiver_file is None or relation_file is None:
                raise TypeError(
                    "SPS loading requires source, receiver, and relation files"
                )
            return cls.from_sps(
                source,
                receiver_file,
                relation_file,
                name=name,
                dimension=dimension,
                units=units,
                system=system,
                source_units=source_units,
                receiver_units=receiver_units,
                source_system=source_system,
                receiver_system=receiver_system,
            )

        if isinstance(source, ABCMapping):
            if {"source_file", "receiver_file", "relation_file"} <= set(source):
                return cls.from_sps(
                    source["source_file"],
                    source["receiver_file"],
                    source["relation_file"],
                    name=name,
                    dimension=dimension,
                    units=units,
                    system=system,
                    source_units=source_units,
                    receiver_units=receiver_units,
                    source_system=source_system,
                    receiver_system=receiver_system,
                )
            raise TypeError("Survey.load mapping input must describe SPS files")

        if (
            isinstance(source, ABCSequence)
            and not isinstance(source, (str, bytes, Path))
            and len(source) == 3
        ):
            return cls.from_sps(
                source[0],
                source[1],
                source[2],
                name=name,
                dimension=dimension,
                units=units,
                system=system,
                source_units=source_units,
                receiver_units=receiver_units,
                source_system=source_system,
                receiver_system=receiver_system,
            )

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                from frequensolve.simulation.artifacts import RunMetadata

                return cls.from_result(
                    type("_SurveyRun", (), {"run_metadata": RunMetadata.read(path)})(),
                    group=group,
                    name=name,
                )
            if cls._is_trace_store(path) or path.suffix.lower() in {
                ".h5",
                ".hdf5",
                ".hdf",
            }:
                return cls.from_trace_file(path, group=group, name=name)
            raise ValueError(f"Cannot infer survey format from path: {path}")

        if any(hasattr(source, attr) for attr in ("paths", "files", "manifest")):
            return cls.from_trace_dataset(source, group=group, name=name)

        if any(
            hasattr(source, attr)
            for attr in ("run_metadata", "trace_manifest", "output_files", "job")
        ):
            return cls.from_result(source, group=group, name=name)

        raise TypeError(f"Cannot load Survey from {type(source).__name__}")

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        group: Optional[str] = None,
        name: str = "survey",
    ) -> "Survey":
        """Load the resolved solver survey from a completed run result."""

        return cls.from_trace_file(
            cls._trace_file_from_result(result),
            group=group,
            name=name,
        )

    @classmethod
    def from_trace_dataset(
        cls,
        traces: Any,
        *,
        group: Optional[str] = None,
        name: str = "survey",
    ) -> "Survey":
        """Load the resolved survey from a ``TraceDataset`` or manifest."""

        files = getattr(traces, "paths", None)
        if files is None:
            files = getattr(traces, "files", None)
        if files is None and hasattr(traces, "manifest"):
            files = getattr(traces.manifest, "files", None)
        if files is None:
            raise TypeError("from_trace_dataset requires trace files or a manifest")
        for file in files:
            path = Path(file)
            if path.exists() and cls._is_trace_store(path):
                return cls.from_trace_file(path, group=group, name=name)
        if hasattr(traces, "consolidate"):
            path = Path(traces.consolidate())
            if path.exists() and cls._is_trace_store(path):
                return cls.from_trace_file(path, group=group, name=name)
        raise FileNotFoundError("No trace HDF5 file with survey metadata was found")

    @classmethod
    def from_trace_file(
        cls,
        file: Union[str, Path],
        *,
        group: Optional[str] = None,
        name: str = "survey",
    ) -> "Survey":
        """Load a resolved survey from a fast solver ``fs_seismic_trace_store_v1`` HDF5 file."""

        try:
            import h5py
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "Survey trace-store loading",
                extra="parallel",
                dependencies=("h5py",),
                error=exc,
            ) from exc

        path = Path(file)
        with h5py.File(path, "r") as h5:
            if "survey" not in h5:
                raise ValueError(f"Trace file has no /survey group: {path}")
            survey_group = h5["survey"]
            schema = _h5_first_string(survey_group["schema_version"][()])
            if schema != _TRACE_SCHEMA:
                raise ValueError(
                    f"Unsupported survey schema {schema!r}; expected {_TRACE_SCHEMA!r}"
                )

            source_ids, source_names, sources, source_units, source_system = (
                cls._read_solver_point_catalog(
                    survey_group,
                    "sources",
                    id_name="source_id",
                    name_name="source_name",
                    name_prefix="source",
                )
            )
            source_field_records = cls._read_solver_source_field_records(survey_group)
            group_specs = cls._read_solver_group_specs(h5, group=group)
            if not group_specs:
                raise ValueError(f"No survey receiver groups found in {path}")
            (
                receiver_ids,
                receiver_names,
                receivers,
                receiver_units,
                receiver_system,
                receiver_id_maps,
            ) = cls._read_solver_receiver_catalogs(h5, group_specs)
            component_ids = cls._read_solver_component_ids(survey_group)
            trace_tables = cls._read_solver_trace_tables(
                h5,
                group_specs=group_specs,
                receiver_id_maps=receiver_id_maps,
                source_ids=source_ids,
                source_field_records=source_field_records,
                component_ids=component_ids,
            )
            if not trace_tables:
                raise ValueError(f"No survey trace tables found in {path}")

            used_sources = _unique_sorted(
                row["source_id"]
                for table in trace_tables.values()
                for row in table
                if int(row["source_id"]) > 0
            )
            used_receivers = _unique_sorted(
                row["receiver_id"]
                for table in trace_tables.values()
                for row in table
                if int(row["receiver_id"]) > 0
            )

            source_idx = _catalog_indices(source_ids, used_sources, len(sources))
            receiver_idx = _catalog_indices(
                receiver_ids, used_receivers, len(receivers)
            )
            if not source_idx or not receiver_idx:
                raise ValueError("Survey catalogs do not cover the resolved trace IDs")

            relations = cls._relations_from_trace_tables(trace_tables)

        return cls(
            name=name,
            kind="SolverTraceStore",
            sources=sources[source_idx],
            receivers=receivers[receiver_idx],
            source_ids=(
                source_ids[source_idx]
                if len(source_ids) == len(sources)
                else np.asarray(used_sources, dtype=np.int64)
            ),
            receiver_ids=(
                receiver_ids[receiver_idx]
                if len(receiver_ids) == len(receivers)
                else np.asarray(used_receivers, dtype=np.int64)
            ),
            source_names=(
                source_names[source_idx]
                if len(source_names) == len(sources)
                else _string_array(None, len(source_idx), "source")
            ),
            receiver_names=(
                receiver_names[receiver_idx]
                if len(receiver_names) == len(receivers)
                else _string_array(None, len(receiver_idx), "receiver")
            ),
            relations=relations,
            source_units=source_units,
            receiver_units=receiver_units,
            source_system=source_system,
            receiver_system=receiver_system,
            trace_tables=trace_tables,
        )

    @classmethod
    def from_solver_output(
        cls,
        file: Union[str, Path],
        *,
        group: Optional[str] = None,
        name: str = "survey",
    ) -> "Survey":
        """Load a resolved survey from a solver trace-store output file."""

        return cls.from_trace_file(file, group=group, name=name)

    @staticmethod
    def _trace_file_from_result(result: Any) -> Path:
        metadata = getattr(result, "run_metadata", None)
        if metadata is None and getattr(result, "job", None) is not None:
            metadata = getattr(result.job, "run_metadata", None)
        if metadata is not None:
            for artifact in getattr(metadata, "artifacts", []):
                if artifact.schema == _TRACE_SCHEMA and Path(artifact.path).exists():
                    return Path(artifact.path)

        manifest = getattr(result, "trace_manifest", None)
        if manifest is None and getattr(result, "job", None) is not None:
            manifest = getattr(result.job, "trace_manifest", None)
        if manifest is not None:
            for file in getattr(manifest, "files", []):
                path = Path(file)
                if path.exists() and Survey._is_trace_store(path):
                    return path

        if hasattr(result, "output_files"):
            for file in result.output_files(kind="hdf5", existing=True):
                path = Path(file)
                if Survey._is_trace_store(path):
                    return path

        raise FileNotFoundError("No fs_seismic_trace_store_v1 HDF5 output was found")

    @staticmethod
    def _is_trace_store(path: Union[str, Path]) -> bool:
        try:
            import h5py
        except ModuleNotFoundError:
            return False

        try:
            with h5py.File(path, "r") as h5:
                if "survey/schema_version" not in h5:
                    return False
                return (
                    _h5_first_string(h5["survey/schema_version"][()]) == _TRACE_SCHEMA
                )
        except OSError:
            return False

    @staticmethod
    def _read_solver_point_catalog(
        survey_group: Any,
        catalog_name: str,
        *,
        id_name: str,
        name_name: str,
        name_prefix: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str], Optional[str]]:
        if catalog_name not in survey_group:
            raise ValueError(f"Trace-store survey is missing /survey/{catalog_name}")
        catalog = survey_group[catalog_name]
        if "coordinates" not in catalog:
            raise ValueError(f"/survey/{catalog_name} is missing coordinates")
        coordinates = np.asarray(catalog["coordinates"][()], dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
            raise ValueError(f"/survey/{catalog_name}/coordinates must be (n, 2|3)")
        size = coordinates.shape[0]
        ids = _h5_dataset_ints(catalog, id_name, np.arange(1, size + 1))
        names = _h5_dataset_strings(catalog, name_name, len(ids), name_prefix)
        coord_dset = catalog["coordinates"]
        units = _h5_first_string(coord_dset.attrs.get("units", []))
        system = _h5_first_string(coord_dset.attrs.get("coordinate_system", []))
        return ids, names, coordinates, units, system

    @staticmethod
    def _read_solver_source_field_records(survey_group: Any) -> Dict[int, int]:
        if "sources" not in survey_group:
            return {}
        sources = survey_group["sources"]
        if "source_id" not in sources or "field_record" not in sources:
            return {}
        source_ids = np.asarray(sources["source_id"][()], dtype=np.int64)
        field_records = np.asarray(sources["field_record"][()], dtype=np.int64)
        return {
            int(source_id): int(field_record)
            for source_id, field_record in zip(source_ids, field_records)
        }

    @staticmethod
    def _read_solver_group_specs(
        h5: Any,
        *,
        group: Optional[str],
    ) -> list[Dict[str, Any]]:
        specs: list[Dict[str, Any]] = []
        survey_group = h5["survey"]
        receiver_groups = survey_group.get("receiver_groups")
        catalog = None
        if receiver_groups is not None:
            catalog = receiver_groups.get("_catalog")

        if catalog is not None:
            group_names = _h5_dataset_string_values(catalog, "group_name")
            dataset_paths = _h5_dataset_string_values(catalog, "dataset_path")
            layout_kinds = _h5_dataset_string_values(catalog, "layout_kind")
            n_group = max(len(group_names), len(dataset_paths), len(layout_kinds))
            for index in range(n_group):
                group_name = (
                    group_names[index]
                    if index < len(group_names) and group_names[index]
                    else f"group_{index + 1}"
                )
                dataset_path = (
                    dataset_paths[index]
                    if index < len(dataset_paths) and dataset_paths[index]
                    else f"/{group_name}"
                )
                dataset = _clean_h5_path(dataset_path)
                layout_kind = (
                    layout_kinds[index]
                    if index < len(layout_kinds) and layout_kinds[index]
                    else ""
                )
                spec = {
                    "group_name": group_name,
                    "dataset_path": dataset_path,
                    "dataset": dataset,
                    "layout_kind": layout_kind,
                }
                if _group_spec_matches(spec, group):
                    specs.append(spec)
            return specs

        if "survey/traces" in h5 or "survey/trace_nodes" in h5:
            spec = {
                "group_name": "traces",
                "dataset_path": "/survey/traces",
                "dataset": "traces",
                "layout_kind": "sparse_trace_v1",
                "flat": True,
            }
            if _group_spec_matches(spec, group):
                specs.append(spec)

        if receiver_groups is not None:
            for group_name, item in receiver_groups.items():
                if group_name == "_catalog":
                    continue
                if "traces" not in item:
                    continue
                spec = {
                    "group_name": group_name,
                    "dataset_path": f"/{group_name}",
                    "dataset": group_name,
                    "layout_kind": "sparse_trace_v1",
                }
                if _group_spec_matches(spec, group):
                    specs.append(spec)

        for dataset, item in h5.items():
            if dataset in _TRACE_SPECIAL_DATASETS or dataset == "survey":
                continue
            if not hasattr(item, "shape"):
                continue
            layout_kind = set(_h5_attr_strings(item, "layout_kind"))
            if "dense_trace_v1" not in layout_kind:
                continue
            spec = {
                "group_name": dataset,
                "dataset_path": f"/{dataset}",
                "dataset": dataset,
                "layout_kind": "dense_trace_v1",
            }
            if _group_spec_matches(spec, group) and not any(
                existing["group_name"] == dataset for existing in specs
            ):
                specs.append(spec)

        return specs

    @classmethod
    def _read_solver_receiver_catalogs(
        cls,
        h5: Any,
        group_specs: Sequence[Mapping[str, Any]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[str],
        Optional[str],
        Dict[str, Dict[int, int]],
    ]:
        all_ids = []
        all_names = []
        all_coordinates = []
        id_maps: Dict[str, Dict[int, int]] = {}
        units = None
        system = None
        seen_ids = set()
        next_id = 1
        force_unique = len(group_specs) > 1

        for spec in group_specs:
            catalog_name = cls._receiver_catalog_name(h5, spec)
            ids, names, coordinates, catalog_units, catalog_system = (
                cls._read_solver_point_catalog(
                    h5["survey"],
                    catalog_name,
                    id_name="receiver_id",
                    name_name="receiver_name",
                    name_prefix="receiver",
                )
            )
            local_ids = [int(value) for value in ids]
            needs_remap = force_unique or any(value in seen_ids for value in local_ids)
            if needs_remap:
                mapped_ids = np.arange(
                    next_id, next_id + len(local_ids), dtype=np.int64
                )
            else:
                mapped_ids = np.asarray(local_ids, dtype=np.int64)

            if len(mapped_ids):
                next_id = max(next_id, int(mapped_ids.max()) + 1)
            seen_ids.update(int(value) for value in mapped_ids)

            group_key = str(spec["group_name"])
            dataset_key = str(spec["dataset"])
            mapping = {
                int(local_id): int(mapped_id)
                for local_id, mapped_id in zip(local_ids, mapped_ids)
            }
            id_maps[group_key] = mapping
            id_maps[dataset_key] = mapping

            all_ids.extend(int(value) for value in mapped_ids)
            all_names.extend(str(value) for value in names)
            all_coordinates.append(coordinates)
            if units is None:
                units = catalog_units
            if system is None:
                system = catalog_system

        if not all_coordinates:
            raise ValueError("Trace-store survey has no receiver coordinates")
        return (
            np.asarray(all_ids, dtype=np.int64),
            np.asarray(all_names, dtype=object),
            np.vstack(all_coordinates),
            units,
            system,
            id_maps,
        )

    @staticmethod
    def _receiver_catalog_name(h5: Any, spec: Mapping[str, Any]) -> str:
        dataset = str(spec["dataset"])
        group_name = str(spec["group_name"])
        for candidate in (
            f"receiver_groups/{dataset}/receivers",
            f"receiver_groups/{group_name}/receivers",
            "receivers",
        ):
            path = f"survey/{candidate}"
            if path in h5 and "coordinates" in h5[path]:
                return candidate
        raise ValueError(
            f"Trace-store survey has no receiver coordinate catalog for {group_name!r}"
        )

    @staticmethod
    def _read_solver_component_ids(survey_group: Any) -> np.ndarray:
        if "components" not in survey_group:
            return np.arange(1, 2, dtype=np.int64)
        components = survey_group["components"]
        if "component_id" in components:
            return np.asarray(components["component_id"][()], dtype=np.int64)
        if "component" in components:
            return np.asarray(components["component"][()], dtype=np.int64)
        return np.arange(1, 2, dtype=np.int64)

    @classmethod
    def _read_solver_trace_tables(
        cls,
        h5: Any,
        *,
        group_specs: Sequence[Mapping[str, Any]],
        receiver_id_maps: Mapping[str, Mapping[int, int]],
        source_ids: np.ndarray,
        source_field_records: Mapping[int, int],
        component_ids: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        tables: Dict[str, np.ndarray] = {}
        for spec in group_specs:
            group_name = str(spec["group_name"])
            dataset = str(spec["dataset"])
            receiver_map = dict(receiver_id_maps.get(group_name) or {})
            layout_kind = str(spec.get("layout_kind", ""))
            component_id_map = cls._read_solver_component_id_map(h5, spec)
            if bool(spec.get("flat")):
                if "survey/trace_nodes" in h5 and cls._is_aligned_sparse_layout(h5):
                    table = cls._read_solver_aligned_sparse_trace_table(
                        h5,
                        receiver_id_map=receiver_map,
                        source_field_records=source_field_records,
                    )
                    tables[group_name] = table
                    continue
                trace_path = "survey/traces"
            else:
                trace_path = f"survey/receiver_groups/{dataset}/traces"
                if trace_path not in h5 and group_name != dataset:
                    trace_path = f"survey/receiver_groups/{group_name}/traces"
            if "sparse_trace_v1" in layout_kind or trace_path in h5:
                if trace_path not in h5:
                    continue
                table = cls._read_solver_sparse_trace_table(
                    h5[trace_path],
                    component_id_map=component_id_map,
                    source_field_records=source_field_records,
                )
                if receiver_map:
                    table["receiver_id"] = [
                        receiver_map.get(int(value), int(value))
                        for value in table["receiver_id"]
                    ]
                tables[group_name] = table
                continue

            dset_name = dataset if dataset in h5 else group_name
            if dset_name not in h5:
                continue
            dset = h5[dset_name]
            if not hasattr(dset, "shape"):
                continue
            dset_layout_kind = set(_h5_attr_strings(dset, "layout_kind"))
            if "dense_trace_v1" not in dset_layout_kind:
                continue
            table = cls._dense_trace_table_from_dataset(
                dset,
                source_ids=source_ids,
                source_field_records=source_field_records,
                receiver_id_map=receiver_map,
                component_ids=cls._read_solver_group_component_ids(
                    h5,
                    spec,
                    fallback=component_ids,
                ),
            )
            tables[group_name] = table
        return tables

    @staticmethod
    def _read_solver_group_component_ids(
        h5: Any,
        spec: Mapping[str, Any],
        *,
        fallback: np.ndarray,
    ) -> np.ndarray:
        dataset = str(spec["dataset"])
        group_name = str(spec["group_name"])
        for candidate in (
            f"survey/receiver_groups/{dataset}/components",
            f"survey/receiver_groups/{group_name}/components",
            "survey/components",
        ):
            if candidate not in h5:
                continue
            components = h5[candidate]
            if "component_id" in components:
                return np.asarray(components["component_id"][()], dtype=np.int64)
            if "component" in components:
                return np.asarray(components["component"][()], dtype=np.int64)
        return np.asarray(fallback, dtype=np.int64)

    @staticmethod
    def _read_solver_component_id_map(
        h5: Any,
        spec: Mapping[str, Any],
    ) -> Dict[int, int]:
        dataset = str(spec["dataset"])
        group_name = str(spec["group_name"])
        for candidate in (
            f"survey/receiver_groups/{dataset}/components",
            f"survey/receiver_groups/{group_name}/components",
            "survey/components",
        ):
            if candidate not in h5:
                continue
            components = h5[candidate]
            if "component" not in components:
                continue
            component = np.asarray(components["component"][()], dtype=np.int64)
            if "component_id" in components:
                component_id = np.asarray(
                    components["component_id"][()], dtype=np.int64
                )
            else:
                component_id = component
            return {
                int(out): int(identifier)
                for out, identifier in zip(component, component_id)
            }
        return {}

    @staticmethod
    def _is_aligned_sparse_layout(h5: Any) -> bool:
        if "survey/traces/layout_encoding" not in h5:
            return False
        encoding = set(_h5_string_list(h5["survey/traces/layout_encoding"][()]))
        return "aligned_components_v1" in encoding

    @classmethod
    def _read_solver_aligned_sparse_trace_table(
        cls,
        h5: Any,
        *,
        receiver_id_map: Mapping[int, int],
        source_field_records: Mapping[int, int],
    ) -> np.ndarray:
        nodes = h5["survey/trace_nodes"]
        components = h5["survey/components"]
        source = np.asarray(nodes["source_id"][()], dtype=np.int64)
        receiver = np.asarray(nodes["receiver_id"][()], dtype=np.int64)
        weight = _h5_dataset_float_array(
            nodes, "weight", np.ones(len(source), dtype=float)
        )
        component_id = np.asarray(components["component_id"][()], dtype=np.int64)
        component = _h5_dataset_ints(components, "component", component_id)

        n = len(source) * len(component_id)
        table = np.zeros(n, dtype=_TRACE_DTYPE)
        row = 0
        for inode, source_id in enumerate(source):
            receiver_id = receiver_id_map.get(
                int(receiver[inode]), int(receiver[inode])
            )
            node_weight = float(weight[inode])
            for ic, comp_id in enumerate(component_id):
                out_component = int(component[ic])
                table[row]["trace_id"] = row + 1
                table[row]["source_id"] = int(source_id)
                table[row]["receiver_id"] = receiver_id
                table[row]["recv_pos_id"] = receiver_id
                table[row]["component_id"] = int(comp_id)
                table[row]["component"] = out_component
                table[row]["channel_number"] = row + 1
                table[row]["field_record"] = int(
                    source_field_records.get(int(source_id), int(source_id))
                )
                table[row]["active"] = node_weight != 0.0
                table[row]["weight"] = node_weight
                row += 1
        return table

    @staticmethod
    def _read_solver_sparse_trace_table(
        trace_group: Any,
        *,
        component_id_map: Mapping[int, int],
        source_field_records: Mapping[int, int],
    ) -> np.ndarray:
        if "source_id" not in trace_group or "receiver_id" not in trace_group:
            raise ValueError(
                "Sparse survey trace table needs source_id and receiver_id"
            )
        source = np.asarray(trace_group["source_id"][()], dtype=np.int64)
        receiver = np.asarray(trace_group["receiver_id"][()], dtype=np.int64)
        n = len(source)
        if len(receiver) != n:
            raise ValueError("Sparse survey trace table columns have different lengths")
        table = np.zeros(n, dtype=_TRACE_DTYPE)
        table["trace_id"] = _h5_dataset_ints(
            trace_group, "trace_id", np.arange(1, n + 1)
        )
        table["source_id"] = source
        table["receiver_id"] = receiver
        table["recv_pos_id"] = _h5_dataset_ints(trace_group, "recv_pos_id", receiver)
        component = _h5_dataset_ints(
            trace_group,
            "component",
            _h5_dataset_ints(trace_group, "component_id", np.ones(n, dtype=np.int64)),
        )
        table["component"] = component
        table["component_id"] = _h5_dataset_ints(
            trace_group,
            "component_id",
            np.asarray(
                [component_id_map.get(int(value), int(value)) for value in component],
                dtype=np.int64,
            ),
        )
        table["channel_number"] = _h5_dataset_ints(
            trace_group, "channel_number", table["trace_id"]
        )
        table["field_record"] = np.asarray(
            [
                source_field_records.get(int(source_id), int(source_id))
                for source_id in table["source_id"]
            ],
            dtype=np.int64,
        )
        if "field_record" in trace_group:
            table["field_record"] = _h5_dataset_ints(
                trace_group, "field_record", table["field_record"]
            )
        weight = _h5_dataset_float_array(trace_group, "weight", np.ones(n, dtype=float))
        table["weight"] = weight
        if "active" in trace_group:
            table["active"] = _h5_dataset_ints(
                trace_group, "active", np.ones(n, dtype=np.int64)
            ).astype(bool)
            if "weight" not in trace_group:
                table["weight"] = table["active"].astype(float)
        else:
            table["active"] = weight != 0.0
        return table

    @staticmethod
    def _dense_trace_table_from_dataset(
        dset: Any,
        *,
        source_ids: np.ndarray,
        source_field_records: Mapping[int, int],
        receiver_id_map: Mapping[int, int],
        component_ids: np.ndarray,
    ) -> np.ndarray:
        sources = _h5_attr_ints(dset, "shot")
        if sources is None:
            sources = np.asarray(source_ids, dtype=np.int64)
        local_receivers = _h5_attr_ints(dset, "receiver")
        if local_receivers is None:
            local_receivers = np.asarray(list(receiver_id_map), dtype=np.int64)
        receivers = np.asarray(
            [
                receiver_id_map.get(int(receiver), int(receiver))
                for receiver in local_receivers
            ],
            dtype=np.int64,
        )
        components = _h5_attr_ints(dset, "component")
        if components is None:
            components = np.asarray(component_ids, dtype=np.int64)

        n = len(sources) * len(receivers) * len(components)
        table = np.zeros(n, dtype=_TRACE_DTYPE)
        row = 0
        for source in sources:
            for receiver in receivers:
                for component in components:
                    table[row]["trace_id"] = row + 1
                    table[row]["source_id"] = int(source)
                    table[row]["receiver_id"] = int(receiver)
                    table[row]["recv_pos_id"] = int(receiver)
                    table[row]["component_id"] = int(component)
                    table[row]["component"] = int(component)
                    table[row]["channel_number"] = row + 1
                    table[row]["field_record"] = int(
                        source_field_records.get(int(source), int(source))
                    )
                    table[row]["active"] = True
                    table[row]["weight"] = 1.0
                    row += 1
        return table

    @staticmethod
    def _relations_from_trace_tables(
        trace_tables: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        rows = []
        seen = set()
        relation_row = 0
        for table in trace_tables.values():
            for trace in table:
                if not bool(trace["active"]):
                    continue
                key = (int(trace["source_id"]), int(trace["receiver_id"]))
                if key in seen:
                    continue
                seen.add(key)
                relation_row += 1
                rows.append(
                    (
                        key[0],
                        key[1],
                        int(trace["field_record"]),
                        int(trace["channel_number"]),
                        relation_row,
                    )
                )
        return np.asarray(rows, dtype=_RELATION_DTYPE)

    @staticmethod
    def _read_sps_points(
        file: Union[str, Path], kind: str, dimension: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[tuple[float, float, int], int]]:
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        coords = []
        names = []
        ids = []
        keys: Dict[tuple[float, float, int], int] = {}
        path = Path(file)
        with path.open("r") as stream:
            for line in stream:
                if not line or line[0:1] != kind:
                    continue
                row_id = len(ids) + 1
                line_number = _field_real(line, 2, 17, 0.0)
                point_number = _field_real(line, 18, 25, float(row_id))
                point_index = _field_int(line, 26, 26, 1)
                x = [
                    _field_real(line, 47, 55, 0.0),
                    _field_real(line, 56, 65, 0.0),
                ]
                if dimension == 3:
                    x.append(_field_real(line, 66, 71, 0.0))
                coords.append(x)
                ids.append(row_id)
                names.append(f"{line_number:g}_{point_number:g}_{point_index}")
                keys[(line_number, point_number, point_index)] = row_id
        return (
            np.asarray(coords, dtype=float).reshape(-1, dimension),
            np.asarray(ids, dtype=np.int64),
            np.asarray(names, dtype=object),
            keys,
        )

    @staticmethod
    def _read_sps_relations(
        file: Union[str, Path],
        *,
        source_keys: Mapping[tuple[float, float, int], int],
        receiver_keys: Mapping[tuple[float, float, int], int],
    ) -> np.ndarray:
        rows = []
        path = Path(file)
        with path.open("r") as stream:
            for relation_row, line in enumerate(stream, start=1):
                if not line or line[0:1] != "X":
                    continue
                field_record = _field_int(line, 8, 15, 0)
                source_key = (
                    _field_real(line, 18, 27, 0.0),
                    _field_real(line, 28, 37, 0.0),
                    _field_int(line, 38, 38, 1),
                )
                try:
                    source_id = source_keys[source_key]
                except KeyError:
                    raise ValueError(
                        f"SPX relation references unknown source {source_key}"
                    ) from None
                from_channel = _field_int(line, 39, 43, 0)
                to_channel = _field_int(line, 44, 48, from_channel)
                channel_increment = max(1, _field_int(line, 49, 49, 1))
                receiver_line = _field_real(line, 50, 59, 0.0)
                from_receiver = _field_real(line, 60, 69, 0.0)
                to_receiver = _field_real(line, 70, 79, from_receiver)
                receiver_index = _field_int(line, 80, 80, 1)
                n_group = ((to_channel - from_channel) // channel_increment) + 1
                point_step = 1 if to_receiver >= from_receiver else -1
                for group in range(max(0, n_group)):
                    receiver_point = from_receiver + group * point_step
                    receiver_key = (receiver_line, receiver_point, receiver_index)
                    try:
                        receiver_id = receiver_keys[receiver_key]
                    except KeyError:
                        raise ValueError(
                            f"SPX relation references unknown receiver {receiver_key}"
                        ) from None
                    rows.append(
                        (
                            source_id,
                            receiver_id,
                            field_record,
                            from_channel + group * channel_increment,
                            relation_row,
                        )
                    )
        return np.asarray(rows, dtype=_RELATION_DTYPE)

    def to_acquisition(
        self,
        device: ReceiverDevice,
        *,
        receiver_group_name: Optional[str] = None,
        source_kind: str = "scalar",
        source_direction: Optional[Sequence[float]] = None,
        source_domain: Optional[int] = None,
        receiver_domain: Optional[int] = None,
        **receiver_kwargs: Any,
    ) -> Acquisition:
        _reject_frame_kwargs(receiver_kwargs)
        acq = Acquisition()
        self.apply_to(
            acq,
            receiver_group_name=receiver_group_name,
            device=device,
            source_kind=source_kind,
            source_direction=source_direction,
            source_domain=source_domain,
            receiver_domain=receiver_domain,
            **receiver_kwargs,
        )
        return acq

    def apply_to(
        self,
        acq: Acquisition,
        *,
        receiver_group_name: Optional[str] = None,
        device: ReceiverDevice,
        source_kind: str = "scalar",
        source_direction: Optional[Sequence[float]] = None,
        source_domain: Optional[int] = None,
        receiver_domain: Optional[int] = None,
        **receiver_kwargs: Any,
    ) -> Acquisition:
        _reject_frame_kwargs(receiver_kwargs)
        group_name = receiver_group_name or self.name
        acq.add_source_group(
            kind=source_kind,
            coords=_coordinate_value_or_array(
                self.sources, units=self.source_units, system=self.source_system
            ),
            direction=source_direction,
            domain=source_domain,
        )

        if self.kind.lower() == "dense":
            acq.add_receiver_group(
                name=group_name,
                device=device,
                coords=_coordinate_value_or_array(
                    self.receivers,
                    units=self.receiver_units,
                    system=self.receiver_system,
                ),
                domain=receiver_domain,
                **receiver_kwargs,
            )
            return acq

        if self.kind.lower() == "spsfiles":
            survey = SparseSurvey.sps(
                self.name,
                source_file=self.source_file,
                receiver_file=self.receiver_file,
                relation_file=self.relation_file,
            )
        elif self.kind.lower() == "offsetdomain":
            survey = SparseSurvey.offset_domain(
                self.name, **(self.offset_domain_spec or {})
            )
        else:
            raise ValueError(f"Unsupported survey kind: {self.kind}")

        acq.add_sparse_receiver_group(
            group_name,
            device,
            coords=_coordinate_value_or_array(
                self.receivers,
                units=self.receiver_units,
                system=self.receiver_system,
            ),
            survey=survey,
            domain=receiver_domain,
            **receiver_kwargs,
        )
        return acq

    def plot(
        self,
        *,
        sources: Optional[Union[int, Sequence[int]]] = None,
        receivers: Optional[Union[int, Sequence[int]]] = None,
        ax: Any = None,
        show_links: bool = True,
        annotate: bool = False,
        source_kwargs: Optional[Mapping[str, Any]] = None,
        receiver_kwargs: Optional[Mapping[str, Any]] = None,
        link_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "Survey plotting",
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc

        if ax is None:
            _, ax = plt.subplots()

        source_set = _as_ids(sources)
        receiver_set = _as_ids(receivers)

        source_ids = set(int(i) for i in self.source_ids)
        receiver_ids = set(int(i) for i in self.receiver_ids)
        relation_rows = self.relations
        if relation_rows.size:
            if source_set is None and receiver_set is not None:
                source_set = {
                    int(row["source_id"])
                    for row in relation_rows
                    if int(row["receiver_id"]) in receiver_set
                }
            if receiver_set is None and source_set is not None:
                receiver_set = {
                    int(row["receiver_id"])
                    for row in relation_rows
                    if int(row["source_id"]) in source_set
                }

        source_set = source_ids if source_set is None else source_set
        receiver_set = receiver_ids if receiver_set is None else receiver_set

        src_mask = np.asarray([int(i) in source_set for i in self.source_ids])
        rcv_mask = np.asarray([int(i) in receiver_set for i in self.receiver_ids])

        src_style = {"marker": "*", "s": 80, "c": "tab:red", "label": "Sources"}
        src_style.update(dict(source_kwargs or {}))
        rcv_style = {"marker": "v", "s": 35, "c": "tab:blue", "label": "Receivers"}
        rcv_style.update(dict(receiver_kwargs or {}))
        line_style = {"color": "0.65", "linewidth": 0.7, "alpha": 0.7}
        line_style.update(dict(link_kwargs or {}))

        if show_links and relation_rows.size:
            source_index = {int(value): i for i, value in enumerate(self.source_ids)}
            receiver_index = {
                int(value): i for i, value in enumerate(self.receiver_ids)
            }
            for row in relation_rows:
                source_id = int(row["source_id"])
                receiver_id = int(row["receiver_id"])
                if source_id not in source_set or receiver_id not in receiver_set:
                    continue
                sxy = self.sources[source_index[source_id], :2]
                rxy = self.receivers[receiver_index[receiver_id], :2]
                ax.plot([sxy[0], rxy[0]], [sxy[1], rxy[1]], **line_style)

        ax.scatter(self.sources[src_mask, 0], self.sources[src_mask, 1], **src_style)
        ax.scatter(
            self.receivers[rcv_mask, 0], self.receivers[rcv_mask, 1], **rcv_style
        )

        if annotate:
            for i, point in zip(self.source_ids[src_mask], self.sources[src_mask]):
                ax.annotate(f"S{i}", point[:2])
            for i, point in zip(self.receiver_ids[rcv_mask], self.receivers[rcv_mask]):
                ax.annotate(f"R{i}", point[:2])

        ax.set_xlabel("x")
        ax.set_ylabel("z" if self.sources.shape[1] == 2 else "y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        return ax
