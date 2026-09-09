"""Sparse seismic survey authoring helpers.

These classes mirror fast solver sparse receiver layout contracts while keeping the
Python-facing syntax compact. They intentionally do not read server-side files
when exporting JSON.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from frequensolve.units import value_and_units_to_fs
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
)

__all__ = [
    "ReceiverSampling",
    "SparseSurvey",
    "SparseTrace",
    "SparseTraceTable",
]


ComponentKey = Union[int, str]


def _component_map_key(name: str) -> str:
    return str(name).strip().lower()


def _resolve_component(
    value: Optional[ComponentKey], component_map: Optional[Mapping[str, int]]
) -> int:
    if value is None:
        return 1
    if isinstance(value, str):
        if component_map is None:
            raise ValueError(
                "Sparse survey component names need a receiver device context. "
                "Use Acquisition.add_sparse_receiver_group(...) or pass numeric component ids."
            )
        try:
            return int(component_map[_component_map_key(value)])
        except KeyError:
            names = ", ".join(sorted(component_map))
            raise ValueError(
                f"Unknown sparse survey component {value!r}. Known components: {names}"
            ) from None
    return int(value)


def _path_to_fs(path: Union[str, Path], ctx: Optional[ExportContext]) -> str:
    if ctx is None:
        return str(path)
    return str(ctx.relative_to_project(Path(path)))


def _as_trace(value: Union["SparseTrace", Mapping[str, Any]]) -> "SparseTrace":
    if isinstance(value, SparseTrace):
        return value
    if isinstance(value, Mapping):
        return SparseTrace.from_fs(value)
    raise TypeError(f"Cannot convert {type(value)} to SparseTrace")


@dataclass
class ReceiverSampling(ExtraFieldsMixin):
    """Receiver-group sampling block.

    Dense receiver groups omit this object. Sparse receiver groups usually use
    ``ReceiverSampling.sparse("survey_name")`` or are created through
    ``Acquisition.add_sparse_receiver_group``.

    Args:
        kind: Solver sampling kind, usually ``"Sparse"``.
        survey: Referenced survey name.
        extra: Additional solver-facing sampling fields.
    """

    kind: str = "Sparse"
    survey: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def sparse(cls, survey: Union[str, "SparseSurvey"]) -> "ReceiverSampling":
        """Create sparse sampling that references a survey by name or object."""

        return cls.from_value(survey)

    @classmethod
    def from_value(cls, value: Any) -> Optional["ReceiverSampling"]:
        """Coerce common public inputs into ``ReceiverSampling``.

        Args:
            value: ``None``, existing sampling object, survey object, survey
                name, or serialized mapping.

        Returns:
            ``ReceiverSampling`` or ``None``.
        """

        if value is None:
            return None
        if isinstance(value, ReceiverSampling):
            return cls(
                kind=value.kind, survey=value.survey, extra=copy.deepcopy(value.extra)
            )
        if isinstance(value, SparseSurvey):
            return cls(kind=value.kind, survey=value.name)
        if isinstance(value, str):
            return cls(kind="Sparse", survey=value)
        if isinstance(value, Mapping):
            return cls.from_fs(value)
        if hasattr(value, "name"):
            return cls(kind=getattr(value, "kind", "Sparse"), survey=value.name)
        raise TypeError(f"Cannot convert {type(value)} to ReceiverSampling")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ReceiverSampling":
        """Deserialize receiver sampling from solver JSON."""

        payload = copy.deepcopy(dict(data))
        kind = payload.pop("_type", payload.pop("kind", "Sparse"))
        survey = payload.pop("survey", None)
        return cls(kind=kind, survey=survey, extra=payload)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize receiver sampling for solver input."""

        payload = {"_type": self.kind}
        if self.survey is not None:
            payload["survey"] = self.survey
        return merge_extra(payload, self.extra, "ReceiverSampling")


@dataclass(init=False)
class SparseTrace(ExtraFieldsMixin):
    """One output trace row in a fast solver sparse receiver layout.

    Parameters use 1-based ids to match the solver. ``source`` and ``receiver``
    are aliases for ``source_id`` and ``receiver_id``. ``point`` is the common
    case where a trace maps to exactly one receiver coordinate row.

    Args:
        source: Alias for ``source_id``.
        receiver: Alias for ``receiver_id``.
        component: Component id or component name.
        point: Receiver-position id for one-point traces.
        points: Inclusive ``(point_first, point_last)`` receiver-position range.
        source_id: One-based source id.
        receiver_id: One-based receiver id.
        receiver_position: Alias for ``receiver_position_id``.
        receiver_position_id: One-based receiver-position id.
        trace_id: Optional one-based trace id. Export assigns this when omitted.
        component_id: Optional component id/name override.
        channel_number: Optional acquisition channel number.
        field_record: Optional field-record/source gather id.
        point_first: First receiver-position id in the trace.
        point_last: Last receiver-position id in the trace.
        active: Whether the trace is active.
        offset: Optional source-receiver offset.
        azimuth: Optional source-receiver azimuth.
        source_name: Optional source display name.
        receiver_name: Optional receiver display name.
        component_name: Optional component display name.
        extra: Additional solver-facing trace fields.
        **kwargs: Additional solver-facing trace fields.

    Raises:
        ValueError: If source or receiver ids are missing.
    """

    source_id: int
    receiver_id: int
    component: Optional[ComponentKey]
    receiver_position_id: Optional[int]
    trace_id: Optional[int]
    component_id: Optional[ComponentKey]
    channel_number: Optional[int]
    field_record: Optional[int]
    point_first: Optional[int]
    point_last: Optional[int]
    active: bool
    offset: Optional[float]
    azimuth: Optional[float]
    source_name: Optional[str]
    receiver_name: Optional[str]
    component_name: Optional[str]
    extra: Dict[str, Any]

    def __init__(
        self,
        *,
        source: Optional[int] = None,
        receiver: Optional[int] = None,
        component: Optional[ComponentKey] = 1,
        point: Optional[int] = None,
        points: Optional[Tuple[int, int]] = None,
        source_id: Optional[int] = None,
        receiver_id: Optional[int] = None,
        receiver_position: Optional[int] = None,
        receiver_position_id: Optional[int] = None,
        trace_id: Optional[int] = None,
        component_id: Optional[ComponentKey] = None,
        channel_number: Optional[int] = None,
        field_record: Optional[int] = None,
        point_first: Optional[int] = None,
        point_last: Optional[int] = None,
        active: bool = True,
        offset: Optional[float] = None,
        azimuth: Optional[float] = None,
        source_name: Optional[str] = None,
        receiver_name: Optional[str] = None,
        component_name: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if source_id is None:
            source_id = source
        if receiver_id is None:
            receiver_id = receiver
        if receiver_position_id is None:
            receiver_position_id = receiver_position
        if points is not None:
            point_first, point_last = int(points[0]), int(points[1])
        if point is not None:
            point_first = point_last = int(point)
        if source_id is None or receiver_id is None:
            raise ValueError(
                "SparseTrace requires source/source_id and receiver/receiver_id"
            )
        if isinstance(component, str) and component_name is None:
            component_name = component

        self.source_id = int(source_id)
        self.receiver_id = int(receiver_id)
        self.component = component
        self.receiver_position_id = (
            None if receiver_position_id is None else int(receiver_position_id)
        )
        self.trace_id = None if trace_id is None else int(trace_id)
        self.component_id = component_id
        self.channel_number = None if channel_number is None else int(channel_number)
        self.field_record = None if field_record is None else int(field_record)
        self.point_first = None if point_first is None else int(point_first)
        self.point_last = None if point_last is None else int(point_last)
        self.active = bool(active)
        self.offset = None if offset is None else float(offset)
        self.azimuth = None if azimuth is None else float(azimuth)
        self.source_name = source_name
        self.receiver_name = receiver_name
        self.component_name = component_name
        self._init_extra(extra, **kwargs)
        for label, value in (
            ("source_id", self.source_id),
            ("receiver_id", self.receiver_id),
            ("receiver_position_id", self.receiver_position_id),
            ("trace_id", self.trace_id),
            ("channel_number", self.channel_number),
            ("field_record", self.field_record),
            ("point_first", self.point_first),
            ("point_last", self.point_last),
        ):
            if value is not None and value < 1:
                raise ValueError(f"SparseTrace {label} must be positive")
        if (
            self.point_first is not None
            and self.point_last is not None
            and self.point_last < self.point_first
        ):
            raise ValueError("SparseTrace point_last must be >= point_first")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SparseTrace":
        """Deserialize a sparse trace row."""

        payload = copy.deepcopy(dict(data))
        if "recv_pos_id" in payload and "receiver_position_id" not in payload:
            payload["receiver_position_id"] = payload.pop("recv_pos_id")
        return cls(**payload)

    def to_fs(
        self,
        trace_id: Optional[int] = None,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Serialize this trace row for solver input.

        Args:
            trace_id: One-based export row id used when the trace does not
                already define one.
            component_map: Optional mapping from component names to one-based
                component ids.

        Returns:
            JSON-compatible sparse trace row.

        Raises:
            ValueError: If no positive trace id is available or a named
                component cannot be resolved.
        """

        out_trace_id = int(self.trace_id or trace_id or 0)
        if out_trace_id <= 0:
            raise ValueError("SparseTrace requires a positive trace_id or export row")

        component = _resolve_component(self.component, component_map)
        component_id = (
            _resolve_component(self.component_id, component_map)
            if self.component_id is not None
            else component
        )
        receiver_position_id = self.receiver_position_id or self.receiver_id
        point_first = self.point_first or receiver_position_id
        point_last = self.point_last or point_first
        n_points = max(0, int(point_last) - int(point_first) + 1)

        payload: Dict[str, Any] = {
            "trace_id": out_trace_id,
            "source_id": self.source_id,
            "receiver_id": self.receiver_id,
            "receiver_position_id": receiver_position_id,
            "component_id": component_id,
            "component": component,
            "channel_number": self.channel_number or out_trace_id,
            "field_record": self.field_record or self.source_id,
            "point_first": point_first,
            "point_last": point_last,
            "n_points": n_points,
            "active": self.active,
        }
        if self.offset is not None:
            payload["offset"] = self.offset
        if self.azimuth is not None:
            payload["azimuth"] = self.azimuth
        if self.source_name is not None:
            payload["source_name"] = self.source_name
        if self.receiver_name is not None:
            payload["receiver_name"] = self.receiver_name
        if self.component_name is not None:
            payload["component_name"] = self.component_name
        return merge_extra(payload, self.extra, "SparseTrace")


class SparseTraceTable:
    """Columnar sparse traces for production-scale survey construction."""

    _OPTIONAL_COLUMNS = {
        "trace_id",
        "component_id",
        "channel_number",
        "field_record",
        "point_first",
        "point_last",
        "active",
        "offset",
        "azimuth",
        "source_name",
        "receiver_name",
        "component_name",
    }

    def __init__(
        self,
        *,
        source_id: Any,
        receiver_id: Any,
        component: Any = 1,
        receiver_position_id: Optional[Any] = None,
        **columns: Any,
    ) -> None:
        unknown = set(columns).difference(self._OPTIONAL_COLUMNS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unsupported sparse trace columns: {names}")
        source = np.asarray(source_id, dtype=np.int64)
        receiver = np.asarray(receiver_id, dtype=np.int64)
        if source.ndim == 0:
            source = source.reshape(1)
        if receiver.ndim == 0:
            receiver = receiver.reshape(1)
        if source.ndim != 1 or receiver.shape != source.shape:
            raise ValueError("source_id and receiver_id must be equal-length vectors")
        if np.any(source < 1) or np.any(receiver < 1):
            raise ValueError("source_id and receiver_id must be positive")
        self._columns: Dict[str, np.ndarray] = {
            "source_id": source,
            "receiver_id": receiver,
            "receiver_position_id": self._column(
                receiver if receiver_position_id is None else receiver_position_id,
                len(source),
                "receiver_position_id",
                dtype=np.int64,
            ),
            "component": self._column(
                component,
                len(source),
                "component",
                dtype=None,
            ),
        }
        component_values = self._columns["component"]
        if "component_name" not in columns:
            if component_values.dtype.kind in {"S", "U"}:
                columns["component_name"] = component_values
            elif component_values.dtype.kind == "O" and any(
                isinstance(value, str) for value in np.unique(component_values)
            ):
                columns["component_name"] = np.asarray(
                    [
                        value if isinstance(value, str) else ""
                        for value in component_values
                    ],
                    dtype=object,
                )
        for name, values in columns.items():
            dtype = (
                None
                if name.endswith("_name")
                else (
                    np.bool_
                    if name == "active"
                    else (
                        np.float64
                        if name in {"offset", "azimuth"}
                        else None if name == "component_id" else np.int64
                    )
                )
            )
            self._columns[name] = self._column(
                values,
                len(source),
                name,
                dtype=dtype,
            )
        for name in (
            "receiver_position_id",
            "trace_id",
            "channel_number",
            "field_record",
            "point_first",
            "point_last",
        ):
            values = self._columns.get(name)
            if values is not None and np.any(values < 1):
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def _column(values: Any, size: int, name: str, *, dtype: Any) -> np.ndarray:
        """Normalize one scalar or vector trace column."""

        if isinstance(values, (str, bytes)) or np.isscalar(values):
            value = values.item() if isinstance(values, np.generic) else values
            return np.full(size, value, dtype=dtype)
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 1 or len(array) != size:
            raise ValueError(f"{name} must be a scalar or a {size}-element vector")
        return array

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[Tuple[int, int]],
        *,
        component: ComponentKey = 1,
        receiver_points: Optional[Mapping[int, int]] = None,
    ) -> "SparseTraceTable":
        """Build a trace table from source/receiver pairs."""

        def pair_values() -> Iterable[int]:
            for pair in pairs:
                try:
                    source_id, receiver_id = pair
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "pairs must contain (source_id, receiver_id) rows"
                    ) from exc
                yield source_id
                yield receiver_id

        values = np.fromiter(
            pair_values(),
            dtype=np.int64,
        )
        if values.size == 0:
            values = values.reshape(0, 2)
        elif values.size % 2:
            raise ValueError("pairs must contain (source_id, receiver_id) rows")
        else:
            values = values.reshape(-1, 2)
        receiver = values[:, 1]
        positions = (
            receiver
            if receiver_points is None
            else np.fromiter(
                (
                    receiver_points.get(int(value), int(value)) or int(value)
                    for value in receiver
                ),
                dtype=np.int64,
                count=len(receiver),
            )
        )
        return cls(
            source_id=values[:, 0],
            receiver_id=receiver,
            receiver_position_id=positions,
            component=component,
        )

    @classmethod
    def from_product(
        cls,
        *,
        sources: Iterable[int],
        receivers: Iterable[int],
        components: Union[ComponentKey, Iterable[ComponentKey]] = 1,
        receiver_points: Optional[Mapping[int, int]] = None,
    ) -> "SparseTraceTable":
        """Build a trace table with vectorized Cartesian-product columns."""

        source = (
            np.asarray(sources, dtype=np.int64)
            if isinstance(sources, np.ndarray)
            else np.fromiter(sources, dtype=np.int64)
        )
        receiver = (
            np.asarray(receivers, dtype=np.int64)
            if isinstance(receivers, np.ndarray)
            else np.fromiter(receivers, dtype=np.int64)
        )
        if source.ndim != 1 or receiver.ndim != 1:
            raise ValueError("sources and receivers must be one-dimensional")
        if isinstance(components, (str, int, np.integer)):
            component = np.asarray([components])
        else:
            component_items = list(components)
            component = np.asarray(
                component_items,
                dtype=(
                    None
                    if all(isinstance(value, str) for value in component_items)
                    or all(
                        isinstance(value, (int, np.integer))
                        for value in component_items
                    )
                    else object
                ),
            )
        if component.ndim != 1 or len(component) == 0:
            raise ValueError("components must not be empty")
        n_receiver_component = len(receiver) * len(component)
        source_id = np.repeat(source, n_receiver_component)
        receiver_id = np.tile(np.repeat(receiver, len(component)), len(source))
        component_values = np.tile(component, len(source) * len(receiver))
        positions = (
            receiver_id
            if receiver_points is None
            else np.fromiter(
                (
                    receiver_points.get(int(value), int(value)) or int(value)
                    for value in receiver_id
                ),
                dtype=np.int64,
                count=len(receiver_id),
            )
        )
        return cls(
            source_id=source_id,
            receiver_id=receiver_id,
            receiver_position_id=positions,
            component=component_values,
        )

    def __len__(self) -> int:
        return len(self._columns["source_id"])

    def columns(
        self,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return resolved solver columns without constructing trace objects."""

        size = len(self)
        trace_id = self._columns.get(
            "trace_id",
            np.arange(1, size + 1, dtype=np.int64),
        )
        component = self._resolve_components(
            self._columns["component"],
            component_map,
        )
        component_id = self._resolve_components(
            self._columns.get("component_id", component),
            component_map,
        )
        if np.any(component < 1) or np.any(component_id < 1):
            raise ValueError("component ids must be positive")
        position = self._columns["receiver_position_id"]
        point_first = self._columns.get("point_first", position)
        point_last = self._columns.get("point_last", point_first)
        if np.any(point_last < point_first):
            raise ValueError("point_last must be >= point_first")
        result = {
            "trace_id": np.asarray(trace_id, dtype=np.int64),
            "source_id": self._columns["source_id"],
            "receiver_id": self._columns["receiver_id"],
            "receiver_position_id": position,
            "component_id": component_id,
            "component": component,
            "channel_number": np.asarray(
                self._columns.get("channel_number", trace_id),
                dtype=np.int64,
            ),
            "field_record": np.asarray(
                self._columns.get("field_record", self._columns["source_id"]),
                dtype=np.int64,
            ),
            "point_first": np.asarray(point_first, dtype=np.int64),
            "point_last": np.asarray(point_last, dtype=np.int64),
            "n_points": np.asarray(point_last - point_first + 1, dtype=np.int64),
            "active": np.asarray(
                self._columns.get("active", np.ones(size, dtype=np.bool_)),
                dtype=np.bool_,
            ),
        }
        for name in (
            "offset",
            "azimuth",
            "source_name",
            "receiver_name",
            "component_name",
        ):
            if name in self._columns:
                result[name] = self._columns[name]
        return result

    @staticmethod
    def _resolve_components(
        values: np.ndarray,
        component_map: Optional[Mapping[str, int]],
    ) -> np.ndarray:
        """Resolve numeric or named components once per distinct value."""

        values = np.asarray(values)
        if values.dtype.kind in {"i", "u"}:
            return values.astype(np.int64, copy=False)
        if values.dtype.kind not in {"O", "S", "U"}:
            return values.astype(np.int64)
        unique_values = np.unique(values)
        if not any(isinstance(value, str) for value in unique_values):
            return values.astype(np.int64)
        resolved = np.empty(len(values), dtype=np.int64)
        for value in unique_values:
            resolved[values == value] = _resolve_component(value, component_map)
        return resolved

    def rows(
        self,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Materialize JSON rows for small examples."""

        columns = self.columns(component_map)
        rows = []
        for index in range(len(self)):
            row = {}
            for name, values in columns.items():
                value = values[index]
                row[name] = value.item() if isinstance(value, np.generic) else value
            rows.append(row)
        return rows

    def trace(self, index: int) -> SparseTrace:
        """Materialize one trace object from the columnar table."""

        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("sparse trace index is out of range")
        values = self._columns
        optional = {
            name: (
                column[index].item()
                if isinstance(column[index], np.generic)
                else column[index]
            )
            for name, column in values.items()
            if name
            not in {
                "source_id",
                "receiver_id",
                "receiver_position_id",
                "component",
            }
        }
        return SparseTrace(
            source_id=int(values["source_id"][index]),
            receiver_id=int(values["receiver_id"][index]),
            receiver_position_id=int(values["receiver_position_id"][index]),
            component=values["component"][index],
            **optional,
        )


@dataclass
class _InlineSparseSurvey:
    table: Optional[SparseTraceTable]
    traces: List[SparseTrace]


@dataclass
class _HDF5SparseSurvey:
    layout_file: Union[str, Path]


@dataclass
class _SPSSparseSurvey:
    source_file: Union[str, Path]
    receiver_file: Union[str, Path]
    relation_file: Union[str, Path]


@dataclass
class _OffsetSparseSurvey:
    offset_domain: Mapping[str, Any]


SparseSurveyStorage = Union[
    _InlineSparseSurvey,
    _HDF5SparseSurvey,
    _SPSSparseSurvey,
    _OffsetSparseSurvey,
]


@dataclass(init=False)
class SparseSurvey(ExtraFieldsMixin):
    """Named fast solver sparse survey layout.

    The default form exports inline JSON accepted by the fast solver's ``Sparse`` layout
    reader. Use ``SparseSurvey.file(...)`` for an existing HDF5 trace store or
    ``SparseSurvey.sps(...)`` for SPS source/receiver/relation files.

    Args:
        name: Survey name referenced by receiver groups.
        traces: Inline sparse trace rows.
        kind: Solver survey kind. Inferred from file/offset arguments when
            omitted.
        layout_file: Existing HDF5 trace-store layout file.
        source_file: SPS source file.
        receiver_file: SPS receiver file.
        relation_file: SPS relation file.
        offset_domain: Offset-domain selection payload.
        extra: Additional solver-facing survey fields.
        **kwargs: Additional solver-facing survey fields.
    """

    name: str
    extra: Dict[str, Any]
    _storage: SparseSurveyStorage

    def _reject_unsupported_sample_tables(self) -> None:
        removed = sorted({"eval_samples", "trace_samples"}.intersection(self.extra))
        if removed:
            raise ValueError(
                "SparseSurvey no longer supports "
                + ", ".join(removed)
                + "; Sauce does not read custom sample tables. "
                "Use receiver-group coordinates and supported receiver sampling."
            )

    def __init__(
        self,
        name: str,
        traces: Optional[Iterable[Union[SparseTrace, Mapping[str, Any]]]] = None,
        *,
        kind: Optional[str] = None,
        trace_table: Optional[SparseTraceTable] = None,
        layout_file: Optional[Union[str, Path]] = None,
        source_file: Optional[Union[str, Path]] = None,
        receiver_file: Optional[Union[str, Path]] = None,
        relation_file: Optional[Union[str, Path]] = None,
        offset_domain: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if kind is None:
            if layout_file is not None:
                kind = "HDF5TraceStore"
            elif offset_domain is not None:
                kind = "OffsetDomain"
            elif (
                source_file is not None
                or receiver_file is not None
                or relation_file is not None
            ):
                kind = "SPSFiles"
            else:
                kind = "Sparse"
        self.name = name
        trace_rows = [_as_trace(trace) for trace in (traces or [])]
        if trace_table is not None and not isinstance(trace_table, SparseTraceTable):
            raise TypeError("trace_table must be a SparseTraceTable")
        self._init_extra(extra, **kwargs)
        self._reject_unsupported_sample_tables()
        normalized_kind = str(kind).strip().lower()
        has_sps = any(
            value is not None for value in (source_file, receiver_file, relation_file)
        )
        if normalized_kind == "sparse":
            if layout_file is not None or has_sps or offset_domain is not None:
                raise ValueError("Sparse survey cannot mix external survey storage")
            self._storage = _InlineSparseSurvey(
                table=trace_table,
                traces=trace_rows,
            )
        elif normalized_kind == "hdf5tracestore":
            if layout_file is None:
                raise ValueError("HDF5TraceStore survey requires layout_file")
            if (
                trace_table is not None
                or trace_rows
                or has_sps
                or offset_domain is not None
            ):
                raise ValueError(
                    "HDF5TraceStore survey cannot include inline traces or SPS files"
                )
            self._storage = _HDF5SparseSurvey(layout_file)
        elif normalized_kind == "spsfiles":
            if not all(
                value is not None
                for value in (
                    source_file,
                    receiver_file,
                    relation_file,
                )
            ):
                raise ValueError("SPSFiles survey requires all three SPS files")
            if (
                trace_table is not None
                or trace_rows
                or layout_file is not None
                or offset_domain is not None
            ):
                raise ValueError("SPSFiles survey cannot mix other survey storage")
            assert source_file is not None
            assert receiver_file is not None
            assert relation_file is not None
            self._storage = _SPSSparseSurvey(
                source_file=source_file,
                receiver_file=receiver_file,
                relation_file=relation_file,
            )
        elif normalized_kind == "offsetdomain":
            if offset_domain is None:
                raise ValueError("OffsetDomain survey requires offset_domain")
            if (
                trace_table is not None
                or trace_rows
                or layout_file is not None
                or has_sps
            ):
                raise ValueError("OffsetDomain survey cannot mix other survey storage")
            self._storage = _OffsetSparseSurvey(copy.deepcopy(dict(offset_domain)))
        else:
            raise ValueError(f"Unsupported sparse survey type: {kind}")

    @property
    def kind(self) -> str:
        if isinstance(self._storage, _InlineSparseSurvey):
            return "Sparse"
        if isinstance(self._storage, _HDF5SparseSurvey):
            return "HDF5TraceStore"
        if isinstance(self._storage, _SPSSparseSurvey):
            return "SPSFiles"
        return "OffsetDomain"

    @property
    def traces(self) -> List[SparseTrace]:
        """Return the legacy mutable trace list, expanding a table on demand."""

        if not isinstance(self._storage, _InlineSparseSurvey):
            return []
        storage = self._storage
        if storage.table is not None:
            storage.traces[:0] = [
                storage.table.trace(index) for index in range(len(storage.table))
            ]
            storage.table = None
        return storage.traces

    @property
    def layout_file(self) -> Optional[Union[str, Path]]:
        return (
            self._storage.layout_file
            if isinstance(self._storage, _HDF5SparseSurvey)
            else None
        )

    @property
    def source_file(self) -> Optional[Union[str, Path]]:
        return (
            self._storage.source_file
            if isinstance(self._storage, _SPSSparseSurvey)
            else None
        )

    @property
    def receiver_file(self) -> Optional[Union[str, Path]]:
        return (
            self._storage.receiver_file
            if isinstance(self._storage, _SPSSparseSurvey)
            else None
        )

    @property
    def relation_file(self) -> Optional[Union[str, Path]]:
        return (
            self._storage.relation_file
            if isinstance(self._storage, _SPSSparseSurvey)
            else None
        )

    @property
    def trace_count(self) -> int:
        """Return the trace count without expanding columnar storage."""

        if not isinstance(self._storage, _InlineSparseSurvey):
            return 0
        table_count = len(self._storage.table) if self._storage.table is not None else 0
        return table_count + len(self._storage.traces)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SparseSurvey":
        """Deserialize sparse survey configuration from solver JSON."""

        payload = copy.deepcopy(dict(data))
        kind = payload.pop("_type", payload.pop("kind", "Sparse"))
        return cls(kind=kind, **payload)

    @classmethod
    def file(
        cls,
        name: str,
        layout_file: Union[str, Path],
        *,
        kind: str = "HDF5TraceStore",
        **kwargs: Any,
    ) -> "SparseSurvey":
        """Create a survey backed by an existing layout file.

        Args:
            name: Survey name.
            layout_file: HDF5 layout file path.
            kind: Solver survey kind.
            **kwargs: Additional ``SparseSurvey`` constructor arguments.
        """

        return cls(name=name, kind=kind, layout_file=layout_file, **kwargs)

    hdf5 = file

    @classmethod
    def sps(
        cls,
        name: str,
        *,
        source_file: Union[str, Path],
        receiver_file: Union[str, Path],
        relation_file: Union[str, Path],
        kind: str = "SPSFiles",
        **kwargs: Any,
    ) -> "SparseSurvey":
        """Create a survey backed by SPS source/receiver/relation files."""

        return cls(
            name=name,
            kind=kind,
            source_file=source_file,
            receiver_file=receiver_file,
            relation_file=relation_file,
            **kwargs,
        )

    @classmethod
    def offset_domain(
        cls,
        name: str,
        *,
        min: Optional[float] = None,
        max: Optional[float] = None,
        metric: str = "horizontal",
        axis: Optional[Sequence[float]] = None,
        absolute: bool = True,
        kind: str = "OffsetDomain",
        **kwargs: Any,
    ) -> "SparseSurvey":
        """Create a survey that selects traces by source-receiver offset."""

        offset_domain: Dict[str, Any] = {
            "metric": metric,
            "absolute": bool(absolute),
        }
        if min is not None:
            offset_domain["min"] = value_and_units_to_fs(min)
        if max is not None:
            offset_domain["max"] = value_and_units_to_fs(max)
        if axis is not None:
            offset_domain["axis"] = list(axis)
        return cls(
            name=name,
            kind=kind,
            offset_domain=offset_domain,
            **kwargs,
        )

    @classmethod
    def from_table(
        cls,
        name: str,
        trace_table: SparseTraceTable,
        **kwargs: Any,
    ) -> "SparseSurvey":
        """Create an inline survey from an existing columnar trace table."""

        return cls(name, trace_table=trace_table, **kwargs)

    @classmethod
    def from_pairs(
        cls,
        name: str,
        pairs: Optional[Iterable[Tuple[int, int]]] = None,
        *,
        source_ids: Optional[Iterable[int]] = None,
        receiver_ids: Optional[Iterable[int]] = None,
        component: ComponentKey = 1,
        receiver_points: Optional[Mapping[int, int]] = None,
        **kwargs: Any,
    ) -> "SparseSurvey":
        """Create a sparse survey from explicit source/receiver pairs.

        Args:
            name: Survey name.
            pairs: Iterable of ``(source_id, receiver_id)`` pairs.
            source_ids: Source ids used with ``receiver_ids`` when ``pairs`` is
                omitted.
            receiver_ids: Receiver ids used with ``source_ids`` when ``pairs``
                is omitted.
            component: Component id/name for created traces.
            receiver_points: Optional mapping from receiver id to receiver point
                id.
            **kwargs: Additional ``SparseSurvey`` constructor arguments.

        Returns:
            Populated ``SparseSurvey``.
        """

        if pairs is None:
            if source_ids is None or receiver_ids is None:
                raise ValueError(
                    "from_pairs requires pairs or source_ids and receiver_ids"
                )
            pairs = zip(source_ids, receiver_ids)
        survey = cls(name, **kwargs)
        if not isinstance(survey._storage, _InlineSparseSurvey):
            raise ValueError("from_pairs requires inline Sparse survey storage")
        survey._storage.table = SparseTraceTable.from_pairs(
            pairs,
            component=component,
            receiver_points=receiver_points,
        )
        return survey

    @classmethod
    def from_product(
        cls,
        name: str,
        *,
        sources: Iterable[int],
        receivers: Iterable[int],
        components: Union[ComponentKey, Iterable[ComponentKey]] = 1,
        receiver_points: Optional[Mapping[int, int]] = None,
        **kwargs: Any,
    ) -> "SparseSurvey":
        """Create traces for the Cartesian product of sources and receivers."""

        survey = cls(name, **kwargs)
        if not isinstance(survey._storage, _InlineSparseSurvey):
            raise ValueError("from_product requires inline Sparse survey storage")
        survey._storage.table = SparseTraceTable.from_product(
            sources=sources,
            receivers=receivers,
            components=components,
            receiver_points=receiver_points,
        )
        return survey

    def add_trace(self, *args: Any, **kwargs: Any) -> SparseTrace:
        """Append a sparse trace row and return it."""

        trace = args[0] if args else SparseTrace(**kwargs)
        trace = _as_trace(trace)
        if not isinstance(self._storage, _InlineSparseSurvey):
            raise ValueError("Cannot add inline traces to external survey storage")
        self._storage.traces.append(trace)
        return trace

    def sampling(self) -> ReceiverSampling:
        """Return receiver sampling that references this survey."""

        return ReceiverSampling(kind=self.kind, survey=self.name)

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        *,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Serialize this sparse survey for solver input."""

        self._reject_unsupported_sample_tables()
        payload: Dict[str, Any] = {"name": self.name, "_type": self.kind}
        kind = self.kind.strip().lower()

        if (
            isinstance(self._storage, _InlineSparseSurvey)
            and self.trace_count > 200
            and ctx is not None
            and ctx.path is not None
            and not any(trace.extra for trace in self._storage.traces)
        ):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name).strip("._")
            if not safe_name:
                safe_name = "survey"
            if safe_name != self.name:
                digest = hashlib.blake2s(
                    self.name.encode("utf-8"),
                    digest_size=4,
                ).hexdigest()
                safe_name = f"{safe_name}-{digest}"
            file = ctx.path / "surveys" / f"{safe_name}.h5"
            self.write_hdf5(file, component_map=component_map)
            return merge_extra(
                {
                    "name": self.name,
                    "_type": "HDF5TraceStore",
                    "layout_file": _path_to_fs(file, ctx),
                },
                self.extra,
                "SparseSurvey",
            )

        if self.layout_file is not None:
            payload["layout_file"] = _path_to_fs(self.layout_file, ctx)
        if self.source_file is not None:
            payload["source_file"] = _path_to_fs(self.source_file, ctx)
        if self.receiver_file is not None:
            payload["receiver_file"] = _path_to_fs(self.receiver_file, ctx)
        if self.relation_file is not None:
            payload["relation_file"] = _path_to_fs(self.relation_file, ctx)
        if isinstance(self._storage, _OffsetSparseSurvey):
            payload["offset_domain"] = copy.deepcopy(dict(self._storage.offset_domain))

        if kind == "sparse":
            payload["traces"] = self._trace_rows(component_map)

        return merge_extra(payload, self.extra, "SparseSurvey")

    def _trace_rows(
        self,
        component_map: Optional[Mapping[str, int]],
    ) -> List[Dict[str, Any]]:
        """Materialize inline trace rows only at the JSON boundary."""

        if not isinstance(self._storage, _InlineSparseSurvey):
            return []
        storage = self._storage
        rows = storage.table.rows(component_map) if storage.table is not None else []
        rows.extend(
            trace.to_fs(trace_id=index, component_map=component_map)
            for index, trace in enumerate(storage.traces, start=len(rows) + 1)
        )
        return rows

    def _trace_columns(
        self,
        component_map: Optional[Mapping[str, int]],
    ) -> Dict[str, np.ndarray]:
        """Return trace columns, preserving the vectorized path when possible."""

        if not isinstance(self._storage, _InlineSparseSurvey):
            return {}
        storage = self._storage
        if storage.table is not None and not storage.traces:
            return storage.table.columns(component_map)
        rows = self._trace_rows(component_map)
        if not rows:
            return {
                name: np.empty(0, dtype=np.int64)
                for name in (
                    "trace_id",
                    "source_id",
                    "receiver_id",
                    "receiver_position_id",
                    "component_id",
                    "component",
                    "channel_number",
                    "field_record",
                    "point_first",
                    "point_last",
                    "n_points",
                    "active",
                )
            }
        columns: Dict[str, np.ndarray] = {}
        required = (
            "trace_id",
            "source_id",
            "receiver_id",
            "receiver_position_id",
            "component_id",
            "component",
            "channel_number",
            "field_record",
            "point_first",
            "point_last",
            "n_points",
            "active",
        )
        for name in required:
            columns[name] = np.asarray([row[name] for row in rows])
        for name in (
            "offset",
            "azimuth",
            "source_name",
            "receiver_name",
            "component_name",
        ):
            if any(name in row for row in rows):
                default = "" if name.endswith("_name") else 0.0
                columns[name] = np.asarray(
                    [row.get(name, default) for row in rows],
                    dtype=object if name.endswith("_name") else np.float64,
                )
        return columns

    def write_hdf5(
        self,
        file: Union[str, Path],
        *,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> Path:
        """Write this survey as a fast solver-compatible HDF5 trace-store layout.

        Args:
            file: Destination HDF5 file path.
            component_map: Optional mapping from component names to one-based
                component ids.

        Returns:
            Path to the written HDF5 file.
        """

        self._reject_unsupported_sample_tables()
        path = Path(file)
        if not isinstance(self._storage, _InlineSparseSurvey):
            raise ValueError("Only inline sparse surveys can be written to HDF5")
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = self._trace_columns(component_map)
        string_dtype = h5py.string_dtype(encoding="utf-8")

        def write_int(group: h5py.Group, name: str, values: Any) -> None:
            group.create_dataset(name, data=np.asarray(values, dtype=np.int32))

        def write_float(group: h5py.Group, name: str, values: Any) -> None:
            group.create_dataset(name, data=np.asarray(values, dtype=np.float64))

        def write_str(group: h5py.Group, name: str, values: Any) -> None:
            group.create_dataset(
                name,
                data=np.asarray([value or "" for value in values], dtype=object),
                dtype=string_dtype,
            )

        with h5py.File(path, "w") as h5:
            survey = h5.require_group("survey")
            write_str(survey, "schema_version", ["fs_seismic_trace_store_v1"])
            write_str(survey, "layout_kind", ["sparse_trace_v1"])

            trace_group = survey.require_group("traces")
            for name in (
                "trace_id",
                "source_id",
                "receiver_id",
                "receiver_position_id",
                "component_id",
                "component",
                "channel_number",
                "field_record",
                "active",
                "point_first",
                "point_last",
                "n_points",
            ):
                write_int(trace_group, name, columns[name])
            if "offset" in columns:
                write_float(trace_group, "offset", columns["offset"])
            if "azimuth" in columns:
                write_float(trace_group, "azimuth", columns["azimuth"])
            self._write_hdf5_catalogs(survey, columns, string_dtype)

        return path

    def _write_hdf5_catalogs(
        self,
        survey: h5py.Group,
        columns: Mapping[str, np.ndarray],
        string_dtype: h5py.Datatype,
    ) -> None:
        """Write compact catalogs directly from trace columns."""

        def write_str(group: h5py.Group, name: str, values: Any) -> None:
            group.create_dataset(
                name,
                data=np.asarray([value or "" for value in values], dtype=object),
                dtype=string_dtype,
            )

        def first_indices(values: np.ndarray) -> np.ndarray:
            _, indices = np.unique(values, return_index=True)
            return np.sort(indices)

        if len(columns["source_id"]):
            indices = first_indices(columns["source_id"])
            group = survey.require_group("sources")
            group.create_dataset(
                "source_id",
                data=np.asarray(columns["source_id"][indices], dtype=np.int32),
            )
            group.create_dataset(
                "field_record",
                data=np.asarray(columns["field_record"][indices], dtype=np.int32),
            )
            names = columns.get("source_name")
            if names is not None and np.any(names[indices] != ""):
                write_str(group, "source_name", names[indices])

        if len(columns["receiver_id"]):
            indices = first_indices(columns["receiver_id"])
            group = survey.require_group("receivers")
            group.create_dataset(
                "receiver_id",
                data=np.asarray(columns["receiver_id"][indices], dtype=np.int32),
            )
            names = columns.get("receiver_name")
            if names is not None and np.any(names[indices] != ""):
                write_str(group, "receiver_name", names[indices])

        if len(columns["receiver_position_id"]):
            indices = first_indices(columns["receiver_position_id"])
            group = survey.require_group("receiver_positions")
            group.create_dataset(
                "receiver_position_id",
                data=np.asarray(
                    columns["receiver_position_id"][indices], dtype=np.int32
                ),
            )
            group.create_dataset(
                "receiver_id",
                data=np.asarray(columns["receiver_id"][indices], dtype=np.int32),
            )
            group.create_dataset(
                "point_first",
                data=np.asarray(columns["point_first"][indices], dtype=np.int32),
            )
            group.create_dataset(
                "point_last",
                data=np.asarray(columns["point_last"][indices], dtype=np.int32),
            )

        if len(columns["component_id"]):
            indices = first_indices(columns["component_id"])
            group = survey.require_group("components")
            group.create_dataset(
                "component_id",
                data=np.asarray(columns["component_id"][indices], dtype=np.int32),
            )
            group.create_dataset(
                "component",
                data=np.asarray(columns["component"][indices], dtype=np.int32),
            )
            names = columns.get("component_name")
            write_str(
                group,
                "component_name",
                names[indices] if names is not None else [""] * len(indices),
            )
