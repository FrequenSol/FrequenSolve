"""Sparse seismic survey authoring helpers.

These classes mirror fast solver sparse receiver layout contracts while keeping the
Python-facing syntax compact. They intentionally do not read server-side files
when exporting JSON.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from frequensolve.units import value_and_units_to_fs
from frequensolve.util.mixins import ExportContext, ExtraFieldsMixin, merge_extra

__all__ = [
    "ReceiverSampling",
    "SparseSurvey",
    "SparseTrace",
    "EvalSample",
    "TraceSample",
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


def _as_eval_sample(value: Union["EvalSample", Mapping[str, Any]]) -> "EvalSample":
    if isinstance(value, EvalSample):
        return value
    if isinstance(value, Mapping):
        return EvalSample.from_fs(value)
    raise TypeError(f"Cannot convert {type(value)} to EvalSample")


def _as_trace_sample(value: Union["TraceSample", Mapping[str, Any]]) -> "TraceSample":
    if isinstance(value, TraceSample):
        return value
    if isinstance(value, Mapping):
        return TraceSample.from_fs(value)
    raise TypeError(f"Cannot convert {type(value)} to TraceSample")


@dataclass
class ReceiverSampling(ExtraFieldsMixin):
    """Receiver-group sampling block.

    Dense receiver groups omit this object. Sparse receiver groups usually use
    ``ReceiverSampling.sparse("survey_name")`` or are created through
    ``Acquisition.add_sparse_receiver_group``.
    """

    kind: str = "Sparse"
    survey: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def sparse(cls, survey: Union[str, "SparseSurvey"]) -> "ReceiverSampling":
        return cls.from_value(survey)

    @classmethod
    def from_value(cls, value: Any) -> Optional["ReceiverSampling"]:
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
        payload = copy.deepcopy(dict(data))
        kind = payload.pop("_type", payload.pop("kind", "Sparse"))
        survey = payload.pop("survey", None)
        return cls(kind=kind, survey=survey, extra=payload)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
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

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SparseTrace":
        payload = copy.deepcopy(dict(data))
        if "recv_pos_id" in payload and "receiver_position_id" not in payload:
            payload["receiver_position_id"] = payload.pop("recv_pos_id")
        return cls(**payload)

    def to_fs(
        self,
        trace_id: Optional[int] = None,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
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


@dataclass(init=False)
class EvalSample(ExtraFieldsMixin):
    """Optional sparse sample row used for weighted/fiber-style traces."""

    sample_id: Optional[int]
    point_id: int
    receiver_position_id: Optional[int]
    x: Optional[Sequence[float]]
    direction: Optional[Sequence[float]]
    extra: Dict[str, Any]

    def __init__(
        self,
        *,
        point: Optional[int] = None,
        point_id: Optional[int] = None,
        sample_id: Optional[int] = None,
        receiver_position: Optional[int] = None,
        receiver_position_id: Optional[int] = None,
        x: Optional[Sequence[float]] = None,
        direction: Optional[Sequence[float]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if point_id is None:
            point_id = point
        if receiver_position_id is None:
            receiver_position_id = receiver_position
        if point_id is None:
            raise ValueError("EvalSample requires point/point_id")
        self.sample_id = None if sample_id is None else int(sample_id)
        self.point_id = int(point_id)
        self.receiver_position_id = (
            None if receiver_position_id is None else int(receiver_position_id)
        )
        self.x = None if x is None else list(x)
        self.direction = None if direction is None else list(direction)
        self._init_extra(extra, **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "EvalSample":
        payload = copy.deepcopy(dict(data))
        if "recveiver_position_id" in payload and "receiver_position_id" not in payload:
            payload["receiver_position_id"] = payload.pop("recveiver_position_id")
        return cls(**payload)

    def to_fs(self, sample_id: Optional[int] = None) -> Dict[str, Any]:
        out_sample_id = int(self.sample_id or sample_id or 0)
        if out_sample_id <= 0:
            raise ValueError("EvalSample requires a positive sample_id or export row")
        payload: Dict[str, Any] = {
            "sample_id": out_sample_id,
            "point_id": self.point_id,
        }
        if self.receiver_position_id is not None:
            payload["receiver_position_id"] = self.receiver_position_id
            # The fast solver's current JSON reader has this misspelling; emit both until
            # the backend accepts the correctly spelled field.
            payload["recveiver_position_id"] = self.receiver_position_id
        if self.x is not None:
            payload["x"] = list(self.x)
        if self.direction is not None:
            payload["direction"] = list(self.direction)
        return merge_extra(payload, self.extra, "EvalSample")


@dataclass(init=False)
class TraceSample(ExtraFieldsMixin):
    """Optional sparse trace/sample weight row."""

    trace_row: int
    sample_id: int
    component: Optional[ComponentKey]
    weight: float
    extra: Dict[str, Any]

    def __init__(
        self,
        *,
        trace: Optional[int] = None,
        trace_row: Optional[int] = None,
        sample: Optional[int] = None,
        sample_id: Optional[int] = None,
        component: Optional[ComponentKey] = 1,
        weight: float = 1.0,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if trace_row is None:
            trace_row = trace
        if sample_id is None:
            sample_id = sample
        if trace_row is None or sample_id is None:
            raise ValueError(
                "TraceSample requires trace/trace_row and sample/sample_id"
            )
        self.trace_row = int(trace_row)
        self.sample_id = int(sample_id)
        self.component = component
        self.weight = float(weight)
        self._init_extra(extra, **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "TraceSample":
        return cls(**copy.deepcopy(dict(data)))

    def to_fs(
        self, component_map: Optional[Mapping[str, int]] = None
    ) -> Dict[str, Any]:
        payload = {
            "trace_row": self.trace_row,
            "sample_id": self.sample_id,
            "component": _resolve_component(self.component, component_map),
            "weight": self.weight,
        }
        return merge_extra(payload, self.extra, "TraceSample")


@dataclass(init=False)
class SparseSurvey(ExtraFieldsMixin):
    """Named fast solver sparse survey layout.

    The default form exports inline JSON accepted by the fast solver's ``Sparse`` layout
    reader. Use ``SparseSurvey.file(...)`` for an existing HDF5 trace store or
    ``SparseSurvey.sps(...)`` for SPS source/receiver/relation files.
    """

    name: str
    kind: str
    traces: List[SparseTrace]
    eval_samples: List[EvalSample]
    trace_samples: List[TraceSample]
    layout_file: Optional[Union[str, Path]]
    source_file: Optional[Union[str, Path]]
    receiver_file: Optional[Union[str, Path]]
    relation_file: Optional[Union[str, Path]]
    offset_domain: Optional[Mapping[str, Any]]
    extra: Dict[str, Any]
    _proj_path: Optional[Path]
    _rel_path: Optional[Path]

    def __init__(
        self,
        name: str,
        traces: Optional[Iterable[Union[SparseTrace, Mapping[str, Any]]]] = None,
        *,
        kind: Optional[str] = None,
        eval_samples: Optional[Iterable[Union[EvalSample, Mapping[str, Any]]]] = None,
        trace_samples: Optional[Iterable[Union[TraceSample, Mapping[str, Any]]]] = None,
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
        self.kind = kind
        self.traces = [_as_trace(trace) for trace in (traces or [])]
        self.eval_samples = [_as_eval_sample(sample) for sample in (eval_samples or [])]
        self.trace_samples = [
            _as_trace_sample(sample) for sample in (trace_samples or [])
        ]
        self.layout_file = layout_file
        self.source_file = source_file
        self.receiver_file = receiver_file
        self.relation_file = relation_file
        self.offset_domain = copy.deepcopy(dict(offset_domain or {})) or None
        self._init_extra(extra, **kwargs)
        self._proj_path = None
        self._rel_path = None

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SparseSurvey":
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
        survey = cls(name, **kwargs)
        if pairs is None:
            if source_ids is None or receiver_ids is None:
                raise ValueError(
                    "from_pairs requires pairs or source_ids and receiver_ids"
                )
            pairs = zip(source_ids, receiver_ids)
        for source_id, receiver_id in pairs:
            point = (
                receiver_points.get(receiver_id)
                if receiver_points is not None
                else None
            )
            survey.add_trace(
                source=source_id, receiver=receiver_id, component=component, point=point
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
        if isinstance(components, (str, int)):
            component_list = [components]
        else:
            component_list = list(components)
        survey = cls(name, **kwargs)
        for source_id in sources:
            for receiver_id in receivers:
                point = (
                    receiver_points.get(receiver_id)
                    if receiver_points is not None
                    else None
                )
                for component in component_list:
                    survey.add_trace(
                        source=source_id,
                        receiver=receiver_id,
                        component=component,
                        point=point,
                    )
        return survey

    def add_trace(self, *args: Any, **kwargs: Any) -> SparseTrace:
        trace = args[0] if args else SparseTrace(**kwargs)
        trace = _as_trace(trace)
        self.traces.append(trace)
        return trace

    def add_eval_sample(self, *args: Any, **kwargs: Any) -> EvalSample:
        sample = args[0] if args else EvalSample(**kwargs)
        sample = _as_eval_sample(sample)
        self.eval_samples.append(sample)
        return sample

    def add_trace_sample(self, *args: Any, **kwargs: Any) -> TraceSample:
        sample = args[0] if args else TraceSample(**kwargs)
        sample = _as_trace_sample(sample)
        self.trace_samples.append(sample)
        return sample

    def sampling(self) -> ReceiverSampling:
        return ReceiverSampling(kind=self.kind, survey=self.name)

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        *,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": self.name, "_type": self.kind}
        kind = self.kind.strip().lower()

        if self.layout_file is not None:
            payload["layout_file"] = _path_to_fs(self.layout_file, ctx)
        if self.source_file is not None:
            payload["source_file"] = _path_to_fs(self.source_file, ctx)
        if self.receiver_file is not None:
            payload["receiver_file"] = _path_to_fs(self.receiver_file, ctx)
        if self.relation_file is not None:
            payload["relation_file"] = _path_to_fs(self.relation_file, ctx)
        if self.offset_domain is not None:
            payload["offset_domain"] = copy.deepcopy(dict(self.offset_domain))

        if kind == "sparse" or self.traces:
            payload["traces"] = [
                trace.to_fs(trace_id=i, component_map=component_map)
                for i, trace in enumerate(self.traces, start=1)
            ]
            if self.eval_samples:
                payload["eval_samples"] = [
                    sample.to_fs(sample_id=i)
                    for i, sample in enumerate(self.eval_samples, start=1)
                ]
            if self.trace_samples:
                payload["trace_samples"] = [
                    sample.to_fs(component_map=component_map)
                    for sample in self.trace_samples
                ]

        return merge_extra(payload, self.extra, "SparseSurvey")

    def write_hdf5(
        self,
        file: Union[str, Path],
        *,
        component_map: Optional[Mapping[str, int]] = None,
    ) -> Path:
        """Write this survey as a fast solver-compatible HDF5 trace-store layout."""

        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        traces = [
            trace.to_fs(trace_id=i, component_map=component_map)
            for i, trace in enumerate(self.traces, start=1)
        ]
        string_dtype = h5py.string_dtype(encoding="utf-8")

        def write_int(group: h5py.Group, name: str, values: List[int]) -> None:
            group.create_dataset(name, data=np.asarray(values, dtype=np.int32))

        def write_float(group: h5py.Group, name: str, values: List[float]) -> None:
            group.create_dataset(name, data=np.asarray(values, dtype=np.float64))

        def write_str(group: h5py.Group, name: str, values: List[str]) -> None:
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
            write_int(trace_group, "trace_id", [row["trace_id"] for row in traces])
            write_int(trace_group, "source_id", [row["source_id"] for row in traces])
            write_int(
                trace_group, "receiver_id", [row["receiver_id"] for row in traces]
            )
            write_int(
                trace_group,
                "receiver_position_id",
                [row["receiver_position_id"] for row in traces],
            )
            write_int(
                trace_group, "component_id", [row["component_id"] for row in traces]
            )
            write_int(trace_group, "component", [row["component"] for row in traces])
            write_int(
                trace_group, "channel_number", [row["channel_number"] for row in traces]
            )
            write_int(
                trace_group, "field_record", [row["field_record"] for row in traces]
            )
            write_int(
                trace_group,
                "active",
                [1 if row.get("active", True) else 0 for row in traces],
            )
            write_int(
                trace_group, "point_first", [row["point_first"] for row in traces]
            )
            write_int(trace_group, "point_last", [row["point_last"] for row in traces])
            write_int(trace_group, "n_points", [row["n_points"] for row in traces])
            write_float(
                trace_group, "offset", [row.get("offset", 0.0) for row in traces]
            )
            write_float(
                trace_group, "azimuth", [row.get("azimuth", 0.0) for row in traces]
            )
            write_str(
                trace_group,
                "source_name",
                [row.get("source_name", "") for row in traces],
            )
            write_str(
                trace_group,
                "receiver_name",
                [row.get("receiver_name", "") for row in traces],
            )
            write_str(
                trace_group,
                "component_name",
                [row.get("component_name", "") for row in traces],
            )

            if self.eval_samples:
                evals = [
                    sample.to_fs(sample_id=i)
                    for i, sample in enumerate(self.eval_samples, start=1)
                ]
                group = survey.require_group("eval_samples")
                write_int(group, "sample_id", [row["sample_id"] for row in evals])
                write_int(group, "point_id", [row["point_id"] for row in evals])
                write_int(
                    group,
                    "receiver_position_id",
                    [row.get("receiver_position_id", 0) for row in evals],
                )

            if self.trace_samples:
                samples = [
                    sample.to_fs(component_map=component_map)
                    for sample in self.trace_samples
                ]
                group = survey.require_group("trace_samples")
                write_int(group, "trace_row", [row["trace_row"] for row in samples])
                write_int(group, "sample_id", [row["sample_id"] for row in samples])
                write_int(group, "component", [row["component"] for row in samples])
                group.create_dataset(
                    "weight",
                    data=np.asarray(
                        [row["weight"] for row in samples], dtype=np.float32
                    ),
                )

            self._write_hdf5_catalogs(survey, traces, string_dtype)

        return path

    def _write_hdf5_catalogs(
        self,
        survey: h5py.Group,
        traces: List[Mapping[str, Any]],
        string_dtype: h5py.Datatype,
    ) -> None:
        sources: Dict[int, Mapping[str, Any]] = {}
        receivers: Dict[int, Mapping[str, Any]] = {}
        receiver_positions: Dict[int, Mapping[str, Any]] = {}
        components: Dict[int, Mapping[str, Any]] = {}

        for row in traces:
            sources.setdefault(row["source_id"], row)
            receivers.setdefault(row["receiver_id"], row)
            receiver_positions.setdefault(row["receiver_position_id"], row)
            components.setdefault(row["component_id"], row)

        def write_str(group: h5py.Group, name: str, values: List[str]) -> None:
            group.create_dataset(
                name,
                data=np.asarray([value or "" for value in values], dtype=object),
                dtype=string_dtype,
            )

        if sources:
            group = survey.require_group("sources")
            ids = list(sources)
            group.create_dataset("source_id", data=np.asarray(ids, dtype=np.int32))
            group.create_dataset(
                "field_record",
                data=np.asarray(
                    [sources[i].get("field_record", i) for i in ids], dtype=np.int32
                ),
            )
            write_str(
                group, "source_name", [sources[i].get("source_name", "") for i in ids]
            )

        if receivers:
            group = survey.require_group("receivers")
            ids = list(receivers)
            group.create_dataset("receiver_id", data=np.asarray(ids, dtype=np.int32))
            write_str(
                group,
                "receiver_name",
                [receivers[i].get("receiver_name", "") for i in ids],
            )

        if receiver_positions:
            group = survey.require_group("receiver_positions")
            ids = list(receiver_positions)
            group.create_dataset(
                "receiver_position_id", data=np.asarray(ids, dtype=np.int32)
            )
            group.create_dataset(
                "receiver_id",
                data=np.asarray(
                    [receiver_positions[i].get("receiver_id", i) for i in ids],
                    dtype=np.int32,
                ),
            )
            group.create_dataset(
                "point_first",
                data=np.asarray(
                    [receiver_positions[i].get("point_first", i) for i in ids],
                    dtype=np.int32,
                ),
            )
            group.create_dataset(
                "point_last",
                data=np.asarray(
                    [receiver_positions[i].get("point_last", i) for i in ids],
                    dtype=np.int32,
                ),
            )

        if components:
            group = survey.require_group("components")
            ids = list(components)
            group.create_dataset("component_id", data=np.asarray(ids, dtype=np.int32))
            group.create_dataset(
                "component",
                data=np.asarray(
                    [components[i].get("component", i) for i in ids], dtype=np.int32
                ),
            )
            write_str(
                group,
                "component_name",
                [components[i].get("component_name", "") for i in ids],
            )

    def _set_path(self, proj_path: Path, rel_path: Path) -> None:
        self._proj_path = proj_path
        self._rel_path = rel_path
