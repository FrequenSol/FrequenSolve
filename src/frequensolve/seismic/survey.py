"""High-level seismic survey authoring helpers."""

from __future__ import annotations

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
