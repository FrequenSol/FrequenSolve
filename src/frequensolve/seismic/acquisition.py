"""Seismic source geometry, source encoding, and receiver acquisition."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from frequensolve.seismic.receivers import (
    CoordsSurfaceCarpet,
    ReceiverDevice,
    ReceiverGroup,
)
from frequensolve.seismic.sources import (
    DistributedSource,
    PointSource,
    SourceEncoding,
    SourceGeometry,
)
from frequensolve.seismic.sparse_survey import ReceiverSampling, SparseSurvey
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
)
from frequensolve.util.named_list import NamedList

__all__ = ["Acquisition"]


def _coerce_source_geometry(value: Any) -> Optional[SourceGeometry]:
    if value is None:
        return None
    if isinstance(value, SourceGeometry):
        return value
    if isinstance(value, Mapping):
        return SourceGeometry.from_fs(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to SourceGeometry")


def _coerce_source_encoding(value: Any) -> Optional[SourceEncoding]:
    if value is None:
        return None
    if isinstance(value, SourceEncoding):
        return value
    if isinstance(value, Mapping):
        return SourceEncoding.from_fs(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to SourceEncoding")


def _encoded_terms(terms: Mapping[Any, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for source, coefficient in terms.items():
        if isinstance(source, PointSource):
            if not source.name:
                raise ValueError("PointSource terms require named source points")
            key = source.name
        else:
            key = str(source)
        out[key] = coefficient
    return out


@dataclass(init=False)
class Acquisition(ExtraFieldsMixin):
    """Source geometry, optional source encoding, and receiver configuration."""

    sources: Optional[SourceGeometry] = None
    source_encoding: Optional[SourceEncoding] = None
    receiver_groups: NamedList = field(default_factory=NamedList)
    surveys: NamedList = field(default_factory=NamedList)
    max_batch: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        sources: Optional[Any] = None,
        source_encoding: Optional[Any] = None,
        receivers: Optional[Any] = None,
        receiver_groups: Optional[Any] = None,
        surveys: Optional[Any] = None,
        max_batch: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if receivers is not None and receiver_groups is not None:
            raise TypeError("Use either receivers or receiver_groups, not both")
        self.sources = sources
        self.source_encoding = source_encoding
        self.receiver_groups = NamedList(
            receiver_groups if receiver_groups is not None else receivers or []
        )
        self.surveys = NamedList(surveys or [])
        self.max_batch = max_batch
        self._init_extra(extra, **kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        self.sources = _coerce_source_geometry(self.sources)
        self.source_encoding = _coerce_source_encoding(self.source_encoding)
        self.receiver_groups = NamedList(
            [
                (
                    group
                    if isinstance(group, ReceiverGroup)
                    else ReceiverGroup.from_fs(group)
                )
                for group in self.receiver_groups
            ]
        )
        self.surveys = NamedList(
            [
                (
                    survey
                    if isinstance(survey, SparseSurvey)
                    else SparseSurvey.from_fs(survey)
                )
                for survey in self.surveys
            ]
        )
        if self.max_batch is not None:
            self.max_batch = int(self.max_batch)

    @property
    def source_geometry(self) -> Optional[SourceGeometry]:
        """Contract-name alias for physical source geometry."""

        return self.sources

    @source_geometry.setter
    def source_geometry(self, value: Any) -> None:
        self.sources = _coerce_source_geometry(value)

    @property
    def receivers(self) -> NamedList:
        """Friendly alias for receiver groups."""

        return self.receiver_groups

    @receivers.setter
    def receivers(self, value: Any) -> None:
        self.receiver_groups = NamedList(value or [])
        self.__post_init__()

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Acquisition":
        """Deserialize acquisition geometry from solver JSON."""

        payload = copy.deepcopy(dict(data))
        payload.pop("schema", None)
        source_geometry = payload.pop("source_geometry", payload.pop("sources", None))
        return cls(
            sources=source_geometry,
            source_encoding=payload.pop("source_encoding", None),
            receiver_groups=NamedList(
                [
                    ReceiverGroup.from_fs(group)
                    for group in payload.pop("receiver_groups", [])
                ]
            ),
            surveys=NamedList(
                [SparseSurvey.from_fs(survey) for survey in payload.pop("surveys", [])]
            ),
            max_batch=payload.pop("max_batch", None),
            extra=payload,
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize acquisition geometry for the v2 solver contract."""

        from ..util.printing import print_warn

        ctx = ctx or ExportContext()

        names = {}
        for group in self.receiver_groups:
            name = group.name
            if name in names:
                i = 1
                while f"{name}_{i}" in names:
                    i += 1
                group.name = f"{name}_{i}"
                print_warn(
                    f"Duplicate receiver group names detected. Renaming receiver group {name} to {group.name}"
                )
            names[group.name] = group.name

        survey_component_maps = self._survey_component_maps()

        payload: Dict[str, Any] = {"schema": "fs-acquisition-2"}
        if self.max_batch is not None:
            payload["max_batch"] = self.max_batch
        if self.sources is not None:
            payload["source_geometry"] = self.sources.to_fs(ctx)
        if self.source_encoding is not None:
            payload["source_encoding"] = self.source_encoding.to_fs(ctx)
        payload["receiver_groups"] = [
            group.to_fs(ctx) for group in self.receiver_groups
        ]
        if self.surveys:
            payload["surveys"] = [
                survey.to_fs(ctx, component_map=survey_component_maps.get(survey.name))
                for survey in self.surveys
            ]
        return merge_extra(payload, self.extra, "Acquisition")

    def set_sources(self, sources: Any) -> SourceGeometry:
        """Set physical source geometry and return it."""

        self.sources = _coerce_source_geometry(sources)
        return self.sources

    def add_sources(
        self,
        *,
        kind: str,
        coords: Any,
        names: Optional[Sequence[str]] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        domain: Optional[int] = None,
        direction: Optional[Any] = None,
        amplitude: Optional[Any] = None,
        mechanism: Optional[Any] = None,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """Append inline physical source points and return their names.

        ``amplitude`` accepts a dimensionless multiplier, a Pint quantity, or
        a ``{"value": ..., "units": ...}`` physical source strength. Sauce
        expects force units for vector/dipole sources and moment units for
        scalar/tensor/monopole sources.
        """

        geometry = SourceGeometry.points(
            kind=kind,
            coords=coords,
            names=names,
            units=units,
            system=system,
            domain=domain,
            direction=direction,
            amplitude=amplitude,
            mechanism=mechanism,
            defaults=defaults,
        )
        return self._append_inline_sources(geometry)

    add_source_points = add_sources

    def _append_inline_sources(self, geometry: SourceGeometry) -> List[str]:
        if self.sources is None:
            self.sources = geometry
            return geometry.point_names()
        if self.sources.geometry_type != "Inline" or geometry.geometry_type != "Inline":
            raise ValueError("Cannot append inline sources to file-backed geometry")
        if self.sources.kind != geometry.kind:
            raise ValueError("All source points in one geometry must share a kind")
        if self.sources.domain != geometry.domain:
            raise ValueError("All source points in one geometry must share a domain")
        if self.sources.defaults != geometry.defaults:
            raise ValueError(
                "All appended source points must share source-geometry defaults"
            )
        existing_names = set(self.sources.point_names())
        new_names = geometry.point_names()
        duplicates = existing_names.intersection(new_names)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate source names: {names}")
        self.sources.sources.extend(geometry.sources)
        return new_names

    def add_distributed_source(
        self, name: str, terms: Mapping[Any, Any]
    ) -> DistributedSource:
        """Append one distributed source field."""

        field_obj = DistributedSource.named(name, _encoded_terms(terms))
        if self.source_encoding is None:
            self.source_encoding = SourceEncoding.named([field_obj])
        elif self.source_encoding.encoding_type != "Named":
            raise ValueError(
                "add_distributed_source can only append to Named source encoding"
            )
        else:
            known = set(self.source_encoding.field_names())
            if name in known:
                raise ValueError(f"Distributed source {name!r} already exists")
            self.source_encoding.fields.append(field_obj)
        return field_obj

    def add_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        domain: Optional[int] = None,
        **kwargs: Any,
    ) -> ReceiverGroup:
        """Add a receiver group with common device and coordinates."""

        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            kwargs
        )
        if deprecated_frame_keys:
            raise TypeError(
                "add_receiver_group frame is no longer supported; receiver coordinates are physical"
            )

        group = ReceiverGroup(
            name=name,
            device=device,
            coordinates=coords,
            domain=domain,
            **kwargs,
        )
        self.receiver_groups.append(group)
        return group

    def add_receiver_carpet(
        self,
        name: str,
        device: ReceiverDevice,
        *,
        surface: Any,
        x: Any,
        y: Optional[Any] = None,
        units: Optional[Any] = None,
        above: Optional[Any] = None,
        below: Optional[Any] = None,
        domain: Optional[int] = None,
        **kwargs: Any,
    ) -> ReceiverGroup:
        """Add a receiver group on a tensor-product carpet."""

        coords = _carpet_coordinates(
            x=x,
            y=y,
            surface=surface,
            units=units,
            above=above,
            below=below,
        )
        return self.add_receiver_group(
            name=name,
            device=device,
            coords=coords,
            domain=domain,
            **kwargs,
        )

    def add_survey(self, survey: SparseSurvey) -> SparseSurvey:
        """Add or replace a named sparse survey layout."""

        if isinstance(survey, dict):
            survey = SparseSurvey.from_fs(survey)
        try:
            self.surveys[survey.name] = survey
        except ValueError:
            self.surveys.append(survey)
        return survey

    def add_sparse_survey(self, name: str, traces=None, **kwargs: Any) -> SparseSurvey:
        """Create and add a named inline sparse survey."""

        return self.add_survey(SparseSurvey(name=name, traces=traces, **kwargs))

    def add_sparse_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        survey: Optional[Union[str, SparseSurvey, Dict]] = None,
        domain: Optional[int] = None,
        **kwargs: Any,
    ) -> ReceiverGroup:
        """Add a receiver group that samples traces from a named sparse survey."""

        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            kwargs
        )
        if deprecated_frame_keys:
            raise TypeError(
                "add_sparse_receiver_group frame is no longer supported; receiver coordinates are physical"
            )

        if survey is None:
            raise ValueError(
                "add_sparse_receiver_group requires a survey name or SparseSurvey"
            )
        if isinstance(survey, dict):
            survey = SparseSurvey.from_fs(survey)
        if isinstance(survey, SparseSurvey):
            self.add_survey(survey)
            sampling = survey.sampling()
        else:
            sampling = ReceiverSampling.sparse(str(survey))

        group = ReceiverGroup(
            name=name,
            device=device,
            coordinates=coords,
            domain=domain,
            sampling=sampling,
            **kwargs,
        )
        self.receiver_groups.append(group)
        return group

    def list_fields(self, recv_name: str = "") -> List[str]:
        """List receiver output field selectors."""

        field_list = []
        groups = [self.receiver_group(recv_name)] if recv_name else self.receiver_groups
        for group in groups:
            for field_obj in group.device.components:
                field_list.append(f"{group.name}:{field_obj.name}")
        return field_list

    def known_source_field_count(self) -> Optional[int]:
        """Return known addressable RHS/source-field count, or ``None``."""

        if self.source_encoding is not None:
            return self.source_encoding.field_count
        if self.sources is None:
            return 0
        return self.sources.point_count

    def source_field_count(self) -> int:
        """Return source-field count, using 0 when external metadata is unknown."""

        count = self.known_source_field_count()
        return int(count or 0)

    def source_field_ids(self) -> List[int]:
        """Return one-based source-field ids when the count is known."""

        count = self.known_source_field_count()
        if count is None:
            return []
        return list(range(1, int(count) + 1))

    def source_field_names(self) -> List[str]:
        """Return source-field names when known."""

        if self.source_encoding is not None:
            return self.source_encoding.field_names()
        if self.sources is None:
            return []
        return self.sources.point_names()

    def source_point_names(self) -> List[str]:
        """Return physical source-point names when known."""

        if self.sources is None:
            return []
        return self.sources.point_names()

    def receiver_group(self, name: str) -> ReceiverGroup:
        """Return a receiver group by name."""

        return self.receiver_groups[name]

    def receiver_coords(self, group: Optional[str] = None):
        """Return receiver coordinates."""

        if group is None:
            group_locations = {}
            for receiver_group in self.receiver_groups:
                group_locations[receiver_group.name] = receiver_group.coordinates.get()
            return group_locations
        return self.receiver_groups[group].coordinates.get()

    def source_point_coords(self) -> np.ndarray:
        """Return physical source-point coordinates."""

        if self.sources is None:
            return np.empty((0, 0), dtype=float)
        return self.sources.coordinates()

    def source_coords(self, src: Optional[int] = None):
        """Return source-field reference coordinates."""

        if self.sources is None:
            coords = np.empty((0, 0), dtype=float)
        elif self.source_encoding is None:
            coords = self.sources.coordinates()
        else:
            coords = self.source_encoding.reference_coordinates(self.sources)
        if src is None:
            return coords
        return coords[int(src) - 1]

    def offsets(self, src: int, group: str) -> Dict:
        """Return horizontal source-field/receiver offsets."""

        diff = self.receiver_coords(group) - self.source_coords(src)
        return np.hypot(diff[:, 0], diff[:, 1])

    def _survey_component_maps(self) -> Dict[str, Dict[str, int]]:
        maps: Dict[str, Dict[str, int]] = {}
        for group in self.receiver_groups:
            survey_name = getattr(group, "survey", None)
            if not survey_name:
                continue
            component_map = maps.setdefault(survey_name, {})
            for index, component in enumerate(group.device.components, start=1):
                component_map.setdefault(str(index), index)
                component_map.setdefault(component.name, index)
                component_map.setdefault(component.name.lower(), index)
                component_map.setdefault(component.field, index)
                component_map.setdefault(component.field.lower(), index)
        return maps


def _carpet_coordinates(
    *,
    surface: Any,
    x: Any,
    y: Optional[Any],
    units: Optional[Any],
    above: Optional[Any],
    below: Optional[Any],
) -> Any:
    points_grid = getattr(surface, "points_grid", None)
    if not callable(points_grid):
        raise TypeError(
            "surface must provide points_grid(...), such as sim.model_surface(...)"
        )
    compact = CoordsSurfaceCarpet.try_from_surface(
        surface,
        x=x,
        y=y,
        units=units,
        above=above,
        below=below,
    )
    if compact is not None and compact.size > 200:
        return compact
    return points_grid(
        x,
        y,
        units=units,
        above=above,
        below=below,
    )
