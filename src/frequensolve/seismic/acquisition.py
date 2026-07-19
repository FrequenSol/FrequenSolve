"""Seismic source geometry, source encoding, and receiver acquisition."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from frequensolve.geometry.frame import CoordinateValue, Direction
from frequensolve.seismic.receivers import (
    ReceiverDevice,
    ReceiverGroup,
    coordinate_array_metadata,
)
from frequensolve.seismic.sources import (
    CompoundSource,
    DistributedSource,
    PointSource,
    RuptureSource,
    SourceEncoding,
    SourceGeometry,
    SourceGroup,
)
from frequensolve.seismic.sparse_survey import ReceiverSampling, SparseSurvey
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
    warn_deprecated_path_api,
)
from frequensolve.util.named_list import NamedList

__all__ = ["Acquisition"]

_SOURCE_KINDS = {"scalar", "vector", "tensor", "monopole", "dipole"}


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
    encoded: Dict[str, Any] = {}
    for source, coefficient in terms.items():
        if isinstance(source, PointSource):
            if not source.name:
                raise ValueError("PointSource terms require named source points")
            source_name = source.name
        else:
            source_name = str(source)
        encoded[source_name] = coefficient
    return encoded


def _deprecated_source_groups() -> None:
    warnings.warn(
        "Acquisition.source_groups is deprecated; use source_geometry, "
        "source_encoding, source_point_names(), and source_field_names().",
        DeprecationWarning,
        stacklevel=3,
    )


class _SourceGroupCompatibilityView(NamedList):
    """Read-only list returned by the deprecated ``source_groups`` property."""

    @staticmethod
    def _reject_mutation(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "Acquisition.source_groups is a read-only compatibility view; "
            "use add_sources(), set_sources(), or set_source_encoding()"
        )

    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    __setitem__ = _reject_mutation


def _coefficient_value(value: Any) -> complex:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    return complex(value)


def _coefficient_magnitude(value: Any) -> float:
    return abs(_coefficient_value(value))


def _coefficient_is_one(value: Any) -> bool:
    return _coefficient_value(value) == complex(1.0, 0.0)


def _coordinate_value(value: Any) -> np.ndarray:
    values, _units, _system = coordinate_array_metadata(value)
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 2 and len(result) == 1:
        return result[0]
    return result


def _weighted_direction(direction: Any, coefficient: Any) -> np.ndarray:
    weight = _coefficient_value(coefficient)
    if weight.imag != 0.0:
        raise ValueError(
            "Complex source encoding cannot be represented by legacy source_groups"
        )
    if isinstance(direction, Direction):
        direction = direction.value
    if direction is None:
        direction = [1.0]
    return np.asarray(direction, dtype=float) * float(weight.real)


@dataclass(init=False)
class Acquisition(ExtraFieldsMixin):
    """Physical sources, optional RHS encoding, receivers, and surveys."""

    source_geometry: Optional[SourceGeometry] = None
    source_encoding: Optional[SourceEncoding] = None
    receiver_groups: NamedList = field(default_factory=NamedList)
    surveys: NamedList = field(default_factory=NamedList)
    max_batch: Optional[int] = None
    write_vtk: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __init__(
        self,
        *,
        source_geometry: Optional[Any] = None,
        sources: Optional[Any] = None,
        source_encoding: Optional[Any] = None,
        source_groups: Optional[Any] = None,
        receivers: Optional[Any] = None,
        receiver_groups: Optional[Any] = None,
        surveys: Optional[Any] = None,
        max_batch: Optional[int] = None,
        write_vtk: Optional[bool] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if source_geometry is not None and sources is not None:
            raise TypeError("Use either source_geometry or sources, not both")
        if source_groups is not None and (
            source_geometry is not None or sources is not None
        ):
            raise TypeError(
                "Legacy source_groups cannot be combined with source_geometry"
            )
        if receivers is not None and receiver_groups is not None:
            raise TypeError("Use either receivers or receiver_groups, not both")

        extra_fields = copy.deepcopy(dict(extra or {}))
        if "source_groups" in extra_fields:
            raise ValueError(
                "Acquisition.extra cannot contain legacy source_groups; pass "
                "source_groups as the migration argument instead"
            )

        self.source_geometry = _coerce_source_geometry(
            source_geometry if source_geometry is not None else sources
        )
        self.source_encoding = _coerce_source_encoding(source_encoding)
        self.receiver_groups = NamedList(
            receiver_groups if receiver_groups is not None else receivers or []
        )
        self.surveys = NamedList(surveys or [])
        self.max_batch = None if max_batch is None else int(max_batch)
        self.write_vtk = None if write_vtk is None else bool(write_vtk)
        self._init_extra(extra_fields, **kwargs)
        self._proj_path = None
        self._rel_path = None
        self._coerce_receivers_and_surveys()
        if source_groups is not None:
            self._load_legacy_source_groups(source_groups)

    @property
    def sources(self) -> Optional[SourceGeometry]:
        """Friendly alias for :attr:`source_geometry`."""

        return self.source_geometry

    @sources.setter
    def sources(self, value: Any) -> None:
        self.source_geometry = _coerce_source_geometry(value)

    @property
    def receivers(self) -> NamedList:
        """Friendly alias for receiver groups."""

        return self.receiver_groups

    @receivers.setter
    def receivers(self, value: Any) -> None:
        self.receiver_groups = NamedList(value or [])
        self._coerce_receivers_and_surveys()

    def _coerce_receivers_and_surveys(self) -> None:
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
        if self.max_batch is not None and self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")

    def __bool__(self) -> bool:
        """Return whether this acquisition contains authored state."""

        return bool(
            self.source_geometry is not None
            or self.source_encoding is not None
            or self.receiver_groups
            or self.surveys
            or self.max_batch is not None
            or self.write_vtk is not None
            or self.extra
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Acquisition":
        """Deserialize current or legacy acquisition JSON."""

        payload = copy.deepcopy(dict(data))
        schema = payload.pop("schema", None)
        if schema not in {None, "fs-acquisition-1", "fs-acquisition-2"}:
            raise ValueError(f"Unsupported acquisition schema {schema!r}")
        source_geometry = payload.pop("source_geometry", payload.pop("sources", None))
        return cls(
            source_geometry=source_geometry,
            source_encoding=payload.pop("source_encoding", None),
            source_groups=payload.pop("source_groups", None),
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
            write_vtk=payload.pop("write_vtk", None),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict:
        """Serialize acquisition geometry for solver input.

        Args:
            ctx: Optional export context used by source, receiver, and survey
                serializers.

        Returns:
            JSON-compatible acquisition block.
        """

        from ..util.printing import print_warn

        self._validate_export_contract()

        ctx = ctx or ExportContext(self._proj_path, self._rel_path)

        # Ensure receiver groups have unique names
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

        payload: Dict[str, Any] = {
            "schema": "fs-acquisition-2",
            "receiver_groups": [group.to_fs(ctx) for group in self.receiver_groups],
        }
        if self.max_batch is not None:
            payload["max_batch"] = self.max_batch
        if self.write_vtk is not None:
            payload["write_vtk"] = self.write_vtk
        if self.source_geometry is not None:
            payload["source_geometry"] = self.source_geometry.to_fs(ctx)
        if self.source_encoding is not None:
            payload["source_encoding"] = self.source_encoding.to_fs(ctx)
        if self.surveys:
            payload["surveys"] = [
                (
                    survey.to_fs(
                        ctx, component_map=survey_component_maps.get(survey.name)
                    )
                )
                for survey in self.surveys
            ]
        return merge_extra(payload, self.extra, "Acquisition")

    def _validate_export_contract(self) -> None:
        """Reject incomplete or inconsistent acquisition-v2 exports."""

        if "source_groups" in self.extra:
            raise ValueError(
                "Acquisition.extra cannot contain legacy source_groups in a "
                "current acquisition export"
            )
        geometry = self.source_geometry
        if geometry is None:
            raise ValueError("fs-acquisition-2 requires source_geometry before export")
        if geometry.kind not in _SOURCE_KINDS:
            choices = ", ".join(sorted(_SOURCE_KINDS))
            raise ValueError(
                f"Unsupported source kind {geometry.kind!r}; use one of: {choices}"
            )
        if geometry.geometry_type == "Inline":
            if not geometry.sources:
                raise ValueError("Inline source_geometry requires source points")
            for index, source in enumerate(geometry.sources):
                if source.kind is not None and (
                    str(source.kind).strip().lower() != geometry.kind
                ):
                    raise ValueError(
                        "Inline source point kind must match source_geometry.kind: "
                        f"sources[{index}] is {source.kind!r}, geometry is "
                        f"{geometry.kind!r}"
                    )
            names = geometry.point_names()
            if len(names) != len(set(names)):
                raise ValueError("Inline source names must be unique")
        else:
            names = []

        encoding = self.source_encoding
        if encoding is None or encoding.encoding_type == "HDF5Dense":
            return
        field_names = encoding.field_names()
        if len(field_names) != len(set(field_names)):
            raise ValueError("Source-encoding field names must be unique")
        if encoding.encoding_type == "Named" and names:
            if geometry.geometry_type == "Inline" and any(
                source.name is None for source in geometry.sources
            ):
                raise ValueError(
                    "Named source encoding requires explicit names for every "
                    "inline physical source point"
                )
            known_names = set(names)
            for field_obj in encoding.fields:
                unknown = sorted(set(field_obj.terms).difference(known_names))
                if unknown:
                    raise ValueError(
                        f"Source encoding references unknown sources: {unknown}"
                    )
        elif encoding.encoding_type == "JsonDense" and names:
            for field_obj in encoding.fields:
                coefficients = field_obj.coefficients
                if coefficients is None or len(coefficients) != len(names):
                    raise ValueError(
                        "JsonDense coefficient count must match physical "
                        "source-point count"
                    )

    def set_sources(self, sources: Any) -> SourceGeometry:
        """Set physical source geometry and return it."""

        geometry = _coerce_source_geometry(sources)
        if geometry is None:
            raise TypeError("sources cannot be None")
        self.source_geometry = geometry
        return geometry

    def set_source_encoding(self, encoding: Optional[Any]) -> Optional[SourceEncoding]:
        """Set or clear explicit source encoding."""

        self.source_encoding = _coerce_source_encoding(encoding)
        return self.source_encoding

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
        """Append physical point sources and return their stable names."""

        existing = self.source_point_count()
        direction_array = None
        if direction is not None and not isinstance(direction, Mapping):
            try:
                candidate = np.asarray(direction, dtype=float)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate.ndim == 2:
                direction_array = candidate

        probe = SourceGeometry.points(
            kind=kind,
            coords=coords,
            names=names,
            units=units,
            system=system,
            domain=domain,
            direction=None if direction_array is not None else direction,
            amplitude=amplitude,
            mechanism=mechanism,
            defaults=defaults,
        )
        if direction_array is not None:
            if len(direction_array) != len(probe.sources):
                raise ValueError("direction must have one row per coordinate")
            for source, source_direction in zip(probe.sources, direction_array):
                source.direction = source_direction.tolist()
        if names is None:
            generated = [
                f"source_{index:03d}"
                for index in range(existing + 1, existing + len(probe.sources) + 1)
            ]
            for source, name in zip(probe.sources, generated):
                source.name = name
        return self._append_inline_sources(probe)

    add_source_points = add_sources

    def add_source_group(
        self,
        kind: str,
        coords: Any,
        direction: Optional[Any] = None,
        domain: Optional[int] = None,
    ) -> List[str]:
        """Deprecated adapter that adds one identity field per point."""

        warnings.warn(
            "add_source_group() is deprecated; use add_sources().",
            DeprecationWarning,
            stacklevel=2,
        )
        if (
            self.source_encoding is not None
            and self.source_encoding.encoding_type != "Named"
        ):
            raise ValueError("Cannot append identity fields to non-Named encoding")
        rows = _source_coordinate_rows(coords)
        first_field = self.source_field_count()
        names = [
            f"source_{index}" for index in range(first_field, first_field + len(rows))
        ]
        names = self.add_sources(
            kind=kind,
            coords=coords,
            names=names,
            direction=direction,
            domain=domain,
        )
        if self.source_encoding is not None:
            self._append_identity_fields(names)
        return names

    def add_compound_source(
        self,
        kind: str,
        coords: np.ndarray,
        weights: np.ndarray,
        direction: Optional[np.ndarray] = None,
        domain: Optional[int] = None,
    ) -> DistributedSource:
        """Deprecated adapter for one weighted RHS field over physical points.

        Args:
            kind: Source kind understood by the solver.
            coords: Coordinate array with one source point per row.
            weights: Scalar weights applied to each point direction.
            direction: Optional direction vector or per-point direction array.
            domain: Optional domain where the source is evaluated.

        Raises:
            ValueError: If ``direction`` does not have one row per coordinate.
        """

        warnings.warn(
            "add_compound_source() is deprecated; use add_sources() plus "
            "add_distributed_source().",
            DeprecationWarning,
            stacklevel=2,
        )
        coords = np.asarray(coords, dtype=np.float64)
        weights = np.asarray(weights, dtype=float)
        if coords.ndim != 2:
            raise ValueError("coords must have shape (n, dim)")
        if weights.ndim != 1 or len(weights) != len(coords):
            raise ValueError("weights must have one value per coordinate row")
        if direction is not None:
            direction = np.asarray(direction, dtype=float)
        if direction is not None and direction.ndim not in {1, 2}:
            raise ValueError("direction must be a 1D vector or one row per coordinate")
        if (
            direction is not None
            and direction.ndim == 2
            and len(direction) != len(coords)
        ):
            raise ValueError("direction must have one row per coordinate")

        if (
            self.source_encoding is not None
            and self.source_encoding.encoding_type != "Named"
        ):
            raise ValueError(
                "add_compound_source cannot extend non-Named source encoding"
            )

        existing_names = self.source_point_names()
        field_index = self.source_field_count()
        field_name = f"source_{field_index}"
        point_names = [
            f"{field_name}_point_{index:03d}" for index in range(1, len(coords) + 1)
        ]
        geometry = SourceGeometry.points(
            kind=kind,
            coords=coords,
            names=point_names,
            domain=domain,
            direction=(
                direction
                if direction is None or np.asarray(direction).ndim == 1
                else None
            ),
        )
        if direction is not None and np.asarray(direction).ndim == 2:
            for source, source_direction in zip(geometry.sources, direction):
                source.direction = source_direction.tolist()
        self._append_inline_sources(geometry)

        field_obj = DistributedSource.named(
            field_name,
            dict(zip(point_names, weights.tolist())),
        )
        if self.source_encoding is None:
            fields = [
                DistributedSource.named(name, {name: 1.0}) for name in existing_names
            ]
            fields.append(field_obj)
            self.source_encoding = SourceEncoding.named(fields)
        else:
            self.source_encoding.fields.append(field_obj)
        return field_obj

    def add_distributed_source(
        self,
        name: str,
        terms: Mapping[Any, Any],
    ) -> DistributedSource:
        """Append one sparse named RHS/source field."""

        encoded = _encoded_terms(terms)
        known = set(self.source_point_names())
        unknown = sorted(set(encoded).difference(known))
        if unknown:
            raise ValueError(f"Unknown physical source names: {unknown}")
        field_obj = DistributedSource.named(name, encoded)
        if self.source_encoding is None:
            self.source_encoding = SourceEncoding.named([field_obj])
        elif self.source_encoding.encoding_type != "Named":
            raise ValueError(
                "add_distributed_source can only extend Named source encoding"
            )
        else:
            if name in set(self.source_encoding.field_names()):
                raise ValueError(f"Distributed source {name!r} already exists")
            self.source_encoding.fields.append(field_obj)
        return field_obj

    def _append_inline_sources(self, geometry: SourceGeometry) -> List[str]:
        if self.source_geometry is None:
            self.source_geometry = geometry
            return geometry.point_names()
        current = self.source_geometry
        if current.geometry_type != "Inline" or geometry.geometry_type != "Inline":
            raise ValueError("Cannot append inline sources to file-backed geometry")
        if current.kind != geometry.kind:
            raise ValueError("All source points in one geometry must share a kind")
        if current.domain != geometry.domain:
            raise ValueError("All source points in one geometry must share a domain")
        if current.defaults != geometry.defaults:
            raise ValueError(
                "All appended source points must share source-geometry defaults"
            )
        known = set(current.point_names())
        new_names = geometry.point_names()
        duplicates = sorted(known.intersection(new_names))
        if duplicates:
            raise ValueError(f"Duplicate source names: {', '.join(duplicates)}")
        current.sources.extend(geometry.sources)
        return new_names

    def _append_identity_fields(self, names: Sequence[str]) -> None:
        if self.source_encoding is None:
            return
        if self.source_encoding.encoding_type != "Named":
            raise ValueError("Cannot append identity fields to non-Named encoding")
        for name in names:
            self.source_encoding.fields.append(
                DistributedSource.named(name, {name: 1.0})
            )

    def _load_legacy_source_groups(self, groups: Any) -> None:
        """Convert pre-v2 logical source groups to v2 geometry and encoding."""

        legacy_groups = [
            group if isinstance(group, SourceGroup) else SourceGroup.from_fs(group)
            for group in groups
        ]
        if not legacy_groups:
            return

        source_kind: Optional[str] = None
        source_domain: Optional[int] = None
        physical_sources: List[PointSource] = []
        fields: List[DistributedSource] = []
        needs_encoding = False
        used_names: set[str] = set()

        def unique_name(proposed: Optional[str], fallback_index: int) -> str:
            stem = str(proposed or f"source_{fallback_index:03d}")
            name = stem
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{stem}_{suffix}"
            used_names.add(name)
            return name

        for field_index, group in enumerate(legacy_groups, start=1):
            source = group.source
            if isinstance(source, RuptureSource):
                raise ValueError(
                    "Legacy RuptureSource inputs cannot be represented by "
                    "fs-acquisition-2; use an Inline, HDF5, or SPS source geometry"
                )

            kind = getattr(source, "kind", None)
            if kind is None:
                raise ValueError("Legacy source groups require source.kind")
            domain = getattr(source, "domain", None)
            if domain is None and isinstance(source, PointSource):
                domain = source.extra.get("domain")
            if source_kind is None:
                source_kind = str(kind)
                source_domain = domain
            elif str(kind) != source_kind or domain != source_domain:
                raise ValueError(
                    "fs-acquisition-2 requires one homogeneous source kind and "
                    "domain per source geometry"
                )

            field_name = unique_name(getattr(source, "name", None), field_index)
            if isinstance(source, PointSource):
                point_extra = copy.deepcopy(source.extra)
                point_extra.pop("domain", None)
                physical_sources.append(
                    PointSource(
                        name=field_name,
                        coordinates=copy.deepcopy(source.coordinates),
                        direction=copy.deepcopy(source.direction),
                        amplitude=copy.deepcopy(source.amplitude),
                        mechanism=copy.deepcopy(source.mechanism),
                        extra=point_extra,
                    )
                )
                fields.append(DistributedSource.named(field_name, {field_name: 1.0}))
                continue

            if not isinstance(source, CompoundSource):
                raise TypeError(
                    f"Unsupported legacy source type {type(source).__name__}"
                )
            needs_encoding = True
            coordinate_rows = _source_coordinate_rows(source.coordinates)
            direction = source.direction
            direction_array = None
            if direction is not None:
                direction_array = np.asarray(direction, dtype=float)
                if direction_array.size == 0:
                    direction_array = None
            if direction_array is None:
                direction_rows: List[Any] = [None] * len(coordinate_rows)
            else:
                if direction_array.ndim == 1:
                    direction_array = np.tile(
                        direction_array, (len(coordinate_rows), 1)
                    )
                if direction_array.ndim != 2 or len(direction_array) != len(
                    coordinate_rows
                ):
                    raise ValueError(
                        "Legacy compound-source direction must have one row per point"
                    )
                direction_rows = direction_array.tolist()

            terms: Dict[str, float] = {}
            for point_index, (coordinates, point_direction) in enumerate(
                zip(coordinate_rows, direction_rows), start=1
            ):
                point_name = unique_name(
                    f"{field_name}_point_{point_index:03d}",
                    len(physical_sources) + 1,
                )
                physical_sources.append(
                    PointSource(
                        name=point_name,
                        coordinates=copy.deepcopy(coordinates),
                        direction=copy.deepcopy(point_direction),
                    )
                )
                terms[point_name] = 1.0
            fields.append(DistributedSource.named(field_name, terms))

        self.source_geometry = SourceGeometry.inline(
            kind=source_kind,
            domain=source_domain,
            sources=physical_sources,
        )
        self.source_encoding = SourceEncoding.named(fields) if needs_encoding else None

    def _compat_source_groups(self) -> NamedList:
        """Build a detached legacy logical-source view from the v2 model."""

        geometry = self.source_geometry
        if geometry is None or geometry.geometry_type != "Inline":
            return _SourceGroupCompatibilityView()

        points = {source.name: source for source in geometry.sources}
        defaults = geometry.defaults

        def point_direction(source: PointSource) -> Any:
            value = source.direction
            if value is None:
                value = defaults.get("direction")
            if isinstance(value, Mapping):
                value = Direction.from_fs(value)
            return copy.deepcopy(value)

        def legacy_point(source: PointSource, *, name: str) -> SourceGroup:
            return SourceGroup(
                source=PointSource(
                    name=name,
                    kind=geometry.kind,
                    coordinates=copy.deepcopy(source.coordinates),
                    direction=point_direction(source),
                    amplitude=copy.deepcopy(
                        source.amplitude
                        if source.amplitude is not None
                        else defaults.get("amplitude")
                    ),
                    mechanism=copy.deepcopy(
                        source.mechanism
                        if source.mechanism is not None
                        else defaults.get("mechanism")
                    ),
                    **(
                        {"domain": geometry.domain}
                        if geometry.domain is not None
                        else {}
                    ),
                )
            )

        encoding = self.source_encoding
        if encoding is None:
            return _SourceGroupCompatibilityView(
                [
                    legacy_point(source, name=name)
                    for source, name in zip(geometry.sources, geometry.point_names())
                ]
            )
        if encoding.encoding_type == "HDF5Dense":
            return _SourceGroupCompatibilityView()

        groups: List[SourceGroup] = []
        for field_index, field_obj in enumerate(encoding.fields, start=1):
            field_name = field_obj.name or f"field_{field_index:03d}"
            if encoding.encoding_type == "Named":
                terms = list(field_obj.terms.items())
            else:
                terms = list(zip(geometry.point_names(), field_obj.coefficients or []))
            nonzero_terms = [
                (name, coefficient)
                for name, coefficient in terms
                if _coefficient_magnitude(coefficient) != 0.0
            ]
            if len(nonzero_terms) == 1 and _coefficient_is_one(nonzero_terms[0][1]):
                point = points.get(str(nonzero_terms[0][0]))
                if point is not None:
                    groups.append(legacy_point(point, name=field_name))
                    continue

            coordinates = []
            directions = []
            for source_name, coefficient in nonzero_terms:
                point = points.get(str(source_name))
                if point is None:
                    continue
                coordinates.append(_coordinate_value(point.coordinates))
                directions.append(
                    _weighted_direction(point_direction(point), coefficient)
                )
            groups.append(
                SourceGroup(
                    source=CompoundSource(
                        name=field_name,
                        kind=geometry.kind,
                        domain=geometry.domain,
                        coordinates=np.asarray(coordinates, dtype=np.float64),
                        direction=np.asarray(directions, dtype=float),
                    )
                )
            )
        return _SourceGroupCompatibilityView(groups)

    @property
    def source_groups(self) -> NamedList:
        """Deprecated computed view of logical source fields."""

        _deprecated_source_groups()
        return self._compat_source_groups()

    @source_groups.setter
    def source_groups(self, _value: Any) -> None:
        _deprecated_source_groups()
        raise TypeError(
            "Acquisition.source_groups is a read-only compatibility view; "
            "pass source_groups to Acquisition(...) for legacy migration or "
            "use set_sources() and set_source_encoding()"
        )

    def known_source_point_count(self) -> Optional[int]:
        """Return the physical point count, or ``None`` for external geometry."""

        if self.source_geometry is None:
            return 0
        return self.source_geometry.point_count

    def source_point_count(self) -> int:
        """Return the number of inline physical source points."""

        return int(self.known_source_point_count() or 0)

    def known_source_field_count(self) -> Optional[int]:
        """Return the logical RHS/source-field count when locally known."""

        if self.source_encoding is not None:
            return self.source_encoding.field_count
        return self.known_source_point_count()

    def source_field_count(self) -> int:
        """Return the number of locally addressable RHS/source fields."""

        return int(self.known_source_field_count() or 0)

    def source_field_ids(self) -> List[int]:
        """Return one-based source-field identifiers."""

        return list(range(1, self.source_field_count() + 1))

    def source_field_names(self) -> List[str]:
        """Return source-field names when locally known."""

        if self.source_encoding is not None:
            return self.source_encoding.field_names()
        return self.source_point_names()

    def source_point_names(self) -> List[str]:
        """Return physical source-point names when locally known."""

        if self.source_geometry is None:
            return []
        return self.source_geometry.point_names()

    def source_point_coords(self) -> np.ndarray:
        """Return inline physical source-point coordinates."""

        if self.source_geometry is None:
            return np.empty((0, 0), dtype=float)
        return self.source_geometry.coordinates()

    def add_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        domain: Optional[int] = None,
        **kwargs,
    ):
        """Add a receiver group with common device and coordinates.

        Args:
            name: Receiver group name.
            device: Device defining receiver type and components.
            coords: Receiver coordinate array or coordinate object.
            domain: Optional domain where the receiver group is evaluated.
            **kwargs: Additional solver-facing receiver group fields.

        Returns:
            Newly added ``ReceiverGroup``.

        Raises:
            TypeError: If deprecated frame arguments are supplied.
        """
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

    def add_survey(self, survey: SparseSurvey) -> SparseSurvey:
        """Add or replace a named sparse survey layout.

        Args:
            survey: Sparse survey instance or serialized survey mapping.

        Returns:
            Stored ``SparseSurvey`` instance.
        """

        if isinstance(survey, dict):
            survey = SparseSurvey.from_fs(survey)
        try:
            self.surveys[survey.name] = survey
        except ValueError:
            self.surveys.append(survey)
        return survey

    def add_sparse_survey(self, name: str, traces=None, **kwargs) -> SparseSurvey:
        """Create and add a named inline sparse survey.

        Args:
            name: Survey name.
            traces: Optional initial trace samples.
            **kwargs: Additional ``SparseSurvey`` constructor arguments.

        Returns:
            Newly added ``SparseSurvey`` instance.
        """

        return self.add_survey(SparseSurvey(name=name, traces=traces, **kwargs))

    def add_sparse_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        survey: Optional[Union[str, SparseSurvey, Dict]] = None,
        domain: Optional[int] = None,
        **kwargs,
    ) -> ReceiverGroup:
        """Add a receiver group that samples traces from a named sparse survey.

        ``survey`` can be a survey name, a ``SparseSurvey`` object, or a survey
        dictionary loaded from JSON. Survey objects are added to
        ``Acquisition.surveys`` automatically.

        Args:
            name: Receiver group name.
            device: Receiver device for the sparse samples.
            coords: Receiver coordinate array or coordinate object.
            survey: Sparse survey name, object, or serialized mapping.
            domain: Optional receiver domain.
            **kwargs: Additional solver-facing receiver group fields.

        Returns:
            Newly added ``ReceiverGroup``.
        """
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
        """List receiver output field selectors.

        Args:
            recv_name: Optional receiver group name. When omitted, all receiver
                groups are included.

        Returns:
            Field selectors of the form ``"<group>:<component>"``.
        """
        field_list = []

        if recv_name:
            group = self.receiver_group(recv_name)
            for field in group.components:
                file = f"{group.name}:{field.name}"
                field_list.append(file)
        else:
            for group in self.receiver_groups:
                for field in group.components:
                    file = f"{group.name}:{field.name}"
                    field_list.append(file)
        return field_list

    def list_sources(self) -> List[int]:
        """Return valid one-based source-field numbers."""

        return self.source_field_ids()

    def source(self, isrc: int) -> SourceGroup:
        """Return a source group by one-based index.

        Args:
            isrc: One-based source index.

        Returns:
            Matching ``SourceGroup``.
        """
        groups = self._compat_source_groups()
        try:
            return groups[isrc - 1]
        except IndexError:
            raise IndexError(f"Source index {isrc} is out of range.") from None

    def receiver_group(self, name: str) -> ReceiverGroup:
        """Return a receiver group by name."""

        return self.receiver_groups[name]

    def _set_path(self, proj_path: Path, rel_path: Path):
        warn_deprecated_path_api(f"{self.__class__.__name__}._set_path")
        self._proj_path = Path(proj_path).expanduser().resolve()
        self._rel_path = Path(rel_path)

    def receiver_coords(self, group: Optional[str] = None):
        """Return receiver coordinates.

        Args:
            group: Optional receiver group name. When omitted, all groups are
                returned as a mapping.
        """
        if group is None:
            group_locations = {}
            for group in self.receiver_groups:
                group_locations[group.name] = group.coordinates.get()
            return group_locations
        else:
            return self.receiver_groups[group].coordinates.get()

    def source_coords(self, src: Optional[int] = None):
        """Return source-field reference coordinates.

        Args:
            src: Optional one-based source index.
        """
        if self.source_geometry is None:
            coords = np.empty((0, 0), dtype=float)
        elif self.source_encoding is None:
            coords = self.source_geometry.coordinates()
        else:
            coords = self.source_encoding.reference_coordinates(self.source_geometry)
        if src is None:
            return coords
        return coords[int(src) - 1]

    def offsets(self, src: int, group: str) -> Dict:
        """Return horizontal source-field/receiver offsets.

        Args:
            src: One-based source index.
            group: Receiver group name.
        """
        diff = self.receiver_coords(group) - self.source_coords(src)
        offsets = np.hypot(diff[:, 0], diff[:, 1])
        return offsets

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

    @property
    def _path(self) -> Path:
        warn_deprecated_path_api(f"{self.__class__.__name__}._path")
        return self._proj_path / self._rel_path


def _source_coordinate_rows(coords):
    extra = {}
    if isinstance(coords, CoordinateValue):
        extra = copy.deepcopy(coords.extra)

    values, units, system = coordinate_array_metadata(coords)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("source coordinates must be a 2D array")

    if units is not None or system is not None:
        return [
            CoordinateValue(
                row.tolist(),
                units=units,
                system=system,
                extra=copy.deepcopy(extra),
            )
            for row in values
        ]

    return values
