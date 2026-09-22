"""Compatibility shims for pre-v2 acquisition source APIs."""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Tuple

import numpy as np

from frequensolve.geometry.frame import CoordinateValue, Direction
from frequensolve.seismic._legacy_source_types import (
    CompoundSource,
    RuptureSource,
    SourceGroup,
)
from frequensolve.seismic.receivers import coordinate_array_metadata
from frequensolve.seismic.sources import (
    EncodedSource,
    PointSource,
    SourceEncoding,
    SourceGeometry,
)
from frequensolve.util.named_list import NamedList


class SourceGroupCompatibilityView(NamedList):
    """Detached, read-only view returned by the deprecated public property."""

    @staticmethod
    def _reject_mutation(*_args: Any, **_kwargs: Any) -> NoReturn:
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


def _warn(name: str, replacement: str, *, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{name} is deprecated; use {replacement}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def add_source_group(
    acquisition: Any,
    *,
    kind: str,
    coords: Any,
    direction: Optional[Any],
    domain: Optional[int],
) -> List[str]:
    """Implement the deprecated identity-source authoring call."""

    _warn("add_source_group()", "add_sources()")

    if (
        acquisition.source_encoding is not None
        and acquisition.source_encoding.encoding_type != "Named"
    ):
        raise ValueError("Cannot append identity fields to non-Named encoding")
    rows = _coordinate_rows(coords)
    first_field = acquisition.source_field_count()
    names = [f"source_{index}" for index in range(first_field, first_field + len(rows))]
    names = acquisition.add_sources(
        kind=kind,
        coords=coords,
        names=names,
        direction=direction,
        domain=domain,
    )
    _append_identity_fields(acquisition, names)
    return names


def add_compound_source(
    acquisition: Any,
    *,
    kind: str,
    coords: Any,
    weights: Any,
    direction: Optional[Any],
    domain: Optional[int],
) -> EncodedSource:
    """Implement the deprecated compound-source authoring call."""

    _warn(
        "add_compound_source()",
        "add_sources() plus add_encoded_source()",
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
    if direction is not None and direction.ndim == 2 and len(direction) != len(coords):
        raise ValueError("direction must have one row per coordinate")

    if (
        acquisition.source_encoding is not None
        and acquisition.source_encoding.encoding_type != "Named"
    ):
        raise ValueError("add_compound_source cannot extend non-Named source encoding")

    existing_names = acquisition.source_point_names()
    field_index = acquisition.source_field_count()
    field_name = f"source_{field_index}"
    point_names = [
        f"{field_name}_point_{index:03d}" for index in range(1, len(coords) + 1)
    ]
    geometry = SourceGeometry.points(
        kind=kind,
        coords=coords,
        names=point_names,
        domain=domain,
        direction=direction if direction is None or direction.ndim == 1 else None,
    )
    if direction is not None and direction.ndim == 2:
        for source, source_direction in zip(geometry.sources, direction):
            source.direction = source_direction.tolist()
    acquisition._append_inline_sources(geometry)

    field = EncodedSource.named(
        field_name,
        dict(zip(point_names, weights.tolist())),
    )
    if acquisition.source_encoding is None:
        fields = [EncodedSource.named(name, {name: 1.0}) for name in existing_names]
        fields.append(field)
        acquisition.source_encoding = SourceEncoding.named(fields)
    else:
        acquisition.source_encoding.fields.append(field)
    return field


def _append_identity_fields(acquisition: Any, names: List[str]) -> None:
    encoding = acquisition.source_encoding
    if encoding is None:
        return
    if encoding.encoding_type != "Named":
        raise ValueError("Cannot append identity fields to non-Named encoding")
    encoding.fields.extend(EncodedSource.named(name, {name: 1.0}) for name in names)


def migrate_source_groups(
    groups: Any,
) -> Tuple[Optional[SourceGeometry], Optional[SourceEncoding]]:
    """Convert pre-v2 logical groups into v2 geometry and encoding."""

    legacy_groups = [
        group if isinstance(group, SourceGroup) else SourceGroup.from_fs(group)
        for group in groups
    ]
    if not legacy_groups:
        return None, None

    source_kind: Optional[str] = None
    source_domain: Optional[int] = None
    physical_sources: List[PointSource] = []
    fields: List[EncodedSource] = []
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
            fields.append(EncodedSource.named(field_name, {field_name: 1.0}))
            continue

        if not isinstance(source, CompoundSource):
            raise TypeError(f"Unsupported legacy source type {type(source).__name__}")
        needs_encoding = True
        coordinate_rows = _coordinate_rows(source.coordinates)
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
                direction_array = np.tile(direction_array, (len(coordinate_rows), 1))
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
        fields.append(EncodedSource.named(field_name, terms))

    assert source_kind is not None
    geometry = SourceGeometry.inline(
        kind=source_kind,
        domain=source_domain,
        sources=physical_sources,
    )
    encoding = SourceEncoding.named(fields) if needs_encoding else None
    return geometry, encoding


def source_groups_view(
    geometry: Optional[SourceGeometry],
    encoding: Optional[SourceEncoding],
) -> NamedList:
    """Build a detached legacy logical-source view from the v2 model."""

    if geometry is None or geometry.geometry_type != "Inline":
        return SourceGroupCompatibilityView()

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
                domain=geometry.domain,
            )
        )

    if encoding is None:
        return SourceGroupCompatibilityView(
            [
                legacy_point(source, name=name)
                for source, name in zip(geometry.sources, geometry.point_names())
            ]
        )
    if encoding.encoding_type == "HDF5Dense":
        return SourceGroupCompatibilityView()

    groups: List[SourceGroup] = []
    for field_index, field_obj in enumerate(encoding.fields, start=1):
        field_name = field_obj.name or f"field_{field_index:06d}"
        if encoding.encoding_type == "Named":
            terms = list(field_obj.terms.items())
        else:
            assert field_obj.coefficients is not None
            terms = list(zip(geometry.point_names(), field_obj.coefficients))
        nonzero_terms = [
            (name, coefficient)
            for name, coefficient in terms
            if _coefficient_magnitude(coefficient) != 0.0
        ]
        if len(nonzero_terms) == 1 and _coefficient_value(
            nonzero_terms[0][1]
        ) == complex(1.0, 0.0):
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
            directions.append(_weighted_direction(point_direction(point), coefficient))
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
    return SourceGroupCompatibilityView(groups)


def source_groups(acquisition: Any) -> NamedList:
    """Return the deprecated logical-source compatibility view."""

    _warn(
        "Acquisition.source_groups",
        "source_geometry, source_encoding, source_point_names(), and "
        "source_field_names()",
    )
    return source_groups_view(
        acquisition.source_geometry,
        acquisition.source_encoding,
    )


def reject_source_groups_assignment() -> None:
    """Reject mutation through the deprecated compatibility property."""

    _warn(
        "Acquisition.source_groups",
        "set_sources() and set_source_encoding()",
    )
    raise TypeError(
        "Acquisition.source_groups is a read-only compatibility view; "
        "pass source_groups to Acquisition(...) for legacy migration or "
        "use set_sources() and set_source_encoding()"
    )


def source_group(acquisition: Any, isrc: int) -> SourceGroup:
    """Return one legacy logical-source view by one-based index."""

    groups = source_groups_view(
        acquisition.source_geometry,
        acquisition.source_encoding,
    )
    if isrc < 1 or isrc > len(groups):
        raise IndexError(f"Source index {isrc} is out of range.") from None
    return groups[isrc - 1]


def _coefficient_value(value: Any) -> complex:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    return complex(value)


def _coefficient_magnitude(value: Any) -> float:
    return abs(_coefficient_value(value))


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


def _coordinate_rows(coords: Any) -> Any:
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
