"""Source geometry and source-encoding authoring helpers."""

from __future__ import annotations

import copy
import hashlib
import warnings
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import h5py
import numpy as np
import xarray as xr

from frequensolve.geometry.frame import (
    CoordinateValue,
    Direction,
    coordinate_value_to_fs,
    direction_to_fs,
)
from frequensolve.units import (
    is_quantity,
    unit_expression,
    ureg,
    value_and_units_to_fs,
)
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
)

__all__ = [
    "Source",
    "SourceGroup",
    "RuptureSource",
    "CompoundSource",
    "PointSource",
    "SourceGeometry",
    "SourceEncoding",
    "EncodedSource",
    "DistributedSource",
]


_SOURCE_KINDS = {"scalar", "vector", "tensor", "monopole", "dipole"}


def _path_to_fs(path: Union[str, Path], ctx: Optional[ExportContext]) -> str:
    if ctx is None:
        return str(path)
    return str(ctx.relative_to_project(Path(path)))


def _mechanism_to_fs(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return {"type": value}
    return copy.deepcopy(value)


def _source_direction_to_fs(value: Any) -> Any:
    """Serialize a direction against the acquisition source-basis schema."""

    if isinstance(value, Mapping) and "type" in value:
        value = Direction.from_fs(value)

    if isinstance(value, Direction):
        if value.system is not None:
            raise ValueError(
                "Source Direction values cannot include a coordinate system; "
                "express the direction in physical coordinates before export"
            )
        if value.extra:
            fields = ", ".join(sorted(value.extra))
            raise ValueError(
                "Source Direction values cannot include extension fields "
                f"({fields}); use a numeric vector or axis direction"
            )
        if value.type == "vector":
            if value.value is None:
                raise ValueError("Source Direction.vector requires a vector value")
            if value.axis is not None or value.components is not None:
                raise ValueError(
                    "Source Direction.vector cannot include axis or components"
                )
            return _source_direction_to_fs(
                value_and_units_to_fs(value.value, value.units)
            )
        if value.type == "coordinate_axis":
            if not value.axis:
                raise ValueError("Source Direction.axis_direction requires an axis")
            if (
                value.value is not None
                or value.units is not None
                or value.components is not None
            ):
                raise ValueError(
                    "Source axis directions cannot include value, units, or components"
                )
            return {"direction": value.axis}
        raise ValueError(
            f"Source Direction type {value.type!r} is not supported by "
            "fs-acquisition-2; use Direction.vector(...) or "
            "Direction.axis_direction(...)"
        )

    direction_payload = direction_to_fs(value)
    if isinstance(direction_payload, Mapping):
        payload = copy.deepcopy(dict(direction_payload))
        unknown = set(payload).difference({"direction", "value", "units"})
        if unknown:
            fields = ", ".join(sorted(unknown))
            raise ValueError(
                "Source direction mappings contain unsupported fields "
                f"({fields}); use {{'direction': <axis>}} or "
                "{'value': <vector>, 'units': <optional units>}"
            )
        has_axis = "direction" in payload
        has_value = "value" in payload
        if has_axis == has_value:
            raise ValueError(
                "Source direction mappings require exactly one of "
                "'direction' or 'value'"
            )
        if has_axis:
            if not isinstance(payload["direction"], str) or not payload["direction"]:
                raise ValueError("Source direction axis must be a non-empty string")
            if "units" in payload:
                payload["units"] = unit_expression(payload["units"])
            return payload

        normalized = value_and_units_to_fs(
            payload["value"],
            payload.get("units"),
        )
        if isinstance(normalized, Mapping):
            return copy.deepcopy(dict(normalized))
        payload = {"value": normalized}
        if "units" in direction_payload:
            payload["units"] = unit_expression(direction_payload["units"])
        return payload

    normalized = value_and_units_to_fs(direction_payload)
    if isinstance(normalized, Mapping):
        return copy.deepcopy(dict(normalized))
    try:
        numeric = np.asarray(normalized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Source direction must be a numeric vector, Direction.vector(...), "
            "or Direction.axis_direction(...)"
        ) from exc
    if numeric.ndim != 1 or numeric.size == 0:
        raise ValueError("Source direction must be a non-empty numeric vector")
    return numeric.tolist()


def _source_direction_from_fs(value: Any) -> Any:
    """Load a source-schema direction without applying the generic shape."""

    if not isinstance(value, Mapping) or "type" in value:
        return Direction.from_fs(value)
    payload = copy.deepcopy(dict(value))
    allowed = {"direction", "value", "units"}
    has_axis = "direction" in payload
    has_value = "value" in payload
    if set(payload).issubset(allowed) and has_axis != has_value:
        return payload
    return Direction.from_fs(payload)


def _source_basis_to_fs(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Serialize the source-basis fields defined by fs-acquisition-2."""

    payload = copy.deepcopy(dict(value))
    if "direction" in payload:
        payload["direction"] = _source_direction_to_fs(payload["direction"])
    if "amplitude" in payload:
        payload["amplitude"] = value_and_units_to_fs(payload["amplitude"])
    if "mechanism" in payload:
        payload["mechanism"] = _mechanism_to_fs(payload["mechanism"])
    return payload


def _basis_to_fs(
    *,
    kind: Optional[str] = None,
    direction: Any = None,
    amplitude: Any = None,
    mechanism: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if kind is not None:
        payload["kind"] = kind
    if direction is not None:
        payload["direction"] = direction
    if amplitude is not None:
        payload["amplitude"] = amplitude
    if mechanism is not None:
        payload["mechanism"] = mechanism
    if extra:
        payload.update(copy.deepcopy(dict(extra)))
    return _source_basis_to_fs(payload)


def _source_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value not in _SOURCE_KINDS:
        choices = ", ".join(sorted(_SOURCE_KINDS))
        raise ValueError(f"Unsupported source kind {kind!r}. Use one of: {choices}.")
    return value


def _coordinate_rows(
    coords: Any,
    *,
    units: Optional[Any] = None,
    system: Optional[str] = None,
) -> List[Any]:
    extra: Dict[str, Any] = {}
    if isinstance(coords, CoordinateValue):
        if units is None:
            units = coords.units
        if system is None:
            system = coords.system
        extra = copy.deepcopy(coords.extra)
        coords = coords.value

    if is_quantity(coords):
        target_units = units or coords.units
        raw_values = coords.to(target_units).magnitude
        units = target_units
    else:
        raw_values = coords
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("source coordinates must have shape (n, dim)") from exc

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("source coordinates must have shape (n, dim)")

    rows: List[Any] = []
    for row in values:
        value = row.tolist()
        if units is not None or system is not None or extra:
            rows.append(
                CoordinateValue(
                    value,
                    units=units,
                    system=system,
                    extra=copy.deepcopy(extra),
                )
            )
        else:
            rows.append(value)
    return rows


def _reference_rows(values: Any, n_fields: int) -> List[Any]:
    if values is None:
        return [None] * n_fields
    rows = _coordinate_rows(values)
    if len(rows) == 1 and n_fields == 1:
        return rows
    if len(rows) != n_fields:
        raise ValueError("reference_coordinates must have one row per encoded field")
    return rows


def _as_source_point(value: Union["PointSource", Mapping[str, Any]]) -> "PointSource":
    if isinstance(value, PointSource):
        return value
    if isinstance(value, Mapping):
        return PointSource.from_fs(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to PointSource")


def _source_names(names: Optional[Iterable[str]], n_sources: int) -> List[str]:
    if names is None:
        return [f"source_{index:06d}" for index in range(1, n_sources + 1)]
    values = [str(name) for name in names]
    if len(values) != n_sources:
        raise ValueError(f"names must have exactly {n_sources} entries")
    if len(set(values)) != len(values):
        raise ValueError("source names must be unique")
    if any(not name for name in values):
        raise ValueError("source names must be non-empty")
    return values


def _complex_to_fs(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        real = float(value.real)
        imag = float(value.imag)
    elif isinstance(value, Number) and not isinstance(value, (bool, np.bool_)):
        real = float(value)
        imag = 0.0
    else:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(item, Number) and not isinstance(item, (bool, np.bool_))
                for item in value
            )
        ):
            raise TypeError(f"Invalid source-encoding coefficient {value!r}")
        real = float(value[0])
        imag = float(value[1])
    if not np.isfinite(real) or not np.isfinite(imag):
        raise ValueError("source-encoding coefficients must be finite")
    return real if imag == 0.0 else [real, imag]


def _source_coefficient_array(values: Any, *, ndim: int, label: str) -> np.ndarray:
    """Normalize dense source coefficients with vectorized validation."""

    if isinstance(values, xr.DataArray):
        values = values.data
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a numeric array")
    try:
        authored = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a numeric array") from exc
    if authored.dtype.kind not in {"i", "u", "f", "c"}:
        raise TypeError(f"{label} must be a numeric array")

    if authored.ndim == ndim + 1 and authored.shape[-1] == 2:
        result = np.asarray(authored[..., 0], dtype=np.complex64)
        result.imag = np.asarray(authored[..., 1], dtype=np.float32)
    elif authored.ndim == ndim:
        result = np.asarray(authored, dtype=np.complex64)
    else:
        shape = "vector" if ndim == 1 else "matrix"
        raise ValueError(f"{label} must be a {shape}")

    if result.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _complex_value(value: Any) -> complex:
    """Return one source-encoding coefficient as a complex scalar."""

    serialized = _complex_to_fs(value)
    if isinstance(serialized, list):
        return complex(serialized[0], serialized[1])
    return complex(serialized)


def _coefficient_abs(value: Any) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return abs(value)
    if isinstance(value, Number):
        return abs(float(value))
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return abs(complex(value[0], value[1]))
    return abs(complex(value))


def _coordinate_array(value: Any) -> np.ndarray:
    """Return one numeric coordinate vector without its metadata."""

    if isinstance(value, CoordinateValue):
        value = value.value
    if is_quantity(value):
        value = value.magnitude
    result = np.asarray(value, dtype=float)
    if result.ndim == 2 and len(result) == 1:
        result = result[0]
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Source reference coordinates must be a single vector")
    return result


def _coordinate_array_with_metadata(
    value: Any,
) -> tuple[np.ndarray, Optional[Any], Optional[str]]:
    """Return one coordinate vector in its declared units and system."""

    units = None
    system = None
    if isinstance(value, CoordinateValue):
        units = value.units
        system = value.system
        value = value.value
    if is_quantity(value):
        target_units = units or value.units
        value = value.to(target_units).magnitude
        units = target_units
    coordinates = np.asarray(value, dtype=float)
    if coordinates.ndim == 2 and len(coordinates) == 1:
        coordinates = coordinates[0]
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("Source reference coordinates must be a single vector")
    return coordinates, units, system


def _source_coordinate_matrix(
    values: Sequence[Any],
) -> tuple[np.ndarray, Optional[Any], Optional[str]]:
    """Normalize source points to one compatible unit and coordinate system."""

    if not values:
        return np.empty((0, 0), dtype=float), None, None
    first, target_units, target_system = _coordinate_array_with_metadata(values[0])
    rows = [first]
    target_system_key = target_system or "global"
    for value in values[1:]:
        coordinates, units, system = _coordinate_array_with_metadata(value)
        if (system or "global") != target_system_key:
            raise ValueError(
                "Source points must use one coordinate system before an "
                "encoded-field reference can be computed"
            )
        if (units is None) != (target_units is None):
            raise ValueError(
                "Source points must all declare compatible coordinate units "
                "before an encoded-field reference can be computed"
            )
        if target_units is not None:
            try:
                coordinates = (
                    (coordinates * ureg(unit_expression(units)))
                    .to(target_units)
                    .magnitude
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Source points must use compatible coordinate units before "
                    "an encoded-field reference can be computed"
                ) from exc
        if coordinates.shape != first.shape:
            raise ValueError("Source points must use one coordinate dimension")
        rows.append(np.asarray(coordinates, dtype=float))
    return np.asarray(rows, dtype=float), target_units, target_system


@dataclass(init=False)
class PointSource(ExtraFieldsMixin):
    """One physical source point in a source geometry catalog."""

    coordinates: Any
    name: Optional[str] = None
    kind: Optional[str] = None
    direction: Optional[Any] = None
    domain: Optional[int] = None
    amplitude: Optional[Any] = None
    mechanism: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name_or_coordinates: Any = None,
        coordinates: Any = None,
        *,
        name: Optional[str] = None,
        coords: Any = None,
        kind: Optional[str] = None,
        direction: Optional[Any] = None,
        domain: Optional[int] = None,
        amplitude: Optional[Any] = None,
        mechanism: Optional[Any] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        extra_fields = copy.deepcopy(dict(extra or {}))
        if domain is None and "domain" in extra_fields:
            domain = extra_fields.pop("domain")
        if coords is not None:
            if coordinates is not None:
                raise TypeError("Use either coords or coordinates, not both")
            coordinates = coords
        legacy_kind_positional = (
            coordinates is not None
            and kind is None
            and isinstance(name_or_coordinates, str)
            and name_or_coordinates.strip().lower() in _SOURCE_KINDS
        )
        if legacy_kind_positional:
            kind = name_or_coordinates
            name_or_coordinates = None
            if name is None:
                name = "point"

        if coordinates is None:
            if name_or_coordinates is None:
                raise TypeError("PointSource requires coordinates")
            coordinates = name_or_coordinates
        elif name_or_coordinates is not None:
            if name is not None:
                raise TypeError("PointSource name was supplied twice")
            name = str(name_or_coordinates)

        self.coordinates = coordinates
        self.name = name
        self.kind = _source_kind(kind) if kind is not None else None
        self.direction = direction
        self.domain = None if domain is None else int(domain)
        self.amplitude = amplitude
        self.mechanism = mechanism
        self._init_extra(extra_fields, **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "PointSource":
        payload = copy.deepcopy(dict(data))
        coordinates = CoordinateValue.from_fs(payload.pop("coordinates"))
        direction = (
            _source_direction_from_fs(payload.pop("direction"))
            if "direction" in payload
            else None
        )
        payload.pop("frame", None)
        return cls(
            coordinates=coordinates,
            name=payload.pop("name", None),
            kind=payload.pop("kind", None),
            direction=direction,
            domain=payload.pop("domain", None),
            amplitude=payload.pop("amplitude", None),
            mechanism=payload.pop("mechanism", None),
            extra=payload,
        )

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        *,
        include_domain: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            **({"name": self.name} if self.name is not None else {}),
            "coordinates": coordinate_value_to_fs(self.coordinates),
            **_basis_to_fs(
                kind=_source_kind(self.kind) if self.kind is not None else None,
                direction=self.direction,
                amplitude=self.amplitude,
                mechanism=self.mechanism,
            ),
            **(
                {"domain": self.domain}
                if include_domain and self.domain is not None
                else {}
            ),
        }
        return merge_extra(payload, self.extra, "PointSource")


@dataclass
class _InlineSourceGeometry:
    sources: List[PointSource]


@dataclass
class _BulkSourceGeometry:
    coordinates: np.ndarray
    names: Optional[List[str]] = None
    directions: Optional[np.ndarray] = None
    units: Optional[Any] = None
    system: Optional[str] = None
    coordinate_extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _HDF5SourceGeometry:
    file: Union[str, Path]
    dataset: str
    names_dataset: Optional[str] = None
    system: Optional[str] = None
    units: Optional[Any] = None
    count: Optional[int] = None


@dataclass
class _SPSSourceGeometry:
    source_file: Union[str, Path]
    system: Optional[str] = None
    units: Optional[Any] = None
    count: Optional[int] = None


SourceGeometryStorage = Union[
    _InlineSourceGeometry,
    _BulkSourceGeometry,
    _HDF5SourceGeometry,
    _SPSSourceGeometry,
]


@dataclass(init=False)
class SourceGeometry(ExtraFieldsMixin):
    """Physical source catalog with one concrete storage representation."""

    kind: str
    name: Optional[str]
    domain: Optional[int]
    defaults: Dict[str, Any]
    extra: Dict[str, Any]
    _storage: SourceGeometryStorage

    def __init__(
        self,
        *,
        kind: str,
        geometry_type: str = "Inline",
        name: Optional[str] = None,
        domain: Optional[int] = None,
        sources: Optional[Iterable[Union[PointSource, Mapping[str, Any]]]] = None,
        file: Optional[Union[str, Path]] = None,
        dataset: Optional[str] = None,
        names_dataset: Optional[str] = None,
        source_file: Optional[Union[str, Path]] = None,
        system: Optional[str] = None,
        units: Optional[Any] = None,
        count: Optional[int] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.kind = _source_kind(kind)
        self.name = name
        self.domain = None if domain is None else int(domain)
        self.defaults = copy.deepcopy(dict(defaults or {}))
        self._init_extra(extra, **kwargs)
        normalized_type = str(geometry_type).strip().lower()
        source_points = [_as_source_point(source) for source in (sources or [])]
        normalized_count = None if count is None else int(count)
        if normalized_count is not None and normalized_count < 1:
            raise ValueError("source geometry count must be >= 1")

        if normalized_type == "inline":
            if not source_points:
                raise ValueError("Inline source geometry requires at least one source")
            if any(
                value is not None
                for value in (
                    file,
                    dataset,
                    names_dataset,
                    source_file,
                    normalized_count,
                    system,
                    units,
                )
            ):
                raise ValueError(
                    "Inline source geometry cannot include external storage"
                )
            names = [source.name for source in source_points if source.name is not None]
            if len(names) != len(set(names)):
                raise ValueError("source names must be unique")
            self._storage = _InlineSourceGeometry(source_points)
        elif normalized_type == "hdf5":
            if file is None or not dataset:
                raise ValueError("HDF5 source geometry requires file and dataset")
            if names_dataset is not None and not str(names_dataset).strip():
                raise ValueError("HDF5 source names_dataset must not be empty")
            if source_points or source_file is not None:
                raise ValueError(
                    "HDF5 source geometry cannot include inline or SPS storage"
                )
            self._storage = _HDF5SourceGeometry(
                file=file,
                dataset=str(dataset),
                names_dataset=names_dataset,
                system=system,
                units=units,
                count=normalized_count,
            )
        elif normalized_type == "spsfiles":
            if not source_file:
                raise ValueError("SPS source geometry requires source_file")
            if (
                source_points
                or file is not None
                or dataset is not None
                or names_dataset is not None
            ):
                raise ValueError(
                    "SPS source geometry cannot include inline or HDF5 storage"
                )
            self._storage = _SPSSourceGeometry(
                source_file=source_file,
                system=system,
                units=units,
                count=normalized_count,
            )
        else:
            raise ValueError("source geometry type must be Inline, HDF5, or SPSFiles")

    @classmethod
    def points(
        cls,
        *,
        kind: str,
        coords: Any,
        names: Optional[Iterable[str]] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        domain: Optional[int] = None,
        name: Optional[str] = None,
        direction: Optional[Any] = None,
        amplitude: Optional[Any] = None,
        mechanism: Optional[Any] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "SourceGeometry":
        """Create inline point-source geometry from coordinate rows."""

        coordinate_extra: Dict[str, Any] = {}
        if isinstance(coords, CoordinateValue):
            units = coords.units if units is None else units
            system = coords.system if system is None else system
            coordinate_extra = copy.deepcopy(coords.extra)
            coords = coords.value
        if is_quantity(coords):
            target_units = units or coords.units
            raw_coordinates = coords.to(target_units).magnitude
            units = target_units
        else:
            raw_coordinates = coords
        try:
            matrix = np.asarray(raw_coordinates, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("source coordinates must have shape (n, dim)") from exc
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("source coordinates must have shape (n, dim)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("source coordinates must be finite")
        source_names = None if names is None else _source_names(names, len(matrix))
        direction_rows = None
        if direction is not None and not isinstance(direction, Mapping):
            try:
                direction_array = np.asarray(direction, dtype=float)
            except (TypeError, ValueError):
                direction_array = None
            if direction_array is not None and direction_array.ndim == 2:
                if len(direction_array) != len(matrix):
                    raise ValueError("direction must have one row per coordinate")
                if not np.all(np.isfinite(direction_array)):
                    raise ValueError("direction must be finite")
                direction_rows = np.asarray(direction_array, dtype=np.float64)
        default_payload = _basis_to_fs(
            direction=None if direction_rows is not None else direction,
            amplitude=amplitude,
            mechanism=mechanism,
            extra=defaults,
        )
        geometry = cls.__new__(cls)
        geometry.kind = _source_kind(kind)
        geometry.name = name
        geometry.domain = None if domain is None else int(domain)
        geometry.defaults = default_payload
        geometry._init_extra(kwargs.pop("extra", None), **kwargs)
        geometry._storage = _BulkSourceGeometry(
            coordinates=np.ascontiguousarray(matrix),
            names=source_names,
            directions=direction_rows,
            units=units,
            system=system,
            coordinate_extra=coordinate_extra,
        )
        return geometry

    @classmethod
    def inline(
        cls,
        *,
        kind: str,
        sources: Iterable[Union[PointSource, Mapping[str, Any]]],
        name: Optional[str] = None,
        domain: Optional[int] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "SourceGeometry":
        """Create inline source geometry from explicit source points."""

        return cls(
            geometry_type="Inline",
            name=name,
            kind=kind,
            domain=domain,
            sources=sources,
            defaults=defaults,
            **kwargs,
        )

    @classmethod
    def hdf5(
        cls,
        file: Union[str, Path],
        *,
        dataset: str,
        kind: str,
        name: Optional[str] = None,
        domain: Optional[int] = None,
        system: Optional[str] = None,
        units: Optional[Any] = None,
        count: Optional[int] = None,
        names_dataset: Optional[str] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "SourceGeometry":
        """Create file-backed source geometry from an HDF5 dataset."""

        return cls(
            geometry_type="HDF5",
            name=name,
            kind=kind,
            domain=domain,
            file=file,
            dataset=dataset,
            names_dataset=names_dataset,
            system=system,
            units=units,
            count=count,
            defaults=defaults,
            **kwargs,
        )

    @classmethod
    def sps(
        cls,
        source_file: Union[str, Path],
        *,
        kind: str,
        name: Optional[str] = None,
        domain: Optional[int] = None,
        system: Optional[str] = None,
        units: Optional[Any] = None,
        count: Optional[int] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "SourceGeometry":
        """Create source geometry from an SPS source file."""

        return cls(
            geometry_type="SPSFiles",
            name=name,
            kind=kind,
            domain=domain,
            source_file=source_file,
            system=system,
            units=units,
            count=count,
            defaults=defaults,
            **kwargs,
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SourceGeometry":
        payload = copy.deepcopy(dict(data))
        geometry_type = payload.pop("_type", payload.pop("geometry_type", "Inline"))
        sources = payload.pop("sources", None)
        return cls(
            geometry_type=geometry_type,
            kind=payload.pop("kind"),
            name=payload.pop("name", None),
            domain=payload.pop("domain", None),
            sources=sources,
            file=payload.pop("file", None),
            dataset=payload.pop("dataset", None),
            names_dataset=payload.pop("names_dataset", None),
            source_file=payload.pop("source_file", None),
            system=payload.pop("system", None),
            units=payload.pop("units", None),
            count=payload.pop("count", payload.pop("source_count", None)),
            defaults=payload.pop("defaults", None),
            extra=payload,
        )

    @property
    def geometry_type(self) -> str:
        if isinstance(self._storage, (_InlineSourceGeometry, _BulkSourceGeometry)):
            return "Inline"
        if isinstance(self._storage, _HDF5SourceGeometry):
            return "HDF5"
        return "SPSFiles"

    @property
    def sources(self) -> List[PointSource]:
        """Return the legacy mutable point list, materializing bulk storage lazily."""

        if isinstance(self._storage, _InlineSourceGeometry):
            return self._storage.sources
        if not isinstance(self._storage, _BulkSourceGeometry):
            return []
        storage = self._storage
        names = storage.names
        sources = []
        for index, row in enumerate(storage.coordinates):
            coordinates: Any = row.tolist()
            if (
                storage.units is not None
                or storage.system is not None
                or storage.coordinate_extra
            ):
                coordinates = CoordinateValue(
                    coordinates,
                    units=storage.units,
                    system=storage.system,
                    extra=copy.deepcopy(storage.coordinate_extra),
                )
            sources.append(
                PointSource(
                    name=(names[index] if names is not None else None),
                    coordinates=coordinates,
                    direction=(
                        storage.directions[index].tolist()
                        if storage.directions is not None
                        else None
                    ),
                )
            )
        self._storage = _InlineSourceGeometry(sources)
        return sources

    @property
    def file(self) -> Optional[Union[str, Path]]:
        return (
            self._storage.file
            if isinstance(self._storage, _HDF5SourceGeometry)
            else None
        )

    @property
    def dataset(self) -> Optional[str]:
        return (
            self._storage.dataset
            if isinstance(self._storage, _HDF5SourceGeometry)
            else None
        )

    @property
    def names_dataset(self) -> Optional[str]:
        return (
            self._storage.names_dataset
            if isinstance(self._storage, _HDF5SourceGeometry)
            else None
        )

    @property
    def source_file(self) -> Optional[Union[str, Path]]:
        return (
            self._storage.source_file
            if isinstance(self._storage, _SPSSourceGeometry)
            else None
        )

    @property
    def system(self) -> Optional[str]:
        return getattr(self._storage, "system", None)

    @property
    def units(self) -> Optional[Any]:
        return getattr(self._storage, "units", None)

    @property
    def count(self) -> Optional[int]:
        return getattr(self._storage, "count", None)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "_type": self.geometry_type,
            **({"name": self.name} if self.name is not None else {}),
            **({"domain": self.domain} if self.domain is not None else {}),
            "kind": self.kind,
        }
        if self.defaults:
            payload["defaults"] = _source_basis_to_fs(self.defaults)
        if self.geometry_type == "Inline":
            materialized = self._materialize_inline(ctx, payload)
            if materialized is not None:
                return merge_extra(materialized, self.extra, "SourceGeometry")
            payload["sources"] = self._inline_rows(ctx)
        elif self.geometry_type == "HDF5":
            assert self.file is not None
            payload["file"] = _path_to_fs(self.file, ctx)
            payload["dataset"] = self.dataset
            if self.names_dataset is not None:
                payload["names_dataset"] = self.names_dataset
            if self.system is not None:
                payload["system"] = self.system
            if self.units is not None:
                payload["units"] = unit_expression(self.units)
            if self.count is not None:
                payload["count"] = self.count
        else:
            payload["source_file"] = _path_to_fs(self.source_file, ctx)
            if self.system is not None:
                payload["system"] = self.system
            if self.units is not None:
                payload["units"] = unit_expression(self.units)
            if self.count is not None:
                payload["count"] = self.count
        return merge_extra(payload, self.extra, "SourceGeometry")

    def _materialize_inline(
        self,
        ctx: Optional[ExportContext],
        base_payload: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Materialize a large homogeneous source catalog in the input store."""

        store = getattr(ctx, "store", None) if ctx is not None else None
        if (self.point_count or 0) <= 200 or store is None:
            return None
        if isinstance(self._storage, _BulkSourceGeometry):
            storage = self._storage
            if storage.directions is not None or storage.coordinate_extra:
                return None
            matrix = storage.coordinates
            units = storage.units
            system = storage.system
            names = storage.names
        else:
            for source in self.sources:
                coordinates = source.coordinates
                if (
                    source.direction is not None
                    or source.amplitude is not None
                    or source.mechanism is not None
                    or source.domain is not None
                    or source.extra
                    or (
                        source.kind is not None
                        and _source_kind(source.kind) != self.kind
                    )
                    or (isinstance(coordinates, CoordinateValue) and coordinates.extra)
                ):
                    return None
            matrix, units, system = _source_coordinate_matrix(self.coordinate_values())
            explicit_names = [source.name for source in self.sources]
            names = (
                self.point_names()
                if any(name is not None for name in explicit_names)
                else None
            )
        coordinate = (
            ["x", "z"]
            if matrix.shape[1] == 2
            else (
                ["x", "y", "z"]
                if matrix.shape[1] == 3
                else list(range(matrix.shape[1]))
            )
        )
        attrs = {"fs_kind": "source_geometry_coordinates"}
        if units is not None:
            attrs["units"] = unit_expression(units)
        if system is not None:
            attrs["system"] = system
        ref = store.put_dataarray(
            "inputs/acquisition/source_geometry/coordinates",
            xr.DataArray(
                matrix,
                dims=("source", "coordinate"),
                coords={"coordinate": coordinate},
            ),
            attrs=attrs,
            coordinate_dims=("coordinate",),
            dtype=np.float64,
        )
        payload = {
            **dict(base_payload),
            "_type": "HDF5",
            "count": len(matrix),
            **ref.to_fs(),
            **({"units": unit_expression(units)} if units is not None else {}),
            **({"system": system} if system is not None else {}),
        }
        default_names = [f"source_{index:06d}" for index in range(1, len(matrix) + 1)]
        if names is not None and list(names) != default_names:
            names_ref = store.put_string_array(
                "inputs/acquisition/source_geometry/names",
                names,
                dimension="source",
                attrs={"fs_kind": "source_geometry_names"},
            )
            payload["names_dataset"] = names_ref.clean_dataset
        return payload

    @property
    def point_count(self) -> Optional[int]:
        if isinstance(self._storage, _InlineSourceGeometry):
            return len(self._storage.sources)
        if isinstance(self._storage, _BulkSourceGeometry):
            return len(self._storage.coordinates)
        return self.count

    @property
    def is_bulk(self) -> bool:
        """Return whether inline points are still held as one NumPy matrix."""

        return isinstance(self._storage, _BulkSourceGeometry)

    def point_names(self) -> List[str]:
        if isinstance(self._storage, _BulkSourceGeometry):
            if self._storage.names is not None:
                return list(self._storage.names)
            return [
                f"source_{index:06d}"
                for index in range(1, len(self._storage.coordinates) + 1)
            ]
        if isinstance(self._storage, _InlineSourceGeometry):
            return [
                source.name if source.name is not None else f"source_{index:06d}"
                for index, source in enumerate(self._storage.sources, start=1)
            ]
        return []

    def has_explicit_names(self) -> bool:
        """Return whether every inline source has an authored name."""

        if isinstance(self._storage, _BulkSourceGeometry):
            return True
        if isinstance(self._storage, _InlineSourceGeometry):
            return all(source.name is not None for source in self._storage.sources)
        return False

    def validate_unique_names(self) -> None:
        """Validate authored inline names without generating default-name lists."""

        if isinstance(self._storage, _BulkSourceGeometry):
            names = self._storage.names
        elif isinstance(self._storage, _InlineSourceGeometry):
            names = [
                source.name if source.name is not None else f"source_{index:06d}"
                for index, source in enumerate(self._storage.sources, start=1)
            ]
        else:
            return
        if names is not None and len(names) != len(set(names)):
            raise ValueError("Inline source names must be unique")

    def set_point_names(self, names: Iterable[str]) -> None:
        """Set stable names without materializing bulk source points."""

        values = _source_names(names, int(self.point_count or 0))
        if isinstance(self._storage, _BulkSourceGeometry):
            self._storage.names = values
            return
        if isinstance(self._storage, _InlineSourceGeometry):
            for source, name in zip(self._storage.sources, values):
                source.name = name
            return
        raise ValueError("Cannot set names on external source geometry")

    def point(self, index: int) -> PointSource:
        """Return one inline point without expanding the rest of a bulk catalog."""

        count = int(self.point_count or 0)
        if index < 0 or index >= count:
            raise IndexError("source point index is out of range")
        if isinstance(self._storage, _InlineSourceGeometry):
            return self._storage.sources[index]
        if not isinstance(self._storage, _BulkSourceGeometry):
            raise ValueError("Source-point metadata is stored externally")
        storage = self._storage
        coordinates: Any = storage.coordinates[index].tolist()
        if (
            storage.units is not None
            or storage.system is not None
            or storage.coordinate_extra
        ):
            coordinates = CoordinateValue(
                coordinates,
                units=storage.units,
                system=storage.system,
                extra=copy.deepcopy(storage.coordinate_extra),
            )
        return PointSource(
            name=(storage.names[index] if storage.names is not None else None),
            coordinates=coordinates,
            direction=(
                storage.directions[index].tolist()
                if storage.directions is not None
                else None
            ),
        )

    def extend_inline(self, other: "SourceGeometry") -> None:
        """Append compatible inline geometry while preserving bulk arrays."""

        if self.geometry_type != "Inline" or other.geometry_type != "Inline":
            raise ValueError("Cannot append inline sources to file-backed geometry")
        if isinstance(self._storage, _BulkSourceGeometry) and isinstance(
            other._storage, _BulkSourceGeometry
        ):
            left, right = self._storage, other._storage
            if (
                (unit_expression(left.units) if left.units is not None else None)
                == (unit_expression(right.units) if right.units is not None else None)
                and left.system == right.system
                and left.coordinate_extra == right.coordinate_extra
                and ((left.directions is None) == (right.directions is None))
            ):
                if left.coordinates.shape[1] != right.coordinates.shape[1]:
                    raise ValueError("Source points must use one coordinate dimension")
                left.coordinates = np.concatenate((left.coordinates, right.coordinates))
                if left.directions is not None:
                    assert right.directions is not None
                    if left.directions.shape[1] != right.directions.shape[1]:
                        raise ValueError("Source directions must use one dimension")
                    left.directions = np.concatenate(
                        (left.directions, right.directions)
                    )
                if left.names is not None or right.names is not None:
                    left_names = left.names or [
                        f"source_{index:06d}"
                        for index in range(
                            1, len(left.coordinates) - len(right.coordinates) + 1
                        )
                    ]
                    right_names = right.names or [
                        f"source_{index:06d}"
                        for index in range(1, len(right.coordinates) + 1)
                    ]
                    left.names = [*left_names, *right_names]
                return
        self.sources.extend(other.sources)

    def _inline_rows(self, ctx: Optional[ExportContext]) -> List[Dict[str, Any]]:
        """Serialize inline storage, expanding rows only for small JSON payloads."""

        if isinstance(self._storage, _InlineSourceGeometry):
            return [
                source.to_fs(ctx, include_domain=False)
                for source in self._storage.sources
            ]
        storage = self._storage
        if not isinstance(storage, _BulkSourceGeometry):
            return []
        rows = []
        for index, coordinates in enumerate(storage.coordinates):
            coordinate_value: Any = coordinates.tolist()
            if (
                storage.units is not None
                or storage.system is not None
                or storage.coordinate_extra
            ):
                coordinate_value = CoordinateValue(
                    coordinate_value,
                    units=storage.units,
                    system=storage.system,
                    extra=copy.deepcopy(storage.coordinate_extra),
                )
            source = PointSource(
                name=(storage.names[index] if storage.names is not None else None),
                coordinates=coordinate_value,
                direction=(
                    storage.directions[index].tolist()
                    if storage.directions is not None
                    else None
                ),
            )
            rows.append(source.to_fs(ctx, include_domain=False))
        return rows

    def coordinates(self) -> np.ndarray:
        """Return inline source-point coordinates as a numeric array."""

        if self.geometry_type != "Inline":
            raise ValueError(
                "Source coordinates are only available for inline geometry"
            )
        if isinstance(self._storage, _BulkSourceGeometry):
            return self._storage.coordinates.copy()
        values = []
        for source in self.sources:
            coords = source.coordinates
            if isinstance(coords, CoordinateValue):
                coords = coords.value
            if is_quantity(coords):
                coords = coords.magnitude
            values.append(np.asarray(coords, dtype=float))
        return np.asarray(values, dtype=float)

    def coordinate_values(self) -> List[Any]:
        """Return inline source-point coordinates with authored metadata."""

        if self.geometry_type != "Inline":
            raise ValueError(
                "Source coordinates are only available for inline geometry"
            )
        if isinstance(self._storage, _BulkSourceGeometry):
            storage = self._storage
            if (
                storage.units is None
                and storage.system is None
                and not storage.coordinate_extra
            ):
                return [row.copy() for row in storage.coordinates]
            return [
                CoordinateValue(
                    row.tolist(),
                    units=storage.units,
                    system=storage.system,
                    extra=copy.deepcopy(storage.coordinate_extra),
                )
                for row in storage.coordinates
            ]
        return [copy.deepcopy(source.coordinates) for source in self.sources]


@dataclass
class EncodedSource:
    """One encoded solver source field over physical source points."""

    name: Optional[str] = None
    terms: Dict[str, Any] = field(default_factory=dict)
    coefficients: Optional[Union[Sequence[Any], np.ndarray]] = None
    reference_coordinates: Optional[Any] = None

    @classmethod
    def named(cls, name: str, terms: Mapping[str, Any]) -> "EncodedSource":
        return cls(name=name, terms=dict(terms))

    @classmethod
    def dense(
        cls,
        coefficients: Sequence[Any],
        *,
        name: Optional[str] = None,
        reference_coordinates: Optional[Any] = None,
    ) -> "EncodedSource":
        if reference_coordinates is not None:
            warnings.warn(
                "EncodedSource reference_coordinates is deprecated; use the "
                "simulation coordinate system and physical source geometry",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            name=name,
            coefficients=_source_coefficient_array(
                coefficients,
                ndim=1,
                label="EncodedSource coefficients",
            ),
            reference_coordinates=reference_coordinates,
        )

    @classmethod
    def from_named_fs(cls, data: Mapping[str, Any]) -> "EncodedSource":
        payload = copy.deepcopy(dict(data))
        terms = {
            str(term["source"]): term["coefficient"]
            for term in payload.pop("terms", [])
        }
        return cls(name=payload.pop("name", None), terms=terms)

    @classmethod
    def from_dense_fs(cls, data: Mapping[str, Any]) -> "EncodedSource":
        payload = copy.deepcopy(dict(data))
        return cls(
            name=payload.pop("name", None),
            coefficients=_source_coefficient_array(
                payload.pop("coefficients"),
                ndim=1,
                label="EncodedSource coefficients",
            ),
            reference_coordinates=(
                CoordinateValue.from_fs(payload.pop("reference_coordinates"))
                if "reference_coordinates" in payload
                else None
            ),
        )

    def to_named_fs(self) -> Dict[str, Any]:
        if not self.terms:
            raise ValueError("EncodedSource requires at least one term")
        payload: Dict[str, Any] = {
            **({"name": self.name} if self.name is not None else {}),
            "terms": [
                {"source": str(source), "coefficient": _complex_to_fs(coefficient)}
                for source, coefficient in self.terms.items()
                if _coefficient_abs(coefficient) != 0.0
            ],
        }
        if not payload["terms"]:
            raise ValueError("EncodedSource needs a nonzero coefficient")
        return payload

    def to_dense_fs(self) -> Dict[str, Any]:
        if self.coefficients is None:
            raise ValueError("Dense EncodedSource requires coefficients")
        coefficients = [_complex_to_fs(value) for value in self.coefficients]
        if not any(_coefficient_abs(value) != 0.0 for value in self.coefficients):
            raise ValueError("Dense EncodedSource needs a nonzero coefficient")
        payload: Dict[str, Any] = {
            **({"name": self.name} if self.name is not None else {}),
            "coefficients": coefficients,
        }
        if self.reference_coordinates is not None:
            payload["reference_coordinates"] = coordinate_value_to_fs(
                self.reference_coordinates
            )
        return payload

    def conjugated(self) -> "EncodedSource":
        """Return a copy with all source-encoding coefficients conjugated."""

        source = copy.deepcopy(self)
        source.terms = {
            name: _complex_value(value).conjugate()
            for name, value in source.terms.items()
        }
        if source.coefficients is not None:
            source.coefficients = np.conjugate(source.coefficients)
        return source

    time_reversed = conjugated


class DistributedSource(EncodedSource):
    """Deprecated name for :class:`EncodedSource`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "DistributedSource is deprecated; use EncodedSource.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@dataclass
class _NamedSourceEncoding:
    fields: List[EncodedSource]


@dataclass
class _DenseSourceEncoding:
    fields: List[EncodedSource]
    coefficients: np.ndarray


@dataclass
class _FrequencyDenseSourceEncoding:
    fields: List[EncodedSource]
    coefficients: np.ndarray
    frequencies: np.ndarray


@dataclass
class _HDF5SourceEncoding:
    file: Union[str, Path]
    dataset: str
    field_names_dataset: Optional[str] = None
    frequencies_dataset: Optional[str] = None
    count: Optional[int] = None


SourceEncodingStorage = Union[
    _NamedSourceEncoding,
    _DenseSourceEncoding,
    _FrequencyDenseSourceEncoding,
    _HDF5SourceEncoding,
]


@dataclass(init=False)
class SourceEncoding(ExtraFieldsMixin):
    """Optional source encoding with one concrete storage representation."""

    name: Optional[str]
    conjugate_coefficients: bool
    extra: Dict[str, Any]
    _storage: SourceEncodingStorage

    def __init__(
        self,
        *,
        encoding_type: str,
        name: Optional[str] = None,
        fields: Optional[Iterable[EncodedSource]] = None,
        weights: Optional[Any] = None,
        coefficients: Optional[Any] = None,
        frequencies: Optional[Any] = None,
        file: Optional[Union[str, Path]] = None,
        dataset: Optional[str] = None,
        field_names_dataset: Optional[str] = None,
        frequencies_dataset: Optional[str] = None,
        count: Optional[int] = None,
        conjugate_coefficients: bool = False,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        normalized_type = str(encoding_type).strip().lower()
        field_objects = list(fields or [])
        if weights is not None and coefficients is not None:
            raise TypeError("Use either weights or coefficients, not both")
        if coefficients is not None:
            warnings.warn(
                "SourceEncoding(coefficients=...) is deprecated; use "
                "weights= with encoding-major axes instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            matrix_values: Optional[Any] = np.swapaxes(np.asarray(coefficients), -1, -2)
        else:
            matrix_values = weights
        matrix = (
            None
            if matrix_values is None
            else _source_coefficient_array(
                matrix_values,
                ndim=3 if normalized_type == "frequencydense" else 2,
                label="SourceEncoding weights",
            )
        )
        frequency_axis = None
        if frequencies is not None:
            frequency_axis = np.asarray(frequencies, dtype=np.float64)
            if frequency_axis.ndim != 1 or frequency_axis.size < 1:
                raise ValueError("SourceEncoding frequencies must be one-dimensional")
            if not np.all(np.isfinite(frequency_axis)):
                raise ValueError("SourceEncoding frequencies must be finite")
            if np.any(frequency_axis < 0.0):
                raise ValueError("SourceEncoding frequencies must be nonnegative")
            if np.unique(frequency_axis).size != frequency_axis.size:
                raise ValueError("SourceEncoding frequencies must be unique")
            frequency_axis = np.ascontiguousarray(frequency_axis)
        legacy_reference_dataset = kwargs.pop("reference_coordinates_dataset", None)
        if legacy_reference_dataset is not None:
            warnings.warn(
                "reference_coordinates_dataset is deprecated and ignored; Sauce "
                "computes encoded-source reference coordinates from source geometry",
                DeprecationWarning,
                stacklevel=2,
            )
        normalized_count = None if count is None else int(count)
        if not isinstance(conjugate_coefficients, (bool, np.bool_)):
            raise TypeError("conjugate_coefficients must be boolean")
        self.conjugate_coefficients = bool(conjugate_coefficients)
        self._init_extra(extra, **kwargs)
        if normalized_type == "named":
            if not field_objects:
                raise ValueError("Named source encoding requires fields")
            if matrix is not None or file is not None or dataset:
                raise ValueError(
                    "Named source encoding cannot include dense or HDF5 storage"
                )
            if (
                field_names_dataset is not None
                or frequencies_dataset is not None
                or normalized_count is not None
                or frequency_axis is not None
            ):
                raise ValueError("Named source encoding cannot include HDF5 metadata")
            self._storage = _NamedSourceEncoding(field_objects)
        elif normalized_type == "jsondense":
            if matrix is None:
                if not field_objects:
                    raise ValueError("JsonDense source encoding requires fields")
                columns = [
                    _source_coefficient_array(
                        field_obj.coefficients,
                        ndim=1,
                        label="EncodedSource coefficients",
                    )
                    for field_obj in field_objects
                ]
                lengths = {len(column) for column in columns}
                if len(lengths) != 1:
                    raise ValueError(
                        "JsonDense fields must have the same source coefficient count"
                    )
                matrix = np.stack(columns).astype(np.complex64, copy=False)
            if not field_objects:
                raise ValueError("JsonDense source encoding requires fields")
            if file is not None or dataset:
                raise ValueError(
                    "JsonDense source encoding cannot include HDF5 storage"
                )
            if (
                field_names_dataset is not None
                or frequencies_dataset is not None
                or normalized_count is not None
                or frequency_axis is not None
            ):
                raise ValueError(
                    "JsonDense source encoding cannot include HDF5 metadata"
                )
            if matrix.shape[0] != len(field_objects):
                raise ValueError(
                    "JsonDense weight rows must match the encoded field count"
                )
            self._storage = _DenseSourceEncoding(field_objects, matrix)
            for index, field_obj in enumerate(field_objects):
                field_obj.coefficients = matrix[index, :]
        elif normalized_type == "frequencydense":
            if matrix is None or frequency_axis is None or not field_objects:
                raise ValueError(
                    "FrequencyDense source encoding requires weights, "
                    "frequencies, and fields"
                )
            if file is not None or dataset:
                raise ValueError(
                    "FrequencyDense source encoding cannot include HDF5 storage"
                )
            if field_names_dataset is not None or frequencies_dataset is not None:
                raise ValueError(
                    "FrequencyDense source encoding cannot include HDF5 metadata"
                )
            if normalized_count is not None:
                raise ValueError("FrequencyDense source encoding infers its count")
            if matrix.shape[0] != frequency_axis.size:
                raise ValueError("FrequencyDense weight slices must match frequencies")
            if matrix.shape[1] != len(field_objects):
                raise ValueError("FrequencyDense weight rows must match encoded names")
            self._storage = _FrequencyDenseSourceEncoding(
                field_objects,
                matrix,
                frequency_axis,
            )
        elif normalized_type == "hdf5dense":
            if file is None or not dataset:
                raise ValueError("HDF5Dense source encoding requires file and dataset")
            if field_names_dataset is not None and not str(field_names_dataset).strip():
                raise ValueError(
                    "HDF5Dense source field_names_dataset must not be empty"
                )
            if frequencies_dataset is not None and not str(frequencies_dataset).strip():
                raise ValueError(
                    "HDF5Dense source frequencies_dataset must not be empty"
                )
            if field_objects or matrix is not None:
                raise ValueError(
                    "HDF5Dense source encoding cannot include inline fields"
                )
            self._storage = _HDF5SourceEncoding(
                file=file,
                dataset=str(dataset),
                field_names_dataset=field_names_dataset,
                frequencies_dataset=frequencies_dataset,
                count=normalized_count,
            )
        else:
            raise ValueError(
                "source encoding type must be Named, JsonDense, FrequencyDense, "
                "or HDF5Dense"
            )
        if normalized_count is not None and normalized_count < 1:
            raise ValueError("source encoding count must be >= 1")
        self._validate_fields()

    @property
    def encoding_type(self) -> str:
        if isinstance(self._storage, _NamedSourceEncoding):
            return "Named"
        if isinstance(self._storage, _DenseSourceEncoding):
            return "JsonDense"
        if isinstance(self._storage, _FrequencyDenseSourceEncoding):
            return "FrequencyDense"
        return "HDF5Dense"

    @property
    def fields(self) -> List[EncodedSource]:
        return getattr(self._storage, "fields", [])

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """Compatibility alias for :attr:`weights`."""

        return self.weights

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Encoding-major complex weights.

        Static weights have shape ``(encoded_source, physical_source)``;
        frequency-dependent weights have shape
        ``(frequency, encoded_source, physical_source)``.
        """

        if isinstance(
            self._storage,
            (_DenseSourceEncoding, _FrequencyDenseSourceEncoding),
        ):
            return self._storage.coefficients
        return None

    @property
    def frequencies(self) -> Optional[np.ndarray]:
        if isinstance(self._storage, _FrequencyDenseSourceEncoding):
            return self._storage.frequencies
        return None

    @property
    def file(self) -> Optional[Union[str, Path]]:
        return (
            self._storage.file
            if isinstance(self._storage, _HDF5SourceEncoding)
            else None
        )

    @property
    def dataset(self) -> Optional[str]:
        return (
            self._storage.dataset
            if isinstance(self._storage, _HDF5SourceEncoding)
            else None
        )

    @property
    def field_names_dataset(self) -> Optional[str]:
        if isinstance(self._storage, _HDF5SourceEncoding):
            return self._storage.field_names_dataset
        return None

    @property
    def frequencies_dataset(self) -> Optional[str]:
        if isinstance(self._storage, _HDF5SourceEncoding):
            return self._storage.frequencies_dataset
        return None

    @property
    def count(self) -> Optional[int]:
        return (
            self._storage.count
            if isinstance(self._storage, _HDF5SourceEncoding)
            else None
        )

    def _validate_fields(self) -> None:
        """Validate fields after the concrete storage variant is selected."""

        if self.encoding_type == "Named":
            if any(
                field.coefficients is not None
                or field.reference_coordinates is not None
                for field in self.fields
            ):
                raise ValueError(
                    "Named source fields cannot include dense coefficients or "
                    "reference coordinates"
                )
            for field_obj in self.fields:
                if not field_obj.terms:
                    raise ValueError("Named source fields require at least one term")
                values = np.asarray(
                    [_complex_value(value) for value in field_obj.terms.values()],
                    dtype=np.complex64,
                )
                if not np.any(values != 0.0):
                    raise ValueError("Named source fields require a nonzero term")
        elif self.encoding_type == "JsonDense":
            if any(field.terms for field in self.fields):
                raise ValueError("JsonDense source fields cannot include named terms")
            self.validate_dense_shape()
            if not np.all(np.any(self.weights != 0.0, axis=1)):
                raise ValueError(
                    "Every JsonDense source field requires a nonzero coefficient"
                )
        elif self.encoding_type == "FrequencyDense":
            if any(
                field.terms
                or field.coefficients is not None
                or field.reference_coordinates is not None
                for field in self.fields
            ):
                raise ValueError(
                    "FrequencyDense fields contain names only; coefficients and "
                    "coordinates live in bulk storage"
                )
            self.validate_dense_shape()
            if not np.all(np.any(self.weights != 0.0, axis=2)):
                raise ValueError(
                    "Every FrequencyDense field requires a nonzero coefficient "
                    "at every frequency"
                )

    def validate_dense_shape(self, source_count: Optional[int] = None) -> None:
        """Validate dense rows and absorb compatible field assignment."""

        if isinstance(self._storage, _FrequencyDenseSourceEncoding):
            if source_count is not None and self._storage.coefficients.shape[2] != int(
                source_count
            ):
                raise ValueError(
                    "FrequencyDense source dimension must match physical "
                    "source-point count"
                )
            return
        if not isinstance(self._storage, _DenseSourceEncoding):
            return
        matrix = self._storage.coefficients
        for index, field_obj in enumerate(self._storage.fields):
            values = _source_coefficient_array(
                field_obj.coefficients,
                ndim=1,
                label="EncodedSource coefficients",
            )
            if len(values) != matrix.shape[1]:
                raise ValueError(
                    "JsonDense coefficient count must match physical source-point count"
                )
            if not np.shares_memory(values, matrix):
                matrix[index, :] = values
                field_obj.coefficients = matrix[index, :]
        if source_count is not None and matrix.shape[1] != int(source_count):
            raise ValueError(
                "JsonDense coefficient count must match physical source-point count"
            )

    @classmethod
    def named(
        cls,
        fields: Union[
            Mapping[str, Mapping[str, Any]],
            Iterable[Union[EncodedSource, Mapping[str, Any]]],
        ],
        *,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "SourceEncoding":
        """Create sparse named source encoding."""

        if isinstance(fields, Mapping):
            field_objects = [
                EncodedSource.named(field_name, terms)
                for field_name, terms in fields.items()
            ]
        else:
            field_objects = [
                (
                    field
                    if isinstance(field, EncodedSource)
                    else EncodedSource.from_named_fs(field)
                )
                for field in fields
            ]
        return cls(
            encoding_type="Named",
            name=name,
            fields=field_objects,
            **kwargs,
        )

    @classmethod
    def dense(
        cls,
        weights: Optional[Any] = None,
        *,
        coefficients: Optional[Any] = None,
        names: Optional[Iterable[str]] = None,
        reference_coordinates: Optional[Any] = None,
        name: Optional[str] = None,
        conjugate: bool = False,
        **kwargs: Any,
    ) -> "SourceEncoding":
        """Create dense encoding from an ``n_encoded x n_source`` matrix.

        Each row defines one encoded field and each column follows physical
        source-geometry order. Positional input and ``weights=`` use the same
        field-major convention as the solver's JSON and HDF5 contracts. The
        deprecated ``coefficients=`` keyword accepts source-major input and
        transposes it; positional input is never transposed implicitly.

        Saved simulations materialize the coefficients in HDF5. Calling
        :meth:`to_fs` without a store retains a compact JSON representation for
        interactive examples and compatibility.

        Args:
            weights: Real or complex encoding-major weight matrix.
            names: Optional encoded-source labels.
            reference_coordinates: Optional reference coordinate per field.
            name: Optional encoding name.
            conjugate: Conjugate the authored coefficients, which implements
                frequency-domain time reversal for forward responses.
        """

        if weights is not None and coefficients is not None:
            raise TypeError("Use either weights or coefficients, not both")
        if coefficients is not None:
            warnings.warn(
                "SourceEncoding.dense(coefficients=...) is deprecated; use "
                "weights= with encoding-major axes instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            matrix = np.asarray(coefficients)
            matrix = (
                matrix.reshape(1, -1)
                if matrix.ndim == 1
                else np.swapaxes(matrix, -1, -2)
            )
        elif weights is not None:
            matrix = np.asarray(weights)
        else:
            raise TypeError("SourceEncoding.dense() requires weights")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        matrix = _source_coefficient_array(
            matrix,
            ndim=2,
            label="dense weights",
        )
        n_fields = int(matrix.shape[0])
        field_names = (
            _source_names(names, n_fields)
            if names is not None
            else [f"field_{index:06d}" for index in range(1, n_fields + 1)]
        )
        if reference_coordinates is not None:
            warnings.warn(
                "SourceEncoding reference_coordinates is deprecated; use the "
                "simulation coordinate system and physical source geometry",
                DeprecationWarning,
                stacklevel=2,
            )
        refs = _reference_rows(reference_coordinates, n_fields)
        fields = [
            EncodedSource(
                name=field_names[index],
                reference_coordinates=refs[index],
            )
            for index in range(n_fields)
        ]
        return cls(
            encoding_type="JsonDense",
            name=name,
            fields=fields,
            weights=matrix,
            conjugate_coefficients=conjugate,
            **kwargs,
        )

    @classmethod
    def frequency_dense(
        cls,
        weights: Optional[Any] = None,
        frequencies: Optional[Any] = None,
        *,
        coefficients: Optional[Any] = None,
        names: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
        conjugate: bool = False,
        **kwargs: Any,
    ) -> "SourceEncoding":
        """Create frequency-dependent dense source encoding.

        Args:
            weights: Complex tensor with shape
                ``(n_frequency, n_encoded, n_source)``.
            frequencies: Physical frequency in Hz for every tensor slice.
            names: Optional encoded-field names.
            name: Optional encoding name.
            conjugate: Conjugate coefficients lazily in Sauce.

        Saved simulations materialize the tensor and its small frequency axis
        in HDF5. Inline JSON serialization is intentionally unsupported.
        """

        if weights is not None and coefficients is not None:
            raise TypeError("Use either weights or coefficients, not both")
        if coefficients is not None:
            warnings.warn(
                "SourceEncoding.frequency_dense(coefficients=...) is "
                "deprecated; use weights= with encoding-major axes instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            weights = np.swapaxes(np.asarray(coefficients), -1, -2)
        if weights is None:
            raise TypeError("SourceEncoding.frequency_dense() requires weights")
        if frequencies is None:
            raise TypeError("SourceEncoding.frequency_dense() requires frequencies")
        tensor = _source_coefficient_array(
            weights,
            ndim=3,
            label="frequency-dependent dense weights",
        )
        n_fields = int(tensor.shape[1])
        field_names = (
            _source_names(names, n_fields)
            if names is not None
            else [f"field_{index:06d}" for index in range(1, n_fields + 1)]
        )
        fields = [EncodedSource(name=field_name) for field_name in field_names]
        return cls(
            encoding_type="FrequencyDense",
            name=name,
            fields=fields,
            weights=tensor,
            frequencies=frequencies,
            conjugate_coefficients=conjugate,
            **kwargs,
        )

    @classmethod
    def hdf5(
        cls,
        file: Union[str, Path],
        *,
        dataset: str,
        name: Optional[str] = None,
        field_names_dataset: Optional[str] = None,
        frequencies_dataset: Optional[str] = None,
        count: Optional[int] = None,
        conjugate: bool = False,
        **kwargs: Any,
    ) -> "SourceEncoding":
        """Create HDF5 dense source encoding.

        Static weights use h5py shape ``(field, source, complex=2)``.
        Frequency-dependent weights use shape
        ``(frequency, field, source, complex=2)`` and require
        ``frequencies_dataset``.
        """

        legacy_reference_dataset = kwargs.pop("reference_coordinates_dataset", None)
        if legacy_reference_dataset is not None:
            warnings.warn(
                "reference_coordinates_dataset is deprecated and ignored; Sauce "
                "computes encoded-source reference coordinates from source geometry",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            encoding_type="HDF5Dense",
            name=name,
            file=file,
            dataset=dataset,
            field_names_dataset=field_names_dataset,
            frequencies_dataset=frequencies_dataset,
            count=count,
            conjugate_coefficients=conjugate,
            **kwargs,
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SourceEncoding":
        payload = copy.deepcopy(dict(data))
        encoding_type = payload.pop("_type", payload.pop("encoding_type", None))
        if encoding_type is None:
            raise ValueError("SourceEncoding payload requires _type or encoding_type")
        if encoding_type == "Named":
            fields = [
                EncodedSource.from_named_fs(field)
                for field in payload.pop("fields", [])
            ]
        elif encoding_type == "JsonDense":
            fields = [
                EncodedSource.from_dense_fs(field)
                for field in payload.pop("fields", [])
            ]
        else:
            fields = []
        legacy_reference_dataset = payload.pop(
            "reference_coordinates_dataset",
            None,
        )
        return cls(
            encoding_type=encoding_type,
            name=payload.pop("name", None),
            fields=fields,
            file=payload.pop("file", None),
            dataset=payload.pop("dataset", None),
            field_names_dataset=payload.pop("field_names_dataset", None),
            frequencies_dataset=payload.pop("frequencies_dataset", None),
            count=payload.pop("count", payload.pop("field_count", None)),
            conjugate_coefficients=payload.pop("conjugate_coefficients", False),
            extra=payload,
            **(
                {"reference_coordinates_dataset": legacy_reference_dataset}
                if legacy_reference_dataset is not None
                else {}
            ),
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        self.validate_dense_shape()
        store = getattr(ctx, "store", None) if ctx is not None else None
        if self.encoding_type == "FrequencyDense":
            if store is None:
                raise ValueError(
                    "FrequencyDense source encoding requires a simulation/project "
                    "store so coefficients can be materialized in HDF5"
                )
            tensor = self.weights
            assert tensor is not None and self.frequencies is not None
            n_frequency, n_field, n_source = tensor.shape

            def coefficient_chunks() -> Iterator[np.ndarray]:
                target_bytes = 32 * 1024 * 1024
                slice_bytes = max(
                    1,
                    n_field * n_source * 2 * np.dtype(np.float32).itemsize,
                )
                slices_per_chunk = max(1, target_bytes // slice_bytes)
                for start in range(0, n_frequency, slices_per_chunk):
                    stop = min(n_frequency, start + slices_per_chunk)
                    values = tensor[start:stop]
                    split = np.empty((*values.shape, 2), dtype=np.float32)
                    split[..., 0] = values.real
                    split[..., 1] = values.imag
                    yield split

            ref = store.put_array_chunks(
                "inputs/acquisition/source_encoding/coefficients",
                (n_frequency, n_field, n_source, 2),
                coefficient_chunks,
                attrs={
                    "fs_kind": "source_encoding_coefficients",
                    "frequency_axis_hash": hashlib.sha256(
                        memoryview(self.frequencies).cast("B")
                    ).hexdigest(),
                },
                dims=("frequency", "field", "source", "complex"),
                dtype=np.float32,
            )
            frequency_ref = store.put_array_chunks(
                "inputs/acquisition/source_encoding/frequencies",
                (n_frequency,),
                lambda: (self.frequencies,),
                attrs={
                    "fs_kind": "source_encoding_frequencies",
                    "units": "Hz",
                },
                dims=("frequency",),
                dtype=np.float64,
            )
            payload = {
                "_type": "HDF5Dense",
                **({"name": self.name} if self.name is not None else {}),
                **ref.to_fs(),
                "frequencies_dataset": frequency_ref.clean_dataset,
                **(
                    {"conjugate_coefficients": True}
                    if self.conjugate_coefficients
                    else {}
                ),
            }
            field_names = self.field_names()
            default_names = [f"field_{index:06d}" for index in range(1, n_field + 1)]
            if field_names != default_names:
                names_ref = store.put_string_array(
                    "inputs/acquisition/source_encoding/field_names",
                    field_names,
                    dimension="field",
                    attrs={"fs_kind": "source_encoding_field_names"},
                )
                payload["field_names_dataset"] = names_ref.clean_dataset
            return payload

        if self.encoding_type == "JsonDense" and store is not None:
            matrix = self.weights
            assert matrix is not None
            n_field, n_source = matrix.shape

            def coefficient_chunks() -> Iterator[np.ndarray]:
                target_bytes = 32 * 1024 * 1024
                row_bytes = max(1, n_source * 2 * np.dtype(np.float32).itemsize)
                rows_per_chunk = max(1, target_bytes // row_bytes)
                for start in range(0, n_field, rows_per_chunk):
                    stop = min(n_field, start + rows_per_chunk)
                    values = matrix[start:stop, :]
                    split = np.empty((*values.shape, 2), dtype=np.float32)
                    split[..., 0] = values.real
                    split[..., 1] = values.imag
                    yield split

            ref = store.put_array_chunks(
                "inputs/acquisition/source_encoding/coefficients",
                (n_field, n_source, 2),
                coefficient_chunks,
                attrs={"fs_kind": "source_encoding_coefficients"},
                dims=("field", "source", "complex"),
                dtype=np.float32,
            )
            payload = {
                "_type": "HDF5Dense",
                **({"name": self.name} if self.name is not None else {}),
                **ref.to_fs(),
                **(
                    {"conjugate_coefficients": True}
                    if self.conjugate_coefficients
                    else {}
                ),
            }
            field_names = self.field_names()
            default_names = [f"field_{index:06d}" for index in range(1, n_field + 1)]
            if field_names != default_names:
                names_ref = store.put_string_array(
                    "inputs/acquisition/source_encoding/field_names",
                    field_names,
                    dimension="field",
                    attrs={"fs_kind": "source_encoding_field_names"},
                )
                payload["field_names_dataset"] = names_ref.clean_dataset
            return payload

        payload = {
            "_type": self.encoding_type,
            **({"name": self.name} if self.name is not None else {}),
        }
        if self.encoding_type == "Named":
            payload["fields"] = [field.to_named_fs() for field in self.fields]
        elif self.encoding_type == "JsonDense":
            assert self.weights is not None
            if self.weights.size > 256:
                raise ValueError(
                    "JsonDense source encoding is limited to 256 coefficients; "
                    "save through a simulation/project context to materialize HDF5"
                )
            payload["fields"] = [field.to_dense_fs() for field in self.fields]
        else:
            assert self.file is not None
            payload["file"] = _path_to_fs(self.file, ctx)
            payload["dataset"] = self.dataset
            if self.field_names_dataset is not None:
                payload["field_names_dataset"] = self.field_names_dataset
            if self.frequencies_dataset is not None:
                payload["frequencies_dataset"] = self.frequencies_dataset
        if self.conjugate_coefficients:
            payload["conjugate_coefficients"] = True
        return merge_extra(payload, self.extra, "SourceEncoding")

    def _resolve_project_reference(self, project_path: Path) -> None:
        """Resolve a loaded local coefficient reference without reading weights."""

        if not isinstance(self._storage, _HDF5SourceEncoding):
            return
        text = str(self._storage.file)
        if text.startswith("remote:") or "://" in text:
            return
        path = Path(text).expanduser()
        if not path.is_absolute():
            self._storage.file = project_path / path

    @property
    def field_count(self) -> Optional[int]:
        if self.encoding_type == "HDF5Dense":
            if (
                self.count is None
                and self.file is not None
                and Path(self.file).is_file()
            ):
                with h5py.File(self.file, "r") as h5:
                    shape = h5[self.dataset].shape
                if len(shape) not in {3, 4} or shape[-1] != 2:
                    raise ValueError(
                        "Source encoding requires a split-complex rank-3 or rank-4 tensor"
                    )
                assert isinstance(self._storage, _HDF5SourceEncoding)
                self._storage.count = shape[-3]
            return self.count
        return len(self.fields)

    def field_names(self) -> List[str]:
        if self.encoding_type == "HDF5Dense":
            if self.file is None or not Path(self.file).is_file():
                return []
            if self.field_names_dataset is None:
                return []
            with h5py.File(self.file, "r") as h5:
                dataset = h5[self.field_names_dataset]
                if dataset.ndim != 1:
                    raise ValueError(
                        "Source field names must be a one-dimensional dataset"
                    )
                names = dataset.asstr()[:].tolist()
            return _source_names(names, self.field_count or len(names))
        return [
            field.name if field.name is not None else f"field_{index:06d}"
            for index, field in enumerate(self.fields, start=1)
        ]

    def reference_coordinate_values(self, geometry: SourceGeometry) -> List[Any]:
        """Return encoded field references while preserving coordinate metadata."""

        if self.encoding_type == "HDF5Dense":
            raise ValueError("HDF5 source-encoding reference coordinates are external")

        needs_computed_reference = any(
            field.reference_coordinates is None for field in self.fields
        )
        bulk_storage = (
            geometry._storage
            if isinstance(geometry._storage, _BulkSourceGeometry)
            else None
        )
        source_values: List[Any] = []
        index_by_name: Dict[str, int] = {}
        if needs_computed_reference and bulk_storage is None:
            source_values = geometry.coordinate_values()
            source_names = geometry.point_names()
            index_by_name = {name: index for index, name in enumerate(source_names)}
        elif needs_computed_reference and self.encoding_type == "Named":
            source_names = geometry.point_names()
            index_by_name = {name: index for index, name in enumerate(source_names)}
        refs = []

        for field_index, field_obj in enumerate(self.fields):
            explicit_ref = field_obj.reference_coordinates
            if explicit_ref is not None:
                refs.append(copy.deepcopy(explicit_ref))
                continue

            if bulk_storage is not None and self.encoding_type == "Named":
                indices = np.fromiter(
                    (index_by_name[str(source)] for source in field_obj.terms),
                    dtype=np.int64,
                    count=len(field_obj.terms),
                )
                weights = np.fromiter(
                    (
                        _coefficient_abs(coefficient)
                        for coefficient in field_obj.terms.values()
                    ),
                    dtype=np.float64,
                    count=len(field_obj.terms),
                )
                active = weights != 0.0
                source_coords = bulk_storage.coordinates[indices[active]]
                weights = weights[active]
                source_units = bulk_storage.units
                source_system = bulk_storage.system
            elif self.encoding_type == "Named":
                weights = np.zeros(len(source_values), dtype=float)
                for source, coefficient in field_obj.terms.items():
                    weights[index_by_name[str(source)]] += _coefficient_abs(coefficient)
                active_indices = np.flatnonzero(weights != 0.0)
                source_coords, source_units, source_system = _source_coordinate_matrix(
                    [source_values[index] for index in active_indices]
                )
                weights = weights[active_indices]
            elif isinstance(self._storage, _FrequencyDenseSourceEncoding):
                weights = np.sqrt(
                    np.mean(
                        np.abs(self._storage.coefficients[:, field_index, :]) ** 2,
                        axis=0,
                    )
                )
                if bulk_storage is not None:
                    source_coords = bulk_storage.coordinates
                    source_units = bulk_storage.units
                    source_system = bulk_storage.system
                else:
                    source_coords, source_units, source_system = (
                        _source_coordinate_matrix(source_values)
                    )
                active = weights != 0.0
                source_coords = source_coords[active]
                weights = weights[active]
            elif bulk_storage is not None:
                weights = np.abs(np.asarray(field_obj.coefficients))
                if len(weights) != len(bulk_storage.coordinates):
                    raise ValueError(
                        "JsonDense coefficient count must match physical "
                        "source-point count"
                    )
                active = weights != 0.0
                source_coords = bulk_storage.coordinates[active]
                weights = weights[active]
                source_units = bulk_storage.units
                source_system = bulk_storage.system
            else:
                assert field_obj.coefficients is not None
                weights = np.asarray(
                    [_coefficient_abs(value) for value in field_obj.coefficients],
                    dtype=float,
                )
                if len(weights) != len(source_values):
                    raise ValueError(
                        "JsonDense coefficient count must match physical "
                        "source-point count"
                    )
                active_indices = np.flatnonzero(weights != 0.0)
                source_coords, source_units, source_system = _source_coordinate_matrix(
                    [source_values[index] for index in active_indices]
                )
                weights = weights[active_indices]
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError("Cannot compute reference coordinates for zero field")
            reference = np.average(
                source_coords,
                axis=0,
                weights=weights,
            )
            if source_units is not None or source_system is not None:
                refs.append(
                    CoordinateValue(
                        reference.tolist(),
                        units=source_units,
                        system=source_system,
                    )
                )
            else:
                refs.append(reference)
        return refs

    def reference_coordinates(self, geometry: SourceGeometry) -> np.ndarray:
        """Return encoded field reference coordinates when computable."""

        refs = self.reference_coordinate_values(geometry)
        return np.asarray(
            [_coordinate_array(reference) for reference in refs],
            dtype=float,
        )

    def conjugated(self) -> "SourceEncoding":
        """Return a lazy conjugated view without copying weight arrays."""

        encoding = copy.copy(self)
        encoding._storage = copy.copy(self._storage)
        if isinstance(
            encoding._storage,
            (
                _NamedSourceEncoding,
                _DenseSourceEncoding,
                _FrequencyDenseSourceEncoding,
            ),
        ):
            encoding._storage.fields = [copy.copy(field) for field in self.fields]
        encoding.conjugate_coefficients = not self.conjugate_coefficients
        return encoding

    time_reversed = conjugated


from frequensolve.seismic._legacy_source_types import (  # noqa: E402
    CompoundSource,
    RuptureSource,
    Source,
    SourceGroup,
)
