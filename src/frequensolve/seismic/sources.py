"""Source geometry and source-encoding authoring helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from frequensolve.geometry.frame import (
    CoordinateValue,
    Direction,
    coordinate_value_to_fs,
    direction_to_fs,
)
from frequensolve.units import is_quantity, unit_expression, value_and_units_to_fs
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
    warn_deprecated_path_api,
)

__all__ = [
    "Source",
    "SourceGroup",
    "RuptureSource",
    "CompoundSource",
    "PointSource",
    "SourceGeometry",
    "SourceEncoding",
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
        direction_payload = direction_to_fs(direction)
        if isinstance(direction_payload, np.ndarray):
            direction_payload = direction_payload.tolist()
        elif isinstance(direction_payload, np.generic):
            direction_payload = direction_payload.item()
        payload["direction"] = direction_payload
    if amplitude is not None:
        payload["amplitude"] = value_and_units_to_fs(amplitude)
    mechanism_payload = _mechanism_to_fs(mechanism)
    if mechanism_payload is not None:
        payload["mechanism"] = mechanism_payload
    if extra:
        payload.update(copy.deepcopy(dict(extra)))
    return payload


def _source_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if not value:
        raise ValueError("source kind must be non-empty")
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
        values = np.asarray(coords.to(target_units).magnitude, dtype=float)
        units = target_units
    else:
        values = np.asarray(coords, dtype=float)

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
        return [f"source_{index:03d}" for index in range(1, n_sources + 1)]
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
        return real if imag == 0.0 else [real, imag]
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, Number) for item in value)
    ):
        real = float(value[0])
        imag = float(value[1])
        return real if imag == 0.0 else [real, imag]
    raise TypeError(f"Invalid source-encoding coefficient {value!r}")


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
            Direction.from_fs(payload.pop("direction"))
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


@dataclass(init=False)
class SourceGeometry(ExtraFieldsMixin):
    """Physical source catalog used by an acquisition."""

    kind: str
    geometry_type: str
    name: Optional[str]
    domain: Optional[int]
    sources: List[PointSource]
    file: Optional[Union[str, Path]]
    dataset: Optional[str]
    source_file: Optional[Union[str, Path]]
    system: Optional[str]
    units: Optional[Any]
    defaults: Dict[str, Any]
    extra: Dict[str, Any]

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
        source_file: Optional[Union[str, Path]] = None,
        system: Optional[str] = None,
        units: Optional[Any] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.kind = _source_kind(kind)
        self.geometry_type = geometry_type
        self.name = name
        self.domain = None if domain is None else int(domain)
        self.sources = [_as_source_point(source) for source in (sources or [])]
        self.file = file
        self.dataset = dataset
        self.source_file = source_file
        self.system = system
        self.units = units
        self.defaults = copy.deepcopy(dict(defaults or {}))
        self._init_extra(extra, **kwargs)
        self._validate()

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

        rows = _coordinate_rows(coords, units=units, system=system)
        source_names = _source_names(names, len(rows))
        direction_rows = None
        if direction is not None and not isinstance(direction, Mapping):
            try:
                direction_array = np.asarray(direction, dtype=float)
            except (TypeError, ValueError):
                direction_array = None
            if direction_array is not None and direction_array.ndim == 2:
                if len(direction_array) != len(rows):
                    raise ValueError("direction must have one row per coordinate")
                direction_rows = direction_array.tolist()

        source_points = [
            PointSource(name=source_name, coordinates=row)
            for source_name, row in zip(source_names, rows)
        ]
        if direction_rows is not None:
            for source, source_direction in zip(source_points, direction_rows):
                source.direction = source_direction
        default_payload = _basis_to_fs(
            direction=None if direction_rows is not None else direction,
            amplitude=amplitude,
            mechanism=mechanism,
            extra=defaults,
        )
        return cls(
            geometry_type="Inline",
            name=name,
            kind=kind,
            domain=domain,
            defaults=default_payload,
            sources=source_points,
            **kwargs,
        )

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
            system=system,
            units=units,
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
            source_file=payload.pop("source_file", None),
            system=payload.pop("system", None),
            units=payload.pop("units", None),
            defaults=payload.pop("defaults", None),
            extra=payload,
        )

    def _validate(self) -> None:
        geometry_type = self.geometry_type
        if geometry_type not in {"Inline", "HDF5", "SPSFiles"}:
            raise ValueError("source geometry type must be Inline, HDF5, or SPSFiles")
        if geometry_type == "Inline":
            if not self.sources:
                raise ValueError("Inline source geometry requires at least one source")
            names = [source.name for source in self.sources if source.name is not None]
            if len(names) != len(set(names)):
                raise ValueError("source names must be unique")
        elif geometry_type == "HDF5":
            if self.file is None or not self.dataset:
                raise ValueError("HDF5 source geometry requires file and dataset")
        elif not self.source_file:
            raise ValueError("SPS source geometry requires source_file")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "_type": self.geometry_type,
            **({"name": self.name} if self.name is not None else {}),
            **({"domain": self.domain} if self.domain is not None else {}),
            "kind": self.kind,
        }
        if self.defaults:
            payload["defaults"] = copy.deepcopy(self.defaults)
        if self.geometry_type == "Inline":
            payload["sources"] = [
                source.to_fs(ctx, include_domain=False) for source in self.sources
            ]
        elif self.geometry_type == "HDF5":
            payload["file"] = _path_to_fs(self.file, ctx)
            payload["dataset"] = self.dataset
            if self.system is not None:
                payload["system"] = self.system
            if self.units is not None:
                payload["units"] = unit_expression(self.units)
        else:
            payload["source_file"] = _path_to_fs(self.source_file, ctx)
            if self.system is not None:
                payload["system"] = self.system
            if self.units is not None:
                payload["units"] = unit_expression(self.units)
        return merge_extra(payload, self.extra, "SourceGeometry")

    @property
    def point_count(self) -> Optional[int]:
        if self.geometry_type != "Inline":
            return None
        return len(self.sources)

    def point_names(self) -> List[str]:
        if self.geometry_type != "Inline":
            return []
        return [
            source.name if source.name is not None else f"source_{index:06d}"
            for index, source in enumerate(self.sources, start=1)
        ]

    def coordinates(self) -> np.ndarray:
        """Return inline source-point coordinates as a numeric array."""

        if self.geometry_type != "Inline":
            raise ValueError(
                "Source coordinates are only available for inline geometry"
            )
        values = []
        for source in self.sources:
            coords = source.coordinates
            if isinstance(coords, CoordinateValue):
                coords = coords.value
            if is_quantity(coords):
                coords = coords.magnitude
            values.append(np.asarray(coords, dtype=float))
        return np.asarray(values, dtype=float)


@dataclass
class DistributedSource:
    """One simulated source field distributed over physical point sources."""

    name: Optional[str] = None
    terms: Dict[str, Any] = field(default_factory=dict)
    coefficients: Optional[Sequence[Any]] = None
    reference_coordinates: Optional[Any] = None

    @classmethod
    def named(cls, name: str, terms: Mapping[str, Any]) -> "DistributedSource":
        return cls(name=name, terms=dict(terms))

    @classmethod
    def dense(
        cls,
        coefficients: Sequence[Any],
        *,
        name: Optional[str] = None,
        reference_coordinates: Optional[Any] = None,
    ) -> "DistributedSource":
        return cls(
            name=name,
            coefficients=list(coefficients),
            reference_coordinates=reference_coordinates,
        )

    @classmethod
    def from_named_fs(cls, data: Mapping[str, Any]) -> "DistributedSource":
        payload = copy.deepcopy(dict(data))
        terms = {
            str(term["source"]): term["coefficient"]
            for term in payload.pop("terms", [])
        }
        return cls(name=payload.pop("name", None), terms=terms)

    @classmethod
    def from_dense_fs(cls, data: Mapping[str, Any]) -> "DistributedSource":
        payload = copy.deepcopy(dict(data))
        return cls(
            name=payload.pop("name", None),
            coefficients=payload.pop("coefficients"),
            reference_coordinates=(
                CoordinateValue.from_fs(payload.pop("reference_coordinates"))
                if "reference_coordinates" in payload
                else None
            ),
        )

    def to_named_fs(self) -> Dict[str, Any]:
        if not self.terms:
            raise ValueError("DistributedSource requires at least one term")
        payload: Dict[str, Any] = {
            **({"name": self.name} if self.name is not None else {}),
            "terms": [
                {"source": str(source), "coefficient": _complex_to_fs(coefficient)}
                for source, coefficient in self.terms.items()
                if _coefficient_abs(coefficient) != 0.0
            ],
        }
        if not payload["terms"]:
            raise ValueError("DistributedSource needs a nonzero coefficient")
        return payload

    def to_dense_fs(self) -> Dict[str, Any]:
        if self.coefficients is None:
            raise ValueError("Dense DistributedSource requires coefficients")
        coefficients = [_complex_to_fs(value) for value in self.coefficients]
        if not any(_coefficient_abs(value) != 0.0 for value in self.coefficients):
            raise ValueError("Dense DistributedSource needs a nonzero coefficient")
        payload: Dict[str, Any] = {
            **({"name": self.name} if self.name is not None else {}),
            "coefficients": coefficients,
        }
        if self.reference_coordinates is not None:
            payload["reference_coordinates"] = coordinate_value_to_fs(
                self.reference_coordinates
            )
        return payload


@dataclass(init=False)
class SourceEncoding(ExtraFieldsMixin):
    """Optional encoding from physical source points to RHS/source fields."""

    encoding_type: str
    name: Optional[str]
    fields: List[DistributedSource]
    file: Optional[Union[str, Path]]
    dataset: Optional[str]
    field_names_dataset: Optional[str]
    reference_coordinates_dataset: Optional[str]
    extra: Dict[str, Any]

    def __init__(
        self,
        *,
        encoding_type: str,
        name: Optional[str] = None,
        fields: Optional[Iterable[DistributedSource]] = None,
        file: Optional[Union[str, Path]] = None,
        dataset: Optional[str] = None,
        field_names_dataset: Optional[str] = None,
        reference_coordinates_dataset: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.encoding_type = encoding_type
        self.name = name
        self.fields = list(fields or [])
        self.file = file
        self.dataset = dataset
        self.field_names_dataset = field_names_dataset
        self.reference_coordinates_dataset = reference_coordinates_dataset
        self._init_extra(extra, **kwargs)
        self._validate()

    @classmethod
    def named(
        cls,
        fields: Union[
            Mapping[str, Mapping[str, Any]],
            Iterable[Union[DistributedSource, Mapping[str, Any]]],
        ],
        *,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "SourceEncoding":
        """Create sparse named source encoding."""

        if isinstance(fields, Mapping):
            field_objects = [
                DistributedSource.named(field_name, terms)
                for field_name, terms in fields.items()
            ]
        else:
            field_objects = [
                (
                    field
                    if isinstance(field, DistributedSource)
                    else DistributedSource.from_named_fs(field)
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
        coefficients: Any,
        *,
        names: Optional[Iterable[str]] = None,
        reference_coordinates: Optional[Any] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "SourceEncoding":
        """Create JSON dense encoding from an ``n_source x n_field`` matrix."""

        matrix = np.asarray(coefficients)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2:
            raise ValueError("dense coefficients must have shape (n_source, n_field)")
        n_fields = int(matrix.shape[1])
        field_names = (
            _source_names(names, n_fields)
            if names is not None
            else [f"field_{index:03d}" for index in range(1, n_fields + 1)]
        )
        refs = _reference_rows(reference_coordinates, n_fields)
        fields = [
            DistributedSource.dense(
                matrix[:, index].tolist(),
                name=field_names[index],
                reference_coordinates=refs[index],
            )
            for index in range(n_fields)
        ]
        return cls(
            encoding_type="JsonDense",
            name=name,
            fields=fields,
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
        reference_coordinates_dataset: Optional[str] = None,
        **kwargs: Any,
    ) -> "SourceEncoding":
        """Create HDF5 dense source encoding."""

        return cls(
            encoding_type="HDF5Dense",
            name=name,
            file=file,
            dataset=dataset,
            field_names_dataset=field_names_dataset,
            reference_coordinates_dataset=reference_coordinates_dataset,
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
                DistributedSource.from_named_fs(field)
                for field in payload.pop("fields", [])
            ]
        elif encoding_type == "JsonDense":
            fields = [
                DistributedSource.from_dense_fs(field)
                for field in payload.pop("fields", [])
            ]
        else:
            fields = []
        return cls(
            encoding_type=encoding_type,
            name=payload.pop("name", None),
            fields=fields,
            file=payload.pop("file", None),
            dataset=payload.pop("dataset", None),
            field_names_dataset=payload.pop("field_names_dataset", None),
            reference_coordinates_dataset=payload.pop(
                "reference_coordinates_dataset", None
            ),
            extra=payload,
        )

    def _validate(self) -> None:
        if self.encoding_type not in {"Named", "JsonDense", "HDF5Dense"}:
            raise ValueError(
                "source encoding type must be Named, JsonDense, or HDF5Dense"
            )
        if self.encoding_type in {"Named", "JsonDense"} and not self.fields:
            raise ValueError(f"{self.encoding_type} source encoding requires fields")
        if self.encoding_type == "HDF5Dense" and (
            self.file is None or not self.dataset
        ):
            raise ValueError("HDF5Dense source encoding requires file and dataset")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "_type": self.encoding_type,
            **({"name": self.name} if self.name is not None else {}),
        }
        if self.encoding_type == "Named":
            payload["fields"] = [field.to_named_fs() for field in self.fields]
        elif self.encoding_type == "JsonDense":
            payload["fields"] = [field.to_dense_fs() for field in self.fields]
        else:
            payload["file"] = _path_to_fs(self.file, ctx)
            payload["dataset"] = self.dataset
            if self.field_names_dataset is not None:
                payload["field_names_dataset"] = self.field_names_dataset
            if self.reference_coordinates_dataset is not None:
                payload["reference_coordinates_dataset"] = (
                    self.reference_coordinates_dataset
                )
        return merge_extra(payload, self.extra, "SourceEncoding")

    @property
    def field_count(self) -> Optional[int]:
        if self.encoding_type == "HDF5Dense":
            return None
        return len(self.fields)

    def field_names(self) -> List[str]:
        if self.encoding_type == "HDF5Dense":
            return []
        return [
            field.name if field.name is not None else f"field_{index:03d}"
            for index, field in enumerate(self.fields, start=1)
        ]

    def reference_coordinates(self, geometry: SourceGeometry) -> np.ndarray:
        """Return encoded field reference coordinates when computable."""

        if self.encoding_type == "HDF5Dense":
            raise ValueError("HDF5 source-encoding reference coordinates are external")

        source_coords = geometry.coordinates()
        source_names = geometry.point_names()
        index_by_name = {name: index for index, name in enumerate(source_names)}
        refs = []

        for field_obj in self.fields:
            explicit_ref = field_obj.reference_coordinates
            if explicit_ref is not None:
                value = (
                    explicit_ref.value
                    if isinstance(explicit_ref, CoordinateValue)
                    else explicit_ref
                )
                refs.append(np.asarray(value, dtype=float))
                continue

            if self.encoding_type == "Named":
                weights = np.zeros(len(source_coords), dtype=float)
                for source, coefficient in field_obj.terms.items():
                    weights[index_by_name[str(source)]] += _coefficient_abs(coefficient)
            else:
                weights = np.asarray(
                    [_coefficient_abs(value) for value in field_obj.coefficients],
                    dtype=float,
                )
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError("Cannot compute reference coordinates for zero field")
            refs.append(np.average(source_coords, axis=0, weights=weights))
        return np.asarray(refs, dtype=float)


class Source:
    """Compatibility dispatcher for legacy source-group payloads."""

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> Any:
        payload = copy.deepcopy(dict(data))
        source_type = payload.pop("_type", "PointSource")
        if source_type == "PointSource":
            return PointSource.from_fs(payload)
        if source_type == "CompoundSource":
            return CompoundSource.from_fs(payload)
        if source_type == "RuptureSource":
            return RuptureSource.from_fs(payload)
        raise ValueError(f"Unsupported legacy source type {source_type!r}")


@dataclass
class RuptureSource(Source):
    """Deprecated legacy SRF source retained for input compatibility."""

    srf_file: str
    name: str = "rupture"

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "RuptureSource":
        return cls(**copy.deepcopy(dict(data)))

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        return {
            "_type": "RuptureSource",
            "srf_file": self.srf_file,
            "name": self.name,
        }


@dataclass
class CompoundSource(Source):
    """Deprecated weighted-point source retained as an adapter input."""

    kind: str
    coordinates: Any = field(default_factory=list)
    direction: Any = field(default_factory=list)
    domain: Optional[int] = None
    name: str = "compound"

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "CompoundSource":
        payload = copy.deepcopy(dict(data))
        payload.pop("n_points", None)
        payload.pop("frame", None)
        if "coordinates" in payload:
            payload["coordinates"] = CoordinateValue.from_fs(payload["coordinates"])
        if "direction" in payload:
            payload["direction"] = Direction.from_fs(payload["direction"])
        return cls(**payload)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        return {
            "_type": "CompoundSource",
            "name": self.name,
            "kind": self.kind,
            "n_points": len(self.coordinates),
            "coordinates": coordinate_value_to_fs(self.coordinates),
            **(
                {"direction": direction_to_fs(self.direction)}
                if self.direction is not None
                else {}
            ),
            **({"domain": self.domain} if self.domain is not None else {}),
        }


@dataclass
class SourceGroup:
    """Deprecated logical-source view used by pre-v2 callers."""

    source: Any
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SourceGroup":
        return cls(source=Source.from_fs(copy.deepcopy(data.get("source", {}))))

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        source = self.source.to_fs(ctx)
        if isinstance(self.source, PointSource):
            source = {"_type": "PointSource", **source}
        return {"source": source}

    def _set_path(self, proj_path: Path, rel_path: Path) -> None:
        warn_deprecated_path_api(f"{self.__class__.__name__}._set_path")
        self._proj_path = Path(proj_path)
        self._rel_path = Path(rel_path)

    def get_coordinates(self) -> np.ndarray:
        """Return source coordinates as a two-dimensional array."""

        coords = self.source.coordinates
        if isinstance(coords, CoordinateValue):
            coords = coords.value
        if is_quantity(coords):
            coords = coords.magnitude
        values = np.asarray(coords, dtype=float)
        if values.ndim == 1:
            return values.reshape(1, -1)
        return values

    def coordinates(self) -> np.ndarray:
        """Compatibility alias for :meth:`get_coordinates`."""

        return self.get_coordinates()

    @property
    def _path(self) -> Path:
        warn_deprecated_path_api(f"{self.__class__.__name__}._path")
        return self._proj_path / self._rel_path
