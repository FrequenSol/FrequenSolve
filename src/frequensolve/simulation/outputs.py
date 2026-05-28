"""Output request objects for traces, ParaView files, wavefields, and units."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
import xarray as xr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.seismic.receivers import (
    ReceiverComponent,
    ReceiverDevice,
)
from frequensolve.units import is_quantity, unit_expression, value_and_units_to_fs
from frequensolve.util.fields import canonical_fields
from frequensolve.util.mixins import (
    ExtraFieldsMixin,
    TypeTaggedMixin,
    fs_serialize,
    merge_extra,
)

__all__ = [
    "Output",
    "OutputUnits",
    "JobOutputs",
    "ParaViewOutput",
    "ParaViewItem",
    "ParaviewOutput",
    "TraceOutput",
    "WavefieldOutput",
    "field",
    "info",
    "output_property",
    "outputs",
    "paraview",
    "wavefield",
]

_OUTPUT_DIMENSIONS = {
    "length",
    "time",
    "mass",
    "frequency",
    "velocity",
    "density",
    "pressure",
    "stress",
    "strain",
    "force",
    "moment",
    "attenuation",
    "wavenumber",
    "conductivity",
    "permittivity",
    "permeability",
    "efield",
    "bfield",
}


def _relative_output_path(path: Union[str, Path], field: str = "path") -> str:
    value = Path(path)
    if value.is_absolute():
        raise ValueError(f"{field} must be relative to the job result directory")
    return str(value)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _drop_none(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _choice(value: Optional[str], choices: set[str], field: str) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).lower()
    if normalized not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{field} must be one of: {allowed}")
    return normalized


def _normalize_parts(parts: Optional[Union[str, Iterable[str]]]) -> Optional[List[str]]:
    if parts is None:
        return None
    aliases = {
        "re": "real",
        "real": "real",
        "im": "imag",
        "imag": "imag",
        "imaginary": "imag",
        "abs": "abs",
        "mag": "abs",
        "magnitude": "abs",
    }
    raw_values = [str(part).lower() for part in _as_list(parts)]
    invalid = [part for part in raw_values if part not in aliases]
    if invalid:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"ParaviewOutput.parts must use only: {allowed}")
    return [aliases[part] for part in raw_values]


def _as_output_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (Output, Mapping)):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _canonical_field_list(fields: Iterable[str]) -> List[str]:
    return canonical_fields(_as_list(fields))


@dataclass(kw_only=True)
class Output(TypeTaggedMixin, ExtraFieldsMixin):
    """Base class for all outputs.

    Paths are relative to the job result directory.
    """

    name: str = ""
    path: Union[str, Path] = None
    extra: Dict = field(default_factory=dict)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this output request to the solver job payload."""

        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path,
        }
        return merge_extra(payload, self.extra, self.__class__.__name__)

    @classmethod
    def from_fs(cls, data: Dict) -> "Output":
        """Deserialize a concrete output request from a solver payload."""

        output_types: Dict[str, Type[Output]] = {
            "TraceOutput": TraceOutput,
            "ParaViewOutput": ParaviewOutput,
            "ParaviewOutput": ParaviewOutput,
            "WavefieldOutput": WavefieldOutput,
        }
        return cls.dispatch_from_fs(data, output_types)


@dataclass(kw_only=True)
class TraceOutput(Output):
    """Trace output request for receiver traces.

    Every job has a trace output. Supplying this object mainly customizes the
    result-directory path used for trace files.
    """

    path: Optional[Union[str, Path]] = None

    def __init__(self, path: Union[str, Path] = "traces", **kwargs):
        """Create a trace output request with a job-relative path."""

        self.path = _relative_output_path(path)
        self._init_extra(None, **kwargs)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this trace output request."""

        return merge_extra(
            {"_type": self.__class__.__name__, "path": self.path},
            self.extra,
            self.__class__.__name__,
        )

    @classmethod
    def from_fs(cls, data: Dict) -> "TraceOutput":
        """Deserialize a trace output request."""

        data = copy.deepcopy(data)
        data.pop("_type", None)
        return cls(path=data.pop("path", "traces"), **data)


@dataclass(kw_only=True)
class OutputUnits(ExtraFieldsMixin):
    """Default units for solver-produced output products."""

    geometry: Optional[str] = None
    dimensions: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        geometry: Optional[str] = None,
        *,
        dimensions: Optional[Mapping[str, str]] = None,
        defaults: Optional[Mapping[str, str]] = None,
        fields: Optional[Mapping[str, str]] = None,
        properties: Optional[Mapping[str, str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **dimension_defaults,
    ):
        """Create output-unit defaults.

        Args:
            geometry: Default coordinate/geometry units.
            dimensions: Units keyed by physical dimension, such as ``length``.
            defaults: Legacy alias for ``dimensions``.
            fields: Units keyed by output field name.
            properties: Units keyed by material-property name.
        """

        self.geometry = unit_expression(geometry) if geometry is not None else None
        merged_dimensions: Dict[str, str] = {}
        for source in (defaults, dimensions, dimension_defaults):
            for key, value in dict(source or {}).items():
                if value is not None:
                    merged_dimensions[str(key)] = unit_expression(value)
        self.dimensions = merged_dimensions
        self.fields = {
            str(key): unit_expression(value)
            for key, value in dict(fields or {}).items()
            if value is not None
        }
        self.properties = {
            str(key): unit_expression(value)
            for key, value in dict(properties or {}).items()
            if value is not None
        }
        self._init_extra(extra)

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize unit defaults for the solver output block."""

        payload = _drop_none({"geometry": self.geometry})
        if self.dimensions:
            payload["dimensions"] = dict(self.dimensions)
        if self.fields:
            payload["fields"] = dict(self.fields)
        if self.properties:
            payload["properties"] = dict(self.properties)
        return merge_extra(payload, self.extra, "OutputUnits")

    @classmethod
    def from_fs(cls, data: Optional[Mapping[str, Any]]) -> "OutputUnits":
        """Deserialize output-unit defaults from a solver payload."""

        data = copy.deepcopy(dict(data or {}))
        dimension_defaults = {
            key: data.pop(key) for key in list(data) if key in _OUTPUT_DIMENSIONS
        }
        return cls(
            geometry=data.pop("geometry", None),
            dimensions=data.pop("dimensions", None),
            defaults=data.pop("defaults", None),
            fields=data.pop("fields", None),
            properties=data.pop("properties", None),
            extra=data,
            **dimension_defaults,
        )


@dataclass(kw_only=True)
class ParaViewItem(ExtraFieldsMixin):
    """One selected field, material property, or metadata value for ParaView."""

    kind: str
    value: str
    name: Optional[str] = None
    units: Optional[Union[str, Sequence[str]]] = None
    parts: Optional[List[str]] = None
    basis: Optional[Union[Sequence[str], Mapping[str, Any]]] = None
    direction: Optional[Union[str, Mapping[str, Any]]] = None
    source_components: Optional[List[str]] = None
    system: str = "global"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        kind: str,
        value: str,
        *,
        name: Optional[str] = None,
        units: Optional[Union[str, Sequence[str]]] = None,
        parts: Optional[Union[str, Iterable[str]]] = None,
        basis: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
        direction: Optional[Union[str, Mapping[str, Any]]] = None,
        components: Optional[Iterable[str]] = None,
        system: str = "global",
        **kwargs,
    ):
        """Create a structured ParaView output selector."""

        normalized = str(kind).lower()
        if normalized not in {"field", "property", "info"}:
            raise ValueError("ParaViewItem kind must be field, property, or info")
        self.kind = normalized
        self.value = str(value)
        self.name = name
        self.units = self._units(units)
        self.parts = _normalize_parts(parts)
        self.basis = copy.deepcopy(basis)
        self.direction = copy.deepcopy(direction)
        self.source_components = (
            [str(component) for component in components]
            if components is not None
            else None
        )
        self.system = str(system)
        self._init_extra(None, **kwargs)

    @staticmethod
    def _units(units: Optional[Union[str, Sequence[str]]]):
        if units is None:
            return None
        if isinstance(units, str):
            return unit_expression(units)
        return [unit_expression(unit) for unit in units]

    @property
    def _selector_key(self) -> str:
        if self.kind == "property":
            return "property"
        return self.kind

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize this ParaView selector."""

        payload: Dict[str, Any] = {
            "kind": self.kind,
            self._selector_key: self.value,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.units is not None:
            payload["units"] = self.units
        if self.parts is not None and self.kind == "field":
            payload["parts"] = self.parts
        if self.source_components is not None:
            payload["components"] = self.source_components
        if self.basis is not None:
            payload["basis"] = self._basis_payload()
        if self.direction is not None:
            payload["direction"] = self._direction_payload()
        return merge_extra(payload, self.extra, "ParaViewItem")

    def _basis_payload(self) -> Dict[str, Any]:
        if isinstance(self.basis, Mapping):
            payload = copy.deepcopy(dict(self.basis))
            payload.setdefault("system", self.system)
            payload.setdefault("type", "coordinate_basis")
            return payload
        return {
            "type": "coordinate_basis",
            "system": self.system,
            "components": [str(component) for component in self.basis],
        }

    def _direction_payload(self) -> Dict[str, Any]:
        if isinstance(self.direction, Mapping):
            payload = copy.deepcopy(dict(self.direction))
            payload.setdefault("system", self.system)
            return payload
        return {"system": self.system, "axis": str(self.direction)}

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ParaViewItem":
        """Deserialize a ParaView selector."""

        payload = copy.deepcopy(dict(data))
        kind = payload.pop("kind", None)
        if kind is None:
            if "field" in payload:
                kind = "field"
            elif "property" in payload:
                kind = "property"
            elif "info" in payload:
                kind = "info"
            else:
                raise ValueError(
                    "ParaView item must define kind, field, property, or info"
                )
        key = "property" if kind == "property" else kind
        value = payload.pop(key)
        return cls(
            kind,
            value,
            name=payload.pop("name", None),
            units=payload.pop("units", None),
            parts=payload.pop("parts", None),
            basis=payload.pop("basis", payload.pop("frame", None)),
            direction=payload.pop("direction", None),
            components=payload.pop("components", None),
            **payload,
        )


@dataclass(kw_only=True)
class ParaviewOutput(Output):
    """ParaView output request.

    The public API intentionally exposes the common cases only: solution fields
    and material properties on the volume, selected surfaces, or a sampling
    grid. Advanced fast solver fields can still be passed through ``extra`` or
    loaded from existing solver JSON.
    """

    name: str = "ParaView"
    path: Union[str, Path] = "ParaView"
    fields: Optional[List[str]] = None
    properties: Optional[List[str]] = None
    items: Optional[List[Any]] = None
    sources: Optional[List[int]] = None
    upscale: int = 1
    show_pml: bool = True
    format: str = "vtu"
    encoding: Optional[str] = None
    execute_on: Optional[str] = None
    order: Optional[int] = None
    parts: Optional[List[str]] = None
    target: Optional[Union[str, Mapping[str, Any]]] = None
    grid_spec: Optional[Any] = None
    surfaces: Optional[List[Union[str, int]]] = None
    boundaries: Optional[List[str]] = None
    shell: bool = False
    plane: Optional[Mapping[str, Any]] = None
    coordinates: Optional[str] = None
    target_coordinates: Optional[str] = None
    writer: Optional[Mapping[str, Any]] = None
    source: Optional[Mapping[str, Any]] = None

    _FORMATS = {"vtu", "xdmf", "xmf", "vtr"}
    _TARGETS = {"volume", "surface", "grid"}
    _EXECUTE_ON = {"adapt", "initial", "special", "solve", "final", "none"}
    _PARTS = {"re", "real", "im", "imag", "imaginary", "abs", "mag", "magnitude"}

    def __init__(
        self,
        name: str = "ParaView",
        path: Union[str, Path] = "ParaView",
        fields: Optional[Iterable[str]] = None,
        properties: Optional[Iterable[str]] = None,
        items: Optional[Iterable[Union[ParaViewItem, Mapping[str, Any]]]] = None,
        sources: Optional[Iterable[int]] = None,
        upscale: int = 1,
        show_pml: bool = True,
        format: str = "vtu",
        encoding: Optional[str] = None,
        execute_on: Optional[str] = None,
        order: Optional[int] = None,
        parts: Optional[Union[str, Iterable[str]]] = None,
        target: Optional[Union[str, Mapping[str, Any]]] = None,
        grid: Optional[Any] = None,
        surfaces: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
        boundaries: Optional[Union[str, Iterable[str]]] = None,
        shell: bool = False,
        plane: Optional[Mapping[str, Any]] = None,
        planes: Optional[Iterable[Mapping[str, Any]]] = None,
        coordinates: Optional[str] = None,
        target_coordinates: Optional[str] = None,
        writer: Optional[Mapping[str, Any]] = None,
        source: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        """Create a ParaView output request.

        Use ``fields`` and ``properties`` for common volume output, ``items``
        when you need per-item units/parts/basis metadata, and class helpers
        such as :meth:`surface` or :meth:`grid` for non-volume targets.
        """

        if sources is None:
            sources = [1]
        if upscale < 0:
            raise ValueError("ParaviewOutput upscale must be >= 0")
        writer_payload = copy.deepcopy(dict(writer or {}))
        if writer_payload:
            requested_format = str(format).lower()
            if requested_format == "xmf":
                requested_format = "xdmf"
            writer_format = _format_from_writer(writer_payload)
            if requested_format != "vtu" and writer_format != requested_format:
                raise ValueError("ParaviewOutput format conflicts with writer format")
            format = writer_format
            if (
                encoding is not None
                and writer_payload.get("encoding") is not None
                and writer_payload.get("encoding") != encoding
            ):
                raise ValueError(
                    "ParaviewOutput encoding conflicts with writer encoding"
                )
            encoding = writer_payload.get("encoding", encoding)
        if items is not None and (
            fields is not None or properties is not None or parts is not None
        ):
            raise ValueError("Pass either items or fields/properties/parts, not both")

        self.name = name
        self.path = _relative_output_path(path)
        self.fields = _canonical_field_list(fields) if fields is not None else None
        self.properties = [str(prop) for prop in _as_list(properties)] or None
        self.items = _paraview_items(items)
        self.sources = [int(source_id) for source_id in sources]
        self.upscale = int(upscale)
        self.show_pml = bool(show_pml)
        self.format = _choice(format, self._FORMATS, "ParaviewOutput.format")
        if self.format == "xmf":
            self.format = "xdmf"
        self.encoding = str(encoding).lower() if encoding is not None else None
        self.execute_on = _choice(
            execute_on, self._EXECUTE_ON, "ParaviewOutput.execute_on"
        )
        self.order = int(order) if order is not None else None
        self.parts = _normalize_parts(parts)
        self.target = copy.deepcopy(target)
        self.grid_spec = copy.deepcopy(grid)
        self.surfaces = copy.deepcopy(_as_list(surfaces)) or None
        self.boundaries = [str(label) for label in _as_list(boundaries)] or None
        self.shell = bool(shell)
        plane_list = [copy.deepcopy(p) for p in _as_list(planes)]
        if plane is not None:
            plane_list.insert(0, copy.deepcopy(plane))
        self.plane = plane_list[0] if len(plane_list) == 1 else None
        self._planes = plane_list
        self.coordinates = coordinates
        self.target_coordinates = target_coordinates
        self.writer = writer_payload or None
        self.source = copy.deepcopy(dict(source)) if source is not None else None
        self._writer_extra = {
            key: value
            for key, value in writer_payload.items()
            if key not in {"format", "encoding"}
        }
        self._init_extra(None, **kwargs)

    @classmethod
    def volume(cls, **kwargs) -> "ParaviewOutput":
        """Create a volume-targeted ParaView output request."""

        return cls(target="volume", **kwargs)

    @classmethod
    def surface(
        cls,
        surfaces: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
        *,
        boundaries: Optional[Union[str, Iterable[str]]] = None,
        shell: bool = False,
        plane: Optional[Mapping[str, Any]] = None,
        planes: Optional[Iterable[Mapping[str, Any]]] = None,
        **kwargs,
    ) -> "ParaviewOutput":
        """Create a surface-targeted ParaView output request."""

        return cls(
            target="surface",
            surfaces=surfaces,
            boundaries=boundaries,
            shell=shell,
            plane=plane,
            planes=planes,
            **kwargs,
        )

    @classmethod
    def grid(cls, grid: Any, **kwargs) -> "ParaviewOutput":
        """Create a grid-targeted ParaView output request."""

        return cls(target="grid", grid=grid, **kwargs)

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this ParaView output request."""

        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path,
            "properties": self.properties,
            "sources": self.sources,
            "upscale": self.upscale,
            "show_pml": self.show_pml,
            "writer": self._writer_payload(),
        }
        extra = copy.deepcopy(self.extra)
        if self.fields is not None:
            payload["fields"] = _canonical_field_list(self.fields)
        if self.source is not None:
            payload["source"] = copy.deepcopy(self.source)

        for key in ["execute_on", "order"]:
            value = getattr(self, key)
            if value is not None:
                payload[key] = value

        if self.coordinates is not None:
            payload["coordinates"] = {"system": self.coordinates}

        target_payload = self._target_payload(ctx)
        if target_payload is not None and (
            target_payload != {"kind": "volume"} or self.target is not None
        ):
            payload["target"] = target_payload

        if self.items is not None:
            payload["items"] = [fs_serialize(item, ctx) for item in self.items]
            payload.pop("fields", None)
            payload.pop("properties", None)
        elif self.parts is not None:
            payload["items"] = self._items_payload()
            payload.pop("fields", None)
            payload.pop("properties", None)
        elif "items" in extra:
            payload["items"] = [
                fs_serialize(item, ctx) for item in _as_list(extra.pop("items"))
            ]
            payload.pop("fields", None)
            payload.pop("properties", None)

        return merge_extra(payload, extra, "ParaviewOutput")

    def _writer_payload(self) -> Dict[str, Any]:
        if self.format in {"xdmf", "xmf"}:
            payload = {"format": "xdmf", "encoding": self.encoding or "hdf5"}
        elif self.format == "vtr":
            payload = {"format": "vtr", "encoding": self.encoding or "appended"}
        else:
            payload = {"format": "vtu", "encoding": self.encoding or "appended"}
        payload.update(copy.deepcopy(self._writer_extra))
        return payload

    def _items_payload(self) -> List[Dict[str, Any]]:
        items = []
        if self.fields is not None:
            items.extend(
                {"kind": "field", "field": field, "parts": self.parts}
                for field in _canonical_field_list(self.fields)
            )
        items.extend(
            {"kind": "property", "property": prop} for prop in (self.properties or [])
        )
        return items

    def _target_payload(self, ctx=None) -> Optional[Dict[str, Any]]:
        if isinstance(self.target, Mapping):
            return copy.deepcopy(dict(self.target))

        target = self._inferred_target()
        payload: Dict[str, Any] = {"kind": target}

        if self.target_coordinates is not None:
            payload["coordinates"] = {"system": self.target_coordinates}
        mesh = _drop_none(
            {
                "order": self.order,
                "upscale": self.upscale,
                "show_pml": self.show_pml,
            }
        )
        if target == "surface" and mesh:
            payload["mesh"] = mesh
        if target == "grid":
            if self.grid_spec is None:
                raise ValueError("grid is required for ParaView grid output")
            payload["grid"] = fs_serialize(self.grid_spec, ctx)
        if target == "surface":
            selections = self._surface_selections()
            if selections:
                payload["selection"] = selections
        return payload

    def _inferred_target(self) -> str:
        if self.target is not None:
            return _choice(str(self.target), self._TARGETS, "ParaviewOutput.target")
        if self.grid_spec is not None:
            return "grid"
        if self.shell or self.surfaces or self.boundaries or self._planes:
            return "surface"
        return "volume"

    def _surface_selections(self) -> List[Dict[str, Any]]:
        selections: List[Dict[str, Any]] = []
        if self.shell:
            selections.append({"kind": "shell"})
        if self.boundaries:
            selections.append({"kind": "boundary", "labels": self.boundaries})
        for surface in self.surfaces or []:
            if isinstance(surface, int):
                selections.append({"kind": "model_surface", "index": surface})
            else:
                selections.append({"kind": "model_surface", "name": str(surface)})
        for plane in self._planes:
            selections.append(self._plane_selection(plane))
        return selections

    def _plane_selection(self, plane: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {
            "kind": "plane",
            "system": plane.get("system", "global"),
            "axis": plane["axis"],
            "value": value_and_units_to_fs(plane["value"], plane.get("units")),
        }
        if "tolerance" in plane:
            payload["tolerance"] = value_and_units_to_fs(
                plane["tolerance"], plane.get("tolerance_units")
            )
        return payload

    @classmethod
    def from_fs(cls, data: Dict) -> "ParaviewOutput":
        """Deserialize a ParaView output request."""

        data = copy.deepcopy(data)
        data.pop("_type", None)
        writer = data.pop("writer", None)
        target = data.pop("target", None)
        source = data.pop("source", None)
        items = data.pop("items", None)
        coordinates = data.pop("coordinates", None)
        if isinstance(coordinates, Mapping):
            coordinates = coordinates.get("system")
        if data.get("path") is not None:
            data["path"] = Path(data["path"])
        return cls(
            name=data.pop("name", "ParaView"),
            path=data.pop("path", "ParaView"),
            fields=data.pop("fields", None),
            properties=data.pop("properties", None),
            items=items,
            sources=data.pop("sources", None),
            upscale=data.pop("upscale", 1),
            show_pml=data.pop("show_pml", True),
            format=_format_from_writer(writer),
            encoding=(writer or {}).get("encoding") if writer is not None else None,
            execute_on=data.pop("execute_on", None),
            order=data.pop("order", None),
            target=target,
            coordinates=coordinates,
            writer=writer,
            source=source,
            **data,
        )


def _format_from_writer(writer: Optional[Mapping[str, Any]]) -> str:
    if not writer:
        return "vtu"
    format_name = str(writer.get("format", "vtu")).lower()
    if format_name in {"xdmf", "xmf"}:
        return "xdmf"
    if format_name == "vtr":
        return "vtr"
    return "vtu"


def _paraview_items(
    items: Optional[Iterable[Union[ParaViewItem, Mapping[str, Any]]]],
) -> Optional[List[Union[ParaViewItem, Mapping[str, Any]]]]:
    if items is None:
        return None
    return [
        item if isinstance(item, ParaViewItem) else ParaViewItem.from_fs(item)
        for item in _as_list(items)
    ]


@dataclass(kw_only=True)
class WavefieldOutput(Output):
    """Grid-backed wavefield output request.

    The grid is described with xarray-style dimensions and coordinates. Pass an
    ``xarray.DataArray``/``Dataset`` directly, or pass ``dims`` and ``coords``.
    Coordinate arrays may be nonuniform, but each coordinate must be 1D and
    strictly monotonic.
    """

    name: str = "wavefield"
    path: Union[str, Path] = "wavefields"
    fields: Optional[List[str]] = None
    device: Optional[ReceiverDevice] = None
    grid: Optional[Dict[str, Any]] = None
    sources: Optional[List[int]] = None

    def __init__(
        self,
        name: str = "wavefield",
        path: Union[str, Path] = "wavefields",
        fields: Optional[Union[str, Iterable[str]]] = None,
        field: Optional[str] = None,
        device: Optional[Union[ReceiverDevice, Mapping[str, Any]]] = None,
        grid: Optional[
            Union[CartesianGrid, xr.DataArray, xr.Dataset, Mapping[str, Any]]
        ] = None,
        dims: Optional[Iterable[str]] = None,
        coords: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
        units: Optional[str] = None,
        system: Optional[str] = None,
        sources: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        """Create a grid-backed wavefield output request.

        Provide either a receiver-like ``device`` or one or more field names,
        plus grid coordinates through ``grid`` or ``dims``/``coords``.
        """

        if field is not None and fields is not None:
            raise ValueError("Pass only one of field or fields")
        if device is not None and (field is not None or fields is not None):
            raise ValueError("Pass either device or field/fields, not both")
        self.name = name
        self.path = _relative_output_path(path)
        self._single_field_key = field is not None
        self.device = _wavefield_device(device)
        if self.device is not None:
            self.fields = [component.field for component in self.device.components]
        else:
            field_value = field if field is not None else fields
            self.fields = (
                canonical_fields(_as_list(field_value)) if field_value else None
            )
        self.grid = _wavefield_grid_payload(
            grid=grid,
            dims=dims,
            coords=coords,
            units=units,
            system=system,
        )
        self.sources = (
            [int(source) for source in sources] if sources is not None else None
        )
        self._init_extra(None, **kwargs)

    @property
    def component_names(self) -> List[str]:
        """Names of the receiver components represented by this wavefield."""

        return [component.name for component in self.components]

    @property
    def components(self) -> List[ReceiverComponent]:
        """Receiver components used to sample the requested wavefield fields."""

        if self.device is not None:
            return list(self.device.components)
        fields = self.fields if self.fields is not None else ["primary"]
        return [
            ReceiverComponent(name=str(field), field=str(field))
            for field in canonical_fields(fields)
        ]

    def component_payloads(self, ctx=None) -> List[Dict[str, Any]]:
        """Serialized component payloads for this wavefield request."""

        return [component.to_fs(ctx) for component in self.components]

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this wavefield output request."""

        if self.grid is None:
            raise ValueError("WavefieldOutput requires a grid")

        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path,
            "grid": copy.deepcopy(self.grid),
        }
        if self.device is not None:
            payload["device"] = self.device.to_fs(ctx)
        else:
            fields = (
                canonical_fields(self.fields)
                if self.fields is not None
                else ["primary"]
            )
            if self._single_field_key and len(fields) == 1:
                payload["field"] = fields[0]
            else:
                payload["fields"] = fields
        if self.sources is not None:
            payload["sources"] = self.sources
        return merge_extra(payload, self.extra, "WavefieldOutput")

    @classmethod
    def from_fs(cls, data: Dict) -> "WavefieldOutput":
        """Deserialize a wavefield output request."""

        data = copy.deepcopy(data)
        data.pop("_type", None)
        grid = data.pop("grid", None)
        device = data.pop("device", None)
        field = data.pop("field", None)
        fields = data.pop("fields", None)
        return cls(
            name=data.pop("name", "wavefield"),
            path=data.pop("path", "wavefields"),
            field=field,
            fields=None if device is not None else fields,
            device=device,
            grid=grid,
            sources=data.pop("sources", None),
            **data,
        )


def _wavefield_device(
    device: Optional[Union[ReceiverDevice, Mapping[str, Any]]],
) -> Optional[ReceiverDevice]:
    if device is None:
        return None
    if isinstance(device, ReceiverDevice):
        resolved = copy.deepcopy(device)
    elif isinstance(device, Mapping):
        payload = copy.deepcopy(dict(device))
        if "_type" not in payload:
            payload["_type"] = "ReceiverNode"
        resolved = ReceiverDevice.from_fs(payload)
    else:
        raise TypeError("device must be a ReceiverDevice or mapping")
    if not resolved.components:
        raise ValueError("WavefieldOutput device requires at least one component")
    return resolved


def _wavefield_grid_payload(
    *,
    grid: Optional[Union[CartesianGrid, xr.DataArray, xr.Dataset, Mapping[str, Any]]],
    dims: Optional[Iterable[str]],
    coords: Optional[Union[Mapping[str, Any], Sequence[Any]]],
    units: Optional[str],
    system: Optional[str],
) -> Optional[Dict[str, Any]]:
    if grid is not None and (coords is not None or dims is not None):
        raise ValueError("Pass either grid or xarray-style dims/coords, not both")
    if grid is None and coords is None:
        return None

    if isinstance(grid, CartesianGrid):
        return _xarray_grid_payload_from_dataarray(
            grid.as_xarray(), units=units or grid.units, system=system or grid.system
        )
    if isinstance(grid, (xr.DataArray, xr.Dataset)):
        return _xarray_grid_payload_from_dataarray(grid, units=units, system=system)
    if isinstance(grid, Mapping):
        payload = dict(grid)
        if "dims" in payload and "coords" in payload:
            return _normalize_xarray_grid_payload(payload, units=units, system=system)
        if payload.get("_type") == "CartesianGrid" or {"n", "x0", "x1"} <= set(payload):
            cart = CartesianGrid.from_fs(payload)
            return _xarray_grid_payload_from_dataarray(
                cart.as_xarray(),
                units=units or cart.units,
                system=system or cart.system,
            )
        raise ValueError("WavefieldOutput grid mapping requires dims/coords")

    if grid is not None:
        raise TypeError(
            "grid must be an xarray DataArray/Dataset, CartesianGrid, or mapping"
        )

    dims_list = _xarray_grid_dims(dims=dims, coords=coords)
    return _xarray_grid_payload_from_coords(
        dims=dims_list,
        coords=coords,
        units=units,
        system=system,
    )


def _xarray_grid_dims(
    *,
    dims: Optional[Iterable[str]],
    coords: Optional[Union[Mapping[str, Any], Sequence[Any]]],
) -> List[str]:
    if dims is not None:
        return [str(dim) for dim in dims]
    if isinstance(coords, Mapping):
        return [str(dim) for dim in coords]
    raise ValueError("WavefieldOutput dims are required when coords is not a mapping")


def _xarray_grid_payload_from_coords(
    *,
    dims: Sequence[str],
    coords: Union[Mapping[str, Any], Sequence[Any]],
    units: Optional[str],
    system: Optional[str],
) -> Dict[str, Any]:
    if isinstance(coords, Mapping):
        coord_map = {str(dim): coords[str(dim)] for dim in dims}
    else:
        coord_values = list(coords)
        if len(coord_values) != len(dims):
            raise ValueError("coords length must match dims length")
        coord_map = dict(zip(dims, coord_values))

    payload: Dict[str, Any] = {
        "_type": "XArrayGrid",
        "dims": list(dims),
        "coords": {},
    }
    coord_units = []
    for dim in dims:
        coord_payload = _coord_payload(dim, coord_map[dim], units=units)
        payload["coords"][dim] = coord_payload
        if coord_payload.get("units"):
            coord_units.append(coord_payload["units"])
    if units is not None:
        payload["units"] = unit_expression(units)
    elif coord_units and all(unit == coord_units[0] for unit in coord_units):
        payload["units"] = coord_units[0]
    if system is not None:
        payload["system"] = system
    return payload


def _xarray_grid_payload_from_dataarray(
    grid: Union[xr.DataArray, xr.Dataset],
    *,
    units: Optional[str],
    system: Optional[str],
) -> Dict[str, Any]:
    dims = list(grid.sizes)
    coords = {
        dim: grid.coords[dim] if dim in grid.coords else np.arange(grid.sizes[dim])
        for dim in dims
    }
    resolved_units = units or grid.attrs.get("units")
    resolved_system = (
        system or grid.attrs.get("system") or grid.attrs.get("coordinate_system")
    )
    return _xarray_grid_payload_from_coords(
        dims=dims,
        coords=coords,
        units=resolved_units,
        system=resolved_system,
    )


def _normalize_xarray_grid_payload(
    payload: Mapping[str, Any],
    *,
    units: Optional[str],
    system: Optional[str],
) -> Dict[str, Any]:
    dims = [str(dim) for dim in payload["dims"]]
    coord_payloads = dict(payload["coords"])
    coords = {}
    for dim in dims:
        if dim not in coord_payloads:
            raise ValueError(f"WavefieldOutput grid is missing coordinate {dim!r}")
        coords[dim] = coord_payloads[dim]
    normalized = _xarray_grid_payload_from_coords(
        dims=dims,
        coords=coords,
        units=units,
        system=system or payload.get("system"),
    )
    if units is None and payload.get("units") is not None:
        default_units = unit_expression(payload["units"])
        normalized["units"] = default_units
        for coord_payload in normalized["coords"].values():
            coord_payload.setdefault("units", default_units)
    normalized["_type"] = payload.get("_type", "XArrayGrid")
    return normalized


def _coord_payload(dim: str, coord: Any, *, units: Optional[str]) -> Dict[str, Any]:
    coord_units = units
    if isinstance(coord, Mapping):
        if coord_units is None and coord.get("units") is not None:
            coord_units = coord["units"]
        coord = _coord_data(coord, field=f"WavefieldOutput coordinate {dim!r}")
    if isinstance(coord, xr.DataArray):
        if coord_units is None:
            coord_units = coord.attrs.get("units")
        coord = coord.values
    if is_quantity(coord):
        if coord_units is None:
            coord_units = coord.units
        coord = coord.magnitude
    values = np.asarray(coord, dtype=float).ravel()
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"WavefieldOutput coordinate {dim!r} must be a 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"WavefieldOutput coordinate {dim!r} must be finite")
    if values.size > 1:
        diffs = np.diff(values)
        if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
            raise ValueError(
                f"WavefieldOutput coordinate {dim!r} must be strictly monotonic"
            )
    payload = {"data": values.tolist()}
    if coord_units is not None:
        payload["units"] = unit_expression(coord_units)
    return payload


def _coord_data(coord: Any, *, field: str) -> Any:
    if isinstance(coord, Mapping):
        if "data" in coord:
            return coord["data"]
        if "values" in coord:
            return coord["values"]
        if "value" in coord:
            return coord["value"]
        raise ValueError(f"{field} requires data")
    return coord


@dataclass
class JobOutputs:
    """Output requests owned by one simulation job.

    Trace output is always present. Adding a ``TraceOutput`` only changes its
    configuration, such as the output directory.
    """

    traces: TraceOutput = field(default_factory=TraceOutput)
    paraview: List[ParaviewOutput] = field(default_factory=list)
    wavefields: List[WavefieldOutput] = field(default_factory=list)
    units: Optional[OutputUnits] = None

    def __init__(
        self,
        outputs: Any = None,
        *,
        traces: Optional[TraceOutput] = None,
        paraview: Optional[Iterable[ParaviewOutput]] = None,
        wavefields: Optional[Iterable[WavefieldOutput]] = None,
        units: Optional[Union[OutputUnits, Mapping[str, Any]]] = None,
    ):
        """Create a collection of job output requests.

        The collection always contains trace output and may also include any
        number of ParaView and wavefield outputs.
        """

        self.traces = traces or TraceOutput()
        self.paraview = []
        self.wavefields = []
        self.units = _output_units(units)
        for item in _as_output_sequence(paraview):
            self.add(item)
        for item in _as_output_sequence(wavefields):
            self.add(item)
        if outputs is not None:
            self.add(outputs)

    def __iadd__(self, output: Any) -> "JobOutputs":
        return self.add(output)

    def add(self, output: Any) -> "JobOutputs":
        """Add or merge an output request and return ``self`` for chaining."""

        if output is None:
            return self
        if isinstance(output, JobOutputs):
            self.traces = output.traces
            self.paraview.extend(output.paraview)
            self.wavefields.extend(output.wavefields)
            self.units = output.units
            return self
        if isinstance(output, OutputUnits):
            self.units = output
            return self
        if isinstance(output, Mapping):
            if any(
                key in output
                for key in ("traces", "receivers", "ParaView", "wavefields", "Units")
            ):
                return self.add(JobOutputs.from_fs(dict(output)))
            return self.add(Output.from_fs(dict(output)))
        if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
            for item in output:
                self.add(item)
            return self
        if isinstance(output, TraceOutput):
            self.traces = output
        elif isinstance(output, ParaviewOutput):
            self.paraview.append(output)
        elif isinstance(output, WavefieldOutput):
            self.wavefields.append(output)
        else:
            raise TypeError(f"Unsupported output type: {type(output).__name__}")
        return self

    def to_fs(self, ctx=None) -> Dict:
        """Serialize the complete job output block."""

        payload = {
            "Units": self.units.to_fs(ctx) if self.units is not None else None,
            "traces": self.traces.to_fs(ctx),
            "ParaView": [pv_out.to_fs(ctx) for pv_out in self.paraview],
            "wavefields": [wf_out.to_fs(ctx) for wf_out in self.wavefields],
        }
        return {key: value for key, value in payload.items() if value}

    @classmethod
    def from_fs(cls, data: Optional[Dict]) -> "JobOutputs":
        """Deserialize a job output block."""

        data = copy.deepcopy(data or {})
        traces = data.get("traces") or data.get("receivers")
        paraview_data = data.get("ParaView", [])
        if isinstance(paraview_data, Mapping):
            paraview_data = [paraview_data]
        paraview = [ParaviewOutput.from_fs(pv_out) for pv_out in paraview_data]
        wavefields = [
            WavefieldOutput.from_fs(wf_out) for wf_out in data.get("wavefields", [])
        ]
        return cls(
            traces=TraceOutput.from_fs(traces) if traces is not None else TraceOutput(),
            paraview=paraview,
            wavefields=wavefields,
            units=OutputUnits.from_fs(data["Units"]) if "Units" in data else None,
        )


def _output_units(
    units: Optional[Union[OutputUnits, Mapping[str, Any]]],
) -> Optional[OutputUnits]:
    if units is None:
        return None
    if isinstance(units, OutputUnits):
        return units
    if isinstance(units, Mapping):
        return OutputUnits.from_fs(units)
    raise TypeError("units must be an OutputUnits or mapping")


def output_property(
    name: str,
    *,
    output_name: Optional[str] = None,
    units: Optional[Union[str, Sequence[str]]] = None,
    **kwargs,
) -> ParaViewItem:
    """Select a material property for ParaView output."""

    return ParaViewItem("property", name, name=output_name, units=units, **kwargs)


def info(
    name: str,
    *,
    output_name: Optional[str] = None,
    units: Optional[Union[str, Sequence[str]]] = None,
    **kwargs,
) -> ParaViewItem:
    """Select unitless mesh/domain metadata for ParaView output."""

    return ParaViewItem("info", name, name=output_name, units=units, **kwargs)


def field(
    name: str,
    *,
    output_name: Optional[str] = None,
    units: Optional[Union[str, Sequence[str]]] = None,
    parts: Optional[Union[str, Iterable[str]]] = None,
    basis: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    direction: Optional[Union[str, Mapping[str, Any]]] = None,
    components: Optional[Iterable[str]] = None,
    system: str = "global",
    **kwargs,
) -> ParaViewItem:
    """Select a solution, xarray, or wavefield field for ParaView output."""

    return ParaViewItem(
        "field",
        name,
        name=output_name,
        units=units,
        parts=parts,
        basis=basis,
        direction=direction,
        components=components,
        system=system,
        **kwargs,
    )


# Explicit imports from this module may use `prop`, while the top-level
# `frequensolve.prop` remains the material-property expression helper.
prop = output_property


class _ParaViewFactory:
    field = staticmethod(field)
    prop = staticmethod(output_property)
    property = staticmethod(output_property)
    info = staticmethod(info)

    def __call__(self, name: str = "paraview", **kwargs) -> ParaviewOutput:
        return ParaviewOutput(name=name, **kwargs)

    def volume(self, name: str = "paraview", **kwargs) -> ParaviewOutput:
        return ParaviewOutput.volume(name=name, **kwargs)

    def surface(self, name: str = "paraview", **kwargs) -> ParaviewOutput:
        return ParaviewOutput.surface(name=name, **kwargs)

    def grid(
        self, name: str = "paraview", grid: Any = None, **kwargs
    ) -> ParaviewOutput:
        if grid is None:
            raise ValueError("paraview.grid requires grid")
        return ParaviewOutput.grid(grid, name=name, **kwargs)


paraview = _ParaViewFactory()
ParaViewOutput = ParaviewOutput


def wavefield(
    fields: Optional[Union[str, Iterable[str]]] = None,
    *,
    name: Optional[str] = None,
    path: Union[str, Path] = "wavefields",
    field: Optional[str] = None,
    device: Optional[Union[ReceiverDevice, Mapping[str, Any]]] = None,
    grid: Optional[
        Union[CartesianGrid, xr.DataArray, xr.Dataset, Mapping[str, Any]]
    ] = None,
    dims: Optional[Iterable[str]] = None,
    coords: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    units: Optional[str] = None,
    system: Optional[str] = None,
    sources: Optional[Iterable[int]] = None,
    **kwargs,
) -> WavefieldOutput:
    """Create a concise grid-backed wavefield output request."""

    if field is not None and fields is not None:
        raise ValueError("Pass only one of the positional field(s) or field")
    requested = field if field is not None else fields
    if name is None:
        name = _wavefield_name(requested, device)
    if isinstance(requested, str):
        return WavefieldOutput(
            name=name,
            path=path,
            field=requested,
            device=device,
            grid=grid,
            dims=dims,
            coords=coords,
            units=units,
            system=system,
            sources=sources,
            **kwargs,
        )
    return WavefieldOutput(
        name=name,
        path=path,
        fields=requested,
        device=device,
        grid=grid,
        dims=dims,
        coords=coords,
        units=units,
        system=system,
        sources=sources,
        **kwargs,
    )


def _wavefield_name(
    fields: Optional[Union[str, Iterable[str]]],
    device: Optional[Union[ReceiverDevice, Mapping[str, Any]]],
) -> str:
    if isinstance(device, ReceiverDevice) and device.name:
        return str(device.name)
    if isinstance(device, Mapping) and device.get("name"):
        return str(device["name"])
    if isinstance(fields, str):
        return f"{canonical_fields([fields])[0].split(':')[-1]}_wavefield"
    return "wavefield"


def outputs(
    value: Any = None,
    *,
    traces: Optional[Union[str, Path, TraceOutput]] = "traces",
    paraview: Any = None,
    wavefields: Any = None,
    units: Optional[Union[OutputUnits, Mapping[str, Any]]] = None,
) -> JobOutputs:
    """Create a complete job output configuration."""

    if isinstance(value, JobOutputs):
        config = JobOutputs(value)
    elif value is not None:
        config = JobOutputs(value)
    else:
        trace_path = "traces" if traces is None else traces
        trace_output = (
            trace_path
            if isinstance(trace_path, TraceOutput)
            else TraceOutput(trace_path)
        )
        config = JobOutputs(traces=trace_output, units=units)
    if value is not None and units is not None:
        config.units = _output_units(units)
    if (
        traces is not None
        and traces != "traces"
        and not isinstance(value, JobOutputs)
        and value is not None
    ):
        config.traces = (
            traces if isinstance(traces, TraceOutput) else TraceOutput(traces)
        )
    for item in _as_output_sequence(paraview):
        config.add(item)
    for item in _as_output_sequence(wavefields):
        config.add(item)
    return config
