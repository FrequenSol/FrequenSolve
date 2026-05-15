from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.units import value_and_units_to_fs
from frequensolve.util.fields import canonical_fields
from frequensolve.util.mixins import (
    ExtraFieldsMixin,
    TypeTaggedMixin,
    fs_serialize,
    merge_extra,
)

__all__ = [
    "Output",
    "JobOutputs",
    "ParaviewOutput",
    "TraceOutput",
    "WavefieldOutput",
]


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
    values = [str(part).lower() for part in _as_list(parts)]
    invalid = [part for part in values if part not in ParaviewOutput._PARTS]
    if invalid:
        allowed = ", ".join(sorted(ParaviewOutput._PARTS))
        raise ValueError(f"ParaviewOutput.parts must use only: {allowed}")
    return values


@dataclass(kw_only=True)
class Output(TypeTaggedMixin, ExtraFieldsMixin):
    """Base class for all outputs.

    Paths are relative to the job result directory.
    """

    name: str = ""
    path: Union[str, Path] = None
    extra: Dict = field(default_factory=dict)

    def to_fs(self, ctx=None) -> Dict:
        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path,
        }
        return merge_extra(payload, self.extra, self.__class__.__name__)

    @classmethod
    def from_fs(cls, data: Dict) -> "Output":
        output_types: Dict[str, Type[Output]] = {
            "TraceOutput": TraceOutput,
            "ParaviewOutput": ParaviewOutput,
            "WavefieldOutput": WavefieldOutput,
        }
        return cls.dispatch_from_fs(data, output_types)


@dataclass(kw_only=True)
class TraceOutput(Output):
    path: Optional[Union[str, Path]] = None

    def __init__(self, path: Union[str, Path] = "traces", **kwargs):
        self.path = _relative_output_path(path)
        self._init_extra(None, **kwargs)

    def to_fs(self, ctx=None) -> Dict:
        return merge_extra(
            {"_type": self.__class__.__name__, "path": self.path},
            self.extra,
            self.__class__.__name__,
        )

    @classmethod
    def from_fs(cls, data: Dict) -> "TraceOutput":
        data = copy.deepcopy(data)
        data.pop("_type", None)
        return cls(path=data.pop("path", "traces"), **data)


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
    sources: Optional[List[int]] = None
    upscale: int = 1
    show_pml: bool = True
    format: str = "vtu"
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

    _FORMATS = {"vtu", "xdmf"}
    _TARGETS = {"volume", "surface", "grid"}
    _EXECUTE_ON = {"adapt", "initial", "special", "solve", "final", "none"}
    _PARTS = {"real", "imag", "abs"}

    def __init__(
        self,
        name: str = "ParaView",
        path: Union[str, Path] = "ParaView",
        fields: Optional[Iterable[str]] = None,
        properties: Optional[Iterable[str]] = None,
        sources: Optional[Iterable[int]] = None,
        upscale: int = 1,
        show_pml: bool = True,
        format: str = "vtu",
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
        **kwargs,
    ):
        if sources is None:
            sources = [1]
        if upscale < 0:
            raise ValueError("ParaviewOutput upscale must be >= 0")
        writer_payload = copy.deepcopy(dict(writer or {}))
        if writer_payload:
            writer_format = _format_from_writer(writer_payload)
            if format != "vtu" and writer_format != format:
                raise ValueError("ParaviewOutput format conflicts with writer format")
            format = writer_format

        self.name = name
        self.path = _relative_output_path(path)
        self.fields = canonical_fields(fields) if fields is not None else None
        self.properties = list(properties) if properties is not None else None
        self.sources = [int(source_id) for source_id in sources]
        self.upscale = int(upscale)
        self.show_pml = bool(show_pml)
        self.format = _choice(format, self._FORMATS, "ParaviewOutput.format")
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
        self._writer_extra = {
            key: value
            for key, value in writer_payload.items()
            if key not in {"format", "encoding"}
        }
        self._init_extra(None, **kwargs)

    @classmethod
    def volume(cls, **kwargs) -> "ParaviewOutput":
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
        return cls(target="grid", grid=grid, **kwargs)

    def to_fs(self, ctx=None) -> Dict:
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
        if self.fields is not None:
            payload["fields"] = canonical_fields(self.fields)

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

        if self.parts is not None:
            payload["items"] = self._items_payload()
            payload.pop("fields", None)
            payload.pop("properties", None)
        elif "items" in self.extra:
            payload.pop("fields", None)
            payload.pop("properties", None)

        return merge_extra(payload, self.extra, "ParaviewOutput")

    def _writer_payload(self) -> Dict[str, Any]:
        if self.format == "xdmf":
            payload = {"format": "xdmf", "encoding": "hdf5"}
        else:
            payload = {"format": "vtu", "encoding": "appended"}
        payload.update(copy.deepcopy(self._writer_extra))
        return payload

    def _items_payload(self) -> List[Dict[str, Any]]:
        items = []
        if self.fields is not None:
            items.extend(
                {"kind": "field", "field": field, "parts": self.parts}
                for field in canonical_fields(self.fields)
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
            sources=data.pop("sources", None),
            upscale=data.pop("upscale", 1),
            show_pml=data.pop("show_pml", True),
            format=_format_from_writer(writer),
            execute_on=data.pop("execute_on", None),
            order=data.pop("order", None),
            target=target,
            coordinates=coordinates,
            writer=writer,
            **({"source": source} if source is not None else {}),
            **({"items": items} if items is not None else {}),
            **data,
        )


def _format_from_writer(writer: Optional[Mapping[str, Any]]) -> str:
    if not writer:
        return "vtu"
    if writer.get("format") == "xdmf":
        return "xdmf"
    return "vtu"


@dataclass(kw_only=True)
class WavefieldOutput(Output):
    """Wavefield output request."""

    name: str = "wavefield"
    path: Union[str, Path] = "wavefields"
    fields: Optional[List[str]] = None
    grid: Optional[CartesianGrid] = None

    def __init__(
        self,
        name: str = "wavefield",
        path: Union[str, Path] = "wavefields",
        fields: Optional[List[str]] = None,
        grid: Optional[CartesianGrid] = None,
        **kwargs,
    ):
        self.name = name
        self.path = _relative_output_path(path)
        self.fields = canonical_fields(fields) if fields is not None else None
        self.grid = grid
        self._init_extra(None, **kwargs)

    def to_fs(self, ctx=None) -> Dict:
        fields = (
            canonical_fields(self.fields) if self.fields is not None else ["primary"]
        )
        grid = self.grid if self.grid is not None else CartesianGrid(x0=[], x1=[], n=[])

        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path,
            "fields": fields,
            "grid": grid.to_fs(ctx),
        }
        return merge_extra(payload, self.extra, "WavefieldOutput")

    @classmethod
    def from_fs(cls, data: Dict) -> "WavefieldOutput":
        data = copy.deepcopy(data)
        data.pop("_type", None)
        grid = data.pop("grid", None)
        return cls(
            name=data.pop("name", "wavefield"),
            path=data.pop("path", "wavefields"),
            fields=data.pop("fields", None),
            grid=CartesianGrid.from_fs(grid) if grid is not None else None,
            **data,
        )


@dataclass
class JobOutputs:
    """Output requests owned by one simulation job.

    Trace output is always present. Adding a ``TraceOutput`` only changes its
    configuration, such as the output directory.
    """

    traces: TraceOutput = field(default_factory=TraceOutput)
    paraview: List[ParaviewOutput] = field(default_factory=list)
    wavefields: List[WavefieldOutput] = field(default_factory=list)

    def __init__(
        self,
        outputs: Any = None,
        *,
        traces: Optional[TraceOutput] = None,
        paraview: Optional[Iterable[ParaviewOutput]] = None,
        wavefields: Optional[Iterable[WavefieldOutput]] = None,
    ):
        self.traces = traces or TraceOutput()
        self.paraview = list(paraview or [])
        self.wavefields = list(wavefields or [])
        if outputs is not None:
            self.add(outputs)

    def __iadd__(self, output: Any) -> "JobOutputs":
        return self.add(output)

    def add(self, output: Any) -> "JobOutputs":
        if output is None:
            return self
        if isinstance(output, JobOutputs):
            self.traces = output.traces
            self.paraview.extend(output.paraview)
            self.wavefields.extend(output.wavefields)
            return self
        if isinstance(output, Mapping):
            if any(
                key in output
                for key in ("traces", "receivers", "ParaView", "wavefields")
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
        payload = {
            "traces": self.traces.to_fs(ctx),
            "ParaView": [pv_out.to_fs(ctx) for pv_out in self.paraview],
            "wavefields": [wf_out.to_fs(ctx) for wf_out in self.wavefields],
        }
        return {key: value for key, value in payload.items() if value}

    @classmethod
    def from_fs(cls, data: Optional[Dict]) -> "JobOutputs":
        data = copy.deepcopy(data or {})
        traces = data.get("traces") or data.get("receivers")
        paraview = [
            ParaviewOutput.from_fs(pv_out) for pv_out in data.get("ParaView", [])
        ]
        wavefields = [
            WavefieldOutput.from_fs(wf_out) for wf_out in data.get("wavefields", [])
        ]
        return cls(
            traces=TraceOutput.from_fs(traces) if traces is not None else TraceOutput(),
            paraview=paraview,
            wavefields=wavefields,
        )
