import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.util.fields import canonical_fields
from frequensolve.util.mixins import ExtraFieldsMixin, TypeTaggedMixin, merge_extra

__all__ = [
    "Output",
    "OutputManager",
    "ParaviewOutput",
    "TraceOutput",
    "WavefieldOutput",
]


def _relative_output_path(path: Union[str, Path], field: str = "path") -> str:
    value = Path(path)
    if value.is_absolute():
        raise ValueError(f"{field} must be relative to the job result directory")
    return str(value)


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
    """
    Represents the Paraview subsection in the Output section.

    Attributes:
       directory (str): The directory to store the Paraview output.
       fields (List[str]): The fields to include in the Paraview output.
       prefix (str): The prefix for the Paraview output files.
       upscale (int): The upscale factor for the Paraview output.
    """

    name: str = "ParaView"
    path: Union[str, Path] = ("ParaView",)
    fields: Optional[List[str]] = None
    properties: Optional[List[str]] = None
    sources: Optional[List[int]] = None
    upscale: int = 1
    show_pml: bool = True

    def __init__(
        self,
        name: str = "ParaView",
        path: Union[str, Path] = "ParaView",
        fields: Optional[List[str]] = None,
        properties: Optional[List[str]] = None,
        sources: Optional[List[int]] = None,
        upscale: int = 1,
        show_pml: bool = True,
        **kwargs,
    ):

        if sources is None:
            sources = [1]
        if upscale < 1:
            raise ValueError("ParaviewOutput upscale must be >= 1")

        self.name = name
        self.path = _relative_output_path(path)
        self.fields = canonical_fields(fields) if fields is not None else None
        self.properties = list(properties) if properties is not None else None
        self.sources = list(sources)
        self.upscale = int(upscale)
        self.show_pml = bool(show_pml)
        self._init_extra(None, **kwargs)

    def to_fs(self, ctx=None) -> Dict:
        fields = canonical_fields(self.fields) if self.fields is not None else ["all"]

        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path,
            "fields": fields,
            "properties": self.properties,
            "sources": self.sources,
            "upscale": self.upscale,
            "show_pml": self.show_pml,
        }
        return merge_extra(payload, self.extra, "ParaviewOutput")

    @classmethod
    def from_fs(cls, data: Dict) -> "ParaviewOutput":
        data = copy.deepcopy(data)
        data.pop("_type", None)
        if data.get("path") is not None:
            data["path"] = Path(data["path"])
        return cls(**data)


@dataclass(kw_only=True)
class WavefieldOutput(Output):
    """
    Represents the WavefieldOutput subsection in the Output section.
    """

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
class OutputManager:
    """
    Manages FrequenSolve outputs.

    Attributes:
       traces (Optional[TraceOutput]):           Trace output request. None disables traces.
       paraview (List[ParaviewOutput]):          List of ParaView outputs.
       wavefields (List[WavefieldOutput]):       List of wavefield outputs.
    """

    traces: Optional[TraceOutput] = field(default_factory=TraceOutput)
    paraview: List[ParaviewOutput] = field(default_factory=list)
    wavefields: List[WavefieldOutput] = field(default_factory=list)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __iadd__(self, output: Output) -> "OutputManager":
        """Overrides += operator to add output"""
        if isinstance(output, TraceOutput):
            self.traces = output
        elif isinstance(output, ParaviewOutput):
            self.paraview.append(output)
        elif isinstance(output, WavefieldOutput):
            self.wavefields.append(output)
        else:
            raise TypeError(f"Unsupported output type: {type(output).__name__}")
        return self

    def add(self, output: Output) -> "OutputManager":
        self += output
        return self

    def disable_traces(self) -> "OutputManager":
        self.traces = None
        return self

    def to_fs(self, ctx=None) -> Dict:
        trace_payload = self.traces.to_fs(ctx) if self.traces is not None else None
        payload = {
            **({"traces": trace_payload} if trace_payload else {}),
            "ParaView": [pv_out.to_fs(ctx) for pv_out in self.paraview],
            "wavefields": [wf_out.to_fs(ctx) for wf_out in self.wavefields],
        }
        return {key: value for key, value in payload.items() if value}

    @classmethod
    def from_fs(cls, data: Dict) -> "OutputManager":
        data = copy.deepcopy(data)
        traces = data.get("traces")
        # Load-only compatibility for pre-traces solver/API payloads.
        receivers = data.get("receivers")
        paraview = [
            ParaviewOutput.from_fs(pv_out) for pv_out in data.get("ParaView", [])
        ]
        wavefields = [
            WavefieldOutput.from_fs(wf_out) for wf_out in data.get("wavefields", [])
        ]
        trace_output = (
            TraceOutput.from_fs(traces)
            if traces is not None
            else (TraceOutput.from_fs(receivers) if receivers is not None else None)
        )
        return cls(
            traces=trace_output,
            paraview=paraview,
            wavefields=wavefields,
        )
