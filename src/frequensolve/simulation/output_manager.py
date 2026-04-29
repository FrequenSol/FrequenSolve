import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.fields import canonical_fields
from frequensolve.util.mixins import merge_extra

__all__ = [
    "Output",
    "OutputManager",
    "ParaviewOutput",
    "TraceOutput",
    "WavefieldOutput",
]


@register_class
@dataclass(kw_only=True)
class Output(ABC):
    """Base class for all outputs.

    Note path will be relative to result_path, specified in the job."""

    name: str = ""
    path: Union[str, Path] = None

    @abstractmethod
    def __dict__(self) -> Dict:
        pass

    @classmethod
    def from_dict(cls, data: Dict) -> "Output":
        data = copy.deepcopy(data)
        class_name = data["_type"]
        if class_name in class_registry:
            out_class = class_registry[class_name]
            return out_class.from_dict(data)
        else:
            raise ValueError(f"Unknown output class: {class_name}")


@register_class
@dataclass(kw_only=True)
class TraceOutput(Output):
    path: Optional[Path] = None

    def __init__(self, path: Union[str, Path] = "traces", **kwargs):
        self.path = path
        self.extra = kwargs

    def to_fs(self, ctx=None) -> Dict:
        return merge_extra(
            {"_type": self.__class__.__name__, "path": self.path},
            self.extra,
            self.__class__.__name__,
        )

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Dict) -> "TraceOutput":
        data = copy.deepcopy(data)
        data.pop("_type", None)
        return cls(path=data.pop("path", "traces"), **data)


@register_class
@dataclass(kw_only=True)
class ReceiverOutput(TraceOutput):
    """Backward-compatible name for trace output configuration."""

    def __init__(self, path: Union[str, Path] = "receivers", **kwargs):
        super().__init__(path=path, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict) -> "ReceiverOutput":
        data = copy.deepcopy(data)
        data.pop("_type", None)
        return cls(path=data.pop("path", "receivers"), **data)


@register_class
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

        self.name = name
        self.path = path
        self.fields = canonical_fields(fields) if fields is not None else None
        self.properties = properties
        self.sources = sources
        self.upscale = upscale
        self.show_pml = show_pml
        self.extra = kwargs

    @property
    def kwargs(self) -> Dict:
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Dict) -> None:
        self.extra = copy.deepcopy(dict(value))

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

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Dict) -> "ParaviewOutput":
        data = copy.deepcopy(data)
        data.pop("_type", None)
        if data.get("path") is not None:
            data["path"] = Path(data["path"])
        return cls(**data)


@register_class
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
        self.path = path
        self.fields = canonical_fields(fields) if fields is not None else None
        self.grid = grid
        self.extra = kwargs

    def to_fs(self, ctx=None) -> Dict:
        fields = (
            canonical_fields(self.fields) if self.fields is not None else ["primary"]
        )
        grid = self.grid if self.grid is not None else CartesianGrid(x0=[], x1=[], n=[])

        path = self.path
        if isinstance(path, Path) and getattr(self, "_proj_path", None) is not None:
            path = path.relative_to(self._proj_path)

        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": path,
            "fields": fields,
            "grid": grid.to_fs(ctx) if hasattr(grid, "to_fs") else grid.__dict__(),
        }
        return merge_extra(payload, self.extra, "WavefieldOutput")

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, dict: Dict) -> "WavefieldOutput":
        dict = copy.deepcopy(dict)
        dict.pop("_type", None)
        grid = dict.pop("grid", None)
        return cls(
            name=dict.pop("name", "wavefield"),
            path=dict.pop("path", "wavefields"),
            fields=dict.pop("fields", None),
            grid=CartesianGrid.from_dict(grid) if grid is not None else None,
            **dict,
        )


@dataclass
class OutputManager:
    """
    Manages FrequenSolve outputs.

    Receiver output is enabled by default.

    Attributes:
       receivers (TraceOutput):                  Backward-compatible trace output path
       paraview (List[ParaviewOutput]):          List of paraview outputs
       wavefields (List[WavefieldOutput]):       List of wavefield outputs
    """

    write_receivers: bool = True
    receivers: TraceOutput = field(default_factory=TraceOutput)
    paraview: List[ParaviewOutput] = field(default_factory=list)
    wavefields: List[WavefieldOutput] = field(default_factory=list)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __iadd__(self, output: Output) -> "OutputManager":
        """Overrides += operator to add output"""
        if isinstance(output, (TraceOutput, ReceiverOutput)):
            self.receivers = output
            self.write_receivers = True
        elif isinstance(output, ParaviewOutput):
            self.paraview.append(output)
        elif isinstance(output, WavefieldOutput):
            self.wavefields.append(output)
        return self

    def to_fs(self, ctx=None) -> Dict:
        trace_payload = self.receivers.to_fs(ctx) if self.write_receivers else None
        receiver_payload = copy.deepcopy(trace_payload) if trace_payload else None
        if receiver_payload is not None:
            receiver_payload["_type"] = "ReceiverOutput"
        dict = {
            **(
                {"traces": trace_payload, "receivers": receiver_payload}
                if trace_payload
                else {}
            ),
            "ParaView": [pv_out.to_fs(ctx) for pv_out in self.paraview],
            "wavefields": [wf_out.to_fs(ctx) for wf_out in self.wavefields],
        }
        return {k: v for k, v in dict.items() if v}

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, dict: Dict) -> None:
        dict = copy.deepcopy(dict)
        traces = dict.get("traces")
        receivers = dict.get("receivers")
        paraview = [
            ParaviewOutput.from_dict(pv_out) for pv_out in dict.get("ParaView", [])
        ]
        wavefields = [
            WavefieldOutput.from_dict(wf_out) for wf_out in dict.get("wavefields", [])
        ]
        receiver_output = (
            TraceOutput.from_dict(traces)
            if traces is not None
            else (
                ReceiverOutput.from_dict(receivers)
                if receivers is not None
                else TraceOutput()
            )
        )
        return cls(
            write_receivers=traces is not None or receivers is not None,
            receivers=receiver_output,
            paraview=paraview,
            wavefields=wavefields,
        )

    @property
    def traces(self) -> TraceOutput:
        return self.receivers

    @traces.setter
    def traces(self, value: TraceOutput) -> None:
        self.receivers = value
