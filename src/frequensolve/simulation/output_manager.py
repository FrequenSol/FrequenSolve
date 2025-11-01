from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.util.class_registry import class_registry, register_class

__all__ = [
    "Output",
    "OutputManager",
    "ParaviewOutput",
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
        class_name = data["_type"]
        if class_name in class_registry:
            out_class = class_registry[class_name]
            return out_class.from_dict(data)
        else:
            raise ValueError(f"Unknown output class: {class_name}")


@register_class
@dataclass(kw_only=True)
class ReceiverOutput(Output):
    path: Optional[Path] = None

    def __init__(self, **kwargs):
        self.path = "receivers"

    def __dict__(self) -> Dict:
        return {"_type": self.__class__.__name__, "path": self.path}


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
    upscale: int = 1
    show_pml: bool = True

    def __init__(
        self,
        name: str = "ParaView",
        path: Union[str, Path] = "ParaView",
        fields: Optional[List[str]] = None,
        properties: Optional[List[str]] = None,
        upscale: int = 1,
        show_pml: bool = True,
        **kwargs,
    ):
        self.name = name
        self.path = path
        self.fields = fields
        self.properties = properties
        self.upscale = upscale
        self.show_pml = show_pml
        self.kwargs = kwargs

    def __dict__(self) -> Dict:
        if self.fields is None:
            self.fields = ["all"]

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path,
            "fields": self.fields,
            "properties": self.properties,
            "upscale": self.upscale,
            "show_pml": self.show_pml,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ParaviewOutput":
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
        self.fields = fields
        self.grid = grid

    def __dict__(self) -> Dict:
        if self.fields is None:
            self.fields = ["primary"]

        if self.grid is None:
            self.grid = CartesianGrid(x0=[], x1=[], n=[])

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path.relative_to(self._proj_path),
            "fields": self.fields,
            "grid": self.grid.__dict__(),
        }

    @classmethod
    def from_dict(cls, dict: Dict) -> "WavefieldOutput":
        return cls(
            name=dict.get("name", "wavefield"),
            path=dict.get("path"),
            fields=dict.get("fields"),
            grid=CartesianGrid.from_dict(dict["grid"]),
        )


@dataclass
class OutputManager:
    """
    Manages FrequenSolve outputs.

    Receiver output is enabled by default.

    Attributes:
       receivers (List[ReceiverOutput]):         Receiver output path
       paraview (List[ParaviewOutput]):          List of paraview outputs
       wavefields (List[WavefieldOutput]):       List of wavefield outputs
    """

    write_receivers: bool = True
    receivers: ReceiverOutput = field(default_factory=ReceiverOutput)
    paraview: List[ParaviewOutput] = field(default_factory=list)
    wavefields: List[WavefieldOutput] = field(default_factory=list)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __iadd__(self, output: Output) -> "OutputManager":
        """Overrides += operator to add output"""
        if isinstance(output, ParaviewOutput):
            self.paraview.append(output)
        elif isinstance(output, WavefieldOutput):
            self.wavefields.append(output)
        return self

    def __dict__(self) -> Dict:
        dict = {
            **(
                {"receivers": self.receivers.__dict__()} if self.write_receivers else {}
            ),
            "ParaView": [pv_out.__dict__() for pv_out in self.paraview],
            "wavefields": [wf_out.__dict__() for wf_out in self.wavefields],
        }
        return {k: v for k, v in dict.items() if v}

    @classmethod
    def from_dict(cls, dict: Dict) -> None:
        receivers = dict.get("receivers")
        paraview = [
            ParaviewOutput.from_dict(pv_out) for pv_out in dict.get("ParaView", [])
        ]
        wavefields = [
            WavefieldOutput.from_dict(wf_out) for wf_out in dict.get("wavefields", [])
        ]
        return cls(
            write_receivers=receivers is not None,
            paraview=paraview,
            wavefields=wavefields,
        )
