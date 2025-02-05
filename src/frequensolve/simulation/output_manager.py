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
    "ReflectivityOutput",
    "WavefieldOutput",
]


@register_class
@dataclass(kw_only=True)
class Output(ABC):
    """Base class for all outputs."""

    name: str = ""
    path: Union[str, Path] = None
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

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

    def _set_path(self, proj_path: Path, rel_path: Path):

        # This is for copying/moving a project; we want path to be resolved in the new project
        try:
            self.path = proj_path / self.path.relative_to(self._proj_path)
        except Exception as e:
            pass

        self._proj_path = proj_path
        self._rel_path = rel_path

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


@register_class
@dataclass(kw_only=True)
class ReceiverOutput(Output):
    path: Optional[Path] = None

    def __init__(self, path: Optional[Union[str, Path]] = None, **kwargs):
        self.path = Path(path).resolve() if path else None

    def __dict__(self) -> Dict:
        if self.path is None:
            self.path = self._path
        if not self.path.exists():
            self.path.mkdir(parents=True)

        return {
            "_type": self.__class__.__name__,
            "path": self.path.relative_to(self._proj_path),
        }


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

    name: str = "paraview"
    path: Optional[Path] = None
    fields: Optional[List[str]] = None
    upscale: int = 1

    def __init__(
        self,
        name: str = "paraview",
        path: Optional[Union[str, Path]] = None,
        fields: Optional[List[str]] = None,
        upscale: int = 1,
        **kwargs,
    ):
        self.name = name
        self.path = Path(path).resolve() if path else None
        self.fields = fields
        self.upscale = upscale

    def __dict__(self) -> Dict:
        if self.path is None:
            self.path = self._path
        if not self.path.exists():
            self.path.mkdir(parents=True)

        if self.fields is None:
            self.fields = ["primary"]

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path.relative_to(self._proj_path),
            "fields": self.fields,
            "upscale": self.upscale,
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
    path: Optional[Path] = None
    fields: Optional[List[str]] = None
    grid: Optional[CartesianGrid] = None

    def __init__(
        self,
        name: str = "wavefield",
        path: Optional[Union[str, Path]] = None,
        fields: Optional[List[str]] = None,
        grid: Optional[CartesianGrid] = None,
        **kwargs,
    ):
        self.name = name
        self.path = Path(path).resolve() if path else None
        self.fields = fields
        self.grid = grid

    def __dict__(self) -> Dict:
        if self.path is None:
            self.path = self._path
        if not self.path.exists():
            self.path.mkdir(parents=True)

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


@register_class
@dataclass(kw_only=True)
class ReflectivityOutput(Output):
    """
    Represents the ReflectivityImage subsection in the Output section.
    """

    name: str = "reflectivity"
    path: Optional[Path] = None
    grid: Optional[CartesianGrid] = None

    def __init__(
        self,
        name: str = "reflectivity",
        path: Optional[Union[str, Path]] = None,
        grid: Optional[CartesianGrid] = None,
        **kwargs,
    ):
        self.name = name
        self.path = Path(path) if path else None
        self.grid = grid

    def __dict__(self) -> Dict:
        if self.path is None:
            self.path = self._path / "reflectivity"
        if not self.path.exists():
            self.path.mkdir(parents=True)

        if self.grid is None:
            self.grid = CartesianGrid()

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "path": self.path.relative_to(self._proj_path),
            "grid": self.grid.__dict__(),
        }

    @classmethod
    def from_dict(cls, dict: Dict) -> "ReflectivityOutput":
        return cls(
            name=dict.get("name", "reflectivity"),
            path=dict.get("path"),
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
       reflectivity (List[ReflectivityOutput]):  List of reflectivity outputs
    """

    write_receivers: bool = True
    receivers: ReceiverOutput = field(default_factory=ReceiverOutput)
    paraview: List[ParaviewOutput] = field(default_factory=list)
    wavefields: List[WavefieldOutput] = field(default_factory=list)
    reflectivity: List[ReflectivityOutput] = field(default_factory=list)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __iadd__(self, output: Output) -> "OutputManager":
        """Overrides += operator to add output"""
        if isinstance(output, ParaviewOutput):
            self.paraview.append(output)
        elif isinstance(output, WavefieldOutput):
            self.wavefields.append(output)
        elif isinstance(output, ReflectivityOutput):
            self.reflectivity.append(output)
        return self

    def __dict__(self) -> Dict:
        dict = {
            **(
                {"receivers": self.receivers.__dict__()} if self.write_receivers else {}
            ),
            "ParaView": [pv_out.__dict__() for pv_out in self.paraview],
            "wavefields": [wf_out.__dict__() for wf_out in self.wavefields],
            "reflectivity": [ri_out.__dict__() for ri_out in self.reflectivity],
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
        reflectivity = [
            ReflectivityOutput.from_dict(ri_out)
            for ri_out in dict.get("reflectivity", [])
        ]
        return cls(
            write_receivers=receivers is not None,
            paraview=paraview,
            wavefields=wavefields,
            reflectivity=reflectivity,
        )

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

        for out in [self.receivers]:
            out._set_path(proj_path, self._rel_path / "receivers")
        for out in self.paraview:
            out._set_path(proj_path, self._rel_path / "ParaView")
        for out in self.wavefields:
            out._set_path(proj_path, self._rel_path / "wavefields")
        for out in self.reflectivity:
            out._set_path(proj_path, self._rel_path / "reflectivity")

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
