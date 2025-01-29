from abc          import ABC, abstractmethod
from dataclasses  import dataclass, field
from typing       import List, Dict, Optional, Union
from pathlib      import Path

from ..geometry.grids      import *    # noqa
from ..util.class_registry import *    # noqa

__all__ = ['OutputManager', 'ParaviewOutput', 'ReflectivityImage']


@register_class
@dataclass
class Output(ABC):
   """Base class for all outputs."""
   _proj_path: Optional[Path] = None
   _rel_path:  Optional[Path] = None
   
   @abstractmethod
   def __dict__(self) -> Dict:
      pass

   @classmethod
   def from_dict(cls, data: Dict) -> 'Output':  
      class_name = data["_type"]
      if class_name in class_registry:
         out_class = class_registry[class_name]
         return out_class.from_dict(data)
      else:
         raise ValueError(f"Unknown output class: {class_name}")
      
   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path

   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path


@register_class
@dataclass
class ParaviewOutput(Output):
   """
   Represents the Paraview subsection in the Output section.

   Attributes:
      directory (str): The directory to store the Paraview output.
      components (List[str]): The components to include in the Paraview output.
      prefix (str): The prefix for the Paraview output files.
      upscale (int): The upscale factor for the Paraview output.
   """
   name:       str = "paraview"
   path:       Optional[Union[str, Path]] = None
   components: List[str] = field(default_factory=lambda: ["pressure"])
   upscale:    int = 1

   def __dict__(self) -> Dict:
      if self.path is None:
         self.path = self._rel_path
      if not self.path.exists():
         self.path.mkdir(parents=True)

      return {
         "name": self.name,
         "path": self.path,
         "components": self.components,
         "upscale": self.upscale,
      }
   
   @classmethod
   def from_dict(cls, dict: Dict) -> 'ParaviewOutput':
      return cls(**dict)


@register_class
@dataclass
class WavefieldOutput:
   """
   Represents the WavefieldOutput subsection in the Output section.
   """
   name: str = "wavefield"
   path: Optional[Union[str, Path]] = None
   grid: CartesianGrid = field(default_factory=CartesianGrid)

   def __dict__(self) -> Dict:
      return {
         "_type": self.__class__.__name__,
         "name": self.name,
         "path": self.path,
         "grid": self.grid.__dict__()
      }

   @classmethod
   def from_dict(cls, dict: Dict) -> 'WavefieldOutput':
      return cls(
         name = dict.get("name","wavefield"),
         path = dict["path"],
         grid = CartesianGrid.from_dict(dict["grid"])
      )


@register_class
@dataclass
class ReflectivityImage:
   """
   Represents the ReflectivityImage subsection in the Output section.
   """
   name: str = "reflectivity"
   path: Union[str, Path] = None
   grid: CartesianGrid = field(default_factory=CartesianGrid)

   def __dict__(self) -> Dict:
      return {
         "_type": self.__class__.__name__,
         "name": self.name,
         "path": self.path,
         "grid": self.grid.__dict__()
      }
   
   @classmethod
   def from_dict(cls, dict: Dict) -> 'ReflectivityImage':
      return cls(
         name = dict.get("name","reflectivity"),
         path = dict["path"],
         grid = CartesianGrid.from_dict(dict["grid"])
      )


@dataclass
class OutputManager:
   """
   Manages FrequenSolve outputs.

   Attributes:
      outputs (List[Output]): List of outputs
   """
   outputs:    List[Output]   = field(default_factory=list)
   _proj_path: Optional[Path] = None
   _rel_path:  Optional[Path] = None

   def __iadd__(self, output: Output) -> "OutputManager":
      """Overrides += operator to add output"""
      self.outputs.append(output)
      return self

   def __dict__(self) -> Dict:
      return {
         "paraview":     [pv_out.__dict__() for pv_out in self.outputs if isinstance(pv_out, ParaviewOutput)],
         "wavefield":    [wf_out.__dict__() for wf_out in self.outputs if isinstance(wf_out, WavefieldOutput)],
         "reflectivity": [ri_out.__dict__() for ri_out in self.outputs if isinstance(ri_out, ReflectivityImage)]
      }

   @classmethod
   def from_dict(cls, dict: Dict) -> None:
      outputs  = [ParaviewOutput.from_dict(pv_out) for pv_out in dict["paraview"]]
      outputs += [WavefieldOutput.from_dict(wf_out) for wf_out in dict["wavefield"]]
      outputs += [ReflectivityImage.from_dict(ri_out) for ri_out in dict["reflectivity"]]
      return cls(outputs = outputs)

   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path/"outputs"
      for out in self.outputs:
         out._set_path(proj_path, self._rel_path)

   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path
   
