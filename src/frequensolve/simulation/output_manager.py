from abc          import ABC, abstractmethod
from dataclasses  import dataclass, field
from typing       import List, Dict, Optional, Union
from pathlib      import Path

from ..geometry.grids import * # noqa

__all__ = ['OutputManager', 'ParaviewOutput', 'ReflectivityImage']


@dataclass
class Output(ABC):
   """Base class for all outputs."""
   

   @abstractmethod
   def __dict__(self) -> Dict:
      pass

   @classmethod
   @abstractmethod
   def from_dict(cls, dict: Dict) -> 'Output':
      pass


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
      return {
         "name": self.name,
         "path": self.path,
         "components": self.components,
         "upscale": self.upscale,
      }
   
   @classmethod
   def from_dict(cls, dict: Dict) -> 'ParaviewOutput':
      return cls(**dict)


@dataclass
class WavefieldOutput:
   """
   Represents the WavefieldOutput subsection in the Output section.
   """
   name: str = "wavefield"
   path: Optional[Union[str, Path]] = None
   type: str = "grid"
   grid: CartesianGrid = field(default_factory=CartesianGrid)

   def __dict__(self) -> Dict:
      return {
         "name": self.name,
         "path": self.path,
         "type": self.type,
         "grid": self.grid.__dict__()
      }

   @classmethod
   def from_dict(cls, dict: Dict) -> 'WavefieldOutput':
      return cls(
         path=dict["path"],
         type=dict["type"],
         grid=CartesianGrid.from_dict(dict["grid"])
      )


@dataclass
class ReflectivityImage:
   """
   Represents the ReflectivityImage subsection in the Output section.
   """
   name: str = "reflectivity"
   path: Union[str, Path] = None
   type: str = "grid"
   grid: CartesianGrid = field(default_factory=CartesianGrid)

   def __dict__(self) -> Dict:
      return {
         "name": self.name,
         "path": self.path,
         "type": self.type,
         "grid": self.grid.__dict__()
      }
   
   @classmethod
   def from_dict(cls, dict: Dict) -> 'ReflectivityImage':
      return cls(
         path=dict["path"],
         type=dict["type"],
         grid=CartesianGrid.from_dict(dict["grid"])
      )


@dataclass
class OutputManager:
   """
   Manages FrequenSolve outputs.

   Attributes:
      outputs (List[Output]): List of outputs
   """
   outputs: List[Output] = field(default_factory=list)

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

   
