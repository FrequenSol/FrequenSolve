from dataclasses import dataclass, field
from typing      import List, Dict, Optional, Union
from pathlib     import Path

from ..geometry.grids import * # noqa

__all__ = ['OutputManager', 'ParaviewOutput', 'ReflectivityImage']


@dataclass
class ParaviewOutput:
   """
   Represents the Paraview subsection in the Output section.

   Attributes:
      directory (str): The directory to store the Paraview output.
      components (List[str]): The components to include in the Paraview output.
      prefix (str): The prefix for the Paraview output files.
      upscale (int): The upscale factor for the Paraview output.
   """
   directory: str = "../output/paraview/"
   components: List[str] = field(default_factory=lambda: ["pressure"])
   prefix: str = "paraview"
   upscale: int = 1

   def to_dict(self) -> Dict:
      """Converts the Paraview output settings to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the Paraview output settings.
      """
      return {
         "directory": self.directory,
         "components": self.components,
         "prefix": self.prefix,
         "upscale": self.upscale
      }
   
   @classmethod
   def from_dict(cls, dict: Dict) -> 'ParaviewOutput':
      return cls(
         directory=dict["directory"],
         components=dict["components"],
         prefix=dict["prefix"],
         upscale=dict["upscale"]
      )
   
   def __str__(self) -> str:
      return (f"[Paraview]\n"
              f"   directory = {self.directory}\n"
              f"   components = {self.components}\n"
              f"   prefix = {self.prefix}\n"
              f"   upscale = {self.upscale}\n"
              f"[]\n\n")

@dataclass
class ReflectivityImage:
   """
   Represents the ReflectivityImage subsection in the Output section.
   """
   path: Union[str, Path]
   type: str = "grid"
   grid: CartesianGrid = field(default_factory=CartesianGrid)

   def to_dict(self) -> Dict:
      return {
         "path": self.path,
         "type": self.type,
         "grid": self.grid.to_dict()
      }
   
   @classmethod
   def from_dict(cls, dict: Dict) -> 'ReflectivityImage':
      return cls(
         path=dict["path"],
         type=dict["type"],
         grid=CartesianGrid.from_dict(dict["grid"])
      )
   
   def __str__(self) -> str:
      return (f"[ReflectivityImage]\n"
              f"   path = {self.path}\n"
              f"   type = {self.type}\n"
              f"   {str(self.grid)}\n"
              f"[]\n\n")


@dataclass
class OutputManager:
   """
   Handles paraview outputs (for now, extensions intended)

   Attributes:
      paraview_output (Optional[ParaviewOutput]): The Paraview output configuration. 
   """
   paraview_outputs: List[ParaviewOutput] = field(default_factory=list)
   reflectivity_images: List[ReflectivityImage] = field(default_factory=list)


   def add_paraview_output(self, pv_out: ParaviewOutput) -> None:
      self.paraview_outputs.append(pv_out)


   def add_reflectivity_image(self, ri: ReflectivityImage) -> None:
      self.reflectivity_images.append(ri)


   def to_dict(self) -> Dict:
      """Converts the Output settings to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the Output settings.
      """
      return {
         "paraview_output": [pv_out.to_dict() for pv_out in self.paraview_outputs],
         "reflectivity_images": [ri.to_dict() for ri in self.reflectivity_images]
      }
   

   @classmethod
   def from_dict(cls, dict: Dict) -> None:
      """Loads the Output settings from a dictionary representation.
      
      Args:
         dict (Dict): Dictionary containing the Output settings.
      """
      paraview_outputs = [ParaviewOutput.from_dict(pv_out) for pv_out in dict["paraview_output"]]
      reflectivity_images = [ReflectivityImage.from_dict(ri) for ri in dict["reflectivity_images"]]
      return cls(paraview_outputs=paraview_outputs, reflectivity_images=reflectivity_images)
   

   def __str__(self) -> str:
      out = "[Output]\n"
      if self.paraview_outputs:
         for pv_out in self.paraview_outputs:
            out += str(pv_out)
      if self.reflectivity_images:
         for ri in self.reflectivity_images:
            out += str(ri)
      out += "[]\n\n"
      return out
   
