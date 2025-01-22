"""Model base classes for managing simulation models."""

import xarray

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Literal

__all__ = ['ModelSubdomain', 'ModelBase']

@dataclass(kw_only=True)
class ModelSubdomain:
   """A subdomain within a model with associated properties.
   
   Attributes:
      mesh_block_id (int):  Unique identifier for the mesh block.
      name (Optional[str]): Optional name for the mesh block.
      frame (str):          Coordinate frame for mapping subdomain materials ('physical' or 'reference').
      properties (Dict[str, Union[float, str, xarray.DataArray]]): Dictionary of subdomain properties.
         Keys are property names, values can be numeric constants, file paths, or xarray DataArrays.
   """
   mesh_block_id: int = -1
   name:          Optional[str] = None
   frame:         str = "physical"
   properties:    Dict[str, Union[float, str, "xarray.DataArray"]] = field(default_factory=dict)

   def to_dict(self) -> Dict:
      return {
         "mesh_block_id": self.mesh_block_id,
         "name": self.name,
         "frame": self.frame,
         "properties": self.properties
      }
   
   @classmethod
   def from_dict(cls, data: Dict) -> "ModelSubdomain":
      raise NotImplementedError("Subclasses must implement from_dict()")


@dataclass(kw_only=True)
class ModelBase:
   """Base class for simulation models.
   
   Provides common attributes and functionality shared by different model types.
   
   Attributes:
      name (str):                Name identifier for the model.
      dimension (Literal[2, 3]): Model dimension (2D or 3D).
      x_limits (List[float]):    Model extent in x-direction [xmin, xmax].
      y_limits (List[float]):    Model extent in y-direction [ymin, ymax].
      z_limits (List[float]):    Model extent in z-direction [zmin, zmax].
      properties (Dict[str, Union[float, str]]): Dictionary of model properties.
         Keys are property names, values can be numeric constants or file paths.
   """
   name:       str = "model"
   dimension:  Literal[2, 3]
   subdomains: List[ModelSubdomain] = field(default_factory=list)

   def to_dict(self) -> Dict:

      # Label any unlabeled subdomains
      labels = {}
      for i,subdomain in enumerate(self.subdomains):
         labels[subdomain.mesh_block_id] = i

      j = 0
      for i, subdomain in enumerate(self.subdomains):
         if subdomain.mesh_block_id == -1:
            while j in labels:
               j += 1
            subdomain.mesh_block_id = j

      return {
         "name": self.name,
         "dimension": self.dimension,
         "subdomains": [ subdomain.to_dict() for subdomain in self.subdomains]
      }

   @classmethod
   def from_dict(cls, data: Dict) -> "ModelBase":
      return cls(
         name=data["name"],
         dimension=data["dimension"],
         subdomains={
            id: ModelSubdomain.from_dict(subdomain_data)
            for id, subdomain_data in data["subdomains"].items()
         }
      )
   
   def add_subdomain(self, subdomain: ModelSubdomain) -> None:
      """Adds a subdomain to the model.

      Args:
         id (int): Unique mesh block identifier
         **kwargs: Additional subdomain parameters.
      """
      self.subdomains.append(subdomain)
