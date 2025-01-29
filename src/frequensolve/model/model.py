"""Model base classes for managing simulation models."""

from xarray import DataArray, Dataset

from pathlib               import Path
from dataclasses           import dataclass, field, asdict
from typing                import Optional, List, Dict, Union, Literal, Tuple

from ..util.class_registry import *

__all__ = ['ModelSubdomain', 'ModelBase']

# TODO: Convert properties to xr.Dataset (will have to update my code, etc.)

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
   properties:    Dict[str, Union[float, str, DataArray]] = field(default_factory=dict)
   _proj_path:    Optional[Path] = None
   _rel_path:     Optional[Path] = None

   def __dict__(self) -> Dict:
      if isinstance(self.properties, xr.Dataset):
         raise NotImplementedError("Subdomains with xr.Dataset properties are not supported yet")
      return {
         "mesh_block_id": self.mesh_block_id,
         "name":          self.name,
         "frame":         self.frame,
         "properties":    self.properties,
      }
   
   @classmethod
   def from_dict(cls, data: Dict) -> "ModelSubdomain":
      raise NotImplementedError("Subclasses must implement from_dict()")

   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path

   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path


@register_class
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
   _proj_path: Optional[Path] = None
   _rel_path:  Optional[Path] = None

   def __dict__(self) -> Dict:

      # Label any unlabeled subdomains
      labels = {}
      for i,subdomain in enumerate(self.subdomains):
         labels[subdomain.mesh_block_id] = i

      j = 1
      for i, subdomain in enumerate(self.subdomains):
         if subdomain.mesh_block_id == -1:
            while j in labels:
               j += 1
            labels[j] = i
            subdomain.mesh_block_id = j

      return {
         "_type": self.__class__.__name__,
         "name": self.name,
         "dimension": self.dimension,
         "subdomains": [subdomain.__dict__() for subdomain in self.subdomains]
      }

   @classmethod
   def from_dict(cls, data: Dict) -> "ModelBase":
      class_name = data["_type"]
      if class_name in class_registry:
         model_class = class_registry[class_name]
         return model_class.from_dict(data)
      else:
         raise ValueError(f"Unknown model class: {class_name}")
   
   def add_subdomain(self, subdomain: ModelSubdomain) -> None:
      """Adds a subdomain to the model.

      Args:
         id (int): Unique mesh block identifier
         **kwargs: Additional subdomain parameters.
      """
      self.subdomains.append(subdomain)

   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path/self.name
      for subdomain in self.subdomains:
         subdomain._set_path(proj_path, self._rel_path)

   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path
