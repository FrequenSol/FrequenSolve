"""Python structures defining mesh API"""

from pathlib     import Path
from abc         import ABC, abstractmethod
from dataclasses import dataclass
from typing      import List, Dict, Optional

from ..seismic.layered_model import *  # noqa
from .mesh                   import *  # noqa
from ..util.class_registry   import *  # noqa

__all__ = ['BaseMeshGenerator', 'HexMeshGenerator']

@register_class
@dataclass
class BaseMeshGenerator(ABC):
   """Base class for mesh generators"""

   _proj_path: Path = Path()
   _rel_path: Path = Path()

   @classmethod
   def from_dict(cls, data: Dict) -> 'BaseMeshGenerator':
      class_name = data["_type"]
      if class_name in class_registry:
         mesh_class = class_registry[class_name]
         return mesh_class.from_dict(data)
      else:
         raise ValueError(f"Unknown mesh generator class: {class_name}")
      
   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path

   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path


@register_class
@dataclass
class HexMeshGenerator(BaseMeshGenerator):
   """Generates a hexahedral mesh

   Attributes:
      model (LayeredModel): 
         Model to use for generating the mesh
      n (List[int]): 
         Number of elements in each direction
      l_bound (List[float], optional): 
         The lower bounds of the mesh (optional, can be inferred from model)
      u_bound (List[float], optional): 
         The upper bounds of the mesh (optional, can be inferred from model)
   """
   n:       Optional[List[int]]    = None
   model:   Optional[LayeredModel] = None
   l_bound: Optional[List[float]]  = None
   u_bound: Optional[List[float]]  = None
   
   def __dict__(self) -> Dict:
      if self.l_bound is not None:
         assert self.u_bound is not None
         l_bound = self.l_bound
         u_bound = self.u_bound
      elif self.model is not None:
         if self.model.dimension == 2:
            x_limits = self.model.x_limits
            z_limits = self.model.z_limits
            l_bound = [x_limits[0], z_limits[0]]
            u_bound = [x_limits[1], z_limits[1]]
         else:
            x_limits = self.model.x_limits
            y_limits = self.model.y_limits
            z_limits = self.model.z_limits
            l_bound = [x_limits[0], y_limits[0], z_limits[0]]
            u_bound = [x_limits[1], y_limits[1], z_limits[1]]

      if self.n is None:
         self.n = [16] * self.model.dimension

      return {
         "_type":   self.__class__.__name__,
         "path":    self._rel_path,
         "n":       self.n,
         "l_bound": l_bound,
         "u_bound": u_bound,
      }
   
   @classmethod
   def from_dict(cls, data: Dict) -> 'HexMeshGenerator':
      return cls(
         n       = data["n"],
         l_bound = data["l_bound"],
         u_bound = data["u_bound"],
      )