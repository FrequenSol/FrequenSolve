"""Python structures defining mesh API"""

from dataclasses import dataclass
from typing      import List, Dict, Optional, Union, Tuple
from pathlib     import Path

from ..seismic.layered_model import *  # noqa
from .mesh                   import *  # noqa
from .mesh_generators        import *  # noqa

__all__ = ['MeshParallelism','MeshAdaptor','MeshManager']

     
@dataclass
class MeshParallelism:
   distribute:       Optional[bool] = True
   ranks_per_part:   Optional[int]  = None
   partitioner:      Optional[str]  = None

   def __dict__(self) -> Dict:
      return {
         "distribute": self.distribute,
         **({"ranks_per_part": self.ranks_per_part} if self.ranks_per_part else {}),
         **({"partitioner": self.partitioner} if self.partitioner else {})
      }
   
@classmethod
def from_dict(cls, data: Dict) -> "MeshParallelism":
   return cls(
      distribute     = data["distribute"],
      ranks_per_part = data.get("ranks_per_part"),
      partitioner    = data.get("partitioner")
   )


@dataclass
class MeshAdaptor:
   """Sets mesh adaptivity options

   Attributes:
      min_epw (float): Minimum # of elements per wavelength.
      adapt_sources (Optional[int]): Number of additional refinements near sources
      adapt_receivers (Optional[int]): Number of additional refinements near receivers
      jump_tolerance (Optional[float]): Maximum relative change in wavespeed that consitutes
      jump_factor (Optional[float]): Multiplicative factor for min_epw on "jump" elements
      smooth_refs (Optional[bool]): Do additional refinements to unconstrain element DOFs
   """
   min_epw:          float
   adapt_sources:    Optional[int]   = None # 2
   adapt_receivers:  Optional[int]   = None # 0
   jump_tolerance:   Optional[float] = None # 0.2
   jump_factor:      Optional[float] = None # 1.0
   smooth_refs:      Optional[bool]  = None # False
#   adapt_order:      bool  = False
#   f_low:            float = 4.0
#   f_med:            float = 8.0

   def __dict__(self) -> Dict:
      return {
         "min_epw": self.min_epw,
         **({"adapt_sources":   self.adapt_sources}   if self.adapt_sources   else {}),
         **({"adapt_receivers": self.adapt_receivers} if self.adapt_receivers else {}),
         **({"jump_tolerance":  self.jump_tolerance}  if self.jump_tolerance  else {}),
         **({"jump_factor":     self.jump_factor}     if self.jump_factor     else {}),
         **({"smooth_refs":     self.smooth_refs}     if self.smooth_refs     else {})
      }
   
   @classmethod
   def from_dict(cls, data: Dict) -> "MeshAdaptor":
      return cls(
         min_epw         = data["min_epw"],
         adapt_sources   = data.get("adapt_sources"),
         adapt_receivers = data.get("adapt_receivers"),
         jump_tolerance  = data.get("jump_tolerance"),
         jump_factor     = data.get("jump_factor"),
         smooth_refs     = data.get("smooth_refs")
      )


@dataclass
class MeshManager:
   """Defines mesh type, dimension, refinement, etc.  

   Attributes:
      mesh (Optional[Mesh]): The mesh object
      mesh_generator (Optional[BaseMeshGenerator]): The mesh generator object
      parallel (Optional[MeshParallelism]): Mesh parallelism options
      adapt (Optional[MeshAdaptor]): Mesh adaptivity options
   """
   mesh:           Optional[Union[Mesh, BaseMeshGenerator]] = None
   mesh_file:      Optional[str]             = None
   mesh_format:    Optional[str]             = None
   parallel:       Optional[MeshParallelism] = None
   adapt:          Optional[MeshAdaptor]     = None
   _proj_path:     Optional[Path]            = None
   _rel_path:      Optional[Path]            = None
   def set_adapt(self,
                 min_epw:         float,
                 adapt_sources:   Optional[int]   = None,
                 adapt_receivers: Optional[int]   = None,
                 jump_tolerance:  Optional[float] = None,
                 jump_factor:     Optional[float] = None,
                 smooth_refs:     Optional[bool]  = None) -> None:
      """Sets mesh adaptivity options

      Attributes:
         min_epw (float):                  Minimum # of elements per wavelength.
         adapt_sources (Optional[int]):    Number of additional refinements near sources
         adapt_receivers (Optional[int]):  Number of additional refinements near receivers
         jump_tolerance (Optional[float]): Maximum relative change in wavespeed that consitutes
                                           a "jump" in material properties
         jump_factor (Optional[float]):    Multiplicative factor for min_epw on "jump" elements
         smooth_refs (Optional[bool]):     Do additional refinements to unconstrain element DOFs
      """
      self.adapt = MeshAdaptor(min_epw         = min_epw,
                               adapt_sources   = adapt_sources,
                               adapt_receivers = adapt_receivers,
                               jump_tolerance  = jump_tolerance,
                               jump_factor     = jump_factor,
                               smooth_refs     = smooth_refs)
                               
   def set_parallel(self,
                    distribute:      bool,
                    ranks_per_part:  Optional[int] = None,
                    partitioner:     Optional[str] = None) -> None:
      """Sets mesh parallel options

      Attributes:
         distribute (bool):               Distribute mesh
         ranks_per_part (Optional[int]):  Number of ranks per mesh part
         partitioner (Optional[str]):     Partitioner type
      """
      self.parallel = MeshParallelism(distribute     = distribute,
                                      ranks_per_part = ranks_per_part,
                                      partitioner    = partitioner)
      
   @classmethod
   def from_dict(cls, data: Dict) -> 'MeshManager':
      manager = cls()
      
      # From file
      mesh_file = data.get("mesh_file")
      mesh_format = data.get("mesh_format")
      if mesh_file is not None and mesh_format is not None:
         manager.mesh_file   = mesh_file
         manager.mesh_format = mesh_format
         manager.mesh        = Mesh.read_mesh(mesh_file, mesh_format)
         
      # From generator
      if "generator" in data:
         manager.mesh = BaseMeshGenerator.from_dict(data["generator"])
         
      # Parallel
      if "parallel" in data:
         p = data["parallel"]
         manager.set_parallel(
            distribute     = p["distribute"],
            ranks_per_part = p.get("ranks_per_part"),
            partitioner    = p.get("partitioner")
         )
         
      if "adapt" in data:
         a = data["adapt"]
         manager.set_adapt(
            min_epw         = a["min_epw"],
            adapt_sources   = a.get("adapt_sources", 0),
            adapt_receivers = a.get("adapt_receivers", 0), 
            jump_tolerance  = a.get("jump_tolerance"),
            jump_factor     = a.get("jump_factor"),
            smooth_refs     = a.get("smooth_refs", False)
         ) 
         
      return manager

   def __dict__(self) -> Dict:
      if self.adapt is None:
         self.set_adapt(min_epw = 2.0)
         
      mesh_dict = {
         "adapt": self.adapt.__dict__(),
      }

      # Mesh determined by file
      if self.mesh is None:
         assert self.mesh_file is not None and self.mesh_format is not None, \
               "if a mesh or mesh generator has not been provided, " \
               "'mesh_file' and 'mesh_format' must be provided"
         mesh_dict["mesh_file"]   = self.mesh.file
         mesh_dict["mesh_format"] = self.mesh.format

      # Mesh determined by generator (defined in backend)
      elif isinstance(self.mesh, BaseMeshGenerator):
         mesh_dict["generator"] = self.mesh.__dict__()

      # Write mesh to file (if mesh is a Mesh object)
      elif isinstance(self.mesh, Mesh):
         path = self._path/"mesh"
         self.mesh.write_mesh(path,"hp3d")
         mesh_dict["mesh_file"]   = path.relative_to(self._proj_path)
         mesh_dict["mesh_format"] = "hp3d"
      
      if self.parallel:
         mesh_dict["parallel"] = self.parallel.__dict__()
         
      return mesh_dict
   
   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path
   
   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path
