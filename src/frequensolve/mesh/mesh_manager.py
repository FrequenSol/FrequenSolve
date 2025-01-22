"""Python structures defining mesh API"""

from dataclasses import dataclass
from typing      import List, Dict, Optional, Union

from ..seismic.layered_model import *  # noqa
from ..simulation.config     import *  # noqa
from .mesh                   import *  # noqa

__all__ = ['HexMeshGenerator','MeshParallelism',
           'MeshAdaptor','MeshManager']


@dataclass
class HexMeshGenerator:
   """Generates a hexahedral mesh

   Attributes:
      n (List[int]): number of elements in each direction
      model (LayeredModel): The model to use for generating the mesh
   """
   n:       List[int]
   model:   LayeredModel
   
   def to_dict(self) -> Dict:
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

      return {
         "type": "hex_mesh_generator",
         "n": self.n,
         "l_bound": l_bound,
         "u_bound": u_bound,
      }

     
@dataclass
class MeshParallelism:
   distribute:       Optional[bool] = True
   ranks_per_part:   Optional[int]  = None
   partitioner:      Optional[str]  = None

   def to_dict(self) -> Dict:
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
   adapt_sources:    Optional[int  ] = None # 2
   adapt_receivers:  Optional[int  ] = None # 0
   jump_tolerance:   Optional[float] = None # 0.2
   jump_factor:      Optional[float] = None # 1.0
   smooth_refs:      Optional[bool ] = None # False
#   adapt_order:      bool  = False
#   f_low:            float = 4.0
#   f_med:            float = 8.0

   def to_dict(self) -> Dict:
      return {
         "min_epw": self.min_epw,
         **({"adapt_sources": self.adapt_sources} if self.adapt_sources else {}),
         **({"adapt_receivers": self.adapt_receivers} if self.adapt_receivers else {}),
         **({"jump_tolerance": self.jump_tolerance} if self.jump_tolerance else {}),
         **({"jump_factor": self.jump_factor} if self.jump_factor else {}),
         **({"smooth_refs": self.smooth_refs} if self.smooth_refs else {})
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
      mesh_generator (Optional[HexMeshGenerator]): The mesh generator object
      parallel (Optional[MeshParallelism]): Mesh parallelism options
      adapt (Optional[MeshAdaptor]): Mesh adaptivity options
   """
   mesh:           Union[Mesh, HexMeshGenerator] = None
   mesh_file:      Optional[str] = None
   mesh_format:    Optional[str] = None
   parallel:       Optional[MeshParallelism] = None
   adapt:          Optional[MeshAdaptor]     = None
   
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

   def to_dict(self) -> Dict:
      if self.adapt is None:
         self.set_adapt(min_epw = 2.0)
         
      mesh_dict = {
         "adapt": self.adapt.to_dict(),
      }

      # Mesh determined by file
      if self.mesh is None:
         assert self.mesh_file is not None and self.mesh_format is not None, \
               "if a mesh or mesh generator has not been provided, " \
               "'mesh_file' and 'mesh_format' must be provided"
         mesh_dict["mesh_file"]   = self.mesh.file
         mesh_dict["mesh_format"] = self.mesh.format
      if isinstance(self.mesh, HexMeshGenerator):
         mesh_dict["generator"] = self.mesh.to_dict()
      elif isinstance(self.mesh, Mesh):
         self.mesh.write_mesh(self.mesh.file, self.mesh.format)
         mesh_dict["mesh_file"]   = self.mesh.file
         mesh_dict["mesh_format"] = self.mesh.format
      
      if self.parallel:
         mesh_dict["parallel"] = self.parallel.to_dict()
         
      return mesh_dict
   
   @classmethod
   def from_dict(cls, sim: SimulationConfig, data: Dict) -> 'MeshManager':
      manager = cls()
      
      # From file
      mesh_file = data.get("mesh_file")
      mesh_format = data.get("mesh_format")
      if mesh_file is not None and mesh_format is not None:
         manager.mesh_file = mesh_file
         manager.mesh_format = mesh_format
         manager.mesh = Mesh.read_mesh(mesh_file, mesh_format)
         
      # From generator
      if "generator" in data:
         g = data["generator"]
         manager.mesh = HexMeshGenerator.from_dict(g)
         
      # Parallel
      if "parallel" in data:
         p = data["parallel"]
         manager.set_parallel(
            distribute=p["distribute"],
            ranks_per_part=p.get("ranks_per_part"),
            partitioner=p.get("partitioner")
         )
         
      if "adapt" in data:
         a = data["adapt"]
         manager.set_adapt(
            min_epw=a["min_epw"],
            adapt_sources=a.get("adapt_sources", False),
            adapt_receivers=a.get("adapt_receivers", False), 
            jump_tolerance=a.get("jump_tolerance"),
            jump_factor=a.get("jump_factor"),
            smooth_refs=a.get("smooth_refs", False)
         )
         
      return manager
