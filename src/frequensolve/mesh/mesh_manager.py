
"""
@file   mesh_manager.py
@brief  Python structures defining mesh API
@date   2025-01-04
"""

from dataclasses import dataclass, field
from typing      import List, Dict, Optional, Union

from ..seismic.model import *  # noqa

__all__ = ['HexMeshGenerator','MeshParallelism',
           'MeshAdaptor','MeshManager']


@dataclass
class HexMeshGenerator:
   """
   @class   Generates a hexahedral mesh
   @brief   Sets adaptivity options
   @param   n:     (list) number of elements in each direction
   """
   
   n:       List[int]
   model:   LayeredModel
   
   def __str__(self) -> str:
      """
      @brief Converts this section to a formatted string.
      """
      self.model.x_limits
      self.model.z_limits
      return (
#         "[Generator]\n"
         f"   type    = hex_mesh_generator\n"
         f"   n       = {'  '.join(map(str, self.n))}\n"
         f"   l_bound = {'  '.join(map(str, self.x_min))}\n"
         f"   u_bound = {'  '.join(map(str, self.x_max))}\n"
#         "[]\n"
      )

     
@dataclass
class MeshParallelism:
   distribute:       Optional[bool] = True
   ranks_per_part:   Optional[int]  = None
   partitioner:      Optional[str]  = None

   def __str__(self) -> str:
      """
      @brief Converts this section to a formatted string.
      """
      out = []
#      out.append( "   [Parallel]\n")
      if self.distribute:
         out.append(f"      distribute     = {self.distribute}")
      if self.ranks_per_part:
         out.append(f"      ranks_per_part = {self.ranks_per_part}")
      if self.partitioner:
         out.append(f"      partitioner    = {self.partitioner}")
#      out.append( "   []\n")
      return "\n".join(out)


@dataclass
class MeshAdaptor:
   min_epw:          float
   adapt_sources:    Optional[int  ] = None # 2
   adapt_receivers:  Optional[int  ] = None # 0
   jump_tolerance:   Optional[float] = None # 0.2
   jump_factor:      Optional[float] = None # 1.0
   smooth_refs:      Optional[bool ] = None # False
#   adapt_order:      bool  = False
#   f_low:            float = 4.0
#   f_med:            float = 8.0

   def __str__(self) -> str:
      """
      @brief Converts this section to a formatted string.
      """
      out = []
      out.append( "   [Adapt]\n")
      out.append(f"      min_epw = {self.min_epw}\n")
      if self.jump_factor:
         out.append(f"      adapt_sources   = {self.adapt_sources}")
      if self.adapt_receivers:
         out.append(f"      adapt_receivers = {self.adapt_receivers}")
      if self.jump_tol:
         out.append(f"      jump_tolerance  = {self.jump_tolerance}")
      if self.jump_factor:
         out.append(f"      jump_factor     = {self.jump_factor}")
      if self.smooth_refs:
         out.append(f"      smooth_refs     = {self.smooth_refs}")
      out.append( "   []\n")
      return "\n".join(out)


@dataclass
class MeshManager:
   """
   @class MeshManager
   @brief Defines mesh type, dimension, refinement, etc.
   """
   source:        object
   parallel:      Optional[MeshParallelism] = None
   adapt:         Optional[MeshAdaptor]     = None
   
   def set_adapt(self,
                 min_epw:         float,
                 adapt_sources:   Optional[int]   = None,
                 adapt_receivers: Optional[int]   = None,
                 jump_tolerance:  Optional[float] = None,
                 jump_factor:     Optional[float] = None,
                 smooth_refs:     Optional[bool]  = None) -> None:
      """
      @brief   Sets mesh adaptivity options
      @param   min_epw         Minimum # of elements per wavelength.
      @param   adapt_sources   Number of additional refinements near sources
      @param   adapt_receivers Number of additional refinements near receivers
      @param   jump_tolerance  Maximum relative change in wavespeed that consitutes
                               a "jump" in material properties
      @param   jump_factor     Multiplicative factor for min_epw on "jump" elements
      @param   smooth_refs     Do additional refinements to unconstrain element DOFs
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
      """
      @brief   Sets mesh parallel options
      @param   distribute      Fully distribute mesh
      @param   ranks_per_part
      @param   partitioner
      """
      self.parallel = MeshParallelism(distribute     = distribute,
                                      ranks_per_part = ranks_per_part,
                                      partitioner    = partitioner)

   def __str__(self) -> str:
      """
      @brief Converts this section to a formatted string.
      """
      out = "[Mesh]\n"
      out += str(self.source)
      if self.parallel:
         out += str(self.parallel)
      if self.adapt is None:
         self.set_adapt(min_epw = 2.0)
      out += str(self.adapt):
      out += "[]\n\n"
      return mesh_str
