import numpy as np

from dataclasses import dataclass, field
from typing import Optional, List

__all__ = ['CartesianGrid']


# ----------------------------------------------------------------------
# Cartesian Grid
# ----------------------------------------------------------------------
@dataclass
class CartesianGrid:
   """A uniform Cartesian grid for defining receiver locations.
   
   This class represents a uniform grid in 2D or 3D space, defined by the number of points,
   starting coordinates, ending coordinates, and/or grid spacing in each dimension.
   Only two of n, dx, and x1 need to be specified - the third will be calculated.
   
   Attributes:
      n  (List[int]):   Number of points in each dimension.
      x0 (List[float]): Starting coordinates in each dimension.
      x1 (List[float]): Ending coordinates in each dimension.
      dx (List[float]): Grid spacing in each dimension.
   """
   n:  Optional[List[int]]   = field(default_factory=list)
   x0: List[float]           = field(default_factory=list)
   x1: Optional[List[float]] = field(default_factory=list)
   dx: Optional[List[float]] = field(default_factory=list)

   def __post_init__(self):
      if self.x1 is None:
         self.x1 = [x0 + (n - 1) * dx for x0, n, dx in zip(self.x0, self.n, self.dx)]
      elif self.dx is None:
         self.dx = [(x1 - x0) / (n - 1) for x0, x1, n in zip(self.x0, self.x1, self.n)]
      elif self.n is None:
         self.n = [int((x1 - x0) / dx + 1) for x0, x1, dx in zip(self.x0, self.x1, self.dx)]

   def get_coords(self, slices: Optional[List[slice]] = None) -> np.ndarray:
      """Gets coordinates for all grid points or a subset defined by slices.
      
      Args:
         slices (Optional[List[slice]]): List of slice objects defining index ranges for each dimension.
            If None, returns coordinates for all points. Length must match grid dimensions.
      
      Returns:
         np.ndarray: Array of coordinate points [x,y] or [x,y,z] for each grid point.
            Returns empty array if grid dimensions are not 2 or 3.
      """
      if len(self.n) == 1:
         return np.array([np.linspace(self.x0[0], self.x1[0], self.n[0])])
      
      elif len(self.n) == 2:
         x = np.linspace(self.x0[0], self.x1[0], self.n[0])
         y = np.linspace(self.x0[1], self.x1[1], self.n[1])
         if slices is not None:
            x = x[slices[0]]
            y = y[slices[1]]
         return np.array([[x_, y_] for x_ in x for y_ in y])
      
      elif len(self.n) == 3:
         x = np.linspace(self.x0[0], self.x1[0], self.n[0])
         y = np.linspace(self.x0[1], self.x1[1], self.n[1])
         z = np.linspace(self.x0[2], self.x1[2], self.n[2])
         if slices is not None:
            x = x[slices[0]]
            y = y[slices[1]] 
            z = z[slices[2]]
         return np.array([[x_, y_, z_] for x_ in x for y_ in y for z_ in z])
      else:
         raise ValueError("Grid must have 1, 2, or 3 dimensions")

   def generate_coords(self, slices: Optional[List[slice]] = None):
      """Generates coordinates for all grid points or a subset defined by slices.
      
      Args:
         slices (Optional[List[slice]]): List of slice objects defining index ranges for each dimension.
            If None, yields coordinates for all points. Length must match grid dimensions.
      
      Yields:
         List[float]: Coordinate point [x,y] or [x,y,z] for each grid point.
      """
      if len(self.n) == 1:
         x = np.linspace(self.x0[0], self.x1[0], self.n[0])
         if slices is not None:
            x = x[slices[0]]
         for x_ in x:
            yield [x_]
            
      elif len(self.n) == 2:
         x = np.linspace(self.x0[0], self.x1[0], self.n[0])
         y = np.linspace(self.x0[1], self.x1[1], self.n[1])
         if slices is not None:
            x = x[slices[0]]
            y = y[slices[1]]
         for x_ in x:
            for y_ in y:
               yield [x_, y_]
               
      elif len(self.n) == 3:
         x = np.linspace(self.x0[0], self.x1[0], self.n[0])
         y = np.linspace(self.x0[1], self.x1[1], self.n[1])
         z = np.linspace(self.x0[2], self.x1[2], self.n[2])
         if slices is not None:
            x = x[slices[0]]
            y = y[slices[1]]
            z = z[slices[2]]
         for x_ in x:
            for y_ in y:
               for z_ in z:
                  yield [x_, y_, z_]
      else:
         raise ValueError("Grid must have 1, 2, or 3 dimensions")
      
   def __str__(self) -> str:
      n  = " ".join(map(str, self.n ))
      x0 = " ".join(map(str, self.x0))
      x1 = " ".join(map(str, self.x1))
      dx = " ".join(map(str, self.dx))
      return (
         f"   [Grid]\n"
         f"      n  = {n}\n"
         f"      x0 = {x0}\n"
         f"      x1 = {x1}\n"
         f"      dx = {dx}\n"
         f"   []\n"
      )