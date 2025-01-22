import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Generator

__all__ = ['Grid','CartesianGrid']


@dataclass
class Grid(ABC):
   """Base class for all grid types."""

   @abstractmethod
   def get_coords(self, slices: Optional[List[slice]] = None) -> np.ndarray:
      """Gets coordinates for all grid points or a subset defined by slices."""
      pass

   @abstractmethod
   def generate_coords(self, slices: Optional[List[slice]] = None) -> Generator[np.ndarray, None, None]:
      """Generates coordinates for all grid points or a subset defined by slices.
      
      Args:
         slices: Optional list of slice objects defining index ranges for each dimension.
            If None, generates coordinates for all points. Length must match grid dimensions.
            
      Yields:
         np.ndarray: Coordinates for each grid point.
      """
      pass
   @abstractmethod
   def to_dict(self) -> Dict:
      """Converts the grid to a dictionary representation."""
      pass

   @classmethod
   @abstractmethod
   def from_dict(cls, data: Dict) -> "Grid":
      """Creates a Grid instance from a dictionary."""
      pass


# ----------------------------------------------------------------------
# Cartesian Grid
# ----------------------------------------------------------------------
@dataclass
class CartesianGrid(Grid):
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
   n:  List[int]   = field(default_factory=list)
   x0: List[float] = field(default_factory=list)
   x1: List[float] = field(default_factory=list)
   dx: List[float] = field(default_factory=list)


   def __post_init__(self):
      if len(self.x1) == 0:
         self.x1 = [x0 + (n - 1) * dx for x0, n, dx in zip(self.x0, self.n, self.dx)]
      elif len(self.dx) == 0:
         self.dx = [(x1 - x0) / (n - 1) for x0, x1, n in zip(self.x0, self.x1, self.n)]
      elif len(self.n) == 0:
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
      

   @property
   def dimension(self) -> int:
      return len(self.n)
   

   def generate_coords(self, slices: Optional[List[slice]] = None):
      """Generates coordinates for all grid points or a subset defined by slices.
      
      Args:
         slices (Optional[List[slice]]): List of slice objects defining index ranges for each dimension.
            If None, yields coordinates for all points. Length must match grid dimensions.
      
      Yields:
         Generator[float]: Coordinate point for each grid point.
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
      
   def to_dict(self) -> Dict:
      """Converts the grid to a dictionary representation.

      Returns:
         Dict: Dictionary containing the grid parameters.
      """
      return {
         "n": self.n,
         "x0": self.x0,
         "x1": self.x1,
         "dx": self.dx
      }

   @classmethod 
   def from_dict(cls, data: Dict) -> "CartesianGrid":
      """Creates a CartesianGrid instance from a dictionary.

      Args:
         data (Dict): Dictionary containing the grid parameters.

      Returns:
         CartesianGrid: A new CartesianGrid instance.
      """
      return cls(
         n  = data["n"],
         x0 = data["x0"],
         x1 = data["x1"],
         dx = data["dx"]
      )