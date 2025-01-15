from dataclasses import dataclass, field
from typing    import List, Dict, Optional, Union, Literal

@dataclass
class Solver:
   """
   Defines solver options for the simulation.

   Attributes:
      solve_on (Literal["final", "all"]): Whether to solve on the final or all time steps.
      max_iter (int): The maximum number of iterations.
      tolerance (float): The tolerance for the solver.
      n_grids (Optional[int]): The number of grids to use.   
   """
   solve_on:     Literal["final", "all"] = "final"
   max_iter:     int           = 500
   tolerance:    float         = 1.e-5
   n_grids:      Optional[int] = None

   def __str__(self) -> str:
      return (
         "[Solver]\n"
         f"   solve_on  = {self.solve_on}\n"
         f"   max_iter  = {self.max_iter}\n"
         f"   tolerance = {self.tolerance}\n"
         f"   n_grids   = {self.n_grids}\n"
         "[]\n"
      )
