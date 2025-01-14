from typing    import List, Dict, Optional, Union, Literal
from pydantic  import BaseModel, Field

class Solver(BaseModel):
    """
    @class Solver
    @brief Defines solver options for the simulation.
    """
    solve_on:     Literal["final", "all"] = "final"
    max_iter:     int           = 500
    tolerance:    float         = 1.e-5
    n_grids:      Optional[int] = None

    def __str__(self) -> str:
        """
        @brief Converts the Solver section to a formatted string.
        """
        return (
            "[Solver]\n"
            f"   solve_on  = {self.solve_on}\n"
            f"   max_iter  = {self.max_iter}\n"
            f"   tolerance = {self.tolerance}\n"
            f"   n_grids   = {self.n_grids}\n"
            "[]\n"
        )
