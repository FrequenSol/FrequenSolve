from dataclasses import dataclass, field
from typing    import List, Dict, Optional, Union, Literal, Any

__all__ = ['Discretization', 'SolverConfig', 'NumericsManager']

@dataclass 
class Discretization:
   """Class representing discretization parameters.
   
   Args:
      method (str): The discretization method, defaults to "DPG".
      DPG_alpha (float): The DPG stabilization parameter, defaults to 1.0.
      DPG_enrich (int): The DPG enrichment parameter, defaults to 1.
      DPG_penalty (float): The DPG penalty parameter (for enforcing constraints like continuity), defaults to 100.
   """
   method: str = "DPG"
   DPG_alpha: float = 1.0 
   DPG_enrich: int = 1
   DPG_penalty: float = 100

   def to_dict(self) -> Dict[str, Any]:
      """Convert discretization parameters to dictionary.
      
      Returns:
         Dict[str, Any]: Dictionary of parameters.
      """
      return {
         "method": self.method,
         "DPG_alpha": self.DPG_alpha,
         "DPG_enrich": self.DPG_enrich,
         "DPG_penalty": self.DPG_penalty
      }

   @classmethod
   def from_dict(cls, d: Dict[str, Any]) -> "Discretization":
      """Create Discretization from dictionary.
      
      Args:
         d (Dict[str, Any]): Dictionary of parameters.
         
      Returns:
         Discretization: New Discretization instance.
      """
      return cls(
         method=d.get("method", "DPG"),
         DPG_alpha=d.get("DPG_alpha", 1.0),
         DPG_enrich=d.get("DPG_enrich", 1),
         DPG_penalty=d.get("DPG_penalty", 100)
      )

   def __str__(self) -> str:
      """String representation of discretization parameters.
      
      Returns:
         str: Formatted string of parameters.
      """
      return (f"[Discretization]\n"
              f"   method           = {self.method}\n"
              f"   DPG_alpha        = {self.DPG_alpha}\n" 
              f"   DPG_enrich       = {self.DPG_enrich}\n"
              f"   DPG_penalty      = {self.DPG_penalty}\n"
              f"[]\n\n")



@dataclass
class SolverConfig:
   """
   Defines solver configuration for the simulation.

   Attributes:
      solve_on (Literal["final", "all"]): Whether to solve on the final or all time steps.
      max_iter (int): The maximum number of iterations.
      tolerance (float): The tolerance for the solver.
      n_grids (Optional[int]): The number of grids to use.   
   """
   solve_on:  Literal["final", "all"] = "final"
   max_iter:  int   = 300
   tolerance: float = 1.e-5
   n_grids:   int   = 4
   refinement_kind:  Literal["uniform", "adapt_indicator", "adapt_wavespeed"] = "adapt_wavespeed"
   refinement_flags:  List[Literal["h", "p"]] = field(default_factory=list)

   # Private (advanced) options, these are not needed in all but the most exotic cases.
   # Setting these will only enable advanced options if a valid advanced user key is provided.
   _use_advanced:          bool = False
   _advanced_user_key:     Optional[str] = None
   _advanced_options:      Dict[str, Any] = field(default_factory=dict)

   @classmethod
   def from_dict(cls, data: Dict) -> "SolverConfig":
      """Creates a SolverConfig instance from a dictionary.

      Args:
         data: Dictionary containing solver configuration data.

      Returns:
         A new SolverConfig instance initialized with the dictionary data.
      """

      return cls(
         solve_on         = data.get("solve_on", "final"),
         max_iter         = data.get("max_iter", 300),
         tolerance        = data.get("tolerance", 1.e-5),
         n_grids          = data.get("n_grids", 4),
         refinement_kind  = data.get("refinement_kind", "adapt_wavespeed"),
         refinement_flags = data.get("refinement_flags", []),
      )


   def to_dict(self) -> Dict:
      """Converts the solver configuration to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the solver configuration data.
      """
      return {
         "solve_on": self.solve_on,
         "max_iter": self.max_iter,
         "tolerance": self.tolerance,
         "n_grids": self.n_grids,
         "refinement_kind": self.refinement_kind,
         "refinement_flags": self.refinement_flags,
         **({"advanced_user_key": self._advanced_user_key,
            **self._advanced_options} if self._use_advanced else {})
      }

   def __str__(self) -> str:
      """Returns a string representation of the solver configuration.

      Returns:
         str: String representation of the solver configuration.
      """
      base_str = (
         "[Solver]\n"
         f"   solve_on         = {self.solve_on}\n"
         f"   max_iter         = {self.max_iter}\n" 
         f"   tolerance        = {self.tolerance}\n"
         f"   n_grids          = {self.n_grids}\n"
         f"   refinement_kind  = {self.refinement_kind}\n"
         f"   refinement_flags = {self.refinement_flags}\n"
         f"[]\n\n"
      )
      return base_str


@dataclass
class NumericsManager:
   """Container for numerical configuration.
   
   Attributes:
      solver (SolverConfig): The solver configuration.
      discretization (Discretization): The discretization configuration.
   """
   solver:         SolverConfig   = field(default_factory=SolverConfig)
   discretization: Discretization = field(default_factory=Discretization)
   
   @classmethod
   def from_dict(cls, data: Dict) -> "NumericsManager":
      """Create a NumericsManager from a dictionary.
      
      Args:
         data (Dict): Dictionary containing numerics configuration.
         
      Returns:
         NumericsManager: A new NumericsManager instance.
      """
      return cls(
         solver = SolverConfig.from_dict(data["solver"]),
         discretization = Discretization.from_dict(data["discretization"])
      )
      
   def to_dict(self) -> Dict:
      """Convert the numerics configuration to a dictionary.
      
      Returns:
         Dict: Dictionary containing the numerics configuration.
      """
      return {
         "solver": self.solver.to_dict(),
         "discretization": self.discretization.to_dict()
      }
      
   def __str__(self) -> str:
      """Get string representation of numerics configuration.
      
      Returns:
         str: String representation of the configuration.
      """
      return (f"[Numerics]\n"
              f"{str(self.solver)}\n"
              f"{str(self.discretization)}\n"
              f"[]\n\n")


