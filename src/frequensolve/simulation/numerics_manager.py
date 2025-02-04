from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

__all__ = ["Discretization", "SolverConfig", "NumericsManager"]


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
    order: int = 3
    DPG_alpha: float = 1.0
    DPG_enrich: int = 1
    DPG_penalty: float = 100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Discretization":
        return cls(
            method=d.get("method", "DPG"),
            order=d.get("order", 3),
            DPG_alpha=d.get("DPG_alpha", 1.0),
            DPG_enrich=d.get("DPG_enrich", 1),
            DPG_penalty=d.get("DPG_penalty", 100),
        )

    def __dict__(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolverConfig:
    """
    Defines solver configuration.

    Attributes:
       solve_on (Literal["final", "all"]): Whether to solve on the final or all time steps.
       max_iter (int): The maximum number of iterations.
       tolerance (float): The tolerance for the solver.
       n_grids (Optional[int]): The number of grids to use.
    """

    solve_on: Literal["final", "all"] = "final"
    max_iter: int = 300
    tolerance: float = 1.0e-5
    n_grids: int = 4
    refinement_kind: Literal["uniform", "adapt_indicator", "adapt_wavespeed"] = (
        "adapt_wavespeed"
    )
    refinement_flags: List[Literal["h", "p"]] = field(default_factory=list)

    # Private (advanced) options, these are not needed in all but the most exotic cases.
    # Setting these will only enable advanced options if a valid advanced user key is provided.
    _use_advanced: bool = False
    _advanced_user_key: Optional[str] = None
    _advanced_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "SolverConfig":
        return cls(
            solve_on=data.get("solve_on", "final"),
            max_iter=data.get("max_iter", 300),
            tolerance=data.get("tolerance", 1.0e-5),
            n_grids=data.get("n_grids", 4),
            refinement_kind=data.get("refinement_kind", "adapt_wavespeed"),
            refinement_flags=data.get("refinement_flags", []),
        )

    def __dict__(self) -> Dict:
        return {
            "solve_on": self.solve_on,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "n_grids": self.n_grids,
            "refinement_kind": self.refinement_kind,
            "refinement_flags": self.refinement_flags,
            **(
                {"advanced_user_key": self._advanced_user_key, **self._advanced_options}
                if self._use_advanced
                else {}
            ),
        }


@dataclass
class NumericsManager:
    """Container for numerical configuration.

    Attributes:
       solver (SolverConfig): The solver configuration.
       discretization (Discretization): The discretization configuration.
    """

    solver: SolverConfig = field(default_factory=SolverConfig)
    discretization: Discretization = field(default_factory=Discretization)

    @classmethod
    def from_dict(cls, data: Dict) -> "NumericsManager":
        return cls(
            solver=SolverConfig.from_dict(data["solver"]),
            discretization=Discretization.from_dict(data["discretization"]),
        )

    def __dict__(self) -> Dict:
        return asdict(self)
