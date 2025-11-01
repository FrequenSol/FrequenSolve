from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

__all__ = ["Discretization", "SolverConfig", "NumericsManager"]


@dataclass
class Discretization:
    """Class representing discretization parameters.

    Args:
       method (str): The discretization method, defaults to "DPG".
       order (Union[int, Dict[str, int]]): The discretization order, defaults to 3.
       DPG_alpha (float): The DPG stabilization parameter, defaults to 1.0.
       DPG_enrich (int): The DPG enrichment parameter, defaults to 1.
       DPG_penalty (float): The DPG penalty parameter (for enforcing constraints like continuity), defaults to 100.
    """

    method: str = "DPG"
    order: Union[int, Dict[str, int]] = 3
    DPG_alpha: float = 1.0
    DPG_enrich: int = 0
    DPG_penalty: float = 100.0
    misc: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        method: str = "DPG",
        order: Union[int, Dict[str, int]] = 3,
        DPG_alpha: float = 1.0,
        DPG_enrich: int = 0,
        DPG_penalty: float = 100.0,
        **kwargs,
    ):
        self.method = method
        self.order = order
        self.DPG_alpha = DPG_alpha
        self.DPG_enrich = DPG_enrich
        self.DPG_penalty = DPG_penalty
        self.misc = kwargs

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Discretization":
        return cls(
            method=d.pop("method", "DPG"),
            order=d.pop("order", 3),
            DPG_alpha=d.pop("DPG_alpha", 1.0),
            DPG_enrich=d.pop("DPG_enrich", 0),
            DPG_penalty=d.pop("DPG_penalty", 100),
            misc=d,
        )

    def __dict__(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "order": self.order,
            "DPG_alpha": self.DPG_alpha,
            "DPG_enrich": self.DPG_enrich,
            "DPG_penalty": self.DPG_penalty,
            **self.misc,
        }


@dataclass
class SolverConfig:
    """
    Defines solver configuration.

    Attributes:
       solve_on (Literal["final", "all"]): Whether to solve on the final or all time steps.
       max_iter (int): The maximum number of iterations.
       tolerance (float): The tolerance for the solver.
       grids (Optional[int]): The number of grids to use.
    """

    solve_on: Literal["final", "all"] = "final"
    max_iter: int = 300
    tolerance: float = 1.0e-4
    grids: int = 4
    hp_switch: Optional[int] = None
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
        ngrids = data.get("grids", 4)
        return cls(
            solve_on=data.get("solve_on", "final"),
            max_iter=data.get("max_iter", 300),
            tolerance=data.get("tolerance", 1.0e-4),
            grids=ngrids,
            hp_switch=data.get("hp_switch", ngrids),
            refinement_kind=data.get("refinement_kind", "adapt_wavespeed"),
            refinement_flags=data.get("refinement_flags", []),
        )

    def __dict__(self) -> Dict:
        hp_switch = self.hp_switch if self.hp_switch is not None else self.grids
        return {
            "solve_on": self.solve_on,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "grids": self.grids,
            "hp_switch": hp_switch,
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
