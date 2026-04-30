import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra

__all__ = ["Discretization", "SolverConfig", "SuperPatch", "NumericsManager"]


@dataclass
class Discretization(ExtraFieldsMixin):
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
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        method: str = "DPG",
        order: Union[int, Dict[str, int]] = 3,
        **kwargs,
    ):
        self.method = method
        self.order = order
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, d: Dict[str, Any]) -> "Discretization":
        d = copy.deepcopy(d)
        return cls(
            method=d.pop("method", "DPG"),
            order=d.pop("order", 3),
            **d,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = {
            "method": self.method,
            "order": self.order,
        }
        return merge_extra(payload, self.extra, "Discretization")


@dataclass
class SuperPatch(ExtraFieldsMixin):
    """*** ADVANCED FEATURE ***

    Class grouping elements in a domain into a single patch.

    Super patches use sparse direct or ILU solvers; they are extremely slow
    compared to the default solver (can be >100x slower, even on a single node);
    but they can be useful for very challenging localized features (e.g. fractures).
    The size of the super patch should be smaller than size of the full problem.

    If you are going to use this feature (especially in workflows that will be
    run repeatedly) we recommend contacting FrequenSol for help setting it up
    correctly and optimizing performance.

    Args:
       grid (int): Grid the patch is defined on.
       domain (Union[int, List[int]]): The domain of the patch.
    """

    grid: int = 0
    domain: List[int] = field(default_factory=list)
    warning_acknowledged: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        grid: int,
        domain: Union[int, List[int]],
        warning_acknowledged: bool = False,
        **kwargs,
    ):
        if not warning_acknowledged:
            raise ValueError(
                "Super patches are an advanced feature that can slow down your simulation by > 100x; they "
                "are typically not beneficial in all but the most challenging cases. If you are confident, "
                "simply set warning_acknowledged = True.\n\n "
                "We recommend contacting FrequenSol for help setting this up correctly and optimizing performance."
            )
        if isinstance(domain, int):
            domain = [domain]
        self.grid = grid
        self.domain = domain
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, data: Dict) -> "SuperPatch":
        data = copy.deepcopy(data)
        return cls(
            grid=data.pop("grid"),
            domain=data.pop("domain"),
            **data,
        )

    def to_fs(self, ctx=None) -> Dict:
        payload = {
            "grid": self.grid,
            "domain": self.domain,
        }
        return merge_extra(payload, self.extra, "SuperPatch")


@dataclass
class SolverConfig(ExtraFieldsMixin):
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
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        solve_on: Literal["final", "all"] = "final",
        max_iter: int = 300,
        tolerance: float = 1.0e-4,
        grids: int = 4,
        hp_switch: Optional[int] = None,
        **kwargs,
    ):
        self.solve_on = solve_on
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.grids = grids
        self.hp_switch = hp_switch
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, data: Dict) -> "SolverConfig":
        data = copy.deepcopy(data)
        ngrids = data.pop("grids", 4)
        obj = cls(
            solve_on=data.pop("solve_on", "final"),
            max_iter=data.pop("max_iter", 300),
            tolerance=data.pop("tolerance", 1.0e-4),
            grids=ngrids,
            hp_switch=data.pop("hp_switch", ngrids),
        )
        obj._init_extra(data)
        return obj

    def __iadd__(self, other: SuperPatch) -> "SolverConfig":
        if isinstance(other, SuperPatch):
            if "super_patches" not in self.extra:
                self.extra["super_patches"] = []
            self.extra["super_patches"].append(other.to_fs())
        return self

    def to_fs(self, ctx=None) -> Dict:
        hp_switch = self.hp_switch if self.hp_switch is not None else self.grids
        dict = {
            "solve_on": self.solve_on,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "grids": self.grids,
            "hp_switch": hp_switch,
        }
        return merge_extra(dict, self.extra, "SolverConfig")


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
    def from_fs(cls, data: Dict) -> "NumericsManager":
        data = copy.deepcopy(data)
        return cls(
            solver=SolverConfig.from_fs(data["solver"]),
            discretization=Discretization.from_fs(data["discretization"]),
        )

    def to_fs(self, ctx=None) -> Dict:
        return {
            "solver": self.solver.to_fs(ctx),
            "discretization": self.discretization.to_fs(ctx),
        }
