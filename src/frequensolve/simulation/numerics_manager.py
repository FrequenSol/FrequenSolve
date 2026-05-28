"""Numerical discretization and solver-configuration objects."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union

from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra

__all__ = ["Discretization", "SolverConfig", "SuperPatch", "NumericsManager"]


@dataclass
class Discretization(ExtraFieldsMixin):
    """Class representing discretization parameters.

    Args:
       method (str): The discretization method, defaults to "DPG".
       DPG_alpha (float): The DPG stabilization parameter, defaults to 1.0.
       DPG_enrich (int): The DPG enrichment parameter, defaults to 1.
       DPG_penalty (float): The DPG penalty parameter (for enforcing constraints like continuity), defaults to 100.
    """

    method: str = "DPG"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        method: str = "DPG",
        **kwargs,
    ):
        """Create discretization settings.

        Solver order is now configured through mesh adaptivity, so legacy
        ``order`` values are rejected with an actionable error.
        """

        if "order" in kwargs:
            raise ValueError(
                "'order' has moved from Discretization to mesh adaptivity; "
                "use mesh.set_adapt(..., order=...) instead."
            )
        self.method = method
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, d: Dict[str, Any]) -> "Discretization":
        """Deserialize discretization settings from a solver payload."""

        d = copy.deepcopy(d)
        if "order" in d:
            raise ValueError(
                "'order' has moved from Discretization to mesh adaptivity; "
                "use Mesh.adapt.order instead."
            )
        return cls(
            method=d.pop("method", "DPG"),
            **d,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize discretization settings to the solver payload."""

        payload = {
            "method": self.method,
        }
        if "order" in self.extra:
            raise ValueError(
                "'order' has moved from Discretization to mesh adaptivity; "
                "use mesh.set_adapt(..., order=...) instead."
            )
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
        """Create an advanced super-patch definition.

        ``warning_acknowledged`` must be set explicitly to avoid accidental use
        of a very expensive solver mode.
        """

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
        """Deserialize a super-patch definition."""

        data = copy.deepcopy(data)
        return cls(
            grid=data.pop("grid"),
            domain=data.pop("domain"),
            **data,
        )

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this super-patch definition."""

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
       grids (int): The number of grids to use.
    """

    solve_on: Literal["final", "all"] = "final"
    max_iter: int = 300
    tolerance: float = 1.0e-4
    grids: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        solve_on: Literal["final", "all"] = "final",
        max_iter: int = 300,
        tolerance: float = 1.0e-4,
        grids: int = 3,
        **kwargs,
    ):
        """Create iterative solver settings."""

        self.solve_on = solve_on
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.grids = grids
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, data: Dict) -> "SolverConfig":
        """Deserialize solver settings from a solver payload."""

        data = copy.deepcopy(data)
        obj = cls(
            solve_on=data.pop("solve_on", "final"),
            max_iter=data.pop("max_iter", 300),
            tolerance=data.pop("tolerance", 1.0e-4),
            grids=data.pop("grids", 3),
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
        """Serialize solver settings to the solver payload."""

        payload = {
            "solve_on": self.solve_on,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "grids": self.grids,
        }
        return merge_extra(payload, self.extra, "SolverConfig")


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
        """Deserialize numerical configuration from a solver payload."""

        data = copy.deepcopy(data)
        return cls(
            solver=SolverConfig.from_fs(data["solver"]),
            discretization=Discretization.from_fs(data["discretization"]),
        )

    def to_fs(self, ctx=None) -> Dict:
        """Serialize numerical configuration to the solver payload."""

        return {
            "solver": self.solver.to_fs(ctx),
            "discretization": self.discretization.to_fs(ctx),
        }
