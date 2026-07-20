"""Linear/nonlinear solver settings for simulations."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union

from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra

__all__ = ["SolverConfig", "SuperPatch"]


def _serialize_solver_extra(extra: Dict[str, Any], ctx=None) -> Dict[str, Any]:
    from frequensolve.mesh.mesh_manager import (
        _serialize_adapt_value,
        _serialize_hp_payload,
    )

    payload = {}
    for key, value in extra.items():
        if key == "hp":
            payload[key] = _serialize_hp_payload(value, ctx)
        else:
            payload[key] = _serialize_adapt_value(value, ctx)
    return payload


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
        self.warning_acknowledged = bool(warning_acknowledged)
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, data: Dict) -> "SuperPatch":
        """Deserialize a super-patch solver payload.

        Args:
            data: Serialized super-patch mapping.

        Returns:
            ``SuperPatch`` instance.
        """

        data = copy.deepcopy(data)
        return cls(
            grid=data.pop("grid"),
            domain=data.pop("domain"),
            warning_acknowledged=data.pop("warning_acknowledged", True),
            **data,
        )

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this super patch for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible super-patch payload.
        """

        payload = {
            "grid": self.grid,
            "domain": self.domain,
        }
        return merge_extra(payload, self.extra, "SuperPatch")


@dataclass
class SolverConfig(ExtraFieldsMixin):
    """Linear/nonlinear solver settings for one simulation.

    Args:
        solve_on: Whether to solve only on the final adaptive mesh or on all
            adaptive steps.
        max_iter: Maximum Krylov/nonlinear iterations.
        tolerance: Solver convergence tolerance.
        precision: Floating-point precision used by the solver executable.
        **kwargs: Additional solver-facing configuration fields.
    """

    solve_on: Literal["final", "all"] = "final"
    max_iter: int = 300
    tolerance: float = 1.0e-4
    precision: Literal["single", "double"] = "single"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        solve_on: Literal["final", "all"] = "final",
        max_iter: int = 300,
        tolerance: float = 1.0e-4,
        precision: Literal["single", "double"] = "single",
        **kwargs,
    ):
        """Create iterative solver settings."""

        if precision not in {"single", "double"}:
            raise ValueError("Solver precision must be 'single' or 'double'")
        self.solve_on = solve_on
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.precision = precision
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, data: Dict) -> "SolverConfig":
        """Deserialize solver settings from solver JSON.

        Args:
            data: Serialized solver configuration mapping.

        Returns:
            ``SolverConfig`` instance.
        """

        data = copy.deepcopy(data)
        obj = cls(
            solve_on=data.pop("solve_on", "final"),
            max_iter=data.pop("max_iter", 300),
            tolerance=data.pop("tolerance", 1.0e-4),
            precision=data.pop("precision", "single"),
        )
        obj._init_extra(data)
        return obj

    def __iadd__(self, other: SuperPatch) -> "SolverConfig":
        """Append an advanced super patch to the solver configuration.

        Args:
            other: Super patch to add.

        Returns:
            This solver configuration.
        """

        if isinstance(other, SuperPatch):
            if "super_patches" not in self.extra:
                self.extra["super_patches"] = []
            self.extra["super_patches"].append(other.to_fs())
        return self

    def to_fs(self, ctx=None) -> Dict:
        """Serialize solver settings for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible solver configuration.
        """

        payload = {
            "solve_on": self.solve_on,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "precision": self.precision,
        }
        return merge_extra(
            payload,
            _serialize_solver_extra(self.extra, ctx),
            "SolverConfig",
        )
