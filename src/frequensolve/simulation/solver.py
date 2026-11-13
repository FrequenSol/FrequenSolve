"""Linear/nonlinear solver settings for simulations."""

import copy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from frequensolve.util.mixins import ExportContext, ExtraFieldsMixin, merge_extra

__all__ = ["SolverConfig", "SuperPatch"]


def _serialize_solver_extra(
    extra: Dict[str, Any],
    ctx: Optional[ExportContext] = None,
) -> Dict[str, Any]:
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
        **kwargs: Any,
    ) -> None:
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
    def from_fs(cls, data: Mapping[str, Any]) -> "SuperPatch":
        """Deserialize a super-patch solver payload.

        Args:
            data: Serialized super-patch mapping.

        Returns:
            ``SuperPatch`` instance.
        """

        payload = copy.deepcopy(dict(data))
        return cls(
            grid=payload.pop("grid"),
            domain=payload.pop("domain"),
            warning_acknowledged=payload.pop("warning_acknowledged", True),
            **payload,
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
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
        tolerance: Solver convergence tolerance. None uses the backend default
            (1e-3 for transient solves, 1e-4 for frequency-domain solves).
        precision: Floating-point precision used by the solver executable.
        cycle: "auto" preserves the solver preset; "V", "W", and "log" select
            a traversal. A mapping supplies a custom visits/anchor/coarse schedule.
        pre_smooth: Number of smoothing sweeps before coarse descent; zero skips it.
        post_smooth: Number of smoothing sweeps after coarse correction; zero skips it.
        **kwargs: Additional solver-facing configuration fields.
    """

    solve_on: Literal["final", "all"] = "final"
    max_iter: int = 300
    tolerance: Optional[float] = None
    precision: Literal["single", "double"] = "single"
    cycle: Union[str, int, Mapping[str, Any]] = "auto"
    pre_smooth: int = 1
    post_smooth: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        solve_on: Literal["final", "all"] = "final",
        max_iter: int = 300,
        tolerance: Optional[float] = None,
        precision: Literal["single", "double"] = "single",
        *,
        cycle: Union[str, int, Mapping[str, Any]] = "auto",
        pre_smooth: int = 1,
        post_smooth: int = 1,
        **kwargs: Any,
    ) -> None:
        """Create iterative solver settings."""

        if precision not in {"single", "double"}:
            raise ValueError("Solver precision must be 'single' or 'double'")
        self.solve_on = solve_on
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.precision = precision
        self.cycle = copy.deepcopy(dict(cycle)) if isinstance(cycle, Mapping) else cycle
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self._multigrid_fields()
        self._init_extra(None, **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SolverConfig":
        """Deserialize solver settings from solver JSON.

        Args:
            data: Serialized solver configuration mapping.

        Returns:
            ``SolverConfig`` instance.
        """

        payload = copy.deepcopy(dict(data))
        obj = cls(
            solve_on=payload.pop("solve_on", "final"),
            max_iter=payload.pop("max_iter", 300),
            tolerance=payload.pop("tolerance", None),
            precision=payload.pop("precision", "single"),
            cycle=payload.pop("cycle", "auto"),
            pre_smooth=payload.pop("pre_smooth", 1),
            post_smooth=payload.pop("post_smooth", 1),
        )
        obj._init_extra(payload)
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

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize solver settings for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible solver configuration.
        """

        payload: Dict[str, Any] = {
            "solve_on": self.solve_on,
            "max_iter": self.max_iter,
            "precision": self.precision,
        }
        payload.update(self._multigrid_fields())
        if self.tolerance is not None:
            payload["tolerance"] = self.tolerance
        return merge_extra(
            payload,
            _serialize_solver_extra(self.extra, ctx),
            "SolverConfig",
        )

    def _multigrid_fields(self) -> Dict[str, Any]:
        """Validate and serialize cycle controls, leaving automatic defaults unset."""
        payload: Dict[str, Any] = {}
        for name in ("pre_smooth", "post_smooth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            if value != 1:
                payload[name] = int(value)
        cycle = self.cycle
        if isinstance(cycle, str):
            if cycle.lower() not in {"auto", "v", "w", "log"}:
                raise ValueError(
                    "cycle must be auto, V, W, log, a visit count or a schedule"
                )
            if cycle.lower() != "auto":
                payload["cycle"] = (
                    cycle.upper() if cycle.lower() in {"v", "w"} else "log"
                )
        elif (
            isinstance(cycle, Integral)
            and not isinstance(cycle, bool)
            and int(cycle) > 0
        ):
            payload["cycle"] = int(cycle)
        elif isinstance(cycle, Mapping):
            schedule = copy.deepcopy(dict(cycle))
            if set(schedule) - {"visits", "anchor", "coarse"}:
                raise ValueError(
                    "cycle schedule accepts only visits, anchor and coarse"
                )
            if schedule.get("anchor", "fine") not in {"fine", "coarse"}:
                raise ValueError("cycle anchor must be fine or coarse")
            visits = schedule.get("visits", [1])
            if not isinstance(visits, (list, tuple)) or not visits:
                raise ValueError(
                    "cycle visits must be a nonempty sequence of positive integers"
                )
            for value in [*visits, schedule.get("coarse", 1)]:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or value < 1
                ):
                    raise ValueError(
                        "cycle visits and coarse must be positive integers"
                    )
            if "visits" in schedule:
                schedule["visits"] = [int(value) for value in visits]
            if "coarse" in schedule:
                schedule["coarse"] = int(schedule["coarse"])
            payload["cycle"] = schedule
        else:
            raise ValueError(
                "cycle must be auto, V, W, log, a visit count or a schedule"
            )
        return payload
