"""Bounded line-search optimizers for real-valued models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar

import numpy as np

__all__ = [
    "InexactNewtonIteration",
    "InexactNewtonOptions",
    "InexactNewtonResult",
    "LBFGSOptions",
    "LBFGSRestart",
    "minimize_inexact_newton",
    "minimize_lbfgs",
    "minimize_proximal_gradient",
]


Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]
HessianProduct = Callable[[np.ndarray, np.ndarray], np.ndarray]
Preconditioner = Callable[[np.ndarray, np.ndarray], np.ndarray]
StepLimit = Callable[[np.ndarray, np.ndarray], float]
StepTransform = Callable[[np.ndarray, np.ndarray], np.ndarray]
IterationCallback = Callable[["InexactNewtonIteration"], None]


def _finite_vector(value: Any, *, name: str, size: Optional[int] = None) -> np.ndarray:
    """Return a finite one-dimensional float64 vector."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size < 1 or (size is not None and array.size != size):
        expected = "non-empty" if size is None else f"size {size}"
        raise ValueError(f"{name} must be a {expected} vector")
    if not _all_finite(array):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=True)


def _restart_vector(value: Any, name: str, size: int) -> np.ndarray:
    """Adopt a read-only float64 restart vector by reference; copy anything else once."""

    if (
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.shape == (size,)
        and not value.flags.writeable
    ):
        if not _all_finite(value):
            raise ValueError(f"{name} must contain only finite values")
        return value
    vector = _finite_vector(value, name=name, size=size)
    vector.flags.writeable = False
    return vector


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Validate one finite positive optimizer option."""

    normalized = float(value)
    valid = normalized >= 0.0 if allow_zero else normalized > 0.0
    if not np.isfinite(normalized) or not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return normalized


def _validate_stopping_options(options: Any) -> None:
    """Validate optional tolerances and the fixed restart reference."""
    for name in (
        "gradient_tolerance",
        "objective_tolerance",
        "grad_rel_tole",
        "obj_rel_tol",
    ):
        value = getattr(options, name)
        if value is not None:
            object.__setattr__(options, name, _positive(value, name, allow_zero=True))
    if options.gradient_tolerance is not None and options.grad_abs_tol != 0:
        raise ValueError("Use grad_abs_tol or gradient_tolerance, not both")
    if options.objective_tolerance is not None and options.obj_rel_tol is not None:
        raise ValueError("Use obj_rel_tol or objective_tolerance, not both")
    if options.initial_objective is not None:
        value = float(options.initial_objective)
        if not np.isfinite(value):
            raise ValueError("initial_objective must be finite")
        object.__setattr__(options, "initial_objective", value)


class _StoppingCriteria:
    """Fixed initial-objective thresholds and an absolute improvement average."""

    def __init__(self, options: Any, initial_value: float):
        self.reference = abs(
            initial_value
            if options.initial_objective is None
            else options.initial_objective
        )
        gradient_atol = (
            options.grad_abs_tol
            if options.gradient_tolerance is None
            else options.gradient_tolerance
        )
        gradient_rtol = options.grad_rel_tole
        if gradient_rtol is None:
            gradient_rtol = 1e-6 if options.gradient_tolerance is None else 0.0
        objective_rtol = options.obj_rel_tol
        if objective_rtol is None:
            objective_rtol = (
                1e-9
                if options.objective_tolerance is None
                else options.objective_tolerance
            )
        self.gradient_threshold = gradient_atol + gradient_rtol * self.reference
        self.objective_threshold = options.obj_abs_tol + objective_rtol * self.reference
        self.momentum = options.objective_tolerance_momentum
        self.improvement: Optional[float] = None

    def relative(self, value: float) -> float:
        """Report relative progress without imposing a unit-dependent floor."""
        if self.reference == 0:
            return 0.0 if value == 0 else float("inf")
        return value / self.reference

    def update(self, old_value: float, new_value: float) -> Tuple[float, float]:
        decrease = max(0.0, old_value - new_value)
        self.improvement = (
            decrease
            if self.improvement is None
            else self.momentum * self.improvement + (1 - self.momentum) * decrease
        )
        return self.relative(decrease), self.relative(self.improvement)

    def objective_converged(self) -> bool:
        return (
            self.improvement is not None
            and self.improvement <= self.objective_threshold
        )


@dataclass(frozen=True)
class InexactNewtonOptions:
    """Controls for Newton-CG, Eisenstat-Walker forcing, and line search.

    Gradient norm and objective-decrease tests each use
    ``absolute_tolerance + relative_tolerance * abs(initial_objective)``.
    Absolute defaults are zero; relative defaults are ``1e-6`` for the
    projected gradient and ``1e-9`` for objective decrease. The reference
    is fixed at the first evaluation, or supplied via ``initial_objective``
    on restart. A zero reference contributes no relative tolerance.
    Gradient norms are measured in optimizer coordinates; these options
    do not rescale the objective, gradient, or search direction.

    ``gradient_tolerance`` is the legacy absolute-gradient spelling and,
    when supplied, disables the default relative-gradient term.
    ``objective_tolerance`` is the legacy relative-objective spelling,
    now also relative to the fixed initial objective. Prefer the explicit
    absolute/relative names; do not supply two spellings of the same term.

    ``objective_tolerance_momentum`` replaces the instantaneous relative
    objective reduction used by the objective stopping test with an
    exponentially weighted reduction.  A value of zero preserves the
    one-iteration test; values nearer one require a sustained plateau.
    ``objective_minimum_iterations`` prevents that test from firing before a
    useful reduction trend has been established.  Gradient and step tests
    remain independent safeguards.
    """

    max_iterations: int = 50
    max_objective_evaluations: Optional[int] = None
    max_cg_iterations: Optional[int] = None
    max_line_search_trials: Optional[int] = None
    gradient_tolerance: Optional[float] = None
    step_tolerance: float = 1.0e-9
    objective_tolerance: Optional[float] = None
    grad_abs_tol: float = 0.0
    grad_rel_tole: Optional[float] = None
    obj_abs_tol: float = 0.0
    obj_rel_tol: Optional[float] = None
    initial_objective: Optional[float] = None
    objective_target: Optional[float] = None
    objective_tolerance_momentum: float = 0.0
    objective_minimum_iterations: int = 1
    initial_forcing: float = 0.7
    minimum_forcing: float = 1.0e-6
    maximum_forcing: float = 0.7
    armijo_constant: float = 1.0e-4
    backtrack_factor: float = 0.5
    minimum_step_length: float = 1.0e-8
    curvature_tolerance: float = 1.0e-14
    bound_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be positive")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        if int(self.objective_minimum_iterations) < 1:
            raise ValueError("objective_minimum_iterations must be positive")
        object.__setattr__(
            self,
            "objective_minimum_iterations",
            int(self.objective_minimum_iterations),
        )
        for field_name in (
            "max_objective_evaluations",
            "max_cg_iterations",
            "max_line_search_trials",
        ):
            value = getattr(self, field_name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{field_name} must be positive when supplied")
            object.__setattr__(self, field_name, None if value is None else int(value))
        for field_name in (
            "step_tolerance",
            "grad_abs_tol",
            "obj_abs_tol",
            "minimum_forcing",
            "minimum_step_length",
            "curvature_tolerance",
            "bound_tolerance",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), field_name, allow_zero=True),
            )
        _validate_stopping_options(self)
        if self.objective_target is not None:
            object.__setattr__(
                self,
                "objective_target",
                _positive(self.objective_target, "objective_target", allow_zero=True),
            )
        for field_name in ("initial_forcing", "maximum_forcing"):
            value = _positive(getattr(self, field_name), field_name)
            if value >= 1.0:
                raise ValueError(f"{field_name} must be less than one")
            object.__setattr__(self, field_name, value)
        momentum = _positive(
            self.objective_tolerance_momentum,
            "objective_tolerance_momentum",
            allow_zero=True,
        )
        if momentum >= 1.0:
            raise ValueError("objective_tolerance_momentum must be less than one")
        object.__setattr__(self, "objective_tolerance_momentum", momentum)
        if self.minimum_forcing > self.maximum_forcing:
            raise ValueError("minimum_forcing cannot exceed maximum_forcing")
        armijo = _positive(self.armijo_constant, "armijo_constant")
        if armijo >= 1.0:
            raise ValueError("armijo_constant must be less than one")
        object.__setattr__(self, "armijo_constant", armijo)
        backtrack = _positive(self.backtrack_factor, "backtrack_factor")
        if backtrack >= 1.0:
            raise ValueError("backtrack_factor must be less than one")
        object.__setattr__(self, "backtrack_factor", backtrack)


@dataclass(frozen=True)
class LBFGSOptions:
    """Controls for projected limited-memory BFGS and Armijo line search.

    ``max_line_search_trials`` caps attempted steps per update across both
    the L-BFGS direction and any projected-gradient recovery direction.

    Absolute/relative stopping tolerances, their defaults and the fixed
    ``initial_objective`` reference follow :class:`InexactNewtonOptions`.
    """

    max_iterations: int = 50
    max_objective_evaluations: Optional[int] = None
    max_line_search_trials: Optional[int] = None
    history_size: int = 10
    gradient_tolerance: Optional[float] = None
    step_tolerance: float = 1.0e-9
    objective_tolerance: Optional[float] = None
    grad_abs_tol: float = 0.0
    grad_rel_tole: Optional[float] = None
    obj_abs_tol: float = 0.0
    obj_rel_tol: Optional[float] = None
    initial_objective: Optional[float] = None
    objective_target: Optional[float] = None
    objective_tolerance_momentum: float = 0.0
    objective_minimum_iterations: int = 1
    armijo_constant: float = 1.0e-4
    backtrack_factor: float = 0.5
    minimum_step_length: float = 1.0e-8
    curvature_tolerance: float = 1.0e-14
    bound_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be positive")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        if int(self.objective_minimum_iterations) < 1:
            raise ValueError("objective_minimum_iterations must be positive")
        object.__setattr__(
            self,
            "objective_minimum_iterations",
            int(self.objective_minimum_iterations),
        )
        if int(self.history_size) < 1:
            raise ValueError("history_size must be positive")
        object.__setattr__(self, "history_size", int(self.history_size))
        for field_name in (
            "max_objective_evaluations",
            "max_line_search_trials",
        ):
            value = getattr(self, field_name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{field_name} must be positive when supplied")
            object.__setattr__(self, field_name, None if value is None else int(value))
        for field_name in (
            "step_tolerance",
            "grad_abs_tol",
            "obj_abs_tol",
            "minimum_step_length",
            "curvature_tolerance",
            "bound_tolerance",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), field_name, allow_zero=True),
            )
        _validate_stopping_options(self)
        if self.objective_target is not None:
            object.__setattr__(
                self,
                "objective_target",
                _positive(self.objective_target, "objective_target", allow_zero=True),
            )
        momentum = _positive(
            self.objective_tolerance_momentum,
            "objective_tolerance_momentum",
            allow_zero=True,
        )
        if momentum >= 1.0:
            raise ValueError("objective_tolerance_momentum must be less than one")
        object.__setattr__(self, "objective_tolerance_momentum", momentum)
        armijo = _positive(self.armijo_constant, "armijo_constant")
        if armijo >= 1.0:
            raise ValueError("armijo_constant must be less than one")
        object.__setattr__(self, "armijo_constant", armijo)
        backtrack = _positive(self.backtrack_factor, "backtrack_factor")
        if backtrack >= 1.0:
            raise ValueError("backtrack_factor must be less than one")
        object.__setattr__(self, "backtrack_factor", backtrack)


LBFGS_RESTART_SCHEMA = "fs-lbfgs-restart-2"


def _all_finite(array: np.ndarray) -> bool:
    """Check finiteness with two reductions instead of a boolean temporary."""

    return bool(np.isfinite(array.min()) and np.isfinite(array.max()))


def _read_only(array: np.ndarray) -> np.ndarray:
    """Share a vector the optimizer never writes again as a read-only view."""

    view = array.view()
    view.flags.writeable = False
    return view


_SHARES_ARRAYS = "_frequensolve_shares_arrays"
_Function = TypeVar("_Function", bound=Callable[..., Any])


def _shares_arrays(function: _Function) -> _Function:
    """Mark an SDK callable that the optimizers call without defensive copies.

    A marked callable never writes its array arguments or keeps them after
    returning (it receives read-only views of the optimizer's vectors), and
    an array it returns is new and never touched by it again, so the
    optimizer adopts it as its own. Unmarked (user) callables keep receiving
    private copies, and their results are copied.
    """

    setattr(function, _SHARES_ARRAYS, True)
    return function


def _shares(function: Any) -> bool:
    """Return whether ``function`` (or a bound method's function) is marked."""

    return getattr(function, _SHARES_ARRAYS, False) is True


def _argument(vector: np.ndarray, shared: bool) -> np.ndarray:
    """Pass a marked callable a read-only view and any other one a private copy."""

    return _read_only(vector) if shared else np.array(vector, copy=True)


def _result(value: Any, shared: bool, *, name: str, size: int) -> np.ndarray:
    """Adopt a marked callable's new float64 vector; validate and copy any other."""

    if (
        shared
        and isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.shape == (size,)
        and value.flags.c_contiguous
        and value.flags.writeable
    ):
        if not _all_finite(value):
            raise ValueError(f"{name} must contain only finite values")
        return value
    return _finite_vector(value, name=name, size=size)


def _hessian_action(
    hessian_product: HessianProduct,
    model: np.ndarray,
    direction: np.ndarray,
    shared: bool,
) -> np.ndarray:
    """Apply ``hessian_product``; unmarked, it gets a model copy and ``direction`` itself."""

    return _result(
        hessian_product(
            _argument(model, shared), _read_only(direction) if shared else direction
        ),
        shared,
        name="Hessian product",
        size=model.size,
    )


@dataclass(frozen=True, eq=False)
class LBFGSRestart:
    """Exact restart state of :func:`minimize_lbfgs` (``fs-lbfgs-restart-2``).

    Every accepted L-BFGS iteration reports one as
    ``InexactNewtonIteration.optimizer_state``. It references the optimizer's
    own read-only arrays, so building it copies nothing: ``model`` is the
    accepted iterate and ``steps``/``gradient_differences`` hold one vector per
    retained curvature pair, oldest first. ``pair_ids`` number each pair by the
    accepted iteration that produced it; they are unique within one run and its
    restarts, so a consumer can persist each immutable pair exactly once.
    Passing the state back as ``minimize_lbfgs(restart=...)`` continues the
    iteration bitwise identically. Read-only restart arrays are adopted by
    reference; writable ones are copied once. Do not modify the arrays.
    """

    model: np.ndarray
    steps: Tuple[np.ndarray, ...]
    gradient_differences: Tuple[np.ndarray, ...]
    pair_ids: Tuple[int, ...]
    history_size: int
    accepted_iterations: int
    initial_objective: float
    improvement: Optional[float] = None
    schema: str = LBFGS_RESTART_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LBFGS_RESTART_SCHEMA:
            raise ValueError(
                f"Unsupported L-BFGS restart schema {self.schema!r}; "
                f"expected {LBFGS_RESTART_SCHEMA!r}"
            )
        model = self.model
        if not isinstance(model, np.ndarray) or model.ndim != 1 or not model.size:
            raise ValueError("L-BFGS restart model must be a non-empty vector")
        steps = tuple(self.steps)
        differences = tuple(self.gradient_differences)
        identifiers = tuple(int(value) for value in self.pair_ids)
        if not len(steps) == len(differences) == len(identifiers):
            raise ValueError("L-BFGS restart pairs and identifiers disagree")
        for vector in steps + differences:
            if not isinstance(vector, np.ndarray) or vector.shape != model.shape:
                raise ValueError("L-BFGS restart pairs must match the model shape")
        accepted = int(self.accepted_iterations)
        if accepted < 0 or int(self.history_size) < 1:
            raise ValueError("L-BFGS restart counters must be nonnegative")
        if any(
            later <= earlier for earlier, later in zip(identifiers, identifiers[1:])
        ):
            raise ValueError("L-BFGS restart pair identifiers must increase")
        if identifiers and (identifiers[0] < 1 or identifiers[-1] > accepted):
            raise ValueError("L-BFGS restart pair identifiers exceed its iterations")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "gradient_differences", differences)
        object.__setattr__(self, "pair_ids", identifiers)
        object.__setattr__(self, "history_size", int(self.history_size))
        object.__setattr__(self, "accepted_iterations", accepted)
        object.__setattr__(self, "initial_objective", float(self.initial_objective))
        if self.improvement is not None:
            object.__setattr__(self, "improvement", float(self.improvement))

    @property
    def pair_count(self) -> int:
        """Return the number of retained curvature pairs."""

        return len(self.steps)


@dataclass(frozen=True)
class InexactNewtonIteration:
    """Diagnostics for the initial point or one accepted outer iteration.

    The vectors are read-only views of the optimizer's own arrays, not copies.
    The optimizer never writes an accepted iterate, gradient or step again, so
    a callback may retain them; they may share memory with other records (one
    iteration's ``gradient`` is the next one's ``linearization_gradient``), the
    restart state and the returned result. Copy a vector before modifying it.
    """

    iteration: int
    model: np.ndarray
    objective: float
    objective_relative_reduction: float
    objective_reduction_momentum: float
    gradient: np.ndarray
    linearization_gradient: np.ndarray
    projected_gradient_norm: float
    step: np.ndarray
    raw_step: np.ndarray
    directional_derivative: float
    raw_directional_derivative: float
    step_length: float
    forcing: float
    cg_iterations: int
    cg_residual_norm: float
    cg_relative_residual: float
    step_transform_cosine: float
    step_transform_norm_ratio: float
    negative_curvature: bool
    steepest_descent_fallback: bool
    line_search_evaluations: int
    objective_evaluations: int
    gradient_evaluations: int
    hessian_products: int
    optimizer_state: Optional[LBFGSRestart] = None


@dataclass(frozen=True)
class InexactNewtonResult:
    """Terminal state and work counters from matrix-free inexact Newton."""

    success: bool
    status: int
    message: str
    model: np.ndarray
    objective: float
    gradient: np.ndarray
    iterations: int
    objective_evaluations: int
    gradient_evaluations: int
    hessian_products: int
    cg_iterations: int
    line_search_evaluations: int
    negative_curvature_events: int
    steepest_descent_fallbacks: int

    @property
    def x(self) -> np.ndarray:
        """Provide the conventional SciPy-compatible name for the model."""

        return self.model

    @property
    def fun(self) -> float:
        """Provide the conventional SciPy-compatible objective name."""

        return self.objective

    @property
    def jac(self) -> np.ndarray:
        """Provide the conventional SciPy-compatible gradient name."""

        return self.gradient


@dataclass(frozen=True)
class _CGResult:
    step: np.ndarray
    hessian_step: np.ndarray
    iterations: int
    residual_norm: float
    negative_curvature: bool


class _EvaluationLimit(RuntimeError):
    """Internal signal for a configured objective-evaluation limit."""


def _bound_flag(
    value: Optional[Sequence[bool]], size: int, name: str
) -> Optional[np.ndarray]:
    """Broadcast and validate one optional dense bound-activation flag."""

    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must contain boolean values")
    try:
        return np.broadcast_to(array, (size,))
    except ValueError as error:
        raise ValueError(f"{name} is not broadcastable to the model") from error


def _bounds(
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]],
    size: int,
    bound_flags: Optional[Tuple[Optional[Sequence[bool]], Optional[Sequence[bool]]]],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Broadcast box bounds and apply optional per-DOF activation flags."""

    if bounds is None:
        if bound_flags is not None:
            raise ValueError("bound_flags require lower and upper bounds")
        return None, None
    if len(bounds) != 2:
        raise ValueError("bounds must contain lower and upper values")
    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    try:
        lower = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), (size,))
        upper = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), (size,))
    except ValueError as error:
        raise ValueError("bounds are not broadcastable to the model") from error
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("bounds cannot contain NaN")

    if bound_flags is not None:
        if len(bound_flags) != 2:
            raise ValueError("bound_flags must contain lower and upper flags")
        lower_active = _bound_flag(bound_flags[0], size, "lower bound flags")
        upper_active = _bound_flag(bound_flags[1], size, "upper bound flags")
        if lower_active is not None:
            if np.any(lower_active & ~np.isfinite(lower)):
                raise ValueError("active lower bounds must be finite")
            if not np.any(lower_active):
                lower = None
            elif not np.all(lower_active):
                lower = np.where(lower_active, lower, -np.inf)
        if upper_active is not None:
            if np.any(upper_active & ~np.isfinite(upper)):
                raise ValueError("active upper bounds must be finite")
            if not np.any(upper_active):
                upper = None
            elif not np.all(upper_active):
                upper = np.where(upper_active, upper, np.inf)

    if lower is not None:
        if np.any(np.isposinf(lower)):
            raise ValueError("lower bounds cannot be positive infinity")
        if np.all(np.isneginf(lower)):
            lower = None
    if upper is not None:
        if np.any(np.isneginf(upper)):
            raise ValueError("upper bounds cannot be negative infinity")
        if np.all(np.isposinf(upper)):
            upper = None
    if lower is not None and upper is not None and np.any(lower > upper):
        raise ValueError("each active lower bound must not exceed its upper bound")
    return lower, upper


def _free_variables(
    model: np.ndarray,
    gradient: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    tolerance: float,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Return the free-variable mask and bound-aware projected gradient."""

    if lower is None and upper is None:
        return None, gradient
    scale = np.maximum(1.0, np.abs(model))
    active = np.zeros(model.size, dtype=bool)
    if lower is not None and upper is not None:
        active |= lower == upper
    if lower is not None:
        active |= (model <= lower + tolerance * scale) & (gradient > 0.0)
    if upper is not None:
        active |= (model >= upper - tolerance * scale) & (gradient < 0.0)
    projected = np.array(gradient, copy=True)
    projected[active] = 0.0
    return ~active, projected


def _inexact_cg(
    model: np.ndarray,
    right_hand_side: np.ndarray,
    hessian_product: HessianProduct,
    *,
    free: Optional[np.ndarray],
    forcing: float,
    max_iterations: int,
    curvature_tolerance: float,
    preconditioner: Optional[Preconditioner],
) -> tuple[_CGResult, int]:
    """Approximately solve the reduced Newton system with truncated PCG."""

    step = np.zeros_like(right_hand_side)
    hessian_step = np.zeros_like(right_hand_side)
    residual = (
        np.array(right_hand_side, copy=True)
        if free is None
        else np.where(free, right_hand_side, 0.0)
    )
    initial_norm = float(np.linalg.norm(residual))
    target = forcing * initial_norm
    if initial_norm == 0.0:
        return _CGResult(step, hessian_step, 0, 0.0, False), 0
    shares_hessian = _shares(hessian_product)
    shares_preconditioner = _shares(preconditioner)

    def apply_preconditioner(value: np.ndarray) -> np.ndarray:
        if preconditioner is None:
            return np.array(value, copy=True)
        result = _result(
            preconditioner(
                _argument(model, shares_preconditioner),
                _argument(value, shares_preconditioner),
            ),
            shares_preconditioner,
            name="preconditioned residual",
            size=value.size,
        )
        return result if free is None else np.where(free, result, 0.0)

    z = apply_preconditioner(residual)
    residual_z = float(np.dot(residual, z))
    if residual_z <= 0.0:
        z = np.array(residual, copy=True)
        residual_z = float(np.dot(residual, residual))
    direction = np.array(z, copy=True)
    hessian_products = 0
    for iteration in range(1, max_iterations + 1):
        restricted_direction = (
            direction if free is None else np.where(free, direction, 0.0)
        )
        hessian_direction = _hessian_action(
            hessian_product, model, restricted_direction, shares_hessian
        )
        hessian_products += 1
        if free is not None:
            hessian_direction[~free] = 0.0
        curvature = float(np.dot(direction, hessian_direction))
        scale = max(
            float(np.linalg.norm(direction) * np.linalg.norm(hessian_direction)),
            float(np.finfo(float).tiny),
        )
        if curvature <= curvature_tolerance * scale:
            if iteration == 1:
                step = np.array(right_hand_side, copy=True)
                if free is not None:
                    step[~free] = 0.0
                hessian_step = _hessian_action(
                    hessian_product, model, step, shares_hessian
                )
                hessian_products += 1
                if free is not None:
                    hessian_step[~free] = 0.0
            return (
                _CGResult(
                    step,
                    hessian_step,
                    iteration - 1,
                    float(np.linalg.norm(residual)),
                    True,
                ),
                hessian_products,
            )
        coefficient = residual_z / curvature
        step += coefficient * direction
        hessian_step += coefficient * hessian_direction
        residual -= coefficient * hessian_direction
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= target:
            return (
                _CGResult(
                    step,
                    hessian_step,
                    iteration,
                    residual_norm,
                    False,
                ),
                hessian_products,
            )
        z_next = apply_preconditioner(residual)
        residual_z_next = float(np.dot(residual, z_next))
        if residual_z_next <= 0.0:
            z_next = np.array(residual, copy=True)
            residual_z_next = float(np.dot(residual, residual))
        direction = z_next + (residual_z_next / residual_z) * direction
        z = z_next
        residual_z = residual_z_next
    return (
        _CGResult(
            step,
            hessian_step,
            max_iterations,
            float(np.linalg.norm(residual)),
            False,
        ),
        hessian_products,
    )


def minimize_inexact_newton(
    objective: Objective,
    gradient: Gradient,
    hessian_product: HessianProduct,
    initial_model: Sequence[float],
    *,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    bound_flags: Optional[
        Tuple[Optional[Sequence[bool]], Optional[Sequence[bool]]]
    ] = None,
    options: Optional[InexactNewtonOptions] = None,
    preconditioner: Optional[Preconditioner] = None,
    step_limit: Optional[StepLimit] = None,
    step_transform: Optional[StepTransform] = None,
    callback: Optional[IterationCallback] = None,
) -> InexactNewtonResult:
    """Minimize a real objective with matrix-free inexact Newton-CG.

    The inner Newton system is truncated according to an Eisenstat-Walker
    forcing term and on non-positive curvature. An Armijo backtracking line
    search globalizes each step. Supplying a Gauss-Newton product gives an
    inexact Gauss-Newton method; supplying the complete reduced-Hessian action
    gives the full inexact Newton method without changing this driver. Bounds
    use the SciPy ``(lower, upper)`` convention. Optional ``bound_flags`` mark
    which lower and upper entries are present; an omitted flag side retains
    the corresponding bounds exactly as authored. ``step_limit`` may provide a
    direction-dependent representation feasibility cap; it is combined with
    the box-bound limit before line-search backtracking. ``step_transform``
    may apply a model-dependent filter or proximal map to every trial step;
    the transformed displacement is used for both bounds and the Armijo test.
    """

    if (
        not callable(objective)
        or not callable(gradient)
        or not callable(hessian_product)
    ):
        raise TypeError("objective, gradient, and hessian_product must be callable")
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be callable")
    if step_limit is not None and not callable(step_limit):
        raise TypeError("step_limit must be callable")
    if step_transform is not None and not callable(step_transform):
        raise TypeError("step_transform must be callable")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    options = InexactNewtonOptions() if options is None else options
    if not isinstance(options, InexactNewtonOptions):
        raise TypeError("options must be InexactNewtonOptions")
    model = _finite_vector(initial_model, name="initial model")
    lower, upper = _bounds(bounds, model.size, bound_flags)
    if (lower is not None and np.any(model < lower)) or (
        upper is not None and np.any(model > upper)
    ):
        raise ValueError("initial model violates its bounds")

    objective_evaluations = 0
    gradient_evaluations = 0
    hessian_products = 0
    total_cg_iterations = 0
    total_line_search_evaluations = 0
    negative_curvature_events = 0
    steepest_descent_fallbacks = 0
    # SDK callables marked by ``_shares_arrays`` are passed read-only views.
    shares_objective, shares_gradient = _shares(objective), _shares(gradient)
    shares_hessian, shares_limit = _shares(hessian_product), _shares(step_limit)
    shares_transform = _shares(step_transform)

    def evaluate(candidate: np.ndarray) -> float:
        nonlocal objective_evaluations
        limit = options.max_objective_evaluations
        if limit is not None and objective_evaluations >= limit:
            raise _EvaluationLimit
        value = float(objective(_argument(candidate, shares_objective)))
        objective_evaluations += 1
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    def evaluate_gradient(candidate: np.ndarray) -> np.ndarray:
        nonlocal gradient_evaluations
        value = _result(
            gradient(_argument(candidate, shares_gradient)),
            shares_gradient,
            name="gradient",
            size=model.size,
        )
        gradient_evaluations += 1
        return value

    value = evaluate(model)
    stopping = _StoppingCriteria(options, value)
    grad = evaluate_gradient(model)
    free, projected_gradient = _free_variables(
        model, grad, lower, upper, options.bound_tolerance
    )
    projected_norm = float(np.linalg.norm(projected_gradient))
    forcing = min(
        options.maximum_forcing, max(options.minimum_forcing, options.initial_forcing)
    )
    if callback is not None:
        # Accepted vectors are never written again: share them read-only.
        zero = _read_only(np.zeros_like(model))
        callback(
            InexactNewtonIteration(
                iteration=0,
                model=_read_only(model),
                objective=value,
                objective_relative_reduction=0.0,
                objective_reduction_momentum=0.0,
                gradient=_read_only(grad),
                linearization_gradient=_read_only(grad),
                projected_gradient_norm=projected_norm,
                step=zero,
                raw_step=zero,
                directional_derivative=0.0,
                raw_directional_derivative=0.0,
                step_length=0.0,
                forcing=forcing,
                cg_iterations=0,
                cg_residual_norm=projected_norm,
                cg_relative_residual=1.0,
                step_transform_cosine=1.0,
                step_transform_norm_ratio=1.0,
                negative_curvature=False,
                steepest_descent_fallback=False,
                line_search_evaluations=0,
                objective_evaluations=objective_evaluations,
                gradient_evaluations=gradient_evaluations,
                hessian_products=hessian_products,
            )
        )
        del zero  # Held by the record only; never pinned for the whole run.
    if options.objective_target is not None and value <= options.objective_target:
        return InexactNewtonResult(
            True,
            4,
            "objective target reached",
            model,
            value,
            grad,
            0,
            objective_evaluations,
            gradient_evaluations,
            hessian_products,
            total_cg_iterations,
            total_line_search_evaluations,
            negative_curvature_events,
            steepest_descent_fallbacks,
        )
    if projected_norm <= stopping.gradient_threshold:
        return InexactNewtonResult(
            True,
            1,
            "projected gradient tolerance reached",
            model,
            value,
            grad,
            0,
            objective_evaluations,
            gradient_evaluations,
            hessian_products,
            total_cg_iterations,
            total_line_search_evaluations,
            negative_curvature_events,
            steepest_descent_fallbacks,
        )

    previous_forcing = forcing
    previous_gradient: Optional[np.ndarray] = None
    previous_hessian_step: Optional[np.ndarray] = None
    for iteration in range(1, options.max_iterations + 1):
        if previous_gradient is not None and previous_hessian_step is not None:
            denominator = max(
                float(np.linalg.norm(previous_gradient)), float(np.finfo(float).tiny)
            )
            forcing = float(
                np.linalg.norm(grad - previous_gradient - previous_hessian_step)
                / denominator
            )
            exponent = 0.5 * (1.0 + np.sqrt(5.0))
            safeguarded = previous_forcing**exponent
            if safeguarded > 0.1:
                forcing = max(forcing, safeguarded)
            forcing = min(
                options.maximum_forcing,
                max(options.minimum_forcing, forcing),
            )

        free_count = model.size if free is None else int(np.count_nonzero(free))
        cg_limit = options.max_cg_iterations or free_count
        cg_limit = max(1, cg_limit)
        cg, products = _inexact_cg(
            model,
            -projected_gradient,
            hessian_product,
            free=free,
            forcing=forcing,
            max_iterations=cg_limit,
            curvature_tolerance=options.curvature_tolerance,
            preconditioner=preconditioner,
        )
        hessian_products += products
        total_cg_iterations += cg.iterations
        cg_rhs_norm = max(projected_norm, float(np.finfo(float).tiny))
        if cg.negative_curvature:
            negative_curvature_events += 1
        # The CG vectors are this iteration's own and never written in place.
        direction = cg.step
        hessian_direction = cg.hessian_step
        directional_derivative = float(np.dot(grad, direction))
        fallback = (
            not np.isfinite(directional_derivative) or directional_derivative >= 0.0
        )
        if fallback:
            direction = -projected_gradient
            hessian_direction = _hessian_action(
                hessian_product, model, direction, shares_hessian
            )
            hessian_products += 1
            if free is not None:
                hessian_direction[~free] = 0.0
            directional_derivative = -float(
                np.dot(projected_gradient, projected_gradient)
            )
            steepest_descent_fallbacks += 1
        # Box feasibility is enforced by projecting each trial model below.  A
        # global maximum feasible scalar step would let one outward component
        # at a bound throttle every otherwise useful component of a coupled
        # Newton direction.
        maximum_step = np.inf
        if step_limit is not None:
            representation_step = float(
                step_limit(
                    _argument(model, shares_limit), _argument(direction, shares_limit)
                )
            )
            if np.isnan(representation_step) or representation_step < 0.0:
                raise ValueError("step_limit must return a nonnegative value")
            maximum_step = min(maximum_step, representation_step)
        if maximum_step <= 0.0 or directional_derivative >= 0.0:
            return InexactNewtonResult(
                False,
                -3,
                "no feasible descent direction",
                model,
                value,
                grad,
                iteration - 1,
                objective_evaluations,
                gradient_evaluations,
                hessian_products,
                total_cg_iterations,
                total_line_search_evaluations,
                negative_curvature_events,
                steepest_descent_fallbacks,
            )

        step_length = min(1.0, maximum_step)
        line_search_evaluations = 0
        line_search_trials = 0
        accepted = False
        # A placeholder: every evaluated trial is a new array.
        trial = model
        trial_value = value
        try:
            while step_length >= options.minimum_step_length and (
                options.max_line_search_trials is None
                or line_search_trials < options.max_line_search_trials
            ):
                line_search_trials += 1
                raw_step = step_length * direction
                step = raw_step
                if step_transform is not None:
                    step = _result(
                        step_transform(
                            _argument(model, shares_transform),
                            _argument(step, shares_transform),
                        ),
                        shares_transform,
                        name="transformed step",
                        size=model.size,
                    )
                trial = model + step
                if lower is not None:
                    np.maximum(trial, lower, out=trial)
                if upper is not None:
                    np.minimum(trial, upper, out=trial)
                step = trial - model
                raw_trial = model + raw_step
                if lower is not None:
                    np.maximum(raw_trial, lower, out=raw_trial)
                if upper is not None:
                    np.minimum(raw_trial, upper, out=raw_trial)
                bounded_raw_step = raw_trial - model
                trial_directional_derivative = float(np.dot(grad, step))
                if step_transform is not None:
                    raw_derivative = float(np.dot(grad, bounded_raw_step))
                    if raw_derivative >= 0.0 or not np.any(bounded_raw_step):
                        step_length *= options.backtrack_factor
                        continue
                    # Preserve every transformed descent step exactly. Mixing
                    # raw Newton content back into a strongly smoothed
                    # direction defeats the filter. A non-descending transform
                    # is rejected and retried at the next line-search scale;
                    # the optimizer must never accept an unsmoothed fallback.
                if trial_directional_derivative >= 0.0 or not np.any(step):
                    step_length *= options.backtrack_factor
                    continue
                trial_value = evaluate(trial)
                line_search_evaluations += 1
                if trial_value <= (
                    value + options.armijo_constant * trial_directional_derivative
                ):
                    accepted = True
                    break
                step_length *= options.backtrack_factor
        except _EvaluationLimit:
            return InexactNewtonResult(
                False,
                -2,
                "objective evaluation limit reached during line search",
                model,
                value,
                grad,
                iteration - 1,
                objective_evaluations,
                gradient_evaluations,
                hessian_products,
                total_cg_iterations,
                total_line_search_evaluations + line_search_evaluations,
                negative_curvature_events,
                steepest_descent_fallbacks,
            )
        total_line_search_evaluations += line_search_evaluations
        if not accepted:
            return InexactNewtonResult(
                False,
                -1,
                "Armijo line search failed",
                model,
                value,
                grad,
                iteration - 1,
                objective_evaluations,
                gradient_evaluations,
                hessian_products,
                total_cg_iterations,
                total_line_search_evaluations,
                negative_curvature_events,
                steepest_descent_fallbacks,
            )

        old_model = model
        old_value = value
        old_gradient = grad
        step = trial - old_model
        model = trial
        value = trial_value
        objective_relative_reduction, objective_reduction_momentum = stopping.update(
            old_value, value
        )
        grad = evaluate_gradient(model)
        free, projected_gradient = _free_variables(
            model, grad, lower, upper, options.bound_tolerance
        )
        projected_norm = float(np.linalg.norm(projected_gradient))
        if step_transform is None:
            previous_gradient = old_gradient
            previous_hessian_step = step_length * hessian_direction
        else:
            # A nonlinear filtered step has no reusable raw Newton Hs action.
            previous_gradient = None
            previous_hessian_step = None
        previous_forcing = forcing
        if callback is not None:
            raw_norm = float(np.linalg.norm(bounded_raw_step))
            transformed_norm = float(np.linalg.norm(step))
            transform_norm_product = raw_norm * transformed_norm
            transform_cosine = (
                1.0
                if transform_norm_product == 0.0
                else float(np.dot(bounded_raw_step, step)) / transform_norm_product
            )
            callback(
                InexactNewtonIteration(
                    iteration=iteration,
                    model=_read_only(model),
                    objective=value,
                    objective_relative_reduction=objective_relative_reduction,
                    objective_reduction_momentum=objective_reduction_momentum,
                    gradient=_read_only(grad),
                    linearization_gradient=_read_only(old_gradient),
                    projected_gradient_norm=projected_norm,
                    step=_read_only(step),
                    raw_step=_read_only(bounded_raw_step),
                    directional_derivative=float(np.dot(old_gradient, step)),
                    raw_directional_derivative=float(
                        np.dot(old_gradient, bounded_raw_step)
                    ),
                    step_length=step_length,
                    forcing=forcing,
                    cg_iterations=cg.iterations,
                    cg_residual_norm=cg.residual_norm,
                    cg_relative_residual=cg.residual_norm / cg_rhs_norm,
                    step_transform_cosine=transform_cosine,
                    step_transform_norm_ratio=transformed_norm
                    / max(raw_norm, float(np.finfo(float).tiny)),
                    negative_curvature=cg.negative_curvature,
                    steepest_descent_fallback=fallback,
                    line_search_evaluations=line_search_evaluations,
                    objective_evaluations=objective_evaluations,
                    gradient_evaluations=gradient_evaluations,
                    hessian_products=hessian_products,
                )
            )

        status = 0
        message = ""
        if options.objective_target is not None and value <= options.objective_target:
            status, message = 4, "objective target reached"
        elif projected_norm <= stopping.gradient_threshold:
            status, message = 1, "projected gradient tolerance reached"
        elif float(np.linalg.norm(step)) <= options.step_tolerance * max(
            1.0, float(np.linalg.norm(model))
        ):
            status, message = 3, "step tolerance reached"
        elif (
            iteration >= options.objective_minimum_iterations
            and stopping.objective_converged()
        ):
            status, message = 2, "objective tolerance reached"
        if status:
            return InexactNewtonResult(
                True,
                status,
                message,
                model,
                value,
                grad,
                iteration,
                objective_evaluations,
                gradient_evaluations,
                hessian_products,
                total_cg_iterations,
                total_line_search_evaluations,
                negative_curvature_events,
                steepest_descent_fallbacks,
            )

    return InexactNewtonResult(
        False,
        0,
        "maximum outer iterations reached",
        model,
        value,
        grad,
        options.max_iterations,
        objective_evaluations,
        gradient_evaluations,
        hessian_products,
        total_cg_iterations,
        total_line_search_evaluations,
        negative_curvature_events,
        steepest_descent_fallbacks,
    )


def minimize_lbfgs(
    objective: Objective,
    gradient: Gradient,
    initial_model: Sequence[float],
    *,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    bound_flags: Optional[
        Tuple[Optional[Sequence[bool]], Optional[Sequence[bool]]]
    ] = None,
    options: Optional[LBFGSOptions] = None,
    preconditioner: Optional[Preconditioner] = None,
    step_limit: Optional[StepLimit] = None,
    step_transform: Optional[StepTransform] = None,
    callback: Optional[IterationCallback] = None,
    restart: Optional[LBFGSRestart] = None,
) -> InexactNewtonResult:
    """Minimize a real objective with projected limited-memory BFGS.

    The method obtains curvature only from successive accepted model and
    gradient differences; it never requests a Hessian product. Bounds use the
    SciPy ``(lower, upper)`` convention. The optional preconditioner supplies
    the initial inverse-Hessian action in the two-loop recursion. Step limits,
    nonlinear step transforms, and Armijo backtracking have the same contract
    as :func:`minimize_inexact_newton`.

    The common result and iteration records are retained so optimization
    history consumers can compare Newton-CG and L-BFGS runs directly. Their CG
    and Hessian counters are identically zero for this method.

    Every iteration record carries an :class:`LBFGSRestart` that references
    (never copies) the optimizer's accepted model and curvature pairs;
    ``restart`` continues bitwise identically from such a state.
    """

    if not callable(objective) or not callable(gradient):
        raise TypeError("objective and gradient must be callable")
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be callable")
    if step_limit is not None and not callable(step_limit):
        raise TypeError("step_limit must be callable")
    if step_transform is not None and not callable(step_transform):
        raise TypeError("step_transform must be callable")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    options = LBFGSOptions() if options is None else options
    if not isinstance(options, LBFGSOptions):
        raise TypeError("options must be LBFGSOptions")

    model = _finite_vector(initial_model, name="initial model")
    lower, upper = _bounds(bounds, model.size, bound_flags)
    if (lower is not None and np.any(model < lower)) or (
        upper is not None and np.any(model > upper)
    ):
        raise ValueError("initial model violates its bounds")

    objective_evaluations = 0
    gradient_evaluations = 0
    total_line_search_evaluations = 0
    steepest_descent_fallbacks = 0
    # SDK callables marked by ``_shares_arrays`` are passed read-only views.
    shares_objective, shares_gradient = _shares(objective), _shares(gradient)
    shares_preconditioner, shares_limit = _shares(preconditioner), _shares(step_limit)
    shares_transform = _shares(step_transform)

    def evaluate(candidate: np.ndarray) -> float:
        nonlocal objective_evaluations
        limit = options.max_objective_evaluations
        if limit is not None and objective_evaluations >= limit:
            raise _EvaluationLimit
        value = float(objective(_argument(candidate, shares_objective)))
        objective_evaluations += 1
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    def evaluate_gradient(candidate: np.ndarray) -> np.ndarray:
        nonlocal gradient_evaluations
        value = _result(
            gradient(_argument(candidate, shares_gradient)),
            shares_gradient,
            name="gradient",
            size=model.size,
        )
        gradient_evaluations += 1
        return value

    def initial_inverse_action(vector: np.ndarray) -> np.ndarray:
        if preconditioner is not None:
            # Written in place by the two-loop recursion: a new array either way.
            return _result(
                preconditioner(
                    _argument(model, shares_preconditioner),
                    _argument(vector, shares_preconditioner),
                ),
                shares_preconditioner,
                name="preconditioned gradient",
                size=model.size,
            )
        if not steps:
            return np.array(vector, copy=True)
        scale = float(np.dot(steps[-1], gradient_differences[-1])) / max(
            float(np.dot(gradient_differences[-1], gradient_differences[-1])),
            float(np.finfo(float).tiny),
        )
        return scale * vector

    def inverse_hessian_action(vector: np.ndarray) -> np.ndarray:
        work = np.array(vector, copy=True)
        coefficients = []
        for step, difference, inverse_curvature in reversed(
            list(zip(steps, gradient_differences, inverse_curvatures))
        ):
            coefficient = inverse_curvature * float(np.dot(step, work))
            coefficients.append(coefficient)
            work -= coefficient * difference
        result = initial_inverse_action(work)
        for (step, difference, inverse_curvature), coefficient in zip(
            zip(steps, gradient_differences, inverse_curvatures),
            reversed(coefficients),
        ):
            result += step * (
                coefficient - inverse_curvature * float(np.dot(difference, result))
            )
        return result

    # Accepted pairs are immutable: the restart state shares them by reference.
    steps: list[np.ndarray] = []
    gradient_differences: list[np.ndarray] = []
    inverse_curvatures: list[float] = []
    pair_ids: list[int] = []
    accepted_before = 0
    if restart is not None:
        if not isinstance(restart, LBFGSRestart):
            raise TypeError(
                "L-BFGS restart must be an LBFGSRestart (fs-lbfgs-restart-2); "
                "JSON-list fs-lbfgs-restart-1 states are no longer supported"
            )
        if restart.model.shape != model.shape or not np.array_equal(
            restart.model, model
        ):
            raise ValueError("L-BFGS restart belongs to a different model or scaling")
        if restart.history_size != options.history_size:
            raise ValueError("L-BFGS restart history size changed")
        if restart.pair_count > options.history_size:
            raise ValueError("L-BFGS restart exceeds its history size")
        for step, difference, identifier in zip(
            restart.steps, restart.gradient_differences, restart.pair_ids
        ):
            step = _restart_vector(step, "restart step", model.size)
            difference = _restart_vector(difference, "restart difference", model.size)
            curvature = float(np.dot(step, difference))
            if not np.isfinite(curvature) or curvature <= 0:
                raise ValueError("L-BFGS restart requires positive finite curvature")
            steps.append(step)
            gradient_differences.append(difference)
            inverse_curvatures.append(1.0 / curvature)
            pair_ids.append(identifier)
        accepted_before = restart.accepted_iterations

    value = evaluate(model)
    reference = (
        value
        if restart is None
        else _positive(
            restart.initial_objective, "restart initial objective", allow_zero=True
        )
    )
    if (
        restart is not None
        and options.initial_objective is not None
        and abs(options.initial_objective) != reference
    ):
        raise ValueError("L-BFGS restart initial objective changed")
    stopping = _StoppingCriteria(options, reference)
    if restart is not None:
        improvement = restart.improvement
        if improvement is not None and (
            not np.isfinite(improvement) or improvement < 0
        ):
            raise ValueError("L-BFGS restart has invalid stopping momentum")
        stopping.improvement = improvement

    def restart_state(iteration: int) -> LBFGSRestart:
        # A read-only view of the accepted iterate; the optimizer never writes
        # an accepted model or pair in place, so no vector is copied here.
        current = model.view()
        current.flags.writeable = False
        return LBFGSRestart(
            model=current,
            steps=tuple(steps),
            gradient_differences=tuple(gradient_differences),
            pair_ids=tuple(pair_ids),
            history_size=options.history_size,
            accepted_iterations=accepted_before + iteration,
            initial_objective=stopping.reference,
            improvement=stopping.improvement,
        )

    grad = evaluate_gradient(model)
    free, projected_gradient = _free_variables(
        model, grad, lower, upper, options.bound_tolerance
    )
    projected_norm = float(np.linalg.norm(projected_gradient))
    if callback is not None:
        # Accepted vectors are never written again: share them read-only.
        zero = _read_only(np.zeros_like(model))
        callback(
            InexactNewtonIteration(
                iteration=0,
                model=_read_only(model),
                objective=value,
                objective_relative_reduction=0.0,
                objective_reduction_momentum=0.0,
                gradient=_read_only(grad),
                linearization_gradient=_read_only(grad),
                projected_gradient_norm=projected_norm,
                step=zero,
                raw_step=zero,
                directional_derivative=0.0,
                raw_directional_derivative=0.0,
                step_length=0.0,
                forcing=0.0,
                cg_iterations=0,
                cg_residual_norm=0.0,
                cg_relative_residual=0.0,
                step_transform_cosine=1.0,
                step_transform_norm_ratio=1.0,
                negative_curvature=False,
                steepest_descent_fallback=False,
                line_search_evaluations=0,
                objective_evaluations=objective_evaluations,
                gradient_evaluations=gradient_evaluations,
                hessian_products=0,
                optimizer_state=restart_state(0),
            )
        )
        del zero  # Held by the record only; never pinned for the whole run.
    if options.objective_target is not None and value <= options.objective_target:
        return InexactNewtonResult(
            True,
            4,
            "objective target reached",
            model,
            value,
            grad,
            0,
            objective_evaluations,
            gradient_evaluations,
            0,
            0,
            total_line_search_evaluations,
            0,
            steepest_descent_fallbacks,
        )
    if projected_norm <= stopping.gradient_threshold:
        return InexactNewtonResult(
            True,
            1,
            "projected gradient tolerance reached",
            model,
            value,
            grad,
            0,
            objective_evaluations,
            gradient_evaluations,
            0,
            0,
            total_line_search_evaluations,
            0,
            steepest_descent_fallbacks,
        )

    for iteration in range(1, options.max_iterations + 1):
        direction = -inverse_hessian_action(projected_gradient)
        if free is not None:
            direction[~free] = 0.0
        raw_directional_derivative = float(np.dot(grad, direction))
        fallback = (
            not np.isfinite(raw_directional_derivative)
            or raw_directional_derivative >= 0.0
        )
        if fallback:
            steps.clear()
            gradient_differences.clear()
            inverse_curvatures.clear()
            pair_ids.clear()
            direction = -projected_gradient
            raw_directional_derivative = -float(
                np.dot(projected_gradient, projected_gradient)
            )
            steepest_descent_fallbacks += 1

        line_search_evaluations = 0
        accepted = False
        # Placeholders: an accepted trial assigns all three new arrays.
        trial = bounded_raw_step = step = model
        trial_value = value
        recovery_attempted = fallback
        line_search_trials = 0
        try:
            while not accepted:
                maximum_step = np.inf
                if step_limit is not None:
                    representation_step = float(
                        step_limit(
                            _argument(model, shares_limit),
                            _argument(direction, shares_limit),
                        )
                    )
                    if np.isnan(representation_step) or representation_step < 0.0:
                        raise ValueError("step_limit must return a nonnegative value")
                    maximum_step = min(maximum_step, representation_step)
                if maximum_step <= 0.0 or raw_directional_derivative >= 0.0:
                    break

                step_length = min(1.0, maximum_step)
                while step_length >= options.minimum_step_length and (
                    options.max_line_search_trials is None
                    or line_search_trials < options.max_line_search_trials
                ):
                    line_search_trials += 1
                    raw_step = step_length * direction
                    step = raw_step
                    if step_transform is not None:
                        step = _result(
                            step_transform(
                                _argument(model, shares_transform),
                                _argument(step, shares_transform),
                            ),
                            shares_transform,
                            name="transformed step",
                            size=model.size,
                        )
                    trial = model + step
                    if lower is not None:
                        np.maximum(trial, lower, out=trial)
                    if upper is not None:
                        np.minimum(trial, upper, out=trial)
                    step = trial - model
                    raw_trial = model + raw_step
                    if lower is not None:
                        np.maximum(raw_trial, lower, out=raw_trial)
                    if upper is not None:
                        np.minimum(raw_trial, upper, out=raw_trial)
                    bounded_raw_step = raw_trial - model
                    directional_derivative = float(np.dot(grad, step))
                    if step_transform is not None:
                        raw_derivative = float(np.dot(grad, bounded_raw_step))
                        if raw_derivative >= 0.0 or not np.any(bounded_raw_step):
                            step_length *= options.backtrack_factor
                            continue
                    if directional_derivative >= 0.0 or not np.any(step):
                        step_length *= options.backtrack_factor
                        continue
                    trial_value = evaluate(trial)
                    line_search_evaluations += 1
                    if trial_value <= (
                        value + options.armijo_constant * directional_derivative
                    ):
                        accepted = True
                        break
                    step_length *= options.backtrack_factor

                budget_exhausted = (
                    options.max_line_search_trials is not None
                    and line_search_trials >= options.max_line_search_trials
                )
                if accepted or recovery_attempted or budget_exhausted:
                    break

                # A stale L-BFGS metric can yield a descent direction whose
                # local model is nevertheless poor enough that every bounded
                # Armijo trial fails.  Discard that metric and give the
                # projected gradient the remaining trial budget in the
                # supplied inverse-Hessian metric before abandoning the
                # nonlinear continuation stage.
                steps.clear()
                gradient_differences.clear()
                inverse_curvatures.clear()
                pair_ids.clear()
                direction = -initial_inverse_action(projected_gradient)
                if free is not None:
                    direction[~free] = 0.0
                raw_directional_derivative = float(np.dot(grad, direction))
                if (
                    not np.isfinite(raw_directional_derivative)
                    or raw_directional_derivative >= 0.0
                ):
                    direction = -projected_gradient
                    raw_directional_derivative = -float(
                        np.dot(projected_gradient, projected_gradient)
                    )
                recovery_attempted = True
                fallback = True
                steepest_descent_fallbacks += 1
        except _EvaluationLimit:
            return InexactNewtonResult(
                False,
                -2,
                "objective evaluation limit reached during line search",
                model,
                value,
                grad,
                iteration - 1,
                objective_evaluations,
                gradient_evaluations,
                0,
                0,
                total_line_search_evaluations + line_search_evaluations,
                0,
                steepest_descent_fallbacks,
            )
        total_line_search_evaluations += line_search_evaluations
        if not accepted:
            return InexactNewtonResult(
                False,
                -1,
                "Armijo line search failed",
                model,
                value,
                grad,
                iteration - 1,
                objective_evaluations,
                gradient_evaluations,
                0,
                0,
                total_line_search_evaluations,
                0,
                steepest_descent_fallbacks,
            )

        old_model = model
        old_value = value
        old_gradient = grad
        model = trial
        value = trial_value
        step = model - old_model
        objective_relative_reduction, objective_reduction_momentum = stopping.update(
            old_value, value
        )
        grad = evaluate_gradient(model)
        free, projected_gradient = _free_variables(
            model, grad, lower, upper, options.bound_tolerance
        )
        projected_norm = float(np.linalg.norm(projected_gradient))

        gradient_difference = grad - old_gradient
        curvature = float(np.dot(step, gradient_difference))
        curvature_scale = max(
            float(np.linalg.norm(step) * np.linalg.norm(gradient_difference)),
            float(np.finfo(float).tiny),
        )
        if curvature > options.curvature_tolerance * curvature_scale:
            # Both vectors were freshly computed above and are never written
            # again; retain them read-only instead of copying.
            step.flags.writeable = False
            gradient_difference.flags.writeable = False
            steps.append(step)
            gradient_differences.append(gradient_difference)
            inverse_curvatures.append(1.0 / curvature)
            pair_ids.append(accepted_before + iteration)
            if len(steps) > options.history_size:
                del steps[0]
                del gradient_differences[0]
                del inverse_curvatures[0]
                del pair_ids[0]

        if callback is not None:
            raw_norm = float(np.linalg.norm(bounded_raw_step))
            transformed_norm = float(np.linalg.norm(step))
            transform_norm_product = raw_norm * transformed_norm
            transform_cosine = (
                1.0
                if transform_norm_product == 0.0
                else float(np.dot(bounded_raw_step, step)) / transform_norm_product
            )
            callback(
                InexactNewtonIteration(
                    iteration=iteration,
                    model=_read_only(model),
                    objective=value,
                    objective_relative_reduction=objective_relative_reduction,
                    objective_reduction_momentum=objective_reduction_momentum,
                    gradient=_read_only(grad),
                    linearization_gradient=_read_only(old_gradient),
                    projected_gradient_norm=projected_norm,
                    step=_read_only(step),
                    raw_step=_read_only(bounded_raw_step),
                    directional_derivative=float(np.dot(old_gradient, step)),
                    raw_directional_derivative=float(
                        np.dot(old_gradient, bounded_raw_step)
                    ),
                    step_length=step_length,
                    forcing=0.0,
                    cg_iterations=0,
                    cg_residual_norm=0.0,
                    cg_relative_residual=0.0,
                    step_transform_cosine=transform_cosine,
                    step_transform_norm_ratio=transformed_norm
                    / max(raw_norm, float(np.finfo(float).tiny)),
                    negative_curvature=False,
                    steepest_descent_fallback=fallback,
                    line_search_evaluations=line_search_evaluations,
                    objective_evaluations=objective_evaluations,
                    gradient_evaluations=gradient_evaluations,
                    hessian_products=0,
                    optimizer_state=restart_state(iteration),
                )
            )

        status = 0
        message = ""
        if options.objective_target is not None and value <= options.objective_target:
            status, message = 4, "objective target reached"
        elif projected_norm <= stopping.gradient_threshold:
            status, message = 1, "projected gradient tolerance reached"
        elif float(np.linalg.norm(step)) <= options.step_tolerance * max(
            1.0, float(np.linalg.norm(model))
        ):
            status, message = 3, "step tolerance reached"
        elif (
            accepted_before + iteration >= options.objective_minimum_iterations
            and stopping.objective_converged()
        ):
            status, message = 2, "objective tolerance reached"
        if status:
            return InexactNewtonResult(
                True,
                status,
                message,
                model,
                value,
                grad,
                iteration,
                objective_evaluations,
                gradient_evaluations,
                0,
                0,
                total_line_search_evaluations,
                0,
                steepest_descent_fallbacks,
            )

    return InexactNewtonResult(
        False,
        0,
        "maximum outer iterations reached",
        model,
        value,
        grad,
        options.max_iterations,
        objective_evaluations,
        gradient_evaluations,
        0,
        0,
        total_line_search_evaluations,
        0,
        steepest_descent_fallbacks,
    )


def minimize_proximal_gradient(
    objective: Objective,
    gradient: Gradient,
    regularization: Objective,
    proximal: Callable[[np.ndarray, float, Tuple[np.ndarray, np.ndarray]], np.ndarray],
    initial_model: Any,
    *,
    bounds: Optional[Tuple[Any, Any]] = None,
    options: Optional[InexactNewtonOptions | LBFGSOptions] = None,
    callback: Optional[IterationCallback] = None,
    step_limit: Optional[StepLimit] = None,
    initial_step: float = 1.0,
    relative_tolerance: Optional[float] = None,
    curvature_steps: bool = False,
) -> InexactNewtonResult:
    """Minimize a smooth objective plus a native convex regularizer.

    ``proximal(v, tau, bounds)`` solves the constrained proximal problem in
    these optimizer coordinates. Backtracking tests the smooth majorization
    inequality and composite decrease. Stationarity uses the proximal-gradient
    mapping ``||prox(x - tau g) - x|| / tau``, including at nonsmooth points
    and fixed bounds.

    ``initial_step`` is the first step ``tau``; backtracking stops below
    ``options.minimum_step_length * initial_step``. After an accepted
    iteration the step grows by ``1 / backtrack_factor``, by default up to
    ``initial_step``. The unit default suits scaled optimizer coordinates.
    Objectives without a natural scale (e.g. a frozen least-squares model in
    physical units) pass an inverse-curvature estimate ``1 / lambda_max``.

    ``relative_tolerance`` adds a scale-free stationarity test: stop once the
    mapping norm is at most ``relative_tolerance`` times its value at the
    initial model (first trial step), in addition to the absolute/relative
    thresholds of ``options``. Value comparisons then allow roundoff relative
    to the objective magnitude only, without the unit floor used otherwise.

    ``curvature_steps`` suits smooth parts with (nearly) constant curvature,
    such as a frozen Gauss-Newton model, where every rejected trial costs a
    full objective and proximal evaluation. Steps then follow the secant
    curvature ``kappa`` along each trial (exact for a quadratic): a trial
    rejected by the majorization test is retried at ``1 / kappa`` (at most
    ``backtrack_factor`` times the rejected step), and an accepted iteration
    grows the step towards the Barzilai-Borwein step ``1 / kappa`` of the
    accepted trial without the ``initial_step`` cap.
    """
    options = InexactNewtonOptions() if options is None else options
    first_step = _positive(initial_step, "initial_step")
    if relative_tolerance is not None:
        relative_tolerance = _positive(
            relative_tolerance, "relative_tolerance", allow_zero=True
        )
    x = _finite_vector(initial_model, name="initial model")
    lower, upper = _bounds(bounds, x.size, None)
    # The same read-only pair reaches every proximal call, so a callable can
    # stage bound-dependent inputs once per solve.
    box = (
        _read_only(np.full(x.size, -np.inf) if lower is None else lower),
        _read_only(np.full(x.size, np.inf) if upper is None else upper),
    )
    if np.any(x < box[0]) or np.any(x > box[1]):
        raise ValueError("initial model violates its bounds")
    evaluations = gradients = searches = 0
    # SDK callables marked by ``_shares_arrays`` are passed read-only views.
    shares_objective, shares_gradient = _shares(objective), _shares(gradient)
    shares_regularization, shares_proximal = _shares(regularization), _shares(proximal)
    shares_limit = _shares(step_limit)

    def evaluate(v: np.ndarray) -> Tuple[float, float]:
        nonlocal evaluations
        if (
            options.max_objective_evaluations is not None
            and evaluations >= options.max_objective_evaluations
        ):
            raise _EvaluationLimit
        f = float(objective(_argument(v, shares_objective)))
        r = float(regularization(_argument(v, shares_regularization)))
        evaluations += 1
        if not np.isfinite(f + r):
            raise ValueError("composite objective must be finite")
        return f, r

    def evaluate_gradient(v: np.ndarray) -> np.ndarray:
        return _result(
            gradient(_argument(v, shares_gradient)),
            shares_gradient,
            name="gradient",
            size=v.size,
        )

    f, reg = evaluate(x)
    stopping = _StoppingCriteria(options, f + reg)
    g = evaluate_gradient(x)
    gradients += 1
    tau = first_step
    minimum_step = options.minimum_step_length * first_step
    stable_tau: Optional[float] = None
    threshold = stopping.gradient_threshold
    reference_mapping: Optional[float] = None
    # Value comparisons allow 32 eps of the objective magnitude, floored at
    # one unless stopping is relative (no unit scale is assumed then).
    roundoff_floor = 1.0 if relative_tolerance is None else 0.0

    def finish(
        success: bool, status: int, message: str, iteration: int
    ) -> InexactNewtonResult:
        return InexactNewtonResult(
            success,
            status,
            message,
            x.copy(),
            f + reg,
            g.copy(),
            iteration,
            evaluations,
            gradients,
            0,
            0,
            searches,
            0,
            0,
        )

    for iteration in range(1, options.max_iterations + 1):
        accepted = False
        trials = 0
        # ``x`` and ``g`` are rebound, never written in place: no copies needed.
        previous = x
        old_gradient = g
        old_total = f + reg
        try:
            while tau >= minimum_step:
                if (
                    options.max_line_search_trials is not None
                    and trials >= options.max_line_search_trials
                ):
                    break
                trials += 1
                trial = _result(
                    proximal(x - tau * g, tau, box),
                    shares_proximal,
                    name="proximal model",
                    size=x.size,
                )
                if np.any(trial < box[0] - options.bound_tolerance) or np.any(
                    trial > box[1] + options.bound_tolerance
                ):
                    raise ValueError("proximal result violates the supplied bounds")
                step = trial - x
                mapping = float(np.linalg.norm(step) / tau)
                if reference_mapping is None:
                    reference_mapping = mapping
                    if relative_tolerance is not None:
                        threshold += relative_tolerance * mapping
                if mapping <= threshold:
                    return finish(
                        True, 0, "proximal gradient tolerance reached", iteration - 1
                    )
                if (
                    step_limit is not None
                    and float(
                        step_limit(
                            _argument(x, shares_limit), _argument(step, shares_limit)
                        )
                    )
                    < 1.0
                ):
                    tau *= options.backtrack_factor
                    continue
                trial_f, trial_reg = evaluate(trial)
                searches += 1
                slope = float(g @ step)
                norm2 = float(step @ step)
                roundoff = (
                    32
                    * np.finfo(float).eps
                    * max(roundoff_floor, abs(f), abs(trial_f), abs(old_total))
                )
                # At machine precision the value test cannot distinguish large
                # unstable steps. Keep the last step certified above roundoff.
                if (
                    stable_tau is not None
                    and norm2 / tau < 100 * roundoff
                    and tau > stable_tau
                ):
                    tau = stable_tau
                    continue
                majorized = trial_f <= f + slope + 0.5 * norm2 / tau + roundoff
                composite_slope = slope + trial_reg - reg
                # Secant curvature along the step (exact for a quadratic).
                excess = trial_f - f - slope
                curvature = 2.0 * excess / norm2 if excess > roundoff else 0.0
                if (
                    majorized
                    and composite_slope <= roundoff - 0.5 * norm2 / tau
                    and trial_f + trial_reg
                    <= old_total
                    + options.armijo_constant * min(composite_slope, 0.0)
                    + roundoff
                ):
                    if norm2 / tau >= 100 * roundoff:
                        stable_tau = tau
                    accepted = True
                    break
                if curvature_steps and not majorized and excess > 0:
                    tau = min(tau * options.backtrack_factor, 0.5 * norm2 / excess)
                    continue
                tau *= options.backtrack_factor
        except _EvaluationLimit:
            return finish(
                False,
                -2,
                "objective evaluation limit reached during line search",
                iteration - 1,
            )
        if not accepted:
            return finish(False, -1, "composite line search failed", iteration - 1)
        x = trial
        f, reg = trial_f, trial_reg
        g = evaluate_gradient(x)
        gradients += 1
        reduction, momentum = stopping.update(old_total, f + reg)
        if callback is not None:
            callback(
                InexactNewtonIteration(
                    iteration=iteration,
                    model=_read_only(x),
                    objective=f + reg,
                    objective_relative_reduction=reduction,
                    objective_reduction_momentum=momentum,
                    gradient=_read_only(g),
                    linearization_gradient=_read_only(old_gradient),
                    projected_gradient_norm=mapping,
                    step=_read_only(step),
                    raw_step=_read_only(-tau * old_gradient),
                    directional_derivative=composite_slope,
                    raw_directional_derivative=-tau
                    * float(old_gradient @ old_gradient),
                    step_length=tau,
                    forcing=0.0,
                    cg_iterations=0,
                    cg_residual_norm=0.0,
                    cg_relative_residual=0.0,
                    step_transform_cosine=1.0,
                    step_transform_norm_ratio=1.0,
                    negative_curvature=False,
                    steepest_descent_fallback=False,
                    line_search_evaluations=trials,
                    objective_evaluations=evaluations,
                    gradient_evaluations=gradients,
                    hessian_products=0,
                )
            )
        if options.objective_target is not None and f + reg <= options.objective_target:
            return finish(True, 0, "objective target reached", iteration)
        if options.step_tolerance > 0 and np.linalg.norm(
            step
        ) <= options.step_tolerance * max(1.0, float(np.linalg.norm(previous))):
            return finish(True, 0, "step tolerance reached", iteration)
        if (
            iteration >= options.objective_minimum_iterations
            and stopping.objective_converged()
        ):
            return finish(True, 0, "objective tolerance reached", iteration)
        if norm2 / tau >= 100 * roundoff:
            growth = tau / options.backtrack_factor
            if not curvature_steps:
                growth = min(growth, first_step)
            elif curvature > 0:
                growth = min(growth, 1.0 / curvature)
            tau = max(tau, growth)
    return finish(False, 1, "maximum iterations reached", options.max_iterations)
