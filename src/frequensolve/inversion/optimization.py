"""Bounded line-search optimizers for real-valued models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "InexactNewtonIteration",
    "InexactNewtonOptions",
    "InexactNewtonResult",
    "LBFGSOptions",
    "minimize_inexact_newton",
    "minimize_lbfgs",
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
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=True)


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Validate one finite positive optimizer option."""

    normalized = float(value)
    valid = normalized >= 0.0 if allow_zero else normalized > 0.0
    if not np.isfinite(normalized) or not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return normalized


@dataclass(frozen=True)
class InexactNewtonOptions:
    """Controls for Newton-CG, Eisenstat-Walker forcing, and line search.

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
    gradient_tolerance: float = 1.0e-6
    step_tolerance: float = 1.0e-9
    objective_tolerance: float = 1.0e-9
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
            "gradient_tolerance",
            "step_tolerance",
            "objective_tolerance",
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
    """Controls for projected limited-memory BFGS and Armijo line search."""

    max_iterations: int = 50
    max_objective_evaluations: Optional[int] = None
    max_line_search_trials: Optional[int] = None
    history_size: int = 10
    gradient_tolerance: float = 1.0e-6
    step_tolerance: float = 1.0e-9
    objective_tolerance: float = 1.0e-9
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
            "gradient_tolerance",
            "step_tolerance",
            "objective_tolerance",
            "minimum_step_length",
            "curvature_tolerance",
            "bound_tolerance",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), field_name, allow_zero=True),
            )
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


@dataclass(frozen=True)
class InexactNewtonIteration:
    """Diagnostics for the initial point or one accepted outer iteration."""

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

    def apply_preconditioner(value: np.ndarray) -> np.ndarray:
        if preconditioner is None:
            return np.array(value, copy=True)
        result = _finite_vector(
            preconditioner(np.array(model, copy=True), np.array(value, copy=True)),
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
        hessian_direction = _finite_vector(
            hessian_product(np.array(model, copy=True), restricted_direction),
            name="Hessian product",
            size=model.size,
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
                hessian_step = _finite_vector(
                    hessian_product(np.array(model, copy=True), step),
                    name="Hessian product",
                    size=model.size,
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

    def evaluate(candidate: np.ndarray) -> float:
        nonlocal objective_evaluations
        limit = options.max_objective_evaluations
        if limit is not None and objective_evaluations >= limit:
            raise _EvaluationLimit
        value = float(objective(np.array(candidate, copy=True)))
        objective_evaluations += 1
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    def evaluate_gradient(candidate: np.ndarray) -> np.ndarray:
        nonlocal gradient_evaluations
        value = _finite_vector(
            gradient(np.array(candidate, copy=True)),
            name="gradient",
            size=model.size,
        )
        gradient_evaluations += 1
        return value

    value = evaluate(model)
    grad = evaluate_gradient(model)
    free, projected_gradient = _free_variables(
        model, grad, lower, upper, options.bound_tolerance
    )
    projected_norm = float(np.linalg.norm(projected_gradient))
    forcing = min(
        options.maximum_forcing, max(options.minimum_forcing, options.initial_forcing)
    )
    zero = np.zeros_like(model)
    if callback is not None:
        callback(
            InexactNewtonIteration(
                iteration=0,
                model=np.array(model, copy=True),
                objective=value,
                objective_relative_reduction=0.0,
                objective_reduction_momentum=0.0,
                gradient=np.array(grad, copy=True),
                linearization_gradient=np.array(grad, copy=True),
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
    if projected_norm <= options.gradient_tolerance:
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
    objective_reduction_momentum: Optional[float] = None
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
        direction = np.array(cg.step, copy=True)
        hessian_direction = np.array(cg.hessian_step, copy=True)
        directional_derivative = float(np.dot(grad, direction))
        fallback = (
            not np.isfinite(directional_derivative) or directional_derivative >= 0.0
        )
        if fallback:
            direction = -projected_gradient
            hessian_direction = _finite_vector(
                hessian_product(np.array(model, copy=True), direction),
                name="Hessian product",
                size=model.size,
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
                step_limit(np.array(model, copy=True), np.array(direction, copy=True))
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
        trial = np.array(model, copy=True)
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
                    step = _finite_vector(
                        step_transform(
                            np.array(model, copy=True), np.array(step, copy=True)
                        ),
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
        objective_relative_reduction = max(
            0.0,
            (old_value - value) / max(1.0, abs(old_value)),
        )
        if objective_reduction_momentum is None:
            objective_reduction_momentum = objective_relative_reduction
        else:
            momentum = options.objective_tolerance_momentum
            objective_reduction_momentum = (
                momentum * objective_reduction_momentum
                + (1.0 - momentum) * objective_relative_reduction
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
                    model=np.array(model, copy=True),
                    objective=value,
                    objective_relative_reduction=objective_relative_reduction,
                    objective_reduction_momentum=objective_reduction_momentum,
                    gradient=np.array(grad, copy=True),
                    linearization_gradient=np.array(old_gradient, copy=True),
                    projected_gradient_norm=projected_norm,
                    step=np.array(step, copy=True),
                    raw_step=np.array(bounded_raw_step, copy=True),
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
        elif projected_norm <= options.gradient_tolerance:
            status, message = 1, "projected gradient tolerance reached"
        elif float(np.linalg.norm(step)) <= options.step_tolerance * max(
            1.0, float(np.linalg.norm(model))
        ):
            status, message = 3, "step tolerance reached"
        elif (
            iteration >= options.objective_minimum_iterations
            and objective_reduction_momentum <= options.objective_tolerance
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

    def evaluate(candidate: np.ndarray) -> float:
        nonlocal objective_evaluations
        limit = options.max_objective_evaluations
        if limit is not None and objective_evaluations >= limit:
            raise _EvaluationLimit
        value = float(objective(np.array(candidate, copy=True)))
        objective_evaluations += 1
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    def evaluate_gradient(candidate: np.ndarray) -> np.ndarray:
        nonlocal gradient_evaluations
        value = _finite_vector(
            gradient(np.array(candidate, copy=True)),
            name="gradient",
            size=model.size,
        )
        gradient_evaluations += 1
        return value

    def initial_inverse_action(vector: np.ndarray) -> np.ndarray:
        if preconditioner is not None:
            return _finite_vector(
                preconditioner(np.array(model, copy=True), np.array(vector, copy=True)),
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

    value = evaluate(model)
    grad = evaluate_gradient(model)
    free, projected_gradient = _free_variables(
        model, grad, lower, upper, options.bound_tolerance
    )
    projected_norm = float(np.linalg.norm(projected_gradient))
    zero = np.zeros_like(model)
    if callback is not None:
        callback(
            InexactNewtonIteration(
                iteration=0,
                model=np.array(model, copy=True),
                objective=value,
                objective_relative_reduction=0.0,
                objective_reduction_momentum=0.0,
                gradient=np.array(grad, copy=True),
                linearization_gradient=np.array(grad, copy=True),
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
            )
        )
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
    if projected_norm <= options.gradient_tolerance:
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

    steps: list[np.ndarray] = []
    gradient_differences: list[np.ndarray] = []
    inverse_curvatures: list[float] = []
    objective_reduction_momentum: Optional[float] = None
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
            direction = -projected_gradient
            raw_directional_derivative = -float(
                np.dot(projected_gradient, projected_gradient)
            )
            steepest_descent_fallbacks += 1

        line_search_evaluations = 0
        accepted = False
        trial = np.array(model, copy=True)
        trial_value = value
        bounded_raw_step = np.zeros_like(model)
        step = np.zeros_like(model)
        recovery_attempted = fallback
        try:
            while not accepted:
                maximum_step = np.inf
                if step_limit is not None:
                    representation_step = float(
                        step_limit(
                            np.array(model, copy=True),
                            np.array(direction, copy=True),
                        )
                    )
                    if np.isnan(representation_step) or representation_step < 0.0:
                        raise ValueError("step_limit must return a nonnegative value")
                    maximum_step = min(maximum_step, representation_step)
                if maximum_step <= 0.0 or raw_directional_derivative >= 0.0:
                    break

                step_length = min(1.0, maximum_step)
                line_search_trials = 0
                while step_length >= options.minimum_step_length and (
                    options.max_line_search_trials is None
                    or line_search_trials < options.max_line_search_trials
                ):
                    line_search_trials += 1
                    raw_step = step_length * direction
                    step = raw_step
                    if step_transform is not None:
                        step = _finite_vector(
                            step_transform(
                                np.array(model, copy=True),
                                np.array(step, copy=True),
                            ),
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

                if accepted or recovery_attempted:
                    break

                # A stale L-BFGS metric can yield a descent direction whose
                # local model is nevertheless poor enough that every bounded
                # Armijo trial fails.  Discard that metric and give the
                # projected gradient one independent line search in the
                # supplied inverse-Hessian metric before abandoning the
                # nonlinear continuation stage.
                steps.clear()
                gradient_differences.clear()
                inverse_curvatures.clear()
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
        objective_relative_reduction = max(
            0.0,
            (old_value - value) / max(1.0, abs(old_value)),
        )
        if objective_reduction_momentum is None:
            objective_reduction_momentum = objective_relative_reduction
        else:
            momentum = options.objective_tolerance_momentum
            objective_reduction_momentum = (
                momentum * objective_reduction_momentum
                + (1.0 - momentum) * objective_relative_reduction
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
            steps.append(np.array(step, copy=True))
            gradient_differences.append(np.array(gradient_difference, copy=True))
            inverse_curvatures.append(1.0 / curvature)
            if len(steps) > options.history_size:
                del steps[0]
                del gradient_differences[0]
                del inverse_curvatures[0]

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
                    model=np.array(model, copy=True),
                    objective=value,
                    objective_relative_reduction=objective_relative_reduction,
                    objective_reduction_momentum=objective_reduction_momentum,
                    gradient=np.array(grad, copy=True),
                    linearization_gradient=np.array(old_gradient, copy=True),
                    projected_gradient_norm=projected_norm,
                    step=np.array(step, copy=True),
                    raw_step=np.array(bounded_raw_step, copy=True),
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
                )
            )

        status = 0
        message = ""
        if options.objective_target is not None and value <= options.objective_target:
            status, message = 4, "objective target reached"
        elif projected_norm <= options.gradient_tolerance:
            status, message = 1, "projected gradient tolerance reached"
        elif float(np.linalg.norm(step)) <= options.step_tolerance * max(
            1.0, float(np.linalg.norm(model))
        ):
            status, message = 3, "step tolerance reached"
        elif (
            iteration >= options.objective_minimum_iterations
            and objective_reduction_momentum <= options.objective_tolerance
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
