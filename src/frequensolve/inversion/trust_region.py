"""Gauss-Newton with a Levenberg-Marquardt trust region for small model spaces.

The method is the trust-region form of the Levenberg-Marquardt algorithm
(More 1978; MINPACK ``lmder``; Nocedal & Wright, *Numerical Optimization*,
2nd ed., ch. 4 and sec. 10.3; Madsen, Nielsen & Tingleff 2004). Every
accepted iterate ``x_k`` supplies the gradient ``g_k`` and an explicit
symmetric Gauss-Newton matrix ``H_k`` on the free coordinates, and each step
solves the scaled subproblem

``min_p q(p) = g^T p + 1/2 p^T H p  subject to  ||D p|| <= Delta``

exactly: the eigendecomposition of ``D^-1 H D^-1`` reduces it to the secular
equation for the multiplier ``lambda >= max(0, -lambda_min)`` (More & Sorensen
1983), including indefinite matrices and the hard case. ``lambda`` is the
Levenberg-Marquardt parameter of the step ``p = -(H + lambda D^T D)^-1 g``.
A trial ``x_k + p`` costs one objective value; the gain ratio
``rho = (f(x_k) - f(x_k + p)) / (q(0) - q(p))`` accepts it (``rho > eta``) and
updates the radius (Nocedal & Wright Algorithm 4.1). A rejected trial re-solves
the subproblem with the same matrix and a smaller radius.

Coordinates whose lower and upper bounds coincide are excluded from the
matrix. Other box bounds use an active-set projection: a coordinate at a
bound whose subproblem step would leave the box is frozen for that subproblem,
which is solved again on the remaining coordinates, and the final step is
clipped to the box. The gain ratio uses the clipped step.

This module knows nothing about the solver: the objective, its gradient and
its Gauss-Newton matrix are plain callables. Stopping tests, tolerances and
iteration records follow :mod:`frequensolve.inversion.optimization`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from frequensolve.inversion.optimization import (
    InexactNewtonIteration,
    InexactNewtonResult,
    _all_finite,
    _argument,
    _bounds,
    _EvaluationLimit,
    _finite_vector,
    _free_variables,
    _positive,
    _read_only,
    _restart_vector,
    _result,
    _shares,
    _StoppingCriteria,
    _validate_stopping_options,
)

__all__ = [
    "GaussNewtonIteration",
    "GaussNewtonOptions",
    "GaussNewtonRestart",
    "GaussNewtonResult",
    "TrustRegionStep",
    "minimize_gauss_newton_trust_region",
    "solve_trust_region_subproblem",
    "trust_region_gain_ratio",
    "update_trust_radius",
]


Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]
GaussNewtonMatrix = Callable[[np.ndarray, np.ndarray], np.ndarray]
IterationCallback = Callable[["GaussNewtonIteration"], None]

GAUSS_NEWTON_RESTART_SCHEMA = "fs-gauss-newton-restart-1"
_SCALINGS = ("identity", "marquardt")
_EPSILON = float(np.finfo(np.float64).eps)
# The multiplier is solved to this relative accuracy of ||D p|| = Delta;
# Newton's iteration on 1/||D p(lambda)|| converges quadratically, so the
# eigendecomposition dominates the cost of an exact solve.
_SECULAR_TOLERANCE = 1.0e-12
_SECULAR_ITERATIONS = 100
# Eigendecompositions kept per matrix: one per active set tried.
_SYSTEM_CACHE = 8


# ---------------------------------------------------------------------------
# options and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussNewtonOptions:
    """Controls for :func:`minimize_gauss_newton_trust_region`.

    ``initial_radius`` is the first trust radius in the scaled norm
    ``||D p||`` of optimizer coordinates; ``None`` uses
    ``min(||D p_GN||, 1)`` with the Gauss-Newton step ``p_GN = -H^+ g`` of the
    first matrix. ``max_radius=None`` leaves the radius unbounded (MINPACK).
    A trial is accepted when its gain ratio exceeds ``eta``
    (``0 <= eta < 1/4``, Nocedal & Wright Algorithm 4.1).

    ``scaling`` selects ``D``: ``"identity"`` (Levenberg) or ``"marquardt"``
    (MINPACK ``mode=1``: ``D_k = max(D_{k-1}, sqrt(diag H_k))`` with a zero
    first diagonal entry set to one), which makes the iteration invariant to a
    diagonal rescaling of the coordinates. ``max_trust_region_trials`` caps the
    subproblems tried in one outer iteration (``None``: no cap);
    ``max_explicit`` caps the number of free coordinates of the explicit
    matrix; ``refresh`` rebuilds the matrix every that many accepted
    iterations (the gradient is always current).

    Absolute/relative stopping tolerances, their defaults and the fixed
    ``initial_objective`` reference follow
    :class:`~frequensolve.inversion.optimization.InexactNewtonOptions`.
    ``step_tolerance`` is also the MINPACK ``xtol`` test: the iteration stops
    once a rejected trial shrinks the radius to at most
    ``step_tolerance * max(1, ||D x||)``.
    """

    max_iterations: int = 50
    max_objective_evaluations: Optional[int] = None
    max_trust_region_trials: Optional[int] = 10
    initial_radius: Optional[float] = None
    max_radius: Optional[float] = None
    eta: float = 1.0e-4
    scaling: str = "identity"
    max_explicit: int = 256
    refresh: int = 1
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
    bound_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        for field_name in (
            "max_iterations",
            "max_explicit",
            "refresh",
            "objective_minimum_iterations",
        ):
            value = int(getattr(self, field_name))
            if value < 1:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        for field_name in ("max_objective_evaluations", "max_trust_region_trials"):
            value = getattr(self, field_name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{field_name} must be positive when supplied")
            object.__setattr__(self, field_name, None if value is None else int(value))
        for field_name in ("initial_radius", "max_radius"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _positive(value, field_name))
        if (
            self.initial_radius is not None
            and self.max_radius is not None
            and self.initial_radius > self.max_radius
        ):
            raise ValueError("initial_radius cannot exceed max_radius")
        eta = _positive(self.eta, "eta", allow_zero=True)
        if eta >= 0.25:
            raise ValueError("eta must be less than 1/4")
        object.__setattr__(self, "eta", eta)
        if self.scaling not in _SCALINGS:
            raise ValueError(f"scaling must be one of {_SCALINGS}")
        for field_name in ("step_tolerance", "grad_abs_tol", "obj_abs_tol"):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), field_name, allow_zero=True),
            )
        object.__setattr__(
            self,
            "bound_tolerance",
            _positive(self.bound_tolerance, "bound_tolerance", allow_zero=True),
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


@dataclass(frozen=True, eq=False)
class GaussNewtonRestart:
    """Exact restart state of :func:`minimize_gauss_newton_trust_region`.

    Every iteration record reports one as ``optimizer_state``
    (``fs-gauss-newton-restart-1``). ``model`` is the accepted iterate in
    optimizer coordinates, ``radius`` the trust radius of the next subproblem
    (``None`` before the first matrix fixed it), ``diagonal`` the Marquardt
    scaling ``D`` (``None`` for identity scaling or before the first matrix)
    and the remaining fields the stopping references. The arrays are
    read-only views of the optimizer's own; do not modify them.

    Passing the state back as ``restart=`` rebuilds the Gauss-Newton matrix at
    ``model`` and continues with the saved radius and scaling; with
    ``refresh=1`` the continued iteration is identical to an uninterrupted
    run.
    """

    model: np.ndarray
    radius: Optional[float]
    scaling: str
    diagonal: Optional[np.ndarray]
    accepted_iterations: int
    initial_objective: float
    improvement: Optional[float] = None
    schema: str = GAUSS_NEWTON_RESTART_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GAUSS_NEWTON_RESTART_SCHEMA:
            raise ValueError(
                f"Unsupported Gauss-Newton restart schema {self.schema!r}; "
                f"expected {GAUSS_NEWTON_RESTART_SCHEMA!r}"
            )
        model = self.model
        if not isinstance(model, np.ndarray) or model.ndim != 1 or not model.size:
            raise ValueError("Gauss-Newton restart model must be a non-empty vector")
        if self.radius is not None:
            object.__setattr__(
                self, "radius", _positive(self.radius, "restart trust radius")
            )
        if self.scaling not in _SCALINGS:
            raise ValueError(f"Gauss-Newton restart scaling must be one of {_SCALINGS}")
        if self.diagonal is not None:
            if (
                not isinstance(self.diagonal, np.ndarray)
                or self.diagonal.shape != model.shape
            ):
                raise ValueError("Gauss-Newton restart diagonal must match the model")
            if self.scaling != "marquardt":
                raise ValueError("only Marquardt scaling keeps a restart diagonal")
        accepted = int(self.accepted_iterations)
        if accepted < 0:
            raise ValueError("Gauss-Newton restart iterations must be nonnegative")
        object.__setattr__(self, "accepted_iterations", accepted)
        object.__setattr__(self, "initial_objective", float(self.initial_objective))
        if self.improvement is not None:
            object.__setattr__(self, "improvement", float(self.improvement))


@dataclass(frozen=True)
class GaussNewtonIteration(InexactNewtonIteration):
    """Diagnostics for the initial point or one accepted Gauss-Newton iteration.

    The common :class:`~frequensolve.inversion.optimization.InexactNewtonIteration`
    fields keep their meaning (``line_search_evaluations`` counts the trial
    objective values of the iteration, ``raw_step`` is the subproblem step
    before clipping to the box, ``negative_curvature`` flags an indefinite
    matrix); the CG and forcing fields are zero. The trust-region fields are:

    ``trust_radius`` the radius of the accepted subproblem and
    ``next_trust_radius`` the updated radius; ``marquardt_parameter`` the
    multiplier ``lambda`` of the accepted step; ``gain_ratio``,
    ``predicted_reduction`` and ``actual_reduction`` its ``rho = actual /
    predicted``; ``trust_region_trials`` the subproblems solved (rejections
    plus the accepted one); ``boundary_step`` and ``hard_case`` how the
    subproblem was solved; ``active_bounds`` the coordinates frozen by the
    active set; ``scaled_step_norm`` ``||D p||`` of the clipped step;
    ``hessian_refreshed`` and ``hessian_seconds`` whether the matrix was
    rebuilt for this iteration and the wall time of that call;
    ``eigenvalue_min`` and ``eigenvalue_max`` the spectrum range of the matrix
    in use. Radii and ``lambda`` are in optimizer coordinates.
    """

    optimizer_state: Optional[GaussNewtonRestart] = None  # type: ignore[assignment]
    trust_radius: Optional[float] = None
    next_trust_radius: Optional[float] = None
    marquardt_parameter: float = 0.0
    gain_ratio: Optional[float] = None
    predicted_reduction: float = 0.0
    actual_reduction: float = 0.0
    trust_region_trials: int = 0
    boundary_step: bool = False
    hard_case: bool = False
    active_bounds: int = 0
    scaled_step_norm: float = 0.0
    hessian_refreshed: bool = False
    hessian_seconds: float = 0.0
    eigenvalue_min: Optional[float] = None
    eigenvalue_max: Optional[float] = None


@dataclass(frozen=True)
class GaussNewtonResult(InexactNewtonResult):
    """Terminal state and work counters of the trust-region Gauss-Newton method.

    ``hessian_products`` counts matrix columns (free coordinates per
    rebuild), ``line_search_evaluations`` trial objective values and
    ``negative_curvature_events`` indefinite matrices. ``trust_radius`` is the
    final radius, ``hessian_refreshes`` the number of matrices built and
    ``trust_region_trials`` the subproblems solved.
    """

    trust_radius: Optional[float] = None
    hessian_refreshes: int = 0
    trust_region_trials: int = 0


@dataclass(frozen=True)
class TrustRegionStep:
    """Solution of one scaled trust-region subproblem.

    ``step`` is ``p``; ``parameter`` the Levenberg-Marquardt multiplier
    ``lambda`` with ``(H + lambda D^T D) p = -g``; ``predicted_reduction``
    ``q(0) - q(p)``; ``scaled_norm`` ``||D p||``. ``boundary`` marks a solution
    on the trust-region boundary and ``hard_case`` the More-Sorensen hard case
    (``lambda = -lambda_min`` with a component along its eigenvector).
    """

    step: np.ndarray
    parameter: float
    predicted_reduction: float
    scaled_norm: float
    boundary: bool
    hard_case: bool


# ---------------------------------------------------------------------------
# subproblem
# ---------------------------------------------------------------------------


def _secular_solution(
    gamma: np.ndarray, values: np.ndarray, radius: float
) -> Tuple[np.ndarray, float, bool, bool]:
    """Solve ``min c.gamma + 1/2 sum(values c^2)`` subject to ``||c|| <= radius``.

    ``values`` are the ascending eigenvalues of the scaled matrix (round-off
    entries already zero) and ``gamma`` the scaled gradient in its eigenbasis.
    Returns the coefficients, the multiplier and the boundary and hard-case
    flags (Nocedal & Wright sec. 4.3, More & Sorensen 1983).
    """

    size = values.size
    tolerance = size * _EPSILON * float(np.max(np.abs(values)))
    lowest = float(values[0])
    shift = max(0.0, -lowest)
    # The unknown is the offset delta = lambda - shift >= 0 of the multiplier
    # from the pole of ||c(lambda)|| at lambda = -lambda_min; the shifted
    # spectrum is formed once, so delta keeps full floating-point resolution
    # next to the pole (iterating on lambda itself cannot resolve
    # delta << shift). Eigenvalues within the numerical-rank tolerance of the
    # lowest one belong to the pole.
    offsets = values + shift
    pole = offsets <= tolerance if lowest <= 0.0 else np.zeros(size, dtype=bool)
    rest = ~pole
    coefficients = np.zeros(size)
    coefficients[rest] = -gamma[rest] / offsets[rest]
    rest_norm = float(np.linalg.norm(coefficients))
    pole_norm = float(np.linalg.norm(gamma[pole]))
    if rest_norm <= radius:
        if not pole.any():
            # Positive definite with an interior minimizer: the Gauss-Newton step.
            return coefficients, 0.0, False, False
        room = math.sqrt(max(radius * radius - rest_norm * rest_norm, 0.0))
        # The root lies within pole_norm / room of the pole; below the
        # resolution of the spectrum the limit solution is exact.
        if pole_norm <= tolerance * room:
            if shift == 0.0:
                # Singular positive semidefinite model: every null-space
                # addition leaves q unchanged; take the minimum-norm minimizer.
                return coefficients, 0.0, False, False
            direction = np.zeros(int(np.count_nonzero(pole)))
            if pole_norm > 0.0:
                direction = -gamma[pole] / pole_norm
            else:
                direction[0] = 1.0
            coefficients[pole] = room * direction
            return coefficients, shift, True, True

    # Boundary solution: Newton's method on 1/||c|| - 1/radius (Nocedal &
    # Wright Algorithm 4.3), safeguarded by bisection on [low, high]; with
    # every shifted eigenvalue nonnegative, ||c(delta)|| <= ||gamma|| / delta.
    low = 0.0
    high = float(np.linalg.norm(gamma)) / radius
    delta = min(pole_norm / radius, high) if pole_norm > 0.0 else 0.0
    best = (coefficients, delta)
    for _ in range(_SECULAR_ITERATIONS):
        shifted = offsets + delta
        with np.errstate(divide="ignore", invalid="ignore"):
            trial = np.where(gamma != 0.0, -gamma / shifted, 0.0)
        norm = float(np.linalg.norm(trial))
        if not math.isfinite(norm):
            low = delta
            candidate = 0.5 * (low + high)
        else:
            best = (trial, delta)
            if abs(norm - radius) <= _SECULAR_TOLERANCE * radius:
                break
            if norm > radius:
                low = delta
            else:
                high = delta
            nonzero = trial != 0.0
            curvature = float(np.sum(trial[nonzero] ** 2 / shifted[nonzero]))
            candidate = delta + (norm * norm / curvature) * (norm - radius) / radius
            if not (low < candidate < high):
                candidate = 0.5 * (low + high)
        if candidate == delta or high - low <= 4.0 * _EPSILON * high:
            break
        delta = candidate
    return best[0], shift + best[1], True, False


class _Eigensystem:
    """Eigendecomposition of one scaled matrix ``D^-1 H D^-1``."""

    def __init__(self, matrix: np.ndarray, diagonal: np.ndarray) -> None:
        scaled = matrix / np.outer(diagonal, diagonal)
        values, vectors = np.linalg.eigh(0.5 * (scaled + scaled.T))
        # Eigenvalues at the numerical-rank tolerance (``matrix_rank``) are
        # exact zeros of a Gauss-Newton matrix in round-off.
        tolerance = values.size * _EPSILON * float(np.max(np.abs(values)))
        values[np.abs(values) <= tolerance] = 0.0
        self.values = values
        self.vectors = vectors
        self.diagonal = diagonal

    def newton_norm(self, gradient: np.ndarray) -> float:
        """Return ``||D p_GN||`` of the pseudo-inverse step ``p_GN = -H^+ g``."""

        gamma = self.vectors.T @ (gradient / self.diagonal)
        nonzero = self.values != 0.0
        return float(np.linalg.norm(gamma[nonzero] / self.values[nonzero]))

    def solve(self, gradient: np.ndarray, radius: float) -> TrustRegionStep:
        gamma = self.vectors.T @ (gradient / self.diagonal)
        coefficients, parameter, boundary, hard = _secular_solution(
            gamma, self.values, radius
        )
        step = (self.vectors @ coefficients) / self.diagonal
        predicted = -float(
            gamma @ coefficients + 0.5 * self.values @ (coefficients * coefficients)
        )
        return TrustRegionStep(
            step=step,
            parameter=float(parameter),
            predicted_reduction=predicted,
            scaled_norm=float(np.linalg.norm(coefficients)),
            boundary=boundary,
            hard_case=hard,
        )


def solve_trust_region_subproblem(
    gradient: Any,
    hessian: Any,
    radius: float,
    *,
    diagonal: Any = None,
) -> TrustRegionStep:
    """Solve ``min g^T p + 1/2 p^T H p`` subject to ``||D p|| <= radius`` exactly.

    ``hessian`` is symmetrized and may be indefinite; ``diagonal`` holds the
    positive scaling ``D`` (identity when omitted). The solution satisfies
    ``(H + lambda D^T D) p = -g`` with ``H + lambda D^T D`` positive
    semidefinite and ``lambda (radius - ||D p||) = 0``. A singular positive
    semidefinite model whose minimizers fit inside the region returns the
    minimum-norm one; the hard case adds the eigenvector component that
    reaches the boundary.
    """

    vector = _finite_vector(gradient, name="gradient")
    matrix = np.asarray(hessian)
    if np.iscomplexobj(matrix):
        raise ValueError("hessian must be real-valued")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (vector.size, vector.size) or not _all_finite(matrix):
        raise ValueError("hessian must be a finite square matrix matching the gradient")
    scale = (
        np.ones(vector.size)
        if diagonal is None
        else _finite_vector(diagonal, name="diagonal", size=vector.size)
    )
    if np.any(scale <= 0.0):
        raise ValueError("diagonal must be positive")
    radius = _positive(radius, "radius")
    solution = _Eigensystem(matrix, scale).solve(vector, radius)
    symmetric = 0.5 * (matrix + matrix.T)
    step = solution.step
    return TrustRegionStep(
        step=step,
        parameter=solution.parameter,
        predicted_reduction=-float(vector @ step + 0.5 * step @ (symmetric @ step)),
        scaled_norm=solution.scaled_norm,
        boundary=solution.boundary,
        hard_case=solution.hard_case,
    )


def trust_region_gain_ratio(
    actual_reduction: float, predicted_reduction: float
) -> float:
    """Return ``rho = actual / predicted``; zero when no decrease is predicted.

    The zero follows MINPACK, which sets the ratio to zero for a vanishing
    predicted reduction, so such a trial is always rejected.
    """

    predicted = float(predicted_reduction)
    if not predicted > 0.0:
        return 0.0
    return float(actual_reduction) / predicted


def update_trust_radius(
    radius: float,
    gain_ratio: float,
    step_norm: float,
    *,
    max_radius: Optional[float] = None,
) -> float:
    """Return the next trust radius (Nocedal & Wright Algorithm 4.1, MINPACK).

    ``rho < 1/4`` shrinks the radius to ``||D p|| / 4``; ``rho > 3/4`` with a
    step on the boundary (``||D p|| >= 0.99 radius``) doubles it up to
    ``max_radius``; otherwise it is kept.
    """

    radius = _positive(radius, "radius")
    step_norm = _positive(step_norm, "step_norm", allow_zero=True)
    ratio = float(gain_ratio)
    if not ratio >= 0.25:
        return 0.25 * step_norm
    if ratio > 0.75 and step_norm >= 0.99 * radius:
        return min(2.0 * radius, math.inf if max_radius is None else float(max_radius))
    return radius


def _system(
    systems: Dict[Optional[bytes], _Eigensystem],
    matrix: np.ndarray,
    diagonal: np.ndarray,
    keep: Optional[np.ndarray],
) -> _Eigensystem:
    """Return the cached eigensystem of the principal submatrix ``keep``."""

    key = None if keep is None or keep.all() else keep.tobytes()
    system = systems.get(key)
    if system is None:
        if len(systems) >= _SYSTEM_CACHE:
            systems.clear()
        if key is None:
            system = _Eigensystem(matrix, diagonal)
        else:
            assert keep is not None
            system = _Eigensystem(matrix[np.ix_(keep, keep)], diagonal[keep])
        systems[key] = system
    return system


def _bounded_step(
    systems: Dict[Optional[bytes], _Eigensystem],
    matrix: np.ndarray,
    diagonal: np.ndarray,
    gradient: np.ndarray,
    model: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    radius: float,
    tolerance: float,
) -> Tuple[np.ndarray, TrustRegionStep, int]:
    """Solve the subproblem with the active-set projection of the box.

    A coordinate at a bound whose step would leave the box is frozen and the
    subproblem solved again on the others, until no step leaves the box from
    a bound. Returns the (unclipped) step on the free coordinates, the last
    subproblem solution and the number of frozen coordinates.
    """

    size = gradient.size
    if lower is None and upper is None:
        solution = _system(systems, matrix, diagonal, None).solve(gradient, radius)
        return solution.step, solution, 0
    scale = np.maximum(1.0, np.abs(model))
    at_lower = (
        np.zeros(size, dtype=bool)
        if lower is None
        else model <= lower + tolerance * scale
    )
    at_upper = (
        np.zeros(size, dtype=bool)
        if upper is None
        else model >= upper - tolerance * scale
    )
    keep = np.ones(size, dtype=bool)
    while keep.any():
        solution = _system(systems, matrix, diagonal, keep).solve(
            gradient[keep], radius
        )
        step = np.zeros(size)
        step[keep] = solution.step
        leaving = keep & ((at_lower & (step < 0.0)) | (at_upper & (step > 0.0)))
        if not leaving.any():
            return step, solution, int(size - np.count_nonzero(keep))
        keep = keep & ~leaving
    empty = TrustRegionStep(np.zeros(0), 0.0, 0.0, 0.0, False, False)
    return np.zeros(size), empty, size


def _marquardt_diagonal(
    previous: Optional[np.ndarray], matrix: np.ndarray, free: np.ndarray, size: int
) -> np.ndarray:
    """Return MINPACK's ``D_k = max(D_{k-1}, sqrt(diag H_k))`` (full, read-only)."""

    column = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    values = (
        np.where(column > 0.0, column, 1.0)
        if previous is None
        else np.maximum(previous[free], column)
    )
    diagonal = np.ones(size)
    diagonal[free] = values
    diagonal.flags.writeable = False
    return diagonal


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def minimize_gauss_newton_trust_region(
    objective: Objective,
    gradient: Gradient,
    hessian: GaussNewtonMatrix,
    initial_model: Sequence[float],
    *,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    options: Optional[GaussNewtonOptions] = None,
    callback: Optional[IterationCallback] = None,
    restart: Optional[GaussNewtonRestart] = None,
) -> GaussNewtonResult:
    """Minimize a least-squares objective with trust-region Gauss-Newton.

    ``objective(x)`` returns the value; ``gradient(x)`` the gradient at an
    accepted iterate; ``hessian(x, free)`` the ``(k, k)`` Gauss-Newton matrix
    on the coordinates ``free`` (a read-only index array, every coordinate
    whose bounds differ), requested only when the next subproblem needs it
    (every ``options.refresh`` accepted iterations, never at a converged
    iterate). The matrix is symmetrized; it may be indefinite. Bounds use the
    SciPy ``(lower, upper)`` convention.

    Each iteration solves the scaled trust-region subproblem exactly (see
    :func:`solve_trust_region_subproblem`) under the active-set projection
    described in the module documentation, evaluates the trial value and
    accepts it when the gain ratio exceeds ``options.eta``; rejected trials
    re-solve with the same matrix and the updated radius
    (:func:`update_trust_radius`). The gradient is evaluated only at accepted
    iterates.

    ``callback`` receives a :class:`GaussNewtonIteration` for the initial
    point and every accepted iterate; each carries a
    :class:`GaussNewtonRestart` from which ``restart=`` continues. Statuses
    follow :func:`~frequensolve.inversion.optimization.minimize_inexact_newton`:
    ``1`` gradient, ``2`` objective decrease, ``3`` step (or radius below the
    step tolerance), ``4`` objective target, ``0`` iteration budget, ``-1``
    trust-region trials exhausted, ``-2`` evaluation limit and ``-3`` no
    feasible step.
    """

    if not callable(objective) or not callable(gradient) or not callable(hessian):
        raise TypeError("objective, gradient, and hessian must be callable")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    options = GaussNewtonOptions() if options is None else options
    if not isinstance(options, GaussNewtonOptions):
        raise TypeError("options must be GaussNewtonOptions")
    model = _finite_vector(initial_model, name="initial model")
    size = model.size
    lower, upper = _bounds(bounds, size, None)
    if (lower is not None and np.any(model < lower)) or (
        upper is not None and np.any(model > upper)
    ):
        raise ValueError("initial model violates its bounds")
    pinned = (
        np.zeros(size, dtype=bool) if lower is None or upper is None else lower == upper
    )
    free = np.flatnonzero(~pinned)
    free.flags.writeable = False
    count = free.size
    if count > options.max_explicit:
        raise ValueError(
            f"the explicit Gauss-Newton matrix has {count} free coordinates, more "
            f"than max_explicit={options.max_explicit}; use matrix-free Newton-CG "
            "(minimize_inexact_newton) for larger spaces or raise max_explicit"
        )
    free_lower = None if lower is None else lower[free]
    free_upper = None if upper is None else upper[free]
    marquardt = options.scaling == "marquardt"
    max_radius = math.inf if options.max_radius is None else options.max_radius

    objective_evaluations = 0
    gradient_evaluations = 0
    hessian_products = 0
    trial_evaluations = 0
    negative_curvature_events = 0
    refreshes = 0
    total_trials = 0
    # SDK callables marked by ``_shares_arrays`` are passed read-only views.
    shares_objective, shares_gradient = _shares(objective), _shares(gradient)
    shares_hessian = _shares(hessian)

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
            size=size,
        )
        gradient_evaluations += 1
        return value

    def evaluate_hessian(candidate: np.ndarray) -> Tuple[np.ndarray, float]:
        started = perf_counter()
        matrix = np.asarray(hessian(_argument(candidate, shares_hessian), free))
        seconds = perf_counter() - started
        if np.iscomplexobj(matrix):
            raise ValueError("Gauss-Newton matrix must be real-valued")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (count, count):
            raise ValueError(
                f"Gauss-Newton matrix must have shape ({count}, {count}); "
                f"got {matrix.shape}"
            )
        if not _all_finite(matrix):
            raise ValueError("Gauss-Newton matrix must contain only finite values")
        return 0.5 * (matrix + matrix.T), seconds

    accepted_before = 0
    radius: Optional[float] = None
    diagonal: Optional[np.ndarray] = None
    if restart is not None:
        if not isinstance(restart, GaussNewtonRestart):
            raise TypeError("Gauss-Newton restart must be a GaussNewtonRestart")
        if restart.model.shape != model.shape or not np.array_equal(
            restart.model, model
        ):
            raise ValueError(
                "Gauss-Newton restart belongs to a different model or scaling"
            )
        if restart.scaling != options.scaling:
            raise ValueError("Gauss-Newton restart scaling mode changed")
        if restart.diagonal is not None:
            diagonal = _restart_vector(restart.diagonal, "restart diagonal", size)
            if np.any(diagonal <= 0.0):
                raise ValueError("Gauss-Newton restart diagonal must be positive")
        if restart.radius is not None:
            radius = min(restart.radius, max_radius)
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
        raise ValueError("Gauss-Newton restart initial objective changed")
    stopping = _StoppingCriteria(options, reference)
    if restart is not None:
        improvement = restart.improvement
        if improvement is not None and (
            not np.isfinite(improvement) or improvement < 0
        ):
            raise ValueError("Gauss-Newton restart has invalid stopping momentum")
        stopping.improvement = improvement

    def restart_state(iteration: int) -> GaussNewtonRestart:
        # Accepted iterates and diagonals are never written again: no copies.
        return GaussNewtonRestart(
            model=_read_only(model),
            radius=radius,
            scaling=options.scaling,
            diagonal=diagonal,
            accepted_iterations=accepted_before + iteration,
            initial_objective=stopping.reference,
            improvement=stopping.improvement,
        )

    def finish(
        success: bool, status: int, message: str, iterations: int
    ) -> GaussNewtonResult:
        return GaussNewtonResult(
            success,
            status,
            message,
            model,
            value,
            grad,
            iterations,
            objective_evaluations,
            gradient_evaluations,
            hessian_products,
            0,
            trial_evaluations,
            negative_curvature_events,
            0,
            trust_radius=radius,
            hessian_refreshes=refreshes,
            trust_region_trials=total_trials,
        )

    grad = evaluate_gradient(model)
    _, projected_gradient = _free_variables(
        model, grad, lower, upper, options.bound_tolerance
    )
    projected_norm = float(np.linalg.norm(projected_gradient))
    if callback is not None:
        zero = _read_only(np.zeros_like(model))
        callback(
            GaussNewtonIteration(
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
                hessian_products=hessian_products,
                optimizer_state=restart_state(0),
                trust_radius=radius,
                next_trust_radius=radius,
            )
        )
        del zero  # Held by the record only.
    if options.objective_target is not None and value <= options.objective_target:
        return finish(True, 4, "objective target reached", 0)
    if projected_norm <= stopping.gradient_threshold:
        return finish(True, 1, "projected gradient tolerance reached", 0)

    matrix: Optional[np.ndarray] = None
    systems: Dict[Optional[bytes], _Eigensystem] = {}
    since_refresh = 0
    eigenvalue_min: Optional[float] = None
    eigenvalue_max: Optional[float] = None
    for iteration in range(1, options.max_iterations + 1):
        refreshed = False
        seconds = 0.0
        if matrix is None or since_refresh >= options.refresh:
            matrix, seconds = evaluate_hessian(model)
            refreshed = True
            refreshes += 1
            hessian_products += count
            since_refresh = 0
            spectrum = np.linalg.eigvalsh(matrix)
            eigenvalue_min, eigenvalue_max = float(spectrum[0]), float(spectrum[-1])
            if eigenvalue_min < 0.0:
                negative_curvature_events += 1
            if marquardt:
                diagonal = _marquardt_diagonal(diagonal, matrix, free, size)
            systems.clear()
        assert matrix is not None
        scale = np.ones(count) if diagonal is None else diagonal[free]
        free_model = model[free]
        free_gradient = grad[free]
        if radius is None:
            if options.initial_radius is not None:
                radius = options.initial_radius
            else:
                newton = _system(systems, matrix, scale, None).newton_norm(
                    free_gradient
                )
                radius = min(newton, 1.0) if newton > 0.0 else 1.0
            radius = min(radius, max_radius)

        trials = 0
        evaluations_before = trial_evaluations
        while True:
            trials += 1
            total_trials += 1
            raw, solution, frozen = _bounded_step(
                systems,
                matrix,
                scale,
                free_gradient,
                free_model,
                free_lower,
                free_upper,
                radius,
                options.bound_tolerance,
            )
            trial = np.array(model, copy=True)
            trial[free] += raw
            if lower is not None:
                np.maximum(trial, lower, out=trial)
            if upper is not None:
                np.minimum(trial, upper, out=trial)
            step_free = trial[free] - free_model
            if not np.any(step_free):
                return finish(False, -3, "no feasible trust-region step", iteration - 1)
            predicted = -float(
                free_gradient @ step_free + 0.5 * step_free @ (matrix @ step_free)
            )
            scaled_norm = float(np.linalg.norm(scale * step_free))
            actual = 0.0
            trial_value = value
            if predicted > 0.0:
                try:
                    trial_value = evaluate(trial)
                except _EvaluationLimit:
                    return finish(
                        False,
                        -2,
                        "objective evaluation limit reached during trust-region trials",
                        iteration - 1,
                    )
                trial_evaluations += 1
                actual = value - trial_value
            # A trial without predicted decrease is rejected unevaluated.
            ratio = trust_region_gain_ratio(actual, predicted)
            accepted_radius = radius
            radius = update_trust_radius(
                radius, ratio, scaled_norm, max_radius=max_radius
            )
            if ratio > options.eta:
                break
            if radius <= options.step_tolerance * max(
                1.0, float(np.linalg.norm(scale * free_model))
            ):
                return finish(
                    True, 3, "trust-region radius below step tolerance", iteration - 1
                )
            if (
                options.max_trust_region_trials is not None
                and trials >= options.max_trust_region_trials
            ):
                return finish(False, -1, "trust-region trials exhausted", iteration - 1)

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
        _, projected_gradient = _free_variables(
            model, grad, lower, upper, options.bound_tolerance
        )
        projected_norm = float(np.linalg.norm(projected_gradient))
        since_refresh += 1
        if callback is not None:
            raw_step = np.zeros(size)
            raw_step[free] = raw
            callback(
                GaussNewtonIteration(
                    iteration=iteration,
                    model=_read_only(model),
                    objective=value,
                    objective_relative_reduction=objective_relative_reduction,
                    objective_reduction_momentum=objective_reduction_momentum,
                    gradient=_read_only(grad),
                    linearization_gradient=_read_only(old_gradient),
                    projected_gradient_norm=projected_norm,
                    step=_read_only(step),
                    raw_step=_read_only(raw_step),
                    directional_derivative=float(np.dot(old_gradient, step)),
                    raw_directional_derivative=float(np.dot(old_gradient, raw_step)),
                    step_length=1.0,
                    forcing=0.0,
                    cg_iterations=0,
                    cg_residual_norm=0.0,
                    cg_relative_residual=0.0,
                    step_transform_cosine=1.0,
                    step_transform_norm_ratio=1.0,
                    negative_curvature=eigenvalue_min is not None
                    and eigenvalue_min < 0.0,
                    steepest_descent_fallback=False,
                    line_search_evaluations=trial_evaluations - evaluations_before,
                    objective_evaluations=objective_evaluations,
                    gradient_evaluations=gradient_evaluations,
                    hessian_products=hessian_products,
                    optimizer_state=restart_state(iteration),
                    trust_radius=accepted_radius,
                    next_trust_radius=radius,
                    marquardt_parameter=solution.parameter,
                    gain_ratio=ratio,
                    predicted_reduction=predicted,
                    actual_reduction=actual,
                    trust_region_trials=trials,
                    boundary_step=solution.boundary,
                    hard_case=solution.hard_case,
                    active_bounds=frozen,
                    scaled_step_norm=scaled_norm,
                    hessian_refreshed=refreshed,
                    hessian_seconds=seconds,
                    eigenvalue_min=eigenvalue_min,
                    eigenvalue_max=eigenvalue_max,
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
            return finish(True, status, message, iteration)

    return finish(False, 0, "maximum outer iterations reached", options.max_iterations)
