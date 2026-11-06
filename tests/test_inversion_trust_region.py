"""Unit tests of the trust-region Gauss-Newton core (no solver)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import lsq_linear

from frequensolve.inversion import (
    GaussNewtonIteration,
    GaussNewtonOptions,
    GaussNewtonRestart,
    minimize_gauss_newton_trust_region,
    solve_trust_region_subproblem,
    trust_region_gain_ratio,
    update_trust_radius,
)

pytestmark = pytest.mark.unit

TIGHT = dict(grad_abs_tol=1e-10, grad_rel_tole=0.0, obj_rel_tol=0.0, step_tolerance=0.0)


class _LinearLeastSquares:
    """``f(x) = 1/2 ||A x - b||^2`` with its exact Gauss-Newton matrix."""

    def __init__(self, A, b):
        self.A, self.b = np.asarray(A, float), np.asarray(b, float)
        self.free = []

    def value(self, x):
        residual = self.A @ x - self.b
        return 0.5 * float(residual @ residual)

    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.b)

    def hessian(self, x, free):
        self.free.append(np.array(free))
        return (self.A.T @ self.A)[np.ix_(free, free)]

    def solve(self, x0, **kwargs):
        return minimize_gauss_newton_trust_region(
            self.value, self.gradient, self.hessian, x0, **kwargs
        )


def _conditioned(condition, rows=10, columns=6, seed=0):
    """Return ``A, b`` with ``cond(A^T A) == condition``."""

    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((rows, columns)))
    right, _ = np.linalg.qr(rng.standard_normal((columns, columns)))
    singular = np.logspace(0.0, -0.5 * np.log10(condition), columns)
    return left @ np.diag(singular) @ right.T, rng.standard_normal(rows)


def _rosenbrock(scaling="identity", **options):
    """Rosenbrock as least squares: r = (10 (x2 - x1^2), 1 - x1)."""

    def residual(x):
        return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

    def jacobian(x):
        return np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])

    def value(x):
        r = residual(x)
        return 0.5 * float(r @ r)

    def gradient(x):
        return jacobian(x).T @ residual(x)

    def hessian(x, free):
        J = jacobian(x)
        return (J.T @ J)[np.ix_(free, free)]

    settings = dict(max_iterations=200, scaling=scaling, **TIGHT)
    settings.update(options)
    return value, gradient, hessian, GaussNewtonOptions(**settings)


def _assert_trust_region_optimal(gradient, hessian, radius, step, diagonal=None):
    """Check the More-Sorensen global optimality conditions of a step."""

    gradient, hessian = np.asarray(gradient, float), np.asarray(hessian, float)
    D = np.ones(gradient.size) if diagonal is None else np.asarray(diagonal, float)
    p = step.step
    shifted = hessian + step.parameter * np.diag(D**2)
    scale = max(np.abs(np.linalg.eigvalsh(hessian)).max(), 1.0)
    size = max(1.0, float(np.linalg.norm(p)))
    assert step.parameter >= 0.0
    assert np.linalg.norm(D * p) <= radius * (1.0 + 1e-10)
    np.testing.assert_allclose(shifted @ p, -gradient, atol=1e-10 * scale * size)
    assert np.linalg.eigvalsh(shifted).min() >= -1e-10 * scale
    if step.parameter > 0.0:
        assert np.linalg.norm(D * p) == pytest.approx(radius, rel=1e-10)
    q = gradient @ p + 0.5 * p @ hessian @ p
    roundoff = 1e-12 * (abs(gradient @ p) + scale * float(p @ p))
    assert step.predicted_reduction == pytest.approx(-q, rel=1e-10, abs=roundoff)


# ---------------------------------------------------------------------------
# subproblem and radius logic
# ---------------------------------------------------------------------------


def test_large_radius_reproduces_gauss_newton_on_an_ill_conditioned_linear_problem():
    A, b = _conditioned(1e6)
    problem = _LinearLeastSquares(A, b)
    records = []
    result = problem.solve(
        np.zeros(6),
        options=GaussNewtonOptions(initial_radius=1e8),
        callback=records.append,
    )

    np.testing.assert_allclose(
        result.x, np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6, atol=1e-8
    )
    assert result.success and result.status == 1 and result.iterations == 1
    step = records[1]
    assert isinstance(step, GaussNewtonIteration)
    # A linear residual has an exact quadratic model.
    assert step.gain_ratio == pytest.approx(1.0, abs=1e-8)
    assert step.marquardt_parameter == 0.0 and not step.boundary_step
    assert step.trust_region_trials == 1 and step.line_search_evaluations == 1
    assert step.hessian_refreshed and step.hessian_products == 6
    eigenvalues = np.linalg.eigvalsh(A.T @ A)
    assert step.eigenvalue_min == pytest.approx(eigenvalues[0], rel=1e-6)
    assert step.eigenvalue_max == pytest.approx(eigenvalues[-1], rel=1e-10)
    # Converged at the first accepted iterate: no matrix is built there.
    assert result.hessian_refreshes == 1 and len(problem.free) == 1


def test_small_radius_gives_the_levenberg_step_with_unit_gain_ratio():
    A, b = _conditioned(1e6)
    problem = _LinearLeastSquares(A, b)
    H, g = A.T @ A, problem.gradient(np.zeros(6))
    radius = 1e-2
    step = solve_trust_region_subproblem(g, H, radius)
    _assert_trust_region_optimal(g, H, radius, step)
    assert step.boundary and step.parameter > 0.0
    np.testing.assert_allclose(
        step.step, -np.linalg.solve(H + step.parameter * np.eye(6), g), rtol=1e-10
    )
    # Marquardt scaling: (H + lambda D^2) p = -g with ||D p|| = radius.
    D = np.linspace(0.5, 3.0, 6)
    scaled = solve_trust_region_subproblem(g, H, radius, diagonal=D)
    _assert_trust_region_optimal(g, H, radius, scaled, diagonal=D)

    records = []
    problem.solve(
        np.zeros(6),
        options=GaussNewtonOptions(initial_radius=radius, max_iterations=3),
        callback=records.append,
    )
    for index, record in enumerate(records[1:]):
        # Boundary steps with rho = 1 double the radius every iteration.
        assert record.gain_ratio == pytest.approx(1.0, abs=1e-8)
        assert record.boundary_step and record.marquardt_parameter > 0.0
        assert record.trust_radius == radius * 2**index
        assert record.next_trust_radius == 2.0 * record.trust_radius
        assert record.scaled_step_norm == pytest.approx(record.trust_radius)
    np.testing.assert_allclose(records[1].step, step.step, rtol=1e-12, atol=1e-15)


def test_default_initial_radius_caps_the_gauss_newton_step_at_one():
    A, b = _conditioned(10.0, seed=4)
    problem = _LinearLeastSquares(A, b)
    records = []
    problem.solve(
        np.full(6, 100.0), options=GaussNewtonOptions(), callback=records.append
    )
    newton = np.linalg.norm(np.linalg.lstsq(A, b, rcond=None)[0] - 100.0)
    assert newton > 1.0 and records[1].trust_radius == 1.0
    small = _LinearLeastSquares(A, A @ np.full(6, 0.01))
    records = []
    small.solve(np.zeros(6), options=GaussNewtonOptions(), callback=records.append)
    assert records[1].trust_radius == pytest.approx(np.linalg.norm(np.full(6, 0.01)))
    assert not records[1].boundary_step


@pytest.mark.parametrize(
    "ratio, step_norm, expected",
    [
        (0.1, 0.8, 0.2),  # rho < 1/4: shrink to ||D p|| / 4
        (-5.0, 1.0, 0.25),
        (0.5, 1.0, 1.0),  # keep
        (0.9, 0.5, 1.0),  # interior step: keep
        (0.9, 0.995, 2.0),  # boundary step with rho > 3/4: double
    ],
)
def test_radius_update_follows_nocedal_wright_algorithm_4_1(ratio, step_norm, expected):
    assert update_trust_radius(1.0, ratio, step_norm) == pytest.approx(expected)


def test_radius_update_respects_the_maximum_and_gain_ratio_needs_a_decrease():
    assert update_trust_radius(1.0, 0.9, 1.0, max_radius=1.5) == 1.5
    assert trust_region_gain_ratio(3.0, 4.0) == 0.75
    assert trust_region_gain_ratio(-1.0, 4.0) == -0.25
    assert trust_region_gain_ratio(1.0, 0.0) == 0.0
    assert trust_region_gain_ratio(1.0, -2.0) == 0.0
    with pytest.raises(ValueError, match="radius"):
        update_trust_radius(0.0, 0.5, 1.0)


def test_indefinite_subproblems_and_the_hard_case_are_solved_globally():
    H = np.diag([-1.0, 2.0])
    # Ordinary indefinite case: lambda > 1 puts the step on the boundary.
    step = solve_trust_region_subproblem([1.0, 1.0], H, 2.0)
    _assert_trust_region_optimal([1.0, 1.0], H, 2.0, step)
    assert step.boundary and not step.hard_case and step.parameter > 1.0

    # Hard case: g has no component along the negative-curvature direction
    # and ||(H + I)^+ g|| = 1/3 < radius, so lambda = 1 and the eigenvector
    # completes the step to the boundary.
    hard = solve_trust_region_subproblem([0.0, 1.0], H, 2.0)
    _assert_trust_region_optimal([0.0, 1.0], H, 2.0, hard)
    assert hard.hard_case and hard.boundary
    assert hard.parameter == pytest.approx(1.0)
    assert hard.step[1] == pytest.approx(-1.0 / 3.0)
    assert abs(hard.step[0]) == pytest.approx(np.sqrt(4.0 - 1.0 / 9.0))

    # Nearly hard: the tiny component still selects the descending sign.
    near = solve_trust_region_subproblem([1e-14, 1.0], H, 2.0)
    _assert_trust_region_optimal([1e-14, 1.0], H, 2.0, near)
    assert near.step[0] == pytest.approx(-np.sqrt(4.0 - 1.0 / 9.0), rel=1e-6)


def test_singular_models_take_the_minimum_norm_or_steepest_descent_step():
    # Positive semidefinite with g in the range: the minimum-norm minimizer.
    singular = solve_trust_region_subproblem([1.0, 0.0], np.diag([1.0, 0.0]), 10.0)
    np.testing.assert_allclose(singular.step, [-1.0, 0.0])
    assert singular.parameter == 0.0 and not singular.boundary
    # A zero matrix is a linear model: the Cauchy step to the boundary.
    linear = solve_trust_region_subproblem([3.0, 4.0], np.zeros((2, 2)), 1.0)
    np.testing.assert_allclose(linear.step, [-0.6, -0.8])
    assert linear.parameter == pytest.approx(5.0) and linear.boundary


def test_random_subproblems_satisfy_the_global_optimality_conditions():
    rng = np.random.default_rng(7)
    for trial in range(300):
        size = int(rng.integers(1, 9))
        basis, _ = np.linalg.qr(rng.standard_normal((size, size)))
        spectrum = rng.standard_normal(size) * 10 ** rng.uniform(-6, 3, size)
        H = basis @ np.diag(spectrum) @ basis.T
        g = rng.standard_normal(size)
        if trial % 3 == 0:  # (nearly) hard: drop the lowest eigenvector part
            lowest = basis[:, np.argmin(spectrum)]
            g -= (g @ lowest) * lowest
        D = np.exp(rng.uniform(-1.0, 1.0, size)) if trial % 2 else None
        radius = 10 ** rng.uniform(-3, 3)
        step = solve_trust_region_subproblem(g, H, radius, diagonal=D)
        _assert_trust_region_optimal(g, H, radius, step, diagonal=D)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", ["identity", "marquardt"])
def test_rosenbrock_least_squares_converges_to_the_minimizer(scaling):
    value, gradient, hessian, options = _rosenbrock(scaling)
    records = []
    result = minimize_gauss_newton_trust_region(
        value, gradient, hessian, [-1.2, 1.0], options=options, callback=records.append
    )

    assert result.success and result.status == 1
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-8)
    objectives = [record.objective for record in records]
    assert np.all(np.diff(objectives) < 0.0)
    assert all(record.gain_ratio > options.eta for record in records[1:])
    assert result.hessian_refreshes == result.iterations
    assert result.trust_region_trials >= result.iterations
    if scaling == "marquardt":
        diagonals = [r.optimizer_state.diagonal for r in records[1:]]
        assert all(np.all(d >= 1e-300) for d in diagonals)
        # MINPACK keeps the running maximum of sqrt(diag H).
        assert all(
            np.all(later >= earlier) for earlier, later in zip(diagonals, diagonals[1:])
        )


def test_rejected_trials_reuse_the_matrix_and_shrink_the_radius():
    calls = []

    def hessian(x, free):
        calls.append(np.array(x))
        return np.ones((1, 1))  # the quadratic part only: a poor model of x^4

    def value(x):
        return 0.5 * x[0] ** 2 + x[0] ** 4

    def gradient(x):
        return np.array([x[0] + 4.0 * x[0] ** 3])

    records = []
    result = minimize_gauss_newton_trust_region(
        value,
        gradient,
        hessian,
        [2.0],
        options=GaussNewtonOptions(initial_radius=10.0, max_iterations=1),
        callback=records.append,
    )
    step = records[1]
    # Delta = 10 is rejected (rho < 0); Delta = ||p|| / 4 = 2.5 is accepted
    # with rho < 1/4, which shrinks the radius again.
    assert step.trust_region_trials == 2 and step.line_search_evaluations == 2
    assert step.trust_radius == 2.5 and step.next_trust_radius == 0.625
    assert 1e-4 < step.gain_ratio < 0.25
    np.testing.assert_allclose(step.model, [-0.5])
    assert len(calls) == 1 and result.status == 0

    exhausted = minimize_gauss_newton_trust_region(
        value,
        gradient,
        hessian,
        [2.0],
        options=GaussNewtonOptions(initial_radius=10.0, max_trust_region_trials=1),
    )
    assert not exhausted.success and exhausted.status == -1
    assert "trials exhausted" in exhausted.message
    np.testing.assert_array_equal(exhausted.x, [2.0])


def test_indefinite_matrix_starts_in_the_hard_case_and_reaches_a_minimizer():
    def value(x):
        return 1.0 - 0.5 * x[0] ** 2 + 0.25 * x[0] ** 4 + x[1] ** 2 + x[1]

    def gradient(x):
        return np.array([x[0] ** 3 - x[0], 2.0 * x[1] + 1.0])

    def hessian(x, free):
        return np.diag([3.0 * x[0] ** 2 - 1.0, 2.0])[np.ix_(free, free)]

    records = []
    result = minimize_gauss_newton_trust_region(
        value,
        gradient,
        hessian,
        [0.0, 0.0],  # a saddle in x1: the gradient is orthogonal to its eigenvector
        options=GaussNewtonOptions(**TIGHT),
        callback=records.append,
    )

    assert records[1].hard_case and records[1].negative_curvature
    assert records[1].eigenvalue_min == -1.0 and records[1].boundary_step
    assert result.success and result.negative_curvature_events >= 1
    np.testing.assert_allclose(np.abs(result.x[0]), 1.0, atol=1e-8)
    assert result.x[1] == pytest.approx(-0.5, abs=1e-10)


def test_pinned_coordinates_are_excluded_from_the_explicit_matrix():
    rng = np.random.default_rng(3)
    A, b = rng.standard_normal((8, 4)), rng.standard_normal(8)
    problem = _LinearLeastSquares(A, b)
    lower = np.array([-np.inf, 0.5, -np.inf, -np.inf])
    upper = np.array([np.inf, 0.5, np.inf, np.inf])
    result = problem.solve(
        [0.0, 0.5, 0.0, 0.0],
        bounds=(lower, upper),
        options=GaussNewtonOptions(initial_radius=1e6),
    )

    assert result.success and result.x[1] == 0.5
    free = [0, 2, 3]
    expected = np.linalg.lstsq(A[:, free], b - 0.5 * A[:, 1], rcond=None)[0]
    np.testing.assert_allclose(result.x[free], expected, rtol=1e-10)
    assert all(np.array_equal(columns, free) for columns in problem.free)
    assert result.hessian_products == 3 * result.hessian_refreshes


def test_box_bounds_freeze_outward_coordinates_and_clip_the_step():
    rng = np.random.default_rng(3)
    A = np.eye(4) + 0.1 * rng.standard_normal((4, 4))
    b = A @ np.array([5.0, -0.3, 0.2, -4.0])
    lower, upper = -np.ones(4), np.ones(4)
    problem = _LinearLeastSquares(A, b)
    records = []
    result = problem.solve(
        [1.0, 0.0, 0.0, 0.0],  # x1 at its upper bound, pushed outward
        bounds=(lower, upper),
        options=GaussNewtonOptions(initial_radius=10.0, **TIGHT),
        callback=records.append,
    )

    expected = lsq_linear(A, b, bounds=(lower, upper), tol=1e-14).x
    assert result.success and result.status == 1
    np.testing.assert_allclose(result.x, expected, atol=1e-8)
    for record in records:
        assert np.all(record.model >= lower) and np.all(record.model <= upper)
    # The first subproblem froze x1; later ones also the clipped x4.
    assert records[1].active_bounds >= 1
    assert np.any(records[1].raw_step != records[1].step)  # clipped


def test_restart_continues_the_radius_and_reproduces_the_uninterrupted_run():
    for scaling in ("identity", "marquardt"):
        value, gradient, hessian, options = _rosenbrock(scaling)
        records = []
        minimize_gauss_newton_trust_region(
            value,
            gradient,
            hessian,
            [-1.2, 1.0],
            options=options,
            callback=records.append,
        )
        cut = 4
        state = records[cut].optimizer_state
        assert isinstance(state, GaussNewtonRestart)
        assert state.radius == records[cut].next_trust_radius
        assert state.accepted_iterations == cut
        assert (state.diagonal is not None) == (scaling == "marquardt")
        assert not state.model.flags.writeable

        resumed = []
        result = minimize_gauss_newton_trust_region(
            value,
            gradient,
            hessian,
            state.model,
            options=options,
            callback=resumed.append,
            restart=state,
        )
        assert resumed[0].trust_radius == state.radius
        assert resumed[1].trust_radius == records[cut + 1].trust_radius
        assert len(resumed) == len(records) - cut
        for continued, original in zip(resumed[1:], records[cut + 1 :]):
            np.testing.assert_array_equal(continued.model, original.model)
            assert continued.trust_radius == original.trust_radius
            assert (
                continued.optimizer_state.accepted_iterations
                == original.optimizer_state.accepted_iterations
            )
        np.testing.assert_array_equal(result.x, records[-1].model)

    with pytest.raises(ValueError, match="different model"):
        minimize_gauss_newton_trust_region(
            value, gradient, hessian, [0.0, 0.0], options=options, restart=state
        )
    with pytest.raises(ValueError, match="scaling mode"):
        minimize_gauss_newton_trust_region(
            value,
            gradient,
            hessian,
            state.model,
            options=GaussNewtonOptions(scaling="identity"),
            restart=state,
        )


def test_objective_target_and_evaluation_limit_stop_the_iteration():
    value, gradient, hessian, options = _rosenbrock()
    target = minimize_gauss_newton_trust_region(
        value,
        gradient,
        hessian,
        [-1.2, 1.0],
        options=GaussNewtonOptions(objective_target=1.0, max_iterations=200),
    )
    assert target.success and target.status == 4 and target.fun <= 1.0
    limited = minimize_gauss_newton_trust_region(
        value,
        gradient,
        hessian,
        [-1.2, 1.0],
        options=GaussNewtonOptions(max_objective_evaluations=3, max_iterations=200),
    )
    assert not limited.success and limited.status == -2
    assert limited.objective_evaluations == 3


@pytest.mark.parametrize(
    "settings, message",
    [
        (dict(eta=0.25), "eta"),
        (dict(scaling="diagonal"), "scaling"),
        (dict(initial_radius=2.0, max_radius=1.0), "initial_radius"),
        (dict(initial_radius=0.0), "initial_radius"),
        (dict(refresh=0), "refresh"),
        (dict(max_explicit=0), "max_explicit"),
        (dict(max_trust_region_trials=0), "max_trust_region_trials"),
    ],
)
def test_options_are_validated(settings, message):
    with pytest.raises(ValueError, match=message):
        GaussNewtonOptions(**settings)


def test_spaces_beyond_max_explicit_point_to_newton_cg():
    problem = _LinearLeastSquares(np.eye(3), np.ones(3))
    with pytest.raises(ValueError, match="max_explicit=2.*Newton-CG"):
        problem.solve(np.zeros(3), options=GaussNewtonOptions(max_explicit=2))
    assert not problem.free  # rejected before any matrix


def test_refresh_reuses_the_matrix_between_rebuilds():
    value, gradient, hessian, _ = _rosenbrock()
    calls = []

    def counted(x, free):
        calls.append(1)
        return hessian(x, free)

    records = []
    result = minimize_gauss_newton_trust_region(
        value,
        gradient,
        counted,
        [-1.2, 1.0],
        options=GaussNewtonOptions(refresh=3, max_iterations=6, **TIGHT),
        callback=records.append,
    )
    refreshed = [record.hessian_refreshed for record in records[1:]]
    assert refreshed == [True, False, False, True, False, False][: len(refreshed)]
    assert len(calls) == result.hessian_refreshes == sum(refreshed)
