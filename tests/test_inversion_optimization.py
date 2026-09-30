from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares

from frequensolve.inversion import (
    ComplexDataRealifier,
    ContinuationSchedule,
    ControlLeastSquaresProblem,
    ControlObjectiveProblem,
    DiagonalInverseHessian,
    GaussNewtonDiagonalEstimate,
    InexactNewtonOptions,
    LBFGSOptions,
    LossTerms,
    OptimizationCheckpoint,
    OptimizationHistory,
    OptimizationResult,
    QuadraticRegularization,
    estimate_gauss_newton_diagonal,
    minimize_inexact_newton,
    minimize_lbfgs,
    run_continuation,
)
from frequensolve.inversion.optimization import minimize_proximal_gradient


@pytest.fixture(params=["newton", "lbfgs"])
def tolerance_solver(request):
    """A half-step quadratic with matched preconditioning at any objective scale."""

    def solve(*, scale=1.0, offset=0.0, **settings):
        options = dict(max_iterations=4, step_tolerance=0.0)
        options.update(settings)
        states = []
        args = (
            lambda x: scale * (offset + 0.5 * float(x @ x)),
            lambda x: scale * x,
        )
        kwargs = dict(
            preconditioner=lambda _x, g: g / scale,
            step_transform=lambda _x, step: step / 2,
            callback=states.append,
        )
        if request.param == "newton":
            result = minimize_inexact_newton(
                *args,
                lambda _x, v: scale * v,
                [2.0],
                options=InexactNewtonOptions(**options),
                **kwargs,
            )
        else:
            result = minimize_lbfgs(
                *args,
                [2.0],
                options=LBFGSOptions(**options),
                **kwargs,
            )
        return result, states

    return solve


@pytest.mark.parametrize("scale", [1e-23, 1.0, 1e9])
def test_gradient_tolerances_add_and_use_initial_objective(tolerance_solver, scale):
    # Initial F=12s, ||g||=2s. After one half-step ||g||=s:
    # 0.02s + 0.082 * 12s = 1.004s. Neither term alone suffices.
    result, _ = tolerance_solver(
        scale=scale,
        offset=10.0,
        grad_abs_tol=0.02 * scale,
        grad_rel_tole=0.082,
        obj_rel_tol=0.0,
    )
    assert result.success and result.status == 1 and result.iterations == 1
    np.testing.assert_allclose(result.model, [1.0])


@pytest.mark.parametrize("scale", [1e-23, 1.0, 1e9])
def test_objective_tolerances_use_fixed_initial_reference(tolerance_solver, scale):
    # Decreases: 1.5s, 0.375s. Stop on 0.05s + 0.17 * F0 = 0.39s.
    result, states = tolerance_solver(
        scale=scale,
        grad_rel_tole=0.0,
        obj_abs_tol=0.05 * scale,
        obj_rel_tol=0.17,
    )
    assert result.success and result.status == 2 and result.iterations == 2
    assert states[-1].objective_relative_reduction == pytest.approx(0.1875)


@pytest.mark.parametrize("initial_objective", [12.0, -12.0])
def test_restart_reference_uses_absolute_initial_objective(
    tolerance_solver, initial_objective
):
    result, _ = tolerance_solver(
        initial_objective=initial_objective,
        grad_rel_tole=0.085,
        obj_rel_tol=0.0,
    )
    assert result.success and result.status == 1 and result.iterations == 1


def test_zero_initial_objective_has_no_relative_floor(tolerance_solver):
    result, states = tolerance_solver(
        offset=-2.0,
        grad_rel_tole=100.0,
        obj_rel_tol=100.0,
        obj_abs_tol=0.4,
        objective_tolerance_momentum=0.0,
    )
    assert result.success and result.status == 2 and result.iterations == 2
    assert states[0].objective == 0.0
    assert np.isinf(states[-1].objective_relative_reduction)
    assert not np.isnan(states[-1].objective_reduction_momentum)


def test_tiny_objective_is_not_initially_converged_by_default(tolerance_solver):
    result, _ = tolerance_solver(scale=1e-23, max_iterations=1)
    assert result.iterations == 1


@pytest.mark.parametrize("options_type", [InexactNewtonOptions, LBFGSOptions])
@pytest.mark.parametrize(
    "field",
    [
        "grad_abs_tol",
        "grad_rel_tole",
        "obj_abs_tol",
        "obj_rel_tol",
    ],
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_absolute_relative_tolerances_are_validated(options_type, field, value):
    with pytest.raises(ValueError, match=field):
        options_type(**{field: value})


@pytest.mark.parametrize("options_type", [InexactNewtonOptions, LBFGSOptions])
def test_restart_reference_must_be_finite(options_type):
    with pytest.raises(ValueError, match="initial_objective"):
        options_type(initial_objective=float("nan"))


def test_proximal_relative_tolerance_includes_regularization():
    result = minimize_proximal_gradient(
        lambda x: 0.5 * float(x @ x),
        lambda x: x,
        lambda x: 10.0,
        lambda v, tau, bounds: v,
        [2.0],
        options=LBFGSOptions(grad_rel_tole=0.17),
    )
    # ||mapping||=2 <= 0.17 * (2+10), but not <= 0.17 * 2.
    assert result.success and result.iterations == 0
    assert result.objective == 12.0


class _ControlSpace:
    """Minimal control space for the toolkit: ``size`` plus mapping ``pack``."""

    def __init__(self, size):
        self.size = size

    def zeros(self):
        return np.zeros(self.size, dtype=np.float64)

    def pack(self, values):
        return np.concatenate(
            [np.asarray(block, dtype=np.float64).ravel() for block in values.values()]
        )


def test_complex_data_realifier_is_an_isometric_real_layout():
    data = np.array([[1.0 + 2.0j, -3.0 + 4.0j], [0.5 - 0.25j, 2.5 + 0.0j]])
    layout = ComplexDataRealifier(data.shape)

    packed = layout.pack(data)
    np.testing.assert_array_equal(packed, [1.0, 2.0, -3.0, 4.0, 0.5, -0.25, 2.5, 0.0])
    np.testing.assert_array_equal(layout.unpack(packed), data)

    dual = np.arange(1.0, layout.real_size + 1.0)
    assert np.dot(packed, dual) == pytest.approx(
        np.vdot(data, layout.unpack(dual)).real
    )

    jacobian = np.arange(1.0, 9.0).reshape(4, 2) + 1j * np.arange(9.0, 17.0).reshape(
        4, 2
    )
    direction = np.array([0.25, -0.5])
    np.testing.assert_allclose(
        layout.pack((jacobian @ direction).reshape(data.shape)),
        layout.pack_jacobian(jacobian, direction.size) @ direction,
    )

    scalar_layout = ComplexDataRealifier.from_data(1.0 + 2.0j)
    np.testing.assert_array_equal(scalar_layout.pack(1.0 + 2.0j), [1.0, 2.0])
    np.testing.assert_array_equal(scalar_layout.unpack([1.0, 2.0]), [1.0 + 2.0j])


def test_control_least_squares_drives_scipy_with_real_controls(tmp_path):
    matrix = np.array(
        [
            [1.0 + 2.0j, 0.5 - 1.0j],
            [-0.25 + 0.5j, 2.0 + 0.25j],
            [1.5, -0.75j],
        ]
    )
    truth = np.array([0.2, -0.1])
    observed = matrix @ truth
    history_path = tmp_path / "loss_history.json"
    history = OptimizationHistory(history_path, metadata={"benchmark": "unit"})
    checkpoints = []
    controls = _ControlSpace(2)
    problem = ControlLeastSquaresProblem(
        controls,
        observed,
        forward=lambda model: matrix @ model,
        jacobian=lambda model: matrix,
        history=history,
        iteration_callback=lambda model, loss: checkpoints.append((model.copy(), loss)),
    )

    result = least_squares(
        problem.residual,
        controls.zeros(),
        jac=problem.jacobian,
        bounds=(-0.5, 0.5),
        gtol=1.0e-12,
        ftol=1.0e-12,
        xtol=1.0e-12,
    )
    final_loss = problem.loss(result.x)
    history.finish("converged", result.message)

    assert result.success
    np.testing.assert_allclose(result.x, truth, rtol=1.0e-10, atol=1.0e-12)
    assert final_loss.total < 1.0e-24
    assert history.evaluation_count == result.nfev
    assert history.iteration_count == result.njev
    assert len(checkpoints) == result.njev
    np.testing.assert_array_equal(checkpoints[-1][0], result.x)
    assert checkpoints[-1][1] == final_loss
    assert np.all(np.diff(history.losses()) <= 0.0)

    loaded = OptimizationHistory.load(history_path)
    assert loaded.status == "converged"
    np.testing.assert_array_equal(loaded.losses(), history.losses())
    with pytest.raises(RuntimeError, match="finished"):
        loaded.record_evaluation(result.x, final_loss)


def test_native_control_objective_caches_atomic_value_and_gradient(tmp_path):
    controls = _ControlSpace(2)
    history = OptimizationHistory(tmp_path / "native_history.json")
    evaluations = []
    hessian = np.array([[3.0, 0.5], [0.5, 2.0]])
    truth = np.array([0.25, -0.4])

    def value_gradient(model):
        evaluations.append(model.copy())
        residual = model - truth
        return 0.5 * residual @ hessian @ residual, hessian @ residual

    problem = ControlObjectiveProblem(
        controls,
        value_gradient=value_gradient,
        history=history,
    )
    model = np.array([0.7, 0.2])

    value = problem.objective(model)
    gradient = problem.gradient(model)

    assert len(evaluations) == 1
    assert value == pytest.approx(0.5 * (model - truth) @ hessian @ (model - truth))
    np.testing.assert_allclose(gradient, hessian @ (model - truth))
    assert history.evaluation_count == 1


def test_control_least_squares_native_jvp_vjp_and_gauss_newton_product():
    matrix = np.array(
        [
            [1.0 + 0.5j, -0.25 + 1.0j],
            [0.5 - 0.75j, 2.0 + 0.25j],
            [-1.5 + 0.0j, 0.75 - 0.5j],
        ]
    )
    weights = np.array([0.5, 2.0, 1.25])
    controls = _ControlSpace(2)
    regularization = QuadraticRegularization([[1.0, -1.0]], weight=0.3)
    problem = ControlLeastSquaresProblem(
        controls,
        np.zeros(3, dtype=np.complex128),
        forward=lambda model: matrix @ model,
        jvp=lambda model, direction: matrix @ direction,
        vjp=lambda model, dual: np.real(matrix.conj().T @ dual),
        weights=weights,
        regularization=regularization,
        record_jacobian_iterations=False,
    )
    model = np.array([0.2, -0.1])
    direction = np.array([0.4, -0.3])
    dual = np.array([0.5 + 0.2j, -0.1 + 0.7j, 0.25 - 0.4j])

    incremental = problem.data_jvp(model, direction)
    transpose = problem.data_vjp(model, dual)
    assert np.vdot(incremental, dual).real == pytest.approx(
        np.dot(direction, transpose)
    )

    weighted_matrix = weights[:, None] * matrix
    regularization_jacobian = regularization.jacobian()
    expected = np.real(weighted_matrix.conj().T @ (weighted_matrix @ direction))
    expected += regularization_jacobian.T @ (regularization_jacobian @ direction)
    np.testing.assert_allclose(problem.gauss_newton_product(model, direction), expected)
    expected_gradient = np.real(weighted_matrix.conj().T @ (weights * (matrix @ model)))
    expected_gradient += regularization_jacobian.T @ regularization.residual(model)
    np.testing.assert_allclose(problem.gradient(model), expected_gradient)


def test_inexact_newton_solves_a_quadratic_with_matrix_free_products():
    hessian = np.array([[5.0, 1.0], [1.0, 2.0]])
    truth = np.array([0.25, -0.4])
    states = []

    def objective(model):
        residual = model - truth
        return 0.5 * residual @ hessian @ residual

    result = minimize_inexact_newton(
        objective,
        lambda model: hessian @ (model - truth),
        lambda model, direction: hessian @ direction,
        [1.5, 0.75],
        options=InexactNewtonOptions(
            gradient_tolerance=1.0e-12,
            step_tolerance=0.0,
            objective_tolerance=0.0,
            max_iterations=10,
        ),
        callback=states.append,
    )

    assert result.success
    np.testing.assert_allclose(result.model, truth, atol=1.0e-12)
    assert result.objective < 1.0e-24
    assert result.hessian_products >= 1
    assert states[0].iteration == 0
    np.testing.assert_array_equal(states[0].raw_step, np.zeros(2))
    assert states[-1].step_length == pytest.approx(1.0)
    np.testing.assert_allclose(states[-1].raw_step, states[-1].step)
    assert states[-1].cg_relative_residual <= states[-1].forcing
    assert states[-1].step_transform_cosine == pytest.approx(1.0)
    assert states[-1].step_transform_norm_ratio == pytest.approx(1.0)


def test_lbfgs_solves_a_bounded_quadratic_without_hessian_products():
    hessian = np.array([[5.0, 1.0], [1.0, 2.0]])
    truth = np.array([0.25, -0.4])
    states = []

    def objective(model):
        residual = model - truth
        return 0.5 * residual @ hessian @ residual

    result = minimize_lbfgs(
        objective,
        lambda model: hessian @ (model - truth),
        [1.5, 0.45],
        bounds=([-2.0, -0.5], [2.0, 0.5]),
        options=LBFGSOptions(
            gradient_tolerance=1.0e-11,
            step_tolerance=0.0,
            objective_tolerance=0.0,
            max_iterations=30,
        ),
        callback=states.append,
    )

    assert result.success
    np.testing.assert_allclose(result.model, truth, atol=1.0e-9)
    assert result.hessian_products == 0
    assert result.cg_iterations == 0
    assert states[0].iteration == 0
    assert all(state.hessian_products == 0 for state in states)


def test_lbfgs_uses_inverse_hessian_metric_without_changing_raw_gradients():
    hessian_diagonal = np.array([100.0, 0.25])
    truth = np.array([-0.5, 0.75])
    initial = np.array([1.0, -1.0])
    states = []

    def gradient(model):
        return hessian_diagonal * (model - truth)

    preconditioner = DiagonalInverseHessian(
        hessian_diagonal,
        relative_damping=0.0,
        maximum_inverse_ratio=None,
    )
    result = minimize_lbfgs(
        lambda model: (
            0.5 * float(np.dot(model - truth, hessian_diagonal * (model - truth)))
        ),
        gradient,
        initial,
        options=LBFGSOptions(
            max_iterations=2,
            gradient_tolerance=1.0e-12,
            step_tolerance=0.0,
            objective_tolerance=0.0,
        ),
        preconditioner=preconditioner,
        callback=states.append,
    )

    assert result.success
    np.testing.assert_allclose(result.model, truth, atol=1.0e-14)
    np.testing.assert_array_equal(states[0].gradient, gradient(initial))
    np.testing.assert_allclose(states[1].raw_step, truth - initial, atol=1.0e-14)
    assert result.hessian_products == 0


def test_randomized_vjp_diagonal_includes_weights_and_regularization():
    matrix = np.diag([2.0, 3.0]).astype(np.complex128)
    weights = np.array([0.5, 2.0])
    controls = _ControlSpace(2)
    regularization = QuadraticRegularization([[1.0, -1.0]], weight=0.25)
    vjp_calls = []
    problem = ControlLeastSquaresProblem(
        controls,
        np.zeros(2, dtype=np.complex128),
        forward=lambda model: matrix @ model,
        jvp=lambda model, direction: matrix @ direction,
        vjp=lambda model, dual: (
            vjp_calls.append(np.array(dual, copy=True))
            or np.real(matrix.conj().T @ dual)
        ),
        weights=weights,
        regularization=regularization,
    )

    estimate = estimate_gauss_newton_diagonal(
        problem,
        np.zeros(2),
        probe_count=3,
        seed=17,
    )

    assert isinstance(estimate, GaussNewtonDiagonalEstimate)
    np.testing.assert_allclose(estimate.data, [1.0, 36.0])
    np.testing.assert_allclose(estimate.regularization, [0.25, 0.25])
    np.testing.assert_allclose(estimate.total, [1.25, 36.25])
    assert estimate.probe_count == 3
    assert estimate.seed == 17
    assert len(vjp_calls) == 3


def test_diagonal_inverse_hessian_damps_and_clips_each_control_block():
    preconditioner = DiagonalInverseHessian(
        [0.0, 4.0, 1.0, 0.0, 0.0],
        block_sizes=[2, 1, 2],
        relative_damping=0.1,
        maximum_inverse_ratio=5.0,
    )

    np.testing.assert_allclose(preconditioner.raw_diagonal, [0.0, 4.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(preconditioner.diagonal, [0.88, 4.4, 1.1, 1.0, 1.0])
    np.testing.assert_allclose(
        preconditioner.apply(np.ones(5)),
        1.0 / np.array([0.88, 4.4, 1.1, 1.0, 1.0]),
    )
    np.testing.assert_allclose(
        preconditioner(np.zeros(5), np.arange(1.0, 6.0)),
        np.arange(1.0, 6.0) / np.array([0.88, 4.4, 1.1, 1.0, 1.0]),
    )


def test_diagonal_inverse_hessian_keeps_partially_dark_blocks_positive():
    preconditioner = DiagonalInverseHessian(
        [0.0, 4.0],
        relative_damping=0.0,
        maximum_inverse_ratio=None,
    )

    assert np.all(np.isfinite(preconditioner.inverse_diagonal))
    assert np.all(preconditioner.diagonal > 0.0)
    assert preconditioner.diagonal[1] == 4.0


def test_lbfgs_applies_step_limits_transforms_and_backtracking():
    states = []

    result = minimize_lbfgs(
        lambda model: 0.5 * float(np.dot(model - 1.0, model - 1.0)),
        lambda model: model - 1.0,
        [0.0, 0.0],
        options=LBFGSOptions(
            gradient_tolerance=1.0e-10,
            step_tolerance=0.0,
            objective_tolerance=0.0,
            max_iterations=30,
        ),
        step_limit=lambda model, direction: min(1.0, 0.5 / np.linalg.norm(direction)),
        step_transform=lambda model, step: 0.8 * step,
        callback=states.append,
    )

    assert result.success
    np.testing.assert_allclose(result.model, [1.0, 1.0], atol=1.0e-8)
    assert result.hessian_products == 0
    assert states[1].step_transform_norm_ratio == pytest.approx(0.8)


def test_lbfgs_restarts_its_metric_after_a_failed_line_search():
    hessian = np.array(
        [
            [1.19039823, 0.24468749, -1.96554502],
            [0.24468749, 2.66853586, 1.52729144],
            [-1.96554502, 1.52729144, 6.31353370],
        ]
    )
    initial = np.array([-2.06485968, 0.47502238, -0.76190204])
    inverse_diagonal = 1.0 / np.array([3.55254131, 5.72702004, 0.97590001])
    states = []

    result = minimize_lbfgs(
        lambda model: 0.5 * float(model @ hessian @ model),
        lambda model: hessian @ model,
        initial,
        preconditioner=lambda _model, vector: inverse_diagonal * vector,
        options=LBFGSOptions(
            max_iterations=2,
            max_line_search_trials=1,
            gradient_tolerance=0.0,
            step_tolerance=0.0,
            objective_tolerance=0.0,
        ),
        callback=states.append,
    )

    assert result.status == 0
    assert result.iterations == 2
    assert result.steepest_descent_fallbacks == 1
    assert states[-1].steepest_descent_fallback
    assert result.objective < 0.56


@pytest.mark.parametrize("kind", ["newton", "lbfgs"])
def test_optimizer_stops_at_an_absolute_objective_target(kind):
    objective = lambda model: 0.5 * float(np.dot(model, model))
    gradient = lambda model: model
    if kind == "newton":
        result = minimize_inexact_newton(
            objective,
            gradient,
            lambda _model, direction: direction,
            [1.0],
            options=InexactNewtonOptions(objective_target=0.6),
        )
    else:
        result = minimize_lbfgs(
            objective,
            gradient,
            [1.0],
            options=LBFGSOptions(objective_target=0.6),
        )

    assert result.success
    assert result.status == 4
    assert result.message == "objective target reached"
    assert result.iterations == 0


@pytest.mark.parametrize("options", [InexactNewtonOptions, LBFGSOptions])
def test_objective_target_must_be_nonnegative(options):
    with pytest.raises(ValueError, match="objective_target"):
        options(objective_target=-1.0)


def test_objective_momentum_requires_a_sustained_plateau():
    def solve(momentum):
        factors = iter((1.0 / 0.9, 100.0, 1.0 / 0.9))
        states = []

        def hessian_product(model, direction):
            return next(factors) * direction

        result = minimize_inexact_newton(
            lambda model: 0.5 * float(np.dot(model, model)),
            lambda model: model,
            hessian_product,
            [10.0],
            options=InexactNewtonOptions(
                max_iterations=3,
                max_cg_iterations=1,
                gradient_tolerance=0.0,
                step_tolerance=0.0,
                objective_tolerance=0.05,
                objective_tolerance_momentum=momentum,
                objective_minimum_iterations=2,
            ),
            callback=states.append,
        )
        return result, states

    instantaneous, instantaneous_states = solve(0.0)
    assert instantaneous.success
    assert instantaneous.status == 2
    assert instantaneous.iterations == 2
    assert instantaneous_states[-1].objective_relative_reduction < 0.05

    smoothed, smoothed_states = solve(0.8)
    assert not smoothed.success
    assert smoothed.message == "maximum outer iterations reached"
    assert smoothed.iterations == 3
    assert smoothed_states[2].objective_relative_reduction < 0.05
    assert smoothed_states[2].objective_reduction_momentum > 0.05


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"objective_tolerance_momentum": 1.0}, "must be less than one"),
        ({"objective_minimum_iterations": 0}, "must be positive"),
    ],
)
def test_objective_momentum_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        InexactNewtonOptions(**options)


def test_inexact_newton_records_line_search_and_inner_solve_history(tmp_path):
    matrix = np.array([[2.0 + 0.5j, -0.25j], [0.5, 1.0 - 0.75j]])
    truth = np.array([0.2, -0.1])
    controls = _ControlSpace(2)
    history = OptimizationHistory(tmp_path / "newton_history.json")
    problem = ControlLeastSquaresProblem(
        controls,
        matrix @ truth,
        forward=lambda model: matrix @ model,
        jacobian=lambda model: matrix,
        history=history,
        record_jacobian_iterations=False,
        history_metrics={"continuation_stage": "low_damped"},
    )

    def record(state):
        problem.record_iteration(
            state.model,
            state.gradient,
            step_norm=np.linalg.norm(state.step),
            step_length=state.step_length,
            metrics={
                "forcing": state.forcing,
                "cg_iterations": state.cg_iterations,
            },
        )

    result = minimize_inexact_newton(
        problem.objective,
        problem.gradient,
        problem.gauss_newton_product,
        controls.zeros(),
        options=InexactNewtonOptions(
            gradient_tolerance=1.0e-12,
            step_tolerance=0.0,
            objective_tolerance=0.0,
            max_iterations=10,
        ),
        callback=record,
    )

    assert result.success
    assert history.iteration_count >= 2
    assert history.iterations[-1].step_length == pytest.approx(1.0)
    assert history.iterations[-1].metrics["continuation_stage"] == "low_damped"
    assert history.iterations[-1].metrics["cg_iterations"] >= 1
    assert OptimizationHistory.load(history.path).records == history.records


def test_inexact_newton_backtracks_and_respects_bounds():
    states = []

    def objective(model):
        return float(np.exp(4.0 * model[0]) - 4.0 * model[0])

    result = minimize_inexact_newton(
        objective,
        lambda model: np.array([4.0 * np.exp(4.0 * model[0]) - 4.0]),
        lambda model, direction: np.array(
            [16.0 * np.exp(4.0 * model[0]) * direction[0]]
        ),
        [-0.4],
        bounds=([-0.5], [0.5]),
        options=InexactNewtonOptions(
            gradient_tolerance=1.0e-10,
            max_iterations=30,
            initial_forcing=0.1,
        ),
        callback=states.append,
    )

    assert result.success
    np.testing.assert_allclose(result.model, [0.0], atol=1.0e-8)
    assert all(-0.5 <= state.model[0] <= 0.5 for state in states)
    assert result.line_search_evaluations > result.iterations


def test_inexact_newton_optional_bound_flags_leave_other_dofs_unconstrained():
    truth = np.array([-1.0, 2.0, 3.0])

    result = minimize_inexact_newton(
        lambda model: 0.5 * float(np.dot(model - truth, model - truth)),
        lambda model: model - truth,
        lambda model, direction: direction,
        [0.5, 0.0, 0.0],
        bounds=([0.0, -100.0, -100.0], [100.0, 100.0, 1.0]),
        bound_flags=([True, False, False], [False, False, True]),
        options=InexactNewtonOptions(
            gradient_tolerance=1.0e-12,
            step_tolerance=0.0,
            objective_tolerance=0.0,
            max_iterations=10,
        ),
    )

    assert result.success
    np.testing.assert_allclose(result.model, [0.0, 2.0, 1.0], atol=1.0e-12)


def test_scipy_infinite_bounds_leave_entries_unconstrained():
    truth = np.array([-1.0, 2.0, 3.0])
    result = minimize_inexact_newton(
        lambda model: 0.5 * float(np.dot(model - truth, model - truth)),
        lambda model: model - truth,
        lambda model, direction: direction,
        [0.5, 0.0, 0.0],
        bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, 1.0]),
        options=InexactNewtonOptions(
            gradient_tolerance=1.0e-12,
            step_tolerance=0.0,
            objective_tolerance=0.0,
            max_iterations=10,
        ),
    )

    assert result.success
    np.testing.assert_allclose(result.model, [0.0, 2.0, 1.0], atol=1.0e-12)


def test_projected_line_search_does_not_let_one_bound_throttle_coupled_step():
    hessian = np.array([[1.0, 2.0], [2.0, 5.0]])
    linear = np.array([-1.0, -10.0])
    states = []

    result = minimize_inexact_newton(
        lambda model: 0.5 * float(model @ hessian @ model) + float(linear @ model),
        lambda model: hessian @ model + linear,
        lambda model, direction: hessian @ direction,
        [0.0, 0.0],
        bounds=([0.0, -np.inf], [np.inf, np.inf]),
        options=InexactNewtonOptions(
            gradient_tolerance=1.0e-12,
            step_tolerance=0.0,
            objective_tolerance=0.0,
            max_iterations=10,
            initial_forcing=1.0e-12,
            minimum_forcing=1.0e-12,
        ),
        callback=states.append,
    )

    assert result.success
    np.testing.assert_allclose(result.model, [0.0, 2.0], atol=1.0e-12)
    assert states[1].raw_step[0] == pytest.approx(0.0)
    assert states[1].raw_step[1] > 0.0


def test_box_constraints_validate_only_overlapping_active_sides():
    objective = lambda model: 0.5 * float(np.dot(model, model))
    gradient = lambda model: model
    hessian_product = lambda model, direction: direction

    result = minimize_inexact_newton(
        objective,
        gradient,
        hessian_product,
        [5.0, 0.0],
        bounds=([5.0, 1.0], [-1.0, 0.0]),
        bound_flags=([True, False], [False, True]),
    )
    assert result.success
    np.testing.assert_array_equal(result.model, [5.0, 0.0])

    with pytest.raises(ValueError, match="lower bound"):
        minimize_inexact_newton(
            objective,
            gradient,
            hessian_product,
            [1.0, 0.0],
            bounds=([1.0, -np.inf], [0.0, np.inf]),
        )


def test_false_optional_bound_flags_disable_authored_bounds():
    truth = np.array([-2.0, 3.0])
    result = minimize_inexact_newton(
        lambda model: 0.5 * float(np.dot(model - truth, model - truth)),
        lambda model: model - truth,
        lambda model, direction: direction,
        [0.0, 0.0],
        bounds=([10.0, 10.0], [-10.0, -10.0]),
        bound_flags=([False, False], [False, False]),
    )
    assert result.success
    np.testing.assert_allclose(result.model, truth, atol=1.0e-12)


@pytest.mark.parametrize(
    ("bounds", "bound_flags", "message"),
    [
        (None, ([True], [False]), "require lower and upper bounds"),
        (([0.0], [1.0]), ([1], [False]), "must contain boolean values"),
        (([-np.inf], [1.0]), ([True], [True]), "must be finite"),
    ],
)
def test_optional_bound_flags_are_validated(bounds, bound_flags, message):
    with pytest.raises(ValueError, match=message):
        minimize_inexact_newton(
            lambda model: 0.5 * float(np.dot(model, model)),
            lambda model: model,
            lambda model, direction: direction,
            [0.0],
            bounds=bounds,
            bound_flags=bound_flags,
        )


def test_inexact_newton_combines_box_and_representation_step_limits():
    seen = []

    def step_limit(model, direction):
        seen.append((model.copy(), direction.copy()))
        return 0.125

    result = minimize_inexact_newton(
        lambda model: 0.5 * float((model[0] - 2.0) ** 2),
        lambda model: np.array([model[0] - 2.0]),
        lambda model, direction: direction,
        [0.0],
        bounds=([-10.0], [10.0]),
        step_limit=step_limit,
        options=InexactNewtonOptions(max_iterations=1, gradient_tolerance=0.0),
    )

    assert seen
    assert result.model[0] == pytest.approx(0.25)


def test_inexact_newton_line_search_uses_transformed_step():
    transformed = []
    states = []

    def step_transform(model, step):
        transformed.append((model.copy(), step.copy()))
        return 2.0 * step

    result = minimize_inexact_newton(
        lambda model: 0.5 * float((model[0] - 1.0) ** 2),
        lambda model: np.array([model[0] - 1.0]),
        lambda model, direction: direction,
        [0.0],
        step_transform=step_transform,
        callback=states.append,
        options=InexactNewtonOptions(
            max_iterations=2,
            gradient_tolerance=1.0e-12,
            step_tolerance=0.0,
            objective_tolerance=0.0,
        ),
    )

    assert result.success
    np.testing.assert_allclose(result.model, [1.0], atol=1.0e-12)
    assert len(transformed) == 2
    assert result.line_search_evaluations == 2
    np.testing.assert_allclose(states[-1].raw_step, [0.5])
    np.testing.assert_allclose(states[-1].step, [1.0])
    np.testing.assert_allclose(states[-1].linearization_gradient, [-1.0])
    assert states[-1].raw_directional_derivative == pytest.approx(-0.5)
    assert states[-1].directional_derivative == pytest.approx(-1.0)


def test_inexact_newton_does_not_mix_raw_step_into_transformed_descent():
    result = minimize_inexact_newton(
        lambda model: 0.5 * float((model[0] - 1.0) ** 2),
        lambda model: np.array([model[0] - 1.0]),
        lambda model, direction: direction,
        [0.0],
        step_transform=lambda model, step: 0.05 * step,
        options=InexactNewtonOptions(
            max_iterations=1,
            gradient_tolerance=0.0,
            step_tolerance=0.0,
            objective_tolerance=0.0,
        ),
    )

    np.testing.assert_allclose(result.model, [0.05], atol=1.0e-12)


def test_inexact_newton_rejects_a_persistently_transformed_ascent_step():
    transformed = []

    def step_transform(model, step):
        transformed.append((model.copy(), step.copy()))
        return -2.0 * step

    result = minimize_inexact_newton(
        lambda model: 0.5 * float((model[0] - 1.0) ** 2),
        lambda model: np.array([model[0] - 1.0]),
        lambda model, direction: direction,
        [0.0],
        step_transform=step_transform,
        options=InexactNewtonOptions(
            max_iterations=1,
            gradient_tolerance=0.0,
            step_tolerance=0.0,
            objective_tolerance=0.0,
        ),
    )

    assert transformed
    assert not result.success
    assert result.message == "Armijo line search failed"
    assert result.objective == pytest.approx(0.5)
    assert result.model[0] == pytest.approx(0.0)


def test_inexact_newton_caps_line_search_trials():
    transformed = []

    def step_transform(model, step):
        transformed.append((model.copy(), step.copy()))
        return -step

    result = minimize_inexact_newton(
        lambda model: 0.5 * float((model[0] - 1.0) ** 2),
        lambda model: np.array([model[0] - 1.0]),
        lambda model, direction: direction,
        [0.0],
        step_transform=step_transform,
        options=InexactNewtonOptions(
            max_iterations=1,
            max_line_search_trials=5,
            gradient_tolerance=0.0,
            step_tolerance=0.0,
            objective_tolerance=0.0,
        ),
    )

    assert not result.success
    assert result.message == "Armijo line search failed"
    assert len(transformed) == 5


def test_inexact_newton_rejects_invalid_transformed_step():
    with pytest.raises(ValueError, match="transformed step"):
        minimize_inexact_newton(
            lambda model: 0.5 * float(np.dot(model, model)),
            lambda model: model,
            lambda model, direction: direction,
            [1.0, -1.0],
            step_transform=lambda model, step: [np.nan],
        )


@pytest.mark.parametrize("limit", [-1.0, np.nan])
def test_inexact_newton_rejects_invalid_representation_step_limits(limit):
    with pytest.raises(ValueError, match="nonnegative"):
        minimize_inexact_newton(
            lambda model: 0.5 * float(np.dot(model, model)),
            lambda model: model,
            lambda model, direction: direction,
            [1.0],
            step_limit=lambda model, direction: limit,
        )


def test_inexact_newton_truncates_negative_curvature_to_a_descent_step():
    def objective(model):
        return 0.25 * model[0] ** 4 - 0.5 * model[0] ** 2

    result = minimize_inexact_newton(
        objective,
        lambda model: np.array([model[0] ** 3 - model[0]]),
        lambda model, direction: np.array([(3.0 * model[0] ** 2 - 1.0) * direction[0]]),
        [0.1],
        options=InexactNewtonOptions(
            gradient_tolerance=1.0e-10,
            max_iterations=30,
            initial_forcing=0.1,
        ),
    )

    assert result.success
    assert result.negative_curvature_events >= 1
    np.testing.assert_allclose(result.model, [1.0], atol=1.0e-7)


def test_joint_frequency_laplace_schedule_round_trip_and_warm_start():
    schedule = ContinuationSchedule.joint_frequency_laplace(
        [[2.0, 4.0], [2.0, 4.0, 8.0], [2.0, 4.0, 8.0, 12.0]],
        [2.0, 0.5, 0.05],
        max_iterations=[2, 3, 4],
        names=["damped", "middle", "final"],
    )
    loaded = ContinuationSchedule.from_fs(schedule.to_fs())
    assert loaded == schedule
    assert loaded.stages[0].frequencies == (2.0 - 2.0j, 4.0 - 2.0j)
    assert loaded.stages[-1].maximum_real_frequency == 12.0
    assert loaded.stages[-1].maximum_laplace_damping == pytest.approx(0.05)
    assert len(loaded.frequencies) == 9

    starts = []

    def solve(stage, model):
        starts.append(model.copy())
        return SimpleNamespace(model=model + 1.0, stage=stage.name)

    result = run_continuation(loaded, [0.0, 0.5], solve)
    np.testing.assert_array_equal(starts[0], [0.0, 0.5])
    np.testing.assert_array_equal(starts[1], [1.0, 1.5])
    np.testing.assert_array_equal(result.model, [3.0, 3.5])
    assert [value.stage for value in result.stage_results] == [
        "damped",
        "middle",
        "final",
    ]


def test_frequency_laplace_bands_form_cartesian_continuation_to_real_axis():
    schedule = ContinuationSchedule.from_fs(
        {
            "bands": [
                {
                    "name": "low",
                    "frequencies_hz": [1.0, 2.0, 3.0],
                    "laplace_damping_hz": [0.8, 0.3, 0.0],
                    "max_iterations": 4,
                    "metadata": {"active_controls": ["vp"]},
                }
            ]
        }
    )

    assert len(schedule.stages) == 3
    assert schedule.frequencies == (
        1.0 - 0.8j,
        2.0 - 0.8j,
        3.0 - 0.8j,
        1.0 - 0.3j,
        2.0 - 0.3j,
        3.0 - 0.3j,
        1.0 + 0.0j,
        2.0 + 0.0j,
        3.0 + 0.0j,
    )
    for index, stage in enumerate(schedule.stages):
        assert stage.max_iterations == 4
        assert stage.metadata["continuation_band"] == "low"
        assert stage.metadata["laplace_index"] == index
        assert stage.metadata["active_controls"] == ["vp"]
        assert {value.real for value in stage.frequencies} == {1.0, 2.0, 3.0}
        assert len({value.imag for value in stage.frequencies}) == 1
    assert schedule.stages[-1].maximum_laplace_damping == 0.0


def test_frequency_laplace_bands_support_triangular_frequency_prefixes():
    schedule = ContinuationSchedule.from_fs(
        {
            "bands": [
                {
                    "name": "triangular",
                    "frequencies_hz": [1.0, 2.0, 3.0],
                    "laplace_damping_hz": [0.8, 0.3, 0.0],
                    "frequency_counts": [1, 2, 3],
                }
            ]
        }
    )

    assert [stage.frequencies for stage in schedule.stages] == [
        (1.0 - 0.8j,),
        (1.0 - 0.3j, 2.0 - 0.3j),
        (1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j),
    ]
    assert [stage.metadata["frequency_count"] for stage in schedule.stages] == [
        1,
        2,
        3,
    ]


def test_frequency_laplace_bands_support_per_laplace_metadata():
    schedule = ContinuationSchedule.from_fs(
        {
            "bands": [
                {
                    "name": "block_continuation",
                    "frequencies_hz": [1.0, 2.0],
                    "laplace_damping_hz": [0.5, 0.0],
                    "metadata": {"block_curvature_max_ratio": 2.0},
                    "laplace_metadata": [
                        {"active_controls": ["ss"]},
                        {"active_controls": ["sp", "ss"]},
                    ],
                }
            ]
        }
    )

    assert schedule.stages[0].metadata["active_controls"] == ["ss"]
    assert schedule.stages[1].metadata["active_controls"] == ["sp", "ss"]
    assert all(
        stage.metadata["block_curvature_max_ratio"] == 2.0 for stage in schedule.stages
    )


@pytest.mark.parametrize(
    "damping, message",
    [([0.5, 0.6, 0.0], "must not increase"), ([0.5, 0.1], "must end at zero")],
)
def test_frequency_laplace_bands_validate_damping_path(damping, message):
    with pytest.raises(ValueError, match=message):
        ContinuationSchedule.from_fs(
            {
                "bands": [
                    {
                        "name": "invalid",
                        "frequencies_hz": [1.0, 2.0],
                        "laplace_damping_hz": damping,
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "counts, message",
    [
        ([1, 2], "must match"),
        ([1, 2.5, 3], "finite integers"),
        ([1, 4, 3], "within"),
        ([2, 1, 3], "not decrease"),
        ([1, 2, 2], "complete band"),
    ],
)
def test_frequency_laplace_bands_validate_frequency_progression(counts, message):
    with pytest.raises(ValueError, match=message):
        ContinuationSchedule.from_fs(
            {
                "bands": [
                    {
                        "name": "invalid",
                        "frequencies_hz": [1.0, 2.0, 3.0],
                        "laplace_damping_hz": [0.8, 0.3, 0.0],
                        "frequency_counts": counts,
                    }
                ]
            }
        )


def test_regularization_history_checkpoint_and_result_round_trip(tmp_path):
    regularization = QuadraticRegularization(
        [[1.0, -1.0]], weight=0.25, reference=[0.1, 0.1]
    )
    model = np.array([0.2, -0.1])
    residual = regularization.residual(model)
    np.testing.assert_allclose(residual, [0.15])
    np.testing.assert_allclose(regularization.jacobian(), [[0.5, -0.5]])

    terms = LossTerms(data=2.0, regularization=0.5)
    history = OptimizationHistory(tmp_path / "history.json")
    history.record_evaluation(model, terms, metrics={"adjoint_mismatch": 1.0e-8})
    first = history.record_iteration(model, terms, gradient_norm=3.0)
    assert history.record_iteration(model, terms) is first

    checkpoint_path = tmp_path / "checkpoint.h5"
    checkpoint = OptimizationCheckpoint(
        model=model,
        iteration=history.iteration_count,
        evaluations=history.evaluation_count,
        loss=terms,
        metadata={"control_id": "sediment_sp"},
    )
    checkpoint.save(checkpoint_path)
    loaded_checkpoint = OptimizationCheckpoint.load(checkpoint_path)
    np.testing.assert_array_equal(loaded_checkpoint.model, model)
    assert loaded_checkpoint.loss == terms

    history.finish("stopped", "iteration limit")
    history.resume("checkpoint restart")
    history.record_evaluation(model + 0.1, terms)
    assert history.status == "running"
    assert history.message == "checkpoint restart"

    history.finish("converged", "done")
    with pytest.raises(RuntimeError, match="converged"):
        history.resume()

    result_path = tmp_path / "result.json"
    result = OptimizationResult(
        success=True,
        status=1,
        message="converged",
        model=model,
        loss=terms,
        iterations=history.iteration_count,
        evaluations=history.evaluation_count,
        history=str(history.path),
        checkpoint=str(checkpoint_path),
        metrics={"objective_ratio": 0.25},
    )
    result.save(result_path)
    loaded_result = OptimizationResult.load(result_path)
    np.testing.assert_array_equal(loaded_result.model, model)
    assert loaded_result.loss == terms
    assert json.loads(result_path.read_text(encoding="utf-8"))["schema"] == (
        "fs-optimization-result-1"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OptimizationCheckpoint(
            model=np.array([1.0 + 0.0j]),
            iteration=0,
            evaluations=0,
            loss=LossTerms(0.0),
        ),
        lambda: ControlObjectiveProblem(
            _ControlSpace(1), value_gradient=lambda model: (0.0, model)
        ).objective(np.array([1.0 + 0.0j])),
    ],
)
def test_optimization_model_vectors_reject_complex_values(factory):
    with pytest.raises(ValueError, match="real-valued"):
        factory()


def test_lbfgs_restart_reproduces_uninterrupted_accepted_models():
    """Curvature pairs and stopping momentum survive an atomic checkpoint."""
    from scipy.optimize import rosen, rosen_der

    from frequensolve.inversion.optimization import LBFGSOptions, minimize_lbfgs

    options = LBFGSOptions(
        max_iterations=8, gradient_tolerance=0, objective_tolerance=0, step_tolerance=0
    )
    initial = np.array([-1.2, 1.0])
    whole = []
    minimize_lbfgs(rosen, rosen_der, initial, options=options, callback=whole.append)
    saved = []

    class Interrupted(RuntimeError):
        pass

    def checkpoint(event):
        if event.iteration == 3:
            saved.append(event)
            raise Interrupted()

    with pytest.raises(Interrupted):
        minimize_lbfgs(rosen, rosen_der, initial, options=options, callback=checkpoint)
    resumed = []
    state = saved[0].optimizer_state
    assert len(state["pairs"]) > 0
    import dataclasses

    minimize_lbfgs(
        rosen,
        rosen_der,
        saved[0].model,
        options=dataclasses.replace(options, max_iterations=5),
        restart=state,
        callback=resumed.append,
    )
    for expected, actual in zip(whole[3:], resumed):
        np.testing.assert_array_equal(expected.model, actual.model)
        if actual.iteration:
            assert (
                actual.objective_reduction_momentum
                == expected.objective_reduction_momentum
            )
    with pytest.raises(ValueError, match="different model"):
        minimize_lbfgs(
            rosen, rosen_der, saved[0].model + 0.01, options=options, restart=state
        )
    invalid = dict(state, pairs=[dict(step=[1.0, 0.0], difference=[-1.0, 0.0])])
    with pytest.raises(ValueError, match="positive finite curvature"):
        minimize_lbfgs(
            rosen, rosen_der, saved[0].model, options=options, restart=invalid
        )
