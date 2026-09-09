from __future__ import annotations

import numpy as np
import pytest

from frequensolve.inversion import (
    ControlLeastSquaresProblem,
    gradient_taylor_test,
    real_adjoint_test,
)
from frequensolve.simulation.jobs.control_sensitivity import ControlBlock, ControlSpace


def test_real_adjoint_test_uses_real_model_complex_data_pairing():
    matrix = np.asarray(
        [
            [1.0 + 2.0j, -0.5j, 0.25],
            [-2.0 + 0.5j, 1.5, 0.75 - 1.0j],
        ]
    )
    direction = np.asarray([0.3, -0.8, 1.2])
    dual = np.asarray([0.5 - 0.25j, -1.0 + 2.0j])

    result = real_adjoint_test(
        lambda value: matrix @ value,
        lambda value: np.real(matrix.conj().T @ value),
        direction,
        dual,
    )

    assert result["passed"]
    assert result["relative_error"] < 1.0e-14


def test_real_adjoint_test_detects_wrong_complex_transpose():
    matrix = np.asarray([[1.0 + 2.0j, -0.5j], [0.25, 2.0 - 1.0j]])
    direction = np.asarray([0.3, -0.8])
    dual = np.asarray([0.5 - 0.25j, -1.0 + 2.0j])

    result = real_adjoint_test(
        lambda value: matrix @ value,
        lambda value: np.real(matrix.T @ value),
        direction,
        dual,
    )

    assert not result["passed"]
    assert result["relative_error"] > 1.0e-2


def test_gradient_taylor_test_reports_quadratic_remainders_and_centered_rates():
    matrix = np.asarray([[3.0, 0.5, -0.25], [0.5, 2.0, 0.75], [-0.25, 0.75, 1.5]])
    linear = np.asarray([0.2, -0.3, 0.5])

    def objective(model):
        return (
            0.5 * np.dot(model, matrix @ model)
            + np.dot(linear, model)
            + 0.1 * np.sum(model**3)
        )

    def gradient(model):
        return matrix @ model + linear + 0.3 * model**2

    result = gradient_taylor_test(
        objective,
        gradient,
        np.asarray([0.4, -0.2, 0.3]),
        np.asarray([0.2, 0.5, -0.4]),
        steps=(0.1, 0.05, 0.025, 0.0125),
        symmetric=True,
    )

    assert result["passed"]
    assert result["first_order_rates"][-1] > 1.95
    assert result["centered_derivative_rates"][-1] > 1.95
    assert 0.9 < result["zeroth_order_rates"][-1] < 1.1


def test_gradient_taylor_test_rejects_an_inconsistent_gradient():
    def objective(model):
        return 0.5 * np.dot(model, model)

    def gradient(model):
        return model + 0.25

    result = gradient_taylor_test(
        objective,
        gradient,
        np.asarray([0.4, -0.2]),
        np.asarray([0.2, 0.5]),
        steps=(0.1, 0.05, 0.025, 0.0125),
    )

    assert not result["passed"]
    assert result["first_order_rates"][-1] < 1.1


def test_derivative_checks_reject_complex_controls_and_unsorted_steps():
    with pytest.raises(ValueError, match="model direction must be real-valued"):
        real_adjoint_test(
            lambda value: value,
            lambda value: value.real,
            np.asarray([1.0 + 1.0j]),
            np.asarray([1.0 + 0.0j]),
        )

    with pytest.raises(ValueError, match="strictly decreasing"):
        gradient_taylor_test(
            lambda model: np.dot(model, model),
            lambda model: 2.0 * model,
            np.asarray([1.0]),
            np.asarray([0.5]),
            steps=(0.1, 0.2),
        )


def test_control_least_squares_exposes_native_dot_and_taylor_tests():
    controls = ControlSpace([ControlBlock("vp", 3)])
    observed = np.asarray([0.25 - 0.5j, -0.75 + 0.1j])
    matrix = np.asarray([[1.0 + 0.5j, -0.25j, 0.75], [-0.5, 1.25 + 0.2j, 0.1j]])

    def forward(model):
        return matrix @ (model + 0.1 * model**2)

    def jvp(model, direction):
        return matrix @ ((1.0 + 0.2 * model) * direction)

    def vjp(model, dual):
        return np.real((1.0 + 0.2 * model) * (matrix.conj().T @ dual))

    problem = ControlLeastSquaresProblem(
        controls,
        observed,
        forward=forward,
        jvp=jvp,
        vjp=vjp,
    )
    model = np.asarray([0.1, -0.2, 0.3])
    direction = np.asarray([0.4, 0.2, -0.3])

    dot = problem.dot_test(model, direction=direction, seed=7)
    taylor = problem.taylor_test(
        model,
        direction,
        steps=(0.1, 0.05, 0.025, 0.0125),
        symmetric=True,
    )

    assert dot["passed"]
    assert dot["relative_error"] < 1.0e-14
    assert taylor["passed"]
    assert taylor["first_order_rates"][-1] > 1.9
