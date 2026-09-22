"""Derivative checks for real controls and complex solver data."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import numpy as np

__all__ = ["gradient_taylor_test", "real_adjoint_test"]


def _real_array(value: Any, *, name: str) -> np.ndarray:
    """Return one finite float64 array without discarding complex values."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _complex_array(value: Any, *, name: str) -> np.ndarray:
    """Return one nonempty finite complex128 array."""

    array = np.asarray(value, dtype=np.complex128)
    if (
        array.size < 1
        or not np.all(np.isfinite(array.real))
        or not np.all(np.isfinite(array.imag))
    ):
        raise ValueError(f"{name} must contain finite values")
    return array


def _real_scalar(value: Any, *, name: str) -> float:
    """Return one finite real scalar."""

    array = np.asarray(value)
    if array.shape or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a real scalar")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _rates(steps: np.ndarray, errors: Sequence[float]) -> list[float]:
    """Return consecutive log-log convergence rates."""

    rates = []
    for previous_step, step, previous_error, error in zip(
        steps[:-1], steps[1:], errors[:-1], errors[1:]
    ):
        if previous_error == 0.0 or error == 0.0:
            rates.append(float("inf"))
        else:
            rates.append(
                float(np.log(error / previous_error) / np.log(step / previous_step))
            )
    return rates


def real_adjoint_test(
    jvp: Callable[[np.ndarray], Any],
    vjp: Callable[[np.ndarray], Any],
    direction: Any,
    dual: Any,
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-12,
) -> Dict[str, Any]:
    """Test a complex-data JVP/VJP pair for real model controls.

    The tested pairing is ``real(vdot(J p, y)) == dot(p, J.T y)``. This is
    the native convention used by Sauce: controls and gradients are real,
    while frequency-domain receiver values and their dual loads are complex.
    """

    if not callable(jvp) or not callable(vjp):
        raise TypeError("JVP and VJP must be callable")
    rtol = _real_scalar(relative_tolerance, name="relative tolerance")
    atol = _real_scalar(absolute_tolerance, name="absolute tolerance")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("adjoint-test tolerances must be nonnegative")

    model_direction = _real_array(direction, name="model direction")
    data_dual = _complex_array(dual, name="data dual")
    tangent = _complex_array(
        jvp(np.array(model_direction, copy=True)), name="JVP result"
    )
    if tangent.shape != data_dual.shape:
        raise ValueError(
            f"JVP result has shape {tangent.shape}; expected {data_dual.shape}"
        )
    adjoint = _real_array(vjp(np.array(data_dual, copy=True)), name="VJP result")
    if adjoint.shape != model_direction.shape:
        raise ValueError(
            f"VJP result has shape {adjoint.shape}; "
            f"expected {model_direction.shape}"
        )

    lhs = float(np.real(np.vdot(tangent.reshape(-1), data_dual.reshape(-1))))
    rhs = float(np.dot(model_direction.reshape(-1), adjoint.reshape(-1)))
    absolute_error = abs(lhs - rhs)
    scale = max(abs(lhs), abs(rhs), float(np.finfo(np.float64).tiny))
    relative_error = absolute_error / scale
    return {
        "lhs": lhs,
        "rhs": rhs,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "passed": bool(absolute_error <= atol + rtol * scale),
    }


def gradient_taylor_test(
    objective: Callable[[np.ndarray], Any],
    gradient: Callable[[np.ndarray], Any],
    model: Any,
    direction: Any,
    *,
    steps: Sequence[float] = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3),
    symmetric: bool = False,
    minimum_order: float = 1.5,
) -> Dict[str, Any]:
    """Check a real scalar objective gradient over decreasing step lengths.

    The zeroth-order remainder should decay linearly, while subtracting
    ``step * dot(gradient, direction)`` should produce a quadratic remainder.
    With ``symmetric=True``, centered directional-derivative errors and their
    rates are also reported.
    """

    if not callable(objective) or not callable(gradient):
        raise TypeError("objective and gradient must be callable")
    base = _real_array(model, name="model")
    perturbation = _real_array(direction, name="direction")
    if perturbation.shape != base.shape:
        raise ValueError(
            f"direction has shape {perturbation.shape}; expected {base.shape}"
        )
    if not np.any(perturbation):
        raise ValueError("Taylor-test direction must be nonzero")

    step_values = _real_array(steps, name="Taylor-test steps").reshape(-1)
    if step_values.size < 2 or np.any(step_values <= 0.0):
        raise ValueError("Taylor test requires at least two positive steps")
    if np.any(step_values[1:] >= step_values[:-1]):
        raise ValueError("Taylor-test steps must be strictly decreasing")
    order = _real_scalar(minimum_order, name="minimum order")

    objective_zero = _real_scalar(
        objective(np.array(base, copy=True)), name="objective"
    )
    gradient_zero = _real_array(gradient(np.array(base, copy=True)), name="gradient")
    if gradient_zero.shape != base.shape:
        raise ValueError(
            f"gradient has shape {gradient_zero.shape}; expected {base.shape}"
        )
    directional_derivative = float(
        np.dot(gradient_zero.reshape(-1), perturbation.reshape(-1))
    )

    objectives_plus = []
    objectives_minus = []
    zeroth_order_remainders = []
    first_order_remainders = []
    centered_derivative_errors = []
    for step in step_values:
        plus = _real_scalar(
            objective(base + step * perturbation), name="perturbed objective"
        )
        objectives_plus.append(plus)
        zeroth_order_remainders.append(abs(plus - objective_zero))
        first_order_remainders.append(
            abs(plus - objective_zero - step * directional_derivative)
        )
        if symmetric:
            minus = _real_scalar(
                objective(base - step * perturbation), name="perturbed objective"
            )
            objectives_minus.append(minus)
            centered_derivative_errors.append(
                abs((plus - minus) / (2.0 * step) - directional_derivative)
            )

    zeroth_order_rates = _rates(step_values, zeroth_order_remainders)
    first_order_rates = _rates(step_values, first_order_remainders)
    result = {
        "steps": step_values.tolist(),
        "objective": objective_zero,
        "objectives_plus": objectives_plus,
        "directional_derivative": directional_derivative,
        "zeroth_order_remainders": zeroth_order_remainders,
        "first_order_remainders": first_order_remainders,
        "zeroth_order_rates": zeroth_order_rates,
        "first_order_rates": first_order_rates,
        "passed": bool(first_order_rates and first_order_rates[-1] >= order),
    }
    if symmetric:
        result.update(
            {
                "objectives_minus": objectives_minus,
                "centered_derivative_errors": centered_derivative_errors,
                "centered_derivative_rates": _rates(
                    step_values, centered_derivative_errors
                ),
            }
        )
    return result
