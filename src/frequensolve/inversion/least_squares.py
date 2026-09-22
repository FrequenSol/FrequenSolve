"""Real-control least-squares adapter for complex solver data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from frequensolve.inversion.data import ComplexDataRealifier
from frequensolve.inversion.history import LossTerms, OptimizationHistory
from frequensolve.inversion.validation import (
    gradient_taylor_test,
    real_adjoint_test,
)
from frequensolve.simulation.jobs.control_sensitivity import ControlSpace

__all__ = [
    "ControlLeastSquaresProblem",
    "ControlObjectiveProblem",
    "QuadraticRegularization",
]


def _real_array(value: Any, *, name: str) -> np.ndarray:
    """Convert a finite real array without dropping complex components."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class QuadraticRegularization:
    """Linear residual ``sqrt(weight) * matrix @ (model - reference)``."""

    matrix: np.ndarray
    weight: float = 1.0
    reference: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        matrix = _real_array(self.matrix, name="regularization matrix")
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("regularization matrix must be non-empty and 2-D")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("regularization weight must be finite and nonnegative")
        reference = self.reference
        if reference is None:
            reference = np.zeros(matrix.shape[1], dtype=np.float64)
        reference = _real_array(reference, name="regularization reference").reshape(-1)
        if reference.size != matrix.shape[1]:
            raise ValueError("regularization reference size does not match its matrix")
        object.__setattr__(self, "matrix", np.array(matrix, copy=True))
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "reference", np.array(reference, copy=True))

    @property
    def size(self) -> int:
        """Return the regularization residual length."""

        return int(self.matrix.shape[0])

    def residual(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the scaled regularization residual."""

        return np.sqrt(self.weight) * (self.matrix @ (model - self.reference))

    def jacobian(self) -> np.ndarray:
        """Return the scaled, model-independent regularization Jacobian."""

        return np.sqrt(self.weight) * self.matrix


class ControlLeastSquaresProblem:
    """Expose complex forward/JVP callbacks as a real least-squares problem.

    Material controls are always float64. Complex receiver samples are
    interleaved into an isometric real data vector only at the optimizer
    boundary. Optional data weights multiply residuals, and optional quadratic
    regularization is appended as additional real residual rows.
    """

    def __init__(
        self,
        control_space: ControlSpace,
        observed: Any,
        *,
        forward: Callable[[np.ndarray], Any],
        jacobian: Optional[Callable[[np.ndarray], Any]] = None,
        jvp: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
        vjp: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
        weights: Optional[Any] = None,
        regularization: Optional[QuadraticRegularization] = None,
        history: Optional[OptimizationHistory] = None,
        iteration_callback: Optional[Callable[[np.ndarray, LossTerms], None]] = None,
        record_jacobian_iterations: bool = True,
        history_metrics: Optional[dict[str, Any]] = None,
    ):
        if not isinstance(control_space, ControlSpace):
            raise TypeError("control least-squares problem requires a ControlSpace")
        observed_values = np.asarray(observed, dtype=np.complex128)
        if observed_values.size < 1:
            raise ValueError("control least-squares problem requires observed data")
        if not np.all(np.isfinite(observed_values.real)) or not np.all(
            np.isfinite(observed_values.imag)
        ):
            raise ValueError("observed data must contain only finite values")
        if not callable(forward):
            raise TypeError("forward must be callable")
        if jacobian is not None and not callable(jacobian):
            raise TypeError("jacobian must be callable")
        if (jvp is None) != (vjp is None):
            raise ValueError("native JVP and VJP callbacks must be supplied together")
        if jvp is not None and (not callable(jvp) or not callable(vjp)):
            raise TypeError("JVP and VJP callbacks must be callable")
        if jacobian is None and jvp is None:
            raise ValueError("provide either a Jacobian or native JVP/VJP callbacks")
        if weights is None:
            data_weights = np.ones(observed_values.shape, dtype=np.float64)
        else:
            data_weights = _real_array(weights, name="data weights")
            try:
                data_weights = np.broadcast_to(
                    data_weights, observed_values.shape
                ).copy()
            except ValueError as error:
                raise ValueError(
                    "data weights are not broadcastable to the observations"
                ) from error
            if np.any(data_weights < 0.0):
                raise ValueError("data weights must be nonnegative")
        if regularization is not None:
            if not isinstance(regularization, QuadraticRegularization):
                raise TypeError("regularization must be QuadraticRegularization")
            if regularization.matrix.shape[1] != control_space.size:
                raise ValueError(
                    "regularization model size does not match the control space"
                )
        if iteration_callback is not None and not callable(iteration_callback):
            raise TypeError("iteration_callback must be callable")
        self.control_space = control_space
        self.observed = np.array(observed_values, copy=True)
        self.forward_callback = forward
        self.jacobian_callback = jacobian
        self.jvp_callback = jvp
        self.vjp_callback = vjp
        self.data_weights = data_weights
        self.regularization = regularization
        self.history = history
        self.iteration_callback = iteration_callback
        self.record_jacobian_iterations = bool(record_jacobian_iterations)
        self.history_metrics = dict(history_metrics or {})
        self.realifier = ComplexDataRealifier(self.observed.shape)
        self._cache_model: Optional[np.ndarray] = None
        self._cache_residual: Optional[np.ndarray] = None
        self._cache_loss: Optional[LossTerms] = None
        self._cache_weighted_residual: Optional[np.ndarray] = None
        self._last_iteration_model: Optional[np.ndarray] = None

    @property
    def data_size(self) -> int:
        """Return the realified receiver-data residual size."""

        return self.realifier.real_size

    @property
    def residual_size(self) -> int:
        """Return the total data-plus-regularization residual size."""

        regularization_size = (
            0 if self.regularization is None else self.regularization.size
        )
        return self.data_size + regularization_size

    def residual(self, model: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Evaluate and record the realified least-squares residual."""

        vector = self.control_space.pack(model)
        self._evaluate(vector)
        return np.array(self._cache_residual, copy=True)

    def objective(self, model: Union[Sequence[float], np.ndarray]) -> float:
        """Return the scalar data-plus-regularization objective."""

        return self.loss(model).total

    def jacobian(self, model: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Evaluate native JVP columns and return a realified Jacobian."""

        vector = self.control_space.pack(model)
        residual = self._evaluate(vector)
        values = self._complex_jacobian(vector)
        weighted = values * self.data_weights.reshape(-1, 1)
        real_jacobian = self.realifier.pack_jacobian(weighted, self.control_space.size)
        if self.regularization is not None:
            real_jacobian = np.vstack((real_jacobian, self.regularization.jacobian()))
        gradient = real_jacobian.T @ residual
        if self.record_jacobian_iterations:
            self._record_iteration(vector, gradient)
        return real_jacobian

    def loss(self, model: Union[Sequence[float], np.ndarray]) -> LossTerms:
        """Return data and regularization loss terms for one model."""

        vector = self.control_space.pack(model)
        self._evaluate(vector)
        if self._cache_loss is None:
            raise RuntimeError("least-squares loss cache was not populated")
        return self._cache_loss

    def gradient(self, model: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Apply the exact real-model VJP to the current weighted residual."""

        vector = self.control_space.pack(model)
        self._evaluate(vector)
        if self._cache_weighted_residual is None:
            raise RuntimeError("weighted data residual cache was not populated")
        dual = self._cache_weighted_residual * self.data_weights
        result = self.data_vjp(vector, dual)
        if self.regularization is not None:
            result = result + (
                self.regularization.jacobian().T @ self.regularization.residual(vector)
            )
        return np.asarray(result, dtype=np.float64)

    def data_jvp(
        self,
        model: Union[Sequence[float], np.ndarray],
        direction: Union[Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Apply the unweighted complex receiver-data Jacobian."""

        vector = self.control_space.pack(model)
        tangent = self.control_space.pack(direction)
        if self.jvp_callback is None:
            values = self._complex_jacobian(vector) @ tangent
            return values.reshape(self.observed.shape)
        values = np.asarray(
            self.jvp_callback(
                np.array(vector, copy=True), np.array(tangent, copy=True)
            ),
            dtype=np.complex128,
        )
        if values.size != self.realifier.complex_size:
            raise ValueError(
                f"native JVP has {values.size} samples; "
                f"expected {self.realifier.complex_size}"
            )
        values = values.reshape(self.observed.shape)
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ValueError("native JVP must contain only finite values")
        return values

    def data_vjp(
        self, model: Union[Sequence[float], np.ndarray], dual: Any
    ) -> np.ndarray:
        """Apply the unweighted transpose under the real/complex pairing."""

        vector = self.control_space.pack(model)
        values = np.asarray(dual, dtype=np.complex128)
        if values.shape != self.observed.shape:
            raise ValueError(
                f"VJP dual has shape {values.shape}; expected {self.observed.shape}"
            )
        if self.vjp_callback is None:
            jacobian = self._complex_jacobian(vector)
            return self.realifier.pack_jacobian(
                jacobian, self.control_space.size
            ).T @ self.realifier.pack(values)
        result = _real_array(
            self.vjp_callback(np.array(vector, copy=True), np.array(values, copy=True)),
            name="native VJP",
        ).reshape(-1)
        if result.size != self.control_space.size:
            raise ValueError(
                f"native VJP has size {result.size}; expected {self.control_space.size}"
            )
        return result

    def gauss_newton_product(
        self,
        model: Union[Sequence[float], np.ndarray],
        direction: Union[Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Apply ``J.T W**2 J + R.T R`` without forming a normal matrix."""

        vector = self.control_space.pack(model)
        tangent = self.control_space.pack(direction)
        self._evaluate(vector)
        incremental_data = self.data_jvp(vector, tangent)
        result = self.data_vjp(
            vector,
            incremental_data * self.data_weights * self.data_weights,
        )
        if self.regularization is not None:
            regularization_jacobian = self.regularization.jacobian()
            result = result + regularization_jacobian.T @ (
                regularization_jacobian @ tangent
            )
        return np.asarray(result, dtype=np.float64)

    def dot_test(
        self,
        model: Union[Sequence[float], np.ndarray],
        *,
        direction: Optional[Union[Sequence[float], np.ndarray]] = None,
        dual: Optional[Any] = None,
        seed: int = 0,
        relative_tolerance: float = 1.0e-8,
        absolute_tolerance: float = 1.0e-12,
    ) -> dict[str, Any]:
        """Check the native JVP/VJP under the real-control pairing."""

        vector = self.control_space.pack(model)
        rng = np.random.default_rng(seed)
        if direction is None:
            direction = rng.standard_normal(self.control_space.size)
        if dual is None:
            dual = rng.standard_normal(self.observed.shape) + 1j * rng.standard_normal(
                self.observed.shape
            )
        return real_adjoint_test(
            lambda tangent: self.data_jvp(vector, tangent),
            lambda load: self.data_vjp(vector, load),
            direction,
            dual,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )

    def taylor_test(
        self,
        model: Union[Sequence[float], np.ndarray],
        direction: Union[Sequence[float], np.ndarray],
        *,
        steps: Sequence[float] = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3),
        symmetric: bool = False,
        minimum_order: float = 1.5,
    ) -> dict[str, Any]:
        """Check the complete scalar objective and its real control gradient."""

        return gradient_taylor_test(
            self.objective,
            self.gradient,
            self.control_space.pack(model),
            self.control_space.pack(direction),
            steps=steps,
            symmetric=symmetric,
            minimum_order=minimum_order,
        )

    def record_iteration(
        self,
        model: Union[Sequence[float], np.ndarray],
        gradient: Union[Sequence[float], np.ndarray],
        *,
        step_norm: Optional[float] = None,
        step_length: Optional[float] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record one accepted iterate from a non-SciPy optimizer."""

        vector = self.control_space.pack(model)
        gradient_vector = _real_array(gradient, name="gradient").reshape(-1)
        if gradient_vector.size != self.control_space.size:
            raise ValueError("gradient size does not match the control space")
        self._evaluate(vector)
        merged_metrics = dict(self.history_metrics)
        merged_metrics.update(metrics or {})
        self._record_iteration(
            vector,
            gradient_vector,
            step_norm=step_norm,
            step_length=step_length,
            metrics=merged_metrics,
        )

    def _complex_jacobian(self, model: np.ndarray) -> np.ndarray:
        """Return an explicit complex Jacobian, using JVP columns if needed."""

        expected_shape = (self.realifier.complex_size, self.control_space.size)
        if self.jacobian_callback is not None:
            values = np.asarray(
                self.jacobian_callback(np.array(model, copy=True)),
                dtype=np.complex128,
            )
        else:
            columns = []
            for index in range(self.control_space.size):
                direction = np.zeros(self.control_space.size, dtype=np.float64)
                direction[index] = 1.0
                columns.append(self.data_jvp(model, direction).reshape(-1))
            values = np.column_stack(columns)
        if values.size != int(np.prod(expected_shape)):
            raise ValueError(
                f"control Jacobian has {values.size} values; expected "
                f"{int(np.prod(expected_shape))}"
            )
        values = values.reshape(expected_shape)
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ValueError("control Jacobian must contain only finite values")
        return values

    def _evaluate(self, model: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear forward map once per exact model vector."""

        if self._cache_model is not None and np.array_equal(model, self._cache_model):
            if self._cache_residual is None:
                raise RuntimeError("least-squares residual cache is incomplete")
            return self._cache_residual

        simulated = np.asarray(
            self.forward_callback(np.array(model, copy=True)), dtype=np.complex128
        )
        if simulated.shape != self.observed.shape:
            raise ValueError(
                f"simulated data has shape {simulated.shape}; "
                f"expected {self.observed.shape}"
            )
        if not np.all(np.isfinite(simulated.real)) or not np.all(
            np.isfinite(simulated.imag)
        ):
            raise ValueError("simulated data must contain only finite values")
        weighted_residual = (simulated - self.observed) * self.data_weights
        data_loss = 0.5 * float(np.vdot(weighted_residual, weighted_residual).real)
        data_residual = self.realifier.pack(weighted_residual)
        regularization_residual = (
            np.empty(0, dtype=np.float64)
            if self.regularization is None
            else self.regularization.residual(model)
        )
        residual = np.concatenate((data_residual, regularization_residual))
        loss = LossTerms(
            data=data_loss,
            regularization=0.5
            * float(np.dot(regularization_residual, regularization_residual)),
        )
        self._cache_model = np.array(model, copy=True)
        self._cache_residual = residual
        self._cache_loss = loss
        self._cache_weighted_residual = np.array(weighted_residual, copy=True)
        if self.history is not None:
            self.history.record_evaluation(
                model, loss, metrics=dict(self.history_metrics)
            )
        return residual

    def _record_iteration(
        self,
        model: np.ndarray,
        gradient: np.ndarray,
        *,
        step_norm: Optional[float] = None,
        step_length: Optional[float] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record each model at which SciPy requests a Jacobian."""

        is_new = self._last_iteration_model is None or not np.array_equal(
            model, self._last_iteration_model
        )
        if self.history is None:
            self._last_iteration_model = np.array(model, copy=True)
        else:
            if self._cache_loss is None:
                raise RuntimeError("cannot record an iteration without a loss")
            if step_norm is None:
                step_norm = (
                    None
                    if self._last_iteration_model is None
                    else float(np.linalg.norm(model - self._last_iteration_model))
                )
            merged_metrics = dict(self.history_metrics)
            merged_metrics.update(metrics or {})
            self.history.record_iteration(
                model,
                self._cache_loss,
                gradient_norm=float(np.linalg.norm(gradient)),
                step_norm=step_norm,
                step_length=step_length,
                metrics=merged_metrics,
            )
        if is_new and self.iteration_callback is not None:
            if self._cache_loss is None:
                raise RuntimeError("cannot checkpoint an iteration without a loss")
            self.iteration_callback(np.array(model, copy=True), self._cache_loss)
        self._last_iteration_model = np.array(model, copy=True)


class ControlObjectiveProblem:
    """Cache a solver-native scalar objective and real control gradient.

    The callback is intentionally loss-agnostic: Sauce may apply L2, Huber,
    Student-t, whitening, trace weights, and the retained-state adjoint in one
    native evaluation. FrequenSolve only adds optional control-space quadratic
    regularization and persists optimizer history.
    """

    def __init__(
        self,
        control_space: ControlSpace,
        *,
        value_gradient: Callable[[np.ndarray], tuple[float, Any]],
        regularization: Optional[QuadraticRegularization] = None,
        history: Optional[OptimizationHistory] = None,
        history_metrics: Optional[dict[str, Any]] = None,
    ):
        if not isinstance(control_space, ControlSpace):
            raise TypeError("control objective requires a ControlSpace")
        if not callable(value_gradient):
            raise TypeError("value_gradient must be callable")
        if regularization is not None:
            if not isinstance(regularization, QuadraticRegularization):
                raise TypeError("regularization must be QuadraticRegularization")
            if regularization.matrix.shape[1] != control_space.size:
                raise ValueError(
                    "regularization model size does not match the control space"
                )
        self.control_space = control_space
        self.value_gradient_callback = value_gradient
        self.regularization = regularization
        self.history = history
        self.history_metrics = dict(history_metrics or {})
        self._cache_model: Optional[np.ndarray] = None
        self._cache_loss: Optional[LossTerms] = None
        self._cache_gradient: Optional[np.ndarray] = None

    def objective(self, model: Union[Sequence[float], np.ndarray]) -> float:
        """Return the complete scalar objective, reusing a fused evaluation."""

        self._evaluate(self.control_space.pack(model))
        if self._cache_loss is None:
            raise RuntimeError("control objective cache was not populated")
        return self._cache_loss.total

    def gradient(self, model: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Return the exact real control gradient from the same evaluation."""

        self._evaluate(self.control_space.pack(model))
        if self._cache_gradient is None:
            raise RuntimeError("control gradient cache was not populated")
        return np.array(self._cache_gradient, copy=True)

    def loss(self, model: Union[Sequence[float], np.ndarray]) -> LossTerms:
        """Return separately recorded data and regularization losses."""

        self._evaluate(self.control_space.pack(model))
        if self._cache_loss is None:
            raise RuntimeError("control objective cache was not populated")
        return self._cache_loss

    def record_iteration(
        self,
        model: Union[Sequence[float], np.ndarray],
        gradient: Union[Sequence[float], np.ndarray],
        *,
        step_norm: Optional[float] = None,
        step_length: Optional[float] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist one accepted optimizer state without reevaluating Sauce."""

        vector = self.control_space.pack(model)
        self._evaluate(vector)
        gradient_vector = _real_array(gradient, name="gradient").reshape(-1)
        if gradient_vector.size != self.control_space.size:
            raise ValueError("gradient size does not match the control space")
        if self.history is None:
            return
        if self._cache_loss is None:
            raise RuntimeError("cannot record an iteration without a loss")
        merged_metrics = dict(self.history_metrics)
        merged_metrics.update(metrics or {})
        self.history.record_iteration(
            vector,
            self._cache_loss,
            gradient_norm=float(np.linalg.norm(gradient_vector)),
            step_norm=step_norm,
            step_length=step_length,
            metrics=merged_metrics,
        )

    def _evaluate(self, model: np.ndarray) -> None:
        """Run one atomic native value-and-gradient evaluation per model."""

        if self._cache_model is not None and np.array_equal(model, self._cache_model):
            return
        value, gradient = self.value_gradient_callback(np.array(model, copy=True))
        data_loss = float(value)
        if not np.isfinite(data_loss):
            raise ValueError("native objective must return a finite scalar")
        data_gradient = _real_array(gradient, name="native gradient").reshape(-1)
        if data_gradient.size != self.control_space.size:
            raise ValueError("native gradient size does not match the control space")
        regularization_loss = 0.0
        total_gradient = np.array(data_gradient, copy=True)
        if self.regularization is not None:
            residual = self.regularization.residual(model)
            jacobian = self.regularization.jacobian()
            regularization_loss = 0.5 * float(np.dot(residual, residual))
            total_gradient += jacobian.T @ residual
        loss = LossTerms(data=data_loss, regularization=regularization_loss)
        self._cache_model = np.array(model, copy=True)
        self._cache_loss = loss
        self._cache_gradient = total_gradient
        if self.history is not None:
            self.history.record_evaluation(
                model,
                loss,
                metrics=dict(self.history_metrics),
            )
