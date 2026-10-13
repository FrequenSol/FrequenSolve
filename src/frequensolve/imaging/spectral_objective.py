"""Fixed spectral attributes in the standard FWI workflow."""

from dataclasses import asdict, replace

import numpy as np

from .operators import ModelOperator
from .spectral_attributes import SpectralAttribute, SpectralJetLinearization


class SpectralObjective:
    """Compose a raw imaging problem with a fixed spectral attribute.

    Pass this view to ``FWI`` to retain native control transfer, regularization,
    checkpoints and line searches. ``feature_weights`` multiply the objective
    outside its robust loss; for normalized jets these index orders 0..order.
    They remain fixed at every trial model. Source derivatives are frozen.
    This version supports one frequency per stage and unprocessed raw inputs.
    """

    def __init__(self, problem, definition, *, order, feature_weights=None):
        if not isinstance(definition, SpectralAttribute):
            raise TypeError("definition must be a SpectralAttribute")
        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order not in range(5)
        ):
            raise ValueError("order must be an integer from zero through four")
        self.raw, self.definition, self.order = problem, definition, int(order)
        weights = np.array(
            [1.0] if feature_weights is None else feature_weights,
            dtype=float,
            copy=True,
        )
        if (
            weights.ndim != 1
            or not len(weights)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError("feature weights must be finite nonnegative values")
        weights.flags.writeable = False
        self.feature_weights = weights
        self._latest = None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    @property
    def state(self):
        return self.raw.state

    @state.setter
    def state(self, value):
        self.raw.state = value

    def identity(self):
        definition = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in asdict(self.definition).items()
        }
        return {
            **self.raw.identity(),
            "spectral_objective": dict(
                definition=definition,
                order=self.order,
                feature_weights=self.feature_weights.tolist(),
            ),
        }

    def restrict(self, **kwargs):
        if any(k in kwargs for k in ("misfit", "loss", "kernel_derivative")):
            raise ValueError(
                "Change the spectral definition explicitly, not its raw inputs"
            )
        return type(self)(
            self.raw.restrict(**kwargs),
            self.definition,
            order=self.order,
            feature_weights=self.feature_weights,
        )

    def with_controls(self, *args, **kwargs):
        return type(self)(
            self.raw.with_controls(*args, **kwargs),
            self.definition,
            order=self.order,
            feature_weights=self.feature_weights,
        )

    def linearize(self, v=None, *, gradient=True):
        if len(self.raw.frequencies) != 1:
            raise ValueError(
                "SpectralObjective currently requires one frequency per stage"
            )
        point = self.raw.vector(None if v is None else self.raw.state_from(v))
        previous = self._latest
        if previous is not None and np.array_equal(point.values, previous.point.values):
            if gradient:
                previous.ensure_gradient()
            return previous
        jets = SpectralJetLinearization.from_problem(
            self.raw, order=self.order, state=v
        )
        bound = jets.attribute(self.definition)
        count = bound.data.residual.shape[1]
        if len(self.feature_weights) not in (1, count):
            raise ValueError("feature weights must match attribute output components")
        scale = np.sqrt(self.feature_weights)
        bound.data = replace(
            bound.data,
            residual=bound.data.residual * scale[None, :],
            jacobian=bound.data.jacobian * scale[None, :, None, None],
        )
        self._latest = _SpectralLinearization(self, bound, gradient=gradient)
        return self._latest

    def value(self, v=None):
        return self.linearize(v, gradient=False).value

    def gradient(self, v=None):
        return self.linearize(v).gradient


class _Jacobian(ModelOperator):
    def __init__(self, bound):
        self.bound = bound
        super().__init__(
            bound.jets.space,
            None,
            shape=(2 * bound.data.residual.size, bound.jets.space.size),
        )

    def _matvec(self, x):
        z = self.bound.jvp(x).ravel()
        return np.concatenate((z.real, z.imag))

    def _rmatvec(self, y):
        n = self.bound.data.residual.size
        z = (np.asarray(y[:n]) + 1j * np.asarray(y[n:])).reshape(
            self.bound.data.residual.shape
        )
        return self.bound.vjp(z).values


class _SpectralLinearization:
    def __init__(self, problem, bound, *, gradient):
        self.problem, self.bound = problem, bound
        first = bound.jets.planes[0]
        self.space, self.state, self.point = first.space, first.state, first.point
        self.frequencies = problem.frequencies
        self.value = bound.value
        self.report = {
            f"order_{k}": float(0.5 * np.sum(abs(r) ** 2))
            for k, r in enumerate(bound.data.residual.T)
        }
        self.gradient = None
        self.jacobian = _Jacobian(bound)
        self.normal = self.jacobian.H @ self.jacobian
        if gradient:
            self.ensure_gradient()

    def ensure_gradient(self):
        if self.gradient is None:
            self.gradient = self.bound.gradient()
            self.gradient._field_factory = self.field

    @property
    def regularization_job(self):
        # Native regularization preparation requires a covector-bearing source
        # job, whereas raw jet planes may be value-only cached states.
        return self.problem.raw.linearize(self.point).regularization_job

    @property
    def jobs(self):
        return tuple(p.job for p in self.bound.jets.planes)

    def field(self, vector=None, **kwargs):
        return self.bound.jets.planes[0].field(
            self.gradient if vector is None else vector, **kwargs
        )

    def trial(self, v, *, gradient=False):
        return self.problem.linearize(v, gradient=gradient)

    def objective_residual(self):
        z = self.bound.data.residual.ravel()
        return np.concatenate((z.real, z.imag))
