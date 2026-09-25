"""Regularization specifications, smooth custom terms, and preconditioners.

FWI and LSRTM dispatch Tikhonov, TV and TGV to Sauce's native model
regularization callbacks. Native TV/TGV use split-Bregman shrinkage and the
same energy in line searches. Quadratic and custom smooth terms retain the
value/gradient/Hessian protocol. Tikhonov/TV/TGV are configuration objects only;
all native model discretization and numerical evaluation belong to Sauce.
The explicit ``smooth(vector, ...)`` operation remains available for processing
an individual vector and does not assemble an inversion objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack
from scipy.sparse.linalg import LinearOperator

from frequensolve.imaging._artifacts import ControlVectorFile, SmoothingConfig
from frequensolve.imaging.controls import (
    ControlSpace,
    ControlState,
    ControlVector,
)
from frequensolve.imaging.jobs import FWIOperatorJob, SmoothJob
from frequensolve.imaging.operators import ModelOperator
from frequensolve.inversion.least_squares import QuadraticRegularization
from frequensolve.inversion.preconditioning import (
    DiagonalInverseHessian,
    GaussNewtonDiagonalEstimate,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frequensolve.imaging.problem import ImagingProblem, Linearization

__all__ = [
    "BoundRegularization",
    "BoundPreconditioner",
    "Diagonal",
    "FromOperator",
    "Identity",
    "Regularization",
    "Preconditioner",
    "Quadratic",
    "Scaled",
    "Smoothing",
    "Sum",
    "TGV",
    "TV",
    "Tikhonov",
    "smooth",
]


# ---------------------------------------------------------------------------
# vectors and references
# ---------------------------------------------------------------------------


def _finite_scalar(value: Any, name: str, *, minimum: float = 0.0) -> float:
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum:g}; got {value!r}")
    return number


def _values_on(space: ControlSpace, v: Any) -> np.ndarray:
    """Return the optimizer-layout array of ``v`` after checking its space."""

    if isinstance(v, ControlVector):
        if v.space is not space and not v.space.equivalent(space):
            raise ValueError("vector belongs to a different control space")
        return v.values
    array = np.asarray(v)
    if np.iscomplexobj(array):
        raise ValueError("control vectors are real")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size != space.size:
        raise ValueError(f"vector has {array.size} entries; the space has {space.size}")
    return array


def _reference_values(reference: Any, space: ControlSpace) -> Optional[np.ndarray]:
    if reference is None:
        return None
    if isinstance(reference, ControlState):
        return np.array(reference.vector(space).values, copy=True)
    return np.array(_values_on(space, reference), copy=True)


def _column_squares(matrix: csr_matrix) -> np.ndarray:
    return np.asarray(matrix.multiply(matrix).sum(axis=0), dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


class _SparseModelOperator(ModelOperator):
    """A sparse matrix acting on optimizer vectors (untyped range)."""

    def __init__(self, matrix: Any, space: ControlSpace) -> None:
        self.matrix = csr_matrix(matrix, dtype=np.float64)
        self.space = space
        super().__init__(space, None, dtype=np.float64, shape=self.matrix.shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ np.asarray(x, dtype=np.float64).reshape(-1))

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix.T @ np.asarray(x, dtype=np.float64).reshape(-1))

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ np.asarray(x, dtype=np.float64))


class _SymmetricModelOperator(ModelOperator):
    """Self-adjoint action ``space -> space`` given as a callable on arrays."""

    def __init__(
        self, action: Callable[[np.ndarray], np.ndarray], space: ControlSpace
    ) -> None:
        self.action = action
        super().__init__(space, space, dtype=np.float64)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.action(np.asarray(x, dtype=np.float64).reshape(-1)),
            dtype=np.float64,
        )

    _rmatvec = _matvec

    def _adjoint(self) -> ModelOperator:
        return self

    def _transpose(self) -> ModelOperator:
        return self


# ---------------------------------------------------------------------------
# regularization
# ---------------------------------------------------------------------------


class Regularization:
    """Unbound, immutable regularization configuration.

    Smooth custom terms implement ``bind(space)`` and return a
    :class:`BoundRegularization`. Native model configurations require the
    problem and stage context supplied by FWI/LSRTM. Terms compose with ``+``
    (a :class:`Sum`) and scale with ``*`` by nonnegative scalars (a :class:`Scaled`).
    """

    def bind(self, space: ControlSpace) -> "BoundRegularization":  # pragma: no cover
        raise NotImplementedError

    def __add__(self, other: Any) -> "Regularization":
        if not isinstance(other, Regularization):
            return NotImplemented
        return Sum(self, other)

    __radd__ = __add__

    def __mul__(self, factor: Any) -> "Regularization":
        if not isinstance(factor, (int, float, np.integer, np.floating)):
            return NotImplemented
        return Scaled(self, float(factor))

    __rmul__ = __mul__


class BoundRegularization:
    """A regularization evaluated on one :class:`ControlSpace` (optimizer layout).

    Vectors are :class:`ControlVector` on ``space`` (or equivalent spaces) or
    ndarrays of ``space.size``; gradients come back as :class:`ControlVector`.
    """

    def __init__(self, regularization: Regularization, space: ControlSpace) -> None:
        self.regularization = regularization
        self.space = space

    # -- helpers ----------------------------------------------------------------

    def _values(self, v: Any) -> np.ndarray:
        return _values_on(self.space, v)

    def _wrap(self, values: np.ndarray) -> ControlVector:
        return ControlVector(np.asarray(values, dtype=np.float64), self.space)

    # -- protocol -----------------------------------------------------------------

    def value(self, v: Any) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def gradient(self, v: Any) -> ControlVector:  # pragma: no cover - abstract
        raise NotImplementedError

    def hessian_operator(self, v: Any) -> ModelOperator:  # pragma: no cover
        """Return the Hessian of this smooth regularization term at ``v``."""

        raise NotImplementedError

    def curvature_diagonal(self, v: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def operator(self) -> Optional[ModelOperator]:
        """Return ``R`` with ``hessian == R.T @ R`` (quadratic regularization terms only)."""

        return None

    def residual(self, v: Any) -> np.ndarray:
        """Return the affine least-squares residual, when an operator exists."""

        raise NotImplementedError(
            "regularization does not expose a least-squares residual"
        )

    def __call__(self, v: Any) -> float:
        return self.value(v)

    def __repr__(self) -> str:
        return f"Bound{self.regularization!r}[{self.space.size}]"


class _BoundQuadraticForm(BoundRegularization):
    """``0.5 * ||R (v - ref)||^2`` for a sparse ``R`` on the optimizer layout."""

    def __init__(
        self,
        regularization: Regularization,
        space: ControlSpace,
        matrix: csr_matrix,
        reference: Optional[np.ndarray],
    ) -> None:
        super().__init__(regularization, space)
        self.matrix = csr_matrix(matrix, dtype=np.float64)
        if self.matrix.shape[1] != space.size:
            raise ValueError(
                f"regularization operator has {self.matrix.shape[1]} columns; the space "
                f"has {space.size}"
            )
        self.reference = reference

    def _shift(self, v: Any) -> np.ndarray:
        values = self._values(v)
        return values if self.reference is None else values - self.reference

    def residual(self, v: Any) -> np.ndarray:
        return np.asarray(self.matrix @ self._shift(v))

    def value(self, v: Any) -> float:
        residual = self.residual(v)
        return 0.5 * float(np.dot(residual, residual))

    def gradient(self, v: Any) -> ControlVector:
        return self._wrap(self.matrix.T @ (self.matrix @ self._shift(v)))

    def hessian_operator(self, v: Any = None) -> ModelOperator:
        matrix = self.matrix

        def action(x: np.ndarray) -> np.ndarray:
            return np.asarray(matrix.T @ (matrix @ x))

        return _SymmetricModelOperator(action, self.space)

    def curvature_diagonal(self, v: Any = None) -> np.ndarray:
        return _column_squares(self.matrix)

    def operator(self) -> Optional[ModelOperator]:
        return _SparseModelOperator(self.matrix, self.space)


class _NativeModelRegularization(Regularization):
    """Configuration for Sauce-owned model regularization."""

    def bind(self, space: ControlSpace) -> BoundRegularization:
        raise NotImplementedError(
            "Model regularization requires native callbacks; pass it to FWI/LSRTM "
            "or bind NativeRegularization with a problem and linearization"
        )


@dataclass(frozen=True)
class Tikhonov(_NativeModelRegularization):
    r"""Native quadratic derivative regularization for FWI/LSRTM.

    Sauce evaluates ``alpha/2 * integral |D^order(m-reference)|^2`` on the
    native material control basis. Spatial axes use km; angular axes use radians.
    Order two requires a spline of degree at least two. Fixed coefficients
    retain their full values in the integral. ``reference`` can be a complete
    ControlState or an active-space vector. This object configures native
    callbacks; it does not expose a Python derivative or matrix discretization.
    """

    alpha: float
    order: int = 1
    reference: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _finite_scalar(self.alpha, "alpha"))
        order = int(self.order)
        if order not in (1, 2):
            raise ValueError("Tikhonov order must be 1 or 2")
        object.__setattr__(self, "order", order)


@dataclass(frozen=True)
class Quadratic(Regularization):
    """Arbitrary quadratic regularization ``0.5 * weight * ||matrix (v - reference)||^2``.

    Wraps the semantics of
    :class:`~frequensolve.inversion.least_squares.QuadraticRegularization`
    (residual ``sqrt(weight) * matrix @ (v - reference)``) for a sparse or
    dense ``matrix`` on the whole optimizer vector.

    Args:
        matrix: ``(rows, space.size)`` sparse or dense real matrix.
        weight: Nonnegative scale.
        reference: Optional reference vector (see :class:`Tikhonov`).
    """

    matrix: Any
    weight: float = 1.0
    reference: Any = None

    def __post_init__(self) -> None:
        matrix = self.matrix
        if not issparse(matrix):
            matrix = np.asarray(matrix)
            if np.iscomplexobj(matrix):
                raise ValueError("Quadratic matrix must be real")
            matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("Quadratic matrix must be non-empty and 2-D")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "weight", _finite_scalar(self.weight, "weight"))

    def bind(self, space: ControlSpace) -> BoundRegularization:
        if self.matrix.shape[1] != space.size:
            raise ValueError(
                f"Quadratic matrix has {self.matrix.shape[1]} columns; the space "
                f"has {space.size} optimizer DOFs"
            )
        return _BoundQuadraticForm(
            self,
            space,
            csr_matrix(self.matrix) * math.sqrt(self.weight),
            _reference_values(self.reference, space),
        )

    def least_squares_term(self, space: ControlSpace) -> QuadraticRegularization:
        """Return the equivalent toolkit :class:`QuadraticRegularization`."""

        dense = self.matrix.toarray() if issparse(self.matrix) else self.matrix
        return QuadraticRegularization(
            dense,
            weight=self.weight,
            reference=_reference_values(self.reference, space),
        )


@dataclass(frozen=True)
class TV(_NativeModelRegularization):
    r"""Native unsmoothed total variation for FWI/LSRTM.

    Sauce evaluates ``sqrt(alpha) * integral |D^order(m-reference)|`` on its
    native mesh, spline or tensor control basis and uses split-Bregman shrinkage.
    ``epsilon`` controls splitting, not smoothing of the norm. Order two
    requires a spline of degree at least two. ``reference`` can be a complete
    ControlState or an active-space vector. Use a complete state to specify
    reference values at fixed coefficients.
    """

    alpha: float
    epsilon: float = 1.0e-3
    order: int = 1
    reference: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _finite_scalar(self.alpha, "alpha"))
        epsilon = _finite_scalar(self.epsilon, "epsilon")
        if epsilon <= 0.0:
            raise ValueError("TV epsilon must be positive")
        object.__setattr__(self, "epsilon", epsilon)
        order = int(self.order)
        if order not in (1, 2):
            raise ValueError("TV order must be 1 or 2")
        object.__setattr__(self, "order", order)


@dataclass(frozen=True)
class TGV(_NativeModelRegularization):
    """Native second-order total generalized variation for FWI and LSRTM.

    Sauce minimizes over an auxiliary vector field using split-Bregman
    shrinkage. ``epsilon`` controls splitting, not smoothing of the norm.
    Use NativeRegularization for a reference state or callback tolerances.
    """

    alpha1: float
    alpha2: float
    epsilon: float = 1.0e-3

    def __post_init__(self) -> None:
        config = SmoothingConfig(
            kind="tgv", alpha1=self.alpha1, alpha2=self.alpha2, epsilon=self.epsilon
        )
        for name in ("alpha1", "alpha2", "epsilon"):
            object.__setattr__(self, name, getattr(config, name))


class Sum(Regularization):
    """Sum of regularization terms (``a + b`` builds one; nested sums are flattened)."""

    def __init__(self, *regularizations: Regularization) -> None:
        terms: List[Regularization] = []
        for regularization in regularizations:
            if not isinstance(regularization, Regularization):
                raise TypeError(
                    f"{type(regularization).__name__} is not a Regularization"
                )
            if isinstance(regularization, Sum):
                terms.extend(regularization.regularizations)
            else:
                terms.append(regularization)
        if not terms:
            raise ValueError("Sum requires at least one regularization")
        self.regularizations: Tuple[Regularization, ...] = tuple(terms)

    def bind(self, space: ControlSpace) -> BoundRegularization:
        return _BoundSum(self, space, [p.bind(space) for p in self.regularizations])

    def __repr__(self) -> str:
        return "Sum(" + ", ".join(repr(p) for p in self.regularizations) + ")"


class _BoundSum(BoundRegularization):
    def residual(self, v: Any) -> np.ndarray:
        return np.concatenate([term.residual(v) for term in self.terms])

    def __init__(
        self,
        regularization: Regularization,
        space: ControlSpace,
        terms: Sequence[BoundRegularization],
    ) -> None:
        super().__init__(regularization, space)
        self.terms = list(terms)

    def value(self, v: Any) -> float:
        return float(sum(term.value(v) for term in self.terms))

    def gradient(self, v: Any) -> ControlVector:
        out = np.zeros(self.space.size, dtype=np.float64)
        for term in self.terms:
            out += term.gradient(v).values
        return self._wrap(out)

    def hessian_operator(self, v: Any) -> ModelOperator:
        operators = [term.hessian_operator(v) for term in self.terms]

        def action(x: np.ndarray) -> np.ndarray:
            out = np.zeros(self.space.size, dtype=np.float64)
            for op in operators:
                out += np.asarray(op.matvec(x)).reshape(-1)
            return out

        return _SymmetricModelOperator(action, self.space)

    def curvature_diagonal(self, v: Any) -> np.ndarray:
        out = np.zeros(self.space.size, dtype=np.float64)
        for term in self.terms:
            out += term.curvature_diagonal(v)
        return out

    def operator(self) -> Optional[ModelOperator]:
        matrices = []
        for term in self.terms:
            op = term.operator()
            if not isinstance(op, _SparseModelOperator):
                return None
            matrices.append(op.matrix)
        return _SparseModelOperator(vstack(matrices, format="csr"), self.space)


class Scaled(Regularization):
    """A regularization multiplied by a nonnegative scalar (``0.5 * TV(...)``)."""

    def __init__(self, regularization: Regularization, factor: float) -> None:
        if not isinstance(regularization, Regularization):
            raise TypeError(f"{type(regularization).__name__} is not a Regularization")
        self.regularization = regularization
        self.factor = _finite_scalar(factor, "regularization scale")

    def bind(self, space: ControlSpace) -> BoundRegularization:
        return _BoundScaled(self, space, self.regularization.bind(space), self.factor)

    def __repr__(self) -> str:
        return f"{self.factor:g} * {self.regularization!r}"


class _BoundScaled(BoundRegularization):
    def residual(self, v: Any) -> np.ndarray:
        return math.sqrt(self.factor) * self.inner.residual(v)

    def __init__(
        self,
        regularization: Regularization,
        space: ControlSpace,
        inner: BoundRegularization,
        factor: float,
    ) -> None:
        super().__init__(regularization, space)
        self.inner = inner
        self.factor = float(factor)

    def value(self, v: Any) -> float:
        return self.factor * self.inner.value(v)

    def gradient(self, v: Any) -> ControlVector:
        return self._wrap(self.factor * self.inner.gradient(v).values)

    def hessian_operator(self, v: Any) -> ModelOperator:
        inner = self.inner.hessian_operator(v)
        factor = self.factor

        def action(x: np.ndarray) -> np.ndarray:
            return factor * np.asarray(inner.matvec(x)).reshape(-1)

        return _SymmetricModelOperator(action, self.space)

    def curvature_diagonal(self, v: Any) -> np.ndarray:
        return self.factor * self.inner.curvature_diagonal(v)

    def operator(self) -> Optional[ModelOperator]:
        op = self.inner.operator()
        if not isinstance(op, _SparseModelOperator):
            return None
        return _SparseModelOperator(math.sqrt(self.factor) * op.matrix, self.space)


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------

Smoothing = SmoothingConfig
"""Sauce's ``control_sensitivities.Smoothing`` configuration (user-facing name)."""


def _smoothing_source_job(problem: "ImagingProblem") -> FWIOperatorJob:
    """Return the job Sauce's ``--smooth`` postprocess is parameterized by.

    The most recent linearize job that wrote a covector is preferred (its
    saved payload already describes the simulation, active blocks and
    frequencies); before any linearization an unrun linearize job shell is
    built the same way, which is all ``SmoothJob`` needs from its source.
    """

    shared = problem._shared
    for linearization in reversed(list(shared.linearizations.values())):
        job = linearization.job
        if isinstance(job, FWIOperatorJob) and job.covector is not None:
            return job
    return problem._linearize_job(problem.space, None, gradient=True)


def smooth(
    vector: Any,
    smoothing: Any,
    problem: "ImagingProblem",
    *,
    weights: Optional[Sequence[float]] = None,
) -> ControlVector:
    """Apply Sauce's smoothing Riesz map to one control vector.

    The vector is written as an ``fs-control-vector-1`` file, a
    :class:`~frequensolve.imaging.jobs.SmoothJob` with ``input_vector`` is run
    through ``problem.backend`` as a postprocess-only submission, and the
    smoothed ``model.*`` blocks are read back.  Blocks Sauce does not smooth
    (source, interface, reflectivity) pass through unchanged.

    Args:
        vector: :class:`ControlVector` on ``problem.space`` or an array of
            that size.
        smoothing: :class:`Smoothing` or a Sauce smoothing mapping.
        problem: The :class:`~frequensolve.imaging.problem.ImagingProblem`
            whose backend runs the job.
        weights: Reserved; Sauce ignores frequency weights when an explicit
            input vector is smoothed, so any value other than ``None`` raises.
    """

    if weights is not None:
        raise ValueError(
            "weights do not apply to explicit-vector smoothing (Sauce ignores "
            "frequency weights with control_sensitivities.input)"
        )
    config = SmoothingConfig.from_value(smoothing)
    if config is None:
        raise ValueError("smooth requires a smoothing configuration")
    return _smooth_with_source(vector, config, problem, _smoothing_source_job(problem))


def _smooth_with_source(
    vector: Any,
    config: SmoothingConfig,
    problem: "ImagingProblem",
    source: FWIOperatorJob,
) -> ControlVector:
    """Process a vector using the model and frequency context of one saved job."""

    space = problem.space
    control = (
        vector if isinstance(vector, ControlVector) else ControlVector(vector, space)
    )
    if control.space is not space and not control.space.equivalent(space):
        raise ValueError("vector does not live on the problem's control space")
    if not any(name.startswith("model.") for name in space.blocks):
        raise ValueError("smoothing requires at least one model.* block")
    job = SmoothJob(
        source,
        smoothing=config,
        input_vector=control.to_file(),
        gradient="smoothed.h5",
        name=problem.backend.job_name("smooth"),
    )
    problem.backend.run(job, postprocess_only=True)
    smoothed = ControlVectorFile.read(job.gradient_file())
    full = space.to_sauce_vector(control)
    for block, sl in zip(space.resolved_blocks, space.full_slices.values()):
        if not block.name.startswith("model."):
            continue
        try:
            values = smoothed[block.name]
        except KeyError as exc:
            raise ValueError(
                f"smoothed vector {job.gradient_file()} has no block {block.name!r}"
            ) from exc
        if values.size != block.size:
            raise ValueError(
                f"smoothed block {block.name!r} has {values.size} entries; "
                f"expected {block.size}"
            )
        full[sl] = values
    return space.from_sauce_vector(full)


# ---------------------------------------------------------------------------
# preconditioners
# ---------------------------------------------------------------------------


class Preconditioner:
    """Unbound preconditioner configuration; ``bind(space)`` makes it usable."""

    def bind(self, space: ControlSpace) -> "BoundPreconditioner":  # pragma: no cover
        raise NotImplementedError


class BoundPreconditioner:
    """Approximate inverse-Hessian action on one control space.

    ``update(linearization, regularization)`` refreshes the approximation at a new
    point (a no-op for point-independent preconditioners); ``apply(g)`` (or
    ``self(g)``) returns ``M^{-1} g`` as a :class:`ControlVector`.
    """

    def __init__(self, preconditioner: Preconditioner, space: ControlSpace) -> None:
        self.preconditioner = preconditioner
        self.space = space

    def update(
        self,
        linearization: "Linearization",
        regularization: Optional[BoundRegularization] = None,
    ) -> None:
        """Refresh the approximation at ``linearization`` (default: no-op)."""

    def apply(self, g: Any) -> ControlVector:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, g: Any) -> ControlVector:
        return self.apply(g)

    def operator(self) -> ModelOperator:
        """Return the action as a self-adjoint :class:`ModelOperator` (CG ``M``)."""

        return _SymmetricModelOperator(lambda x: self.apply(x).values, self.space)

    def __repr__(self) -> str:
        return f"Bound{self.preconditioner!r}[{self.space.size}]"


@dataclass(frozen=True)
class Identity(Preconditioner):
    """The identity metric (``apply(g) == g``)."""

    def bind(self, space: ControlSpace) -> BoundPreconditioner:
        return _BoundIdentity(self, space)


class _BoundIdentity(BoundPreconditioner):
    def apply(self, g: Any) -> ControlVector:
        return ControlVector(np.array(_values_on(self.space, g), copy=True), self.space)


@dataclass(frozen=True)
class FromOperator(Preconditioner):
    """User-supplied inverse-Hessian action.

    Args:
        operator: Callable ``g -> M^{-1} g``, a SciPy ``LinearOperator`` /
            :class:`ModelOperator`, a sparse matrix or a dense
            ``(size, size)`` array.
    """

    operator: Any

    def __post_init__(self) -> None:
        op = self.operator
        if not (
            callable(op)
            or isinstance(op, LinearOperator)
            or issparse(op)
            or isinstance(op, np.ndarray)
        ):
            raise TypeError("FromOperator needs a callable, operator or matrix")

    def bind(self, space: ControlSpace) -> BoundPreconditioner:
        op = self.operator
        if hasattr(op, "shape") and tuple(op.shape) != (space.size, space.size):
            raise ValueError(
                f"preconditioner operator has shape {tuple(op.shape)}; expected "
                f"({space.size}, {space.size})"
            )
        return _BoundFromOperator(self, space, op)


class _BoundFromOperator(BoundPreconditioner):
    def __init__(
        self, preconditioner: Preconditioner, space: ControlSpace, op: Any
    ) -> None:
        super().__init__(preconditioner, space)
        self.op = op

    def apply(self, g: Any) -> ControlVector:
        values = _values_on(self.space, g)
        op = self.op
        if isinstance(op, (LinearOperator, np.ndarray)) or issparse(op):
            result = op @ values
        else:
            result = op(g if isinstance(g, ControlVector) else values)
        out = np.asarray(result)
        if isinstance(result, ControlVector):
            out = result.values
        if np.iscomplexobj(out):
            raise ValueError("preconditioner output must be real")
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        if out.size != self.space.size:
            raise ValueError(
                f"preconditioner returned {out.size} entries; expected {self.space.size}"
            )
        return ControlVector(out, self.space)


@dataclass(frozen=True)
class Diagonal(Preconditioner):
    """Damped inverse of a randomized Gauss-Newton diagonal estimate.

    ``update`` estimates ``diag(Re J^H W J)`` with Rademacher probes of
    ``linearization.normal`` (Hutchinson: ``E[z * (N z)] = diag(N)``, one
    normal action per probe; negative samples are clipped to zero), adds the
    regularization's ``curvature_diagonal`` at the linearization point and
    wraps the sum in
    :class:`~frequensolve.inversion.preconditioning.DiagonalInverseHessian`
    (block-wise relative damping and inverse-ratio clipping).  ``apply``
    raises until the first ``update``.

    Args:
        probe_count: Number of Rademacher probes (``probes="rademacher"``).
        seed: Probe seed.
        relative_damping: Damping relative to each block's maximum diagonal.
        maximum_inverse_ratio: Cap on each block's inverse dynamic range.
        probes: ``"rademacher"`` (randomized, ``probe_count`` normal actions)
            or ``"unit"`` (exact diagonal, ``space.size`` normal actions).
    """

    probe_count: int = 4
    seed: Optional[int] = 0
    relative_damping: float = 1.0e-2
    maximum_inverse_ratio: Optional[float] = 1.0e3
    probes: str = "rademacher"

    def __post_init__(self) -> None:
        count = int(self.probe_count)
        if count < 1:
            raise ValueError("probe_count must be positive")
        object.__setattr__(self, "probe_count", count)
        probes = str(self.probes).strip().lower()
        if probes not in {"rademacher", "unit"}:
            raise ValueError("probes must be 'rademacher' or 'unit'")
        object.__setattr__(self, "probes", probes)
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        _finite_scalar(self.relative_damping, "relative_damping")
        if self.maximum_inverse_ratio is not None:
            _finite_scalar(
                self.maximum_inverse_ratio, "maximum_inverse_ratio", minimum=1.0
            )

    def bind(self, space: ControlSpace) -> BoundPreconditioner:
        return _BoundDiagonal(self, space)


class _BoundDiagonal(BoundPreconditioner):
    def __init__(self, preconditioner: Diagonal, space: ControlSpace) -> None:
        super().__init__(preconditioner, space)
        self.config: Diagonal = preconditioner
        self.estimate: Optional[GaussNewtonDiagonalEstimate] = None
        self.inverse: Optional[DiagonalInverseHessian] = None

    def update(
        self,
        linearization: "Linearization",
        regularization: Optional[BoundRegularization] = None,
    ) -> None:
        """Re-estimate the diagonal at ``linearization`` (adopting its space)."""

        space = linearization.space
        if space is not self.space and not space.equivalent(self.space):
            # Support masks are adopted at the first linearize; follow them.
            self.space = space
        if regularization is not None and not (
            regularization.space is space or regularization.space.equivalent(space)
        ):
            raise ValueError("regularization is bound to a different control space")
        normal = linearization.normal
        size = space.size
        config = self.config
        data = np.zeros(size, dtype=np.float64)
        if config.probes == "unit":
            for i in range(size):
                probe = np.zeros(size)
                probe[i] = 1.0
                data[i] = float(normal.matvec(probe).values[i])
            probe_count = size
        else:
            rng = np.random.default_rng(config.seed)
            for _ in range(config.probe_count):
                probe = rng.choice((-1.0, 1.0), size=size)
                data += probe * normal.matvec(probe).values
            data /= config.probe_count
            probe_count = config.probe_count
        data = np.maximum(data, 0.0)
        regularization_diagonal = (
            np.zeros(size)
            if regularization is None
            else np.asarray(regularization.curvature_diagonal(linearization.point))
        )
        self.estimate = GaussNewtonDiagonalEstimate(
            data, np.maximum(regularization_diagonal, 0.0), probe_count, config.seed
        )
        self.inverse = DiagonalInverseHessian(
            self.estimate.total,
            block_sizes=[sl.stop - sl.start for sl in space.slices.values()],
            relative_damping=config.relative_damping,
            maximum_inverse_ratio=config.maximum_inverse_ratio,
        )

    def apply(self, g: Any) -> ControlVector:
        if self.inverse is None:
            raise RuntimeError(
                "Diagonal preconditioner has no estimate yet; call "
                "update(linearization, regularization) first"
            )
        return ControlVector(self.inverse.apply(_values_on(self.space, g)), self.space)

    @property
    def diagonal(self) -> Optional[np.ndarray]:
        """Return the damped curvature diagonal of the last update."""

        return None if self.inverse is None else self.inverse.diagonal
