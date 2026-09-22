"""SciPy-style linear operators over control and data spaces.

:class:`ModelOperator` is a :class:`scipy.sparse.linalg.LinearOperator` that
knows its ``domain`` and ``range`` descriptors (a
:class:`~frequensolve.imaging.controls.ControlSpace` in the optimizer layout
or a :class:`~frequensolve.imaging.data.DataSpace`).  ``matvec`` / ``rmatvec``
/ ``@`` accept typed vectors or plain arrays and return typed vectors;
composition with scalars, SciPy operators and sparse matrices (``H + alpha *
R.T @ R``) keeps the typed interface, so results drop straight into ``cg``,
``lsqr`` or PyLops.

Convention: ``J.H @ r`` returns the **real** control covector (real part of
the Hermitian pairing, no factor 2), matching Sauce's ``vjp``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from frequensolve.imaging.controls import ControlSpace, ControlVector
from frequensolve.imaging.data import DataSpace, DataVector
from frequensolve.inversion.validation import real_adjoint_test

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frequensolve.imaging.problem import Linearization

__all__ = ["Jacobian", "ModelOperator", "Normal"]

Descriptor = Union[ControlSpace, DataSpace, None]


def _size(descriptor: Descriptor, fallback: Optional[int] = None) -> int:
    if descriptor is None:
        if fallback is None:
            raise ValueError("an untyped operator side needs an explicit size")
        return int(fallback)
    return int(descriptor.size)


def _values(x: Any, descriptor: Descriptor) -> np.ndarray:
    """Return the raw array of ``x`` after checking it lives on ``descriptor``."""

    if isinstance(x, ControlVector):
        if isinstance(descriptor, ControlSpace) and not (
            x.space is descriptor or x.space.equivalent(descriptor)
        ):
            raise ValueError("control vector belongs to a different control space")
        return x.values
    if isinstance(x, DataVector):
        if isinstance(descriptor, DataSpace) and x.space != descriptor:
            raise ValueError("data vector belongs to a different data space")
        return x.values
    return np.asarray(x)


def _wrap(values: np.ndarray, descriptor: Descriptor) -> Any:
    """Wrap raw operator output in the typed vector of ``descriptor``."""

    if isinstance(descriptor, ControlSpace):
        if np.iscomplexobj(values):
            if np.max(np.abs(values.imag), initial=0.0) > 0.0:
                raise ValueError("control-side operator output must be real")
            values = values.real
        return ControlVector(values, descriptor)
    if isinstance(descriptor, DataSpace):
        return DataVector(np.asarray(values, dtype=descriptor.dtype), descriptor)
    return np.asarray(values)


def _is_operator_like(x: Any) -> bool:
    return isinstance(x, LinearOperator) or issparse(x)


class ModelOperator(LinearOperator):
    """Typed :class:`LinearOperator` between control and data spaces.

    Subclasses implement ``_matvec`` / ``_rmatvec`` on plain arrays; the
    public ``matvec`` / ``rmatvec`` / ``dot`` / ``@`` accept typed vectors and
    return :class:`ControlVector` / :class:`DataVector` according to
    ``range`` / ``domain`` (plain arrays when a side is untyped).

    Args:
        domain: Descriptor of the input side (or ``None`` with ``shape``).
        range: Descriptor of the output side (or ``None`` with ``shape``).
        dtype: Operator dtype.
        shape: Explicit ``(rows, columns)`` when a side is untyped.
    """

    def __init__(
        self,
        domain: Descriptor,
        range: Descriptor,
        *,
        dtype: Any = np.float64,
        shape: Optional[Any] = None,
    ) -> None:
        self.domain = domain
        self.range = range
        rows = _size(range, None if shape is None else shape[0])
        cols = _size(domain, None if shape is None else shape[1])
        super().__init__(np.dtype(dtype), (rows, cols))

    # -- raw actions (subclasses) --------------------------------------------

    def _matvec(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- typed actions --------------------------------------------------------

    def matvec(self, x: Any) -> Any:
        """Return ``A @ x`` as a typed vector of ``range``."""

        return _wrap(super().matvec(_values(x, self.domain)), self.range)

    def rmatvec(self, x: Any) -> Any:
        """Return ``A^H @ x`` as a typed vector of ``domain``."""

        return _wrap(super().rmatvec(_values(x, self.range)), self.domain)

    def dot(self, x: Any) -> Any:
        """Apply to a vector, or compose with a scalar, operator or matrix."""

        if isinstance(x, (ControlVector, DataVector)):
            return self.matvec(x)
        if np.isscalar(x):
            return self._scaled(x)
        if _is_operator_like(x):
            other = aslinearoperator(x)
            return _CompositeModelOperator(
                LinearOperator.dot(self, other),
                domain=getattr(x, "domain", None),
                range=self.range,
            )
        array = np.asarray(x)
        if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1):
            return self.matvec(array.reshape(-1))
        return super().matmat(array)

    __mul__ = dot

    def __matmul__(self, other: Any) -> Any:
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self.dot(other)

    def __rmul__(self, x: Any) -> Any:
        if np.isscalar(x):
            return self._scaled(x)
        return super().__rmul__(x)

    def __neg__(self) -> "ModelOperator":
        return self._scaled(-1.0)

    def _scaled(self, alpha: Any) -> "ModelOperator":
        return _CompositeModelOperator(
            LinearOperator.dot(self, alpha), domain=self.domain, range=self.range
        )

    def __add__(self, x: Any) -> Any:
        if not _is_operator_like(x):
            if isinstance(x, np.ndarray) and x.ndim == 2:
                x = aslinearoperator(x)
            else:
                return NotImplemented
        other = aslinearoperator(x)
        return _CompositeModelOperator(
            LinearOperator.__add__(self, other), domain=self.domain, range=self.range
        )

    __radd__ = __add__

    def __sub__(self, x: Any) -> Any:
        return self.__add__(-x)

    def __rsub__(self, x: Any) -> Any:
        return (-self).__add__(x)

    def _adjoint(self) -> "ModelOperator":
        return _AdjointModelOperator(self)

    def _transpose(self) -> "ModelOperator":
        return _TransposedModelOperator(self)

    # -- interop --------------------------------------------------------------

    def to_pylops(self) -> Any:
        """Return a ``pylops.LinearOperator`` wrapping this operator."""

        try:
            import pylops
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "to_pylops requires PyLops; install it with 'pip install pylops'"
            ) from exc
        return pylops.LinearOperator(
            Op=self, dtype=self.dtype, shape=self.shape, name=type(self).__name__
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"domain={_describe(self.domain)}, range={_describe(self.range)})"
        )


def _describe(descriptor: Descriptor) -> str:
    if isinstance(descriptor, ControlSpace):
        return f"ControlSpace[{descriptor.size}]"
    if isinstance(descriptor, DataSpace):
        return f"DataSpace[{descriptor.size}]"
    return "ndarray"


class _CompositeModelOperator(ModelOperator):
    """A SciPy operator (sum, product, scaling) re-typed on model descriptors."""

    def __init__(
        self, inner: LinearOperator, *, domain: Descriptor, range: Descriptor
    ) -> None:
        self.inner = inner
        super().__init__(domain, range, dtype=inner.dtype, shape=inner.shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.matvec(x))

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.rmatvec(x))

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.matmat(x))

    def _adjoint(self) -> ModelOperator:
        return _CompositeModelOperator(
            self.inner.H, domain=self.range, range=self.domain
        )

    def _transpose(self) -> ModelOperator:
        return _CompositeModelOperator(
            self.inner.T, domain=self.range, range=self.domain
        )


class _AdjointModelOperator(ModelOperator):
    def __init__(self, parent: ModelOperator) -> None:
        self.parent = parent
        super().__init__(
            parent.range,
            parent.domain,
            dtype=parent.dtype,
            shape=(parent.shape[1], parent.shape[0]),
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.parent._rmatvec(x)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self.parent._matvec(x)

    def _adjoint(self) -> ModelOperator:
        return self.parent


class _TransposedModelOperator(ModelOperator):
    def __init__(self, parent: ModelOperator) -> None:
        self.parent = parent
        super().__init__(
            parent.range,
            parent.domain,
            dtype=parent.dtype,
            shape=(parent.shape[1], parent.shape[0]),
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return np.conj(self.parent._rmatvec(np.conj(x)))

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return np.conj(self.parent._matvec(np.conj(x)))

    def _transpose(self) -> ModelOperator:
        return self.parent


# ---------------------------------------------------------------------------
# Jacobian and normal operators
# ---------------------------------------------------------------------------


def _real_direction(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x).reshape(-1)
    if np.iscomplexobj(array):
        if np.max(np.abs(array.imag), initial=0.0) > 0.0:
            raise ValueError("control directions are real")
        array = array.real
    return np.asarray(array, dtype=np.float64)


class Jacobian(ModelOperator):
    """Jacobian of the data with respect to the active controls.

    ``J @ dv`` (Sauce ``jvp``) maps a real :class:`ControlVector` on
    ``linearization.space`` to a complex :class:`DataVector`; ``J.H @ r``
    (``vjp``) returns the real covector.  Frozen DOFs are expanded as zeros
    on the way in and dropped on the way out.
    """

    def __init__(self, linearization: "Linearization") -> None:
        self.linearization = linearization
        super().__init__(
            linearization.space, linearization.data_space, dtype=np.complex128
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.linearization.jvp(_real_direction(x)).values

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        dual = np.asarray(x, dtype=np.complex128).reshape(-1)
        return self.linearization.vjp(dual).values

    def dot_test(self, seed: int = 0, *, tolerance: float = 1.0e-8) -> Dict[str, Any]:
        """Check ``<J dv, r>_Re == <dv, J^H r>`` on random vectors."""

        dv = self.linearization.space.random(seed)
        r = self.linearization.data_space.random(seed + 1)
        return real_adjoint_test(
            lambda p: np.asarray(self @ p),
            lambda y: np.asarray(self.H @ y),
            dv.values,
            r.values,
            relative_tolerance=tolerance,
        )


class Normal(ModelOperator):
    """Frozen Gauss-Newton normal operator ``Re(J^H W J)`` (Sauce ``normal``).

    Self-adjoint on ``linearization.space``; ``.H`` returns the operator
    itself.
    """

    def __init__(self, linearization: "Linearization") -> None:
        self.linearization = linearization
        super().__init__(linearization.space, linearization.space, dtype=np.float64)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.linearization.apply_normal(_real_direction(x)).values

    _rmatvec = _matvec

    def _adjoint(self) -> ModelOperator:
        return self

    def _transpose(self) -> ModelOperator:
        return self
