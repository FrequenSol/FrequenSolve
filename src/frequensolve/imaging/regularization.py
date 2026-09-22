"""Penalties, smoothing and preconditioners for the imaging API.

Three small families live here (spec section 5.3):

- **Penalties** (:class:`Tikhonov`, :class:`TV`, :class:`Quadratic`,
  :class:`Sum`) are part of the objective and are evaluated in Python on the
  optimizer coordinates of a :class:`~frequensolve.imaging.controls.ControlSpace`
  (block coefficients after the Sauce transform, frozen DOFs excluded).  A
  penalty is an unbound, frozen configuration; :meth:`Penalty.bind` returns a
  :class:`BoundPenalty` exposing ``value``, ``gradient``, ``hessian_operator``,
  ``curvature_diagonal`` and, for quadratic penalties, ``operator`` (``R`` with
  ``hessian == R.T @ R``) so ``H + R.T @ R`` enters Newton-CG.
- **Smoothing** (:class:`Smoothing`, :func:`smooth`) is Sauce's native
  ``control_sensitivities.Smoothing`` Riesz map, a step transform on
  gradients that is *not* part of the objective.  :func:`smooth` runs it on
  one explicit vector through a :class:`~frequensolve.imaging.jobs.SmoothJob`.
- **Preconditioners** (:class:`Diagonal`, :class:`FromOperator`,
  :class:`Identity`) change the linear solve only; a bound preconditioner is
  refreshed with :meth:`BoundPreconditioner.update` at a new linearization
  and applied with :meth:`BoundPreconditioner.apply`.

Lattice conventions
-------------------

Difference operators are scaled by the block's physical spacing.  1-D
profiles (hat) differentiate along the profile axis at the node coordinates;
B-spline profiles difference their coefficients at the Greville abscissae;
lattices (:class:`~frequensolve.imaging.controls.GridParameters`) sum one
term per lattice axis with that axis' spacing.  Blocks without a lattice
(source, interface, registry-only blocks) fall back to the identity
"difference" (a ridge toward the reference) when a nonzero weight is
requested for them; mesh blocks have no local topology and raise.

Frozen DOFs
-----------

Penalties operate on the optimizer layout.  A finite difference that
straddles a frozen node treats the node as absent: the row is dropped
(Tikhonov / Quadratic-style stacks) or zeroed (node-based TV gradients), so
frozen DOFs neither contribute to nor receive penalty curvature.

Illumination preconditioning
----------------------------

There is no cheap source of per-DOF illumination in the ``fwi_operator``
contract (it would need per-source wavefield energies that Sauce does not
export), so no ``Illumination`` preconditioner is provided; use
:class:`Diagonal` (randomized Gauss-Newton diagonal) instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.sparse import csr_matrix, identity, issparse, kron, vstack
from scipy.sparse.linalg import LinearOperator

from frequensolve.imaging._artifacts import ControlVectorFile, SmoothingConfig
from frequensolve.imaging.controls import (
    ControlSpace,
    ControlState,
    ControlVector,
    ResolvedBlock,
)
from frequensolve.imaging.jobs import FWIOperatorJob, SmoothJob
from frequensolve.imaging.operators import ModelOperator
from frequensolve.inversion.least_squares import QuadraticRegularization
from frequensolve.inversion.preconditioning import (
    DiagonalInverseHessian,
    GaussNewtonDiagonalEstimate,
)
from frequensolve.model.parameterization import (
    BSplineControl,
    HatControl,
    TensorHatControl,
)
from frequensolve.model.representation import ControlRepresentation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frequensolve.imaging.problem import ImagingProblem, Linearization

__all__ = [
    "BoundPenalty",
    "BoundPreconditioner",
    "Diagonal",
    "FromOperator",
    "Identity",
    "Penalty",
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

_LATTICE_KINDS = {"profile", "grid"}
_ZERO_DEFAULT_KINDS = {"source", "interface", "reflectivity", "registry"}


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


def _block_weights(
    space: ControlSpace,
    weights: Optional[Mapping[str, Any]],
    default: Callable[[ResolvedBlock], float],
) -> Dict[str, float]:
    table = {block.name: default(block) for block in space.resolved_blocks}
    for key, weight in dict(weights or {}).items():
        value = _finite_scalar(weight, f"weight of {key!r}")
        for block in space._select(key):
            table[block.name] = value
    return table


def _material_default(block: ResolvedBlock) -> float:
    return 0.0 if block.kind in _ZERO_DEFAULT_KINDS else 1.0


# ---------------------------------------------------------------------------
# lattice difference operators
# ---------------------------------------------------------------------------


def _greville(control: BSplineControl) -> np.ndarray:
    representation = ControlRepresentation(control)
    knots = representation.knots
    degree = representation.degree
    size = representation.size
    return np.array([knots[i + 1 : i + degree + 1].mean() for i in range(size)])


def _axis_nodes(block: ResolvedBlock) -> Optional[List[np.ndarray]]:
    """Return per-axis node coordinates (first axis fastest) or ``None``."""

    control = block.control
    if isinstance(control, HatControl):
        return [np.asarray(control.coordinates, dtype=np.float64)]
    if isinstance(control, BSplineControl):
        return [_greville(control)]
    if isinstance(control, TensorHatControl):
        return [
            np.asarray(values, dtype=np.float64) for values in control.axis_coordinates
        ]
    return None


def _difference_1d(nodes: np.ndarray, order: int) -> csr_matrix:
    """Return the ``(n - order) x n`` edge/interior finite-difference matrix."""

    n = int(nodes.size)
    if order == 1:
        if n < 2:
            return csr_matrix((0, n), dtype=np.float64)
        h = np.diff(nodes)
        rows = np.repeat(np.arange(n - 1), 2)
        cols = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).reshape(-1)
        data = np.stack([-1.0 / h, 1.0 / h], axis=1).reshape(-1)
        return csr_matrix((data, (rows, cols)), shape=(n - 1, n))
    if order == 2:
        if n < 3:
            return csr_matrix((0, n), dtype=np.float64)
        h = np.diff(nodes)
        left, right = h[:-1], h[1:]
        rows = np.repeat(np.arange(n - 2), 3)
        cols = np.stack(
            [np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)], axis=1
        ).reshape(-1)
        data = np.stack(
            [
                2.0 / (left * (left + right)),
                -2.0 / (left * right),
                2.0 / (right * (left + right)),
            ],
            axis=1,
        ).reshape(-1)
        return csr_matrix((data, (rows, cols)), shape=(n - 2, n))
    raise ValueError("difference order must be 1 or 2")


def _node_difference_1d(nodes: np.ndarray, order: int) -> csr_matrix:
    """Return the ``n x n`` node-based difference (zero rows at the boundary)."""

    n = int(nodes.size)
    inner = _difference_1d(nodes, order).tocoo()
    start = 0 if order == 1 else 1
    return csr_matrix(
        (inner.data, (inner.row + start, inner.col)), shape=(n, n), dtype=np.float64
    )


def _lattice_operator(shape: Sequence[int], axis: int, op1d: csr_matrix) -> csr_matrix:
    """Lift a 1-D operator along ``axis`` of a first-axis-fastest lattice."""

    result: Optional[Any] = None
    for j in reversed(range(len(shape))):
        factor = op1d if j == axis else identity(int(shape[j]), format="csr")
        result = factor if result is None else kron(result, factor, format="csr")
    assert result is not None
    return csr_matrix(result)


def _restrict(
    matrix: csr_matrix,
    mask: np.ndarray,
    offset: int,
    columns: int,
    *,
    drop_rows: bool,
) -> csr_matrix:
    """Map a block-local operator onto the optimizer layout.

    Rows touching a frozen column are dropped (``drop_rows``) or zeroed;
    active columns are shifted to ``offset`` in a ``columns``-wide matrix.
    """

    coo = matrix.tocoo()
    m = int(matrix.shape[0])
    keep_row = np.ones(m, dtype=bool)
    if coo.nnz:
        keep_row[np.unique(coo.row[~mask[coo.col]])] = False
    if drop_rows:
        row_map = np.full(m, -1, dtype=np.int64)
        row_map[keep_row] = np.arange(int(np.count_nonzero(keep_row)))
        rows_out = int(np.count_nonzero(keep_row))
    else:
        row_map = np.arange(m, dtype=np.int64)
        rows_out = m
    col_map = np.full(mask.size, -1, dtype=np.int64)
    col_map[mask] = np.arange(int(np.count_nonzero(mask))) + int(offset)
    select = keep_row[coo.row]
    return csr_matrix(
        (
            coo.data[select],
            (row_map[coo.row[select]], col_map[coo.col[select]]),
        ),
        shape=(rows_out, int(columns)),
        dtype=np.float64,
    )


def _block_operators(
    space: ControlSpace,
    block: ResolvedBlock,
    *,
    order: int,
    node_based: bool,
) -> List[csr_matrix]:
    """Return the per-axis difference operators of ``block`` on the space."""

    if block.kind == "mesh":
        raise NotImplementedError(
            f"mesh block {block.name!r} has no lattice: nodal differences need the "
            "property-space topology Sauce freezes in its artifact. Exclude the "
            f"block with weights={{{block.address!r}: 0}} or use Sauce-side "
            "smoothing (im.Smoothing / ImagingProblem(smoothing=...)) where the "
            "solver supports it"
        )
    mask = np.asarray(space.support[block.name], dtype=bool)
    offset = space.slices[block.name].start
    columns = space.size
    nodes = _axis_nodes(block)
    if nodes is None:
        eye = identity(block.size, format="csr")
        return [_restrict(eye, mask, offset, columns, drop_rows=True)]
    shape = tuple(int(axis.size) for axis in nodes)
    if int(np.prod(shape)) != block.size:
        raise ValueError(
            f"block {block.name!r} lattice {shape} does not match its {block.size} DOFs"
        )
    operators: List[csr_matrix] = []
    for k, axis_nodes in enumerate(nodes):
        one_d = (
            _node_difference_1d(axis_nodes, order)
            if node_based
            else _difference_1d(axis_nodes, order)
        )
        if one_d.shape[0] == 0 or one_d.nnz == 0:
            continue
        full = _lattice_operator(shape, k, one_d)
        operators.append(
            _restrict(full, mask, offset, columns, drop_rows=not node_based)
        )
    return operators


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
# penalties
# ---------------------------------------------------------------------------


class Penalty:
    """Unbound, immutable penalty configuration.

    ``penalty.bind(space)`` returns the :class:`BoundPenalty` evaluating it on
    that space.  Penalties compose with ``+`` (a :class:`Sum`) and scale with
    ``*`` by nonnegative scalars (a :class:`Scaled`).
    """

    def bind(self, space: ControlSpace) -> "BoundPenalty":  # pragma: no cover
        raise NotImplementedError

    def __add__(self, other: Any) -> "Penalty":
        if not isinstance(other, Penalty):
            return NotImplemented
        return Sum(self, other)

    __radd__ = __add__

    def __mul__(self, factor: Any) -> "Penalty":
        if not isinstance(factor, (int, float, np.integer, np.floating)):
            return NotImplemented
        return Scaled(self, float(factor))

    __rmul__ = __mul__


class BoundPenalty:
    """A penalty evaluated on one :class:`ControlSpace` (optimizer layout).

    Vectors are :class:`ControlVector` on ``space`` (or equivalent spaces) or
    ndarrays of ``space.size``; gradients come back as :class:`ControlVector`.
    """

    def __init__(self, penalty: Penalty, space: ControlSpace) -> None:
        self.penalty = penalty
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
        raise NotImplementedError

    def curvature_diagonal(self, v: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def operator(self) -> Optional[ModelOperator]:
        """Return ``R`` with ``hessian == R.T @ R`` (quadratic penalties only)."""

        return None

    def __call__(self, v: Any) -> float:
        return self.value(v)

    def __repr__(self) -> str:
        return f"Bound{self.penalty!r}[{self.space.size}]"


class _BoundQuadraticForm(BoundPenalty):
    """``0.5 * ||R (v - ref)||^2`` for a sparse ``R`` on the optimizer layout."""

    def __init__(
        self,
        penalty: Penalty,
        space: ControlSpace,
        matrix: csr_matrix,
        reference: Optional[np.ndarray],
    ) -> None:
        super().__init__(penalty, space)
        self.matrix = csr_matrix(matrix, dtype=np.float64)
        if self.matrix.shape[1] != space.size:
            raise ValueError(
                f"penalty operator has {self.matrix.shape[1]} columns; the space "
                f"has {space.size}"
            )
        self.reference = reference

    def _shift(self, v: Any) -> np.ndarray:
        values = self._values(v)
        return values if self.reference is None else values - self.reference

    def value(self, v: Any) -> float:
        residual = self.matrix @ self._shift(v)
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


@dataclass(frozen=True)
class Tikhonov(Penalty):
    """Finite-difference Tikhonov penalty ``0.5 * alpha * sum_b w_b ||D_b (v - v_ref)||^2``.

    One term per block, summed.  ``D_b`` is the order-``order`` finite
    difference scaled by the block's physical spacing (per lattice axis for
    :class:`~frequensolve.imaging.controls.GridParameters`, Greville abscissae
    for B-spline profiles).  Material blocks (profile, grid) have unit weight;
    source, interface and reflectivity blocks are unpenalized unless
    ``weights`` names them, in which case blocks without a lattice receive an
    identity (ridge) term.  Mesh blocks raise unless weighted zero.

    Args:
        alpha: Penalty strength (nonnegative).
        order: Difference order, ``1`` or ``2``.
        weights: Optional ``block -> weight`` (user key, address or qualified
            name) multiplying that block's term.
        reference: ``v_ref`` as a :class:`ControlVector`, :class:`ControlState`
            or array on the bound space; ``None`` penalizes ``v`` itself.
    """

    alpha: float
    order: int = 1
    weights: Optional[Mapping[str, Any]] = None
    reference: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _finite_scalar(self.alpha, "alpha"))
        order = int(self.order)
        if order not in (1, 2):
            raise ValueError("Tikhonov order must be 1 or 2")
        object.__setattr__(self, "order", order)
        if self.weights is not None:
            object.__setattr__(self, "weights", dict(self.weights))

    def bind(self, space: ControlSpace) -> BoundPenalty:
        weights = _block_weights(space, self.weights, _material_default)
        rows: List[csr_matrix] = []
        for block in space.resolved_blocks:
            weight = weights[block.name]
            if weight == 0.0:
                continue
            operators = _block_operators(
                space, block, order=self.order, node_based=False
            )
            if not operators:
                continue
            rows.append(
                math.sqrt(self.alpha * weight) * vstack(operators, format="csr")
            )
        matrix = (
            vstack(rows, format="csr")
            if rows
            else csr_matrix((0, space.size), dtype=np.float64)
        )
        return _BoundQuadraticForm(
            self, space, matrix, _reference_values(self.reference, space)
        )


@dataclass(frozen=True)
class Quadratic(Penalty):
    """Arbitrary quadratic penalty ``0.5 * weight * ||matrix (v - reference)||^2``.

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

    def bind(self, space: ControlSpace) -> BoundPenalty:
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
class TV(Penalty):
    """Smoothed total variation ``alpha * sum_b w_b sum_i (sqrt(|G_b v|_i^2 + eps^2) - eps)``.

    ``G_b`` collects node-based forward differences along every lattice axis
    of block ``b`` (order ``1``) or node-based second differences (``order=2``,
    a second-order TV); ``|G_b v|_i`` is the Euclidean norm over axes at node
    ``i``.  ``value`` and ``gradient`` are exact; ``hessian_operator`` is the
    lagged-diffusivity (IRLS) operator ``alpha * sum_k G_k^T diag(1 / r) G_k``
    with ``r = sqrt(|G v|^2 + eps^2)`` frozen at ``v``, which is self-adjoint
    and positive semidefinite.  Sums run over nodes without a lattice measure,
    consistent with :class:`Tikhonov`.  The ``- eps`` offset makes constant
    fields cost zero.  Frozen nodes zero every difference that touches them.

    Args:
        alpha: Penalty strength.
        epsilon: Smoothing parameter of the absolute value (same units as
            ``|G v|``).
        order: ``1`` (TV) or ``2`` (second-order TV).
        weights: Per-block weights (see :class:`Tikhonov`).
        reference: Optional reference vector (see :class:`Tikhonov`).
    """

    alpha: float
    epsilon: float = 1.0e-3
    order: int = 1
    weights: Optional[Mapping[str, Any]] = None
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
        if self.weights is not None:
            object.__setattr__(self, "weights", dict(self.weights))

    def bind(self, space: ControlSpace) -> BoundPenalty:
        weights = _block_weights(space, self.weights, _material_default)
        terms: List[Tuple[float, List[csr_matrix]]] = []
        for block in space.resolved_blocks:
            weight = weights[block.name]
            if weight == 0.0:
                continue
            operators = _block_operators(
                space, block, order=self.order, node_based=True
            )
            if operators:
                terms.append((self.alpha * weight, operators))
        return _BoundTV(
            self, space, terms, self.epsilon, _reference_values(self.reference, space)
        )


class _BoundTV(BoundPenalty):
    def __init__(
        self,
        penalty: Penalty,
        space: ControlSpace,
        terms: Sequence[Tuple[float, Sequence[csr_matrix]]],
        epsilon: float,
        reference: Optional[np.ndarray],
    ) -> None:
        super().__init__(penalty, space)
        self.terms = [(float(scale), list(ops)) for scale, ops in terms]
        self.epsilon = float(epsilon)
        self.reference = reference

    def _shift(self, v: Any) -> np.ndarray:
        values = self._values(v)
        return values if self.reference is None else values - self.reference

    def _norms(
        self, u: np.ndarray
    ) -> List[Tuple[float, List[csr_matrix], List[np.ndarray], np.ndarray]]:
        out = []
        for scale, ops in self.terms:
            components = [np.asarray(op @ u) for op in ops]
            squares = sum(c * c for c in components)
            radius = np.sqrt(squares + self.epsilon**2)
            out.append((scale, ops, components, radius))
        return out

    def value(self, v: Any) -> float:
        total = 0.0
        for scale, _ops, _components, radius in self._norms(self._shift(v)):
            total += scale * float(np.sum(radius - self.epsilon))
        return total

    def gradient(self, v: Any) -> ControlVector:
        out = np.zeros(self.space.size, dtype=np.float64)
        for scale, ops, components, radius in self._norms(self._shift(v)):
            for op, component in zip(ops, components):
                out += scale * np.asarray(op.T @ (component / radius))
        return self._wrap(out)

    def _weights(self, v: Any) -> List[Tuple[float, List[csr_matrix], np.ndarray]]:
        return [
            (scale, ops, 1.0 / radius)
            for scale, ops, _components, radius in self._norms(self._shift(v))
        ]

    def hessian_operator(self, v: Any) -> ModelOperator:
        frozen = self._weights(v)

        def action(x: np.ndarray) -> np.ndarray:
            out = np.zeros(self.space.size, dtype=np.float64)
            for scale, ops, weight in frozen:
                for op in ops:
                    out += scale * np.asarray(op.T @ (weight * np.asarray(op @ x)))
            return out

        return _SymmetricModelOperator(action, self.space)

    def curvature_diagonal(self, v: Any) -> np.ndarray:
        out = np.zeros(self.space.size, dtype=np.float64)
        for scale, ops, weight in self._weights(v):
            for op in ops:
                out += scale * np.asarray(op.multiply(op).T @ weight).reshape(-1)
        return out


@dataclass(frozen=True)
class TGV(Penalty):
    """Total generalized variation (not implemented in Python).

    Full TGV needs an auxiliary vector field solved jointly with the controls,
    which the value/gradient penalty protocol cannot express.  Use
    ``TV(order=2)`` for a second-order TV penalty, or Sauce-side
    ``Smoothing(kind="tgv", alpha1=..., alpha2=...)`` for TGV smoothing of
    gradients.
    """

    alpha1: float
    alpha2: float
    epsilon: float = 1.0e-3

    def bind(self, space: ControlSpace) -> BoundPenalty:
        raise NotImplementedError(
            "TGV penalties are not implemented in Python: use TV(order=2) for a "
            "second-order TV penalty, or Sauce-side Smoothing(kind='tgv') on "
            "gradients"
        )


class Sum(Penalty):
    """Sum of penalties (``a + b`` builds one; nested sums are flattened)."""

    def __init__(self, *penalties: Penalty) -> None:
        terms: List[Penalty] = []
        for penalty in penalties:
            if not isinstance(penalty, Penalty):
                raise TypeError(f"{type(penalty).__name__} is not a Penalty")
            if isinstance(penalty, Sum):
                terms.extend(penalty.penalties)
            else:
                terms.append(penalty)
        if not terms:
            raise ValueError("Sum requires at least one penalty")
        self.penalties: Tuple[Penalty, ...] = tuple(terms)

    def bind(self, space: ControlSpace) -> BoundPenalty:
        return _BoundSum(self, space, [p.bind(space) for p in self.penalties])

    def __repr__(self) -> str:
        return "Sum(" + ", ".join(repr(p) for p in self.penalties) + ")"


class _BoundSum(BoundPenalty):
    def __init__(
        self, penalty: Penalty, space: ControlSpace, terms: Sequence[BoundPenalty]
    ) -> None:
        super().__init__(penalty, space)
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


class Scaled(Penalty):
    """A penalty multiplied by a nonnegative scalar (``0.5 * TV(...)``)."""

    def __init__(self, penalty: Penalty, factor: float) -> None:
        if not isinstance(penalty, Penalty):
            raise TypeError(f"{type(penalty).__name__} is not a Penalty")
        self.penalty = penalty
        self.factor = _finite_scalar(factor, "penalty scale")

    def bind(self, space: ControlSpace) -> BoundPenalty:
        return _BoundScaled(self, space, self.penalty.bind(space), self.factor)

    def __repr__(self) -> str:
        return f"{self.factor:g} * {self.penalty!r}"


class _BoundScaled(BoundPenalty):
    def __init__(
        self, penalty: Penalty, space: ControlSpace, inner: BoundPenalty, factor: float
    ) -> None:
        super().__init__(penalty, space)
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
    space = problem.space
    control = (
        vector if isinstance(vector, ControlVector) else ControlVector(vector, space)
    )
    if control.space is not space and not control.space.equivalent(space):
        raise ValueError("vector does not live on the problem's control space")
    if not any(name.startswith("model.") for name in space.blocks):
        raise ValueError("smoothing requires at least one model.* block")
    source = _smoothing_source_job(problem)
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

    ``update(linearization, penalty)`` refreshes the approximation at a new
    point (a no-op for point-independent preconditioners); ``apply(g)`` (or
    ``self(g)``) returns ``M^{-1} g`` as a :class:`ControlVector`.
    """

    def __init__(self, preconditioner: Preconditioner, space: ControlSpace) -> None:
        self.preconditioner = preconditioner
        self.space = space

    def update(
        self, linearization: "Linearization", penalty: Optional[BoundPenalty] = None
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
    penalty's exact ``curvature_diagonal`` at the linearization point and
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
        self, linearization: "Linearization", penalty: Optional[BoundPenalty] = None
    ) -> None:
        """Re-estimate the diagonal at ``linearization`` (adopting its space)."""

        space = linearization.space
        if space is not self.space and not space.equivalent(self.space):
            # Support masks are adopted at the first linearize; follow them.
            self.space = space
        if penalty is not None and not (
            penalty.space is space or penalty.space.equivalent(space)
        ):
            raise ValueError("penalty is bound to a different control space")
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
        regularization = (
            np.zeros(size)
            if penalty is None
            else np.asarray(penalty.curvature_diagonal(linearization.point))
        )
        self.estimate = GaussNewtonDiagonalEstimate(
            data, np.maximum(regularization, 0.0), probe_count, config.seed
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
                "update(linearization, penalty) first"
            )
        return ControlVector(self.inverse.apply(_values_on(self.space, g)), self.space)

    @property
    def diagonal(self) -> Optional[np.ndarray]:
        """Return the damped curvature diagonal of the last update."""

        return None if self.inverse is None else self.inverse.diagonal
