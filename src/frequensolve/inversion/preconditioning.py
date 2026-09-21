"""Reduced-space inverse-Hessian preconditioners for waveform inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from frequensolve.inversion.least_squares import ControlLeastSquaresProblem

__all__ = [
    "DiagonalInverseHessian",
    "GaussNewtonDiagonalEstimate",
    "estimate_gauss_newton_diagonal",
]


def _finite_real_vector(
    value: Any, *, name: str, size: Optional[int] = None
) -> np.ndarray:
    """Return one finite, real float64 vector."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size < 1 or (size is not None and array.size != size):
        expected = "non-empty" if size is None else f"size {size}"
        raise ValueError(f"{name} must be a {expected} vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=True)


@dataclass(frozen=True)
class GaussNewtonDiagonalEstimate:
    """Data and regularization contributions to a reduced Gauss--Newton diagonal."""

    data: np.ndarray
    regularization: np.ndarray
    probe_count: int
    seed: Optional[int]

    def __post_init__(self) -> None:
        data = _finite_real_vector(self.data, name="data diagonal")
        regularization = _finite_real_vector(
            self.regularization,
            name="regularization diagonal",
            size=data.size,
        )
        if np.any(data < 0.0) or np.any(regularization < 0.0):
            raise ValueError("Gauss-Newton diagonal contributions must be nonnegative")
        probe_count = int(self.probe_count)
        if probe_count < 1:
            raise ValueError("probe_count must be positive")
        seed = None if self.seed is None else int(self.seed)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "regularization", regularization)
        object.__setattr__(self, "probe_count", probe_count)
        object.__setattr__(self, "seed", seed)

    @property
    def total(self) -> np.ndarray:
        """Return the undamped data-plus-regularization diagonal."""

        return self.data + self.regularization


def estimate_gauss_newton_diagonal(
    problem: ControlLeastSquaresProblem,
    model: Union[Sequence[float], np.ndarray],
    *,
    probe_count: int = 4,
    seed: Optional[int] = 0,
) -> GaussNewtonDiagonalEstimate:
    r"""Estimate ``diag(J.T W**2 J + R.T R)`` with randomized VJP probes.

    A Rademacher vector :math:`z` in the isometric real data layout satisfies

    .. math::

       E[(J^T W z) \odot (J^T W z)] = \operatorname{diag}(J^T W^2 J).

    Consequently each probe needs one VJP and no Born solve, normal product,
    or explicit Jacobian. The estimate is nonnegative for every finite sample
    and is independent of the current data residual. Quadratic-regularization
    curvature is added exactly.
    """

    if not isinstance(problem, ControlLeastSquaresProblem):
        raise TypeError("Gauss-Newton diagonal estimation requires a control problem")
    probes = int(probe_count)
    if probes < 1:
        raise ValueError("probe_count must be positive")
    vector = problem.control_space.pack(model)
    rng = np.random.default_rng(seed)
    data_diagonal = np.zeros(problem.control_space.size, dtype=np.float64)
    for _ in range(probes):
        real_probe = rng.choice((-1.0, 1.0), size=problem.data_size)
        weighted_dual = problem.realifier.unpack(real_probe) * problem.data_weights
        transpose_probe = problem.data_vjp(vector, weighted_dual)
        data_diagonal += transpose_probe * transpose_probe
    data_diagonal /= probes

    regularization_diagonal = np.zeros_like(data_diagonal)
    if problem.regularization is not None:
        jacobian = problem.regularization.jacobian()
        regularization_diagonal = np.einsum(
            "ij,ij->j", jacobian, jacobian, optimize=True
        )
    return GaussNewtonDiagonalEstimate(
        data_diagonal,
        regularization_diagonal,
        probes,
        seed,
    )


class DiagonalInverseHessian:
    """Damped positive inverse-Hessian action for an optimizer's initial metric.

    Damping and inverse-dynamic-range clipping are applied independently to
    each supplied control block. A completely unilluminated block receives an
    identity metric, avoiding division by zero without inventing a large step.
    """

    def __init__(
        self,
        diagonal: Union[Sequence[float], np.ndarray],
        *,
        block_sizes: Optional[Sequence[int]] = None,
        relative_damping: float = 1.0e-2,
        maximum_inverse_ratio: Optional[float] = 1.0e3,
    ):
        raw = _finite_real_vector(diagonal, name="inverse-Hessian diagonal")
        if np.any(raw < 0.0):
            raise ValueError("inverse-Hessian diagonal must be nonnegative")
        damping = float(relative_damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("relative_damping must be finite and nonnegative")
        if maximum_inverse_ratio is None:
            ratio = None
        else:
            ratio = float(maximum_inverse_ratio)
            if not np.isfinite(ratio) or ratio < 1.0:
                raise ValueError(
                    "maximum_inverse_ratio must be finite and at least one"
                )
        if block_sizes is None:
            sizes: tuple[int, ...] = (raw.size,)
        else:
            sizes = tuple(int(size) for size in block_sizes)
            if not sizes or any(size < 1 for size in sizes):
                raise ValueError("block_sizes must contain positive integers")
            if sum(sizes) != raw.size:
                raise ValueError("block_sizes must span the diagonal")

        damped = np.empty_like(raw)
        offset = 0
        for size in sizes:
            block_slice = slice(offset, offset + size)
            block = raw[block_slice]
            scale = float(np.max(block))
            if scale <= np.finfo(np.float64).tiny:
                damped[block_slice] = 1.0
            else:
                local = block + damping * scale
                local = np.maximum(
                    local,
                    np.finfo(np.float64).eps * scale,
                )
                if ratio is not None:
                    local = np.maximum(local, float(np.max(local)) / ratio)
                damped[block_slice] = local
            offset += size

        self._raw_diagonal = raw
        self._diagonal = damped
        self._inverse_diagonal = 1.0 / damped
        self.block_sizes = sizes
        self.relative_damping = damping
        self.maximum_inverse_ratio = ratio

    @property
    def size(self) -> int:
        """Return the control-vector size."""

        return int(self._diagonal.size)

    @property
    def raw_diagonal(self) -> np.ndarray:
        """Return the undamped curvature estimate."""

        return np.array(self._raw_diagonal, copy=True)

    @property
    def diagonal(self) -> np.ndarray:
        """Return the positive damped curvature diagonal."""

        return np.array(self._diagonal, copy=True)

    @property
    def inverse_diagonal(self) -> np.ndarray:
        """Return the initial inverse-Hessian diagonal."""

        return np.array(self._inverse_diagonal, copy=True)

    def apply(self, vector: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Apply the damped inverse metric to one control covector."""

        values = _finite_real_vector(
            vector,
            name="inverse-Hessian input",
            size=self.size,
        )
        return self._inverse_diagonal * values

    def __call__(
        self,
        model: Union[Sequence[float], np.ndarray],
        vector: Union[Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Apply the optimizer preconditioner; the frozen baseline is implicit."""

        _finite_real_vector(model, name="preconditioner model", size=self.size)
        return self.apply(vector)
