# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Sparse pullback of forward-energy metrics into material-control coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from frequensolve.imaging.controls import ControlSpace, ControlVector
from frequensolve.imaging.regularization import BoundPreconditioner, Preconditioner
from frequensolve.inversion.optimization import _shares_arrays
from frequensolve.inversion.preconditioning import DiagonalInverseHessian

__all__ = ["SourceEnergy"]


def _by_active_block(
    space: ControlSpace, values: Mapping[str, Any], active: set, name: str
) -> Dict[str, Any]:
    """Resolve keys to active block names, rejecting duplicates and inactive blocks."""
    resolved: Dict[str, Any] = {}
    for key, value in values.items():
        block = space.block(key).name
        if block in resolved:
            raise ValueError(f"{name} names block {block!r} more than once")
        if block not in active:
            raise ValueError(
                f"{name} supplied for {block!r}, which has no active controls"
            )
        resolved[block] = value
    return resolved


def _grid_axes(grid: Any) -> Dict[str, np.ndarray]:
    """Return the sample coordinates of each named grid axis."""
    return {
        dim: np.linspace(lo, hi, n)
        for dim, n, lo, hi in zip(grid.dims, grid.n, grid.x0, grid.x1)
    }


def _samples(value: Any, grid: Any, name: str) -> np.ndarray:
    """Validate a real scalar or a field in Cartesian storage order.

    Labelled arrays (``xarray.DataArray``) are aligned by dimension name and must
    carry the grid's sample coordinates on any coordinate they declare.
    """
    shape = tuple(grid.shape)
    if hasattr(value, "dims") and hasattr(value, "transpose"):
        order = tuple(reversed(grid.dims))
        if set(value.dims) != set(order) or len(value.dims) != len(order):
            raise ValueError(f"{name} dims {tuple(value.dims)} must be {order}")
        value = value.transpose(*order)
        for dim, expected in _grid_axes(grid).items():
            if dim not in value.coords:
                continue
            actual = np.asarray(value.coords[dim], dtype=float)
            tolerance = 1.0e-9 * max(abs(expected[-1] - expected[0]), 1.0)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=0.0, atol=tolerance
            ):
                raise ValueError(f"{name} coordinate {dim!r} does not match the grid")
        value = value.values
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.shape not in ((), shape):
        raise ValueError(f"{name} must be real and scalar or have shape {shape}")
    array = np.broadcast_to(np.asarray(array, dtype=float), shape)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class SourceEnergy(Preconditioner):
    r"""Diagonal forward-energy metric pulled back through the control basis.

    For each block, assemble the lumped diagonal ``B.T @ (w * E * t**2)``:
    ``B`` samples the control basis, ``w`` is physical quadrature, ``E`` is
    the supplied forward-energy density, and ``t = dm/d(Bc)`` is the material
    transform derivative. Where the basis is a partition of unity this is the
    row sum of the pseudo-Hessian ``B.T @ diag(w * E * t**2) @ B``, so a
    covector ``B.T @ (w * k)`` of a smooth kernel maps to the physical update
    ``k / (E * t**2)``; the consistent diagonal ``diag(B.T @ ... @ B)`` would
    overstate it by 1.5 per dimension for multilinear hats. This is a
    pseudo-Hessian, not a GN diagonal. No normal solves, dense Jacobians,
    amplitude normalization or RMS gain are used.

    ``energy`` maps control keys to nonnegative arrays in ``grid.shape`` order
    (z, [y,] x). Sum frequency/RHS contributions with the same weights as the
    covector *before* constructing this object. All active blocks must be
    material profile/tensor-hat controls with an entry; omissions never silently
    receive an identity preconditioner. Zero-energy controls get zero updates.

    ``transform_derivative`` is optional for identity controls (defaults to 1),
    but required for other transforms: physical ``m`` for log, ``-m**2`` for
    inverse, and ``m*(1-m)`` for logit. Evaluate these at the current model, in
    the physical-property convention of ``energy``. Transforms must be refreshed
    when the model changes.

    Default quadrature is tensor trapezoidal integration in the grid's numeric
    length units. Supply ``quadrature_weights`` for another sampled measure,
    such as an axisymmetric metric. Masks follow the bound material
    domain, and frozen columns are excluded without discarding their neighbors.

    Energy/objective scaling is deliberately not guessed: ``abs(p)**2`` alone does
    not have the units of every objective's Vp Hessian. Physical update units
    require ``E*w`` to have objective/property-squared units. In particular, WRI and
    normalized waveform objectives need their own scaling.
    """

    energy: Mapping[str, Any]
    grid: Any
    transform_derivative: Optional[Mapping[str, Any]] = None
    quadrature_weights: Any = None
    relative_damping: float = 1.0e-2
    maximum_inverse_ratio: Optional[float] = 1.0e3
    context: Any = None

    def bind(self, space: ControlSpace) -> BoundPreconditioner:
        return _BoundSourceEnergy(self, space)


class _BoundSourceEnergy(BoundPreconditioner):
    def __init__(self, config: SourceEnergy, space: ControlSpace) -> None:
        super().__init__(config, space)
        self.config = config
        self._assemble()

    def _assemble(self) -> None:
        """Integrate sparse basis entries against the energy density (lumped metric)."""
        config, space = self.config, self.space
        from frequensolve.geometry.grids import CartesianGrid

        if not isinstance(config.grid, CartesianGrid):
            raise TypeError("SourceEnergy requires a CartesianGrid")
        if space.size == 0:
            raise ValueError("SourceEnergy needs at least one active control")
        shape = tuple(config.grid.shape)
        if config.quadrature_weights is None:
            weights = np.ones(shape)
            for axis, (n, lo, hi) in enumerate(
                zip(config.grid.n, config.grid.x0, config.grid.x1)
            ):
                if n < 2 or hi <= lo:
                    raise ValueError("energy quadrature needs increasing grid axes")
                rule = np.full(n, (hi - lo) / (n - 1))
                rule[[0, -1]] *= 0.5
                broadcast = [1] * len(shape)
                broadcast[len(shape) - 1 - axis] = n
                weights *= rule.reshape(broadcast)
        else:
            weights = _samples(
                config.quadrature_weights, config.grid, "quadrature_weights"
            )
        if np.any(weights < 0):
            raise ValueError("quadrature_weights must be nonnegative")
        active = {
            block.name
            for block in space.resolved_blocks
            if space.slices[block.name].start != space.slices[block.name].stop
        }
        energy = _by_active_block(space, config.energy, active, "energy")
        factors = _by_active_block(
            space, config.transform_derivative or {}, active, "transform_derivative"
        )
        raw = np.zeros(space.size)
        vector = space.zeros()
        for block in space.resolved_blocks:
            sl = space.slices[block.name]
            if sl.start == sl.stop:
                continue
            if not block.name.startswith("model.") or block.complex:
                raise ValueError("SourceEnergy supports real material controls only")
            if block.name not in energy:
                raise ValueError(f"missing source energy for {block.name!r}")
            field = _samples(energy[block.name], config.grid, "energy")
            if np.any(field < 0):
                raise ValueError("source energy must be nonnegative")
            if block.transform != "identity" and block.name not in factors:
                raise ValueError(
                    f"{block.name!r} needs its physical transform derivative"
                )
            factor = _samples(
                factors.get(block.name, 1.0), config.grid, "transform derivative"
            )
            _, basis, valid = vector._grid_sampling(
                config.grid, block.name, context=config.context
            )
            basis = basis[:, space._mask_of(block)]
            density = (weights * field * factor**2 * valid).ravel()
            raw[sl] = np.asarray(basis.T @ density).ravel()
        self.raw_diagonal = raw
        # Coverage is a property of the supplied energy alone: regularization or
        # damping curvature never turns an unilluminated coefficient into an update.
        self.illuminated = raw > 0
        self._transformed = any(
            block.transform != "identity"
            for block in space.resolved_blocks
            if block.name in active
        )
        self._point: Optional[np.ndarray] = None
        self._set_inverse(raw)

    def _set_inverse(self, diagonal: np.ndarray) -> None:
        """Damp the metric; unilluminated entries stay frozen in ``apply``."""
        sizes = [sl.stop - sl.start for sl in self.space.slices.values()]
        self.inverse = DiagonalInverseHessian(
            diagonal,
            block_sizes=[size for size in sizes if size],
            relative_damping=self.config.relative_damping,
            maximum_inverse_ratio=self.config.maximum_inverse_ratio,
        )

    def update(self, linearization: Any, regularization: Any = None) -> None:
        """Check the space and optionally add regularization curvature.

        Energy is fixed supplied data; this does not refresh forward fields.
        Construct a new SourceEnergy when the baseline or energy changes. With
        non-identity transforms, ``transform_derivative`` is only valid at one
        model, so a refresh at a different point is rejected.
        """
        if not self.space.equivalent(linearization.space):
            raise ValueError("source energy is bound to a different control space")
        point = np.asarray(linearization.point.values, dtype=float)
        if self._transformed:
            if self._point is None:
                self._point = point.copy()
            elif not np.array_equal(point, self._point):
                raise ValueError(
                    "SourceEnergy transform derivatives were evaluated at a different "
                    "model; rebuild SourceEnergy at the current model"
                )
        diagonal = self.raw_diagonal.copy()
        if regularization is not None:
            if not self.space.equivalent(regularization.space):
                raise ValueError("regularization is bound to a different control space")
            diagonal += np.asarray(
                regularization.curvature_diagonal(linearization.point)
            )
        self._set_inverse(diagonal)

    @property
    def diagonal(self) -> np.ndarray:
        """Damped diagonal; raw coverage remains in ``raw_diagonal``."""
        return self.inverse.diagonal

    @_shares_arrays  # Returns a new vector.
    def apply(self, g: Any) -> ControlVector:
        """Apply the inverse metric; leave the input covector unchanged."""
        from frequensolve.imaging.regularization import _values_on

        values = self.inverse.apply(_values_on(self.space, g))
        return ControlVector(np.where(self.illuminated, values, 0.0), self.space)
