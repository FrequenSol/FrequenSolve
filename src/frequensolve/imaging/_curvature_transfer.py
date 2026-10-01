# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Stage curvature policies and immutable local inverse actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .curvature import (
    BFGSHistory,
    CurvatureResult,
    _finite,
    _frozen,
    _integer,
    _real_input,
    _relative_tolerance,
)

__all__ = ["CurvatureTransfer"]


@dataclass(frozen=True)
class CurvatureTransfer:
    """Choose how curvature moves between frequency bands and control meshes.

    ``reset()`` starts each stage from its declared diagonal. ``warm_start()``
    carries the reduced inverse correction relative to the source diagonal
    (whitened) onto the new stage's diagonal as an optimizer metric; it is
    not a posterior for the new objective. ``refresh()`` retains a
    reduced direction basis and reevaluates its curvature with the new
    stage's Gauss-Newton normal plus regularization curvature. Sauce performs
    mesh transfer, basis reduction and the reduced inverse construction. ``rank`` limits
    retained directions. For ``refresh()``, ``basis_tolerance`` rejects
    dependent directions before target-stage Hessian evaluation, and
    ``symmetry_tolerance`` (``None``: Sauce's ``1e-2``) bounds the relative
    asymmetry of the measured reduced Hessian, which Sauce symmetrizes; the
    stage's ``curvature_transfer`` metrics record the measured
    ``hessian_asymmetry``.
    """

    method: str = "reset"
    rank: int = 20
    basis_tolerance: float = 1e-4
    symmetry_tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        if self.method not in {"reset", "warm_start", "refresh"}:
            raise ValueError(
                "Curvature transfer method must be reset, warm_start or refresh"
            )
        _integer(self.rank, "curvature transfer rank", minimum=1)
        minimum = np.sqrt(np.finfo(np.float64).eps)
        if (
            not np.isfinite(self.basis_tolerance)
            or not minimum <= self.basis_tolerance < 1
        ):
            raise ValueError(
                "basis_tolerance must be at least sqrt(float64 eps) and below one"
            )
        if self.symmetry_tolerance is not None:
            if self.method != "refresh":
                raise ValueError("symmetry_tolerance applies only to refresh()")
            object.__setattr__(
                self,
                "symmetry_tolerance",
                _relative_tolerance(self.symmetry_tolerance, "symmetry_tolerance"),
            )

    @classmethod
    def reset(cls) -> CurvatureTransfer:
        """Start every stage from its declared inverse diagonal."""
        return cls("reset")

    @classmethod
    def warm_start(cls, rank: int = 20) -> CurvatureTransfer:
        """Carry a reduced inverse correction as a new stage's optimizer metric."""
        return cls("warm_start", rank)

    @classmethod
    def refresh(
        cls,
        rank: int = 20,
        *,
        basis_tolerance: float = 1e-4,
        symmetry_tolerance: Optional[float] = None,
    ) -> CurvatureTransfer:
        """Reevaluate directions against the new normal and regularization curvature."""
        return cls("refresh", rank, basis_tolerance, symmetry_tolerance)


def _seed_factors(
    factors: CurvatureResult, *, whitened: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read verified factors once and convert them in place, without copies.

    Returns the physical inverse diagonal ``B``, the modes and eigenvalues.
    Modes are physical (``prior * V``), or with ``whitened=True`` relative to
    the stored diagonal (``V / sqrt(B0)``, i.e. ``B**-0.5`` times physical
    modes), the coordinates in which the inverse is ``I + W diag(e) W^T``.
    """
    rank = _integer(factors.metadata.get("rank", 0), "seed rank")
    arrays = factors.read_verified(
        "base_inverse_diagonal",
        "prior_std",
        *(("modes", "eigenvalues") if rank else ()),
    )
    # The verified HDF5 reads are owned here; validation and the coordinate
    # change work in place so no second controls-by-rank block is allocated.
    base = _real_input(arrays["base_inverse_diagonal"], "seed base")
    prior = _real_input(arrays["prior_std"], "seed prior scale")
    if base.ndim != 1 or not base.size or prior.shape != base.shape:
        raise ValueError("Seed diagonal and prior have inconsistent coordinates")
    if base.min() <= 0 or prior.min() <= 0:
        raise ValueError("Seed diagonal and prior must be positive")
    modes = (
        _real_input(arrays["modes"], "seed modes") if rank else np.empty((0, base.size))
    )
    eigenvalues = (
        _real_input(arrays["eigenvalues"], "seed eigenvalues") if rank else np.empty(0)
    )
    if modes.shape != (rank, base.size) or eigenvalues.shape != (rank,):
        raise ValueError("Seed modes and eigenvalues have inconsistent coordinates")
    with np.errstate(over="ignore"):
        modes *= (1 / np.sqrt(base) if whitened else prior)[None, :]
        base *= prior**2
    if not _finite(base) or base.min() <= 0 or not _finite(modes):
        raise ValueError("Physical seed factors overflow their coordinates")
    return base, modes, eigenvalues


class _CurvatureSeed:
    """Pinned diagonal-plus-low-rank inverse in physical control coordinates.

    Factors are read once and each array is checked in memory against the
    digest Sauce recorded when it wrote it. A prior-whitened source factor is
    converted to physical coordinates by its stored scale; this is a change
    of coordinates, not a curvature eigensolve. A stage moves the modes into
    its history with :meth:`take`; the optimizer then applies the
    history's copy (:func:`_history_inverse`), so one block stays resident.
    """

    def __init__(
        self, factors: CurvatureResult, *, provenance: Optional[dict] = None
    ) -> None:
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("Seed provenance must be a JSON mapping")
        base, modes, eigenvalues = _seed_factors(factors)
        identity = factors.identity
        recorded = dict(provenance or {})
        recorded.setdefault("factors_identity", identity)
        recorded.setdefault("source_state", factors.metadata.get("state"))
        recorded.setdefault("source_coordinates", factors.metadata.get("coordinates"))
        for key in ("symmetry_tolerance", "hessian_asymmetry"):
            if key in factors.metadata:  # refreshed factors
                recorded.setdefault(key, factors.metadata[key])
        self._provenance = json.dumps(recorded, sort_keys=True, allow_nan=False)
        self.metadata = json.loads(json.dumps(factors.metadata, allow_nan=False))
        self.factors: Optional[CurvatureResult] = factors
        self.digest = identity
        self.physical_base = _frozen(base)
        self.modes = _frozen(modes)
        self.eigenvalues = _frozen(eigenvalues)

    @property
    def rank(self) -> int:
        return len(self.eigenvalues)

    @property
    def provenance(self) -> dict:
        return json.loads(self._provenance)

    def take(self, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return modes in optimizer coordinates ``x = scale * y`` and eigenvalues.

        The modes are rescaled in place and the seed keeps rank zero
        afterwards, so a history adopts the block while at most one other
        copy exists.
        """
        modes, eigenvalues = self.modes, self.eigenvalues
        if scale.shape != self.physical_base.shape or scale.min() <= 0:
            raise ValueError("Seed coordinate scale must be a positive matching vector")
        self.modes = _frozen(np.empty((0, modes.shape[1])))
        self.eigenvalues = _frozen(np.empty(0))
        if not modes.flags.owndata:
            return _frozen(modes / scale[None, :]), eigenvalues
        modes.flags.writeable = True
        modes /= scale[None, :]
        # Returned read-only: a history adopts the block instead of copying it.
        return _frozen(modes), eigenvalues

    def apply(self, directions: Any) -> np.ndarray:
        """Apply the pinned physical inverse to directions with controls last."""
        values = np.asarray(directions)
        if np.iscomplexobj(values):
            raise ValueError("Seed directions must be real")
        values = np.asarray(values, dtype=float)
        if values.ndim not in (1, 2) or values.shape[-1] != len(self.physical_base):
            raise ValueError("Seed directions have inconsistent coordinates")
        if not np.isfinite(values).all():
            raise ValueError("Seed directions must be finite")
        result = values * self.physical_base
        if self.rank:
            result += ((values @ self.modes.T) * self.eigenvalues) @ self.modes
        return result


def _history_inverse(
    history: BFGSHistory, scale: np.ndarray
) -> Callable[[Any, np.ndarray], np.ndarray]:
    """Physical action ``S (B + V diag(e) V^T) S`` of a history's frozen initial inverse.

    ``B``, ``V`` and ``e`` are the history's own read-only arrays in optimizer
    coordinates ``x = S y``; the closure shares them instead of keeping a
    physical copy of the seed modes for the whole stage.
    """
    base = history.base_inverse_diagonal
    modes, eigenvalues = history.seed_modes, history.seed_eigenvalues
    if scale.shape != base.shape or scale.min() <= 0:
        raise ValueError("History coordinate scale must be a positive matching vector")

    def apply(model: Any, gradient: np.ndarray) -> np.ndarray:
        scaled = scale * gradient  # Controls last: one vector or a batch of rows.
        result = base * scaled
        if len(eigenvalues):
            result += ((scaled @ modes.T) * eigenvalues) @ modes
        result *= scale
        return result

    return apply
