# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Orchestrate physical curvature directions across immutable stage spaces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

import numpy as np

from ._backend import fingerprint
from ._curvature_transfer import CurvatureTransfer, _CurvatureSeed, _seed_factors
from .controls import ControlVector, _same_basis
from .statistics import UncertaintyResult, _linear_materials, mesh_descriptor


@dataclass(frozen=True)
class _CurvatureSource:
    index: int
    result: UncertaintyResult


def _regularization_identity(bound: Any) -> Any:
    """Identify the complete penalty before replaying target-stage secants."""
    from ._native_regularization import BoundNativeRegularization
    from .regularization import _BoundQuadraticForm, _BoundScaled, _BoundSum

    if bound is None:
        return None
    identity = getattr(bound, "identity", None)
    if isinstance(identity, str) and identity:
        return identity
    if isinstance(bound, BoundNativeRegularization):
        # Native Tikhonov: the checkpoint record pins the configuration, factor,
        # reference and the stage context Sauce froze (basis identities, weights
        # and amplitudes), exactly what a mid-stage restore reinstates.
        return fingerprint(native=bound.checkpoint())
    if isinstance(bound, _BoundSum):
        return fingerprint(
            terms=[_regularization_identity(term) for term in bound.terms]
        )
    if isinstance(bound, _BoundScaled):
        return fingerprint(
            factor=bound.factor, inner=_regularization_identity(bound.inner)
        )
    if isinstance(bound, _BoundQuadraticForm):
        digest = hashlib.sha256(np.asarray(bound.matrix.shape, np.int64).tobytes())
        for values in (
            bound.matrix.indptr,
            bound.matrix.indices,
            bound.matrix.data,
            bound.reference,
        ):
            if values is not None:
                array = np.ascontiguousarray(values)
                digest.update(array.dtype.str.encode())
                digest.update(array.data)
        return "sha256:" + digest.hexdigest()
    raise ValueError(
        "Curvature transfer requires regularization with a stable identity"
    )


def _configuration_value(value: Any) -> Any:
    """Hash declared arrays without materializing large JSON lists or binding a PDE."""
    from scipy.sparse import issparse

    from .controls import ControlState

    if isinstance(value, (ControlState, ControlVector)):
        return dict(
            values=_configuration_value(value.values),
            blocks=[
                (k, value.space.block(k).basis_identity) for k in value.space.blocks
            ],
        )
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return dict(value=_configuration_value(value.magnitude), units=str(value.units))
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            return _configuration_value(array.tolist())
        digest = hashlib.sha256(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.data)
        return dict(array_sha256=digest.hexdigest())
    if issparse(value):
        matrix = value.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        return dict(
            shape=matrix.shape,
            indptr=_configuration_value(matrix.indptr),
            indices=_configuration_value(matrix.indices),
            data=_configuration_value(matrix.data),
        )
    if isinstance(value, Mapping):
        return {str(key): _configuration_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_configuration_value(item) for item in value]
    if isinstance(value, np.generic):
        return _configuration_value(value.item())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return dict(
            type=f"{type(value).__module__}.{type(value).__qualname__}",
            fields={
                field.name: _configuration_value(getattr(value, field.name))
                for field in fields(value)
            },
        )
    if hasattr(value, "dims") and hasattr(value, "coords") and hasattr(value, "values"):
        return dict(
            values=_configuration_value(np.asarray(value.values)),
            dims=list(value.dims),
            coords={
                str(k): _configuration_value(np.asarray(v))
                for k, v in value.coords.items()
            },
            attrs=_configuration_value(dict(value.attrs)),
        )
    identity = getattr(value, "identity", None)
    if isinstance(identity, str) and identity:
        return dict(identity=identity)
    to_fs = getattr(value, "to_fs", None)
    if callable(to_fs):
        return _configuration_value(to_fs())
    raise ValueError("Curvature checkpoint configuration needs a stable identity")


def _checkpoint_regularization_identity(specification: Any, space: Any) -> Any:
    """Identify authored penalties even when a completed stage will be skipped."""
    from ._artifacts import SmoothingConfig
    from ._native_regularization import NativeRegularization
    from .regularization import TGV, TV, Quadratic, Scaled, Sum, Tikhonov
    from .statistics import GaussianPrior

    if specification is None or specification is False:
        return None
    if isinstance(specification, Sum):
        return fingerprint(
            terms=[
                _checkpoint_regularization_identity(term, space)
                for term in specification.regularizations
            ]
        )
    if isinstance(specification, Scaled):
        return fingerprint(
            factor=specification.factor,
            inner=_checkpoint_regularization_identity(
                specification.regularization, space
            ),
        )
    if isinstance(
        specification,
        (
            GaussianPrior,
            Quadratic,
            SmoothingConfig,
            NativeRegularization,
            Tikhonov,
            TV,
            TGV,
            Mapping,
        ),
    ):
        # Native penalties are declarations here: preparing a native context
        # merely to admit a checkpoint would require needless execution.
        return fingerprint(configuration=_configuration_value(specification))
    identity = getattr(specification, "identity", None)
    if isinstance(identity, str) and identity:
        return identity
    return _regularization_identity(specification.bind(space))


def _checkpoint_operator_configuration(scaling: Any, preconditioner: Any) -> str:
    """Pin authored scaling and fixed-diagonal preconditioner options."""
    return fingerprint(
        scaling=_configuration_value(scaling),
        preconditioner=_configuration_value(preconditioner),
    )


_NONSMOOTH = (
    "Curvature transfer requires a smooth objective without proximal regularization"
)


def _require_smooth_regularization(specification: Any) -> None:
    """Reject proximal penalties a fixed-metric stage history cannot follow.

    Runs on declarations before any solve. TV and TGV (also as native
    configurations) are nonsmooth; native Tikhonov, Gaussian priors and
    quadratic forms are identified when bound. Custom terms need ``identity``
    on the declaration or on the bound term; the latter is checked at binding.
    """
    from ._artifacts import SmoothingConfig, control_smoothing
    from ._native_regularization import NativeRegularization
    from .regularization import TGV, TV, Scaled, Sum

    if specification is None or specification is False:
        return
    if isinstance(specification, Sum):
        for term in specification.regularizations:
            _require_smooth_regularization(term)
    elif isinstance(specification, Scaled):
        if specification.factor != 0:
            _require_smooth_regularization(specification.regularization)
    elif isinstance(specification, TGV) or (
        isinstance(specification, TV) and specification.alpha != 0
    ):
        raise ValueError(_NONSMOOTH)
    elif isinstance(specification, (NativeRegularization, SmoothingConfig, Mapping)):
        config = (
            specification.smoothing
            if isinstance(specification, NativeRegularization)
            else control_smoothing(specification)
        )
        if config is not None and config.kind in {"tv", "tgv"}:
            raise ValueError(_NONSMOOTH)


def _initial_scale(
    objective: Any,
    point: np.ndarray,
    base: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> float:
    """Scale a frozen inverse diagonal ``B`` by ``g'Bg / (Bg)'H(Bg)`` at the stage start.

    A fixed-metric stage history cannot follow L-BFGS's dynamic gamma, so one
    Hessian action (data normal plus regularization curvature) on the
    bound-projected, ``B``-preconditioned gradient sets the scalar once; the
    first step is then exact on the local quadratic model. Returns one when
    that gradient vanishes or its measured curvature is not positive.
    """
    measured = _scale_direction(objective, point, base, bounds)
    if measured is None:
        return 1.0
    direction, slope = measured
    return _measured_scale(slope, direction, objective.hessian_action(point, direction))


def _scale_direction(
    objective: Any,
    point: np.ndarray,
    base: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float] | None:
    """Return the bound-projected ``Bg`` and ``g'Bg``, or ``None`` when ``g'Bg <= 0``."""
    gradient = objective.gradient(point)
    lower, upper = bounds
    gradient[
        ((point <= lower) & (gradient > 0)) | ((point >= upper) & (gradient < 0))
    ] = 0
    direction = base * gradient
    slope = float(gradient @ direction)
    if not np.isfinite(slope) or slope <= 0:
        return None
    return direction, slope


def _measured_scale(slope: float, direction: np.ndarray, image: np.ndarray) -> float:
    """Return ``g'Bg / (Bg)'H(Bg)``, or one for a nonpositive or nonfinite curvature."""
    curvature = float(direction @ image)
    scale = slope / curvature if np.isfinite(curvature) and curvature > 0 else 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _hessian_images(
    objective: Any, point: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Return Hessian actions on every row, from one normal job when batched."""
    batched = getattr(objective, "hessian_actions", None)
    if batched is not None:
        return np.asarray(batched(point, directions), dtype=np.float64)
    images = np.empty_like(directions)
    for i, direction in enumerate(directions):
        images[i] = objective.hessian_action(point, direction)
    return images


def _move(
    source: np.ndarray,
    source_mask: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
) -> None:
    """Copy rows of one block between support layouts without block temporaries.

    ``source`` and ``target`` hold the active columns of a full block with
    supports ``source_mask``/``target_mask``; target entries outside the
    shared support keep their values (zeros).
    """
    if np.array_equal(source_mask, target_mask):
        target[...] = source
    elif target_mask.all():
        for into, row in zip(target, source):
            into[source_mask] = row
    elif source_mask.all():
        for into, row in zip(target, source):
            into[...] = row[target_mask]
    else:
        positions = np.flatnonzero(target_mask)
        shared = source_mask[positions]
        index = (np.cumsum(source_mask) - 1)[positions[shared]]
        for into, row in zip(target, source):
            into[shared] = row[index]


def _directions(
    source: _CurvatureSource,
    target: Any,
    native: Any,
    rank: int,
    *,
    whitened: bool = False,
) -> tuple:
    """Lift source modes block by block; no gradient covectors are interpolated.

    Modes are physical, or with ``whitened`` relative to the source diagonal
    (``B**-0.5`` times physical): the correction ``I + W diag(e) W^T`` that a
    warm start re-colors with the target diagonal. A whitened block whose basis
    changes keeps its source norm, so interpolation onto more coefficients
    does not inflate the correction. Support changes and new or dropped
    blocks restrict it. At most two controls-by-rank blocks are resident: the
    source modes are released before the last block is lifted.
    """
    old = source.result.space
    _, modes, eigenvalues = _seed_factors(source.result.factors, whitened=whitened)
    count = min(rank, len(eigenvalues))
    modes = modes[:count]
    result = np.zeros((count, target.size))
    _linear_materials(old)
    _linear_materials(target)
    # Newly active properties retain the fresh target baseline.
    shared = [name for name in target.blocks if name in old.blocks]
    for name in shared:
        before, after = old.block(name), target.block(name)
        if (before.transform, before.prop, before.subdomain, before.units) != (
            after.transform,
            after.prop,
            after.subdomain,
            after.units,
        ):
            raise ValueError(
                f"Curvature transfer changes the physical meaning of {name}"
            )
        source_mask, target_mask = old._mask_of(before), target._mask_of(after)
        block = modes[:, old.slices[name]]
        into = result[:, target.slices[name]]
        if _same_basis(before, after) and before.size == after.size:
            _move(block, source_mask, into, target_mask)
            continue
        values = np.zeros((count, before.size))
        _move(block, source_mask, values, np.ones(before.size, dtype=bool))
        if name == shared[-1]:
            del block, modes  # Last use: free the source block before lifting.
        norm = float(np.linalg.norm(values)) if whitened else 0.0
        if before.kind == after.kind == "mesh":
            output = native.mesh_directions(
                source.result.meshes[name], mesh_descriptor(target, after), values
            )
            del values
            lifted = output.read("directions")
        else:
            # Profiles and lattices use the existing field representation map.
            old_block = old.without_support().restrict(name)
            new_block = target.without_support().restrict(name)
            lifted = np.empty((count, after.size))
            for row, value in zip(lifted, values):
                row[:] = old_block.transfer_to(
                    new_block, ControlVector(value, old_block)
                ).values
            del values
        if lifted.shape != (count, after.size):
            raise ValueError(
                "Transferred curvature directions have inconsistent coordinates"
            )
        if whitened and norm > 0:
            lifted_norm = float(np.linalg.norm(lifted))
            if lifted_norm > 0:
                lifted *= norm / lifted_norm
        _move(lifted, np.ones(after.size, dtype=bool), into, target_mask)
    return result, eigenvalues[:count]


def _prepare_seed(
    policy: CurvatureTransfer,
    source: _CurvatureSource,
    space: Any,
    native: Any,
    base: np.ndarray,
    objective: Any,
    point: np.ndarray,
    *,
    state: str,
    coordinates: str,
    index: int,
    scale_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> _CurvatureSeed:
    """Schedule backend transfer/reduction and target-objective Hessian actions.

    With ``scale_bounds`` a refresh also measures the initial scale ``gamma``
    of :func:`_initial_scale` in its normal job and seeds ``gamma * base``;
    ``provenance["initial_scale"]`` records it.
    """
    factors, provenance = _transfer_factors(
        policy,
        source,
        space,
        native,
        base,
        objective,
        point,
        state=state,
        coordinates=coordinates,
        index=index,
        scale_bounds=scale_bounds,
    )
    # The staged direction and image blocks were local to the transfer and are
    # released before the seed reads its own copy of the factors.
    return _CurvatureSeed(factors, provenance=provenance)


def _transfer_factors(
    policy: CurvatureTransfer,
    source: _CurvatureSource,
    space: Any,
    native: Any,
    base: np.ndarray,
    objective: Any,
    point: np.ndarray,
    *,
    state: str,
    coordinates: str,
    index: int,
    scale_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Any, dict]:
    """Return the target-stage factors and their provenance."""
    if scale_bounds is not None and policy.method != "refresh":
        raise ValueError("Only a curvature refresh measures the initial scale")
    directions, eigenvalues = _directions(
        source, space, native, policy.rank, whitened=policy.method == "warm_start"
    )
    provenance = dict(
        mode=policy.method,
        source_stage=source.index,
        target_stage=index,
        source_state=source.result.factors.metadata["state"],
        source_coordinates=source.result.factors.metadata["coordinates"],
        source_factors=source.result.factors.identity,
        target_state=state,
        target_coordinates=coordinates,
        requested_rank=policy.rank,
        inherited=policy.method == "warm_start" and len(directions) > 0,
        refreshed=policy.method == "refresh",
        hessian_actions=0,
    )
    if policy.method == "warm_start":
        # The seed is B^1/2 (I + W diag(e) W^T) B^1/2: the whitened source
        # correction follows a re-estimated target diagonal B instead of being
        # added in the source's units, so Sauce damps only a non-SPD transfer.
        directions *= np.sqrt(base)[None, :]
        factors = native.warm_start_curvature(
            base, directions, eigenvalues, state=state, coordinates=coordinates
        )
        provenance["transfer_damping"] = factors.metadata.get("transfer_damping", 1.0)
    else:
        basis = native.transfer_basis(
            base,
            directions,
            state=state,
            coordinates=coordinates,
            rank=min(policy.rank, len(directions)),
            basis_tolerance=policy.basis_tolerance,
        )
        directions = basis.read("directions")
        measured = None
        if scale_bounds is not None:
            measured = _scale_direction(objective, point, base, scale_bounds)
        # One normal job evaluates every retained direction and the scale probe.
        rows = directions if measured is None else np.vstack([directions, measured[0]])
        images = _hessian_images(objective, point, rows)
        if scale_bounds is not None:
            scale = 1.0
            if measured is not None:
                scale = _measured_scale(measured[1], measured[0], images[-1])
                images = images[:-1]
            # The basis is metric-orthonormal and so scale-equivariant: it is
            # sqrt(scale) times the basis of ``base``, and its images follow.
            directions = np.sqrt(scale) * directions
            images = np.sqrt(scale) * images
            base = scale * base
            provenance["initial_scale"] = scale
        provenance["hessian_actions"] = len(directions)
        factors = native.refresh_curvature(
            base,
            directions,
            images,
            state=state,
            coordinates=coordinates,
            basis_tolerance=policy.basis_tolerance,
            symmetry_tolerance=policy.symmetry_tolerance,
        )
    provenance["retained_rank"] = factors.metadata["rank"]
    return factors, provenance
