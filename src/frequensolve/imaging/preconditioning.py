# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Reusable local and balanced coarse inverse actions on frozen material spaces.

Sauce constructs and reduces the transferred coarse basis and factors its
measured curvature. Application uses cached factors in memory; it performs no
backend calls. Setup always measures the current full regularized normal.
"""

from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ._backend import fingerprint
from .controls import ControlSpace, ControlVector
from .curvature import _integer, _relative_tolerance
from .regularization import BoundPreconditioner, Diagonal, Preconditioner, _values_on


def _positive(value: float, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ReceiverBlockDiagonal(Preconditioner):
    """Factor collocated cross-parameter receiver-probe blocks once per update.

    All active controls must be the selected mesh controls with shared support.
    ``scales`` are optional positive physical parameter scales for stabilizing
    the inverse; they do not change the objective or the returned units.
    ``relative_damping`` bounds unilluminated directions by a positive ridge.
    The regularization curvature diagonal is included before factoring.
    """

    controls: tuple[str, ...] = ("ss", "sp", "rho")
    probe_count: int = 16
    source_samples: int | None = None
    seed: int = 0
    relative_damping: float = 0.01
    scales: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.controls, str):
            raise TypeError("controls must be a sequence of material names")
        names = tuple(str(n).removeprefix("model.") for n in self.controls)
        if len(names) < 2 or len(set(names)) != len(names) or any(not n for n in names):
            raise ValueError(
                "controls must contain at least two distinct material names"
            )
        object.__setattr__(self, "controls", names)
        _integer(self.probe_count, "probe_count", minimum=1)
        _integer(self.seed, "seed")
        if self.source_samples is not None:
            _integer(self.source_samples, "source_samples", minimum=1)
        _positive(self.relative_damping, "relative_damping")
        if self.scales is not None:
            from types import MappingProxyType

            if not isinstance(self.scales, Mapping):
                raise TypeError("scales must map material names to positive values")
            for value in self.scales.values():
                _positive(value, "scale")
            scales = {
                str(k).removeprefix("model."): float(v) for k, v in self.scales.items()
            }
            if set(scales) != set(names) or len(scales) != len(self.scales):
                raise ValueError(
                    "scales must specify each selected control exactly once"
                )
            for value in scales.values():
                _positive(value, "scale")
            object.__setattr__(self, "scales", MappingProxyType(scales))

    def receiver_probe_request(self) -> dict:
        request = dict(
            probes=self.probe_count,
            seed=self.seed,
            blocks=dict(controls=list(self.controls)),
        )
        if self.source_samples is not None:
            request["source_samples"] = self.source_samples
        return request

    def bind(self, space: ControlSpace) -> BoundPreconditioner:
        return _BoundReceiverBlocks(self, space)


class _BoundReceiverBlocks(BoundPreconditioner):
    def __init__(self, config: ReceiverBlockDiagonal, space: ControlSpace) -> None:
        super().__init__(config, space)
        self._inverse = self._indices = None
        self.info: dict[str, Any] = {}

    def update(self, linearization: Any, regularization: Any = None) -> None:
        self._inverse = self._indices = None
        space, config = linearization.space, self.preconditioner
        blocks = linearization.receiver_blocks
        names = [space.block(n).name for n in config.controls]
        if set(names) != set(space.blocks) or tuple(blocks.controls) != config.controls:
            raise ValueError(
                "ReceiverBlockDiagonal needs all active blocks in the requested order"
            )
        if regularization is not None and not regularization.space.equivalent(space):
            raise ValueError("regularization belongs to another control space")
        lookup = np.full(space.full_size, -1, dtype=np.int64)
        lookup[space.active_indices] = np.arange(space.size)
        indices = np.column_stack([lookup[space.full_slices[n]] for n in names])
        if not np.all((indices >= 0) == (indices[:, :1] >= 0)):
            raise ValueError(
                "ReceiverBlockDiagonal requires matching parameter support"
            )
        selected = indices[:, 0] >= 0
        indices = indices[selected]
        scale = np.array(
            [
                1.0 if config.scales is None else config.scales[n]
                for n in config.controls
            ]
        )
        matrix = np.array(blocks.matrices[selected], copy=True)
        if regularization is not None:
            diagonal = np.asarray(
                regularization.curvature_diagonal(linearization.point)
            )
            if (
                diagonal.shape != (space.size,)
                or not np.isfinite(diagonal).all()
                or np.any(diagonal < 0)
            ):
                raise ValueError(
                    "regularization diagonal must be finite and nonnegative"
                )
            for k in range(len(names)):
                matrix[:, k, k] += diagonal[indices[:, k]]
        matrix *= scale[None, :, None] * scale[None, None, :]
        if not np.isfinite(matrix).all() or not len(matrix):
            raise ValueError("receiver blocks must be finite and nonempty")
        eigenvalues, vectors = np.linalg.eigh(matrix)
        largest = float(np.max(eigenvalues))
        if largest <= 0:
            raise ValueError("receiver blocks contain no positive curvature")
        if np.min(eigenvalues) < -1e-8 * largest:
            raise ValueError("receiver blocks have negative curvature")
        ridge = config.relative_damping * largest
        inverse = (
            vectors / (np.maximum(eigenvalues, 0) + ridge)[:, None, :]
        ) @ vectors.transpose(0, 2, 1)
        inverse *= scale[None, :, None] * scale[None, None, :]
        inverse.flags.writeable = indices.flags.writeable = False
        self.space, self._indices, self._inverse = space, indices, inverse
        self.info = dict(blocks=len(matrix), ridge=ridge)

    def apply(self, g: Any) -> ControlVector:
        if self._inverse is None:
            raise RuntimeError("Call update before applying ReceiverBlockDiagonal")
        values = _values_on(self.space, g)
        result = np.empty(self.space.size)
        result[self._indices] = np.einsum(
            "nij,nj->ni", self._inverse, values[self._indices], optimize=False
        )
        return ControlVector(result, self.space)


@dataclass(frozen=True)
class CoarseCorrection(Preconditioner):
    """Balanced coarse correction around an existing local inverse.

    ``coarse`` is a resolved, frozen native material ControlSpace with the same
    named physical parameters as the fine space. Sauce transfers its active
    cardinal functions; no Python interpolation or reverse model transfer is
    used. All fine controls remain unknowns. Setup batches exact normal actions
    and includes the complete regularization supplied to ``update``.

    ``max_directions`` and ``max_memory_bytes`` bound setup before allocating
    controls-by-rank arrays. Oversized spaces raise; no modes are silently
    truncated. Budget six such arrays and one full-block lift for transient
    setup, two controls-by-rank arrays for application.
    Rebind/update after changes to the model, objective, weights or regularizer.
    """

    coarse: ControlSpace
    local: Preconditioner = field(default_factory=lambda: Diagonal(probes="receiver"))
    max_directions: int = 256
    max_memory_bytes: int = 512 * 1024**2
    basis_tolerance: float = 1e-4
    symmetry_tolerance: float = 1e-2

    def __post_init__(self) -> None:
        if not isinstance(self.coarse, ControlSpace):
            raise TypeError("coarse must be a resolved native material ControlSpace")
        if not isinstance(self.local, Preconditioner):
            raise TypeError("local must be a preconditioner specification")
        _integer(self.max_directions, "max_directions", minimum=1)
        try:
            budget = operator.index(self.max_memory_bytes)
        except TypeError as error:
            raise ValueError("max_memory_bytes must be a positive integer") from error
        if isinstance(self.max_memory_bytes, (bool, np.bool_)) or budget < 1:
            raise ValueError("max_memory_bytes must be a positive integer")
        _relative_tolerance(self.basis_tolerance, "basis_tolerance")
        _relative_tolerance(self.symmetry_tolerance, "symmetry_tolerance")

    def receiver_probe_request(self) -> dict | None:
        from .workflows import _receiver_probe_request

        return _receiver_probe_request(self.local)

    def bind(self, space: ControlSpace) -> BoundPreconditioner:
        return _BoundCoarseCorrection(self, space)


def _normal_images(
    linearization: Any, rows: np.ndarray, regularization: Any
) -> np.ndarray:
    """Batch data and native Tikhonov together, then add other smooth terms."""
    from ._native_regularization import BoundNativeRegularization
    from .regularization import _BoundSum

    def terms(term):
        if isinstance(term, _BoundSum):
            for child in term.terms:
                yield from terms(child)
        elif term is not None:
            yield term

    remaining = list(terms(regularization))
    native = next(
        (t for t in remaining if isinstance(t, BoundNativeRegularization)), None
    )
    if native is not None:
        remaining.remove(native)
    options = dict(regularization=native)
    if "cache" in inspect.signature(linearization.apply_normal_batch).parameters:
        options["cache"] = False
    products = linearization.apply_normal_batch(rows, **options)
    images = np.stack([p.values for p in products])
    for term in remaining:
        operator = term.hessian_operator(linearization.point)
        for i, row in enumerate(rows):
            images[i] += np.asarray(operator @ row).reshape(-1)
    return images


class _BoundCoarseCorrection(BoundPreconditioner):
    def __init__(self, config: CoarseCorrection, space: ControlSpace) -> None:
        super().__init__(config, space)
        self.local = config.local.bind(space)
        self._directions = self._images = None
        self.info: dict[str, Any] = {}

    def update(self, linearization: Any, regularization: Any = None) -> None:
        from .statistics import _linear_materials, mesh_descriptor

        # A failed refresh cannot leave an apparently usable stale inverse.
        self._directions = self._images = None
        self.info = {}
        start = perf_counter()
        space, coarse, config = (
            linearization.space,
            self.preconditioner.coarse,
            self.preconditioner,
        )
        _linear_materials(space)
        _linear_materials(coarse)
        if set(space.blocks) != set(coarse.blocks):
            raise ValueError(
                "coarse and fine spaces must have the same named material controls"
            )
        if regularization is not None and not regularization.space.equivalent(space):
            raise ValueError("regularization belongs to another control space")
        count = coarse.size
        if not count or count > config.max_directions:
            raise ValueError(
                f"coarse space has {count} directions; choose a smaller mesh or increase max_directions"
            )
        # Native transfer covers the full target block before support restriction.
        # Include that temporary even when most fine controls are frozen.
        lifted_entries = max(
            np.count_nonzero(coarse._mask_of(coarse.block(name)))
            * space.block(name).size
            for name in space.blocks
        )
        memory = 8 * (6 * space.size * count + lifted_entries)
        if memory > config.max_memory_bytes:
            raise ValueError(
                f"coarse setup needs approximately {memory} bytes; choose a smaller mesh or increase max_memory_bytes"
            )
        descriptors = []
        for name in space.blocks:
            before, after = coarse.block(name), space.block(name)
            if before.kind != "mesh" or after.kind != "mesh":
                raise ValueError("coarse correction requires native material meshes")
            if (before.prop, before.subdomain, before.transform, before.units) != (
                after.prop,
                after.subdomain,
                after.transform,
                after.units,
            ):
                raise ValueError(
                    f"coarse correction changes physical coordinates of {name}"
                )
            descriptors.append(
                (name, mesh_descriptor(coarse, before), mesh_descriptor(space, after))
            )
        native = linearization.problem.backend.curvature()
        local = config.local.bind(space)
        local.update(linearization, regularization=regularization)
        directions = np.zeros((count, space.size))
        first = 0
        previous_transfer = None
        lifted = None
        for name, source, target in descriptors:
            columns = np.flatnonzero(coarse._mask_of(coarse.block(name)))
            if not len(columns):
                continue
            transfer_key = (
                source["identity"],
                target["identity"],
                source["material"],
                columns.tobytes(),
            )
            if transfer_key != previous_transfer:
                lifted = native.mesh_basis(source, target, columns).read("directions")
                previous_transfer = transfer_key
            assert lifted is not None
            if lifted.shape != (len(columns), target["size"]):
                raise ValueError(
                    "Sauce returned inconsistent coarse transfer dimensions"
                )
            directions[first : first + len(columns), space.slices[name]] = lifted[
                :, space._mask_of(space.block(name))
            ]
            first += len(columns)
        del lifted
        # Normalize individual columns before rank revelation so units and
        # disparate mesh support do not preferentially discard one parameter.
        norms = np.linalg.norm(directions, axis=1)
        directions /= np.where(norms > 0, norms, 1)[:, None]
        coordinates = fingerprint(meshes=descriptors, active=space.active_indices)
        state = fingerprint(
            point=linearization.point.values,
            job=str(linearization.job.job_file),
        )
        reduced = native.transfer_basis(
            np.ones(space.size),
            directions,
            state=state,
            coordinates=coordinates,
            rank=min(count, space.size),
            basis_tolerance=config.basis_tolerance,
        )
        del directions
        directions = reduced.read("directions")
        if not len(directions):
            raise ValueError(
                "coarse space has no active independent directions on the fine mesh"
            )
        setup = perf_counter()
        images = _normal_images(linearization, directions, regularization)
        normal_seconds = perf_counter() - setup
        factors = native.coarse_factor(
            directions,
            images,
            state=state,
            coordinates=coordinates,
            symmetry_tolerance=config.symmetry_tolerance,
        )
        del directions, images
        q, aq = factors.read("directions"), factors.read("images")
        if (
            q.shape != aq.shape
            or q.shape != (factors.metadata["rank"], space.size)
            or not np.isfinite(q).all()
            or not np.isfinite(aq).all()
        ):
            raise ValueError("Sauce returned invalid coarse correction factors")
        q.flags.writeable = aq.flags.writeable = False
        self.space, self._directions, self._images = space, q, aq
        self.local = local
        self.info = dict(
            rank=len(q),
            requested_directions=count,
            normal_directions=len(q),
            setup_seconds=perf_counter() - start,
            normal_seconds=normal_seconds,
            resident_bytes=q.nbytes + aq.nbytes,
            condition=factors.metadata["condition"],
            hessian_asymmetry=factors.metadata["hessian_asymmetry"],
            factors=str(factors.path),
        )

    def apply(self, g: Any) -> ControlVector:
        if self._directions is None:
            raise RuntimeError("Call update before applying CoarseCorrection")
        r = _values_on(self.space, g)
        q, aq = self._directions, self._images
        coefficients = q @ r
        z = self.local.apply(ControlVector(r - coefficients @ aq, self.space)).values
        return ControlVector(z + (coefficients - aq @ z) @ q, self.space)
