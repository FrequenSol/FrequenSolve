# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Reference-model illumination calibration on a frozen imaging linearization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .controls import ControlVector
from .statistics import _linear_materials


@dataclass(frozen=True)
class ReferenceIllumination:
    """Rickett reference-image envelope calibration.

    ``reference`` is a ControlVector, a same-space array, or a DataArray for a
    single block. ``smoothing`` maps physical axis names to box half-widths
    (quantities or values in model length units). Relative damping is a fraction
    of the maximum smoothed remigrated envelope, independently per block.
    """

    reference: Any
    smoothing: Mapping[str, Any] = field(default_factory=dict)
    relative_damping: float = 0.01
    padding: Any = "reflect"

    def __post_init__(self) -> None:
        if not np.isfinite(self.relative_damping) or self.relative_damping < 0:
            raise ValueError("relative_damping must be nonnegative finite")
        if self.padding != "reflect":
            from .curvature import _integer

            _integer(self.padding, "padding")
        if not isinstance(self.smoothing, Mapping):
            raise TypeError("smoothing must map axes to physical half-widths")

    def bind(self, linearization: Any) -> IlluminationCalibration:
        return IlluminationCalibration(linearization, self)


class IlluminationCalibration:
    """Reusable calibration; reference modeling, remigration and Rickett run once.

    Sauce forms the envelope weights at construction. ``apply`` multiplies
    images by those stored weights, which is Sauce's ``weights * image``.
    """

    def __init__(self, lin: Any, config: ReferenceIllumination):
        self.space = lin.space
        _linear_materials(self.space)
        self.native = lin.problem.backend.curvature()
        self.config = config
        self.state = lin.state_fingerprint
        reference = config.reference
        if isinstance(reference, xr.DataArray):
            if len(self.space.blocks) != 1:
                raise ValueError("A DataArray reference needs a single active block")
            block = self.space.block(self.space.blocks[0])
            if set(reference.dims) != set(block.dims):
                raise ValueError("Reference dimensions must match the control block")
            reference = reference.transpose(*block.dims)
            for dim in block.dims:
                if not np.array_equal(reference.coords[dim], block.coords[dim]):
                    raise ValueError(
                        "Reference coordinates must match the control block"
                    )
            reference = reference.values.ravel(order="F")
        if isinstance(reference, ControlVector):
            if not reference.space.equivalent(self.space):
                raise ValueError("Reference and linearization control spaces disagree")
            reference = reference.values
        reference = np.asarray(reference).copy()
        if (
            np.iscomplexobj(reference)
            or reference.shape != (self.space.size,)
            or not np.isfinite(reference).all()
        ):
            raise ValueError(
                "Reference must be a finite real vector on the active control space"
            )
        self._blocks = []
        axes_used = set()
        for name in self.space.blocks:
            block = self.space.block(name)
            if (
                not block.dims
                or not 1 <= len(block.shape) <= 3
                or not self.space._mask_of(block).all()
            ):
                raise ValueError(
                    "Reference illumination requires complete regular lattice blocks"
                )
            radii = []
            axes_used.update(block.dims)
            for dim, length in zip(block.dims, block.shape):
                coordinate = np.asarray(block.coords[dim], dtype=float)
                distance = config.smoothing.get(dim, 0.0)
                if hasattr(distance, "to"):
                    if block.units is None:
                        raise ValueError(
                            "Smoothing quantities require model length units"
                        )
                    distance = distance.to(block.units).magnitude
                distance = float(distance)
                if not np.isfinite(distance) or distance < 0:
                    raise ValueError("Smoothing half-widths must be nonnegative finite")
                if length > 1:
                    increments = np.diff(coordinate)
                    if not np.allclose(increments, increments[0]) or increments[0] == 0:
                        raise ValueError(
                            "Reference illumination requires regularly spaced control coordinates"
                        )
                    radius = int(np.ceil(distance / abs(increments[0])))
                else:
                    radius = 0
                radii.append(min(radius, length))
            depth = next(
                (i for i, d in enumerate(block.dims) if d in ("z", "depth", "s")), None
            )
            if depth is None:
                raise ValueError(
                    "Reference illumination needs a depth axis named z, depth or s"
                )
            section = self.space.slices[name]
            values = reference[section].reshape(block.shape, order="F")
            self._blocks.append((block, section, values, depth, radii))
        unknown = set(config.smoothing) - axes_used
        if unknown:
            raise ValueError(f"Unknown smoothing axes: {sorted(unknown)}")
        self.coordinates = hashlib.sha256(
            repr(
                [(b.name, b.shape, b.dims, b.basis_identity) for b, *_ in self._blocks]
            ).encode()
        ).hexdigest()
        # Joint action retains coupling between multiple material blocks.
        remigrated = np.asarray(lin.normal @ reference)
        weights = np.empty(self.space.size)
        for block, section, values, depth, radii in self._blocks:
            result = self.native.rickett(
                values,
                np.ones(block.shape),
                normal_reference=remigrated[section].reshape(block.shape, order="F"),
                state=self.state,
                coordinates=self.coordinates,
                depth_axis=depth,
                smoothing_radii=radii,
                relative_damping=self.config.relative_damping,
                padding=(
                    values.shape[depth]
                    if self.config.padding == "reflect"
                    else self.config.padding
                ),
            )
            packed_shape = np.moveaxis(values, depth, -1).shape
            restored = np.moveaxis(
                result.read("weights").reshape(packed_shape), -1, depth
            )
            weights[section] = restored.ravel(order="F")
        weights.flags.writeable = False
        self._weights = weights
        self.weights = ControlVector(weights, self.space)

    def apply(self, image: ControlVector) -> ControlVector:
        """Weight an image by the calibrated envelopes without running Sauce."""
        if not isinstance(image, ControlVector) or not self.space.equivalent(
            image.space
        ):
            raise ValueError("Image must be a ControlVector on the calibration space")
        if not np.isfinite(image.values).all():
            raise ValueError("Rickett migration values must be finite")
        with np.errstate(over="ignore"):
            normalized = self._weights * image.values
        if not np.isfinite(normalized).all():
            raise ValueError("Rickett normalization overflow")
        return ControlVector(normalized, self.space)
