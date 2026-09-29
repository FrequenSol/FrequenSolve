"""Native material-mass inverse metrics for coefficient optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._artifacts import SmoothingConfig
from ._native_regularization import NativeRegularization
from .controls import ControlSpace, ControlVector
from .regularization import BoundPreconditioner, Preconditioner, _values_on


@dataclass(frozen=True)
class NativeMass(Preconditioner):
    """Use gamma*M^{-1} as an optimizer inverse metric on material controls.

    M is integrated on the native geometry with the constrained control basis.
    Raw coefficient derivatives remain unchanged. By default gamma is calibrated
    once per update from a Gauss-Newton plus regularization Rayleigh quotient.
    With approximation="diagonal", cache diag(M) by native basis identity across
    stage bindings and apply its reciprocal without native calls or iterative
    solves. This is an approximate L2 map. Only completely active material blocks
    are supported; partial coefficient masks are rejected explicitly.
    """

    curvature_scale: bool = True
    iterations: int = 300
    relative_tolerance: float = 1e-9
    absolute_tolerance: float = 1e-30
    approximation: str = "consistent"
    _diagonal_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.approximation not in {"consistent", "diagonal"}:
            raise ValueError("NativeMass approximation must be consistent or diagonal")
        if self.iterations < 1 or any(
            not np.isfinite(v) or v <= 0
            for v in (self.relative_tolerance, self.absolute_tolerance)
        ):
            raise ValueError("NativeMass requires positive iterations and tolerances")

    def bind(self, space: ControlSpace) -> BoundNativeMass:
        return BoundNativeMass(self, space)


class BoundNativeMass(BoundPreconditioner):
    """A frozen native mass map with optional scalar curvature calibration."""

    def __init__(self, specification: NativeMass, space: ControlSpace) -> None:
        super().__init__(specification, space)
        self.scale = 1.0
        self.callbacks = None
        self._diagonal = None
        for block in space.resolved_blocks:
            if not block.name.startswith("model."):
                raise ValueError("NativeMass currently supports only material controls")
            if not np.all(space._mask_of(block)):
                raise ValueError("NativeMass requires fully active material blocks")

    def update(self, linearization: Any, regularization: Any = None) -> None:
        spec = self.preconditioner
        self.callbacks = NativeRegularization(
            SmoothingConfig(kind="tikhonov", alpha=0.0, normalize_amplitude=False),
            iterations=spec.iterations,
            relative_tolerance=spec.relative_tolerance,
            absolute_tolerance=spec.absolute_tolerance,
        ).bind(self.space, problem=linearization.problem, linearization=linearization)
        cache_hit = False
        if spec.approximation == "diagonal":
            # Native identities include geometry, refinement, basis and constraints.
            # Layout/order is part of the key; model values and frequencies are not.
            identities = tuple(
                sorted(
                    (name, block.get("identity", ""))
                    for name, block in self.callbacks.context_identity.items()
                )
            )
            key = (
                identities,
                tuple(
                    (b.name, b.size, b.basis_identity)
                    for b in self.space.resolved_blocks
                ),
            )
            reusable = bool(identities) and all(identity for _, identity in identities)
            diagonal = spec._diagonal_cache.get(key) if reusable else None
            cache_hit = diagonal is not None
            if diagonal is None:
                diagonal = self._action(
                    "mass_diagonal", np.zeros(self.space.size)
                ).values.copy()
                if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
                    raise ValueError("NativeMass diagonal must be finite and positive")
                diagonal.setflags(write=False)
                if reusable:
                    spec._diagonal_cache[key] = diagonal
            self._diagonal = diagonal
        self.scale = 1.0
        numerator = denominator = None
        if spec.curvature_scale:
            if linearization.gradient is None:
                raise ValueError("NativeMass curvature calibration requires a gradient")
            gradient = linearization.gradient.values.copy()
            if regularization is not None:
                gradient += regularization.gradient(linearization.point).values
            direction = self.riesz(gradient).values
            numerator = float(gradient @ direction)
            curvature = np.asarray(linearization.normal @ direction)
            if regularization is not None:
                curvature += np.asarray(
                    regularization.hessian_operator(linearization.point) @ direction
                )
            denominator = float(direction @ curvature)
            if numerator < 0 or denominator < 0 or not np.isfinite(denominator):
                raise ValueError("NativeMass curvature calibration is not positive")
            if numerator > 0:
                if denominator == 0:
                    raise ValueError("NativeMass has zero directional curvature")
                self.scale = numerator / denominator
        self.diagnostics = {
            "metric": f"gamma * native {spec.approximation} mass inverse",
            "gamma": self.scale,
            "gradient_mass_inverse_pairing": numerator,
            "directional_curvature": denominator,
            "controls": self.space.size,
            "relative_tolerance": spec.relative_tolerance,
            "diagonal_cache_hit": cache_hit,
        }
        (self.callbacks.context.parent / "native_mass_metric.json").write_text(
            json.dumps(self.diagnostics, indent=2) + "\n"
        )

    def _action(self, operation: str, values: Any) -> ControlVector:
        if self.callbacks is None:
            raise RuntimeError("NativeMass must be updated before applying it")
        values = _values_on(self.space, values)
        full = np.zeros_like(self.callbacks.baseline.values)
        for block in self.space.resolved_blocks:
            full[self.callbacks.baseline.space.full_slices[block.name]] = values[
                self.space.slices[block.name]
            ]
        job = self.callbacks._run(operation, full, context=self.callbacks.context)
        self.callbacks._read_value(job)
        return ControlVector(self.callbacks._read_gradient(job), self.space)

    def mass(self, v: Any) -> ControlVector:
        """Map primal coefficients to their native L2 covector."""
        return self._action("mass", v)

    def riesz(self, b: Any) -> ControlVector:
        """Return the unscaled mass inverse action (approximate in diagonal mode)."""
        if self.preconditioner.approximation == "diagonal":
            if self._diagonal is None:
                raise RuntimeError("NativeMass must be updated before applying it")
            return ControlVector(_values_on(self.space, b) / self._diagonal, self.space)
        return self._action("mass_inverse", b)

    def apply(self, g: Any) -> ControlVector:
        return ControlVector(self.scale * self.riesz(g).values, self.space)
