# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Native HV geometry objectives for acoustic and elastic FWI.

HV acts on complete receiver signals, with independent real and imaginary
auxiliary solves. Its reduced Gauss–Newton model linearizes the coupled
signal and deformation equations before eliminating auxiliary increments.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, Sequence

if TYPE_CHECKING:
    from .problem import ImagingProblem
    from .data import ObservedGroup

import numpy as np

from ._artifacts import ControlVectorFile
from ._backend import fingerprint
from .controls import ControlState, ControlVector, unqualified_block_name
from .data import TraceStoreRef
from .jobs import ControlGradientJob
from .misfit import Misfit, Normalization

__all__ = ["HV", "HVProblem", "HVLinearization"]


@dataclass(frozen=True)
class HV:
    """HV auxiliary problem settings.

    ``scale`` is a fixed positive pressure amplitude (prefer a quantity in Pa).
    Coordinates along each straight receiver line are normalized to [0, 1].
    ``lambda_`` is the velocity first-derivative penalty; ``epsilon`` penalizes
    its second derivative. Boundary-error refinement starts from the coarse
    ``spatial_intervals`` by ``path_intervals`` mesh. Both inner convergence
    criteria must pass; reaching the iteration cap produces no gradient.
    """

    scale: Any
    kappa: float = 0.05
    lambda_: float = 0.01
    epsilon: float = 0.001
    path_intervals: int = 4
    spatial_intervals: int = 4
    order: int = 2
    max_refinements: int = 4
    relative_mesh_tolerance: float = 0.01
    max_iterations: int = 150
    relative_energy_tolerance: float = 1e-6
    relative_stationarity_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        Normalization.explicit(self.scale, reduction="sum")
        for name in (
            "kappa",
            "lambda_",
            "epsilon",
            "relative_mesh_tolerance",
            "relative_energy_tolerance",
            "relative_stationarity_tolerance",
        ):
            value = float(getattr(self, name))
            if (
                not np.isfinite(value)
                or value < 0
                or (value == 0 and name != "lambda_")
            ):
                raise ValueError(
                    f"HV {name} must be finite and {'nonnegative' if name == 'lambda_' else 'positive'}"
                )
        for name in (
            "path_intervals",
            "spatial_intervals",
            "order",
            "max_refinements",
            "max_iterations",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isfinite(value)
                or int(value) != value
                or value < (0 if name == "max_refinements" else 1)
            ):
                raise ValueError(
                    f"HV {name} must be a {'nonnegative' if name == 'max_refinements' else 'positive'} integer"
                )

    def to_fs(self) -> dict[str, Any]:
        """Return the native objective configuration (excluding normalization)."""
        return {
            "kind": "hv",
            "kappa": float(self.kappa),
            "lambda": float(self.lambda_),
            "epsilon": float(self.epsilon),
            "discretization": {
                name: int(getattr(self, name))
                for name in (
                    "path_intervals",
                    "spatial_intervals",
                    "order",
                    "max_refinements",
                )
            }
            | {"relative_mesh_tolerance": float(self.relative_mesh_tolerance)},
            "inner_solver": {
                "max_iterations": int(self.max_iterations),
                "relative_energy_tolerance": float(self.relative_energy_tolerance),
                "relative_stationarity_tolerance": float(
                    self.relative_stationarity_tolerance
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.to_fs(),
            "normalization": Normalization.explicit(
                self.scale, reduction="sum"
            ).to_fs(),
        }

    def misfit(self) -> Misfit:
        """Waveform scaffold used for observation binding and state discovery."""
        return Misfit.l2(
            normalization=Normalization.explicit(self.scale, reduction="sum"),
            include_default_preprocess=False,
        )

    def payload(self, groups: Sequence[ObservedGroup]) -> dict[str, Any]:
        payload = self.misfit().to_fs(groups)
        for term in payload["objective_terms"]:
            term["objective"] = self.to_fs()
        return payload


class HVLinearization:
    """Native HV value and material covector at one model state."""

    def __init__(
        self,
        problem: HVProblem,
        state: ControlState,
        value: float,
        gradient: ControlVector,
        job: ControlGradientJob,
    ) -> None:
        self.problem, self.state, self.job = problem, state, job
        self.space = problem.space
        self.point = state.vector(self.space)
        self.value = float(value)
        self.gradient = gradient
        self.curvature_job = None
        self._normal = None
        self._receiver_normal = None

    def field(self, vector=None, *, input_role="dual", local_solver=None):
        """Sample a model or gradient on the retained native control basis."""
        from .problem import Linearization

        return Linearization.field(
            self, vector, input_role=input_role, local_solver=local_solver
        )

    @property
    def normal(self):
        """Reduced HV Gauss–Newton curvature, prepared lazily and reused."""
        from .operators import Normal

        if self._normal is None:
            self._normal = Normal(self)
        return self._normal

    def apply_normal(self, direction):
        """Pull the coupled HV receiver curvature through the wave Jacobian."""
        from ._hv_curvature import HVReceiverNormal

        if self._receiver_normal is None:
            if self.curvature_job is None:
                prepared = self.problem.linearize(self.point, curvature=True)
                self.curvature_job = prepared.curvature_job
            self._receiver_normal = HVReceiverNormal(self)
        return self._receiver_normal.apply(direction)

    def apply_normal_batch(self, directions, *, regularization=None):
        """Reuse auxiliary factors for successive material directions."""
        if regularization is not None:
            raise NotImplementedError("HV normal does not fuse model regularization")
        return [self.apply_normal(direction) for direction in directions]

    @property
    def jacobian(self) -> NoReturn:
        raise NotImplementedError("HV has no residual-space Jacobian")


class HVProblem:
    """HV objective view over serial Cartesian 2D scalar receiver data.

    Requires independent physical sources, dense uniform straight scalar
    receiver lines, and grid/profile or frozen meshed material controls. No source
    controls, spectral derivatives, preprocessing or receiver projections are
    supported. The native receiver adapter validates line geometry. Gradients
    use frozen DPG Gram matrices. Use Galerkin for a full discretized derivative.
    Frequency weights multiply both values and covectors.
    """

    def __init__(self, problem: ImagingProblem, hv: HV) -> None:
        if not isinstance(hv, HV):
            raise TypeError("hv must be an HV configuration")
        sim = problem.simulation
        if (
            sim.dimension != 2
            or sim.physics not in {"acoustic", "elastic", "coupled"}
            or getattr(sim, "_axisymmetric", False)
        ):
            raise ValueError(
                "HV requires Cartesian 2D acoustic, elastic, or coupled simulations"
            )
        if sim.acquisition.source_encoding is not None:
            raise ValueError(
                "HV requires independent physical sources; remove source encoding"
            )
        if problem.kernel_derivative is not None:
            raise ValueError("HV does not support spectral kernel derivatives")
        if any(
            b.kind not in {"grid", "profile", "mesh"}
            for b in problem.full_space.resolved_blocks
        ):
            raise ValueError(
                "HV currently requires grid/profile/mesh material-only controls"
            )
        for group in problem.observed_groups:
            if group.observed is None or group.derivatives:
                raise ValueError(
                    "HV requires observed scalar data without spectral derivatives"
                )
        if any(group.survey is not None for group in sim.acquisition.receiver_groups):
            raise ValueError("HV requires dense receiver sampling")
        self.hv = hv
        self._problem = problem.restrict(misfit=hv.misfit(), smoothing=None)
        self._cache: OrderedDict[str, HVLinearization] = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._problem, name)

    @property
    def _shared(self):
        return self._problem._shared

    @property
    def state(self) -> ControlState | None:
        return self._problem.state

    @state.setter
    def state(self, value: ControlState) -> None:
        self._problem.state = value

    @property
    def smoothing(self) -> None:
        return None

    def identity(self) -> dict[str, Any]:
        return {**self._problem.identity(), "hv": self.hv.to_dict()}

    def restrict(
        self,
        *args: Any,
        misfit: Any = None,
        loss: Any = None,
        smoothing: Any = None,
        **kwargs: Any,
    ) -> HVProblem:
        if misfit is not None or loss is not None or smoothing is not None:
            raise ValueError(
                "HV's objective cannot be replaced or smoothed; use model regularization in FWI"
            )
        view = HVProblem(self._problem.restrict(*args, **kwargs), self.hv)
        view._cache = self._cache
        return view

    def clear_cache(self) -> None:
        self._cache.clear()
        self._problem.clear_cache()

    def linearize(
        self, v: Any = None, *, gradient: bool = True, curvature: bool = False
    ) -> HVLinearization:
        """Evaluate HV, optionally retaining coupled auxiliary GN curvature.

        ``curvature=True`` avoids a second auxiliary evaluation when the normal
        operator will be used. Otherwise ``linearization.normal`` prepares it
        on its first application. Ordinary value/gradient calls retain no
        curvature matrices.
        """
        # The native HV workflow always computes the value and covector together.
        base = self._problem
        state = base._state_at(v)
        key = fingerprint(**self.identity(), hv_state=state.values)
        cached = self._cache.get(key)
        if cached is None or (curvature and cached.curvature_job is None):
            base._sync_simulation(state)
            current = None
            active_space = base.space
            if not base._inline_authoring():
                # Native overrides apply to the active subspace. Include every
                # material block so a restricted view preserves inactive values.
                active_space = base.full_space
                directory = base.workdir / key.split(":")[-1][:16]
                directory.mkdir(parents=True, exist_ok=True)
                current = directory / "current.h5"
                state.vector(active_space).to_file(native=True).write(current)
            groups = base.observed_groups
            observed = {}
            for group in groups:
                source = group.observed
                if source is None:
                    raise ValueError("HV requires observed pressure data")
                observed[group.name] = (
                    source.file if isinstance(source, TraceStoreRef) else source
                )
            payload = self.hv.payload(groups)
            if curvature:
                for term in payload["objective_terms"]:
                    term["objective"]["export_receiver_curvature"] = True
            job = ControlGradientJob(
                base.backend.job_name("hv"),
                base.simulation,
                base.frequencies,
                kind="rtm",
                observed=observed,
                gradient="gradient.h5",
                objective_file="objective.h5",
                gram_derivative="frozen",
                current=current,
                active=[unqualified_block_name(name) for name in active_space.blocks],
                misfit=payload,
                weights=base.weights,
                preserve_task_outputs=True,
            )
            base.backend.run(job)
            failures = job.failed_tasks()
            if failures:
                raise RuntimeError(
                    f"HV job {job.name!r} failed: {failures}; see {job._stdout_path}"
                )
            from .problem import _vector_from_file

            covector = _vector_from_file(
                ControlVectorFile.read(job.gradient_file()), base.space
            )
            value = job.objective_value
            if not np.isfinite(value) or not np.all(np.isfinite(covector.values)):
                raise RuntimeError("HV produced a nonfinite objective or gradient")
            self._cache[key] = HVLinearization(self, state, value, covector, job)
            if curvature:
                self._cache[key].curvature_job = job
            while len(self._cache) > base._shared.cache.capacity:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def value(self, v: Any = None) -> float:
        return self.linearize(v, gradient=False).value

    def gradient(self, v: Any = None) -> ControlVector:
        return self.linearize(v).gradient

    def normal(self, v: Any = None):
        """Return HV curvature rather than the underlying waveform normal."""
        return self.linearize(v, curvature=True).normal

    def jacobian(self, v: Any = None) -> NoReturn:
        raise NotImplementedError("HV has no residual-space Jacobian")
