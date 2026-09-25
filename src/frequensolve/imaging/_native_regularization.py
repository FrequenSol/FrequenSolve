"""Sauce-owned model energies and proximal callbacks for imaging workflows."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ._artifacts import ControlVectorFile, SmoothingConfig
from .controls import ControlSpace, ControlState, ControlVector
from .jobs import RegularizationJob
from .regularization import (
    TGV,
    TV,
    BoundRegularization,
    Regularization,
    Scaled,
    Sum,
    Tikhonov,
)


@dataclass(frozen=True)
class NativeRegularization(Regularization):
    """Native Tikhonov, TV or TGV on the full material control field.

    Sauce evaluates the energy and solves constrained proximal problems using
    native control bases, including mesh controls. TV/TGV use split-Bregman
    shrinkage; epsilon controls splitting, not smoothing of the norm. Weights
    and amplitude normalization are resolved once at the stage's first model.
    ``reference`` is an optional complete ControlState; fixed DOFs contribute
    their actual values to the energy. Without it, the model itself is regularized.
    """

    smoothing: Any = None
    reference: Optional[ControlState] = None
    iterations: int = 1000
    relative_tolerance: float = 1e-6
    absolute_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        config = SmoothingConfig.from_value(self.smoothing or SmoothingConfig())
        object.__setattr__(self, "smoothing", config)
        if self.reference is not None and not isinstance(self.reference, ControlState):
            raise TypeError(
                "native regularization reference must be a complete ControlState"
            )
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if any(
            not np.isfinite(t) or t <= 0
            for t in (self.relative_tolerance, self.absolute_tolerance)
        ):
            raise ValueError(
                "native regularization tolerances must be finite and positive"
            )

    def bind(
        self, space: ControlSpace, *, problem: Any = None, linearization: Any = None
    ) -> BoundNativeRegularization:
        if problem is None or linearization is None:
            raise ValueError(
                "native regularization binds to a problem and its stage linearization"
            )
        return BoundNativeRegularization(self, space, problem, linearization)


class BoundNativeRegularization(BoundRegularization):
    """Fixed native context and explicit full-state value/proximal requests."""

    regularization: NativeRegularization

    def __init__(
        self,
        specification: NativeRegularization,
        space: ControlSpace,
        problem: Any,
        linearization: Any,
    ) -> None:
        self.regularization = specification
        self.space = space
        self.problem = problem
        self.baseline: ControlState = linearization.state
        self.factor = 1.0
        self.reference = np.zeros_like(self.baseline.values)
        if specification.reference is not None:
            reference = specification.reference
            if not reference.space.without_support().equivalent(
                self.baseline.space.without_support()
            ):
                raise ValueError("native reference has a different full control layout")
            self.reference = reference.values.copy()
        self.source = copy.copy(linearization.job)
        # The native basis and material wavelengths belong to this stage, not
        # the mutable simulation last installed by a rejected PDE trial.
        self.source.simulation = copy.deepcopy(linearization.job.simulation)
        self.source.simulation.name = problem.backend.job_name("regularization_model")
        self.source.simulation.save()
        self._value_cache: dict[bytes, float] = {}
        job = self._run("prepare", self.baseline.values, context=None)
        self.context = job.context
        self.context_identity = json.loads(self.context.read_text())

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": self.regularization.smoothing.to_control_fs(),
            "factor": self.factor,
            "reference": hashlib.sha256(self.reference.tobytes()).hexdigest(),
            "context": self.context_identity,
        }

    def restore(self, recorded: Mapping[str, Any]) -> None:
        current = self.checkpoint()
        if any(recorded[k] != current[k] for k in ("config", "factor", "reference")):
            raise ValueError(
                "checkpoint regularization configuration or reference changed"
            )
        # Material weights can change with the new model; reuse the weights
        # frozen at the beginning of the interrupted stage.
        previous = recorded["context"]
        for name, block in self.context_identity.items():
            if block.get("identity") != previous.get(name, {}).get("identity"):
                raise ValueError("checkpoint regularization control basis changed")
        self.context_identity = previous
        self.context.write_text(json.dumps(previous))
        self._value_cache.clear()

    def _model_file(self, full: np.ndarray) -> ControlVectorFile:
        space = self.baseline.space
        return ControlVectorFile(
            {
                block.name: np.asarray(full[sl], dtype=float)
                for block, sl in zip(space.resolved_blocks, space.full_slices.values())
                if block.name.startswith("model.") and block.name in self.space.blocks
            },
            control_spaces={
                block.name: block.basis_identity
                for block in space.resolved_blocks
                if block.name.startswith("model.")
                and block.name in self.space.blocks
                and block.basis_identity
            },
        )

    def _full(self, v: Any) -> np.ndarray:
        vector = v if isinstance(v, ControlVector) else ControlVector(v, self.space)
        return self.baseline.with_update(vector).values.copy()

    def _run(
        self,
        operation: str,
        full: np.ndarray,
        *,
        context: Optional[Path],
        tau: float = 1.0,
        metric: Optional[ControlVectorFile] = None,
        lower: Optional[ControlVectorFile] = None,
        upper: Optional[ControlVectorFile] = None,
    ) -> RegularizationJob:
        spec = self.regularization
        job = RegularizationJob(
            self.source,
            smoothing=spec.smoothing,
            input_vector=self._model_file(full),
            operation=operation,
            context=context,
            tau=tau,
            metric=metric,
            lower=lower,
            upper=upper,
            iterations=spec.iterations,
            relative_tolerance=spec.relative_tolerance,
            absolute_tolerance=spec.absolute_tolerance,
            name=self.problem.backend.job_name("regularization_" + operation),
        )
        self.problem.backend.run(job, postprocess_only=True)
        return job

    @staticmethod
    def _read_value(job: RegularizationJob) -> float:
        report = json.loads(job.result.read_text())
        if report.get(
            "schema"
        ) != "fs-control-regularization-result-1" or not report.get("converged"):
            raise RuntimeError("Sauce returned an incomplete regularization result")
        value = float(report["value"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Sauce returned an invalid regularization value")
        return value

    def value(self, v: Any) -> float:
        full = self._full(v) - self.reference
        key = hashlib.sha256(full.tobytes()).digest()
        if key not in self._value_cache:
            job = self._run("value", full, context=self.context)
            self._value_cache[key] = self._read_value(job)
        return self.factor * self._value_cache[key]

    def prox(
        self,
        v: np.ndarray,
        tau: float,
        metric: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        full = self._full(np.clip(v, bounds[0], bounds[1]))
        # Fidelity targets may lie outside the box; only non-material coordinates
        # are projected here. Native material targets retain their full values.
        target = self._full(v)
        for block in self.space.resolved_blocks:
            if block.name.startswith("model."):
                sl = self.baseline.space.full_slices[block.name]
                full[sl] = target[sl]
        full_metric = np.ones_like(full)
        # with_update maps active coordinates to the complete model layout;
        # zeros from tangent-vector expansion are never used as frozen values.
        limit = np.finfo(float).max / 100
        lower = self._full(np.clip(bounds[0], -limit, limit))
        upper = self._full(np.clip(bounds[1], -limit, limit))
        # Material blocks outside the active space are excluded from the job;
        # inactive DOFs inside each included block have equal fixed bounds.
        for block in self.space.resolved_blocks:
            block_slice = self.baseline.space.full_slices[block.name]
            active = self.space._mask_of(block)
            values = np.ones(block.size)
            values[active] = np.asarray(metric)[self.space.slices[block.name]]
            full_metric[block_slice] = values
        job = self._run(
            "proximal",
            full - self.reference,
            context=self.context,
            tau=tau * self.factor,
            metric=self._model_file(full_metric),
            lower=self._model_file(np.clip(lower - self.reference, -limit, limit)),
            upper=self._model_file(np.clip(upper - self.reference, -limit, limit)),
        )
        self._read_value(job)
        result = ControlVectorFile.read(job.gradient_file(), native=True)
        for block in self.space.resolved_blocks:
            if block.name.startswith("model."):
                sl = self.baseline.space.full_slices[block.name]
                full[sl] = result[block.name] + self.reference[sl]
        state = ControlState(
            self.baseline.space,
            full,
            scaling=self.baseline.scaling,
            scaling_units=self.baseline.scaling_units,
        )
        return state.vector(self.space).values


def bind_workflow_regularization(
    specification: Any, space: ControlSpace, problem: Any, linearization: Any
) -> tuple[Optional[BoundRegularization], Optional[BoundNativeRegularization]]:
    """Separate smooth user terms from one native proximal model term."""
    native: Optional[BoundNativeRegularization] = None
    smooth: list[Regularization] = []

    def add(spec: Any, factor: float = 1.0) -> None:
        nonlocal native
        if spec is None or spec is False or factor == 0:
            return
        if isinstance(spec, Scaled):
            add(spec.regularization, factor * spec.factor)
            return
        if isinstance(spec, Sum):
            for term in spec.regularizations:
                add(term, factor)
            return
        if isinstance(spec, (SmoothingConfig, Mapping)):
            spec = NativeRegularization(spec)
        if isinstance(spec, (Tikhonov, TV)) and spec.alpha == 0:
            return
        if isinstance(spec, (Tikhonov, TV, TGV)):
            reference = getattr(spec, "reference", None)
            if reference is not None and not isinstance(reference, ControlState):
                reference = linearization.state.with_update(
                    ControlVector(np.asarray(reference), space)
                )
            config = (
                SmoothingConfig(
                    kind="tgv",
                    alpha1=spec.alpha1,
                    alpha2=spec.alpha2,
                    epsilon=spec.epsilon,
                )
                if isinstance(spec, TGV)
                else SmoothingConfig(
                    kind="tv" if isinstance(spec, TV) else "tikhonov",
                    alpha=spec.alpha,
                    derivative_order=spec.order,
                    epsilon=getattr(spec, "epsilon", 1e-3),
                )
            )
            spec = NativeRegularization(config, reference=reference)
        if isinstance(spec, NativeRegularization):
            if spec.smoothing.kind == "none" or not any(
                n.startswith("model.") for n in space.blocks
            ):
                return
            if native is not None:
                raise ValueError(
                    "one native model regularizer may be combined with smooth custom terms"
                )
            native = spec.bind(space, problem=problem, linearization=linearization)
            native.factor = factor
        else:
            smooth.append(spec if factor == 1 else factor * spec)

    add(specification)
    smooth_bound = (
        None
        if not smooth
        else (smooth[0] if len(smooth) == 1 else Sum(*smooth)).bind(space)
    )
    return smooth_bound, native
