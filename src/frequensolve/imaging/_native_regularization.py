"""Sauce-owned model energies and proximal callbacks for imaging workflows."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from frequensolve.inversion.optimization import _shares_arrays

from ._artifacts import ControlVectorFile, SmoothingConfig, control_smoothing
from ._block_digest import block_digest
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
    _BoundScaled,
    _BoundSum,
    _SymmetricModelOperator,
)


@dataclass(frozen=True)
class NativeRegularization(Regularization):
    """Native Tikhonov, TV or TGV on the full material control field.

    Sauce evaluates the energy and solves constrained proximal problems using
    native control bases, including mesh controls. TV/TGV use split-Bregman
    shrinkage; epsilon controls splitting, not smoothing of the norm. Weights
    and amplitude normalization are resolved once at the stage's first model;
    wavelength weights use the local length ``L(x) = lambda*v(x)/f`` (sizing
    wavespeed of that model, largest frequency, no ``2*pi``).
    ``reference`` is an optional complete ControlState; fixed DOFs contribute
    their actual values to the energy. Without it, the model itself is regularized.
    """

    smoothing: Any = None
    reference: Optional[ControlState] = None
    iterations: int = 1000
    relative_tolerance: float = 1e-4
    absolute_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        config = control_smoothing(self.smoothing or SmoothingConfig())
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
        source = getattr(linearization, "regularization_job", None)
        if source is None:
            source = linearization.job
        self.source = copy.copy(source)
        # The native basis and the stage wavespeed v(x) behind the local
        # smoothing length L(x) = lambda*v(x)/f belong to this stage, not the
        # mutable simulation last installed by a rejected PDE trial.
        self.source.simulation = copy.deepcopy(self.source.simulation)
        self.source.simulation.name = problem.backend.job_name("regularization_model")
        self.source.simulation.save()
        self._value_cache: dict[tuple, float] = {}
        self._gradient_cache: dict[tuple, np.ndarray] = {}
        # Staged prox metric/bounds: (weak references to read-only sources,
        # their digests, the published files).
        self._proximal: Optional[tuple] = None
        job = self._run("prepare", self.baseline.values, context=None)
        self.context = job.context
        self.context_identity = json.loads(self.context.read_text())

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": self.regularization.smoothing.to_control_fs(),
            "factor": self.factor,
            # Same digest as hashing tobytes(), without copying the vector.
            "reference": hashlib.sha256(
                np.ascontiguousarray(self.reference).data
            ).hexdigest(),
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
        self._gradient_cache.clear()

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

    @functools.cached_property
    def _model_layout(self) -> tuple:
        """``(block, state slice, active slice, support mask)`` of each native block.

        In :meth:`_model_file` order; the baseline, reference and space are
        fixed for this binding, so the layout is resolved once.
        """
        space = self.baseline.space
        active = self.space.slices
        blocks = {block.name: block for block in self.space.resolved_blocks}
        return tuple(
            (
                block.name,
                full,
                active[block.name],
                self.space._mask_of(blocks[block.name]),
            )
            for block, full in zip(space.resolved_blocks, space.full_slices.values())
            if block.name.startswith("model.") and block.name in blocks
        )

    @functools.cached_property
    def _tangent_origin(self) -> np.ndarray:
        """The full model at zero active coordinates (fixed values elsewhere)."""
        origin = self._full(np.zeros(self.space.size))
        origin.flags.writeable = False
        return origin

    def _at_reference(self, offset: np.ndarray) -> bool:
        """Whether every regularized coefficient equals its reference.

        The native energies vanish there with a zero Tikhonov gradient, so no
        job is needed (e.g. an LS-RTM image at zero without a reference).
        """
        space = self.baseline.space
        return not any(
            np.any(offset[space.full_slices[block.name]])
            for block in space.resolved_blocks
            if block.name.startswith("model.") and block.name in self.space.blocks
        )

    def _key(self, v: Any) -> tuple:
        """Cache key of an input, computed without forming the full model.

        A vector on this space is identified by its own coordinates: the
        baseline, reference and space expanding it to Sauce's model are fixed
        for this instance (rebinding them resets the caches), so equal keys
        mean equal native inputs within a run. The zero-copy
        ``fs-block-sha256-1`` digest hashes the caller's buffer in parallel; a
        cache hit therefore costs one pass over the vector, no copies.
        """
        if isinstance(v, ControlVector):
            if v.space is not self.space and not v.space.equivalent(self.space):
                return ("model", block_digest(self._full(v) - self.reference))
            v = v.values
        values = np.asarray(v)
        if np.iscomplexobj(values):
            raise ValueError("control vectors are real; complex blocks interleave")
        return ("active", block_digest(values.reshape(-1)))

    def _run(
        self,
        operation: str,
        full: Union[np.ndarray, ControlVectorFile],
        *,
        context: Optional[Path],
        tau: float = 1.0,
        metric: Optional[Union[Path, ControlVectorFile]] = None,
        lower: Optional[Union[Path, ControlVectorFile]] = None,
        upper: Optional[Union[Path, ControlVectorFile]] = None,
    ) -> RegularizationJob:
        spec = self.regularization
        job = RegularizationJob(
            self.source,
            smoothing=spec.smoothing,
            input_vector=(
                full if isinstance(full, ControlVectorFile) else self._model_file(full)
            ),
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

    @_shares_arrays
    def value(self, v: Any) -> float:
        key = self._key(v)
        if key not in self._value_cache:
            full = self._full(v) - self.reference
            if self._at_reference(full):
                if self.is_smooth:
                    self._gradient_cache = {key: np.zeros(self.space.size)}
                self._value_cache[key] = 0.0
            else:
                job = self._run(
                    "gradient" if self.is_smooth else "value",
                    full,
                    context=self.context,
                )
                if self.is_smooth:
                    self._gradient_cache = {key: self._read_gradient(job)}
                self._value_cache[key] = self._read_value(job)
        return self.factor * self._value_cache[key]

    @property
    def is_smooth(self) -> bool:
        """Tikhonov has an exact native quadratic gradient; TV/TGV use prox."""
        return self.regularization.smoothing.kind == "tikhonov"

    def _read_gradient(self, job: RegularizationJob) -> np.ndarray:
        result = ControlVectorFile.read(job.gradient_file(), native=True)
        gradient = np.zeros(self.space.size)
        for block in self.space.resolved_blocks:
            if block.name.startswith("model."):
                gradient[self.space.slices[block.name]] = result[block.name][
                    self.space._mask_of(block)
                ]
        return gradient

    def gradient(self, v: Any) -> ControlVector:
        if not self.is_smooth:
            raise ValueError("TV/TGV require their proximal callback")
        key = self._key(v)
        if key not in self._gradient_cache:
            full = self._full(v) - self.reference
            if self._at_reference(full):
                self._value_cache[key] = 0.0
                self._gradient_cache = {key: np.zeros(self.space.size)}
            else:
                job = self._run("gradient", full, context=self.context)
                self._value_cache[key] = self._read_value(job)
                self._gradient_cache = {key: self._read_gradient(job)}
        return self._wrap(self.factor * self._gradient_cache[key])

    def hessian_operator(self, v: Any = None) -> Any:
        if not self.is_smooth:
            raise ValueError("TV/TGV require their proximal callback")

        def action(direction: np.ndarray) -> np.ndarray:
            # Tangents have zero fixed entries, unlike full model values.
            full = self._full(direction) - self._tangent_origin
            job = self._run("gradient", full, context=self.context)
            self._read_value(job)
            return self.factor * self._read_gradient(job)

        return _SymmetricModelOperator(action, self.space)

    def curvature_diagonal(self, v: Any = None) -> np.ndarray:
        if not self.is_smooth:
            raise ValueError("TV/TGV require their proximal callback")
        job = self._run(
            "diagonal", np.zeros_like(self.baseline.values), context=self.context
        )
        self._read_value(job)
        return self.factor * self._read_gradient(job)

    @functools.cached_property
    def _model_spaces(self) -> dict:
        """Basis identities of the native blocks, as :meth:`_model_file` records them."""
        return {
            block.name: block.basis_identity
            for block in self.baseline.space.resolved_blocks
            if block.name.startswith("model.")
            and block.name in self.space.blocks
            and block.basis_identity
        }

    def _proximal_inputs(
        self, metric: Any, bounds: tuple[np.ndarray, np.ndarray]
    ) -> dict[str, Path]:
        """Return the metric and bound files of :meth:`prox`, staged once per content.

        They depend only on ``metric`` and ``bounds`` (baseline, reference and
        space are fixed for this binding), so each distinct content is written
        once beside the stage's native context and every later prox job
        references it in place (a remote site receives one stable path).
        Read-only arrays are immutable inputs: passing the same ones again is
        free; other arrays are identified by their ``fs-block-sha256-1``
        digests (one parallel pass each).
        """
        arrays = tuple(np.asarray(a, dtype=float) for a in (metric, *bounds))
        if any(a.shape != (self.space.size,) for a in arrays):
            raise ValueError("proximal metric and bounds must match the control space")
        staged = self._proximal
        if staged is not None and not all(p.is_file() for p in staged[2].values()):
            staged = None
        if staged is not None and all(
            ref is not None and ref() is array for ref, array in zip(staged[0], arrays)
        ):
            return staged[2]
        digests = tuple(block_digest(array) for array in arrays)
        paths = (
            staged[2]
            if staged is not None and staged[1] == digests
            else self._stage_proximal(arrays, digests)
        )
        # Weak references never keep a caller's arrays alive.
        refs = tuple(
            None if array.flags.writeable else weakref.ref(array) for array in arrays
        )
        self._proximal = (refs, digests, paths)
        return paths

    def _stage_proximal(self, arrays: tuple, digests: tuple) -> dict[str, Path]:
        metric, lower, upper = arrays
        # Bounds are finite in the native files; inactive DOFs inside each
        # included block keep equal fixed bounds (their baseline values).
        limit = np.finfo(float).max / 100
        baseline = self.baseline.values
        blocks: dict[str, dict[str, np.ndarray]] = dict(metric={}, lower={}, upper={})
        for name, full, active, mask in self._model_layout:
            values = np.ones(mask.size)
            values[mask] = metric[active]
            blocks["metric"][name] = values
            for label, bound in (("lower", lower), ("upper", upper)):
                segment = baseline[full].copy()
                segment[mask] = np.clip(bound[active], -limit, limit)
                blocks[label][name] = np.clip(
                    segment - self.reference[full], -limit, limit
                )
        directory = self.context.parent
        return {
            label: RegularizationJob.stage_input(
                ControlVectorFile(values, control_spaces=self._model_spaces),
                directory / f"regularization_{label}_{digest.split(':')[-1][:16]}.h5",
            )
            for (label, values), digest in zip(blocks.items(), digests)
        }

    @_shares_arrays
    def prox(
        self,
        v: np.ndarray,
        tau: float,
        metric: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Solve Sauce's constrained proximal problem around ``v``.

        Only the target changes between calls: each call writes the native
        target, while the metric and bound files are staged once per content
        (:meth:`_proximal_inputs`).
        """
        values = np.asarray(v, dtype=float).reshape(-1)
        if values.size != self.space.size:
            raise ValueError(
                f"vector has {values.size} entries; the space has {self.space.size}"
            )
        # Fidelity targets may lie outside the box; only non-material coordinates
        # are projected here. Native material targets retain their full values.
        model = np.clip(values, bounds[0], bounds[1])
        baseline = self.baseline.values
        target = {}
        for name, full, active, mask in self._model_layout:
            segment = baseline[full].copy()
            segment[mask] = values[active]
            segment -= self.reference[full]
            target[name] = segment
        job = self._run(
            "proximal",
            ControlVectorFile(target, control_spaces=self._model_spaces),
            context=self.context,
            tau=tau * self.factor,
            **self._proximal_inputs(metric, bounds),
        )
        del target
        energy = self._read_value(job)
        result = ControlVectorFile.read(job.gradient_file(), native=True)
        for name, full, active, mask in self._model_layout:
            block = result[name] + self.reference[full]
            if not (np.isfinite(block.min()) and np.isfinite(block.max())):
                raise ValueError("control state values must be finite")
            model[active] = block[mask]
        # Sauce reports the energy of the returned model: the composite line
        # search's value() of this trial needs no separate value job.
        self._value_cache[self._key(model)] = energy
        return model


def bind_workflow_regularization(
    specification: Any, space: ControlSpace, problem: Any, linearization: Any
) -> tuple[Optional[BoundRegularization], Optional[BoundNativeRegularization]]:
    """Separate smooth user terms from one native proximal model term."""
    from .statistics import GaussianPrior

    if isinstance(specification, GaussianPrior):
        return (
            specification.bind(space, problem=problem, linearization=linearization),
            None,
        )
    native: Optional[BoundNativeRegularization] = None
    smooth: list[BoundRegularization] = []

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
            bound = (
                spec.bind(space, problem=problem, linearization=linearization)
                if isinstance(spec, GaussianPrior)
                else spec.bind(space)
            )
            smooth.append(
                bound
                if factor == 1
                else _BoundScaled(Scaled(spec, factor), space, bound, factor)
            )

    add(specification)
    smooth_bound = (
        None
        if not smooth
        else smooth[0] if len(smooth) == 1 else _BoundSum(specification, space, smooth)
    )
    return smooth_bound, native
