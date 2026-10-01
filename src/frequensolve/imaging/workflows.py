"""Imaging workflows: staged FWI, LSRTM, RTM, sensitivity kernels.

The workflow layer owns the outer loop of an inversion while
:class:`~frequensolve.imaging.problem.ImagingProblem` owns the physics:

- :class:`Stage` names the frequencies, active blocks and iteration budget of
  one continuation stage (plus optional loss, regularization, smoothing, optimizer
  and frequency-weight overrides) and produces the stage view of a problem.
- :class:`LBFGS` and :class:`NewtonCG` are thin configurations of the
  generic optimizers in :mod:`frequensolve.inversion.optimization`; they add
  an RMS step cap per block and an optional diagonal change of variables
  (per-block curvature scaling), both ported from the TCCS production driver.
- :class:`FWI` runs the stage sequence through
  :func:`~frequensolve.inversion.continuation.run_continuation`, threads the
  complete :class:`~frequensolve.imaging.controls.ControlState` through the
  stages (transferring it when a stage changes the control layout with
  ``Stage(controls=...)``), adopts support masks from each stage's first linearization,
  records every evaluation and iteration in an
  :class:`~frequensolve.inversion.history.OptimizationHistory`, checkpoints
  after every accepted iteration and resumes from a matching checkpoint.
- :class:`LSRTM`, :func:`rtm` and :func:`sensitivity_kernel` are the
  single-linearization workflows.

Sign convention: Sauce's covector is the gradient of the misfit, i.e. the
adjoint applied to ``simulated - observed``.  :func:`rtm` returns that
gradient; the classic RTM image in the ``observed - simulated`` convention is
its negative.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import glob
import hashlib
import inspect
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr as scipy_lsqr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.imaging._artifacts import (
    ImageSet,
    control_smoothing,
)
from frequensolve.imaging._backend import fingerprint
from frequensolve.imaging._block_digest import block_digest
from frequensolve.imaging._curvature_continuation import (
    _checkpoint_operator_configuration,
    _checkpoint_regularization_identity,
    _configuration_value,
    _CurvatureSource,
    _initial_scale,
    _prepare_seed,
    _regularization_identity,
    _require_smooth_regularization,
)
from frequensolve.imaging._curvature_transfer import (
    CurvatureTransfer,
    _history_inverse,
)
from frequensolve.imaging._restart_store import (
    RESTART_SCHEMA,
    RestartStore,
    StageFiles,
    state_files,
)
from frequensolve.imaging.controls import (
    ControlSpace,
    ControlState,
    ControlVector,
    _BlockSpec,
)
from frequensolve.imaging.data import DataVector, TraceStoreRef
from frequensolve.imaging.jobs import ImageKernelJob, ImageSpec, _kernel_derivative
from frequensolve.imaging.misfit import Loss, Misfit, Normalization
from frequensolve.imaging.problem import ImagingProblem, Linearization, _MisfitPayload
from frequensolve.imaging.results import FWIResult, StageResult
from frequensolve.imaging.transfer import Transfer, resolve_transfer
from frequensolve.inversion.continuation import (
    ContinuationSchedule,
    ContinuationStage,
    run_continuation,
)
from frequensolve.inversion.history import (
    LossTerms,
    OptimizationCheckpoint,
    OptimizationHistory,
    OptimizationRecord,
)
from frequensolve.inversion.optimization import (
    InexactNewtonIteration,
    InexactNewtonOptions,
    InexactNewtonResult,
    LBFGSOptions,
    LBFGSRestart,
    _read_only,
    minimize_inexact_newton,
    minimize_lbfgs,
    minimize_proximal_gradient,
)
from frequensolve.util.atomic import atomic_output_path

__all__ = [
    "FWI",
    "FWIIteration",
    "LBFGS",
    "LSRTM",
    "NewtonCG",
    "PatchUpdates",
    "Stage",
    "block_curvatures",
    "curvature_scaling",
    "rms_step_limit",
    "rtm",
    "sensitivity_kernel",
    "sensitivity_kernel_job",
]

from ._patch_updates import PatchUpdates

CHECKPOINT_SCHEMA = "fs-imaging-fwi-checkpoint-2"
# Earlier checkpoints kept optimizer restart state as JSON metadata.
_LEGACY_CHECKPOINT_SCHEMAS = ("fs-imaging-fwi-checkpoint-1",)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stage_frequencies(values: Any) -> Tuple[Any, ...]:
    """Normalize a stage frequency list (floats, or complex Laplace samples)."""

    if np.isscalar(values):
        values = [values]
    array = np.asarray(list(values))
    if array.size == 0:
        raise ValueError("a stage requires at least one frequency")
    out: List[Any] = []
    for value in array.reshape(-1):
        z = complex(value)
        if not math.isfinite(z.real) or not math.isfinite(z.imag):
            raise ValueError("stage frequencies must be finite")
        out.append(float(z.real) if z.imag == 0.0 else complex(z.real, -abs(z.imag)))
    if len({complex(v) for v in out}) != len(out):
        raise ValueError("stage frequencies must be unique")
    return tuple(out)


def _mechanism_scaling(problem: Any) -> Dict[str, float]:
    """Return a problem's mechanism reference scales (``{}`` when none)."""

    scaling = getattr(problem, "mechanism_scaling", None)
    return dict(scaling) if isinstance(scaling, Mapping) else {}


def _frequency_pairs(values: Sequence[Any]) -> List[List[float]]:
    return [[complex(v).real, complex(v).imag] for v in values]


def _state_epoch(state: ControlState) -> str:
    """Identify complete coefficients together with their basis and reference scales.

    The state's own coefficient blocks are hashed in place (``fingerprint``
    digests large arrays by their bytes), without a file representation copy.
    """
    space = state.space
    return fingerprint(
        blocks={name: state.values[sl] for name, sl in space.full_slices.items()},
        control_spaces={
            block.name: block.basis_identity
            for block in space.resolved_blocks
            if block.basis_identity
        },
        scaling=dict(state.scaling),
        scaling_units=dict(state.scaling_units),
    ).removeprefix("sha256:")


def _real_vector(value: Any, *, size: Optional[int] = None, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"{name} has size {array.size}; expected {size}")
    return array


def rms_step_limit(
    direction: Any, block_slices: Sequence[slice], maximum_rms: float
) -> float:
    """Return the largest step length keeping every block's RMS update below a cap.

    Ported from the TCCS driver's ``_rms_step_limit``: for each active block
    the RMS of the direction restricted to that block is compared with
    ``maximum_rms`` (a size-independent update magnitude in optimizer
    coordinates).  Returns ``inf`` for a zero direction.
    """

    maximum = float(maximum_rms)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("step_limit must be finite and positive")
    values = np.asarray(direction, dtype=np.float64).reshape(-1)
    limit = math.inf
    for block_slice in block_slices:
        block = values[block_slice]
        if block.size == 0:
            continue
        rms = float(np.linalg.norm(block) / math.sqrt(block.size))
        if rms > 0.0:
            limit = min(limit, maximum / rms)
    return limit


def block_curvatures(
    linearization: Linearization,
    *,
    seed: int = 0,
    blocks: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Estimate the mean data curvature of each block with one Rademacher JVP.

    For block ``b`` with ``n_b`` active DOFs and a random ``±1`` direction
    ``z_b`` supported on the block, ``c_b = <J z_b, W J z_b> / n_b``
    approximates the mean diagonal of ``J^H W J`` over the block (TCCS
    ``estimate_block_curvatures``).  Blocks not listed in ``blocks`` receive
    a tiny positive curvature.  Returns ``qualified block -> curvature``.
    """

    space = linearization.space
    rng = np.random.default_rng(seed)
    enabled = None if blocks is None else {space.block(b).name for b in blocks}
    tiny = float(np.finfo(np.float64).tiny)
    estimates: Dict[str, float] = {}
    for name, block_slice in space.slices.items():
        size = block_slice.stop - block_slice.start
        if size == 0 or (enabled is not None and name not in enabled):
            estimates[name] = tiny
            continue
        direction = np.zeros(space.size, dtype=np.float64)
        direction[block_slice] = rng.choice((-1.0, 1.0), size=size)
        increment = linearization.jacobian @ ControlVector(direction, space)
        weighted = linearization.weight_data(increment)
        curvature = float(np.real(np.vdot(increment.values, weighted.values))) / size
        estimates[name] = max(curvature, tiny)
    return estimates


def curvature_scaling(
    curvatures: Mapping[str, float],
    space: ControlSpace,
    *,
    max_ratio: Optional[float] = None,
) -> np.ndarray:
    """Return the per-DOF change-of-variables scale from block curvatures.

    With ``x = S y`` and ``S = diag(1 / sqrt(c_b))`` the scaled Hessian
    ``S H S`` has unit mean curvature per block.  ``max_ratio`` floors every
    curvature at ``max(c) / max_ratio`` (TCCS ``_block_curvature_scales``)
    so poorly illuminated blocks are not amplified without bound.  Blocks
    missing from ``curvatures`` keep unit scale.
    """

    table = {space.block(name).name: float(value) for name, value in curvatures.items()}
    values = np.asarray(list(table.values()), dtype=np.float64)
    if values.size and (np.any(~np.isfinite(values)) or np.any(values <= 0.0)):
        raise ValueError("block curvatures must be finite and positive")
    if max_ratio is not None and values.size:
        ratio = float(max_ratio)
        if not math.isfinite(ratio) or ratio < 1.0:
            raise ValueError("max_ratio must be finite and at least one")
        floor = float(np.max(values)) / ratio
        table = {name: max(value, floor) for name, value in table.items()}
    scale = np.ones(space.size, dtype=np.float64)
    for name, block_slice in space.slices.items():
        if name in table:
            scale[block_slice] = 1.0 / math.sqrt(table[name])
    return scale


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One continuation stage of an inversion.

    Args:
        frequencies: Frequencies (real or complex Laplace samples) solved in
            the stage; must be a subset of the problem's frequencies.
        iterations: Accepted optimizer iterations budget for the stage.
        active: Block keys, addresses or qualified names activated by the
            stage; ``None`` activates the problem's blocks.
        loss: Replace the loss of every misfit term for the stage
            (``"huber"``, :class:`~frequensolve.imaging.misfit.Loss`, ...).
        misfit: Replace the whole misfit for the stage (exclusive with
            ``loss``).
        regularization: Regularization overriding the workflow-level regularization.
        smoothing: Gradient-output and search-step smoothing overriding the workflow/problem
            smoothing; ``False`` switches smoothing off for the stage.
        optimizer: :class:`LBFGS` / :class:`NewtonCG` overriding the
            workflow optimizer.
        weights: One nonnegative objective weight per stage frequency.
        patches: PatchSet overriding the problem's patch policy for this stage.
        name: Stage label (default ``stage_<index>``).
        min_support: Relative support threshold for the stage.
        metadata: Free-form scalar metadata recorded with the stage.
        mesh_averaging_wavelengths: Half-width of the material-aware slowness
            averaging window for explicit mesh changes, in material-average
            wavelengths at the largest requested control sizing frequency.
        transfer: ``im.Transfer.nodal()`` (default) or
            ``im.Transfer.l2(smooth=100 * u.m)`` for explicit mesh changes.
            Smoothing lengths accept length quantities or numbers in meters.
        mesh_transfer: Legacy spelling: ``"nodal"`` or ``"l2"``.
        mesh_smoothing_length: L2 projection smoothing length in meters (zero
            for unsmoothed projection). Applied once at the stage boundary.
        controls: Change the control layout from this stage on: a complete
            :class:`~frequensolve.imaging.controls.ControlSpace` or a mapping
            ``{block key: new block spec}`` replacing those blocks (e.g.
            ``{"vp": im.DepthProfile("vp", "sediment", spacing=25*u.m)}``).
            :class:`FWI` builds the stage problem with
            :meth:`ImagingProblem.with_controls`, transferring the accepted
            state of the previous stage; later stages keep the new layout
            until another stage changes it.
        kernel_derivative: Spectral-data selection for this stage, using
            ``residual="derivative"`` or ``"window"``. Defaults to the problem's selection.
    """

    frequencies: Tuple[Any, ...]
    iterations: int
    active: Optional[Tuple[str, ...]] = None
    loss: Optional[Loss] = None
    misfit: Optional[Misfit] = None
    regularization: Any = None
    smoothing: Any = None
    optimizer: Any = None
    weights: Optional[Tuple[float, ...]] = None
    name: Optional[str] = None
    min_support: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    controls: Any = None
    mesh_averaging_wavelengths: float = 0.5
    mesh_transfer: Optional[str] = None
    mesh_smoothing_length: Optional[float] = None
    kernel_derivative: Optional[Mapping[str, Any]] = None
    transfer: Optional[Transfer] = None
    patches: Any = None
    curvature: Optional[CurvatureTransfer] = None

    def __post_init__(self) -> None:
        if self.curvature is not None and not isinstance(
            self.curvature, CurvatureTransfer
        ):
            raise TypeError("stage curvature must be a CurvatureTransfer policy")
        if self.patches is not None:
            from frequensolve.mesh.patches import PatchSet

            if not isinstance(self.patches, PatchSet):
                raise TypeError("stage patches must be a PatchSet")
        policy = resolve_transfer(
            self.transfer, self.mesh_transfer, self.mesh_smoothing_length
        )
        if self.controls is None and (
            policy.method != "nodal" or policy.smoothing_length
        ):
            raise ValueError("mesh transfer options require stage controls")
        if self.kernel_derivative is not None:
            object.__setattr__(
                self,
                "kernel_derivative",
                _kernel_derivative(
                    self.kernel_derivative, residuals=("derivative", "window")
                ),
            )
        if (
            not np.isfinite(self.mesh_averaging_wavelengths)
            or self.mesh_averaging_wavelengths <= 0
        ):
            raise ValueError("mesh_averaging_wavelengths must be finite and positive")
        object.__setattr__(self, "frequencies", _stage_frequencies(self.frequencies))
        iterations = int(self.iterations)
        if iterations < 1:
            raise ValueError("stage iterations must be positive")
        object.__setattr__(self, "iterations", iterations)
        active = self.active
        if active is not None:
            names = [active] if isinstance(active, str) else list(active)
            if not names:
                raise ValueError("stage active must name at least one block")
            object.__setattr__(self, "active", tuple(str(n) for n in names))
        if self.loss is not None and self.misfit is not None:
            raise ValueError("pass either loss or misfit to a Stage, not both")
        if self.loss is not None:
            object.__setattr__(self, "loss", Loss.from_value(self.loss))
        if self.misfit is not None and not isinstance(self.misfit, Misfit):
            raise TypeError("stage misfit must be a Misfit")
        if self.smoothing is not None and self.smoothing is not False:
            object.__setattr__(self, "smoothing", control_smoothing(self.smoothing))
        if self.optimizer is not None and not callable(
            getattr(self.optimizer, "solve", None)
        ):
            raise TypeError("stage optimizer must provide solve()")
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
            if weights.size != len(self.frequencies):
                raise ValueError("stage weights must match the stage frequencies")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("stage weights must be finite and nonnegative")
            object.__setattr__(self, "weights", tuple(float(w) for w in weights))
        if self.name is not None:
            name = str(self.name).strip()
            if not name:
                raise ValueError("stage name cannot be empty")
            object.__setattr__(self, "name", name)
        if self.min_support is not None:
            threshold = float(self.min_support)
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError("min_support must be finite and non-negative")
            object.__setattr__(self, "min_support", threshold)
        object.__setattr__(self, "metadata", dict(self.metadata))
        controls = self.controls
        if controls is not None:
            if isinstance(controls, Mapping) and not isinstance(controls, ControlSpace):
                if not controls:
                    raise ValueError("stage controls mapping cannot be empty")
                if not all(isinstance(spec, _BlockSpec) for spec in controls.values()):
                    raise TypeError(
                        "stage controls mapping values must be block specs "
                        "(DepthProfile, GridParameters, ...)"
                    )
                controls = {str(key): spec for key, spec in controls.items()}
            elif isinstance(controls, _BlockSpec):
                controls = ControlSpace(controls)
            elif not isinstance(controls, ControlSpace):
                raise TypeError(
                    "stage controls must be a ControlSpace or a {key: block} mapping"
                )
            object.__setattr__(self, "controls", controls)

    # -- constructors ---------------------------------------------------------

    @classmethod
    def bands(
        cls,
        bands: Sequence[Any],
        iterations: Union[int, Sequence[int]],
        **common: Any,
    ) -> List["Stage"]:
        """Return one stage per frequency band.

        ``iterations`` is one budget for every band or one per band; every
        other keyword is passed to each :class:`Stage`.
        """

        bands = list(bands)
        if not bands:
            raise ValueError("Stage.bands requires at least one band")
        if np.isscalar(iterations):
            budgets = [int(iterations)] * len(bands)  # type: ignore[arg-type]
        else:
            budgets = [int(v) for v in iterations]  # type: ignore[union-attr]
            if len(budgets) != len(bands):
                raise ValueError("iterations must match the number of bands")
        return [cls(band, budget, **common) for band, budget in zip(bands, budgets)]

    @classmethod
    def from_schedule(
        cls,
        schedule: ContinuationSchedule,
        iterations: Optional[Union[int, Sequence[Optional[int]]]] = None,
        **common: Any,
    ) -> List["Stage"]:
        """Wrap a :class:`ContinuationSchedule` as stages.

        ``iterations`` (scalar or per stage) overrides the schedule's
        ``max_iterations``; a stage without either raises.
        """

        if not isinstance(schedule, ContinuationSchedule):
            raise TypeError("schedule must be a ContinuationSchedule")
        stages = list(schedule.stages)
        if iterations is None:
            budgets: List[Optional[int]] = [s.max_iterations for s in stages]
        elif np.isscalar(iterations):
            budgets = [int(iterations)] * len(stages)  # type: ignore[arg-type]
        else:
            budgets = [None if v is None else int(v) for v in iterations]  # type: ignore[union-attr]
            if len(budgets) != len(stages):
                raise ValueError("iterations must match the number of stages")
        shared_metadata = dict(common.pop("metadata", {}))
        out = []
        for stage, budget in zip(stages, budgets):
            if budget is None:
                budget = stage.max_iterations
            if budget is None:
                raise ValueError(
                    f"continuation stage {stage.name!r} has no iteration budget"
                )
            out.append(
                cls(
                    stage.frequencies,
                    budget,
                    name=stage.name,
                    metadata={**dict(stage.metadata), **shared_metadata},
                    **common,
                )
            )
        return out

    @classmethod
    def frequency_laplace_bands(
        cls,
        bands: Sequence[Mapping[str, Any]],
        iterations: Optional[Union[int, Sequence[Optional[int]]]] = None,
        *,
        damping_sign: float = -1.0,
        **common: Any,
    ) -> List["Stage"]:
        """Expand frequency/Laplace bands into stages.

        See :meth:`ContinuationSchedule.frequency_laplace_bands`; each band
        may carry ``max_iterations`` which ``iterations`` overrides.
        """

        schedule = ContinuationSchedule.frequency_laplace_bands(
            bands, damping_sign=damping_sign
        )
        return cls.from_schedule(schedule, iterations, **common)

    @staticmethod
    def alternate(stages: Sequence["Stage"], rounds: int) -> List["Stage"]:
        """Repeat ``stages`` ``rounds`` times (variable projection, joint updates).

        Named stages receive a ``_r<round>`` suffix so labels stay unique.
        """

        rounds = int(rounds)
        if rounds < 1:
            raise ValueError("rounds must be positive")
        stages = list(stages)
        if not stages or not all(isinstance(s, Stage) for s in stages):
            raise TypeError("alternate requires a non-empty sequence of Stage")
        out: List[Stage] = []
        for round_index in range(1, rounds + 1):
            for stage in stages:
                name = None if stage.name is None else f"{stage.name}_r{round_index}"
                out.append(
                    dataclasses.replace(
                        stage,
                        name=name,
                        metadata={**stage.metadata, "round": round_index},
                    )
                )
        return out

    # -- views ----------------------------------------------------------------

    def label(self, index: int = 0) -> str:
        """Return the stage name or ``stage_<index + 1>``."""

        return self.name if self.name is not None else f"stage_{index + 1:02d}"

    def view(self, problem: ImagingProblem) -> ImagingProblem:
        """Return the restricted problem view of this stage.

        The view selects the stage frequencies and active blocks, applies the
        misfit/loss, smoothing, support threshold and weight overrides and
        adopts fresh support masks from its own first linearization
        (spec §4.1.1).
        """

        kwargs: Dict[str, Any] = {}
        if self.misfit is not None:
            kwargs["misfit"] = self.misfit
        elif self.loss is not None:
            kwargs["loss"] = self.loss
        if self.smoothing is not None:
            kwargs["smoothing"] = None if self.smoothing is False else self.smoothing
        if self.min_support is not None:
            kwargs["min_support"] = self.min_support
        if self.weights is not None:
            kwargs["weights"] = self.weights
        if self.kernel_derivative is not None:
            kwargs["kernel_derivative"] = self.kernel_derivative
        if self.patches is not None:
            kwargs["patches"] = self.patches
        return problem.restrict(
            frequencies=self.frequencies,
            active=None if self.active is None else list(self.active),
            support="refresh",
            **kwargs,
        )

    def to_continuation_stage(self, index: int = 0) -> ContinuationStage:
        """Return the :class:`ContinuationStage` used by :func:`run_continuation`."""

        metadata: Dict[str, Any] = {
            **self.metadata,
            "stage_index": int(index),
        }
        if self.active is not None:
            metadata["active"] = list(self.active)
        if self.controls is not None:
            metadata["controls"] = (
                sorted(self.controls)
                if isinstance(self.controls, Mapping)
                else list(self.controls.keys)
            )
        return ContinuationStage(
            self.label(index),
            tuple(complex(v) for v in self.frequencies),
            max_iterations=self.iterations,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Stage(frequencies={list(self.frequencies)}, iterations={self.iterations}, "
            f"active={None if self.active is None else list(self.active)}, "
            f"name={self.name!r})"
        )


def _stage_list(stages: Any) -> List[Stage]:
    if isinstance(stages, Stage):
        return [stages]
    if isinstance(stages, ContinuationSchedule):
        return Stage.from_schedule(stages)
    out = list(stages)
    if not out:
        raise ValueError("FWI requires at least one stage")
    for stage in out:
        if not isinstance(stage, Stage):
            raise TypeError("stages must be Stage objects")
    return out


# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------


def _preconditioner_callable(
    preconditioner: Any, space: Optional[ControlSpace]
) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Adapt a bound preconditioner or plain callable to ``(model, g) -> ndarray``."""

    if preconditioner is None:
        return None
    apply = getattr(preconditioner, "apply", None)
    if callable(apply):

        def bound(_model: np.ndarray, g: np.ndarray) -> np.ndarray:
            vector = g if space is None else ControlVector(g, space)
            return np.asarray(apply(vector), dtype=np.float64).reshape(-1)

        return bound
    if callable(preconditioner):
        return preconditioner
    raise TypeError("preconditioner must be callable or provide apply()")


@dataclass(frozen=True)
class _OptimizerConfig:
    """Shared options of :class:`LBFGS` and :class:`NewtonCG`.

    Gradient and objective-decrease thresholds are the corresponding
    ``absolute_tolerance + relative_tolerance * abs(initial_objective)``.
    Absolute defaults are zero; relative defaults are ``1e-6`` (gradient)
    and ``1e-9`` (objective decrease). FWI retains the initial stage objective
    through checkpoint restarts. See :class:`InexactNewtonOptions` for the
    legacy tolerance spellings.

    ``step_limit`` caps the RMS update of every block per iteration
    (optimizer coordinates); ``scaling_max_ratio`` floors curvature-based
    scaling (see :func:`curvature_scaling`); ``preconditioner_refresh``
    re-estimates a bound preconditioner every that many accepted iterations
    inside :class:`FWI`.
    """

    max_iterations: Optional[int] = None
    gradient_tolerance: Optional[float] = None
    step_tolerance: float = 1.0e-9
    objective_tolerance: Optional[float] = None
    grad_abs_tol: float = 0.0
    grad_rel_tole: Optional[float] = None
    obj_abs_tol: float = 0.0
    obj_rel_tol: Optional[float] = None
    objective_target: Optional[float] = None
    objective_tolerance_momentum: float = 0.0
    objective_minimum_iterations: int = 1
    max_objective_evaluations: Optional[int] = None
    max_line_search_trials: Optional[int] = None
    armijo_constant: float = 1.0e-4
    backtrack_factor: float = 0.5
    minimum_step_length: float = 1.0e-8
    curvature_tolerance: float = 1.0e-14
    bound_tolerance: float = 1.0e-12
    step_limit: Optional[float] = None
    scaling_max_ratio: Optional[float] = None
    preconditioner_refresh: Optional[int] = None

    kind: str = dataclasses.field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_iterations is not None and int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be positive when supplied")
        if self.step_limit is not None:
            limit = float(self.step_limit)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("step_limit must be finite and positive")
            object.__setattr__(self, "step_limit", limit)
        if (
            self.preconditioner_refresh is not None
            and int(self.preconditioner_refresh) < 1
        ):
            raise ValueError("preconditioner_refresh must be positive when supplied")

    def _common(self, max_iterations: Optional[int]) -> Dict[str, Any]:
        limit = max_iterations if max_iterations is not None else self.max_iterations
        return {
            "max_iterations": 50 if limit is None else int(limit),
            "max_objective_evaluations": self.max_objective_evaluations,
            "max_line_search_trials": self.max_line_search_trials,
            "gradient_tolerance": self.gradient_tolerance,
            "step_tolerance": self.step_tolerance,
            "objective_tolerance": self.objective_tolerance,
            "grad_abs_tol": self.grad_abs_tol,
            "grad_rel_tole": self.grad_rel_tole,
            "obj_abs_tol": self.obj_abs_tol,
            "obj_rel_tol": self.obj_rel_tol,
            "objective_target": self.objective_target,
            "objective_tolerance_momentum": self.objective_tolerance_momentum,
            "objective_minimum_iterations": self.objective_minimum_iterations,
            "armijo_constant": self.armijo_constant,
            "backtrack_factor": self.backtrack_factor,
            "minimum_step_length": self.minimum_step_length,
            "curvature_tolerance": self.curvature_tolerance,
            "bound_tolerance": self.bound_tolerance,
        }

    def options(self, max_iterations: Optional[int] = None) -> Any:
        """Return the :mod:`frequensolve.inversion.optimization` options."""

        raise NotImplementedError  # pragma: no cover - subclasses

    def solve(
        self,
        objective: Any,
        x0: Any,
        *,
        bounds: Optional[Tuple[Any, Any]] = None,
        preconditioner: Any = None,
        hessian_action: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        step_transform: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        history: Optional[OptimizationHistory] = None,
        callback: Optional[Callable[[InexactNewtonIteration], None]] = None,
        max_iterations: Optional[int] = None,
        initial_objective: Optional[float] = None,
        step_limit: Optional[float] = None,
        scaling: Optional[Any] = None,
        block_slices: Optional[Sequence[slice]] = None,
        space: Optional[ControlSpace] = None,
        restart: Optional[LBFGSRestart] = None,
    ) -> InexactNewtonResult:
        """Minimize ``objective`` from ``x0``.

        Args:
            objective: Object with ``value(x) -> float`` and
                ``gradient(x) -> ndarray`` (optionally ``loss(x)`` for history
                records and ``hessian_action(x, dx)`` for Newton-CG).
            x0: Initial point in optimizer coordinates.
            bounds: ``(lower, upper)`` arrays; ``space.bounds`` when a space
                is given and ``bounds`` is omitted.
            preconditioner: ``(model, g) -> ndarray`` callable or an object
                with ``apply(g)`` (a bound imaging preconditioner).
            hessian_action: ``(model, dx) -> ndarray`` (Newton-CG only);
                defaults to ``objective.hessian_action``.
            step_transform: Optional ``(model, step) -> step`` in physical
                optimizer coordinates. Applied outside CG; line searches
                check the transformed step against the raw gradient.
            history: Optional history receiving one iteration record per
                accepted iterate (``objective.loss`` required).
            callback: Called with every accepted
                :class:`InexactNewtonIteration` (physical coordinates).
            max_iterations: Iteration budget overriding the configuration.
            initial_objective: Original stage objective on restart; relative
                stopping tolerances retain this fixed reference.
            step_limit: RMS step cap overriding the configuration.
            scaling: Per-DOF change-of-variables scale ``s`` (``x = s * y``)
                or a ``block -> curvature`` mapping resolved with ``space``.
            block_slices: Block slices for the RMS step cap (``space.slices``
                when a space is given).
            space: Control space of ``x0`` (typed preconditioners, bounds,
                slices, curvature scaling).
        """

        x0 = _real_vector(x0, name="initial model")
        size = x0.size
        if bounds is None and space is not None:
            bounds = space.bounds
        if block_slices is None:
            block_slices = (
                tuple(space.slices.values()) if space is not None else (slice(0, size),)
            )
        limit = self.step_limit if step_limit is None else float(step_limit)
        if isinstance(scaling, Mapping):
            if space is None:
                raise ValueError("curvature scaling by block requires a space")
            scaling = curvature_scaling(
                scaling, space, max_ratio=self.scaling_max_ratio
            )
        scale = None
        if scaling is not None:
            scale = _real_vector(scaling, size=size, name="scaling")
            if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
                raise ValueError("scaling must be finite and positive")
        value_fn = objective.value
        gradient_fn = objective.gradient
        curvature_fn = hessian_action
        if curvature_fn is None:
            curvature_fn = getattr(objective, "hessian_action", None)
        precondition = _preconditioner_callable(preconditioner, space)

        def to_physical(y: np.ndarray) -> np.ndarray:
            return y if scale is None else scale * y

        def to_scaled(x: np.ndarray) -> np.ndarray:
            return x if scale is None else x / scale

        def f(y: np.ndarray) -> float:
            return float(value_fn(to_physical(y)))

        def g(y: np.ndarray) -> np.ndarray:
            grad = _real_vector(gradient_fn(to_physical(y)), size=size, name="gradient")
            return grad if scale is None else scale * grad

        def h(y: np.ndarray, dy: np.ndarray) -> np.ndarray:
            assert curvature_fn is not None
            out = _real_vector(
                curvature_fn(to_physical(y), to_physical(dy)),
                size=size,
                name="curvature product",
            )
            return out if scale is None else scale * out

        def p(y: np.ndarray, r: np.ndarray) -> np.ndarray:
            assert precondition is not None
            if scale is None:
                return _real_vector(
                    precondition(y, r), size=size, name="preconditioned"
                )
            out = precondition(to_physical(y), r / scale)
            return _real_vector(out, size=size, name="preconditioned") / scale

        def step_limit_fn(y: np.ndarray, dy: np.ndarray) -> float:
            assert limit is not None
            return rms_step_limit(to_physical(dy), block_slices, limit)

        def transform(y: np.ndarray, dy: np.ndarray) -> np.ndarray:
            assert step_transform is not None
            transformed = _real_vector(
                step_transform(to_physical(y), to_physical(dy)),
                size=size,
                name="transformed step",
            )
            return to_scaled(transformed)

        def unscale_iteration(it: InexactNewtonIteration) -> InexactNewtonIteration:
            if scale is None:
                return it
            # Read-only like the optimizer's own records, scaled or not.
            return dataclasses.replace(
                it,
                model=_read_only(to_physical(it.model)),
                gradient=_read_only(it.gradient / scale),
                linearization_gradient=_read_only(it.linearization_gradient / scale),
                step=_read_only(to_physical(it.step)),
                raw_step=_read_only(to_physical(it.raw_step)),
            )

        def on_iteration(it: InexactNewtonIteration) -> None:
            physical = unscale_iteration(it)
            if history is not None:
                loss_fn = getattr(objective, "loss", None)
                loss = (
                    loss_fn(physical.model)
                    if callable(loss_fn)
                    else LossTerms(data=physical.objective)
                )
                history.record_iteration(
                    physical.model,
                    loss,
                    gradient_norm=float(np.linalg.norm(physical.gradient)),
                    step_norm=float(np.linalg.norm(physical.step)),
                    step_length=physical.step_length,
                    metrics={"optimizer": self.kind},
                )
            if callback is not None:
                callback(physical)

        scaled_bounds = None
        if bounds is not None:
            lower = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), (size,))
            upper = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), (size,))
            scaled_bounds = (to_scaled(lower), to_scaled(upper))
        options = self.options(max_iterations)
        if initial_objective is not None:
            options = dataclasses.replace(options, initial_objective=initial_objective)
        kwargs: Dict[str, Any] = {
            "bounds": scaled_bounds,
            "options": options,
            "preconditioner": None if precondition is None else p,
            "step_limit": None if limit is None else step_limit_fn,
            "step_transform": None if step_transform is None else transform,
            "callback": on_iteration,
        }
        y0 = cast(Sequence[float], to_scaled(x0))
        if restart is not None:
            # Preserve the exact saved iterate: dividing S*y by S can round
            # differently even after its physical checkpoint was validated.
            restarted = _real_vector(restart.model, size=size, name="restart model")
            if not np.array_equal(to_physical(restarted), x0):
                raise ValueError(
                    "L-BFGS restart belongs to a different model or scaling"
                )
            y0 = cast(Sequence[float], restarted)
        native = getattr(objective, "native_regularization", None)
        if native is not None:
            if step_transform is not None:
                raise ValueError(
                    "native proximal optimization does not accept an extra step transform"
                )
            # Read-only and built once: the native term stages its metric and
            # bound files once for the whole solve instead of once per prox.
            metric = _read_only(np.ones(size) if scale is None else 1.0 / scale**2)
            boxes: Dict[str, Any] = {}

            def prox(
                y: np.ndarray, tau: float, box: Tuple[np.ndarray, np.ndarray]
            ) -> np.ndarray:
                if boxes.get("box") is not box:
                    boxes.update(
                        box=box,
                        physical=tuple(_read_only(to_physical(b)) for b in box),
                    )
                return to_scaled(
                    native.prox(to_physical(y), tau, metric, boxes["physical"])
                )

            result = minimize_proximal_gradient(
                lambda y: objective.smooth_value(to_physical(y)),
                g,
                lambda y: native.value(to_physical(y)),
                prox,
                y0,
                bounds=scaled_bounds,
                options=options,
                callback=on_iteration,
                step_limit=None if limit is None else step_limit_fn,
            )
        elif self.kind == "lbfgs":
            result = minimize_lbfgs(f, g, y0, restart=restart, **kwargs)
        else:
            if curvature_fn is None:
                raise ValueError("NewtonCG requires a hessian_action")
            result = minimize_inexact_newton(f, g, h, y0, **kwargs)
        if scale is None:
            return result
        return dataclasses.replace(
            result, model=to_physical(result.model), gradient=result.gradient / scale
        )


@dataclass(frozen=True)
class LBFGS(_OptimizerConfig):
    """Projected L-BFGS configuration (see :func:`minimize_lbfgs`).

    Args:
        memory: Number of curvature pairs kept (``history_size``).
        max_iterations: Default iteration budget (a stage's ``iterations``
            overrides it).
        step_limit: RMS step cap per block in optimizer coordinates.
        scaling_max_ratio: Curvature floor ratio for block scaling.
        preconditioner_refresh: Re-estimate a bound preconditioner every
            that many accepted iterations.
    """

    memory: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.memory) < 1:
            raise ValueError("memory must be positive")
        object.__setattr__(self, "memory", int(self.memory))
        object.__setattr__(self, "kind", "lbfgs")

    def options(self, max_iterations: Optional[int] = None) -> LBFGSOptions:
        return LBFGSOptions(history_size=self.memory, **self._common(max_iterations))


@dataclass(frozen=True)
class NewtonCG(_OptimizerConfig):
    """Inexact (Gauss-)Newton-CG configuration (see :func:`minimize_inexact_newton`).

    Args:
        max_cg_iterations: Truncation limit of the inner CG solve.
        initial_forcing, minimum_forcing, maximum_forcing: Eisenstat-Walker
            forcing controls.
    """

    max_cg_iterations: Optional[int] = 20
    initial_forcing: float = 0.7
    minimum_forcing: float = 1.0e-6
    maximum_forcing: float = 0.7

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "kind", "newton_cg")

    def options(self, max_iterations: Optional[int] = None) -> InexactNewtonOptions:
        return InexactNewtonOptions(
            max_cg_iterations=self.max_cg_iterations,
            initial_forcing=self.initial_forcing,
            minimum_forcing=self.minimum_forcing,
            maximum_forcing=self.maximum_forcing,
            **self._common(max_iterations),
        )


# ---------------------------------------------------------------------------
# stage objective
# ---------------------------------------------------------------------------


class _StageObjective:
    """Cached misfit-plus-regularization objective of one stage.

    One ``linearize`` (value and covector) per distinct point; the value,
    gradient, curvature action (Gauss-Newton normal plus regularization curvature) and
    loss terms of the last point are reused.  Every new evaluation is
    appended to the history with the stage metrics.
    """

    def __init__(
        self,
        view: ImagingProblem,
        space: ControlSpace,
        regularization: Any,
        history: Optional[OptimizationHistory],
        metrics: Mapping[str, Any],
    ) -> None:
        self.view = view
        self.space = space
        self.regularization = regularization
        self.native_regularization: Any = None
        self.history = history
        self.metrics = dict(metrics)
        self.evaluations = 0
        self._point: Optional[np.ndarray] = None
        self._linearization: Optional[Linearization] = None
        self._loss: Optional[LossTerms] = None
        self._gradient: Optional[np.ndarray] = None

    def vector(self, x: Any) -> ControlVector:
        return ControlVector(
            _real_vector(x, size=self.space.size, name="model"), self.space
        )

    def _evaluate(self, x: Any) -> None:
        point = _real_vector(x, size=self.space.size, name="model")
        if self._point is not None and np.array_equal(point, self._point):
            return
        vector = ControlVector(point, self.space)
        lin = self.view.linearize(vector, gradient=True)
        if lin.gradient is None:  # pragma: no cover - defensive
            raise RuntimeError("linearize returned no gradient")
        gradient = np.array(lin.gradient.values, dtype=np.float64, copy=True)
        regularization = 0.0
        if self.regularization is not None:
            regularization = float(self.regularization.value(vector))
            gradient += _real_vector(
                self.regularization.gradient(vector),
                size=point.size,
                name="regularization gradient",
            )
        if self.native_regularization is not None:
            regularization += self.native_regularization.value(vector)
        loss = LossTerms(data=lin.value, regularization=regularization)
        self._point = np.array(point, copy=True)
        self._linearization = lin
        self._loss = loss
        self._gradient = gradient
        self.evaluations += 1
        if self.history is not None:
            self.history.record_evaluation(
                point,
                loss,
                gradient_norm=float(np.linalg.norm(gradient)),
                metrics=self.metrics,
            )

    def linearization(self, x: Any) -> Linearization:
        self._evaluate(x)
        assert self._linearization is not None
        return self._linearization

    def value(self, x: Any) -> float:
        return self.loss(x).total

    def smooth_value(self, x: Any) -> float:
        loss = self.loss(x)
        return (
            loss.total
            if self.native_regularization is None
            else loss.total - self.native_regularization.value(self.vector(x))
        )

    def gradient(self, x: Any) -> np.ndarray:
        self._evaluate(x)
        assert self._gradient is not None
        return np.array(self._gradient, copy=True)

    def loss(self, x: Any) -> LossTerms:
        self._evaluate(x)
        assert self._loss is not None
        return self._loss

    def hessian_action(self, x: Any, dx: Any) -> np.ndarray:
        lin = self.linearization(x)
        direction = self.vector(dx)
        out = np.array(
            np.asarray(lin.normal @ direction, dtype=np.float64).reshape(-1), copy=True
        )
        if self.regularization is not None:
            operator = self.regularization.hessian_operator(self.vector(x))
            out += _real_vector(
                operator @ direction,
                size=out.size,
                name="regularization curvature product",
            )
        return out

    def hessian_actions(self, x: Any, directions: Any) -> np.ndarray:
        """Return :meth:`hessian_action` for every row of ``directions``.

        One ``normal`` job serves all rows; a native Tikhonov term is applied
        in that job when it can be, other smooth terms row by row.
        """
        from ._native_regularization import BoundNativeRegularization
        from .regularization import _BoundSum

        rows = np.asarray(directions, dtype=np.float64).reshape(-1, self.space.size)
        if not len(rows):
            return np.zeros((0, self.space.size))
        lin = self.linearization(x)
        terms = [] if self.regularization is None else [self.regularization]
        if isinstance(self.regularization, _BoundSum):
            terms = list(self.regularization.terms)
        natives = [t for t in terms if isinstance(t, BoundNativeRegularization)]
        native = natives[0] if len(natives) == 1 else None
        vectors = [self.vector(row) for row in rows]
        out = np.array(
            [p.values for p in lin.apply_normal_batch(vectors, regularization=native)]
        )
        for term in terms:
            if term is native:
                continue
            operator = term.hessian_operator(self.vector(x))
            for i, direction in enumerate(vectors):
                out[i] += _real_vector(
                    operator @ direction,
                    size=out.shape[1],
                    name="regularization curvature product",
                )
        return out


# ---------------------------------------------------------------------------
# FWI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FWIIteration:
    """Event passed to the :class:`FWI` callback after every accepted iteration."""

    stage_index: int
    stage: Stage
    iteration: int
    stage_iteration: int
    model: ControlVector
    loss: LossTerms
    record: OptimizationRecord
    diagnostics: InexactNewtonIteration
    mode: str = "global"
    patch_name: Optional[str] = None
    sweep: Optional[int] = None


@dataclass(frozen=True)
class _ResumePlan:
    state: ControlState
    start: int
    start_iteration: int
    checkpoint: OptimizationCheckpoint
    stage_index: int
    # The stage's optimization finished; only end-of-stage work remains.
    finalize: bool = False
    summary: Optional[Dict[str, Any]] = None


class _ContinuationResult:
    """``run_continuation`` stage result carrying the unmasked active vector."""

    def __init__(self, model: np.ndarray, stage_result: StageResult) -> None:
        self.model = model
        self.stage_result = stage_result


def _gaussian_declaration(prior: Any) -> str:
    """Fingerprint authored prior fields independently of stage mesh refinement."""

    def scale(value: Any) -> Any:
        # Arrays stay arrays: ``fingerprint`` digests large ones by their bytes.
        if isinstance(value, Mapping):
            return {str(k): scale(v) for k, v in value.items()}
        if hasattr(value, "magnitude"):
            return dict(value=np.asarray(value.magnitude), units=str(value.units))
        return np.asarray(value)

    return fingerprint(
        reference=prior.reference.values,
        basis=[
            (k, prior.reference.space.block(k).basis_identity)
            for k in prior.reference.space.blocks
        ],
        std=scale(prior.std),
        mesh_measure=prior.mesh_measure,
    )


class FWI:
    """Staged full-waveform inversion over one :class:`ImagingProblem`.

    A stage with ``controls`` switches the run (from that stage on) to
    ``problem.with_controls(stage.controls)`` with the accepted state
    transferred to the new layout; :attr:`final_problem` and
    :attr:`FWIResult.problem` are the last stage's problem.

    Args:
        problem: The problem (or any object with the same ``restrict`` /
            ``linearize`` / ``state`` protocol, e.g. an extended problem) the
            first stage runs on.
        stages: :class:`Stage` sequence, a single stage, or a
            :class:`ContinuationSchedule`.
        optimizer: :class:`LBFGS` (default) or :class:`NewtonCG`; a stage's
            ``optimizer`` overrides it.
        regularization: Regularization (``bind(space)`` protocol) added to the misfit; a
            stage's ``regularization`` overrides it.
        preconditioner: Preconditioner (``bind(space)`` protocol) or plain
            ``(model, g) -> ndarray`` callable; bound preconditioners are
            updated at every stage start and every
            ``optimizer.preconditioner_refresh`` iterations.
        smoothing: Search-step smoothing applied by Sauce for every stage
            (overrides the problem's; ``False`` switches it off).
        step_limit: RMS step cap per block (overrides the optimizer's).
        scaling: ``None``, ``"curvature"`` (estimate block curvatures with
            one JVP per block at every stage start) or a ``block ->
            curvature`` mapping; applied as a diagonal change of variables.
        checkpoint: Checkpoint path (``fs-optimization-checkpoint-1`` plus a
            ``<stem>.state.h5`` control state); relative paths live in
            ``problem.workdir``.  ``None`` disables checkpointing.
        history: History path, an :class:`OptimizationHistory`, or ``None``
            for an in-memory history.
        callback: Called with an :class:`FWIIteration` after every accepted
            iteration.
    """

    def __init__(
        self,
        problem: ImagingProblem,
        stages: Any,
        *,
        optimizer: Any = None,
        regularization: Any = None,
        preconditioner: Any = None,
        smoothing: Any = None,
        step_limit: Optional[float] = None,
        scaling: Any = None,
        checkpoint: Optional[Union[str, Path]] = None,
        history: Any = None,
        callback: Optional[Callable[[FWIIteration], None]] = None,
        patch_updates: Optional[PatchUpdates] = None,
        uncertainty: Any = None,
        curvature: Optional[CurvatureTransfer] = None,
    ) -> None:
        if patch_updates is not None and not isinstance(patch_updates, PatchUpdates):
            raise TypeError("patch_updates must be a PatchUpdates configuration")
        from .curvature import BFGSHistory
        from .statistics import BFGSUncertainty, GaussianPrior

        if uncertainty is not None and not isinstance(uncertainty, BFGSUncertainty):
            raise TypeError("uncertainty must be a BFGSUncertainty configuration")
        if curvature is not None and not isinstance(curvature, CurvatureTransfer):
            raise TypeError("curvature must be a CurvatureTransfer policy")
        self.curvature = CurvatureTransfer.reset() if curvature is None else curvature
        self._curvature_archive: Optional[BFGSHistory] = None
        self._curvature_source: Optional[_CurvatureSource] = None
        self._curvature_output: Optional[str] = None
        self.uncertainty = uncertainty
        self._uncertainty_archive: Optional[BFGSHistory] = None
        self._uncertainty_outputs: dict[str, str] = {}
        # Restart arrays of the stage being solved (see ``_write_checkpoint``).
        self._stage_files: Optional[Tuple[int, StageFiles]] = None
        self.patch_updates = patch_updates
        self.problem = problem
        self.stages: List[Stage] = _stage_list(stages)
        if patch_updates is not None and any(
            stage.patches is None and getattr(problem, "patches", None) is None
            for stage in self.stages
        ):
            raise ValueError("Local patch updates require patches in every stage")
        self._transfer_enabled = any(
            self._curvature_policy(s).method != "reset" for s in self.stages
        )
        self.optimizer = LBFGS() if optimizer is None else optimizer
        if self._transfer_enabled:
            if (
                patch_updates is not None
                or getattr(problem, "patches", None) is not None
                or any(s.patches is not None for s in self.stages)
            ):
                raise ValueError(
                    "Curvature transfer currently requires full-domain FWI"
                )
            if any(
                not isinstance(
                    self.optimizer if s.optimizer is None else s.optimizer, LBFGS
                )
                for s in self.stages
            ):
                raise ValueError("Curvature transfer requires LBFGS in every stage")
            problem.backend.curvature()
        if not callable(getattr(self.optimizer, "solve", None)):
            raise TypeError("optimizer must provide solve()")
        if uncertainty is not None:
            if (
                patch_updates is not None
                or getattr(problem, "patches", None) is not None
            ):
                raise ValueError("BFGS uncertainty currently requires full-domain FWI")
            if scaling is not None:
                raise ValueError(
                    "BFGS uncertainty automatically uses Gaussian prior scaling"
                )
            problem.backend.curvature()  # fail before any propagation on unsupported sites
            for i, stage in enumerate(self.stages):
                if uncertainty.stages == "final" and i != len(self.stages) - 1:
                    continue
                selected = (
                    self.optimizer if stage.optimizer is None else stage.optimizer
                )
                prior = (
                    regularization
                    if stage.regularization is None
                    else stage.regularization
                )
                if not isinstance(selected, LBFGS) or not isinstance(
                    prior, GaussianPrior
                ):
                    raise ValueError(
                        "BFGS uncertainty requires LBFGS and GaussianPrior"
                    )
                misfit = problem.misfit if stage.misfit is None else stage.misfit
                if getattr(misfit, "noise_std", None) is None:
                    raise ValueError(
                        "BFGS uncertainty requires Misfit.l2(noise_std=...)"
                    )
                if stage.loss is not None:
                    raise ValueError(
                        "BFGS uncertainty cannot override the Gaussian data loss"
                    )
        self.regularization = regularization
        self.preconditioner = preconditioner
        if _receiver_probe_request(preconditioner) is not None and (
            patch_updates is not None
            or getattr(problem, "patches", None) is not None
            or any(stage.patches is not None for stage in self.stages)
        ):
            raise ValueError(
                "Diagonal(probes='receiver') needs full-domain receiver probes; "
                "patch FWI supports probes='rademacher' or 'unit'"
            )
        if smoothing is not None and smoothing is not False:
            smoothing = control_smoothing(smoothing)
        self.smoothing = smoothing
        if self._transfer_enabled:
            # Reject proximal penalties before any solve; an explicit penalty
            # replaces the stage, workflow and problem smoothing in that order.
            for stage in self.stages:
                declared = (
                    regularization
                    if stage.regularization is None
                    else stage.regularization
                )
                if declared is None:
                    declared = next(
                        (s for s in (stage.smoothing, smoothing) if s is not None),
                        getattr(problem, "smoothing", None),
                    )
                _require_smooth_regularization(declared)
        if step_limit is not None:
            limit = float(step_limit)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("step_limit must be finite and positive")
            step_limit = limit
        self.step_limit = step_limit
        if (
            scaling is not None
            and scaling != "curvature"
            and not isinstance(scaling, Mapping)
        ):
            raise ValueError("scaling must be None, 'curvature' or a block mapping")
        self.scaling = scaling
        self.checkpoint_path: Optional[Path] = None
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
            if not path.is_absolute():
                path = Path(problem.workdir) / path
            self.checkpoint_path = path
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        self.callback = callback
        self._history_spec = history
        self._history: Optional[OptimizationHistory] = (
            history if isinstance(history, OptimizationHistory) else None
        )
        self.results: List[StageResult] = []
        self._views: Dict[str, ImagingProblem] = {}
        self._problems: Dict[int, ImagingProblem] = {}
        self._stage_inputs: Dict[str, str] = {}

    def _preconditioner_linearization(
        self, view: Any, objective: "_StageObjective", model: Any
    ) -> Linearization:
        """Return the linearization a bound preconditioner is updated at.

        ``Diagonal(probes="receiver")`` reads the receiver-probe diagonal of
        the linearize job, so that point is linearized with the probes. The
        stage's first linearization requests them already; a refresh point
        costs one more linearize.
        """
        lin = objective.linearization(model)
        probes = _receiver_probe_request(self.preconditioner)
        if probes is None:
            return lin
        return view.linearize(
            objective.vector(model), gradient=True, receiver_diagonal=probes
        )

    def _curvature_policy(self, stage: Stage) -> CurvatureTransfer:
        """Resolve a stage override without changing the run's default policy."""
        return self.curvature if stage.curvature is None else stage.curvature

    def _curvature_configuration(self) -> list:
        return [
            dataclasses.asdict(self._curvature_policy(stage)) for stage in self.stages
        ]

    # -- stage problems

    def _problem_for(self, index: int) -> ImagingProblem:
        """Return the problem stage ``index`` runs on (built lazily).

        A stage with ``controls`` gets
        ``previous_problem.with_controls(stage.controls)``, built when first
        requested; the state is transferred from the previous problem's
        current state at that moment (the accepted state of the previous
        stage during :meth:`run`).  Stages without ``controls`` share the
        problem of the stage before them; stage 0 starts from
        :attr:`problem`.
        """

        cached = self._problems.get(index)
        if cached is not None:
            return cached
        previous = self.problem if index == 0 else self._problem_for(index - 1)
        stage = self.stages[index]
        if stage.controls is not None and not callable(
            getattr(previous, "with_controls", None)
        ):
            raise TypeError(
                f"stage {stage.label(index)!r} changes the control layout, which "
                f"needs an ImagingProblem; {type(previous).__name__} has no "
                "with_controls"
            )
        if stage.controls is None:
            problem = previous
        elif isinstance(previous, ImagingProblem):
            previous._ensure_registry()
            entry = self._stage_inputs.get(str(index))
            if entry is not None:
                previous.state = ControlState.load(
                    entry, previous.full_space.without_support()
                )
            elif self.checkpoint_path is not None:
                accepted = previous._require_state()
                digest = block_digest(accepted.values).rsplit(":", 1)[-1][:20]
                path = (
                    self.checkpoint_path.parent
                    / f"{self.checkpoint_path.stem}.stage_{index}.{digest}.h5"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                accepted.save(path)
                self._stage_inputs[str(index)] = str(path)
            problem = previous.with_controls(
                stage.controls,
                mesh_averaging_wavelengths=stage.mesh_averaging_wavelengths,
                transfer=resolve_transfer(
                    stage.transfer, stage.mesh_transfer, stage.mesh_smoothing_length
                ),
                discovery_frequencies=stage.frequencies,
            )
        else:
            problem = previous.with_controls(stage.controls)
        self._problems[index] = problem
        return problem

    @property
    def final_problem(self) -> ImagingProblem:
        """Return the problem of the last stage built so far (:attr:`problem` before a run)."""

        if not self._problems:
            return self.problem
        return self._problems[max(self._problems)]

    # -- bookkeeping ----------------------------------------------------------

    @property
    def history(self) -> Optional[OptimizationHistory]:
        """Return the history of the current/last run."""

        return self._history

    @property
    def state_path(self) -> Optional[Path]:
        """Return the control-state file written beside the checkpoint."""

        if self.checkpoint_path is None:
            return None
        return self.checkpoint_path.with_name(self.checkpoint_path.stem + ".state.h5")

    def _history_path(self) -> Optional[Path]:
        spec = self._history_spec
        if spec is None or isinstance(spec, OptimizationHistory):
            return None
        path = Path(spec).expanduser()
        if not path.is_absolute():
            path = Path(self.problem.workdir) / path
        return path

    def _history_metadata(self) -> Dict[str, Any]:
        problem = self.problem
        return {
            "problem": problem.name,
            "control_ids": ",".join(problem.full_space.blocks),
            "stages": len(self.stages),
            "optimizer": getattr(self.optimizer, "kind", type(self.optimizer).__name__),
        }

    def _open_history(self, *, resume: bool) -> OptimizationHistory:
        path = self._history_path()
        if resume and self._history is not None:
            self._history.resume("checkpoint restart")
            return self._history
        if resume and path is not None and path.is_file():
            history = OptimizationHistory.load(path)
            history.resume("checkpoint restart")
        elif self._history is not None and self._history.status == "running":
            history = self._history
        else:
            history = OptimizationHistory(path, metadata=self._history_metadata())
        self._history = history
        return history

    def _identity(self, problem: Optional[ImagingProblem] = None) -> str:
        return fingerprint(**(self.problem if problem is None else problem).identity())

    @staticmethod
    def _layout(problem: ImagingProblem) -> Tuple[str, str]:
        """Return the ``(control_ids, control_sizes)`` record of a problem."""

        full = problem.full_space
        return (
            ",".join(full.blocks),
            ",".join(str(size) for size in full.sizes.values()),
        )

    def _stage_metrics(
        self,
        index: int,
        stage: Stage,
        space: ControlSpace,
        optimizer: Any,
        regularization: Any,
    ) -> Dict[str, Any]:
        view = self._views[stage.label(index)]
        smoothing = view.smoothing
        return {
            "stage": stage.label(index),
            "stage_index": int(index),
            "frequencies": json.dumps(_frequency_pairs(stage.frequencies)),
            "active": ",".join(space.blocks),
            "optimizer": getattr(optimizer, "kind", type(optimizer).__name__),
            "regularization": (
                None if regularization is None else type(regularization).__name__
            ),
            "smoothing": None if smoothing is None else smoothing.kind,
        }

    def _checkpoint_metadata(
        self,
        index: int,
        stage: Stage,
        space: ControlSpace,
        *,
        stage_iteration: int,
        completed: bool,
        history: OptimizationHistory,
    ) -> Dict[str, Any]:
        problem = self._problem_for(index)
        assert self.state_path is not None
        support = fingerprint(support=space.support_masks()).removeprefix("sha256:")
        return {
            "schema": CHECKPOINT_SCHEMA,
            "stage_inputs": json.dumps(self._stage_inputs, sort_keys=True),
            "native_regularization": json.dumps(
                getattr(self, "_native_checkpoint", None)
            ),
            "problem": problem.name,
            "identity": self._identity(problem),
            # The control layout of this stage's problem (stages with
            # ``controls`` change it); resume rebuilds the same layout.
            "control_ids": self._layout(problem)[0],
            "control_sizes": self._layout(problem)[1],
            "stage_index": int(index),
            "stage_name": stage.label(index),
            "initial_objective": self._stage_initial_objective,
            "stage_iteration": int(stage_iteration),
            "stage_iterations": int(stage.iterations),
            "stage_completed": bool(completed),
            "active": ",".join(space.blocks),
            "frequencies": json.dumps(_frequency_pairs(stage.frequencies)),
            "support": support,
            "state_path": str(self.state_path),
            "history_iteration": history.iteration_count,
            # The optimizer coordinates of mechanism blocks are physical /
            # s_ref; the checkpointed model is only meaningful under the same
            # reference scales (resume pins or checks them).
            "mechanism_scaling": json.dumps(
                dict(sorted(_mechanism_scaling(problem).items()))
            ),
        }

    def _write_checkpoint(
        self,
        index: int,
        stage: Stage,
        view: ImagingProblem,
        space: ControlSpace,
        model: np.ndarray,
        loss: LossTerms,
        *,
        stage_iteration: int,
        completed: bool,
        history: OptimizationHistory,
        finished: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Publish one checkpoint generation.

        Control-sized restart arrays (curvature pairs, the optimizer iterate,
        coordinate scaling and the archive seed) go to the bounded
        ``<stem>.restart`` store, each written once; the checkpoint metadata
        only references them. Once the checkpoint has atomically replaced its
        predecessor, restart files and patch model epochs it no longer
        references are deleted. ``finished`` marks an optimized stage whose
        end-of-stage factorizations are still pending.
        """
        if self.checkpoint_path is None:
            return
        state = view.state_from(ControlVector(model, space))
        assert self.state_path is not None
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state_path = self.state_path
        patched = getattr(view, "patches", None) is not None
        if patched:
            # Publish a new immutable state before changing the checkpoint
            # pointer, so an interrupted write cannot change an older epoch.
            epoch = _state_epoch(state)[:20]
            state_path = state_path.with_name(
                f"{state_path.stem}_{epoch}{state_path.suffix}"
            )
        # Replace atomically: the published checkpoint may reference this file.
        with atomic_output_path(state_path) as temporary:
            state.save(temporary)
        metadata = self._checkpoint_metadata(
            index,
            stage,
            space,
            stage_iteration=stage_iteration,
            completed=completed,
            history=history,
        )
        metadata["state_path"] = str(state_path)
        metadata.update(self._stage_declarations(index, stage, view, space))
        metadata["uncertainty_config"] = json.dumps(
            None if self.uncertainty is None else dataclasses.asdict(self.uncertainty)
        )
        metadata["uncertainty_outputs"] = json.dumps(self._uncertainty_outputs)
        metadata["curvature_config"] = json.dumps(self._curvature_configuration())
        metadata["curvature_output"] = self._curvature_output
        metadata["curvature_source_stage"] = (
            None if self._curvature_source is None else self._curvature_source.index
        )
        restart, local = self._restart_record(index, completed)
        metadata["restart"] = json.dumps(restart)
        if restart is not None or patched:
            metadata["optimizer_config"] = getattr(self, "_optimizer_config", None)
        if self._uncertainty_archive is not None:
            metadata["uncertainty_prior"] = self._uncertainty_prior
        if finished is not None:
            metadata["stage_finished"] = True
            metadata["stage_status"] = int(finished["status"])
            metadata["stage_message"] = str(finished["message"])
            metadata["stage_success"] = bool(finished["success"])
        if patched:
            metadata["stage_identity"] = self._identity(view)
            assert view._patch_runtime is not None
            metadata["patch_runtime"] = json.dumps(
                view._patch_runtime.checkpoint(), sort_keys=True
            )
            metadata["model_epoch"] = epoch
            metadata["patch_updates"] = json.dumps(
                None if self.patch_updates is None else self.patch_updates.to_dict()
            )
            metadata["local_updates"] = json.dumps(local)
        OptimizationCheckpoint(
            model=model,
            iteration=history.iteration_count,
            evaluations=history.evaluation_count,
            loss=loss,
            metadata=metadata,
        ).save(self.checkpoint_path)
        # The new generation is published: drop everything it superseded.
        self._restart_store().commit(restart)
        if patched:
            pattern = f"{glob.escape(self.state_path.stem)}_*{self.state_path.suffix}"
            for stale in state_path.parent.glob(pattern):
                if stale != state_path:
                    stale.unlink(missing_ok=True)

    # -- restart arrays ---------------------------------------------------------

    def _restart_store(self) -> RestartStore:
        assert self.checkpoint_path is not None
        store = getattr(self, "_restarts", None)
        if store is None or store.root != self.checkpoint_path.with_name(
            f"{self.checkpoint_path.stem}.restart"
        ):
            store = RestartStore(self.checkpoint_path)
            self._restarts = store
        return store

    def _stage_files_for(self, index: int) -> StageFiles:
        """Return the restart directory of this run of stage ``index``."""
        current = getattr(self, "_stage_files", None)
        if current is None or current[0] != index:
            current = (index, self._restart_store().new_stage(index))
            self._stage_files = current
        return current[1]

    def _stage_declarations(
        self, index: int, stage: Stage, view: ImagingProblem, space: ControlSpace
    ) -> Dict[str, Any]:
        """Return penalty and operator declarations a resume must match.

        Computed once per stage space: fingerprinting a Gaussian prior or
        binding a custom penalty per checkpoint would repeat control-sized work.
        """
        cached = getattr(self, "_declarations", None)
        if cached is not None and cached[0] == index and cached[1] is space:
            return dict(cached[2])
        from .statistics import GaussianPrior

        prior_spec = (
            self.regularization
            if stage.regularization is None
            else stage.regularization
        )
        declarations: Dict[str, Any] = {
            "gaussian_prior": (
                None
                if not isinstance(prior_spec, GaussianPrior)
                else _gaussian_declaration(prior_spec)
            )
        }
        if self._transfer_enabled or self.uncertainty is not None:
            effective_regularization = (
                view.smoothing if prior_spec is None else prior_spec
            )
            declarations["curvature_regularization_declaration"] = (
                _checkpoint_regularization_identity(effective_regularization, space)
            )
            declarations["curvature_operator_configuration"] = (
                _checkpoint_operator_configuration(self.scaling, self.preconditioner)
            )
        self._declarations = (index, space, declarations)
        return dict(declarations)

    def _restart_record(
        self, index: int, completed: bool
    ) -> Tuple[Optional[Dict[str, Any]], Any]:
        """Persist this generation's new restart arrays; return its references.

        Returns the JSON restart record (``None`` when the checkpoint needs no
        arrays) and the local patch progress with its optimizer states and
        scalings replaced by file references. Pairs shared by the optimizer
        restart and the curvature archive are stored once.
        """
        local = getattr(self, "_local_checkpoint", None)
        live = local is not None and any(
            isinstance(entry.get("optimizer_state"), LBFGSRestart)
            or isinstance(entry.get("scaling"), np.ndarray)
            for entry in local.get("proposals", {}).values()
        )
        # A completed stage resumes at the next one: it keeps no restart arrays.
        state = None if completed else getattr(self, "_optimizer_checkpoint", None)
        archive = None if completed else self._curvature_archive
        if archive is None and not completed:
            archive = self._uncertainty_archive
        scaling = None if completed else getattr(self, "_optimizer_scaling", None)
        if state is None and archive is None and scaling is None and not live:
            return None, local
        files = self._stage_files_for(index)
        names = set()
        if archive is not None or scaling is not None:
            arrays: Dict[str, Optional[np.ndarray]] = dict(scaling=scaling)
            if archive is not None:
                arrays["base_inverse_diagonal"] = archive.base_inverse_diagonal
                if archive.seed_rank:
                    arrays["seed_modes"] = archive.seed_modes
                    arrays["seed_eigenvalues"] = archive.seed_eigenvalues
            names.add(
                files.put(
                    "stage.h5",
                    arrays,
                    attrs=dict(
                        archive=(
                            None
                            if archive is None
                            else dict(
                                state=archive.state,
                                coordinates=archive.coordinates,
                                provenance=archive.provenance,
                            )
                        ),
                        units=getattr(self, "_uncertainty_units", None),
                    ),
                )
            )
        optimizer = None
        if state is not None:
            # An unscaled optimizer iterate is the checkpoint model itself.
            optimizer = files.save_state(state, model=scaling is not None)
            names |= state_files(optimizer)
        archived = None
        if archive is not None:
            steps = () if state is None else state.steps
            if len(archive.pairs[0]) != len(steps) or any(
                a is not b for a, b in zip(archive.pairs[0], steps)
            ):
                raise RuntimeError(
                    "The curvature archive and the optimizer restart diverged"
                )
            archived = dict(
                accepted_iterations=archive.accepted_iterations, stage=archive._stage
            )
        if live:
            assert local is not None
            proposals = {}
            for key, entry in local["proposals"].items():
                entry = dict(entry)
                prefix = f"patch_{key}_"
                if isinstance(entry.get("optimizer_state"), LBFGSRestart):
                    entry["optimizer_state"] = files.save_state(
                        entry["optimizer_state"], prefix
                    )
                    names |= state_files(entry["optimizer_state"])
                if isinstance(entry.get("scaling"), np.ndarray):
                    name = files.put(
                        f"{prefix}scaling_{int(entry['iteration'])}.h5",
                        dict(scaling=entry["scaling"]),
                    )
                    entry["scaling"] = {"file": name}
                    names.add(name)
                proposals[key] = entry
            local = {**local, "proposals": proposals}
        record = dict(
            schema=RESTART_SCHEMA,
            directory=files.name,
            files=sorted(names),
            optimizer=optimizer,
            archive=archived,
        )
        return record, local

    def _restore_restart(
        self,
        index: int,
        record: Optional[Mapping[str, Any]],
        checkpoint: OptimizationCheckpoint,
        meta: Mapping[str, Any],
    ) -> None:
        """Load the restart arrays a checkpoint references for stage ``index``."""
        from .curvature import BFGSHistory

        state = None
        scaling = None
        files = None
        if record is not None:
            files = self._restart_store().open_stage(record)
            self._resume_files = (index, files)
            arrays: Dict[str, np.ndarray] = {}
            attrs: Dict[str, Any] = {}
            if "stage.h5" in record["files"]:
                arrays, attrs = files.get("stage.h5")
            scaling = arrays.get("scaling")
            if record["optimizer"] is not None:
                state = files.load_state(record["optimizer"], checkpoint.model)
            if record["archive"] is not None:
                identities = attrs["archive"]
                archive = BFGSHistory(
                    arrays["base_inverse_diagonal"],
                    state=identities["state"],
                    coordinates=identities["coordinates"],
                    seed_modes=arrays.get("seed_modes"),
                    seed_eigenvalues=arrays.get("seed_eigenvalues"),
                    provenance=identities["provenance"],
                )
                if state is not None:
                    # The archive and the optimizer share the loaded pairs.
                    archive._set_pairs(
                        state.steps, state.gradient_differences, state.pair_ids
                    )
                archive._accepted_iterations = int(
                    record["archive"]["accepted_iterations"]
                )
                archive._stage = record["archive"]["stage"]
                self._resume_curvature = (index, archive)
            self._resume_units = attrs.get("units")
        self._resume_optimizer = (index, state, meta.get("optimizer_config"), scaling)
        local = getattr(self, "_resume_local", None)
        if local is not None and local[0] == index and local[1] is not None:
            for entry in local[1]["proposals"].values():
                if isinstance(entry.get("optimizer_state"), Mapping):
                    if files is None:
                        raise ValueError("Checkpoint local updates lack restart files")
                    entry["optimizer_state"] = files.load_state(
                        entry["optimizer_state"]
                    )
                if isinstance(entry.get("scaling"), Mapping):
                    if files is None:
                        raise ValueError("Checkpoint local updates lack restart files")
                    entry["scaling"] = files.get(entry["scaling"]["file"])[0]["scaling"]

    # -- resume ---------------------------------------------------------------

    def _resume_plan(self) -> Optional[_ResumePlan]:
        path = self.checkpoint_path
        if path is None or not path.is_file():
            return None
        checkpoint = OptimizationCheckpoint.load(path)
        meta = checkpoint.metadata
        schema = meta.get("schema")
        if schema in _LEGACY_CHECKPOINT_SCHEMAS:
            raise ValueError(
                f"{path} is an {schema} checkpoint, which kept optimizer restart "
                "state as JSON lists; this FrequenSolve version cannot resume it. "
                "Rerun with run(resume=False) or remove the checkpoint."
            )
        if schema != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path} is not an FWI checkpoint")
        if meta.get("problem") != self.problem.name:
            raise ValueError(
                f"checkpoint {path} belongs to problem {meta.get('problem')!r}, "
                f"not {self.problem.name!r}"
            )
        index = int(meta["stage_index"])
        if not 0 <= index < len(self.stages):
            raise ValueError("Checkpoint stage index is outside the configured stages")
        from .statistics import GaussianPrior

        prior_spec = (
            self.regularization
            if self.stages[index].regularization is None
            else self.stages[index].regularization
        )
        declaration = (
            None
            if not isinstance(prior_spec, GaussianPrior)
            else _gaussian_declaration(prior_spec)
        )
        if meta.get("gaussian_prior") != declaration:
            raise ValueError("Checkpoint Gaussian prior declaration changed")
        configured = (
            None if self.uncertainty is None else dataclasses.asdict(self.uncertainty)
        )
        if json.loads(meta.get("uncertainty_config", "null")) != configured:
            raise ValueError("Checkpoint uncertainty configuration changed")
        self._uncertainty_outputs = json.loads(meta.get("uncertainty_outputs", "{}"))
        configured_curvature = meta.get("curvature_config")
        if configured_curvature is None:
            if self._transfer_enabled:
                raise ValueError("Checkpoint has no curvature-transfer configuration")
        elif json.loads(configured_curvature) != self._curvature_configuration():
            raise ValueError("Checkpoint curvature-transfer configuration changed")
        self._curvature_output = meta.get("curvature_output")
        if meta.get("uncertainty_prior") is not None:
            self._resume_uncertainty = (index, meta["uncertainty_prior"])
        self._resume_initial_objective = (index, meta.get("initial_objective"))
        self._resume_regularization = (
            index,
            json.loads(meta.get("native_regularization", "null")),
        )
        if index >= len(self.stages):
            raise ValueError(
                f"checkpoint stage index {index} exceeds the {len(self.stages)} stages"
            )
        # Rebuild the control layout the checkpointed stage ran on (stage
        # ``controls`` of every stage up to it) before comparing layouts.
        self._stage_inputs = json.loads(meta.get("stage_inputs", "{}"))
        problem = self._problem_for(index)
        if self._curvature_output:
            from .statistics import UncertaintyResult

            source_index = int(meta["curvature_source_stage"])
            source_problem = self._problem_for(source_index)
            self._curvature_source = _CurvatureSource(
                source_index,
                UncertaintyResult.load(
                    self._curvature_output,
                    source_problem.full_space,
                    native=problem.backend.curvature(),
                ),
            )
        patch_record = meta.get("patch_runtime")
        if patch_record is not None:
            configured = (
                None if self.patch_updates is None else self.patch_updates.to_dict()
            )
            if json.loads(meta.get("patch_updates", "null")) != configured:
                raise ValueError("Checkpoint patch-update mode or settings changed")
            self._resume_local = (index, json.loads(meta.get("local_updates", "null")))
        if patch_record is not None:
            from ._patch_problem import _PatchRuntime

            view = self.stages[index].view(problem)
            runtime = _PatchRuntime.restore(view, json.loads(patch_record))
            if meta.get("stage_identity") != self._identity(
                self._smoothed_view(view, self.stages[index])
            ):
                raise ValueError("Checkpoint patch stage objective or settings changed")
            view._patch_runtime = runtime
            runtime.problem = view
            self._views[self.stages[index].label(index)] = view
        elif getattr(problem, "patches", None) is not None:
            raise ValueError("Patch checkpoint omits its frozen stage and meshes")
        control_ids, control_sizes = self._layout(problem)
        if meta.get("control_ids") != control_ids:
            raise ValueError(
                "checkpoint control blocks do not match the problem: "
                f"{meta.get('control_ids')} != {control_ids}"
            )
        if meta.get("control_sizes") != control_sizes:
            raise ValueError(
                "checkpoint control sizes do not match the problem: "
                f"{meta.get('control_sizes')} != {control_sizes}"
            )
        if meta.get("identity") != self._identity(problem):
            raise ValueError(
                "checkpoint identity does not match the problem (simulation, "
                "control layout, observed data, misfit or smoothing changed)"
            )
        stage = self.stages[index]
        expected_active = ",".join(stage.view(problem).space.blocks)
        if meta.get("active") != expected_active:
            raise ValueError(
                f"checkpoint stage {index} activates {meta.get('active')!r}; the "
                f"configured stage activates {expected_active!r}"
            )
        expected_frequencies = json.dumps(_frequency_pairs(stage.frequencies))
        if meta.get("frequencies") != expected_frequencies:
            raise ValueError(
                f"checkpoint stage {index} frequencies {meta.get('frequencies')} "
                f"differ from the configured {expected_frequencies}"
            )
        state_path = Path(str(meta.get("state_path", "")))
        if not state_path.is_file():
            raise FileNotFoundError(
                f"checkpoint {path} references a missing control state {state_path}"
            )
        state = ControlState.load(state_path, problem.full_space.without_support())
        if self._transfer_enabled or self.uncertainty is not None:
            # Declarations were written from the stage view a solve uses,
            # including the workflow-level smoothing override.
            checkpoint_view = self._smoothed_view(stage.view(problem), stage)
            declaration_space = checkpoint_view.space.without_support()
            declaration_space = declaration_space.with_support(
                {
                    name: mask
                    for name, mask in state.space.support_masks().items()
                    if name in declaration_space.blocks
                }
            )
            effective_regularization = (
                checkpoint_view.smoothing if prior_spec is None else prior_spec
            )
            expected_declaration = _checkpoint_regularization_identity(
                effective_regularization, declaration_space
            )
            if "curvature_regularization_declaration" not in meta:
                raise ValueError(
                    "Checkpoint lacks curvature regularization declaration; start a fresh run"
                )
            if meta["curvature_regularization_declaration"] != expected_declaration:
                raise ValueError(
                    "Checkpoint curvature regularization declaration changed"
                )
            expected_operator = _checkpoint_operator_configuration(
                self.scaling, self.preconditioner
            )
            if meta.get("curvature_operator_configuration") != expected_operator:
                raise ValueError(
                    "Checkpoint scaling or fixed preconditioner configuration changed"
                )
        if (
            patch_record is not None
            and meta.get("model_epoch") != _state_epoch(state)[:20]
        ):
            raise ValueError("Checkpoint complete model epoch changed")
        recorded = meta.get("mechanism_scaling")
        if recorded:
            scaling = {str(k): float(v) for k, v in json.loads(recorded).items()}
            pin = getattr(problem, "_pin_mechanism_scaling", None)
            if scaling and callable(pin):
                try:
                    pin(scaling, state.scaling_units)
                except ValueError as exc:
                    raise ValueError(
                        f"checkpoint {path} was written with other mechanism "
                        f"reference scales: {exc}"
                    ) from exc
        stage_iteration = int(meta.get("stage_iteration", 0))
        completed = bool(meta.get("stage_completed"))
        # A stage that still owes end-of-stage factorizations is complete only
        # once its completed checkpoint exists: its last iteration checkpoint
        # already reaches the budget, and an optimizer may also stop early.
        finalize = (
            not completed
            and any(self._stage_factorizations(index))
            and (
                bool(meta.get("stage_finished")) or stage_iteration >= stage.iterations
            )
        )
        if completed or (not finalize and stage_iteration >= stage.iterations):
            return _ResumePlan(state, index + 1, 0, checkpoint, index)
        self._restore_restart(
            index, json.loads(meta.get("restart", "null")), checkpoint, meta
        )
        summary = None
        if meta.get("stage_finished"):
            summary = dict(
                status=int(meta["stage_status"]),
                message=str(meta["stage_message"]),
                success=bool(meta["stage_success"]),
            )
        return _ResumePlan(
            state,
            index,
            stage_iteration,
            checkpoint,
            index,
            finalize=finalize,
            summary=summary,
        )

    def _skipped_result(
        self, index: int, stage: Stage, history: OptimizationHistory
    ) -> StageResult:
        """Summarize a stage completed by an earlier run from the history."""

        records = [
            r
            for r in history.iterations
            if r.metrics.get("stage_index") == index and r.accepted is not False
        ]
        if records:
            initial, final = records[0].loss, records[-1].loss
            stage_iteration = int(records[-1].metrics.get("stage_iteration", 0))
        else:
            initial = final = LossTerms(data=0.0)
            stage_iteration = stage.iterations
        view = stage.view(self._problem_for(index))
        uncertainty_result = None
        if str(index) in self._uncertainty_outputs:
            from .statistics import UncertaintyResult

            uncertainty_result = UncertaintyResult.load(
                self._uncertainty_outputs[str(index)],
                view.full_space,
                native=view.backend.curvature(),
            )
        return StageResult(
            index=index,
            name=stage.label(index),
            frequencies=stage.frequencies,
            active=view.space.blocks,
            iterations=0,
            stage_iteration=stage_iteration,
            success=True,
            status=0,
            message="completed by an earlier run",
            initial_loss=initial,
            final_loss=final,
            resumed=True,
            skipped=True,
            uncertainty=uncertainty_result,
        )

    # -- stage solves ---------------------------------------------------------

    def _scaling_for(self, lin: Linearization, space: ControlSpace) -> Optional[Any]:
        if self.scaling is None:
            return None
        if isinstance(self.scaling, Mapping):
            return dict(self.scaling)
        return block_curvatures(lin, seed=20260829)

    def solve_stage(
        self,
        stage: Stage,
        state: Optional[ControlState] = None,
        *,
        index: Optional[int] = None,
        start_iteration: int = 0,
    ) -> StageResult:
        """Solve one stage from ``state`` (default: the problem's current state).

        The accepted vector is written back into ``problem.state``.  Custom
        loops call this directly; :meth:`run` uses it through
        :func:`run_continuation`.
        """

        if not isinstance(stage, Stage):
            raise TypeError("stage must be a Stage")
        if index is None:
            index = self.stages.index(stage) if stage in self.stages else 0
        if state is not None:
            self._problem_for(index).state = state
        if self._history is None:
            self._history = self._open_history(resume=False)
        return self._solve_stage(index, stage, start_iteration)

    def _solve_stage(
        self,
        index: int,
        stage: Stage,
        start_iteration: int,
        *,
        expected_model: Optional[np.ndarray] = None,
    ) -> StageResult:
        stage_started = perf_counter()
        problem = self._problem_for(index)
        history = self._history
        assert history is not None
        label = stage.label(index)
        view = self._views.get(label)
        if view is None:
            view = stage.view(problem)
            self._views[label] = view
        if self.smoothing is not None and stage.smoothing is None:
            view = view.restrict(
                smoothing=None if self.smoothing is False else self.smoothing
            )
            self._views[label] = view
        optimizer = self.optimizer if stage.optimizer is None else stage.optimizer
        regularization = (
            self.regularization
            if stage.regularization is None
            else stage.regularization
        )
        remaining = stage.iterations - int(start_iteration)
        if remaining < 1:
            raise ValueError(
                f"stage {label!r} has no iterations left ({start_iteration} of "
                f"{stage.iterations} done)"
            )

        capture_uncertainty = self.uncertainty is not None and (
            self.uncertainty.stages == "all" or index == len(self.stages) - 1
        )
        capture_curvature = capture_uncertainty or self._transfer_enabled
        self._uncertainty_archive = None
        self._curvature_archive = None
        self._optimizer_checkpoint = None
        self._optimizer_scaling = None
        self._local_checkpoint = None
        self._uncertainty_units = None
        resumed_files = getattr(self, "_resume_files", None)
        self._stage_files = (
            resumed_files
            if resumed_files is not None
            and resumed_files[0] == index
            and start_iteration > 0
            else None
        )
        if self._transfer_enabled:
            optimizer = dataclasses.replace(optimizer, preconditioner_refresh=None)
        if capture_uncertainty:
            assert self.uncertainty is not None
            from .statistics import GaussianPrior

            if not isinstance(optimizer, LBFGS) or not isinstance(
                regularization, GaussianPrior
            ):
                raise ValueError("BFGS uncertainty requires LBFGS and GaussianPrior")
            if getattr(view.misfit, "noise_std", None) is None:
                raise ValueError(
                    "BFGS uncertainty requires a Gaussian noise declaration"
                )
            optimizer = dataclasses.replace(
                optimizer,
                memory=max(optimizer.memory, stage.iterations),
                curvature_tolerance=max(
                    optimizer.curvature_tolerance, self.uncertainty.curvature_tolerance
                ),
                preconditioner_refresh=None,
            )

        # First linearization of the stage: adopts the support masks that
        # define the stage space (held fixed until the next transition).
        timing_backend = getattr(view, "backend", None)
        timing_snapshot = (
            timing_backend.timing_snapshot() if timing_backend is not None else None
        )
        probes = _receiver_probe_request(self.preconditioner)
        first = (
            view.linearize(gradient=True)
            if probes is None
            else view.linearize(gradient=True, receiver_diagonal=probes)
        )
        space = view.space
        x0 = np.array(first.point.values, dtype=np.float64, copy=True)
        if expected_model is not None:
            if expected_model.shape != x0.shape or not np.allclose(
                expected_model, x0, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    f"checkpoint model for stage {label!r} does not match the "
                    "state it references (support masks or block layout changed)"
                )
            x0 = np.array(expected_model, copy=True)
        lower, upper = space.bounds
        x0 = np.clip(x0, lower, upper)

        from frequensolve.imaging._native_regularization import (
            bind_workflow_regularization,
        )

        # Explicit stage/workflow regularization overrides the inherited native
        # smoothing configuration; it is never silently counted twice.
        regularization = view.smoothing if regularization is None else regularization
        bound_regularization, native_regularization = bind_workflow_regularization(
            regularization, space, view, first
        )
        resumed = getattr(self, "_resume_regularization", None)
        if resumed is not None and resumed[0] == index:
            if (resumed[1] is None) != (native_regularization is None):
                raise ValueError(
                    "checkpoint native regularization configuration changed"
                )
            if native_regularization is not None:
                native_regularization.restore(resumed[1])
        self._native_checkpoint = (
            None
            if native_regularization is None
            else native_regularization.checkpoint()
        )
        # Preserve native contexts in checkpoints, while smooth Tikhonov uses
        # the requested optimizer and its exact coefficient gradient/Hessian.
        if native_regularization is not None and native_regularization.is_smooth:
            if bound_regularization is None:
                bound_regularization = native_regularization
            else:
                from .regularization import Sum, _BoundSum

                bound_regularization = _BoundSum(
                    Sum(
                        bound_regularization.regularization,
                        native_regularization.regularization,
                    ),
                    space,
                    [bound_regularization, native_regularization],
                )
            native_regularization = None
        if self.patch_updates is not None:
            from ._patch_updates import solve_local_stage

            local_result = solve_local_stage(
                self,
                index,
                stage,
                start_iteration,
                view,
                first,
                bound_regularization,
                native_regularization,
                stage_started=stage_started,
            )
            if timing_snapshot is not None:
                assert timing_backend is not None
                local_result = dataclasses.replace(
                    local_result,
                    metrics={
                        **local_result.metrics,
                        **timing_backend.cost_since(timing_snapshot),
                    },
                )
            return local_result
        metrics = self._stage_metrics(index, stage, space, optimizer, regularization)
        if native_regularization is not None:
            metrics["optimizer"] = "proximal_gradient"
        objective = _StageObjective(view, space, bound_regularization, history, metrics)
        objective.native_regularization = native_regularization
        initial_loss = objective.loss(x0)
        self._stage_initial_objective = initial_loss.total
        if start_iteration > 0:
            reference_index, reference = getattr(
                self, "_resume_initial_objective", (index, None)
            )
            if reference_index != index:
                reference = None
            if reference is None:
                records = [
                    r
                    for r in history.iterations
                    if r.metrics.get("stage_index") == index
                ]
                if not records:
                    raise ValueError(
                        "Cannot resume relative tolerances without the stage's initial objective"
                    )
                reference = records[0].loss.total
            self._stage_initial_objective = float(reference)
        linearizations_before = objective.evaluations

        preconditioner = self.preconditioner if native_regularization is None else None
        bound_preconditioner: Any = None
        if preconditioner is not None and callable(
            getattr(preconditioner, "bind", None)
        ):
            bound_preconditioner = preconditioner.bind(space)
            bound_preconditioner.update(
                self._preconditioner_linearization(view, objective, x0),
                regularization=bound_regularization,
            )
            preconditioner = bound_preconditioner
        refresh = getattr(optimizer, "preconditioner_refresh", None)
        scaling = self._scaling_for(objective.linearization(x0), space)
        curvature_provenance = {
            "mode": "reset",
            "target_stage": index,
            "hessian_actions": 0,
        }
        if capture_curvature:
            from .curvature import BFGSHistory
            from .regularization import Identity
            from .statistics import BoundGaussianPrior, _linear_materials

            if native_regularization is not None:
                raise ValueError(
                    "Curvature transfer requires a smooth objective without proximal regularization"
                )
            _linear_materials(space)
            if capture_uncertainty:
                assert isinstance(bound_regularization, BoundGaussianPrior)
                scaling = bound_regularization.std.copy()
            elif isinstance(scaling, Mapping):
                scaling = curvature_scaling(
                    scaling, space, max_ratio=optimizer.scaling_max_ratio
                )
            resumed_optimizer = getattr(self, "_resume_optimizer", None)
            if (
                resumed_optimizer is not None
                and resumed_optimizer[0] == index
                and start_iteration > 0
            ):
                if resumed_optimizer[2] != repr(optimizer):
                    raise ValueError("Checkpoint optimizer configuration changed")
                scaling = (
                    None
                    if resumed_optimizer[3] is None
                    else np.asarray(resumed_optimizer[3])
                )
            scale = np.ones(space.size) if scaling is None else np.asarray(scaling)
            if self.preconditioner is None:
                physical_base = scale**2
            elif isinstance(self.preconditioner, Identity):
                physical_base = np.ones(space.size)
            elif (
                bound_preconditioner is not None
                and getattr(bound_preconditioner, "inverse", None) is not None
            ):
                physical_base = bound_preconditioner.inverse.inverse_diagonal
            else:
                raise ValueError(
                    "BFGS curvature requires a known fixed diagonal preconditioner"
                )
            prior_identity = _regularization_identity(bound_regularization)
            # Array digests, not JSON lists: identities stay O(1) in memory.
            coordinates = fingerprint(
                prior=prior_identity,
                scaling=_configuration_value(scale),
                blocks=space.blocks,
                support={
                    k: _configuration_value(v) for k, v in space.support_masks().items()
                },
                basis=[space.block(k).basis_identity for k in space.blocks],
            )
            objective_id = fingerprint(stage=view.identity(), prior=prior_identity)
            resumed_archive = None
            resumed_curvature = getattr(self, "_resume_curvature", None)
            if (
                resumed_curvature is not None
                and resumed_curvature[0] == index
                and start_iteration > 0
            ):
                resumed_archive = resumed_curvature[1]
                resumed_uq = getattr(self, "_resume_uncertainty", None)
                if (
                    capture_uncertainty
                    and resumed_uq is not None
                    and resumed_uq[0] == index
                    and resumed_uq[1] != prior_identity
                ):
                    raise ValueError("Checkpoint Gaussian prior changed")
            if resumed_archive is not None:
                archive = resumed_archive
                if archive.coordinates != coordinates or archive.state != objective_id:
                    raise ValueError(
                        "Checkpoint curvature objective or coordinates changed"
                    )
                curvature_provenance = archive.provenance
            else:
                initial_scale = None
                seed = None
                policy = self._curvature_policy(stage)
                source = self._curvature_source
                transfer = policy.method != "reset" and source is not None
                # Plain L-BFGS rescales its initial inverse by a dynamic gamma
                # that a frozen stage history cannot follow; measure the scale
                # once instead (one Hessian action per stage, inside the
                # refresh's normal job when the stage refreshes curvature).
                measure = self.preconditioner is None and not capture_uncertainty
                fold = measure and transfer and policy.method == "refresh"
                if measure and not fold:
                    initial_scale = _initial_scale(
                        objective, x0, physical_base, (lower, upper)
                    )
                    physical_base *= initial_scale
                if transfer:
                    assert source is not None
                    if source.index >= index:
                        raise ValueError(
                            "Curvature source must precede the target stage"
                        )
                    native = view.backend.curvature()
                    seed = _prepare_seed(
                        policy,
                        source,
                        space,
                        native,
                        physical_base,
                        objective,
                        x0,
                        state=objective_id,
                        coordinates=coordinates,
                        index=index,
                        scale_bounds=(lower, upper) if fold else None,
                    )
                    curvature_provenance = seed.provenance
                    if fold:
                        initial_scale = float(seed.provenance["initial_scale"])
                        physical_base *= initial_scale
                    if capture_uncertainty and curvature_provenance.get("inherited"):
                        raise ValueError(
                            "BFGS uncertainty requires refreshed curvature; warm-start inheritance is an optimizer metric"
                        )
                curvature_provenance["history_scope"] = (
                    "full_stage" if capture_uncertainty else "limited_memory"
                )
                curvature_provenance["initial_scale"] = initial_scale
                # The history adopts the seed modes in optimizer coordinates;
                # the seed releases its block so one copy stays resident.
                modes, eigenvalues = (None, None) if seed is None else seed.take(scale)
                seed = None
                archive = BFGSHistory(
                    physical_base / scale**2,
                    state=objective_id,
                    coordinates=coordinates,
                    seed_modes=modes,
                    seed_eigenvalues=eigenvalues,
                    provenance=curvature_provenance,
                )
                del modes
            preconditioner = _history_inverse(archive, scale)
            bound_preconditioner = None
            refresh = None
            if capture_uncertainty:
                self._uncertainty_archive = archive
                self._uncertainty_prior = prior_identity
                self._uncertainty_units = dict(
                    getattr(bound_regularization, "units", {})
                )
            if self._transfer_enabled:
                self._curvature_archive = archive

        optimizer_restart = None
        if getattr(view, "patches", None) is not None or capture_curvature:
            if isinstance(scaling, Mapping):
                scaling = curvature_scaling(
                    scaling, space, max_ratio=optimizer.scaling_max_ratio
                )
            resumed_optimizer = getattr(self, "_resume_optimizer", None)
            if (
                resumed_optimizer is not None
                and resumed_optimizer[0] == index
                and start_iteration > 0
            ):
                if resumed_optimizer[2] != repr(optimizer):
                    raise ValueError("Checkpoint optimizer configuration changed")
                optimizer_restart = resumed_optimizer[1]
                scaling = (
                    None
                    if resumed_optimizer[3] is None
                    else np.asarray(resumed_optimizer[3])
                )
            self._optimizer_checkpoint = optimizer_restart
            self._optimizer_config = repr(optimizer)
            self._optimizer_scaling = (
                None if scaling is None else np.asarray(scaling, dtype=np.float64)
            )
        step_limit = self.step_limit
        if step_limit is None:
            step_limit = getattr(optimizer, "step_limit", None)

        self._write_checkpoint(
            index,
            stage,
            view,
            space,
            x0,
            initial_loss,
            stage_iteration=start_iteration,
            completed=False,
            history=history,
        )

        def on_iteration(it: InexactNewtonIteration) -> None:
            if capture_uncertainty:
                archive(it)
            elif capture_curvature:
                if it.optimizer_state is None:
                    raise ValueError(
                        "Curvature transfer requires an optimizer checkpoint"
                    )
                # Share the optimizer's limited-memory pairs (no copies).
                archive._follow(it.optimizer_state)
            if getattr(view, "patches", None) is not None or capture_curvature:
                self._optimizer_checkpoint = it.optimizer_state
            loss = objective.loss(it.model)
            stage_iteration = start_iteration + it.iteration
            record = history.record_iteration(
                it.model,
                loss,
                gradient_norm=float(np.linalg.norm(it.gradient)),
                step_norm=float(np.linalg.norm(it.step)),
                step_length=it.step_length,
                metrics={
                    **metrics,
                    "stage_iteration": stage_iteration,
                    "projected_gradient_norm": it.projected_gradient_norm,
                    "forcing": it.forcing,
                    "cg_iterations": it.cg_iterations,
                    "cg_relative_residual": it.cg_relative_residual,
                    "negative_curvature": it.negative_curvature,
                    "steepest_descent_fallback": it.steepest_descent_fallback,
                    "line_search_evaluations": it.line_search_evaluations,
                    "directional_derivative": it.directional_derivative,
                    "objective_relative_reduction": it.objective_relative_reduction,
                },
            )
            self._write_checkpoint(
                index,
                stage,
                view,
                space,
                it.model,
                loss,
                stage_iteration=stage_iteration,
                completed=False,
                history=history,
            )
            if (
                bound_preconditioner is not None
                and refresh is not None
                and it.iteration > 0
                and it.iteration % int(refresh) == 0
            ):
                bound_preconditioner.update(
                    self._preconditioner_linearization(view, objective, it.model),
                    regularization=bound_regularization,
                )
            if self.callback is not None:
                self.callback(
                    FWIIteration(
                        stage_index=index,
                        stage=stage,
                        iteration=it.iteration,
                        stage_iteration=stage_iteration,
                        model=ControlVector(it.model, space),
                        loss=loss,
                        record=record,
                        diagnostics=it,
                    )
                )

        curvature = (
            objective.hessian_action
            if getattr(optimizer, "kind", None) == "newton_cg"
            else None
        )
        result = optimizer.solve(
            objective,
            x0,
            bounds=(lower, upper),
            preconditioner=preconditioner,
            hessian_action=curvature,
            callback=on_iteration,
            max_iterations=remaining,
            initial_objective=self._stage_initial_objective,
            step_limit=step_limit,
            scaling=scaling,
            space=space,
            **(
                {"restart": optimizer_restart}
                if (getattr(view, "patches", None) is not None or capture_curvature)
                and getattr(optimizer, "kind", None) == "lbfgs"
                else {}
            ),
        )
        final = ControlVector(result.model, space)
        problem.state = view.state_from(final)
        final_loss = objective.loss(result.model)
        stage_iteration = start_iteration + int(result.iterations)
        uncertainty_result = None
        if capture_curvature:
            if any(self._stage_factorizations(index)):
                # A walltime kill during the factorizations below resumes in
                # finalize-only mode instead of optimizing the stage again.
                self._write_checkpoint(
                    index,
                    stage,
                    view,
                    space,
                    result.model,
                    final_loss,
                    stage_iteration=stage_iteration,
                    completed=False,
                    history=history,
                    finished=dict(
                        status=int(result.status),
                        message=str(result.message),
                        success=bool(result.success) or int(result.status) >= 0,
                    ),
                )
            uncertainty_result = self._factorize_stage(
                index,
                view,
                space,
                final,
                archive,
                scale,
                optimizer,
                curvature_provenance,
                getattr(self, "_uncertainty_units", None),
            )
        self._write_checkpoint(
            index,
            stage,
            view,
            space,
            result.model,
            final_loss,
            stage_iteration=stage_iteration,
            completed=True,
            history=history,
        )
        # Exhausting a stage's iteration budget (status 0) is the normal way a
        # continuation stage ends; only negative optimizer statuses (failed
        # line search, evaluation limit, no descent direction) are failures.
        stage_result = StageResult(
            index=index,
            name=label,
            frequencies=stage.frequencies,
            active=space.blocks,
            iterations=int(result.iterations),
            stage_iteration=stage_iteration,
            success=bool(result.success) or int(result.status) >= 0,
            status=int(result.status),
            message=str(result.message),
            initial_loss=initial_loss,
            final_loss=final_loss,
            evaluations=int(result.objective_evaluations),
            linearizations=objective.evaluations - linearizations_before + 1,
            resumed=start_iteration > 0,
            vector=final,
            space=space,
            uncertainty=uncertainty_result,
            metrics={
                "optimizer": metrics["optimizer"],
                "curvature_transfer": curvature_provenance,
                "elapsed_seconds": perf_counter() - stage_started,
                "elapsed_scope": "stage preparation, evaluations, reductions and checkpoint writes",
                "gradient_evaluations": int(result.gradient_evaluations),
                "hessian_products": int(result.hessian_products),
                "cg_iterations": int(result.cg_iterations),
                "line_search_evaluations": int(result.line_search_evaluations),
                "negative_curvature_events": int(result.negative_curvature_events),
                "steepest_descent_fallbacks": int(result.steepest_descent_fallbacks),
                "frozen_dofs": int(space.support.frozen_count),
                "gradient_norm": float(np.linalg.norm(result.gradient)),
            },
        )
        if timing_snapshot is not None:
            assert timing_backend is not None
            stage_result = dataclasses.replace(
                stage_result,
                metrics={
                    **stage_result.metrics,
                    **timing_backend.cost_since(timing_snapshot),
                },
            )
        self.results.append(stage_result)
        return stage_result

    def _smoothed_view(self, view: ImagingProblem, stage: Stage) -> ImagingProblem:
        """Apply the workflow-level smoothing override exactly as a stage solve does."""
        if self.smoothing is not None and stage.smoothing is None:
            return view.restrict(
                smoothing=None if self.smoothing is False else self.smoothing
            )
        return view

    def _stage_factorizations(self, index: int) -> Tuple[bool, bool]:
        """Return whether stage ``index`` ends with (uncertainty, transfer) factors.

        Transfer factors exist only for a following stage that seeds from
        them; nothing is factorized after a stage nobody reads.
        """
        uncertainty = self.uncertainty is not None and (
            self.uncertainty.stages == "all" or index == len(self.stages) - 1
        )
        transfer = (
            self._transfer_enabled
            and index + 1 < len(self.stages)
            and self._curvature_policy(self.stages[index + 1]).method != "reset"
        )
        return uncertainty, transfer

    def _factorize_stage(
        self,
        index: int,
        view: ImagingProblem,
        space: ControlSpace,
        final: ControlVector,
        archive: Any,
        scale: np.ndarray,
        optimizer: Any,
        provenance: Mapping[str, Any],
        units: Optional[Mapping[str, Any]],
    ) -> Any:
        """Run the stage's single end-of-stage BFGS factorization, if anyone uses it.

        Posterior uncertainty for this stage and the next stage's curvature
        transfer share one ``bfgs_rsvd`` of the archive at the larger of their
        ranks (an unlimited uncertainty rank covers both); its saved factors
        serve both. Returns the uncertainty result, or ``None``.
        """
        uncertainty, transfer = self._stage_factorizations(index)
        if self._transfer_enabled:
            # A later stage must never seed from an older band's factors.
            self._curvature_source = None
            self._curvature_output = None
        if not (uncertainty or transfer):
            return None
        from .statistics import UncertaintyResult

        if uncertainty:
            lower, upper = space.bounds
            tolerance = optimizer.bound_tolerance
            if np.any(
                np.isclose(final.values, lower, rtol=0, atol=tolerance)
            ) or np.any(np.isclose(final.values, upper, rtol=0, atol=tolerance)):
                raise ValueError(
                    "BFGS posterior approximation does not support active parameter bounds"
                )
        transfer_rank = (
            min(space.size, max(self._curvature_policy(s).rank for s in self.stages))
            if transfer
            else None
        )
        if uncertainty:
            assert self.uncertainty is not None
            rank = self.uncertainty.rank
            if rank is not None and transfer_rank is not None:
                rank = max(rank, transfer_rank)
            curvature_tolerance = max(
                optimizer.curvature_tolerance, self.uncertainty.curvature_tolerance
            )
        else:
            rank, curvature_tolerance = transfer_rank, optimizer.curvature_tolerance
        native = view.backend.curvature()
        factors = native.bfgs_uncertainty(
            archive,
            prior_std=scale,
            rank=rank,
            curvature_tolerance=curvature_tolerance,
        )
        result = UncertaintyResult(
            factors,
            final,
            native=native,
            units=units if uncertainty else None,
            provenance=dict(provenance),
        )
        directory = (
            Path(self._problem_for(index).workdir)
            / ("uncertainty" if uncertainty else "curvature")
            / f"stage_{index}_{factors.path.parent.name}"
        )
        saved = str(result.save(directory))
        if uncertainty:
            self._uncertainty_outputs[str(index)] = saved
        if transfer:
            self._curvature_output = saved
            self._curvature_source = _CurvatureSource(index, result)
        return result if uncertainty else None

    def _stage_optimizer(self, index: int, stage: Stage) -> Any:
        """Return the optimizer configuration a curvature-capturing stage ran with."""
        optimizer = self.optimizer if stage.optimizer is None else stage.optimizer
        if self._transfer_enabled:
            optimizer = dataclasses.replace(optimizer, preconditioner_refresh=None)
        if self._stage_factorizations(index)[0]:
            assert self.uncertainty is not None
            optimizer = dataclasses.replace(
                optimizer,
                memory=max(optimizer.memory, stage.iterations),
                curvature_tolerance=max(
                    optimizer.curvature_tolerance, self.uncertainty.curvature_tolerance
                ),
                preconditioner_refresh=None,
            )
        return optimizer

    def _finalize_stage(
        self, index: int, stage: Stage, plan: _ResumePlan
    ) -> StageResult:
        """Finish a stage whose optimization an earlier run completed.

        The earlier run reached the stage's last accepted iterate (or its
        optimizer stopped) but was interrupted before the end-of-stage
        factorizations or the completed checkpoint. Restore the archived
        curvature pairs, run only those factorizations and publish the
        completed checkpoint; no objective is evaluated and the optimizer
        does not run again.
        """
        started = perf_counter()
        problem = self._problem_for(index)
        history = self._history
        assert history is not None and problem.state is not None
        label = stage.label(index)
        view = self._smoothed_view(stage.view(problem), stage)
        view._adopt_masks(
            {
                name: mask
                for name, mask in problem.state.space.support_masks().items()
                if name in view.space.blocks
            }
        )
        self._views[label] = view
        space = view.space
        restored = getattr(self, "_resume_curvature", None)
        resumed_optimizer = getattr(self, "_resume_optimizer", None)
        if (
            restored is None
            or restored[0] != index
            or resumed_optimizer is None
            or resumed_optimizer[0] != index
        ):
            raise ValueError(
                f"checkpoint of stage {label!r} lacks the curvature archive its "
                "end-of-stage factorization needs"
            )
        optimizer = self._stage_optimizer(index, stage)
        if resumed_optimizer[2] != repr(optimizer):
            raise ValueError("Checkpoint optimizer configuration changed")
        archive = restored[1]
        scaling = resumed_optimizer[3]
        scale = np.ones(space.size) if scaling is None else np.asarray(scaling)
        model = np.array(plan.checkpoint.model, copy=True)
        final = ControlVector(model, space)
        reference = getattr(self, "_resume_initial_objective", (index, None))
        if reference[0] == index and reference[1] is not None:
            self._stage_initial_objective = float(reference[1])
        native_record = getattr(self, "_resume_regularization", (index, None))
        self._native_checkpoint = (
            native_record[1] if native_record[0] == index else None
        )
        self._optimizer_config = resumed_optimizer[2]
        self._optimizer_checkpoint = resumed_optimizer[1]
        self._optimizer_scaling = scaling
        self._local_checkpoint = None
        if self._transfer_enabled:
            self._curvature_archive = archive
        uncertainty = self._stage_factorizations(index)[0]
        self._uncertainty_archive = archive if uncertainty else None
        resumed_prior = getattr(self, "_resume_uncertainty", None)
        self._uncertainty_prior = (
            resumed_prior[1]
            if resumed_prior is not None and resumed_prior[0] == index
            else None
        )
        resumed_files = getattr(self, "_resume_files", None)
        self._stage_files = (
            resumed_files if resumed_files and resumed_files[0] == index else None
        )
        provenance = archive.provenance
        uncertainty_result = self._factorize_stage(
            index,
            view,
            space,
            final,
            archive,
            scale,
            optimizer,
            provenance,
            getattr(self, "_resume_units", None),
        )
        self._write_checkpoint(
            index,
            stage,
            view,
            space,
            model,
            plan.checkpoint.loss,
            stage_iteration=plan.start_iteration,
            completed=True,
            history=history,
        )
        records = [
            r
            for r in history.iterations
            if r.metrics.get("stage_index") == index and r.accepted is not False
        ]
        summary: Dict[str, Any] = dict(
            status=0,
            message="stage iteration budget reached by an earlier run",
            success=True,
        )
        summary.update(plan.summary or {})
        result = StageResult(
            index=index,
            name=label,
            frequencies=stage.frequencies,
            active=space.blocks,
            iterations=0,
            stage_iteration=plan.start_iteration,
            success=bool(summary["success"]),
            status=int(summary["status"]),
            message=str(summary["message"]),
            initial_loss=records[0].loss if records else plan.checkpoint.loss,
            final_loss=plan.checkpoint.loss,
            resumed=True,
            vector=final,
            space=space,
            uncertainty=uncertainty_result,
            metrics={
                "optimizer": getattr(optimizer, "kind", type(optimizer).__name__),
                "curvature_transfer": provenance,
                "finalized_after_resume": True,
                "elapsed_seconds": perf_counter() - started,
                "elapsed_scope": "end-of-stage factorizations and checkpoint writes",
            },
        )
        self.results.append(result)
        return result

    # -- run ------------------------------------------------------------------

    def run(self, resume: bool = True) -> FWIResult:
        """Run every stage and return the :class:`FWIResult`.

        With ``resume=True`` (default) a checkpoint written by an earlier run
        of the same problem and stage list is picked up: completed stages are
        skipped and an interrupted stage continues from its last accepted
        iterate with the remaining budget.  A checkpoint of a different
        problem, block layout, stage active set or frequencies is rejected.

        Stages with ``controls`` switch to a problem built with
        :meth:`ImagingProblem.with_controls` at their transition (the
        accepted state is transferred to the new layout);
        :attr:`FWIResult.problem` and :attr:`FWIResult.state` are those of
        the last stage.
        """

        self.results = []
        self._views = {}
        self._problems = {}
        self._stage_inputs = {}
        self._curvature_source = None
        self._curvature_output = None
        self._curvature_archive = None
        self._uncertainty_archive = None
        self._uncertainty_outputs = {}
        for key in (
            "_resume_curvature",
            "_resume_uncertainty",
            "_resume_optimizer",
            "_resume_local",
            "_resume_files",
            "_resume_units",
            "_stage_files",
            "_declarations",
            "_restarts",
        ):
            self.__dict__.pop(key, None)
        plan = self._resume_plan() if resume else None
        if plan is not None:
            # ``_resume_plan`` built the checkpointed stage's problem.
            self._problem_for(plan.stage_index).state = plan.state
            start, start_iteration = plan.start, plan.start_iteration
        else:
            start, start_iteration = 0, 0
            if self.problem.state is None:
                self.problem.linearize(gradient=False)  # registry discovery
        remaining = self.stages[start:]
        if not remaining:
            history = self._history
            if history is None:
                path = self._history_path()
                if path is not None and path.is_file():
                    history = OptimizationHistory.load(path)
                else:
                    history = OptimizationHistory(
                        path, metadata=self._history_metadata()
                    )
                self._history = history
            if history.status == "running":
                history.finish("converged", "every stage completed by an earlier run")
            stages = tuple(
                self._skipped_result(i, s, history) for i, s in enumerate(self.stages)
            )
            final_problem = self._problem_for(len(self.stages) - 1)
            assert final_problem.state is not None
            return FWIResult(
                state=final_problem.state,
                history=history,
                stages=stages,
                checkpoint=self.checkpoint_path,
                problem=final_problem,
            )
        history = self._open_history(resume=plan is not None)
        skipped = [
            self._skipped_result(i, s, history)
            for i, s in enumerate(self.stages[:start])
        ]
        expected = (
            None if plan is None or plan.start_iteration == 0 else plan.checkpoint.model
        )
        index_of = {s.label(i): i for i, s in enumerate(self.stages)}
        schedule = ContinuationSchedule(
            tuple(
                stage.to_continuation_stage(i) for i, stage in enumerate(self.stages)
            )[start:]
        )
        first_stage = self.stages[start]
        first_problem = self._problem_for(start)
        first_view = self._views.get(first_stage.label(start))
        if first_view is None:
            first_view = first_stage.view(first_problem)
            self._views[first_stage.label(start)] = first_view
        assert first_problem.state is not None
        initial = first_view.vector(first_problem.state).values

        def solve(cs: ContinuationStage, _model: np.ndarray) -> _ContinuationResult:
            index = index_of[cs.name]
            stage = self.stages[index]
            resumed = index == start
            if resumed and plan is not None and plan.finalize:
                result = self._finalize_stage(index, stage, plan)
            else:
                result = self._solve_stage(
                    index,
                    stage,
                    start_iteration if resumed else 0,
                    expected_model=expected if resumed else None,
                )
            view = self._views[cs.name]
            state = self._problem_for(index).state
            assert state is not None
            model = state.vector(view.space.without_support()).values
            return _ContinuationResult(model, result)

        def transition(
            previous: ContinuationStage,
            following: ContinuationStage,
            accepted: np.ndarray,
        ) -> Sequence[float]:
            # Write the accepted vector into the previous stage's state
            # (idempotent: the stage solve already did) and build the next
            # stage's view, which adopts fresh support masks from its first
            # linearize.  A stage with ``controls`` gets a new problem whose
            # state is the accepted state transferred to the new layout, so
            # the returned vector may differ in size from ``accepted``.
            previous_index = index_of[previous.name]
            previous_problem = self._problem_for(previous_index)
            previous_view = self._views[previous.name]
            assert previous_problem.state is not None
            previous_problem.state = previous_problem.state.with_update(
                previous_view.space.without_support(), accepted
            )
            next_index = index_of[following.name]
            next_problem = self._problem_for(next_index)
            next_view = self.stages[next_index].view(next_problem)
            self._views[following.name] = next_view
            assert next_problem.state is not None
            return cast(Sequence[float], next_view.vector(next_problem.state).values)

        try:
            run_continuation(
                schedule, cast(Sequence[float], initial), solve, transition=transition
            )
        except BaseException as exc:
            if history.status == "running":
                history.finish("stopped", f"{type(exc).__name__}: {exc}")
            raise
        stages = tuple(skipped) + tuple(self.results)
        success = all(s.success for s in self.results)
        history.finish("converged" if success else "stopped", stages[-1].message)
        final_problem = self._problem_for(len(self.stages) - 1)
        assert final_problem.state is not None
        return FWIResult(
            state=final_problem.state,
            history=history,
            stages=stages,
            checkpoint=self.checkpoint_path,
            problem=final_problem,
        )


# ---------------------------------------------------------------------------
# LSRTM
# ---------------------------------------------------------------------------

# Normal jobs spent on the 1/lambda_max step estimate of proximal LS-RTM.
_POWER_ITERATIONS = 2
# Rejected proximal trials per iteration (each costs a normal and a prox job).
_LINE_SEARCH_TRIALS = 10
# Fold native Tikhonov Hessian products into the Sauce normal job through
# ``Linearization.apply_normal(..., regularization=...)``; requires a Sauce build
# whose normal action carries the regularization request outside the linearization
# fingerprint.
_FUSE_NATIVE_REGULARIZATION = True
# Sauce task-result ``misc`` counters reported in ``LSRTM.info["background"]``.
_BACKGROUND_COUNTERS = {
    "background_solves_reused": "reused_solves",
    "background_reuse_misses": "misses",
}


def _receiver_probe_request(preconditioner: Any) -> Optional[Dict[str, Any]]:
    """Return the linearize request behind ``Diagonal(probes="receiver")``."""

    from .regularization import Diagonal

    if isinstance(preconditioner, Diagonal) and preconditioner.probes == "receiver":
        return {"probes": preconditioner.probe_count, "seed": preconditioner.seed or 0}
    return None


def _normal_with_regularization(
    lin: Any, native: Any, space: ControlSpace
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return ``x -> (Re(J^H W J) + R'') x`` from one normal job, if supported.

    A linearization whose ``apply_normal`` accepts ``regularization=`` folds
    the native Tikhonov Hessian into the Sauce normal action, instead of one
    native regularization job per product (a new batch submission on SLURM
    sites and a full control vector transferred each way).
    """

    apply = getattr(lin, "apply_normal", None)
    if not _FUSE_NATIVE_REGULARIZATION or native is None or apply is None:
        return None
    try:
        parameters = inspect.signature(apply).parameters
    except (TypeError, ValueError):
        return None
    if "regularization" not in parameters:
        return None

    def action(x: np.ndarray) -> np.ndarray:
        result = apply(ControlVector(x, space), regularization=native)
        values = getattr(result, "values", result)
        return _real_vector(values, size=space.size, name="normal product")

    return action


def _preconditioned_cg(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    *,
    iterations: int,
    tolerance: float,
    precondition: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    callback: Optional[Callable[[np.ndarray], None]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve ``A x = b`` from ``x = 0`` by safeguarded preconditioned CG.

    Each iteration applies ``A`` once and nothing else does: convergence is
    tested on the recurrence residual after every update (a solve converging
    on its last allowed iteration succeeds), and the final residual is
    reported from the recurrence rather than recomputed. ``r^T M r <= 0``
    raises because the preconditioner is not SPD. ``p^T A p <= 0`` (an
    indefinite operator, or an FP32 normal that is only approximately
    symmetric) stops with status ``-1`` and returns the previous iterate.
    Status is ``0`` when converged and ``iterations`` at the limit.
    """

    b = _real_vector(rhs, name="right-hand side")
    x = np.zeros_like(b)
    b_norm = float(np.linalg.norm(b))
    info: Dict[str, Any] = {
        "iterations": 0,
        "status": 0,
        "converged": True,
        "residual_norm": b_norm,
        "rhs_norm": b_norm,
        "message": "zero right-hand side",
    }
    if b_norm == 0.0:
        return x, info
    if not math.isfinite(b_norm):
        raise ValueError("CG right-hand side must be finite")

    def preconditioned(r: np.ndarray) -> Tuple[np.ndarray, float]:
        z = (
            r
            if precondition is None
            else _real_vector(precondition(r), size=r.size, name="preconditioner")
        )
        rho = float(r @ z)
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError(
                f"LSRTM preconditioner is not SPD: r^T M r = {rho:.6g} for a "
                "nonzero residual"
            )
        return z, rho

    r = b.copy()
    z, rho = preconditioned(r)
    p = np.array(z, copy=True)
    residual_norm = b_norm
    status, message, done = iterations, "iteration limit reached", 0
    for k in range(1, iterations + 1):
        q = _real_vector(matvec(p), size=b.size, name="operator product")
        curvature = float(p @ q)
        if not math.isfinite(curvature):
            raise ValueError("CG operator product is not finite")
        if curvature <= 0.0:
            status = -1
            message = (
                f"nonpositive curvature p^T A p = {curvature:.6g} at iteration {k}; "
                "the operator is indefinite or inaccurate along this direction"
            )
            break
        alpha = rho / curvature
        x += alpha * p
        r -= alpha * q
        del q
        done = k
        residual_norm = float(np.linalg.norm(r))
        if callback is not None:
            callback(x)
        if residual_norm <= tolerance * b_norm:
            status, message = 0, "relative residual tolerance reached"
            break
        if k == iterations:
            break
        z, rho_next = preconditioned(r)
        p *= rho_next / rho
        p += z
        rho = rho_next
    info.update(
        iterations=done,
        status=status,
        converged=status == 0,
        residual_norm=residual_norm,
        message=message,
    )
    return x, info


def _task_counters(job: Any) -> Optional[Dict[str, int]]:
    """Sum the background ``misc`` counters of a job's local task results."""

    from frequensolve.simulation.artifact_contract import (
        ArtifactContractError,
        TaskResult,
        task_result_path,
    )

    root = getattr(job, "_result_path", None)
    if root is None:
        return None
    totals: Optional[Dict[str, int]] = None
    for task in range(1, int(getattr(job, "n_tasks", 1) or 1) + 1):
        path = task_result_path(root, task)
        if not path.is_file():
            continue
        try:
            misc = TaskResult.read(path, result_path=root).metadata.get("misc", {})
        except (ArtifactContractError, OSError, ValueError):
            continue
        totals = {} if totals is None else totals
        for key in _BACKGROUND_COUNTERS:
            value = misc.get(key, 0)
            if isinstance(value, (int, float)) and math.isfinite(value):
                totals[key] = totals.get(key, 0) + int(value)
    return totals


class _RealJacobian(LinearOperator):
    """``[Re; Im]`` stacking of a complex Jacobian over a real domain.

    ``matvec`` interleaves real and imaginary parts (optionally scaled by
    ``sqrt(w)`` per row) so ``||A x - b||`` equals the weighted complex
    residual norm; ``rmatvec`` restores the complex dual and applies
    ``J.H``.  Optional regularization rows ``R`` are appended.
    """

    def __init__(
        self,
        jacobian: Any,
        row_weights: np.ndarray,
        regularization_operator: Any = None,
    ) -> None:
        self.jacobian = jacobian
        self.sqrt_weights = np.sqrt(np.asarray(row_weights, dtype=np.float64))
        self.regularization_operator = regularization_operator
        rows, cols = jacobian.shape
        extra = (
            0
            if regularization_operator is None
            else int(regularization_operator.shape[0])
        )
        self.data_rows = 2 * int(rows)
        super().__init__(np.dtype(np.float64), (self.data_rows + extra, int(cols)))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        data = np.asarray(self.jacobian @ x).reshape(-1) * self.sqrt_weights
        out = np.stack((data.real, data.imag), axis=-1).reshape(-1)
        if self.regularization_operator is not None:
            out = np.concatenate(
                [
                    out,
                    np.asarray(
                        self.regularization_operator @ x, dtype=np.float64
                    ).reshape(-1),
                ]
            )
        return out

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        pairs = y[: self.data_rows].reshape(-1, 2)
        dual = (pairs[:, 0] + 1j * pairs[:, 1]) * self.sqrt_weights
        out = np.asarray(self.jacobian.H @ dual, dtype=np.float64).reshape(-1)
        if self.regularization_operator is not None:
            out = out + np.asarray(
                self.regularization_operator.H @ y[self.data_rows :], dtype=np.float64
            ).reshape(-1)
        return out

    def pack(self, data: Any) -> np.ndarray:
        """Realify a complex data vector with the row weights applied."""

        values = np.asarray(data, dtype=np.complex128).reshape(-1) * self.sqrt_weights
        return np.stack((values.real, values.imag), axis=-1).reshape(-1)


class LSRTM:
    """Least-squares reverse-time migration around one linearization.

    Solves ``min_dm 0.5 ||J dm + r||_W^2 + 0.5 damping ||dm||^2 + P(dm)`` with
    ``r = F(v0) - d`` (``J``, ``W`` and ``r`` frozen at ``v0``). Each solver
    iteration costs one Sauce ``normal`` job (a Born forward plus adjoint
    solve set over all frequencies) or one JVP/VJP pair:

    - ``method="cg"``, and any smooth native regularization (``Tikhonov``)
      whatever ``method``: preconditioned conjugate gradients on
      ``(H + damping I + P'') dm = -(g + P'(0))`` from ``lin.normal`` and
      ``lin.gradient`` only. Each native Tikhonov Hessian product is one
      native regularization job unless the normal job applies it
      (``info["fused_regularization"]``). ``info["method"]`` is ``"cg"``.
    - ``method="lsqr"`` otherwise: SciPy's LSQR on the real-stacked Jacobian
      with the data-space residual from ``problem.residual(v0)``; a
      regularization must expose ``operator()`` and is appended as extra rows
      ``||R dm||^2``.
    - Nonsmooth TV/TGV: composite proximal-gradient iterations with Sauce
      solving each constrained proximal problem (``"proximal_gradient"``).
      The first step is ``1/lambda_max`` of the frozen normal (two power
      iterations: two normal jobs), every trial costs one normal and one
      proximal job, and the stopping test is relative to the initial
      proximal-gradient mapping, so rescaling the data or image units does
      not change the iterates.

    ``tolerance`` is relative for every method: the CG recurrence residual
    against its right-hand side, LSQR's ``atol``/``btol`` and the
    proximal-gradient mapping against its initial value.

    CG stops on nonpositive curvature ``p^T A p <= 0`` with
    ``info["status"] == -1`` (an FP32 normal at loose solver tolerance is
    only approximately symmetric) and raises when the preconditioner is not
    SPD. ``info["jobs"]`` counts the jobs this run submitted by action
    (``regularization_<operation>`` for native regularization callbacks) and
    ``info["background"]`` reports whether background reuse was used (it
    needs ``reuse_background``, an eligible problem and a site with
    ``supports_background_reuse``), the forward solves Sauce restored from
    the checkpoint, the cache misses it recomputed (``misses``; a
    :class:`RuntimeWarning` is issued) and the checkpoint bytes released.

    Args:
        problem: Problem (typically over ``GridParameters``/reflectivity).
        iterations: Solver iteration limit.
        regularization: Optional regularization (``bind(space)`` protocol) on the image.
        method: ``"lsqr"`` or ``"cg"``.
        damping: Tikhonov damping ``||dm||^2`` weight.
        preconditioner: Fixed SPD inverse metric (``bind(space)`` protocol),
            applied inside CG. ``Diagonal(probes="receiver", probe_count=16)``
            requests shared receiver probes with the initial gradient.
        reuse_background: Retain native background fields for subsequent
            ordinary material JVP/VJP/normal actions on the same partition;
            unsupported controls/objectives and sites automatically use
            uncached actions.
        keep_background: Keep the background checkpoint this run created.
            By default it is deleted when the run ends, locally and in the
            site's result directory (full forward-field checkpoints hold
            ``n_dof * n_src`` complex values per frequency). Kept checkpoints
            are deleted when their linearization leaves ``problem.cache`` or
            exceeds ``problem.cache.background_budget``.
        tolerance: Relative solver tolerance.
        callback: Called with the current image after every iteration.
    """

    def __init__(
        self,
        problem: ImagingProblem,
        iterations: int = 15,
        *,
        regularization: Any = None,
        method: str = "lsqr",
        damping: float = 0.0,
        tolerance: float = 1.0e-4,
        preconditioner: Any = None,
        reuse_background: bool = True,
        keep_background: bool = False,
        callback: Optional[Callable[[ControlVector], None]] = None,
    ) -> None:
        self.problem = problem
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        self.regularization = regularization
        method = str(method).strip().lower()
        if method not in {"lsqr", "cg"}:
            raise ValueError("method must be 'lsqr' or 'cg'")
        self.method = method
        self.damping = float(damping)
        if not math.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and nonnegative")
        self.tolerance = float(tolerance)
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if preconditioner is not None and method != "cg":
            raise ValueError("LSRTM preconditioning currently requires method='cg'")
        self.preconditioner = preconditioner
        self.reuse_background = bool(reuse_background)
        self.keep_background = bool(keep_background)
        self.bound_preconditioner: Any = None
        self.callback = callback
        self.info: Dict[str, Any] = {}
        self.linearization: Optional[Linearization] = None
        self._background: Dict[str, Any] = {}

    def run(self, v0: Any = None) -> ControlVector:
        """Return the image ``dm`` on the problem space, linearized at ``v0``."""

        self.info = {}
        self.linearization = None
        recording = getattr(self.problem.backend, "recording", None)
        empty: List[Any] = []
        context = contextlib.nullcontext(empty) if recording is None else recording()
        with context as jobs:
            try:
                image = self._solve(v0)
            finally:
                released = self._release_background(jobs)
        self.info.update(self._job_diagnostics(jobs, released))
        return image

    def _background_decision(self) -> Tuple[bool, Optional[str]]:
        """Use background checkpoints only where problem and site support them."""

        if not self.reuse_background:
            return False, "reuse_background=False"
        site = getattr(self.problem.backend, "site", None)
        if not getattr(site, "supports_background_reuse", False):
            return False, f"{type(site).__name__} does not support background reuse"
        if not self.problem._supports_background_reuse():
            return False, "controls, objective or patches require uncached actions"
        return True, None

    def _release_background(self, jobs: Sequence[Any]) -> Optional[int]:
        """Delete the background checkpoint this run created (unless kept)."""

        lin = self.linearization
        job = None if lin is None else lin.job
        release = getattr(self.problem.backend, "release_background", None)
        if (
            self.keep_background
            or job is None
            or release is None
            or getattr(job, "background", None) is None
            or not any(item is job for item in jobs)
        ):
            return None
        try:
            return int(release(job))
        except Exception as exc:  # cleanup must not discard a computed image
            warnings.warn(
                f"LS-RTM could not delete the background checkpoint of "
                f"{job.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            return None

    def _job_diagnostics(
        self, jobs: Sequence[Any], released: Optional[int]
    ) -> Dict[str, Any]:
        """Count this run's jobs and sum Sauce's background reuse counters."""

        from .jobs import RegularizationJob

        counts: Dict[str, int] = {}
        totals: Optional[Dict[str, int]] = None
        for job in jobs:
            kind = (
                f"regularization_{job.operation}"
                if isinstance(job, RegularizationJob)
                else str(
                    getattr(job, "action", None)
                    or getattr(job, "workflow", None)
                    or type(job).__name__
                )
            )
            counts[kind] = counts.get(kind, 0) + 1
            if getattr(job, "background", None) is None:
                continue
            counters = _task_counters(job)
            if counters is not None:
                totals = {} if totals is None else totals
                for key, value in counters.items():
                    totals[key] = totals.get(key, 0) + value
        background = dict(self._background)
        for key, label in _BACKGROUND_COUNTERS.items():
            background[label] = None if totals is None else totals.get(key, 0)
        background["released_bytes"] = released
        if background.get("misses"):
            warnings.warn(
                f"LS-RTM background reuse missed {background['misses']} source "
                "batch(es); Sauce recomputed their forward fields (see the "
                "FS_BACKGROUND_REUSE_MISS warnings in the task logs)",
                RuntimeWarning,
                stacklevel=3,
            )
        return {"jobs": counts, "background": background}

    def _solve(self, v0: Any) -> ControlVector:
        # Operators need the covector parts (per-task registry fingerprints),
        # so the linearization always carries the gradient.
        reuse_background, reason = self._background_decision()
        self._background = {
            "requested": self.reuse_background,
            "enabled": reuse_background,
            "reason": reason,
        }
        lin = self.problem.linearize(
            v0,
            gradient=True,
            receiver_diagonal=_receiver_probe_request(self.preconditioner),
            background=reuse_background,
        )
        if not reuse_background:
            # A replay view must not reuse checkpoint-derived action memoization.
            lin = copy.copy(lin)
            lin._reuse_background = False
            lin._ops = {}
            lin._normal = lin._jacobian = None
        self.linearization = lin
        space = lin.space
        from types import SimpleNamespace

        from ._native_regularization import bind_workflow_regularization

        # LSRTM regularizes the image/update; frozen image coefficients are zero.
        image_state = ControlState(
            lin.state.space,
            np.zeros_like(lin.state.values),
            scaling=lin.state.scaling,
            scaling_units=lin.state.scaling_units,
        )
        specification = (
            self.problem.smoothing
            if self.regularization is None
            else self.regularization
        )
        bound, native = bind_workflow_regularization(
            specification,
            space,
            self.problem,
            SimpleNamespace(job=lin.job, state=image_state),
        )
        if native is not None and not native.is_smooth:
            if self.preconditioner is not None:
                raise ValueError(
                    "TV/TGV use proximal iterations; CG preconditioning requires quadratic regularization"
                )
            return self._run_proximal(lin, space, bound, native)
        # Native Tikhonov has no explicit rows for LSQR; its exact Hessian
        # makes the regularized normal equations a CG problem.
        if native is not None or self.method == "cg":
            return self._run_cg(lin, space, bound, native)
        return self._run_lsqr(lin, space, bound)

    def _run_proximal(
        self, lin: Linearization, space: ControlSpace, bound: Any, native: Any
    ) -> ControlVector:
        """Composite proximal gradient on the frozen quadratic data model.

        The data model ``q(x) = g0.x + x.Hx/2`` is evaluated from steps off
        the current iterate, ``q(x + s) = q(x) + (g0 + Hx).s + s.Hs/2``: one
        normal job per trial, with decreases and curvatures from ``s.Hs``
        directly, so approximate (FP32) normal products never enter the
        majorization test through the cancellation of ``O(q)`` values.
        """

        assert lin.gradient is not None
        size = space.size
        g0 = _real_vector(lin.gradient.values, size=size, name="gradient").copy()
        damping = self.damping
        zero = np.zeros(size)

        def normal(direction: np.ndarray) -> np.ndarray:
            product = lin.normal @ direction
            return _real_vector(product, size=size, name="normal product")

        def key(x: np.ndarray) -> bytes:
            data = np.ascontiguousarray(x).data
            return hashlib.blake2b(data, digest_size=20).digest()

        # Start from the inverse of the largest curvature: power iterations
        # from the steepest-descent direction, one normal job each.
        curvature = None
        start = g0.copy()
        if bound is not None:
            origin = ControlVector(zero, space)
            start += _real_vector(
                bound.gradient(origin), size=size, name="regularization gradient"
            )
            try:
                curvature = bound.hessian_operator(origin)
            except NotImplementedError:
                curvature = None
        if not np.any(start):
            start = space.random(0).values
        direction = start / float(np.linalg.norm(start))
        largest = 0.0
        for _ in range(_POWER_ITERATIONS):
            product = normal(direction) + damping * direction
            if curvature is not None:
                product += _real_vector(
                    curvature @ direction, size=size, name="regularization product"
                )
            norm = float(np.linalg.norm(product))
            if not math.isfinite(norm) or norm == 0.0:
                break
            largest = norm
            direction = product / norm
        del start, direction
        initial_step = 1.0 / largest if largest > 0.0 else 1.0

        base: Dict[str, Any] = {"key": key(zero), "x": zero, "q": 0.0, "dq": g0}
        trial: Dict[str, Any] = {}

        def model(x: np.ndarray) -> Tuple[bytes, float, np.ndarray]:
            name = key(x)
            if name == base["key"]:
                return name, base["q"], base["dq"]
            if trial.get("key") != name:
                step = x - base["x"]
                hs = normal(step)
                q = base["q"] + float(base["dq"] @ step) + 0.5 * float(step @ hs)
                trial.update(key=name, q=q, dq=base["dq"] + hs)
            return name, trial["q"], trial["dq"]

        def smooth_value(x: np.ndarray) -> float:
            _, q, _ = model(x)
            value = lin.value + q + 0.5 * damping * float(x @ x)
            if bound is not None:
                value += float(bound.value(ControlVector(x, space)))
            return value

        def smooth_gradient(x: np.ndarray) -> np.ndarray:
            name, q, dq = model(x)
            if name != base["key"]:
                # Gradients are requested at accepted iterates only; the next
                # trials step from this point.
                base.update(key=name, x=np.array(x, copy=True), q=q, dq=dq)
                trial.clear()
            gradient = dq + damping * x
            if bound is not None:
                gradient = gradient + _real_vector(
                    bound.gradient(ControlVector(x, space)),
                    size=size,
                    name="regularization gradient",
                )
            return gradient

        def iteration(it: InexactNewtonIteration) -> None:
            if self.callback is not None:
                self.callback(ControlVector(it.model, space))

        # Read-only, like the optimizer's box: metric and bound files are staged once.
        metric = _read_only(np.ones(size))
        result = minimize_proximal_gradient(
            smooth_value,
            smooth_gradient,
            native.value,
            lambda x, tau, box: native.prox(x, tau, metric, box),
            zero,
            options=InexactNewtonOptions(
                max_iterations=self.iterations,
                max_line_search_trials=_LINE_SEARCH_TRIALS,
                gradient_tolerance=0.0,
                objective_tolerance=0.0,
                step_tolerance=0.0,
            ),
            callback=iteration,
            initial_step=initial_step,
            relative_tolerance=self.tolerance,
            curvature_steps=True,
        )
        self.info = {
            "method": "proximal_gradient",
            "iterations": result.iterations,
            "status": result.status,
            "converged": result.success,
            "message": result.message,
            "objective": result.objective,
            "initial_step": initial_step,
            "objective_evaluations": result.objective_evaluations,
        }
        return ControlVector(result.model, space)

    def _run_cg(
        self,
        lin: Linearization,
        space: ControlSpace,
        bound: Any,
        native: Any = None,
    ) -> ControlVector:
        zero = space.zeros()
        size = space.size
        assert lin.gradient is not None
        rhs = -_real_vector(lin.gradient.values, size=size, name="gradient")
        curvature = None
        if bound is not None:
            rhs = rhs - _real_vector(
                bound.gradient(zero), size=size, name="regularization"
            )
            curvature = bound.hessian_operator(zero)
        fused = _normal_with_regularization(lin, native, space)
        native_curvature = None
        if native is not None:
            # Zero, without a native job, unless the Tikhonov term has a reference.
            rhs = rhs - _real_vector(
                native.gradient(zero), size=size, name="native regularization"
            )
            if fused is None:
                native_curvature = native.hessian_operator(zero)
        normal = lin.normal
        damping = self.damping

        def matvec(x: np.ndarray) -> np.ndarray:
            if fused is not None:
                out = fused(x)
            else:
                out = _real_vector(normal @ x, size=size, name="normal product")
            out = out + damping * x
            for operator in (native_curvature, curvature):
                if operator is not None:
                    out = out + _real_vector(
                        operator @ x, size=size, name="regularization product"
                    )
            return out

        precondition: Optional[Callable[[np.ndarray], np.ndarray]] = None
        if self.preconditioner is not None:
            from .regularization import Quadratic, _BoundSum

            bound_metric = self.preconditioner.bind(space)
            parts = [term for term in (bound, native) if term is not None]
            if damping:
                from scipy.sparse import eye

                parts.append(Quadratic(eye(size), weight=damping).bind(space))
            # The metric only reads the terms' curvature diagonals.
            metric_regularization = (
                None
                if not parts
                else (
                    parts[0]
                    if len(parts) == 1
                    else _BoundSum(cast(Any, None), space, parts)
                )
            )
            # Image regularization is evaluated at zero, not the background.
            image_linearization = copy.copy(lin)
            image_linearization.point = zero
            bound_metric.update(
                image_linearization, regularization=metric_regularization
            )
            self.bound_preconditioner = bound_metric

            def apply_metric(r: np.ndarray) -> np.ndarray:
                return np.asarray(bound_metric.apply(ControlVector(r, space)).values)

            precondition = apply_metric

        def on_iteration(x: np.ndarray) -> None:
            if self.callback is not None:
                self.callback(ControlVector(x, space))

        solution, info = _preconditioned_cg(
            matvec,
            rhs,
            iterations=self.iterations,
            tolerance=self.tolerance,
            precondition=precondition,
            callback=on_iteration,
        )
        self.info = {
            "method": "cg",
            "requested_method": self.method,
            "preconditioned": precondition is not None,
            "fused_regularization": fused is not None,
            **info,
        }
        return ControlVector(solution, space)

    def _run_lsqr(
        self, lin: Linearization, space: ControlSpace, bound: Any
    ) -> ControlVector:
        try:
            residual = lin.objective_residual()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "LSRTM with method='lsqr' needs the saved objective-space residual; "
                "regenerate the linearization with a current Sauce build or use "
                "method='cg' which works from lin.normal and lin.gradient"
            ) from exc
        regularization_operator = None
        if bound is not None:
            regularization_operator = bound.operator()
            if regularization_operator is None:
                raise ValueError(
                    "LSRTM with method='lsqr' needs a regularization exposing operator(); "
                    "use method='cg' for this regularization"
                )
        weights = np.ones(lin.data_space.size, dtype=np.float64)
        weights = lin.weight_data(
            DataVector(weights.astype(np.complex128), lin.data_space)
        ).values.real
        operator = _RealJacobian(lin.jacobian, weights, regularization_operator)
        b = np.concatenate(
            [
                -operator.pack(residual.values),
                (
                    np.zeros(0, dtype=np.float64)
                    if bound is None
                    else -np.asarray(bound.residual(space.zeros()))
                ),
            ]
        )
        solution, istop, itn, r1norm, r2norm, *_ = scipy_lsqr(
            operator,
            b,
            damp=math.sqrt(self.damping),
            atol=self.tolerance,
            btol=self.tolerance,
            iter_lim=self.iterations,
        )
        image = ControlVector(np.asarray(solution, dtype=np.float64), space)
        self.info = {
            "method": "lsqr",
            "iterations": int(itn),
            "status": int(istop),
            "converged": int(istop) in {1, 2},
            "residual_norm": float(r1norm),
            "damped_residual_norm": float(r2norm),
            "rhs_norm": float(np.linalg.norm(b)),
        }
        if self.callback is not None:
            self.callback(image)
        return image


# ---------------------------------------------------------------------------
# single-linearization workflows
# ---------------------------------------------------------------------------


def rtm(
    problem: ImagingProblem, v: Any = None, *, illumination: Any = None
) -> ControlVector:
    """Return the misfit gradient (RTM image on the control space) at ``v``.

    Equals ``problem.gradient(v)``: the objective's real control covector
    (the raw objective derivative). The Jacobian and its frozen
    comparison residual carry Sauce's objective-space sign and normalization.
    """

    lin = problem.linearize(v, gradient=True)
    assert lin.gradient is not None
    return (
        lin.gradient
        if illumination is None
        else lin.illumination(illumination).apply(lin.gradient)
    )


def _simulation_for(problem: ImagingProblem, v: Any) -> Any:
    """Return the simulation with the state at ``v`` installed."""

    if v is None and problem.is_authored():
        return problem.simulation
    return problem.simulation_at(v)


def _observed_paths(problem: ImagingProblem) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for group in problem.observed_groups:
        ref = group.observed
        if ref is None:
            raise ValueError(f"observed group {group.name!r} has no observed data")
        paths[group.name] = (
            Path(ref.file) if isinstance(ref, TraceStoreRef) else Path(ref)
        )
    return paths


def sensitivity_kernel_job(
    problem: ImagingProblem,
    grid: Union[CartesianGrid, Mapping[str, Any]],
    *,
    properties: Sequence[str] = ("vp",),
    condition: str = "fwi",
    frequencies: Optional[Iterable[Any]] = None,
    observed: Any = None,
    v: Any = None,
    keep: str = "none",
    smoothing: Any = None,
    weights: Optional[Sequence[float]] = None,
    name: Optional[str] = None,
) -> ImageKernelJob:
    """Build the :class:`ImageKernelJob` behind :func:`sensitivity_kernel`.

    Args:
        problem: Problem providing simulation, misfit, observed data and site.
        grid: Cartesian image grid.
        properties: One image per property (``fwi`` conditions).
        condition: Imaging condition; ``"fwi"`` resolves to
            ``fwi:<physics>``; anything else is passed verbatim.
        frequencies: Subset of the problem frequencies (default: all).
        observed: ``None`` for zero-data sensitivity kernels, ``True`` for
            the problem's observed data, or an explicit path / mapping.
        v: Linearization point (default: the current state).
        keep: Field retention ``"none"``/``"forward"``/``"adjoint"``/``"all"``.
        smoothing: Image smoothing (``Imaging.Smoothing``); defaults to the
            problem's smoothing configuration.
        weights: Optional per-frequency weights.
        name: Job name (default from the backend).
    """

    if condition == "fwi":
        physics = getattr(problem.simulation, "physics", None)
        condition = f"fwi:{physics}" if physics else "fwi"
    names = [
        str(p) for p in ([properties] if isinstance(properties, str) else properties)
    ]
    if not names:
        raise ValueError("sensitivity_kernel requires at least one property")
    images = {prop: ImageSpec(condition, prop) for prop in names}
    view = problem if frequencies is None else problem.restrict(frequencies=frequencies)
    groups = problem.observed_groups
    if observed is True:
        observed_arg: Any = _observed_paths(problem)
        misfit: Any = problem._misfit_payload
    elif observed is None:
        observed_arg = None
        # Observed RMS is undefined without observations. Retain the reduction
        # and all other term settings, using a unit scale for zero-data kernels.
        kernel_misfit = copy.copy(problem.misfit)
        kernel_misfit._terms = tuple(
            (
                dataclasses.replace(
                    term,
                    normalization=Normalization(
                        kind="explicit",
                        value=1.0,
                        reduction=term.normalization.reduction,
                    ),
                )
                if term.normalization.kind == "observed_rms"
                else term
            )
            for term in problem.misfit.objective_terms([g.name for g in groups])
        )
        misfit = _MisfitPayload(
            kernel_misfit,
            [dataclasses.replace(g, observed=None, derivatives={}) for g in groups],
        )
    else:
        observed_arg = observed
        misfit = problem._misfit_payload
    smoothing = problem.smoothing if smoothing is None else smoothing
    keep = str(keep).strip().lower()
    return ImageKernelJob(
        name or problem.backend.job_name("kernel"),
        _simulation_for(problem, v),
        list(view.frequencies),
        grid=grid,
        images=images,
        observed=observed_arg,
        misfit=misfit,
        weights=None if weights is None else list(weights),
        smoothing=smoothing,
        field_retention=None if keep == "none" else keep,
        workflow="rtm",
    )


def sensitivity_kernel(
    problem: ImagingProblem,
    grid: Union[CartesianGrid, Mapping[str, Any]],
    *,
    properties: Sequence[str] = ("vp",),
    condition: str = "fwi",
    frequencies: Optional[Iterable[Any]] = None,
    observed: Any = None,
    v: Any = None,
    keep: str = "none",
    smoothing: Any = None,
    weights: Optional[Sequence[float]] = None,
) -> ImageSet:
    """Run a Cartesian sensitivity-kernel (``Imaging.grid``) job and return its images.

    With ``observed=None`` Sauce uses zero observed data, so the images are
    the pure model sensitivity kernels; observed-RMS normalization uses a
    unit scale while retaining its reduction. ``observed=True`` images the misfit
    residual instead.  See :func:`sensitivity_kernel_job` for the arguments.
    """

    job = sensitivity_kernel_job(
        problem,
        grid,
        properties=properties,
        condition=condition,
        frequencies=frequencies,
        observed=observed,
        v=v,
        keep=keep,
        smoothing=smoothing,
        weights=weights,
    )
    problem.backend.run(job)
    return job.load_images()
