"""Imaging workflows: staged FWI, LSRTM, RTM, sensitivity kernels, focusing.

The workflow layer owns the outer loop of an inversion while
:class:`~frequensolve.imaging.problem.ImagingProblem` owns the physics:

- :class:`Stage` names the frequencies, active blocks and iteration budget of
  one continuation stage (plus optional loss, penalty, smoothing, optimizer
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
- :class:`LSRTM`, :func:`rtm`, :func:`sensitivity_kernel` and
  :class:`TimeReversalFocus` are the single-linearization workflows.

Sign convention: Sauce's covector is the gradient of the misfit, i.e. the
adjoint applied to ``simulated - observed``.  :func:`rtm` returns that
gradient; the classic RTM image in the ``observed - simulated`` convention is
its negative.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
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
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import lsqr as scipy_lsqr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.imaging._artifacts import (
    ControlVectorFile,
    ImageSet,
    SmoothingConfig,
    unqualified_block_name,
)
from frequensolve.imaging._backend import fingerprint
from frequensolve.imaging.controls import (
    ControlSpace,
    ControlState,
    ControlVector,
    _BlockSpec,
)
from frequensolve.imaging.data import DataVector, TraceStoreRef
from frequensolve.imaging.jobs import ControlGradientJob, ImageKernelJob, ImageSpec
from frequensolve.imaging.misfit import Loss, Misfit, Normalization
from frequensolve.imaging.problem import ImagingProblem, Linearization, _MisfitPayload
from frequensolve.imaging.results import FWIResult, StageResult
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
    minimize_inexact_newton,
    minimize_lbfgs,
)

__all__ = [
    "FWI",
    "FWIIteration",
    "LBFGS",
    "LSRTM",
    "NewtonCG",
    "Stage",
    "TimeReversalFocus",
    "block_curvatures",
    "curvature_scaling",
    "rms_step_limit",
    "rtm",
    "sensitivity_kernel",
    "sensitivity_kernel_job",
]

CHECKPOINT_SCHEMA = "fs-imaging-fwi-checkpoint-1"


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


def _digest(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


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
        penalty: Penalty overriding the workflow-level penalty.
        smoothing: Gradient smoothing overriding the workflow/problem
            smoothing; ``False`` switches smoothing off for the stage.
        optimizer: :class:`LBFGS` / :class:`NewtonCG` overriding the
            workflow optimizer.
        weights: One nonnegative objective weight per stage frequency.
        name: Stage label (default ``stage_<index>``).
        min_support: Relative support threshold for the stage.
        metadata: Free-form scalar metadata recorded with the stage.
        controls: Change the control layout from this stage on: a complete
            :class:`~frequensolve.imaging.controls.ControlSpace` or a mapping
            ``{block key: new block spec}`` replacing those blocks (e.g.
            ``{"vp": im.DepthProfile("vp", "sediment", spacing=25*u.m)}``).
            :class:`FWI` builds the stage problem with
            :meth:`ImagingProblem.with_controls`, transferring the accepted
            state of the previous stage; later stages keep the new layout
            until another stage changes it.
    """

    frequencies: Tuple[Any, ...]
    iterations: int
    active: Optional[Tuple[str, ...]] = None
    loss: Optional[Loss] = None
    misfit: Optional[Misfit] = None
    penalty: Any = None
    smoothing: Any = None
    optimizer: Any = None
    weights: Optional[Tuple[float, ...]] = None
    name: Optional[str] = None
    min_support: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    controls: Any = None

    def __post_init__(self) -> None:
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
            object.__setattr__(
                self, "smoothing", SmoothingConfig.from_value(self.smoothing)
            )
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

    ``step_limit`` caps the RMS update of every block per iteration
    (optimizer coordinates); ``scaling_max_ratio`` floors curvature-based
    scaling (see :func:`curvature_scaling`); ``preconditioner_refresh``
    re-estimates a bound preconditioner every that many accepted iterations
    inside :class:`FWI`.
    """

    max_iterations: Optional[int] = None
    gradient_tolerance: float = 1.0e-6
    step_tolerance: float = 1.0e-9
    objective_tolerance: float = 1.0e-9
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
        history: Optional[OptimizationHistory] = None,
        callback: Optional[Callable[[InexactNewtonIteration], None]] = None,
        max_iterations: Optional[int] = None,
        step_limit: Optional[float] = None,
        scaling: Optional[Any] = None,
        block_slices: Optional[Sequence[slice]] = None,
        space: Optional[ControlSpace] = None,
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
            history: Optional history receiving one iteration record per
                accepted iterate (``objective.loss`` required).
            callback: Called with every accepted
                :class:`InexactNewtonIteration` (physical coordinates).
            max_iterations: Iteration budget overriding the configuration.
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
        hessian_fn = hessian_action
        if hessian_fn is None:
            hessian_fn = getattr(objective, "hessian_action", None)
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
            assert hessian_fn is not None
            out = _real_vector(
                hessian_fn(to_physical(y), to_physical(dy)),
                size=size,
                name="Hessian product",
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

        def unscale_iteration(it: InexactNewtonIteration) -> InexactNewtonIteration:
            if scale is None:
                return it
            return dataclasses.replace(
                it,
                model=to_physical(it.model),
                gradient=it.gradient / scale,
                linearization_gradient=it.linearization_gradient / scale,
                step=to_physical(it.step),
                raw_step=to_physical(it.raw_step),
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
        kwargs: Dict[str, Any] = {
            "bounds": scaled_bounds,
            "options": self.options(max_iterations),
            "preconditioner": None if precondition is None else p,
            "step_limit": None if limit is None else step_limit_fn,
            "callback": on_iteration,
        }
        y0 = cast(Sequence[float], to_scaled(x0))
        if self.kind == "lbfgs":
            result = minimize_lbfgs(f, g, y0, **kwargs)
        else:
            if hessian_fn is None:
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
    """Cached misfit-plus-penalty objective of one stage.

    One ``linearize`` (value and covector) per distinct point; the value,
    gradient, Hessian action (Gauss-Newton normal plus penalty Hessian) and
    loss terms of the last point are reused.  Every new evaluation is
    appended to the history with the stage metrics.
    """

    def __init__(
        self,
        view: ImagingProblem,
        space: ControlSpace,
        penalty: Any,
        history: Optional[OptimizationHistory],
        metrics: Mapping[str, Any],
    ) -> None:
        self.view = view
        self.space = space
        self.penalty = penalty
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
        if self.penalty is not None:
            regularization = float(self.penalty.value(vector))
            gradient += _real_vector(
                self.penalty.gradient(vector), size=point.size, name="penalty gradient"
            )
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
        if self.penalty is not None:
            operator = self.penalty.hessian_operator(self.vector(x))
            out += _real_vector(
                operator @ direction, size=out.size, name="penalty Hessian product"
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


@dataclass(frozen=True)
class _ResumePlan:
    state: ControlState
    start: int
    start_iteration: int
    checkpoint: OptimizationCheckpoint
    stage_index: int


class _ContinuationResult:
    """``run_continuation`` stage result carrying the unmasked active vector."""

    def __init__(self, model: np.ndarray, stage_result: StageResult) -> None:
        self.model = model
        self.stage_result = stage_result


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
        penalty: Penalty (``bind(space)`` protocol) added to the misfit; a
            stage's ``penalty`` overrides it.
        preconditioner: Preconditioner (``bind(space)`` protocol) or plain
            ``(model, g) -> ndarray`` callable; bound preconditioners are
            updated at every stage start and every
            ``optimizer.preconditioner_refresh`` iterations.
        smoothing: Gradient smoothing applied by Sauce for every stage
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
        penalty: Any = None,
        preconditioner: Any = None,
        smoothing: Any = None,
        step_limit: Optional[float] = None,
        scaling: Any = None,
        checkpoint: Optional[Union[str, Path]] = None,
        history: Any = None,
        callback: Optional[Callable[[FWIIteration], None]] = None,
    ) -> None:
        self.problem = problem
        self.stages: List[Stage] = _stage_list(stages)
        self.optimizer = LBFGS() if optimizer is None else optimizer
        if not callable(getattr(self.optimizer, "solve", None)):
            raise TypeError("optimizer must provide solve()")
        self.penalty = penalty
        self.preconditioner = preconditioner
        if smoothing is not None and smoothing is not False:
            smoothing = SmoothingConfig.from_value(smoothing)
        self.smoothing = smoothing
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
        problem = (
            previous
            if stage.controls is None
            else previous.with_controls(stage.controls)
        )
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
        penalty: Any,
    ) -> Dict[str, Any]:
        view = self._views[stage.label(index)]
        smoothing = view.smoothing
        return {
            "stage": stage.label(index),
            "stage_index": int(index),
            "frequencies": json.dumps(_frequency_pairs(stage.frequencies)),
            "active": ",".join(space.blocks),
            "optimizer": getattr(optimizer, "kind", type(optimizer).__name__),
            "penalty": None if penalty is None else type(penalty).__name__,
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
        masks = space.support_masks()
        support = hashlib.sha256(
            json.dumps(
                {
                    name: mask.astype(int).tolist()
                    for name, mask in sorted(masks.items())
                }
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema": CHECKPOINT_SCHEMA,
            "problem": problem.name,
            "identity": self._identity(problem),
            # The control layout of this stage's problem (stages with
            # ``controls`` change it); resume rebuilds the same layout.
            "control_ids": self._layout(problem)[0],
            "control_sizes": self._layout(problem)[1],
            "stage_index": int(index),
            "stage_name": stage.label(index),
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
    ) -> None:
        if self.checkpoint_path is None:
            return
        state = view.state_from(ControlVector(model, space))
        assert self.state_path is not None
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state.save(self.state_path)
        OptimizationCheckpoint(
            model=model,
            iteration=history.iteration_count,
            evaluations=history.evaluation_count,
            loss=loss,
            metadata=self._checkpoint_metadata(
                index,
                stage,
                space,
                stage_iteration=stage_iteration,
                completed=completed,
                history=history,
            ),
        ).save(self.checkpoint_path)

    # -- resume ---------------------------------------------------------------

    def _resume_plan(self) -> Optional[_ResumePlan]:
        path = self.checkpoint_path
        if path is None or not path.is_file():
            return None
        checkpoint = OptimizationCheckpoint.load(path)
        meta = checkpoint.metadata
        if meta.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path} is not an FWI checkpoint")
        if meta.get("problem") != self.problem.name:
            raise ValueError(
                f"checkpoint {path} belongs to problem {meta.get('problem')!r}, "
                f"not {self.problem.name!r}"
            )
        index = int(meta["stage_index"])
        if index >= len(self.stages):
            raise ValueError(
                f"checkpoint stage index {index} exceeds the {len(self.stages)} stages"
            )
        # Rebuild the control layout the checkpointed stage ran on (stage
        # ``controls`` of every stage up to it) before comparing layouts.
        problem = self._problem_for(index)
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
        if bool(meta.get("stage_completed")) or stage_iteration >= stage.iterations:
            return _ResumePlan(state, index + 1, 0, checkpoint, index)
        return _ResumePlan(state, index, stage_iteration, checkpoint, index)

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
        penalty = self.penalty if stage.penalty is None else stage.penalty
        remaining = stage.iterations - int(start_iteration)
        if remaining < 1:
            raise ValueError(
                f"stage {label!r} has no iterations left ({start_iteration} of "
                f"{stage.iterations} done)"
            )

        # First linearization of the stage: adopts the support masks that
        # define the stage space (held fixed until the next transition).
        first = view.linearize(gradient=True)
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
        lower, upper = space.bounds
        x0 = np.clip(x0, lower, upper)

        bound_penalty = None if penalty is None else penalty.bind(space)
        metrics = self._stage_metrics(index, stage, space, optimizer, penalty)
        objective = _StageObjective(view, space, bound_penalty, history, metrics)
        initial_loss = objective.loss(x0)
        linearizations_before = objective.evaluations

        preconditioner = self.preconditioner
        bound_preconditioner: Any = None
        if preconditioner is not None and callable(
            getattr(preconditioner, "bind", None)
        ):
            bound_preconditioner = preconditioner.bind(space)
            bound_preconditioner.update(
                objective.linearization(x0), penalty=bound_penalty
            )
            preconditioner = bound_preconditioner
        refresh = getattr(optimizer, "preconditioner_refresh", None)
        scaling = self._scaling_for(objective.linearization(x0), space)
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
                    objective.linearization(it.model), penalty=bound_penalty
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

        hessian = (
            objective.hessian_action
            if getattr(optimizer, "kind", None) == "newton_cg"
            else None
        )
        result = optimizer.solve(
            objective,
            x0,
            bounds=(lower, upper),
            preconditioner=preconditioner,
            hessian_action=hessian,
            callback=on_iteration,
            max_iterations=remaining,
            step_limit=step_limit,
            scaling=scaling,
            space=space,
        )
        final = ControlVector(result.model, space)
        problem.state = view.state_from(final)
        final_loss = objective.loss(result.model)
        stage_iteration = start_iteration + int(result.iterations)
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
            metrics={
                "optimizer": metrics["optimizer"],
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
        self.results.append(stage_result)
        return stage_result

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
        first_view = first_stage.view(first_problem)
        self._views[first_stage.label(start)] = first_view
        assert first_problem.state is not None
        initial = first_view.vector(first_problem.state).values

        def solve(cs: ContinuationStage, _model: np.ndarray) -> _ContinuationResult:
            index = index_of[cs.name]
            stage = self.stages[index]
            resumed = index == start
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


class _RealJacobian(LinearOperator):
    """``[Re; Im]`` stacking of a complex Jacobian over a real domain.

    ``matvec`` interleaves real and imaginary parts (optionally scaled by
    ``sqrt(w)`` per row) so ``||A x - b||`` equals the weighted complex
    residual norm; ``rmatvec`` restores the complex dual and applies
    ``J.H``.  Optional penalty rows ``R`` are appended.
    """

    def __init__(
        self,
        jacobian: Any,
        row_weights: np.ndarray,
        penalty_operator: Any = None,
    ) -> None:
        self.jacobian = jacobian
        self.sqrt_weights = np.sqrt(np.asarray(row_weights, dtype=np.float64))
        self.penalty_operator = penalty_operator
        rows, cols = jacobian.shape
        extra = 0 if penalty_operator is None else int(penalty_operator.shape[0])
        self.data_rows = 2 * int(rows)
        super().__init__(np.dtype(np.float64), (self.data_rows + extra, int(cols)))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        data = np.asarray(self.jacobian @ x).reshape(-1) * self.sqrt_weights
        out = np.stack((data.real, data.imag), axis=-1).reshape(-1)
        if self.penalty_operator is not None:
            out = np.concatenate(
                [
                    out,
                    np.asarray(self.penalty_operator @ x, dtype=np.float64).reshape(-1),
                ]
            )
        return out

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        pairs = y[: self.data_rows].reshape(-1, 2)
        dual = (pairs[:, 0] + 1j * pairs[:, 1]) * self.sqrt_weights
        out = np.asarray(self.jacobian.H @ dual, dtype=np.float64).reshape(-1)
        if self.penalty_operator is not None:
            out = out + np.asarray(
                self.penalty_operator.H @ y[self.data_rows :], dtype=np.float64
            ).reshape(-1)
        return out

    def pack(self, data: Any) -> np.ndarray:
        """Realify a complex data vector with the row weights applied."""

        values = np.asarray(data, dtype=np.complex128).reshape(-1) * self.sqrt_weights
        return np.stack((values.real, values.imag), axis=-1).reshape(-1)


class LSRTM:
    """Least-squares reverse-time migration around one linearization.

    Solves ``min_dm 0.5 ||J dm + r||_W^2 + 0.5 damping ||dm||^2 + P(dm)`` with
    ``r = F(v0) - d`` (``J``, ``W`` and ``r`` frozen at ``v0``):

    - ``method="lsqr"``: SciPy's LSQR on the real-stacked Jacobian with the
      data-space residual from ``problem.residual(v0)`` (a site ``forward``
      hook or a forward job); a penalty must expose ``operator()`` and is
      appended as extra rows ``||R dm||^2``.
    - ``method="cg"``: conjugate gradients on the Gauss-Newton normal
      equations ``(H + damping I + P'') dm = -(g + P'(0))`` using
      ``lin.normal`` and ``lin.gradient`` only (no data-space residual
      needed; the gradient is smoothed when the problem smooths gradients).

    Args:
        problem: Problem (typically over ``GridParameters``/reflectivity).
        iterations: Solver iteration limit.
        penalty: Optional penalty (``bind(space)`` protocol) on the image.
        method: ``"lsqr"`` or ``"cg"``.
        damping: Tikhonov damping ``||dm||^2`` weight.
        tolerance: Relative solver tolerance.
        callback: Called with the current image after every iteration.
    """

    def __init__(
        self,
        problem: ImagingProblem,
        iterations: int = 15,
        *,
        penalty: Any = None,
        method: str = "lsqr",
        damping: float = 0.0,
        tolerance: float = 1.0e-8,
        callback: Optional[Callable[[ControlVector], None]] = None,
    ) -> None:
        self.problem = problem
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        self.penalty = penalty
        method = str(method).strip().lower()
        if method not in {"lsqr", "cg"}:
            raise ValueError("method must be 'lsqr' or 'cg'")
        self.method = method
        self.damping = float(damping)
        if not math.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and nonnegative")
        self.tolerance = float(tolerance)
        self.callback = callback
        self.info: Dict[str, Any] = {}
        self.linearization: Optional[Linearization] = None

    def run(self, v0: Any = None) -> ControlVector:
        """Return the image ``dm`` on the problem space, linearized at ``v0``."""

        # Operators need the covector parts (per-task registry fingerprints),
        # so the linearization always carries the gradient.
        lin = self.problem.linearize(v0, gradient=True)
        self.linearization = lin
        space = lin.space
        bound = None if self.penalty is None else self.penalty.bind(space)
        if self.method == "cg":
            return self._run_cg(lin, space, bound)
        return self._run_lsqr(lin, space, bound)

    def _run_cg(
        self, lin: Linearization, space: ControlSpace, bound: Any
    ) -> ControlVector:
        zero = space.zeros()
        assert lin.gradient is not None
        rhs = -np.asarray(lin.gradient.values, dtype=np.float64)
        hessian = None
        if bound is not None:
            rhs = rhs - _real_vector(
                bound.gradient(zero), size=space.size, name="penalty"
            )
            hessian = bound.hessian_operator(zero)
        normal = lin.normal
        damping = self.damping

        def matvec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            out = np.asarray(normal @ x, dtype=np.float64).reshape(-1) + damping * x
            if hessian is not None:
                out = out + np.asarray(hessian @ x, dtype=np.float64).reshape(-1)
            return out

        operator = LinearOperator(
            (space.size, space.size), matvec=matvec, dtype=np.float64
        )
        count = 0

        def on_iteration(x: np.ndarray) -> None:
            nonlocal count
            count += 1
            if self.callback is not None:
                self.callback(ControlVector(x, space))

        solution, status = scipy_cg(
            operator,
            rhs,
            maxiter=self.iterations,
            rtol=self.tolerance,
            callback=on_iteration,
        )
        residual = rhs - matvec(solution)
        self.info = {
            "method": "cg",
            "iterations": count,
            "status": int(status),
            "converged": int(status) == 0,
            "residual_norm": float(np.linalg.norm(residual)),
            "rhs_norm": float(np.linalg.norm(rhs)),
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
        penalty_operator = None
        if bound is not None:
            penalty_operator = bound.operator()
            if penalty_operator is None:
                raise ValueError(
                    "LSRTM with method='lsqr' needs a penalty exposing operator(); "
                    "use method='cg' for this penalty"
                )
        weights = np.ones(lin.data_space.size, dtype=np.float64)
        weights = lin.weight_data(
            DataVector(weights.astype(np.complex128), lin.data_space)
        ).values.real
        operator = _RealJacobian(lin.jacobian, weights, penalty_operator)
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


def rtm(problem: ImagingProblem, v: Any = None) -> ControlVector:
    """Return the misfit gradient (RTM image on the control space) at ``v``.

    Equals ``problem.gradient(v)``: the objective's real control covector
    (smoothed when the problem smooths gradients). The Jacobian and its frozen
    comparison residual carry Sauce's objective-space sign and normalization.
    """

    lin = problem.linearize(v, gradient=True)
    assert lin.gradient is not None
    return lin.gradient


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


class TimeReversalFocus:
    """Time-reversal focusing objective over the problem's material blocks.

    Non-material blocks of the problem view (sources, reflectivity) are held
    at the current state; their gradient entries are zero.

    Wraps ``ControlGradientJob(kind="focus")``: Sauce back-propagates the
    observed data, measures the focusing of the time-reversed wavefield with
    softening length ``softening`` (km) and returns the focusing objective and
    its gradient with respect to the active ``model.*`` blocks.

    Args:
        problem: Problem providing simulation, observed data, site and the
            active material blocks.
        softening: Focusing softening length in km (or a length quantity).
        kind: ``"trfwi"`` or ``"weft"``.
        weights: Optional per-frequency weights.
        smoothing: Gradient smoothing (default: the problem's).
        spatial_window: Optional ``control_sensitivities.spatial_window``.
    """

    def __init__(
        self,
        problem: ImagingProblem,
        softening: Any,
        *,
        kind: str = "trfwi",
        weights: Optional[Sequence[float]] = None,
        smoothing: Any = None,
        spatial_window: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.problem = problem
        magnitude = getattr(softening, "magnitude", None)
        if magnitude is not None and hasattr(softening, "to"):
            softening = softening.to("km").magnitude
        self.softening = float(softening)
        if not math.isfinite(self.softening) or self.softening <= 0.0:
            raise ValueError("softening must be a positive length (km)")
        self.kind = str(kind).strip().lower()
        self.weights = None if weights is None else [float(w) for w in weights]
        self.smoothing = (
            problem.smoothing
            if smoothing is None
            else SmoothingConfig.from_value(smoothing)
        )
        self.spatial_window = None if spatial_window is None else dict(spatial_window)
        self._results: Dict[str, Tuple[float, ControlVector]] = {}
        self._jobs: Dict[str, ControlGradientJob] = {}

    @property
    def active(self) -> List[str]:
        """Return the unqualified material block ids of the problem view.

        Only ``model.*`` blocks enter Sauce's focus workflow; other blocks of
        the view (sources, reflectivity) are held at the state and receive a
        zero gradient.
        """

        blocks = list(self.problem.space.blocks)
        model = [name for name in blocks if name.startswith("model.")]
        if not model:
            raise ValueError(
                "time-reversal focusing needs at least one model.* block "
                f"(active blocks: {blocks})"
            )
        return [unqualified_block_name(name) for name in model]

    def _key(self, state: ControlState) -> str:
        return _digest(state.values)[:16]

    def job(self, v: Any = None) -> ControlGradientJob:
        """Build the focus job at ``v`` (default: the current state)."""

        problem = self.problem
        state = problem.state_from(v) if v is not None else problem.state
        if state is None:
            raise ValueError("the control state is unknown; linearize once first")
        key = self._key(state)
        job = self._jobs.get(key)
        if job is not None:
            return job
        active = self.active
        full = problem.full_space
        for block, sl in zip(full.resolved_blocks, full.full_slices.values()):
            if block.name.startswith("model."):
                continue
            baseline = problem._authored_block(block.name, sl)
            if baseline is None or not np.array_equal(state.values[sl], baseline):
                raise NotImplementedError(
                    f"time-reversal focusing runs on the authored simulation, but "
                    f"block {block.name!r} differs from its authored baseline"
                )
        current: Optional[Path] = None
        if v is not None or not problem.is_authored(state):
            staging = problem.backend.staging_dir("focus", key)
            blocks = {
                name: values
                for name, values in state.blocks().items()
                if name.startswith("model.")
            }
            current = ControlVectorFile(blocks, native=True).write(
                staging / "current.h5"
            )
        job = ControlGradientJob(
            problem.backend.job_name("focus"),
            problem.simulation,
            list(problem.frequencies),
            kind="focus",
            observed=_observed_paths(problem),
            gradient="focus_gradient.h5",
            objective_file="focus_objective.h5",
            focus={"softening": self.softening, "kind": self.kind},
            active=active,
            current=current,
            misfit=problem._misfit_payload,
            weights=self.weights,
            smoothing=self.smoothing,
            raw_gradient=(
                "focus_gradient_raw.h5" if self.smoothing is not None else None
            ),
            spatial_window=self.spatial_window,
        )
        self._jobs[key] = job
        return job

    def dry_run(self, v: Any = None) -> Dict[str, Any]:
        """Describe the focus job at ``v`` without submitting it."""

        return self.problem.backend.dry_run(self.job(v))

    def objective(self, v: Any = None) -> Tuple[float, ControlVector]:
        """Return ``(value, gradient)`` of the focusing objective at ``v``."""

        problem = self.problem
        state = problem.state_from(v) if v is not None else problem.state
        assert state is not None
        key = self._key(state)
        cached = self._results.get(key)
        if cached is not None:
            return cached
        job = self.job(v)
        problem.backend.run(job)
        value = float(job.objective_value)
        file = ControlVectorFile.read(job.gradient_file())
        space = problem.space
        # Non-material blocks do not enter the focus workflow: zero gradient.
        blocks = {
            b.name: (
                file[b.name]
                if b.name.startswith("model.")
                else np.zeros(
                    b.coefficient_count, dtype=complex if b.complex else float
                )
            )
            for b in space.resolved_blocks
        }
        gradient = space.pack(blocks)
        self._results[key] = (value, gradient)
        return value, gradient

    def value(self, v: Any = None) -> float:
        """Return the focusing objective at ``v``."""

        return self.objective(v)[0]

    def gradient(self, v: Any = None) -> ControlVector:
        """Return the focusing gradient on the problem space at ``v``."""

        return self.objective(v)[1]
