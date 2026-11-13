# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Local patch optimization, ordered publication and common-epoch proposals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import asdict, dataclass
from threading import RLock
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from frequensolve.inversion import LossTerms
from frequensolve.mesh._stage_snapshot import _contained, _digest

from ._block_digest import block_digest
from ._patch_masks import core_masks, local_patch_view
from .controls import ControlSpace, ControlState, ControlVector
from .problem import ImagingProblem, Linearization
from .results import StageResult


@dataclass(frozen=True)
class PatchUpdates:
    """Local material updates; one stage iteration is one prepared-order sweep.

    Parallel proposals share one sweep-start model and use equal nonnegative
    overlap weights. Combined-objective increases are retained and recorded.
    ``check_every=None`` disables periodic checks; endpoints are still evaluated.
    """

    mode: str
    local_steps: int = 1
    check_every: int | None = 1

    def __post_init__(self) -> None:
        if self.mode not in {"local_serial", "local_parallel"}:
            raise ValueError("PatchUpdates mode must be local_serial or local_parallel")
        for name in ("local_steps", "check_every"):
            value = getattr(self, name)
            if value is None and name == "check_every":
                continue
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _RestrictedRegularization:
    """Pull back a global term while retaining all fixed-exterior connections."""

    def __init__(
        self,
        bound: Any,
        baseline: ControlState,
        global_space: ControlSpace,
        local_space: ControlSpace,
        lock: Any,
    ) -> None:
        self.bound = bound
        self.baseline = baseline
        self.global_space = global_space
        self.space = local_space
        self.lock = lock

    def _point(self, vector: ControlVector) -> ControlVector:
        return self.baseline.with_update(vector).vector(self.global_space)

    def value(self, vector: ControlVector) -> float:
        with self.lock:
            return self.bound.value(self._point(vector))

    def gradient(self, vector: ControlVector) -> ControlVector:
        with self.lock:
            gradient = self.bound.gradient(self._point(vector))
            return self.space.from_sauce_vector(
                gradient.space.to_sauce_vector(gradient)
            )


def combine_proposals(
    baseline: ControlState,
    proposals: Sequence[ControlState],
    masks: Sequence[Mapping[str, np.ndarray]],
    space: ControlSpace,
    *,
    step_limit: float | None = None,
) -> ControlState:
    """Average increments in native coordinates, retaining untouched coefficients."""
    if len(proposals) != len(masks):
        raise ValueError("Each parallel proposal requires its own core mask")
    total = np.zeros_like(baseline.values)
    count = np.zeros_like(baseline.values)
    for proposal, coverage in zip(proposals, masks):
        if not proposal.space.without_support().equivalent(
            baseline.space.without_support()
        ):
            raise ValueError("Parallel proposal has a different parent control basis")
        active = np.zeros(baseline.size, bool)
        for name, flags in coverage.items():
            flags = np.asarray(flags, dtype=bool)
            region = baseline.space.full_slices[name]
            if flags.shape != (region.stop - region.start,):
                raise ValueError("Parallel proposal core mask has the wrong size")
            active[baseline.space.full_slices[name]] = flags
        total[active] += (proposal.values - baseline.values)[active]
        count[active] += 1
    values = baseline.values.copy()
    np.divide(total, count, out=total, where=count > 0)
    values[count > 0] += total[count > 0]
    candidate = ControlState(
        baseline.space,
        values,
        scaling=baseline.scaling,
        scaling_units=baseline.scaling_units,
    ).vector(space)
    lower, upper = space.bounds
    accepted = np.clip(candidate.values, lower, upper)
    current = baseline.vector(space)
    increment = accepted - current.values
    if step_limit is not None:
        from .workflows import rms_step_limit

        scale = min(
            1.0, rms_step_limit(increment, tuple(space.slices.values()), step_limit)
        )
        increment *= scale
    return baseline.with_update(ControlVector(current.values + increment, space))


def solve_local_stage(
    workflow: Any,
    index: int,
    stage: Any,
    start: int,
    view: ImagingProblem,
    first: Linearization,
    regularization: Any,
    native: Any,
    *,
    stage_started: float | None = None,
) -> StageResult:
    """Optimize cores and publish only completed, epoch-consistent updates."""
    from .workflows import FWIIteration, _StageObjective

    wall_started = perf_counter() if stage_started is None else stage_started
    settings = workflow.patch_updates
    optimizer = workflow.optimizer if stage.optimizer is None else stage.optimizer
    if getattr(optimizer, "kind", None) != "lbfgs":
        raise ValueError("Local patch updates require LBFGS")
    if view.patches is None:
        raise ValueError("Local patch updates require an ImagingProblem with patches")
    history = workflow._history
    space = view.space
    state = first.state
    runtime = view._patch_runtime
    assert runtime is not None
    prepared = runtime.prepared
    names = [p["name"] for p in prepared.geometry["patches"]]
    lock = RLock()
    objective = _StageObjective(
        view,
        space,
        regularization,
        history,
        {"stage": stage.label(index), "stage_index": index, "optimizer": settings.mode},
    )
    objective.native_regularization = native
    initial_loss = objective.loss(state.vector(space).values)
    workflow._stage_initial_objective = initial_loss.total
    workflow._optimizer_config = repr(optimizer)
    workflow._optimizer_scaling = None
    previous_check = initial_loss.total
    resumed = getattr(workflow, "_resume_local", None)
    if resumed is not None and resumed[0] == index:
        reference_index, reference = getattr(
            workflow, "_resume_initial_objective", (index, None)
        )
        if reference_index != index or reference is None:
            raise ValueError("Local checkpoint omits the original stage objective")
        workflow._stage_initial_objective = float(reference)
    progress = None if resumed is None or resumed[0] != index else resumed[1]
    if progress is not None:
        if progress["config"] != settings.to_dict() or progress["optimizer"] != repr(
            optimizer
        ):
            raise ValueError("Checkpoint local-update configuration changed")
        previous_check = progress["last_combined_objective"]
    accepted_updates = 0
    checks = 0
    increases = 0
    worker_seconds = 0.0
    evaluations = 0
    limit = (
        workflow.step_limit if workflow.step_limit is not None else optimizer.step_limit
    )
    policy = settings.to_dict()

    def epoch(model: ControlState) -> str:
        # Hashes the coefficients in place (threaded), without a bytes copy.
        return block_digest(model.values).rsplit(":", 1)[-1]

    def save_proposal(model: ControlState) -> dict[str, str]:
        root = runtime.stage.manifest.parent
        path = root / "proposals" / (epoch(model) + ".h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        return {"file": path.relative_to(root).as_posix(), "sha256": _digest(path)}

    def load_proposal(record: Mapping[str, Any]) -> ControlState:
        path = _contained(runtime.stage.manifest.parent, record["file"])
        if _digest(path) != record["sha256"]:
            raise ValueError("Checkpoint local proposal changed")
        return ControlState.load(path, state.space.without_support())

    def publish(sweep: int, *, completed: bool = False) -> None:
        workflow._local_checkpoint = progress
        view._shared.set_state(state)
        workflow._problem_for(index).state = state
        vector = state.vector(space)
        # This checkpoint loss is explicitly a combined value from its last
        # configured check, rather than a sum of local optimization objectives.
        checkpoint_loss = LossTerms(previous_check)
        workflow._write_checkpoint(
            index,
            stage,
            view,
            space,
            vector.values,
            checkpoint_loss,
            stage_iteration=sweep,
            completed=completed,
            history=history,
        )
        # Proposals the published checkpoint no longer references are
        # superseded; keep at most the baseline and one state per patch.
        root = runtime.stage.manifest.parent
        keep = set()
        if progress is not None:
            keep.add(progress["baseline"]["file"])
            keep.update(r["state"]["file"] for r in progress["proposals"].values())
        for stale in (root / "proposals").glob("*.h5"):
            if stale.relative_to(root).as_posix() not in keep:
                stale.unlink(missing_ok=True)

    def proposal(
        patch: int, base: ControlState, recorded: Mapping[str, Any] | None
    ) -> tuple[
        ControlState, list[tuple[int, ControlState, LossTerms, Any, Any]], int, float
    ]:
        began = perf_counter()
        point = base if recorded is None else load_proposal(recorded["state"])
        local = local_patch_view(view, patch, point)
        if local.space.size == 0:
            return point, [], 0, perf_counter() - began
        local_reg = (
            None
            if regularization is None
            else _RestrictedRegularization(
                regularization, base, space, local.space, lock
            )
        )
        local_native = None
        if native is not None:
            local_native = copy(native)
            local_native.space = local.space
            local_native.baseline = base
            local_native._value_cache = {}
            local_native._gradient_cache = {}
        local_objective = _StageObjective(local, local.space, local_reg, None, {})
        local_objective.native_regularization = local_native
        events: list[tuple[int, ControlState, LossTerms, Any, Any]] = []
        done = 0 if recorded is None else recorded["iteration"]
        remaining = settings.local_steps - done
        if remaining <= 0 or (recorded is not None and recorded["completed"]):
            return point, [], 0, perf_counter() - began
        bound_preconditioner = workflow.preconditioner
        if callable(getattr(bound_preconditioner, "bind", None)):
            bound_preconditioner = bound_preconditioner.bind(local.space)
            bound_preconditioner.update(local.linearize(), regularization=local_reg)
        scaling = workflow._scaling_for(local.linearize(), local.space)
        restart = None if recorded is None else recorded.get("optimizer_state")
        if recorded is not None:
            scaling = recorded.get("scaling")
            scaling = None if scaling is None else np.asarray(scaling)
        if isinstance(scaling, dict):
            from .workflows import curvature_scaling

            scaling = curvature_scaling(
                scaling, local.space, max_ratio=optimizer.scaling_max_ratio
            )

        def accepted(event: Any) -> None:
            if event.iteration > 0:
                model = point.with_update(ControlVector(event.model, local.space))
                events.append(
                    (
                        done + event.iteration,
                        model,
                        local_objective.loss(event.model),
                        event,
                        None if scaling is None else np.asarray(scaling, float),
                    )
                )

        result = optimizer.solve(
            local_objective,
            point.vector(local.space).values,
            bounds=local.space.bounds,
            max_iterations=remaining,
            callback=accepted,
            step_limit=limit,
            scaling=scaling,
            space=local.space,
            preconditioner=bound_preconditioner,
            restart=restart,
        )
        if result.status < 0:
            raise RuntimeError(f"Local patch {names[patch]!r} failed: {result.message}")
        return (
            point.with_update(ControlVector(result.model, local.space)),
            events,
            local_objective.evaluations,
            perf_counter() - began,
        )

    site: Any = view.site
    keep_cluster = bool(getattr(site, "shutdown_on_completion", False))
    if keep_cluster:
        site.shutdown_on_completion = False
    try:
        for sweep in range(start, stage.iterations):
            base = state
            if progress is None or progress["sweep"] != sweep:
                progress = {
                    "config": policy,
                    "optimizer": repr(optimizer),
                    "sweep": sweep,
                    "baseline_epoch": epoch(base),
                    "baseline": save_proposal(base),
                    "proposals": {},
                    "next_patch": 0,
                    "last_combined_objective": previous_check,
                }
                publish(sweep)
            else:
                base = load_proposal(progress["baseline"])
                if epoch(base) != progress["baseline_epoch"]:
                    raise ValueError("Checkpoint sweep baseline epoch changed")
            start_patch = int(progress["next_patch"])
            if settings.mode == "local_parallel":
                # Bound scheduler threads independently of the number of survey patches.
                executor = ThreadPoolExecutor()
                futures = {
                    patch: executor.submit(
                        proposal, patch, base, progress["proposals"].get(str(patch))
                    )
                    for patch in range(start_patch, len(names))
                }
            else:
                executor, futures = None, {}
            try:
                for patch in range(start_patch, len(names)):
                    recorded = progress["proposals"].get(str(patch))
                    model, events, used, elapsed = (
                        futures[patch].result()
                        if executor is not None
                        else proposal(patch, state, recorded)
                    )
                    evaluations += used
                    worker_seconds += elapsed
                    for (
                        local_iteration,
                        accepted_model,
                        loss,
                        diagnostics,
                        scaling,
                    ) in events:
                        if settings.mode == "local_serial":
                            state = accepted_model
                        progress["proposals"][str(patch)] = {
                            "baseline_epoch": progress["baseline_epoch"],
                            "state": save_proposal(accepted_model),
                            "iteration": local_iteration,
                            "completed": False,
                            "optimizer_state": diagnostics.optimizer_state,
                            "scaling": scaling,
                        }
                        workflow._optimizer_checkpoint = None
                        record = history.record_iteration(
                            accepted_model.vector(space).values,
                            loss,
                            gradient_norm=float(np.linalg.norm(diagnostics.gradient)),
                            step_norm=float(np.linalg.norm(diagnostics.step)),
                            step_length=diagnostics.step_length,
                            metrics={
                                "stage": stage.label(index),
                                "stage_index": index,
                                "optimizer": settings.mode,
                                "patch": names[patch],
                                "sweep": sweep + 1,
                                "local_iteration": local_iteration,
                                "objective_scope": "local",
                            },
                        )
                        accepted_updates += 1
                        publish(sweep)
                        if workflow.callback is not None:
                            workflow.callback(
                                FWIIteration(
                                    index,
                                    stage,
                                    history.iteration_count,
                                    sweep,
                                    accepted_model.vector(space),
                                    loss,
                                    record,
                                    diagnostics,
                                    mode=settings.mode,
                                    patch_name=names[patch],
                                    sweep=sweep + 1,
                                )
                            )
                    if settings.mode == "local_serial":
                        state = model
                    progress["proposals"][str(patch)] = {
                        "baseline_epoch": progress["baseline_epoch"],
                        "state": save_proposal(model),
                        "iteration": settings.local_steps,
                        "completed": True,
                        "optimizer_state": None,
                        "scaling": None,
                    }
                    progress["next_patch"] = patch + 1
                    publish(sweep)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
            if settings.mode == "local_parallel":
                records = [
                    progress["proposals"][str(patch)] for patch in range(len(names))
                ]
                if any(
                    record["baseline_epoch"] != progress["baseline_epoch"]
                    or not record["completed"]
                    for record in records
                ):
                    raise ValueError(
                        "Cannot combine parallel proposals from different baseline epochs"
                    )
                state = combine_proposals(
                    base,
                    [load_proposal(r["state"]) for r in records],
                    [core_masks(view, prepared, patch) for patch in range(len(names))],
                    space,
                    step_limit=limit,
                )
            if (
                settings.check_every is not None
                and (sweep + 1) % settings.check_every == 0
            ):
                checked = objective.loss(state.vector(space).values)
                change = checked.total - previous_check
                checks += 1
                increases += int(change > 0)
                history.record_evaluation(
                    state.vector(space).values,
                    checked,
                    metrics={
                        "stage": stage.label(index),
                        "stage_index": index,
                        "optimizer": settings.mode,
                        "sweep": sweep + 1,
                        "objective_scope": "combined",
                        "objective_change": change,
                        "updates_retained": True,
                    },
                )
                previous_check = checked.total
            progress = None
            publish(sweep + 1)
        final_loss = objective.loss(state.vector(space).values)
        previous_check = final_loss.total
        publish(stage.iterations, completed=True)
    finally:
        if keep_cluster:
            site.shutdown_on_completion = True
            site.close(wait=True, retire=True)
    result = StageResult(
        index=index,
        name=stage.label(index),
        frequencies=stage.frequencies,
        active=space.blocks,
        iterations=stage.iterations - start,
        stage_iteration=stage.iterations,
        success=True,
        status=0,
        message=f"{settings.mode} sweeps completed; accepted updates retained",
        initial_loss=initial_loss,
        final_loss=final_loss,
        evaluations=evaluations,
        linearizations=evaluations,
        resumed=resumed is not None,
        vector=state.vector(space),
        space=space,
        metrics={
            "optimizer": settings.mode,
            "local_updates": accepted_updates,
            "combined_checks": checks,
            "combined_objective_increases": increases,
            "elapsed_seconds": perf_counter() - wall_started,
            "elapsed_scope": "stage preparation, evaluations, reductions and checkpoint writes",
            "summed_local_visit_seconds": worker_seconds,
            "local_visit_time_scope": "local proposal preparation and optimization; excludes combined checks",
        },
    )
    workflow.results.append(result)
    return result
