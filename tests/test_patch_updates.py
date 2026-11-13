# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Native-coordinate overlap combination and fixed-exterior regularization."""

from copy import copy
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

from frequensolve.imaging import (
    ControlState,
    ControlVector,
    DepthProfile,
    ImagingProblem,
    PatchUpdates,
)
from frequensolve.imaging._patch_updates import (
    _RestrictedRegularization,
    combine_proposals,
)
from frequensolve.imaging.regularization import Quadratic
from tests.imaging_fakes import FakeImagingSite, layered_simulation


@pytest.fixture
def problem(tmp_path):
    problem = ImagingProblem(
        layered_simulation(tmp_path),
        controls=DepthProfile("vp", "sediment", count=4, transform="log"),
        observed=None,
        frequencies=[4],
        site=FakeImagingSite(),
    )
    problem.linearize()
    return problem


@pytest.mark.parametrize(
    "settings",
    [
        dict(mode="bad"),
        dict(mode="local_serial", local_steps=0),
        dict(mode="local_parallel", check_every=0),
        dict(mode="local_serial", local_steps=True),
    ],
)
def test_patch_update_configuration_rejects_invalid_values(settings):
    with pytest.raises(ValueError):
        PatchUpdates(**settings)


def test_parallel_overlap_averages_native_log_increments_and_preserves_exterior(
    problem,
):
    baseline = ControlState(problem.full_space, np.array([0.1, 0.2, 0.3, 0.4]))
    one = ControlState(problem.full_space, np.array([0.4, 0.6, 999.0, 0.4]))
    two = ControlState(problem.full_space, np.array([999.0, 0.8, 1.1, 0.4]))
    masks = [
        {"model.vp": [True, True, False, False]},
        {"model.vp": [False, True, True, False]},
    ]
    merged = combine_proposals(baseline, [one, two], masks, problem.space)
    np.testing.assert_allclose(merged.values, [0.4, 0.7, 1.1, 0.4])
    limited = combine_proposals(
        baseline, [one, two], masks, problem.space, step_limit=0.05
    )
    assert np.linalg.norm(limited.values - baseline.values) / 2 <= 0.05 + 1e-15
    assert limited.values[-1] == baseline.values[-1]


def test_restricted_regularization_retains_fixed_exterior_connections(problem):
    global_space = problem.space
    # First differences couple the local coefficient to its fixed neighbours.
    matrix = np.diff(np.eye(4), axis=0)
    bound = Quadratic(matrix).bind(global_space)
    baseline = ControlState(problem.full_space, np.array([1.0, 2.0, 4.0, 8.0]))
    local = global_space.with_support({"model.vp": [False, True, False, False]})
    term = _RestrictedRegularization(bound, baseline, global_space, local, RLock())
    point = ControlVector([3.0], local)
    assembled = baseline.with_update(point).vector(global_space)
    assert term.value(point) == pytest.approx(bound.value(assembled))
    expected = bound.gradient(assembled).values[1]
    assert term.gradient(point).values == pytest.approx([expected])
    epsilon = 1e-5
    fd = (
        term.value(ControlVector([3.0 + epsilon], local))
        - term.value(ControlVector([3.0 - epsilon], local))
    ) / (2 * epsilon)
    assert fd == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["local_serial", "local_parallel"])
def test_local_orchestration_baselines_and_combined_increase_are_retained(
    problem, tmp_path, monkeypatch, mode
):
    from frequensolve.imaging import Stage
    from frequensolve.imaging._patch_updates import solve_local_stage
    from frequensolve.inversion import LossTerms, OptimizationHistory
    from frequensolve.mesh.patches import PatchSet
    from frequensolve.units import ureg

    state = ControlState(problem.full_space, np.zeros(4))
    problem.state = state
    problem._patches = PatchSet.around_sources(
        shots_per_patch=1, max_offset=1 * ureg.km, padding=0 * ureg.m
    )
    problem._patch_runtime = SimpleNamespace(
        prepared=SimpleNamespace(
            geometry={"patches": [{"name": "one"}, {"name": "two"}]}
        ),
        stage=SimpleNamespace(manifest=tmp_path / "manifest.json"),
    )
    masks = [[True, True, False, False], [False, True, True, False]]
    starts, callbacks, checkpoints = [], [], []

    def local_view(view, patch, baseline):
        local = view.restrict(patches=None)
        local._shared = copy(view._shared)
        local._shared.state = baseline
        local._masks = {"model.vp": np.asarray(masks[patch])}
        local._masks_adopted = True
        local.linearize = lambda: SimpleNamespace()
        return local

    class Objective:
        # Local solvers below accept their own steps. The combined quadratic is
        # deliberately minimized at the baseline, so every combined update rises.
        def __init__(self, view, space, *args):
            self.evaluations = 0

        def loss(self, values):
            self.evaluations += 1
            return LossTerms(data=float(np.asarray(values) @ np.asarray(values) / 2))

    class Optimizer:
        kind = "lbfgs"
        step_limit = None
        scaling_max_ratio = 100

        def solve(self, objective, initial, *, callback, **kwargs):
            starts.append(initial.copy())
            model = initial + 1
            callback(
                SimpleNamespace(
                    iteration=1,
                    model=model,
                    gradient=np.ones(2),
                    step=np.ones(2),
                    step_length=1,
                    optimizer_state=None,
                )
            )
            return SimpleNamespace(status=0, model=model)

    monkeypatch.setattr(
        "frequensolve.imaging._patch_updates.local_patch_view", local_view
    )
    monkeypatch.setattr(
        "frequensolve.imaging._patch_updates.core_masks",
        lambda view, prepared, patch: {"model.vp": masks[patch]},
    )
    monkeypatch.setattr("frequensolve.imaging.workflows._StageObjective", Objective)
    workflow = SimpleNamespace(
        patch_updates=PatchUpdates(mode),
        optimizer=Optimizer(),
        _history=OptimizationHistory(),
        preconditioner=None,
        step_limit=None,
        _scaling_for=lambda *args: None,
        _problem_for=lambda index: problem,
        _write_checkpoint=lambda *args, **kwargs: checkpoints.append(kwargs),
        callback=callbacks.append,
        results=[],
    )
    result = solve_local_stage(
        workflow,
        0,
        Stage([4], 1),
        0,
        problem,
        SimpleNamespace(state=state),
        None,
        None,
    )
    np.testing.assert_array_equal(starts[0], [0, 0])
    np.testing.assert_array_equal(
        starts[1], [1, 0] if mode == "local_serial" else [0, 0]
    )
    expected = [1, 2, 1, 0] if mode == "local_serial" else [1, 1, 1, 0]
    np.testing.assert_array_equal(problem.state.values, expected)
    assert result.final_loss.total > result.initial_loss.total
    assert result.metrics["combined_objective_increases"] == 1
    combined = [
        r
        for r in workflow._history.evaluations
        if r.metrics.get("objective_scope") == "combined"
    ]
    assert len(combined) == 1 and combined[0].metrics["updates_retained"]
    assert combined[0].metrics["objective_change"] > 0
    assert len(callbacks) == 2 and all(e.mode == mode for e in callbacks)
    assert checkpoints[-1]["completed"]


@pytest.mark.parametrize("increment", [0.0, 0.02])
def test_parallel_step_cap_preserves_zero_and_small_updates(problem, increment):
    baseline = ControlState(problem.full_space, np.full(4, 0.1))
    proposal = ControlState(problem.full_space, baseline.values + increment)
    accepted = combine_proposals(
        baseline,
        [proposal],
        [{"model.vp": np.ones(4, dtype=bool)}],
        problem.space,
        step_limit=0.05,
    )
    assert np.isfinite(accepted.values).all()
    np.testing.assert_allclose(accepted.values, proposal.values, rtol=0, atol=1e-15)
