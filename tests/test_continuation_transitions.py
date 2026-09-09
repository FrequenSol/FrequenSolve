"""Dimension-changing continuation preserves fields, not coefficient indices."""

from types import SimpleNamespace

import numpy as np
import pytest

from frequensolve import (
    ContinuationSchedule,
    ContinuationStage,
    ControlRepresentation,
    EvaluationContext,
    HatControl,
    minimize_inexact_newton,
    minimize_lbfgs,
    run_continuation,
)


def schedule():
    return ContinuationSchedule(
        tuple(
            ContinuationStage(f"level_{size}", (float(size),), metadata={"size": size})
            for size in (3, 5, 9)
        )
    )


@pytest.mark.parametrize("method", ["lbfgs", "newton"])
def test_refinement_transfers_accepted_field_with_fresh_optimizer(method):
    context = EvaluationContext({"x": np.linspace(0, 1, 101)})
    representations = {
        size: ControlRepresentation(
            HatControl(
                origin=0, spacing=1 / (size - 1), coefficients=np.zeros(size), axis="x"
            )
        )
        for size in (3, 5, 9)
    }
    target = 0.2 + context.coordinate("x") * 0.1
    transitions = []

    def solve(stage, initial):
        representation = representations[stage.metadata["size"]]
        matrix = representation.sampling_operator(context)
        objective = lambda model: 0.5 * np.linalg.norm(matrix @ model - target) ** 2
        gradient = lambda model: np.asarray(matrix.T @ (matrix @ model - target))
        if method == "lbfgs":
            return minimize_lbfgs(objective, gradient, initial)
        return minimize_inexact_newton(
            objective,
            gradient,
            lambda model, direction: matrix.T @ (matrix @ direction),
            initial,
        )

    def transition(previous, following, accepted):
        source = representations[previous.metadata["size"]]
        target_representation = representations[following.metadata["size"]]
        result = source.transfer_to(
            target_representation, accepted, context, tolerance=1e-12
        )
        np.testing.assert_allclose(
            source.evaluate(accepted, context),
            target_representation.evaluate(result, context),
            atol=1e-6,
        )
        transitions.append((accepted.size, result.size))
        return result

    result = run_continuation(schedule(), np.zeros(3), solve, transition=transition)
    assert transitions == [(3, 5), (5, 9)]
    assert [model.size for model in result.stage_initial_models] == [3, 5, 9]
    assert [stage.model.size for stage in result.stage_results] == [3, 5, 9]
    np.testing.assert_allclose(
        representations[9].evaluate(result.model, context), target, atol=1e-5
    )


@pytest.mark.parametrize("invalid", [[], [np.nan], [np.inf], [1j]])
def test_bad_transfer_stops_before_next_solve(invalid):
    starts = []

    def solve(stage, initial):
        starts.append(stage.name)
        return SimpleNamespace(x=initial)

    with pytest.raises(ValueError, match="continuation model"):
        run_continuation(schedule(), [0], solve, transition=lambda *_: invalid)
    assert starts == ["level_3"]


@pytest.mark.parametrize(
    "transition", [None, lambda previous, stage, model: np.zeros(5)]
)
def test_dimension_changes_inside_solve_are_always_rejected(transition):
    with pytest.raises(ValueError, match="control space"):
        run_continuation(
            schedule(),
            [0],
            lambda stage, model: SimpleNamespace(model=np.zeros(model.size + 1)),
            transition=transition,
        )


def test_same_size_transition_and_inplace_hooks_do_not_mutate_previous_results():
    def solve(stage, initial):
        initial[:] += 1
        return SimpleNamespace(model=initial)

    def transition(previous, stage, accepted):
        accepted[:] *= 2
        return accepted

    result = run_continuation(schedule(), [0], solve, transition=transition)
    assert [x.tolist() for x in result.stage_initial_models] == [[0], [2], [6]]
    assert [x.model.tolist() for x in result.stage_results] == [[1], [3], [7]]


def test_transition_must_be_callable():
    with pytest.raises(TypeError, match="transition"):
        run_continuation(schedule(), [0], lambda *_: None, transition=1)


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf])
def test_projection_tolerance_validation(tolerance):
    representation = ControlRepresentation(
        HatControl(axis="x", origin=0, spacing=1, coefficients=[0, 0])
    )
    with pytest.raises(ValueError, match="projection tolerance"):
        representation.project(
            [0, 1], EvaluationContext({"x": [0, 1]}), tolerance=tolerance
        )
