"""Weighted multi-frequency objectives retain FWI values and derivatives."""

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging.weighted_objective import WeightedObjective
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_workflows import _problem


def test_weighted_multifrequency_value_gradient_and_diagonal(tmp_path):
    raw = _problem(tmp_path, FakeImagingSite())
    parts = [raw.restrict(weights=[1 / 2, 1 / 2]), raw.restrict(weights=[1 / 4, 3 / 4])]
    objective = WeightedObjective(parts, [2 / 3, 1 / 3])
    point = raw.vector()
    lin = objective.linearize(point, receiver_diagonal={"probes": 4, "seed": 17})
    expected = [
        p.linearize(point, receiver_diagonal={"probes": 4, "seed": 17}) for p in parts
    ]
    assert lin.value == pytest.approx(
        sum(w * p.value for w, p in zip(objective.weights, expected))
    )
    np.testing.assert_allclose(
        lin.gradient.values,
        sum(w * p.gradient.values for w, p in zip(objective.weights, expected)),
    )
    np.testing.assert_allclose(
        lin.receiver_diagonal.values,
        sum(
            w * p.receiver_diagonal.values for w, p in zip(objective.weights, expected)
        ),
    )
    direction = np.linspace(-0.1, 0.1, point.size)
    h = 1e-4
    fd = (
        objective.value(im.ControlVector(point.values + h * direction, point.space))
        - objective.value(im.ControlVector(point.values - h * direction, point.space))
    ) / (2 * h)
    assert fd == pytest.approx(lin.gradient.values @ direction, rel=1e-5, abs=1e-8)
    child = objective.restrict(frequencies=[4.0], support="refresh")
    assert child.weights == objective.weights
    assert all(list(p.frequencies) == [4.0] for p in child.problems)


def test_weighted_objective_standard_fwi(tmp_path):
    raw = _problem(tmp_path, FakeImagingSite())
    objective = WeightedObjective(
        [raw.restrict(weights=[0.5, 0.5]), raw.restrict(weights=[0.2, 0.8])],
        [2 / 3, 1 / 3],
    )
    before = objective.value()
    result = im.FWI(
        objective,
        [im.Stage(frequencies=(4.0, 6.0), iterations=2)],
        optimizer=im.LBFGS(),
    ).run()
    assert result.stages[-1].final_loss.data < before


@pytest.mark.parametrize("weights", [[-1, 2], [0, 0], [float("nan"), 1], [1]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        WeightedObjective([object(), object()], weights)


def test_weighted_objective_public_export():
    assert im.WeightedObjective is WeightedObjective
