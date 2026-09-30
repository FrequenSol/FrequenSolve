# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Exact composite row maps, shared coefficients and parent coverage."""

from types import SimpleNamespace

import numpy as np
import pytest

from frequensolve.imaging import ControlVector, DataVector, DepthProfile, ImagingProblem
from frequensolve.imaging._objective import _ObjectiveSpace
from frequensolve.imaging._patch_problem import CompositeLinearization
from frequensolve.imaging.data import _DataSegment
from tests.imaging_fakes import FakeImagingSite, layered_simulation


class _Child:
    def __init__(self, problem, frequency, patch, matrix, keys, mask, weight):
        self.space = problem.space.without_support()
        self.frequencies = [frequency]
        self.data_space = _ObjectiveSpace(
            [frequency], [_DataSegment("p", ("objective",), (1,), (1, 2))]
        )
        self.data_space._dense = set()
        self.data_space._keys = [{"p": keys}]
        self.matrix = matrix
        self.weight = weight
        self.job = SimpleNamespace(state_file=lambda task: (frequency, patch))
        self.problem = SimpleNamespace(
            simulation=SimpleNamespace(acquisition=SimpleNamespace(to_fs=lambda: {}))
        )
        self.reports = []
        self.report = {"p": float(weight * np.linalg.norm(matrix) ** 2)}
        self.value = self.report["p"]
        self.gradient = ControlVector(
            weight * np.real(matrix.conj().T @ np.ones(2)), self.space
        )
        self.support_masks = {"model.vp": mask}
        self.residual_sign = 1

    def jvp(self, direction):
        return DataVector(self.matrix @ direction.values, self.data_space)

    def vjp(self, dual):
        return ControlVector(np.real(self.matrix.conj().T @ dual.values), self.space)

    def apply_normal(self, direction):
        return self.vjp(self.weight_data(self.jvp(direction)))

    def weight_data(self, dual):
        return DataVector(self.weight * dual.values, self.data_space)

    def simulated(self):
        return DataVector(self.matrix.sum(axis=1), self.data_space)

    objective_residual = observed = simulated


@pytest.fixture
def composite(tmp_path, monkeypatch):
    problem = ImagingProblem(
        layered_simulation(tmp_path),
        controls=DepthProfile("vp", "sediment", count=3),
        observed=None,
        frequencies=[4, 6],
        site=FakeImagingSite(),
    )
    parent = problem.linearize()
    children, mappings = [], {}
    rng = np.random.default_rng(29)
    for frequency, weight in [(4, 0.5), (6, 2.0)]:
        for patch, source in enumerate([3, 1]):
            # Deliberately reverse receiver IDs and noncontiguous original shots.
            keys = np.array([[source, 7, 2], [source, 2, 1]])
            matrix = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
            mask = np.array([True, patch == 0, patch == 1])
            child = _Child(problem, frequency, patch, matrix, keys, mask, weight)
            children.append(child)
            mappings[(frequency, patch)] = {"p": keys}
    monkeypatch.setattr(
        "frequensolve.imaging._patch_problem.patch_objective_keys",
        lambda path, acquisition: mappings[path],
    )
    lin = CompositeLinearization(problem, parent.state, "composite", children)
    return problem, lin, children


def test_shared_control_sum_and_global_coverage(composite):
    problem, lin, children = composite
    assert lin.space.size == 3
    assert lin.support_masks["model.vp"].all()
    assert problem.space.size == 3
    np.testing.assert_allclose(
        lin.gradient.values, sum(c.gradient.values for c in children)
    )
    assert lin.value == sum(c.value for c in children)
    assert len(lin.jobs) == 4
    with pytest.raises(ValueError, match="multiple jobs"):
        _ = lin.job


def test_exact_composite_adjoint_weighting_and_frequency_parts(composite):
    _, lin, children = composite
    direction = lin.space.random(12)
    dual = lin.data_space.random(13)
    tangent = lin.jvp(direction)
    np.testing.assert_allclose(
        np.vdot(tangent.values, dual.values).real,
        direction.values @ lin.vjp(dual).values,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        lin.apply_normal(direction).values,
        lin.vjp(lin.weight_data(tangent)).values,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        sum(v.values for v in lin.vjp_tasks(dual)), lin.vjp(dual).values
    )
    for child in children:
        target = lin.data_space.term_layout("p", frequency=child.frequencies[0])
        source = child.data_space.term_layout("p")
        lookup = {
            tuple(row): i for row, i in zip(target.coordinate_keys, target.indices)
        }
        np.testing.assert_allclose(
            tangent.values[[lookup[tuple(k)] for k in source.coordinate_keys]],
            child.matrix @ direction.values,
        )
    np.testing.assert_allclose((lin.jacobian.H @ dual).values, lin.vjp(dual).values)


def test_duplicate_physical_observation_rejected(composite):
    problem, lin, children = composite
    with pytest.raises(ValueError, match="duplicate"):
        CompositeLinearization(
            problem, lin.state, "duplicate", [*children, children[0]]
        )
