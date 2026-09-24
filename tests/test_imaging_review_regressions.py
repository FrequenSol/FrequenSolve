"""Regressions for saved-state replay and objective-space solver consistency."""

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._objective import (
    ObjectiveState,
    objective_residual,
    objective_space,
)
from frequensolve.imaging.data import DataSpace, DataVector, file_sha256
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_extension import _extension
from tests.test_imaging_extension import _problem as _extension_problem
from tests.test_imaging_problem import _problem

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("extension", [False, True])
def test_derivative_actions_replay_the_frozen_control_state(tmp_path, extension):
    fake = FakeImagingSite()
    make = _extension_problem if extension else _problem
    options = {
        "controls": im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", count=5),
            rho=im.DepthProfile("rho", "sediment", count=3),
            src=im.SourceParameters(),
        )
    }
    _, problem = make(tmp_path, fake, **options)
    target = (
        problem.restrict(active=["vp", "rho"]).extend(_extension())
        if extension
        else problem
    )
    point = problem.vector() + 0.001
    lin = target.linearize(problem.state.with_update(point))
    assert lin.job.control_state is not None
    # A subsequent linearization must not move the saved derivative baseline.
    target.linearize(problem.state.with_update(point + 0.001))
    direction = lin.extension_space.random(1) if extension else lin.space.random(1)
    data = lin.jvp(direction)
    lin.vjp(data)
    if extension:
        lin.apply_tap_normal(direction)
        assert lin.solutions
    else:
        lin.apply_normal(direction)
    actions = [job for job in fake.jobs if getattr(job, "action", None) != "linearize"]
    assert actions
    assert all(job.control_state == lin.job.control_state for job in actions)


def test_cached_linearization_adopts_support_before_a_stage_uses_its_space(tmp_path):
    fake = FakeImagingSite(support_masks={"model.vp": [1, 0, 1, 1, 0]})
    _, problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    view = problem.restrict(support="refresh")
    assert view.space.size == 8
    assert view.linearize() is lin
    assert view.space.equivalent(lin.space)
    run = im.FWI(problem, im.Stage([4.0, 6.0], iterations=1))
    run.run()


@pytest.mark.parametrize("reference", [None, np.arange(8.0) + 10])
def test_lsqr_uses_frozen_objective_residual_and_affine_penalty(tmp_path, reference):
    class ScaledSite(FakeImagingSite):
        def surrogate(self, *args, **kwargs):
            J, d, space = super().surrogate(*args, **kwargs)
            return 2.5 * J, 2.5 * d, space

    _, problem = _problem(tmp_path, ScaledSite())
    # Raw observations can have a different comparison/normalization entirely.
    problem.residual = lambda *_: (_ for _ in ()).throw(
        AssertionError("raw residual used")
    )
    penalty = 0.3 * (
        im.Quadratic(np.eye(8), weight=10, reference=reference)
        + im.Quadratic(np.eye(8), weight=2, reference=np.ones(8))
    )
    lin = problem.linearize()
    expected = np.real(lin.jacobian.H @ lin.objective_residual())
    np.testing.assert_allclose(expected, lin.gradient.values, atol=1e-10)
    outputs = [
        im.LSRTM(
            problem,
            method=method,
            iterations=120,
            tolerance=1e-12,
            damping=0.1,
            penalty=penalty,
        )
        .run()
        .values
        for method in ("cg", "lsqr")
    ]
    np.testing.assert_allclose(*outputs, rtol=1e-8, atol=1e-7)


@pytest.mark.parametrize("labels", [("amplitude",), ("amplitude", "phase")])
def test_named_objective_terms_share_a_receiver_group(tmp_path, labels):
    class TermSite(FakeImagingSite):
        def surrogate(self, *args, **kwargs):
            J, d, space = super().surrogate(*args, **kwargs)
            return (
                np.concatenate([J] * len(labels)),
                np.concatenate([d] * len(labels)),
                DataSpace(
                    space.frequencies,
                    [replace(space.segments[0], group=label) for label in labels],
                ),
            )

        def _write_state(self, job, task, lin, frequency):
            super()._write_state(job, task, lin, frequency)
            path = job.state_file(task)
            manifest = json.loads(path.read_text())
            for term in manifest["terms"]:
                term["receiver_group"] = "surface"
            path.write_text(json.dumps(manifest))

    fake = TermSite()
    _, problem = _problem(
        tmp_path,
        fake,
        misfit=im.Misfit.terms(
            *(im.ObjectiveTerm("surface", id=label) for label in labels)
        ),
    )
    lin = problem.linearize()
    assert lin.data_space.groups == labels
    assert lin.jacobian.dot_test(tolerance=1e-10)["passed"]
    vector = lin.objective_residual()
    np.testing.assert_allclose(lin.vjp(vector).values, lin.gradient.values, atol=1e-10)
    vjp = [job for job in fake.jobs if getattr(job, "action", None) == "vjp"][-1]
    for task in range(1, len(lin.frequencies) + 1):
        manifest = DataVector.read_objective_manifest(
            vjp.task_input(vjp.objective_vector, task)
        )
        assert Path(manifest["file"]).is_file()
        assert [term["id"] for term in manifest["terms"]] == list(labels)


def test_saved_sparse_rows_and_missing_legacy_residual(tmp_path):
    _, problem = _problem(tmp_path, FakeImagingSite())
    lin = problem.linearize()
    # Change keys to a valid sparse/projected coordinate sequence, keeping rows.
    for task in (1, 2):
        path = lin.job.state_file(task)
        manifest = json.loads(path.read_text())
        term = manifest["terms"][0]
        cache = path.parent / term["cache"]["file"]
        with h5py.File(cache, "a") as h5:
            group = h5["terms/0"]
            keys = group["coordinate_keys"][...]
            keys[:, 1] *= 2
            group["coordinate_keys"][...] = keys
            if task == 2:
                del group["objective_residual"]
        term["runtime"]["cache_fingerprint"] = file_sha256(cache)
        path.write_text(json.dumps(manifest))
    states = [ObjectiveState(lin.job.state_file(t)) for t in (1, 2)]
    space = objective_space(problem.simulation, lin.frequencies, states)
    layout = space.term_layout("surface", frequency=4.0)
    assert np.all(layout.coordinate_keys[:, 1] % 2 == 0)
    assert layout.complete
    with pytest.raises(NotImplementedError, match="regenerate"):
        objective_residual(space, states)


def test_cached_extension_linearization_adopts_support_on_a_refreshed_view(tmp_path):
    fake = FakeImagingSite(support_masks={"model.vp": [1, 0, 1, 1, 0]})
    _, problem = _extension_problem(tmp_path, fake)
    extended = problem.extend(_extension())
    lin = extended.linearize()
    view = extended.restrict(support="refresh")
    assert view.space.size == 8
    assert view.linearize() is lin
    assert view.space.equivalent(lin.space)
