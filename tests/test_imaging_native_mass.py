"""Native Riesz maps remain inverse metrics, never substitute derivatives."""

import numpy as np
import pytest

from frequensolve import imaging as im
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_workflows import FREQUENCIES, _problem

pytestmark = pytest.mark.unit


def test_mass_inverse_pairing_and_curvature_calibration(tmp_path):
    problem = _problem(tmp_path, FakeImagingSite(seed=7))
    lin = problem.linearize()
    bound = im.NativeMass().bind(lin.space)
    regularization = im.Quadratic(np.eye(lin.space.size), weight=0.3).bind(lin.space)
    bound.update(lin, regularization)
    a = lin.space.random(12).values
    b = lin.space.random(13).values
    np.testing.assert_allclose(bound.riesz(bound.mass(a)).values, a, atol=1e-12)
    pa, pb = bound.riesz(a).values, bound.riesz(b).values
    assert a @ pb == pytest.approx(b @ pa)
    assert a @ pa > 0
    gradient = lin.gradient.values + regularization.gradient(lin.point).values
    d = bound.riesz(gradient).values
    hd = lin.normal @ d + regularization.hessian_operator(lin.point) @ d
    assert bound.scale == pytest.approx((gradient @ d) / (d @ np.asarray(hd)))


@pytest.mark.parametrize("approximation", ["consistent", "diagonal"])
def test_mass_preconditioned_lbfgs_keeps_true_derivative_and_descent(
    tmp_path, approximation
):
    problem = _problem(tmp_path, FakeImagingSite(seed=7))
    events = []
    result = im.FWI(
        problem,
        im.Stage(FREQUENCIES, 100),
        optimizer=im.LBFGS(
            objective_tolerance=0, step_tolerance=0, gradient_tolerance=1e-7
        ),
        regularization=im.Tikhonov(0.3),
        preconditioner=im.NativeMass(approximation=approximation),
        scaling={"vp": 3.0, "rho": 0.5},
        callback=events.append,
    ).run()
    x = result.state.vector(problem.space).values
    np.testing.assert_allclose(problem.gradient(x).values + 0.3 * x, 0, atol=3e-6)
    losses = [e.loss.total for e in events]
    assert np.all(np.diff(losses) <= 1e-12)
    for previous, current in zip(events, events[1:]):
        g = problem.gradient(previous.model).values + 0.3 * previous.model.values
        step = current.model.values - previous.model.values
        assert g @ step < 0
        assert current.diagnostics.directional_derivative == pytest.approx(g @ step)


def test_diagonal_mass_is_cached_and_matches_mass_columns(tmp_path, monkeypatch):
    site = FakeImagingSite(seed=7)
    lin = _problem(tmp_path, site).linearize()
    specification = im.NativeMass(approximation="diagonal", curvature_scale=False)
    bound = specification.bind(lin.space)
    bound.update(lin)
    diagonal = np.array(
        [bound.mass(e).values[i] for i, e in enumerate(np.eye(lin.space.size))]
    )
    n_jobs = len(site.jobs)
    b = lin.space.random(12).values
    for _ in range(3):
        np.testing.assert_allclose(bound.apply(b).values, b / diagonal)
    assert len(site.jobs) == n_jobs  # No native callback or solve per application.
    bound.update(lin)
    next_stage = specification.bind(lin.space)
    next_stage.update(lin)
    np.testing.assert_array_equal(next_stage.apply(b).values, bound.apply(b).values)
    assert sum(getattr(j, "operation", None) == "mass_diagonal" for j in site.jobs) == 1
    assert not any(getattr(j, "operation", None) == "mass_inverse" for j in site.jobs)
    # Changed native geometry/basis identity must invalidate cross-stage reuse.
    import json

    original = site._postprocess_regularization

    def changed_basis(job):
        result = original(job)
        if job.operation == "prepare":
            context = json.loads(job.context.read_text())
            for block in context.values():
                block["identity"] += "/refined"
            job.context.write_text(json.dumps(context))
        return result

    monkeypatch.setattr(site, "_postprocess_regularization", changed_basis)
    specification.bind(lin.space).update(lin)
    assert sum(getattr(j, "operation", None) == "mass_diagonal" for j in site.jobs) == 2


def test_mass_rejects_partial_support_without_a_reduced_solve(tmp_path):
    problem = _problem(
        tmp_path, FakeImagingSite(seed=7, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    )
    lin = problem.linearize()
    with pytest.raises(ValueError, match="fully active"):
        im.NativeMass().bind(lin.space)
