"""Composite objectives use Sauce value/prox callbacks without changing derivatives."""

import json
from pathlib import Path

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._artifacts import ControlVectorFile
from frequensolve.imaging._native_regularization import bind_workflow_regularization
from frequensolve.imaging.jobs import RegularizationJob
from frequensolve.imaging.workflows import _StageObjective
from frequensolve.inversion.optimization import (
    InexactNewtonOptions,
    minimize_proximal_gradient,
)
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_workflows import FREQUENCIES, _problem

pytestmark = pytest.mark.unit


def test_problem_derivatives_are_unchanged_by_regularization_configuration(tmp_path):
    problem = _problem(
        tmp_path, FakeImagingSite(seed=7), smoothing=im.Smoothing(kind="tv", alpha=0.1)
    )
    x, d = problem.space.random(5), problem.space.random(6)
    lin = problem.linearize(x)
    assert lin.job.smoothing is None
    assert not hasattr(lin, "smoothed_gradient")
    bound = im.Quadratic(np.eye(problem.space.size), weight=0.3).bind(problem.space)
    objective = _StageObjective(problem, problem.space, bound, None, {})
    h = 1e-5
    fd = (
        objective.value(x.values + h * d.values)
        - objective.value(x.values - h * d.values)
    ) / (2 * h)
    assert objective.gradient(x.values) @ d.values == pytest.approx(fd, rel=1e-8)
    fdg = (
        objective.gradient(x.values + h * d.values)
        - objective.gradient(x.values - h * d.values)
    ) / (2 * h)
    np.testing.assert_allclose(
        objective.hessian_action(x.values, d.values), fdg, rtol=1e-8
    )


def test_composite_line_search_can_leave_a_data_stationary_point():
    # At x=3 the data gradient is zero. Regularization still decreases by moving.
    events = []
    result = minimize_proximal_gradient(
        lambda x: 0.5 * float((x - 3) @ (x - 3)),
        lambda x: x - 3,
        lambda x: float(np.abs(x).sum()),
        lambda x, t, box: np.clip(np.sign(x) * np.maximum(np.abs(x) - t, 0), *box),
        np.array([3.0]),
        options=InexactNewtonOptions(
            max_iterations=20,
            gradient_tolerance=1e-10,
            step_tolerance=0,
            objective_tolerance=0,
        ),
        callback=events.append,
    )
    assert result.success
    np.testing.assert_allclose(result.model, [2.0], atol=1e-10)
    assert result.objective == pytest.approx(2.5)
    assert events[0].objective < 3.0


def test_composite_backtracking_and_nonsmooth_stationarity():
    events = []
    target = np.array([0.01, 2.0, -4.0])
    result = minimize_proximal_gradient(
        lambda x: 5 * float((x - target) @ (x - target)),
        lambda x: 10 * (x - target),
        lambda x: float(np.abs(x).sum()),
        lambda x, t, box: np.clip(np.sign(x) * np.maximum(np.abs(x) - t, 0), *box),
        np.array([0.0, 0.0, 0.0]),
        bounds=(np.array([-5.0, 0.0, -2.0]), np.array([5.0, 1.0, 5.0])),
        options=InexactNewtonOptions(
            max_iterations=100,
            gradient_tolerance=1e-8,
            step_tolerance=0,
            objective_tolerance=0,
        ),
        callback=events.append,
    )
    assert result.success
    np.testing.assert_allclose(result.model, [0.0, 1.0, -2.0], atol=1e-7)
    assert events[0].line_search_evaluations > 1
    assert np.all(np.diff([e.objective for e in events]) <= 1e-12)


@pytest.mark.parametrize("optimizer", [im.NewtonCG, im.LBFGS])
def test_fwi_uses_native_value_and_prox_callbacks(tmp_path, optimizer):
    site = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, site)
    spec = im.NativeRegularization(
        im.Smoothing(kind="tv", alpha=0.04, normalize_amplitude=False)
    )
    result = im.FWI(
        problem,
        im.Stage(FREQUENCIES, 150),
        regularization=spec,
        optimizer=optimizer(
            objective_tolerance=0, step_tolerance=0, gradient_tolerance=1e-7
        ),
        scaling={"vp": 4.0, "rho": 0.5},
    ).run()
    x = result.state.vector(problem.space).values
    g = problem.gradient(x).values
    # Exact KKT conditions for the fake site's independent L1 energy.
    nonzero = np.abs(x) > 1e-6
    np.testing.assert_allclose(g[nonzero] + 0.2 * np.sign(x[nonzero]), 0, atol=2e-5)
    assert np.all(np.abs(g[~nonzero]) <= 0.2 + 2e-5)
    assert result.loss.regularization == pytest.approx(0.2 * np.abs(x).sum())
    jobs = [j for j in site.jobs if isinstance(j, RegularizationJob)]
    assert {j.operation for j in jobs} == {"prepare", "value", "proximal"}
    assert len([j for j in jobs if j.operation == "prepare"]) == 1
    assert all(j.smoothing.kind == "tv" for j in jobs)
    losses = [e.loss.total for e in result.history.iterations]
    assert np.all(np.diff(losses) <= 1e-10)


def test_native_callbacks_keep_frozen_values_and_scaled_metric(tmp_path):
    site = FakeImagingSite(seed=7, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    problem = _problem(tmp_path, site)
    problem.state = problem.state_from(np.arange(1.0, 9.0))
    lin = problem.linearize()
    spec = im.NativeRegularization(im.Smoothing(alpha=0.5, normalize_amplitude=False))
    _, native = bind_workflow_regularization(spec, lin.space, problem, lin)
    x = lin.point.values
    assert native.value(x) == pytest.approx(0.25 * np.sum(lin.state.values**2))
    metric = np.linspace(1.0, 2.0, lin.space.size)
    got = native.prox(x, 0.3, metric, lin.space.bounds)
    np.testing.assert_allclose(got, x / (1 + 0.15 / metric))
    job = next(j for j in reversed(site.jobs) if isinstance(j, RegularizationJob))
    full = ControlVectorFile.read(job.gradient_file(), native=True)
    np.testing.assert_array_equal(full["vp"][[1, 4]], lin.state["model.vp"][[1, 4]])


def test_native_checkpoint_reuses_frozen_normalization(tmp_path):
    site = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, site)
    problem.state = problem.state_from(np.arange(1.0, 9.0))
    lin = problem.linearize()
    spec = im.NativeRegularization(
        im.Smoothing(kind="tv", alpha=0.04, normalize_amplitude=True)
    )
    _, first = bind_workflow_regularization(spec, lin.space, problem, lin)
    saved = first.checkpoint()
    trial = lin.point.values * 0.25
    expected = first.value(trial)
    other = problem.linearize(trial)
    _, resumed = bind_workflow_regularization(spec, other.space, problem, other)
    assert resumed.value(trial) != pytest.approx(expected)
    resumed.restore(saved)
    assert resumed.value(trial) == pytest.approx(expected)
    assert json.loads(resumed.context.read_text()) == saved["context"]


@pytest.mark.parametrize(
    "spec, expected",
    [
        (
            im.Tikhonov(0.3, order=2),
            {"type": "tikhonov", "alpha": 0.3, "derivative_order": 2},
        ),
        (im.TV(0.4, epsilon=0.02), {"type": "tv", "alpha": 0.4, "epsilon": 0.02}),
        (
            im.TGV(0.5, 0.6, epsilon=0.03),
            {"type": "tgv", "alpha1": 0.5, "alpha2": 0.6, "epsilon": 0.03},
        ),
    ],
)
def test_model_specifications_dispatch_to_sauce(tmp_path, spec, expected):
    site = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, site)
    lin = problem.linearize()
    quadratic = im.Quadratic(np.eye(lin.space.size), weight=0.2)
    smooth, native = bind_workflow_regularization(
        quadratic + 2.0 * spec, lin.space, problem, lin
    )
    prepare = next(j for j in site.jobs if isinstance(j, RegularizationJob))
    request = prepare.to_fs()["control_sensitivities"]["Regularization"]
    for key, value in expected.items():
        assert request[key] == value
    assert request["operation"] == "prepare"
    assert native.factor == 2.0
    x = lin.space.random(3).values
    assert smooth.value(x) == pytest.approx(0.1 * (x @ x))
    native.value(x)
    native.prox(x, 0.3, np.ones(x.size), lin.space.bounds)
    assert [j.operation for j in site.jobs if isinstance(j, RegularizationJob)] == [
        "prepare",
        "gradient" if isinstance(spec, im.Tikhonov) else "value",
        "proximal",
    ]
    assert site.jobs[-1].tau == pytest.approx(0.6)


def test_native_specification_preserves_reference_state(tmp_path):
    site = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, site)
    reference = problem.state_from(np.arange(1.0, 9.0))
    lin = problem.linearize()
    _, native = bind_workflow_regularization(
        im.TV(0.04, reference=reference), lin.space, problem, lin
    )
    assert native.value(reference.vector(lin.space)) == pytest.approx(0.0)


def test_native_tikhonov_derivatives_with_reference_and_frozen_dofs(tmp_path):
    site = FakeImagingSite(seed=7, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    problem = _problem(tmp_path, site)
    problem.state = problem.state_from(np.arange(1.0, 9.0))
    lin = problem.linearize()
    reference = im.ControlState(lin.state.space, lin.state.values * 0.2)
    _, native = bind_workflow_regularization(
        2 * im.Tikhonov(0.3, reference=reference), lin.space, problem, lin
    )
    x = lin.point.values
    direction = lin.space.random(17).values
    h = 1e-4
    fd = (native.value(x + h * direction) - native.value(x - h * direction)) / (2 * h)
    assert native.gradient(x).values @ direction == pytest.approx(fd, rel=1e-9)
    fdg = (
        native.gradient(x + h * direction).values
        - native.gradient(x - h * direction).values
    ) / (2 * h)
    np.testing.assert_allclose(native.hessian_operator() @ direction, fdg, rtol=1e-9)
    np.testing.assert_allclose(fdg, 0.6 * direction, rtol=1e-9)
    np.testing.assert_allclose(native.curvature_diagonal(), 0.6)


def test_fwi_tikhonov_keeps_lbfgs_and_adds_energy_once(tmp_path):
    site = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, site)
    result = im.FWI(
        problem,
        im.Stage(FREQUENCIES, 100),
        regularization=im.Tikhonov(0.3),
        optimizer=im.LBFGS(
            objective_tolerance=0, step_tolerance=0, gradient_tolerance=1e-7
        ),
    ).run()
    x = result.state.vector(problem.space).values
    np.testing.assert_allclose(problem.gradient(x).values + 0.3 * x, 0, atol=3e-6)
    assert result.loss.regularization == pytest.approx(0.15 * (x @ x))
    jobs = [j for j in site.jobs if isinstance(j, RegularizationJob)]
    assert {j.operation for j in jobs} == {"prepare", "gradient"}
    assert all(e.metrics["optimizer"] == "lbfgs" for e in result.history.iterations)


def test_smooth_native_tikhonov_resumes_with_context(tmp_path):
    site = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, site)
    options = dict(
        regularization=im.Tikhonov(0.3),
        optimizer=im.LBFGS(
            objective_tolerance=0, step_tolerance=0, gradient_tolerance=1e-7
        ),
        checkpoint=tmp_path / "checkpoint.h5",
        history=tmp_path / "history.json",
    )

    def interrupt(event):
        if event.stage_iteration == 2:
            raise RuntimeError("intentional interruption")

    with pytest.raises(RuntimeError, match="intentional interruption"):
        im.FWI(problem, im.Stage(FREQUENCIES, 100), callback=interrupt, **options).run()
    result = im.FWI(problem, im.Stage(FREQUENCIES, 100), **options).run(resume=True)
    x = result.state.vector(problem.space).values
    np.testing.assert_allclose(problem.gradient(x).values + 0.3 * x, 0, atol=3e-6)
    assert result.stages[0].resumed
    assert result.stages[0].metrics["optimizer"] == "lbfgs"


def test_native_cache_hits_hash_the_vector_without_forming_the_model(
    tmp_path, monkeypatch
):
    site = FakeImagingSite(seed=7, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    problem = _problem(tmp_path, site)
    problem.state = problem.state_from(np.arange(1.0, 9.0))
    lin = problem.linearize()
    spec = im.NativeRegularization(
        im.Smoothing(kind="tv", alpha=0.04, normalize_amplitude=False)
    )
    _, native = bind_workflow_regularization(spec, lin.space, problem, lin)
    x = lin.point.values
    expected = native.value(x)
    built = []
    full = native._full
    monkeypatch.setattr(native, "_full", lambda v: built.append(v) or full(v))
    jobs = len(site.jobs)
    frozen = x.copy()
    frozen.flags.writeable = False
    # Equal coordinates in any container are one key, stable within the run.
    for same in (x.copy(), frozen, im.ControlVector(x, lin.space), list(x)):
        assert native.value(same) == expected
    assert len(site.jobs) == jobs and not built
    model = native.prox(x, 0.3, np.ones(x.size), lin.space.bounds)
    jobs, built[:] = len(site.jobs), []
    # Sauce's prox energy of the returned model serves the line search's value.
    native.value(np.array(model))
    assert len(site.jobs) == jobs and not built
    native.value(0.5 * x)
    assert len(site.jobs) == jobs + 1 and len(built) == 1


def test_prox_stages_metric_and_bounds_once_per_content(tmp_path, monkeypatch):
    site = FakeImagingSite(seed=7, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    problem = _problem(tmp_path, site)
    problem.state = problem.state_from(np.arange(1.0, 9.0))
    lin = problem.linearize()
    spec = im.NativeRegularization(
        im.Smoothing(kind="tv", alpha=0.04, normalize_amplitude=False)
    )
    _, native = bind_workflow_regularization(spec, lin.space, problem, lin)
    hashed = []
    fingerprint = RegularizationJob._path_content_fingerprint.__func__
    monkeypatch.setattr(
        RegularizationJob,
        "_path_content_fingerprint",
        classmethod(
            lambda cls, path: hashed.append(Path(path)) or fingerprint(cls, path)
        ),
    )
    x = lin.point.values
    metric = np.linspace(1.0, 2.0, x.size)
    bounds = tuple(np.array(b, dtype=float) for b in lin.space.bounds)
    for array in (metric, *bounds):
        array.flags.writeable = False

    def proximal_jobs():
        return [
            j
            for j in site.jobs
            if isinstance(j, RegularizationJob) and j.operation == "proximal"
        ]

    def status(paths):
        return {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in paths.values()}

    remote = tmp_path / "remote"
    native.prox(x, 0.3, metric, bounds)
    (first,) = proximal_jobs()
    staged = dict(first.regularization_inputs)
    # Beside the stage's native context, named by content: a stable path each.
    assert set(staged) == {"metric", "lower", "upper"}
    assert {path.parent for path in staged.values()} == {native.context.parent}
    written = status(staged)
    assert [hashed.count(path) for path in staged.values()] == [1, 1, 1]
    uploads = {
        pair for pair in first.remote_input_files(remote) if Path(pair[0]) in written
    }
    assert len(uploads) == 3
    del hashed[:]
    # Later calls (the same read-only arrays, or equal content in new ones)
    # write only their target vector and reuse the recorded fingerprints.
    native.prox(0.5 * x, 0.3, metric, bounds)
    native.prox(0.25 * x, 0.2, np.array(metric), tuple(np.array(b) for b in bounds))
    jobs = proximal_jobs()
    assert len(jobs) == 3
    for job in jobs[1:]:
        assert job.regularization_inputs == staged
        # The same local files at the same remote paths: uploaded once.
        assert {
            pair for pair in job.remote_input_files(remote) if Path(pair[0]) in written
        } == uploads
        assert not any(
            (job._result_path / f"regularization_{label}.h5").exists()
            for label in staged
        )
    assert status(staged) == written
    assert not set(hashed) & set(staged.values())
    # New metric content stages one new file; the unchanged bounds stay put.
    native.prox(x, 0.3, 2.0 * metric, bounds)
    changed = proximal_jobs()[-1].regularization_inputs
    assert changed["metric"] != staged["metric"] and staged["metric"].exists()
    assert {k: changed[k] for k in ("lower", "upper")} == {
        k: staged[k] for k in ("lower", "upper")
    }
    assert status(staged) == written
    np.testing.assert_array_equal(
        ControlVectorFile.read(changed["metric"], native=True)["vp"][[0, 2, 3]],
        2.0 * metric[:3],
    )
