"""Independent dense oracles for shared-frequency reflectivity elimination."""

import numpy as np
import pytest

from frequensolve.imaging._rwi import _ReflectivitySystem
from frequensolve.inversion._krylov import _preconditioned_cg

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("weights", [[0.3, 2.0, 0.0], [1.0, 1.0, 1.0]])
def test_shared_fit_and_schur_use_inverse_of_sum(weights):
    rng = np.random.default_rng(901)
    matrices = [rng.normal(size=(9, 5)) for _ in weights]
    data = [rng.normal(size=9) for _ in weights]
    normal = sum(w * j.T @ j for w, j in zip(weights, matrices))
    rhs = sum(w * j.T @ d for w, j, d in zip(weights, matrices, data))
    outer, inner = [0, 3], [1, 2, 4]
    damping = 0.4
    system = _ReflectivitySystem(
        lambda x: normal @ x,
        5,
        outer,
        inner,
        damping=damping,
        solver=dict(
            max_iterations=10, relative_tolerance=1e-12, absolute_tolerance=1e-14
        ),
    )
    point = rng.normal(size=5)
    gradient = normal @ point - rhs
    delta, report = system.fit_step(gradient[inner], point[inner])
    D = normal[np.ix_(inner, inner)] + damping * np.eye(3)
    C = normal[np.ix_(inner, outer)]
    expected = np.linalg.solve(D, rhs[inner] - C @ point[outer])
    np.testing.assert_allclose(point[inner] + delta, expected, atol=2e-13)
    assert report["converged"] and report["true_residual_norm"] < 1e-12
    expected_schur = normal[np.ix_(outer, outer)] - C.T @ np.linalg.solve(D, C)
    actual = np.column_stack([system.reduced_normal(d)[0] for d in np.eye(2)])
    np.testing.assert_allclose(actual, expected_schur, atol=2e-13)
    np.testing.assert_allclose(actual, actual.T, atol=2e-13)
    assert np.linalg.eigvalsh(actual).min() > 0
    independent = np.zeros((2, 2))
    for w, j in zip(weights, matrices):
        n = j.T @ j
        c = n[np.ix_(inner, outer)]
        independent += w * (
            n[np.ix_(outer, outer)]
            - c.T @ np.linalg.solve(n[np.ix_(inner, inner)] + damping * np.eye(3), c)
        )
    assert np.linalg.norm(actual - independent) > 0.1


def test_true_residual_rejects_inconsistent_action():
    calls = 0

    def action(x):
        nonlocal calls
        calls += 1
        return x if calls % 2 else 2 * x

    _, report = _preconditioned_cg(
        action, np.ones(3), iterations=3, tolerance=1e-8, verify_residual=True
    )
    assert not report["converged"]
    assert report["true_residual_norm"] > 1
    assert report["iterations"] == 3


@pytest.mark.parametrize("iterations,converged", [(1, False), (3, True)])
def test_true_residual_restarts_after_an_inexact_product(iterations, converged):
    calls = 0

    def action(x):
        nonlocal calls
        calls += 1
        # Model a small inexact PDE action; subsequent products resolve it.
        return (1.001 if calls == 1 else 1.0) * x

    rhs = np.array([1.0, 2.0, 3.0])
    solution, report = _preconditioned_cg(
        action, rhs, iterations=iterations, tolerance=1e-4, verify_residual=True
    )
    assert report["converged"] == converged
    assert report["iterations"] <= iterations
    np.testing.assert_allclose(
        report["true_residual_norm"], np.linalg.norm(rhs - solution)
    )
    if converged:
        assert report["residual_replacements"] == 1
        assert report["true_residual_norm"] <= 1e-4 * np.linalg.norm(rhs)
    else:
        assert report["residual_replacements"] == 0
        assert report["true_residual_norm"] > 1e-4 * np.linalg.norm(rhs)


def test_response_failure_and_absolute_tolerance():
    options = dict(max_iterations=0, relative_tolerance=1e-4, absolute_tolerance=1e-9)
    system = _ReflectivitySystem(
        lambda x: np.array([[2.0, 1.0], [1.0, 2.0]]) @ x,
        2,
        [0],
        [1],
        damping=0.1,
        solver=options,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        system.reduced_normal([1.0])
    response, report = system.solve([1e-10])
    np.testing.assert_array_equal(response, 0)
    assert report["converged"]


@pytest.mark.parametrize("weights", [[0.3, 2.0], [0.0, 1.0]])
def test_sdk_dispatch_and_fitted_normal(tmp_path, weights):
    from frequensolve import imaging as im
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_reflectivity import _problem

    fake = FakeImagingSite(block_sizes={"reflectivity.ip": 5}, seed=3)
    _, original = _problem(tmp_path, fake)
    problem = original.restrict(weights=weights)
    rwi = problem.rwi(damping=0.2, relative_tolerance=1e-10, absolute_tolerance=1e-12)
    baseline = problem.state
    fit = rwi.linearize()
    assert (fit._joint.entry.directory / "reflectivity.h5").is_file()
    assert (fit._joint.entry.directory / "reflectivity.json").is_file()
    joint = problem.linearize(baseline)
    matrix = np.asarray(joint.normal @ np.eye(joint.space.size))
    inner = np.arange(5, 10)
    outer = np.arange(5)
    D = matrix[np.ix_(inner, inner)] + 0.2 * np.eye(5)
    C = matrix[np.ix_(inner, outer)]
    expected = baseline.vector(joint.space).values[inner] - np.linalg.solve(
        D,
        joint.gradient.values[inner] + 0.2 * baseline.vector(joint.space).values[inner],
    )
    np.testing.assert_allclose(fit.state["refl"], expected, atol=1e-10)
    schur = matrix[:5, :5] - C.T @ np.linalg.solve(D, C)
    actual = np.asarray(fit.normal @ np.eye(5))
    np.testing.assert_allclose(actual, schur, atol=2e-10)
    assert fit.normal.H is not None
    assert all(job.action != "solve" for job in fake.jobs)
    assert all(job.frequency_groups == 1 for job in fake.jobs)
    assert all(job.n_tasks == 2 for job in fake.jobs)
    for job in fake.jobs:
        if job.action == "normal":
            assert job.active == ["model.vp", "reflectivity.ip"]
    np.testing.assert_array_equal(problem.state.values, baseline.values)
    assert rwi.linearize() is fit
    direction = np.arange(1.0, 6.0) / 100
    h = 1e-4
    fd = (
        rwi.value(fit.point + h * direction) - rwi.value(fit.point - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(
        fd, fit.gradient.values @ direction, rtol=1e-7, atol=1e-9
    )
    # Later fitted points or solver settings must not change an earlier operator.
    rwi.solver["damping"] = 3.0
    replay = fit.apply_normal_batch(np.eye(5), cache=False)
    np.testing.assert_allclose(
        np.column_stack([v.values for v in replay]), schur, atol=2e-10
    )
    preconditioner = im.Diagonal(probes="unit").bind(fit.space)
    preconditioner.update(fit)
    np.testing.assert_allclose(preconditioner.estimate.data, np.diag(schur), atol=2e-10)


def test_fit_preserves_frozen_fields_support_and_state_identity(tmp_path):
    from frequensolve import imaging as im
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_reflectivity import _problem, _reflectivity

    controls = im.ControlSpace(
        vp=im.DepthProfile("vp", "sediment", count=5),
        refl=_reflectivity(),
        frozen=im.ReflectivityParameters(
            "vp_ip", fields=[im.ReflectivityField("dv", layer=2, axis=1, basis="vp")]
        ),
    )
    fake = FakeImagingSite(
        block_sizes={"reflectivity.ip": 5, "reflectivity.dv": 5},
        seed=4,
        support_masks={"reflectivity.ip": [1, 0, 1, 1, 1], "model.vp": [1, 1, 0, 1, 1]},
    )
    _, problem = _problem(tmp_path, fake, controls=controls)
    state = problem.state.with_update(
        problem.full_space.pack(
            {
                "vp": problem.state["vp"],
                "refl": np.ones(5) * 0.2,
                "frozen": np.ones(5) * 0.1,
            }
        )
    )
    rwi = problem.rwi(
        reflectivity="refl",
        damping=0.1,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-11,
    )
    fit = rwi.linearize(state)
    assert fit.space.size == 4
    assert fit.state["refl"][1] == 0.2
    np.testing.assert_array_equal(fit.state["frozen"], 0.1)
    for job in fake.jobs:
        if job.action == "normal":
            assert len(job.reflectivity["fields"]) == 2
            assert job.active == ["model.vp", "reflectivity.ip"]
    changed = fit.state.with_update(
        problem.full_space.pack(
            {
                "vp": fit.state["vp"],
                "refl": np.array([0.0, 0.7, 0.0, 0.0, 0.0]),
                "frozen": np.ones(5) * 0.1,
            }
        )
    )
    new = rwi.linearize(changed)
    assert new is not fit and new.state["refl"][1] == 0.7
    assert rwi.linearize(changed) is new
    assert new.fingerprint != fit.fingerprint


def test_outer_curvature_scaling_regularization_and_newton_reuse(tmp_path):
    from frequensolve import imaging as im
    from frequensolve.imaging.workflows import block_curvatures
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_reflectivity import _problem

    fake = FakeImagingSite(block_sizes={"reflectivity.ip": 5}, seed=3)
    _, problem = _problem(tmp_path, fake)
    rwi = problem.rwi(damping=0.2, relative_tolerance=1e-10, absolute_tolerance=1e-11)
    fit = rwi.linearize()
    direction = fit.space.random(22)
    penalty = im.Quadratic(np.eye(fit.space.size), weight=0.3).bind(fit.space)
    plain = fit.apply_normal(direction)
    regularized = fit.apply_normal(direction, regularization=penalty)
    np.testing.assert_allclose(
        regularized.values,
        plain.values
        + np.asarray(penalty.hessian_operator(fit.point) @ direction.values),
    )
    assert block_curvatures(fit)["model.vp"] > 0
    result = im.FWI(
        rwi,
        [im.Stage(rwi.frequencies, iterations=3)],
        optimizer=im.NewtonCG(),
        scaling="curvature",
    ).run(resume=False)
    assert rwi.value(result.state) < fit.value


def test_failed_frequency_action_cannot_publish_fit(tmp_path, monkeypatch):
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_reflectivity import _problem

    fake = FakeImagingSite(block_sizes={"reflectivity.ip": 5}, seed=3)
    _, problem = _problem(tmp_path, fake)
    rwi = problem.rwi(damping=0.2)
    original = problem._run_job

    def run(job):
        if job.action == "normal":
            raise RuntimeError("frequency task failed")
        return original(job)

    monkeypatch.setattr(type(problem), "_run_job", lambda self, job: run(job))
    with pytest.raises(RuntimeError, match="frequency task failed"):
        rwi.linearize()
    assert not rwi._fits and rwi._warm is None


def test_recomputed_stationarity_is_required(tmp_path, monkeypatch):
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_reflectivity import _problem

    fake = FakeImagingSite(block_sizes={"reflectivity.ip": 5}, seed=3)
    _, problem = _problem(tmp_path, fake)
    rwi = problem.rwi(damping=0.2, require_convergence=False)
    linearize = type(problem)._linearize_state

    def inconsistent(self, state, **kwargs):
        result = linearize(self, state, **kwargs)
        if result.gradient is not None and np.any(result.state["refl"]):
            result.gradient = result.gradient + 0.1
        return result

    monkeypatch.setattr(type(problem), "_linearize_state", inconsistent)
    with pytest.raises(RuntimeError, match="not stationary"):
        rwi.linearize()
    assert not rwi._fits and rwi._warm is None


def test_warm_start_keeps_the_same_absolute_stationarity_target():
    matrix = np.diag([2.0, 4.0])
    options = dict(max_iterations=5, relative_tolerance=1e-4, absolute_tolerance=0.0)
    system = _ReflectivitySystem(
        lambda x: matrix @ x, 2, [], [0, 1], damping=0.5, solver=options
    )
    rhs = np.array([1.0, 2.0])
    reports = []
    for initial in (np.zeros(2), np.array([1e6, -1e6]), rhs / np.array([2.5, 4.5])):
        update, report = system.fit_step(matrix @ initial - rhs, initial)
        reports.append(report)
        np.testing.assert_allclose(
            initial + update, rhs / np.array([2.5, 4.5]), atol=1e-9
        )
    np.testing.assert_allclose(
        [r["tolerance"] for r in reports], 1e-4 * np.linalg.norm(rhs)
    )


def test_outer_restart_preserves_fitted_reflectivity(tmp_path):
    from frequensolve import imaging as im
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_reflectivity import _problem

    fake = FakeImagingSite(block_sizes={"reflectivity.ip": 5}, seed=3)
    _, problem = _problem(tmp_path, fake)
    rwi = problem.rwi(damping=0.2, relative_tolerance=1e-10, absolute_tolerance=1e-10)
    checkpoint = tmp_path / "rwi.ckpt.h5"
    options = dict(
        checkpoint=checkpoint,
        step_limit=0.05,
        optimizer=im.LBFGS(
            gradient_tolerance=0.0, objective_tolerance=0.0, step_tolerance=0.0
        ),
    )

    def interrupt(event):
        if event.stage_iteration == 1:
            raise RuntimeError("test interruption")

    with pytest.raises(RuntimeError, match="test interruption"):
        im.FWI(
            rwi,
            [im.Stage(rwi.frequencies, iterations=3)],
            callback=interrupt,
            **options,
        ).run(resume=False)
    assert checkpoint.is_file()
    resumed = im.FWI(rwi, [im.Stage(rwi.frequencies, iterations=3)], **options).run(
        resume=True
    )
    assert resumed.stages[0].resumed
    fit = rwi.linearize(resumed.state)
    np.testing.assert_allclose(
        resumed.state["refl"], fit.state["refl"], rtol=1e-8, atol=1e-9
    )


@pytest.mark.parametrize("chart", ["vp_vs_ip", "ip_is_rho"])
def test_elastic_sdk_fits_all_coordinates_and_reduces_joint_normal(tmp_path, chart):
    from frequensolve import imaging as im
    from frequensolve.model.layered import LayeredModel
    from frequensolve.project import Project
    from frequensolve.seismic import Acquisition, ReceiverNode
    from frequensolve.simulation import Discretization
    from tests.imaging_fakes import FakeImagingSite

    project = Project(name="elastic", path=tmp_path / "project", load_if_exists=False)
    simulation = project.new_simulation(name="tti", physics="elastic", dimension=2)
    model = LayeredModel(dimension=2, x_limits=[0, 1])
    model.add_surface(name="top", depth=0)
    model.add_layer(
        name="solid",
        physics="elastic:tti",
        properties=dict(vp=2.5, vs=1.3, rho=2.2, epsilon=0.18, delta=0.05, theta=0.37),
    )
    model.add_surface(name="bottom", depth=1)
    simulation += model
    simulation += model.hex_mesh_generator(n=[2, 2])
    acquisition = Acquisition()
    acquisition.add_sources(kind="vector", coords=[[0.5, 0.1]], direction=[0, 1])
    receiver = ReceiverNode(name="geophone")
    receiver.add_component(name="vz", field="velocity", direction=[0, 1])
    acquisition.add_receiver_group(name="surface", device=receiver, coords=[[0.2, 0.1]])
    simulation += acquisition
    simulation += Discretization(method="DPG")
    simulation.save()
    names = ["vp", "vs", "ip"] if chart == "vp_vs_ip" else ["ip", "is", "rho"]
    fake = FakeImagingSite(
        block_sizes={f"reflectivity.{name}": 3 for name in names}, seed=7
    )
    problem = im.ImagingProblem(
        simulation,
        controls=im.ControlSpace(
            theta=im.DepthProfile("theta", "solid", count=3),
            refl=im.ReflectivityParameters(
                chart,
                fields=[
                    im.ReflectivityField(name, layer=1, axis=i + 1, basis="theta")
                    for i, name in enumerate(names)
                ],
            ),
        ),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=[2.0, 3.0],
        site=fake,
    ).restrict(weights=[0.5, 1.5])
    rwi = problem.rwi(damping=0.2, relative_tolerance=1e-10)
    baseline = problem.linearize()
    fit = rwi.linearize()
    normal = np.asarray(baseline.normal @ np.eye(baseline.space.size))
    outer, inner = np.arange(3), np.arange(3, 12)
    D = normal[np.ix_(inner, inner)] + 0.2 * np.eye(9)
    C = normal[np.ix_(inner, outer)]
    expected = -np.linalg.solve(D, baseline.gradient.values[inner])
    np.testing.assert_allclose(
        fit.state.vector(baseline.space).values[inner], expected, atol=1e-9
    )
    schur = normal[np.ix_(outer, outer)] - C.T @ np.linalg.solve(D, C)
    np.testing.assert_allclose(np.asarray(fit.normal @ np.eye(3)), schur, atol=1e-9)
    assert all(job.action != "solve" for job in fake.jobs)
    assert all(job.reflectivity["parameterization"] == chart for job in fake.jobs)
    with pytest.raises(ValueError, match="use execution='sdk'"):
        problem.rwi(execution="native")
