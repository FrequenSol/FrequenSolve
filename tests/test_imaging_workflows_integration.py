"""End-to-end imaging workflows through the public API and a local Sauce build."""

import json

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic.sparse_survey import (
    ReceiverSampling,
    SparseSurvey,
    SparseTrace,
)
from frequensolve.simulation import FrequencyDomainJob
from tests.test_imaging_integration import (
    FREQUENCY,
    START_VP,
    TRUTH_VP,
    _executable,
    _simulation,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]


@pytest.fixture
def site():
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=4,
        shutdown_on_completion=False,
        # Unreleased local builds have no immutable release-pair declaration.
        solver_policy="warn",
    ) as local:
        yield local


def _problem(tmp_path, site, *, sparse=False, loss="l2"):
    project = Project(name="workflows", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)
    if sparse:
        for simulation in (truth, initial):
            survey = SparseSurvey(
                "selected",
                traces=[
                    SparseTrace(
                        source=1, receiver=receiver, component="p", trace_id=trace
                    )
                    for receiver, trace in [(1, 1), (9, 2), (17, 3)]
                ],
            )
            simulation.acquisition.add_survey(survey)
            simulation.acquisition.receiver_groups[0].sampling = (
                ReceiverSampling.sparse(survey)
            )
            simulation.save()
    observed = FrequencyDomainJob("observed", truth, [FREQUENCY])
    assert site.run(observed, check=True).successful
    return im.ImagingProblem(
        initial,
        controls=im.DepthProfile("vp", "layer_2", count=4),
        observed=im.ObservedData(observed),
        site=site,
        name="workflows",
        misfit=im.Misfit(loss=loss),
    )


def test_lsrtm_solvers_and_fwi_checkpoint_through_public_api(tmp_path, site):
    problem = _problem(tmp_path, site)
    lin = problem.linearize()
    reference = np.linspace(0.01, 0.04, lin.space.size)
    regularization = im.Quadratic(
        np.eye(lin.space.size), weight=0.2, reference=reference
    )
    images = {}
    for method in ("cg", "lsqr"):
        workflow = im.LSRTM(
            problem,
            method=method,
            iterations=25,
            tolerance=1e-4,
            damping=0.1,
            regularization=regularization,
        )
        image = workflow.run()
        assert workflow.info["converged"], workflow.info
        assert np.all(np.isfinite(image.values)) and image.norm() > 0
        residual = lin.weight_data(lin.jvp(image) + lin.objective_residual())
        # Stationarity of the actual regularized objective, independently of
        # the iterative solver's own stopping report.
        gradient = (
            lin.vjp(residual).values
            + 0.1 * image.values
            + regularization.bind(lin.space).gradient(image).values
        )
        assert np.linalg.norm(gradient) < 1e-3 * max(
            np.linalg.norm(lin.gradient.values), 0.01
        )
        images[method] = image.values
    np.testing.assert_allclose(images["cg"], images["lsqr"], rtol=1e-2, atol=1e-4)

    checkpoint = tmp_path / "checkpoint.h5"
    workflow = im.FWI(
        problem,
        im.Stage([FREQUENCY], iterations=1),
        step_limit=0.1,
        checkpoint=checkpoint,
        history=tmp_path / "history.json",
    )
    result = workflow.run()
    assert result.stages[0].iterations == 1
    assert result.stages[-1].final_loss.total < result.stages[0].initial_loss.total
    assert checkpoint.is_file() and workflow.state_path.is_file()
    # A new workflow instance must accept and restore the completed checkpoint.
    restored = im.FWI(
        problem,
        im.Stage([FREQUENCY], iterations=1),
        step_limit=0.1,
        checkpoint=checkpoint,
        history=tmp_path / "history.json",
    ).run()
    np.testing.assert_array_equal(restored.state.values, result.state.values)
    (tmp_path / "workflow_metrics.json").write_text(
        json.dumps(
            {
                "cg_image": images["cg"].tolist(),
                "lsqr_image": images["lsqr"].tolist(),
                "initial_objective": result.stages[0].initial_loss.total,
                "final_objective": result.stages[-1].final_loss.total,
            },
            indent=2,
        )
    )


def _background_files(job):
    root = job._result_path
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and "background" in path.relative_to(root).as_posix()
    ]


def test_lsrtm_native_regularization_job_counts_and_checkpoint_cleanup(tmp_path, site):
    problem = _problem(tmp_path, site)
    problem.linearize()
    # The default method is LSQR; native Tikhonov still runs CG.
    tikhonov = im.LSRTM(
        problem, iterations=25, regularization=im.Tikhonov(1e-3, order=1)
    )
    image = tikhonov.run()
    info = tikhonov.info
    assert info["method"] == "cg" and info["converged"], info
    assert np.all(np.isfinite(image.values)) and image.norm() > 0
    iterations = info["iterations"]
    folded = 0 if info["fused_regularization"] else iterations
    assert info["jobs"] == {
        "linearize": 1,
        "regularization_prepare": 1,
        "normal": iterations,
        **({"regularization_gradient": folded} if folded else {}),
    }
    background = info["background"]
    assert background["enabled"] and background["misses"] == 0, background
    assert background["reused_solves"] >= iterations
    job = tikhonov.linearization.job
    assert job.background is None and background["released_bytes"] > 0
    assert not _background_files(job)

    tv = im.LSRTM(problem, iterations=40, regularization=im.TV(1e-6))
    image = tv.run()
    assert tv.info["method"] == "proximal_gradient", tv.info
    assert np.all(np.isfinite(image.values))
    jobs = tv.info["jobs"]
    # Two power iterations, then one normal and one prox job per trial (the
    # prox meeting the tolerance needs no normal) and no energy evaluations.
    assert "regularization_value" not in jobs
    trials = jobs["regularization_proximal"]
    assert jobs["normal"] - 2 in {trials, trials - 1}
    assert not _background_files(tv.linearization.job)


@pytest.mark.parametrize("loss", ["l2", "huber"])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_saved_residual_and_adjoint_through_public_api(tmp_path, site, loss, sparse):
    problem = _problem(tmp_path, site, sparse=sparse, loss=loss)
    lin = problem.linearize()
    layout = lin.data_space.term_layout("surface", frequency=FREQUENCY)
    assert layout.n_global_rows == (3 if sparse else 17)
    if sparse:
        assert set(layout.coordinate_keys[:, 1]) == {1, 2, 3}
    residual = lin.objective_residual()
    np.testing.assert_allclose(
        lin.vjp(residual).values, lin.gradient.values, rtol=3e-3, atol=1e-6
    )
    assert lin.jacobian.dot_test(seed=3, tolerance=3e-3)["passed"]


def test_native_tv_fwi_objective_through_local_orchestration(tmp_path, site):
    """Execute native callbacks and PDE line-search trials through LocalSite."""
    from frequensolve.imaging._native_regularization import bind_workflow_regularization

    problem = _problem(tmp_path, site)
    specification = im.TV(alpha=0.002)
    result = im.FWI(
        problem,
        im.Stage([FREQUENCY], iterations=2),
        regularization=specification,
        step_limit=0.05,
        optimizer=im.LBFGS(objective_tolerance=0, step_tolerance=0),
    ).run()
    stage = result.stages[0]
    assert stage.final_loss.total < stage.initial_loss.total
    assert result.history.iterations[-1].metrics["optimizer"] == "proximal_gradient"
    point = result.state.vector(problem.space)
    linearization = problem.linearize(point)
    _, native = bind_workflow_regularization(
        specification, linearization.space, problem, linearization
    )
    assert result.loss.regularization == pytest.approx(
        native.value(point), rel=1e-6, abs=1e-10
    )
    assert result.loss.data == pytest.approx(linearization.value, rel=1e-5, abs=1e-10)
    assert result.loss.total == pytest.approx(
        result.loss.data + result.loss.regularization
    )


def test_reference_wavelength_weights_match_the_sdk_resolution(tmp_path, site):
    """Sauce resolves ``lambda*reference_wavelength`` (no 2*pi) like the SDK."""
    from frequensolve.imaging._native_regularization import bind_workflow_regularization

    problem = _problem(tmp_path, site)
    linearization = problem.linearize()
    point = np.linspace(-0.1, 0.2, problem.space.size)
    for kind in ("tikhonov", "tv", "tgv"):
        wavelength = im.Smoothing(
            kind=kind,
            wavelength_fraction=0.25,
            reference_wavelength=0.4,
            tgv_ratio=2.0,
            normalize_amplitude=False,
        )
        if kind == "tgv":
            alpha1, alpha2 = wavelength.resolved_tgv_weights()
            explicit = im.Smoothing(
                kind=kind, alpha1=alpha1, alpha2=alpha2, normalize_amplitude=False
            )
        else:
            explicit = im.Smoothing(
                kind=kind, alpha=wavelength.resolved_alpha(), normalize_amplitude=False
            )
        values = []
        for config in (wavelength, explicit):
            _, native = bind_workflow_regularization(
                config, linearization.space, problem, linearization
            )
            values.append(native.value(point))
        assert values[0] > 0
        assert values[0] == pytest.approx(values[1], rel=1e-6)


def test_multifrequency_zero_data_kernel_stacks_native_task_images(tmp_path, site):
    """Observed-RMS problems can request a zero-data kernel and native frequency mean."""
    from frequensolve.geometry.grids import CartesianGrid

    project = Project(name="kernel", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)
    frequencies = [FREQUENCY - 1, FREQUENCY]
    observed = FrequencyDomainJob("observed", truth, frequencies)
    assert site.run(observed, check=True).successful
    problem = im.ImagingProblem(
        initial,
        controls=im.DepthProfile("vp", "layer_2", count=4),
        observed=im.ObservedData(observed),
        frequencies=frequencies,
        misfit=im.Misfit.huber(),
        site=site,
        name="kernel",
    )
    grid = CartesianGrid(n=[9, 7], x0=[0, 0], x1=[1.0, 0.6])
    images = im.sensitivity_kernel(problem, grid, observed=None)
    assert images.parts == 2
    raw = images.raw["vp"].values
    parts = [images.read_images("raw", part=task)["vp"].values for task in (1, 2)]
    assert np.all(np.isfinite(raw)) and np.linalg.norm(raw) > 0
    np.testing.assert_allclose(raw, (parts[0] + parts[1]) / 2, rtol=1e-4, atol=1e-12)
