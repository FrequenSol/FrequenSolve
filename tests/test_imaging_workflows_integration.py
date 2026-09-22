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
    penalty = im.Quadratic(np.eye(lin.space.size), weight=0.2, reference=reference)
    images = {}
    for method in ("cg", "lsqr"):
        workflow = im.LSRTM(
            problem,
            method=method,
            iterations=25,
            tolerance=1e-4,
            damping=0.1,
            penalty=penalty,
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
            + penalty.bind(lin.space).gradient(image).values
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
