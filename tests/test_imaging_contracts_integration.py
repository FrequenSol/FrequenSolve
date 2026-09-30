"""Public multi-frequency imaging contracts exercised against a real solver."""

import hashlib
import json

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.geometry.grids import CartesianGrid
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.simulation import FrequencyDomainJob
from frequensolve.simulation.artifact_catalog import load_artifact_catalog
from frequensolve.simulation.outputs import TraceOutput
from tests.test_imaging_integration import START_VP, TRUTH_VP, _executable, _simulation

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]


def exercise_imaging_contracts(site, root):
    project = Project(name="imaging_contracts", path=root, load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)
    observed = FrequencyDomainJob("observed", truth, [4.0, 6.0, 8.0])
    assert site.run(observed, check=True, fetch=True).successful
    traces = site.fetch_traces(observed)
    assert traces is not None
    _assert_stable_trace_layout(observed)
    observed.save()
    assert observed.traces() is not None
    problem = im.ImagingProblem(
        initial,
        controls=im.DepthProfile("vp", "layer_2", count=4),
        observed=im.ObservedData(observed),
        site=site,
        name="contracts",
        misfit=im.Misfit(normalization="observed_rms"),
    )
    lin = problem.linearize()
    assert np.isfinite(lin.value) and lin.gradient.norm() > 0
    # These exports must be discoverable by remote fetch, not just exist on disk.
    catalog = load_artifact_catalog(lin.job._result_path, tasks=(1, 2, 3))
    schemas = {artifact.schema for artifact in catalog.query()}
    assert {"fs-control-state-1", "fs-control-registry-1"} <= schemas
    assert lin.jacobian.dot_test(seed=3, tolerance=3e-3)["passed"]
    direction = lin.space.ones() * 0.001
    np.testing.assert_allclose(
        (lin.normal @ direction).values,
        lin.vjp(lin.jvp(direction)).values,
        rtol=3e-3,
        atol=1e-7,
    )
    assert im.rtm(problem).norm() > 0
    grid = CartesianGrid(n=[21, 13], x0=[0.0, 0.0], x1=[1.0, 0.6])
    for use_observed in (None, True):
        images = im.sensitivity_kernel(problem, grid, observed=use_observed)
        raw = images.raw.vp.values
        assert np.all(np.isfinite(raw)) and np.linalg.norm(raw) > 0
        # The stacked image is the frequency mean of the parts in output units.
        stacked = np.mean(
            [images.read_images("raw", part=task).vp.values for task in (1, 2, 3)],
            axis=0,
        )
        np.testing.assert_allclose(
            raw, stacked, rtol=1e-4, atol=1e-6 * np.abs(raw).max()
        )
    result = im.FWI(
        problem, im.Stage([4.0, 6.0, 8.0], iterations=1), step_limit=0.05
    ).run()
    assert result.stages[-1].final_loss.total < result.stages[0].initial_loss.total
    return problem


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_stable_trace_layout(job):
    trace_dir = job._result_path / "traces"
    assert (trace_dir / "traces.h5").is_file()
    assert (trace_dir / "manifest.json").is_file()
    assert not (trace_dir / "generations").exists()
    assert not (trace_dir / "shards").exists()
    assert job.trace_manifest.packed_files == [trace_dir / "traces.h5"]


def exercise_trace_history(site, root):
    project = Project(name="trace_history", path=root, load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    job = FrequencyDomainJob(
        "history", truth, [4.0, 6.0], outputs=TraceOutput(keep_history=True)
    )
    assert site.run(job, check=True, fetch=True).successful
    _assert_stable_trace_layout(job)
    first = json.loads((job._result_path / "traces" / "manifest.json").read_text())
    first_digest = _sha256(job._result_path / "traces" / "traces.h5")
    assert site.run(job, check=True, fetch=True, force=True).successful
    _assert_stable_trace_layout(job)
    archived = job._result_path / "traces" / "history" / first["generation"]
    assert (archived / "traces.h5").is_file()
    history = json.loads((archived / "manifest.json").read_text())
    assert [s["path"] for s in history["segments"]] == [
        f"traces/history/{first['generation']}/traces.h5"
    ]
    assert history["entries"] == first["entries"]
    # The archive is the first run's exact product, not a copy of the new one.
    assert _sha256(archived / "traces.h5") == first_digest
    assert _sha256(job._result_path / "traces" / "traces.h5") != first_digest


def test_multifrequency_imaging_contracts(tmp_path):
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        exercise_imaging_contracts(site, tmp_path / "project")


def test_trace_history_is_kept_on_request(tmp_path):
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        exercise_trace_history(site, tmp_path / "project")
