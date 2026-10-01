"""Solver-backed check that one ``normal`` job equals one job per direction.

Uses the two-frequency acoustic fixture of ``test_imaging_integration`` (same
executable lookup: ``FS_SAUCE_EXECUTABLE`` or ``LOCAL_SOLVER_EXECUTABLE``).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging import DepthProfile, ImagingProblem, ObservedData
from frequensolve.imaging._native_regularization import bind_workflow_regularization
from frequensolve.project import Project
from frequensolve.simulation import FrequencyDomainJob
from tests.test_imaging_integration import (
    FREQUENCY,
    START_VP,
    TRUTH_VP,
    _executable,
    _simulation,
)

pytestmark = pytest.mark.integration


def _spy(problem):
    jobs = []
    run = problem._run_job

    def record(job):
        jobs.append(job)
        run(job)

    problem._run_job = record
    return jobs


@pytest.mark.parametrize("background", [False, True])
def test_batched_normal_matches_single_direction_jobs(tmp_path, background):
    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="batch", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)
    frequencies = [FREQUENCY - 1.0, FREQUENCY]
    observed_job = FrequencyDomainJob("observed", truth, frequencies)
    assert site.run(observed_job, check=True).successful
    problem = ImagingProblem(
        initial,
        controls=DepthProfile("vp", "layer_2", count=4),
        observed=ObservedData(observed_job),
        frequencies=frequencies,
        site=site,
        name="batch",
    ).restrict(weights=[0.5, 2.0])
    lin = problem.linearize(background=background)
    assert (lin.job.background is not None) == background
    single = copy.copy(lin)  # an independent memo: one job per direction
    single._ops, single._normal = {}, None
    jobs = _spy(problem)
    directions = [lin.space.random(seed) for seed in (1, 2, 3)]

    batch = lin.normal @ np.column_stack([d.values for d in directions])

    (job,) = jobs
    assert job.action == "normal" and len(job.directions) == 3 and job.n_tasks == 2
    assert (job.background is not None) == background
    for j, direction in enumerate(directions):
        reference = single.apply_normal(direction).values
        error = np.linalg.norm(batch[:, j] - reference) / np.linalg.norm(reference)
        assert error < 1.0e-4, (j, error)
    assert len(jobs) == 1 + len(directions)

    # A bound native Tikhonov Hessian rides in the same job's postprocess.
    _, native = bind_workflow_regularization(
        2.0 * im.Tikhonov(0.3), lin.space, problem, lin
    )
    fresh = [lin.space.random(seed) for seed in (4, 5)]
    before = len(jobs)
    regularized = lin.apply_normal_batch(fresh, regularization=native)
    (folded,) = jobs[before:]
    assert folded.regularization is not None and len(folded.directions) == 2
    operator = native.hessian_operator()
    for product, direction in zip(regularized, fresh):
        expected = lin.apply_normal(direction).values + np.asarray(operator @ direction)
        error = np.linalg.norm(product.values - expected) / np.linalg.norm(expected)
        assert error < 1.0e-6, error
