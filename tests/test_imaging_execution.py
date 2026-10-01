"""Workflow execution scopes retain caller sites and reach every problem view."""

import numpy as np
import pytest

from frequensolve.imaging._backend import Backend
from frequensolve.imaging.workflows import FWI, LBFGS, NewtonCG
from frequensolve.orchestrator.sites.execution import (
    PersistentAllocation,
    execution_scope,
)
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_workflows import _problem, _stages

pytestmark = pytest.mark.unit


class TrackedSession:
    def __init__(self, site):
        self.base_site = site
        self.opened = False
        self.closed = False
        self.jobs = []
        self.options = {}

    def __getattr__(self, name):
        return getattr(self.base_site, name)

    def __enter__(self):
        self.opened = True
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def _require_ready(self):
        assert self.opened and not self.closed

    def submit(self, job, **options):
        self._require_ready()
        self.jobs.append(job)
        return self.base_site.submit(job, **options)


@pytest.mark.parametrize("optimizer", [LBFGS(), NewtonCG()])
def test_fwi_owned_execution_matches_existing_result_and_restores_sites(
    tmp_path, monkeypatch, optimizer
):
    site = FakeImagingSite(seed=11)
    reference_site = FakeImagingSite(seed=11)
    problem = _problem(tmp_path, site)
    reference = _problem(tmp_path, reference_site, subdir="reference")
    expected = FWI(reference, _stages(), optimizer=optimizer).run()
    backend = problem.backend  # Created before entering the execution scope.
    session = TrackedSession(site)

    def factory(**options):
        session.options = options
        return session

    monkeypatch.setattr(site, "session", factory, raising=False)
    observed_sites = []
    result = FWI(
        problem,
        _stages(),
        optimizer=optimizer,
        callback=lambda event: observed_sites.append(problem.site),
    ).run(execution=PersistentAllocation(nodes=1))
    np.testing.assert_allclose(result.state.values, expected.state.values, rtol=1e-8)
    assert session.opened and session.closed
    assert session.options["nodes"] == 1
    assert session.jobs == site.jobs
    assert observed_sites and all(s is session for s in observed_sites)
    assert problem.site is site and backend.site is site
    assert result.problem.site is site


def test_fwi_borrows_open_session_without_closing_it(tmp_path):
    site = FakeImagingSite(seed=11)
    problem = _problem(tmp_path, site)
    session = TrackedSession(site)
    session.opened = True
    FWI(problem, _stages()).run(execution=session)
    assert session.jobs and not session.closed
    assert problem.site is site


def test_fwi_interrupt_closes_owned_session_and_restores_scope(tmp_path, monkeypatch):
    site = FakeImagingSite(seed=11)
    problem = _problem(tmp_path, site)
    session = TrackedSession(site)
    monkeypatch.setattr(site, "session", lambda **kwargs: session, raising=False)

    def interrupt(_event):
        assert problem.site is session
        raise KeyboardInterrupt("stop inversion")

    with pytest.raises(KeyboardInterrupt):
        FWI(problem, _stages(), callback=interrupt).run(
            execution=PersistentAllocation()
        )
    assert session.closed
    assert problem.site is site and problem.backend.site is site


def test_new_control_views_do_not_keep_closed_executor(tmp_path):
    site = FakeImagingSite(seed=11)
    problem = _problem(tmp_path, site)
    session = TrackedSession(site)
    session.opened = True
    with execution_scope(site, session):
        view = problem.with_controls(problem.full_space)
        assert view.site is session
        assert view.backend.site is session
        view.linearize(gradient=False)
    assert view.site is site
    assert view.backend.site is site


def test_backend_scope_routes_preparation_and_job_families(tmp_path):
    site = FakeImagingSite(seed=11)
    first = Backend(site, tmp_path)
    second = Backend(site, tmp_path / "other")
    executor = TrackedSession(site)
    with execution_scope(site, executor):
        assert first.site is executor and second.site is executor
        with execution_scope(site, object()) as nested:
            assert first.site is nested
        assert first.site is executor
    assert first.site is site and second.site is site
