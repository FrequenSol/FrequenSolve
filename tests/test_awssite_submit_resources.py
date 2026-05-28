from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")

from frequensolve.orchestrator.sites.aws import aws as aws_module
from frequensolve.orchestrator.sites.aws.aws import AWSSite


class FakeGraphQLClient:
    def __init__(self):
        self.submit_calls = []

    def _check_compute_stack_exists(self):
        return True

    def submit_job(self, **kwargs):
        self.submit_calls.append(kwargs)
        return {"simulationId": "simulation-1", "status": "PENDING"}


class FakeJob:
    name = "demo-job"

    def __init__(self):
        self.project_path = Path("project-a")
        self.simulation = SimpleNamespace(_remote_path=Path("project-a/simulation-a"))
        self._job_id = None

    def is_run_current(self):
        return False

    def save_for_remote(self, site_name, project):
        assert site_name == "AWSSite"
        assert project == "project-a"
        return "local-job.json", "project-a/jobs/job.json"


def make_graphql_site():
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = FakeGraphQLClient()
    site.prepare_job = lambda job, sync_project=False, validate=True: None
    site.sync_s3 = lambda local, remote: remote
    site._emit = lambda message: None
    site._make_run_handle = (
        lambda job, simulation_id, poll_interval, fetch: SimpleNamespace(
            job=job,
            simulation_id=simulation_id,
            poll_interval=poll_interval,
            fetch=fetch,
        )
    )
    return site


def make_rest_site():
    site = make_graphql_site()
    site.graphql_client = None
    site.config = SimpleNamespace(
        api_token=None,
        api_base_url="https://api.example.invalid",
    )
    return site


def test_graphql_submit_preserves_backend_resource_defaults_when_omitted():
    site = make_graphql_site()
    job = FakeJob()

    site.submit(job)

    assert site.graphql_client.submit_calls[0]["vcpu"] is None
    assert site.graphql_client.submit_calls[0]["memory"] is None


def test_graphql_submit_sends_explicit_resource_overrides():
    site = make_graphql_site()
    job = FakeJob()

    site.submit(job, vcpu=8, memory=16384)

    assert site.graphql_client.submit_calls[0]["vcpu"] == 8
    assert site.graphql_client.submit_calls[0]["memory"] == 16384


def test_rest_submit_preserves_backend_resource_defaults_when_omitted(monkeypatch):
    site = make_rest_site()
    job = FakeJob()
    requests = []

    def fake_post(url, json, headers, timeout):
        requests.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "success", "simulation_id": "simulation-1"},
        )

    monkeypatch.setattr(aws_module.requests, "post", fake_post)

    site.submit(job)

    assert "vcpu" not in requests[0]["json"]
    assert "memory" not in requests[0]["json"]


def test_rest_submit_sends_only_explicit_resource_overrides(monkeypatch):
    site = make_rest_site()
    job = FakeJob()
    requests = []

    def fake_post(url, json, headers, timeout):
        requests.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "success", "simulation_id": "simulation-1"},
        )

    monkeypatch.setattr(aws_module.requests, "post", fake_post)

    site.submit(job, vcpu=8)

    assert requests[0]["json"]["vcpu"] == 8
    assert "memory" not in requests[0]["json"]
