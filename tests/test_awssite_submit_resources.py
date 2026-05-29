import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")

from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.aws import aws as aws_module
from frequensolve.orchestrator.sites.aws.aws import AWSSite
from frequensolve.project.project import Project
from frequensolve.simulation.jobs import FrequencyDomainJob


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

    def save_for_remote(
        self, site_name, project, *, result_path_relative_to_project=False
    ):
        assert site_name == "AWSSite"
        assert project == "project-a"
        assert result_path_relative_to_project is True
        return "local-job.json", "project-a/jobs/job.json"


def make_graphql_site():
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = FakeGraphQLClient()
    site.prepare_job = lambda job, sync_project=False, validate=True: None
    site.sync_s3 = lambda local, remote: remote
    site._emit = lambda message: None
    site._make_run_handle = (
        lambda job, simulation_id, poll_interval, fetch, check=False: SimpleNamespace(
            job=job,
            simulation_id=simulation_id,
            poll_interval=poll_interval,
            fetch=fetch,
            check=check,
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


def test_graphql_submit_stages_cloud_result_path_relative_to_project(tmp_path):
    project = Project(name="project", path=tmp_path / "ex_1_1")
    sim = project.new_simulation(
        name="simple_acoustic", physics="acoustic", dimension=2
    )
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="time", simulation=sim, f_list=[1.0])
    site = make_graphql_site()
    synced = {}

    def fake_sync(local, remote):
        synced[str(remote)] = Path(local)
        return remote

    site.sync_s3 = fake_sync

    site.submit(job, force_run=True)

    s3_job_key = "ex_1_1/jobs/simple_acoustic/time/time.json"
    assert site.graphql_client.submit_calls[0]["job_file_s3_key"] == s3_job_key
    payload = json.loads(synced[s3_job_key].read_text())
    assert payload["project_path"] == "ex_1_1"
    assert payload["simulation"] == (
        "ex_1_1/simulations/simple_acoustic/simple_acoustic.json"
    )
    assert payload["result_path"] == "jobs/simple_acoustic/time/results"
    assert "ex_1_1/ex_1_1" not in json.dumps(payload)


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
