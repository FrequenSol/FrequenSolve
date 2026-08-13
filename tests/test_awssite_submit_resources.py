from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")

from frequensolve.orchestrator.sites.aws.aws import AWSSite


class FakeGraphQLClient:
    def __init__(self):
        self.submit_calls = []
        self.compute_stack_checks = 0
        self.compute_mode = "shared"
        self.compute_stack_exists = True
        self.compute_deployments = 0
        self.compute_waits = []

    def get_compute_provisioning_mode(self):
        return self.compute_mode

    def _check_compute_stack_exists(self):
        self.compute_stack_checks += 1
        return self.compute_stack_exists

    def deploy_compute_stack(self):
        self.compute_deployments += 1
        return {"stackId": "legacy-compute-stack"}

    def wait_for_stack_ready(self, stack_type, expected_stack_id=None):
        self.compute_waits.append((stack_type, expected_stack_id))
        return {"stackId": expected_stack_id, "status": "CREATE_COMPLETE"}

    def submit_job(self, **kwargs):
        self.submit_calls.append(kwargs)
        return {"simulationId": "simulation-1", "status": "PENDING"}


class FakeJob:
    name = "demo-job"

    def __init__(self):
        self.project_path = Path("project-a")
        self.simulation = SimpleNamespace(project_path=Path("project-a"))
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
        lambda job, simulation_id, poll_interval, fetch, check=False: SimpleNamespace(
            job=job,
            simulation_id=simulation_id,
            poll_interval=poll_interval,
            fetch=fetch,
            check=check,
        )
    )
    return site


def test_aws_cli_environment_replaces_credentials_and_removes_profiles(
    monkeypatch,
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "inherited-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "inherited-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "inherited-token")
    monkeypatch.setenv("AWS_PROFILE", "inherited-profile")
    monkeypatch.setenv("HPC_PASSWORD", "hpc-secret")
    credentials = SimpleNamespace(
        get_frozen_credentials=lambda: SimpleNamespace(
            access_key="temporary-access",
            secret_key="temporary-secret",
            token="temporary-token",
        )
    )
    site = AWSSite.__new__(AWSSite)
    site.session = SimpleNamespace(get_credentials=lambda: credentials)
    site.config = SimpleNamespace(region="us-test-1")

    environment = site._aws_cli_env()

    assert environment["AWS_ACCESS_KEY_ID"] == "temporary-access"
    assert environment["AWS_SECRET_ACCESS_KEY"] == "temporary-secret"
    assert environment["AWS_SESSION_TOKEN"] == "temporary-token"
    assert environment["AWS_DEFAULT_REGION"] == "us-test-1"
    assert "AWS_PROFILE" not in environment
    assert "HPC_PASSWORD" not in environment


def test_graphql_submit_preserves_backend_resource_defaults_when_omitted():
    site = make_graphql_site()
    job = FakeJob()

    site.submit(job)

    assert site.graphql_client.submit_calls[0]["vcpu"] is None
    assert site.graphql_client.submit_calls[0]["memory"] is None
    assert site.graphql_client.compute_stack_checks == 0


def test_graphql_submit_current_job_fetches_once_when_waited():
    site = make_graphql_site()
    site._emit_status = lambda *args, **kwargs: None
    fetch_calls = []
    site.fetch_outputs = lambda job: fetch_calls.append(job)
    job = FakeJob()
    job.is_run_current = lambda: True
    job.write_run_state = lambda **kwargs: None

    run = site.submit(job, fetch=True)

    assert fetch_calls == []
    result = run.wait()
    assert result.successful
    assert fetch_calls == [job]
    assert run.wait() is result
    assert fetch_calls == [job]
    assert site.graphql_client.submit_calls == []


def test_graphql_submit_checks_legacy_per_user_compute_stack():
    site = make_graphql_site()
    site.graphql_client.compute_mode = "per-user"

    site.submit(FakeJob())

    assert site.graphql_client.compute_stack_checks == 1


def test_graphql_submit_provisions_missing_legacy_per_user_compute_stack():
    site = make_graphql_site()
    site.graphql_client.compute_mode = "per-user"
    site.graphql_client.compute_stack_exists = False

    site.submit(FakeJob())

    assert site.graphql_client.compute_deployments == 1
    assert site.graphql_client.compute_waits == [("compute", "legacy-compute-stack")]
    assert len(site.graphql_client.submit_calls) == 1


def test_graphql_submit_fails_closed_when_compute_capability_is_unknown():
    site = make_graphql_site()

    def fail_capability_probe():
        raise RuntimeError("introspection unavailable")

    site.graphql_client.get_compute_provisioning_mode = fail_capability_probe

    with pytest.raises(RuntimeError, match="Failed to determine whether"):
        site.submit(FakeJob())

    assert site.graphql_client.submit_calls == []
    assert site.graphql_client.compute_stack_checks == 0


def test_graphql_submit_sends_explicit_resource_overrides():
    site = make_graphql_site()
    job = FakeJob()

    site.submit(job, vcpu=8, memory=16384)

    assert site.graphql_client.submit_calls[0]["vcpu"] == 8
    assert site.graphql_client.submit_calls[0]["memory"] == 16384


@pytest.mark.parametrize("skip", [False, "false"])
def test_graphql_submit_skip_false_requests_fresh_run(skip):
    site = make_graphql_site()
    job = FakeJob()

    site.submit(job, skip=skip)

    assert site.graphql_client.submit_calls[0]["fresh"] is True


def test_submit_requires_the_current_authenticated_graphql_contract():
    site = make_graphql_site()
    site.graphql_client = None
    job = FakeJob()

    with pytest.raises(RuntimeError, match="requires Cognito authentication"):
        site.submit(job)


def test_connectivity_uses_the_same_graphql_capability_probe_as_submission():
    site = make_graphql_site()

    assert site.test_api_connectivity() is True


def test_connectivity_returns_false_without_graphql_authentication():
    site = make_graphql_site()
    site.graphql_client = None

    assert site.test_api_connectivity() is False


def test_connectivity_returns_false_when_capability_probe_fails():
    site = make_graphql_site()

    def fail_capability_probe():
        raise RuntimeError("request timed out")

    site.graphql_client.get_compute_provisioning_mode = fail_capability_probe

    assert site.test_api_connectivity() is False
