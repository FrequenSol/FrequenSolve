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
        self.simulation = SimpleNamespace(
            name="model",
            project_path=Path("project-a"),
            _project=SimpleNamespace(name="project-a", pretty_name="Project A"),
        )
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
    assert site.graphql_client.submit_calls[0]["project_name"] == "project-a"
    assert site.graphql_client.submit_calls[0]["project_display_name"] == "Project A"
    assert site.graphql_client.submit_calls[0]["simulation_name"] == "model"
    assert site.graphql_client.submit_calls[0]["simulation_job_name"] == "demo-job"
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


def test_graphql_submit_stages_inputs_without_inventing_project_metadata():
    site = make_graphql_site()
    site._ensure_storage_bucket = lambda: None
    sync_calls = []
    site.sync_s3 = lambda local, remote: sync_calls.append((local, remote)) or remote

    job = FakeJob()
    job.simulation._project = None
    job.save_simulation_for_remote = lambda site_name, project: (
        "staged-simulation.json",
        "project-a/simulations/model/model.json",
    )
    job.remote_input_files = lambda project: [
        ("local-input.h5", "project-a/inputs/model.h5")
    ]

    site.submit(job)

    assert site.graphql_client.submit_calls[0]["project_name"] is None
    assert site.graphql_client.submit_calls[0]["project_display_name"] is None

    assert sync_calls == [
        (
            "staged-simulation.json",
            "project-a/simulations/model/model.json",
        ),
        ("local-input.h5", "project-a/inputs/model.h5"),
        ("local-job.json", "project-a/jobs/job.json"),
    ]


def test_graphql_submit_recovers_saved_project_metadata_for_loaded_job(tmp_path):
    project_path = tmp_path / "opaque-cache-directory"
    project_path.mkdir()
    (project_path / "customer-model.json").write_text(
        '{"name":"customer-model","pretty_name":"Customer Model",'
        '"version":"1.0","simulations":[]}'
    )
    site = make_graphql_site()
    site._ensure_storage_bucket = lambda: None
    site.sync_s3 = lambda local, remote: remote

    job = FakeJob()
    job.project_path = project_path
    job.simulation.project_path = project_path
    job.simulation._project = None
    job.save_simulation_for_remote = lambda site_name, project: (
        "staged-simulation.json",
        f"{project}/simulations/model/model.json",
    )
    job.remote_input_files = lambda project: []
    job.save_for_remote = lambda site_name, project: (
        "local-job.json",
        f"{project}/jobs/job.json",
    )

    site.submit(job)

    submitted = site.graphql_client.submit_calls[0]
    assert submitted["project_name"] == "customer-model"
    assert submitted["project_display_name"] == "Customer Model"
    assert submitted["project_name"] != project_path.name


def test_poll_run_preserves_customer_safe_cloud_failure_message():
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = SimpleNamespace(
        get_simulation_status_details=lambda simulation_id: {
            "id": simulation_id,
            "status": "FAILED",
            "failureCode": "SCU_BALANCE_INSUFFICIENT",
            "failureMessage": "This simulation needs more SCUs. No solver work was charged.",
        }
    )

    status = site._poll_run(SimpleNamespace(id="simulation-1"))

    assert status.state == "failed"
    assert status.message == (
        "This simulation needs more SCUs. No solver work was charged."
    )
    assert status.raw["failureCode"] == "SCU_BALANCE_INSUFFICIENT"


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "CANCELED"])
def test_cancel_job_treats_terminal_states_as_idempotent(status):
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = SimpleNamespace(
        get_simulation_status=lambda simulation_id: status
    )

    assert site.cancel_job("private-simulation-id") is None


def test_cancel_job_running_state_has_actionable_sdk_boundary_without_id():
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = SimpleNamespace(
        get_simulation_status=lambda simulation_id: "RUNNING"
    )

    with pytest.raises(NotImplementedError) as exc_info:
        site.cancel_job("private-simulation-id")

    diagnostic = str(exc_info.value)
    assert "Cancel it through the Cloud application" in diagnostic
    assert "private-simulation-id" not in diagnostic


def test_cancel_job_sanitizes_status_lookup_failure():
    site = AWSSite.__new__(AWSSite)

    def fail_status(simulation_id):
        raise RuntimeError(
            "token=private-token account=private-account object=private/key"
        )

    site.graphql_client = SimpleNamespace(get_simulation_status=fail_status)

    with pytest.raises(RuntimeError, match="confirming Cloud connectivity") as exc_info:
        site.cancel_job("private-simulation-id")

    diagnostic = str(exc_info.value)
    for secret in ("private-token", "private-account", "private/key"):
        assert secret not in diagnostic


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


def test_legacy_status_entry_point_uses_graphql_details():
    site = make_graphql_site()
    site.graphql_client.get_simulation_status_details = lambda simulation_id: {
        "id": simulation_id,
        "status": "RUNNING",
    }

    with pytest.deprecated_call(match="get_job_status_from_api"):
        status = site.get_job_status_from_api("simulation-1")

    assert status == {"id": "simulation-1", "status": "RUNNING"}
