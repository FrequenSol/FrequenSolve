from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")

from frequensolve.orchestrator.sites.aws.aws import AWSSite
from frequensolve.orchestrator.sites.aws.execution_profile import (
    ManagedExecutionProfile,
)


class FakeGraphQLClient:
    def __init__(self):
        self.submit_calls = []

    def execute(self, query, variables=None):
        assert query == "query CloudConnectivity { __typename }"
        assert variables is None
        return {"__typename": "Query"}

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
        self.outputs = SimpleNamespace()
        self.f_list = [10.0]

    @property
    def _result_path(self):
        return self.project_path / "jobs" / self.name / "results"

    def is_run_current(self):
        return False

    def save_for_remote(self, site_name, project):
        assert site_name == "AWSSite"
        assert project == "project-a"
        return "local-job.json", "project-a/jobs/job.json"


def make_graphql_site():
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = FakeGraphQLClient()
    site.execution_profile = ManagedExecutionProfile.from_mapping({})
    site.prepare_job = lambda job, sync_project=False, validate=True: None
    site.sync_s3 = lambda local, remote: remote
    site._emit = lambda message: None
    site._make_run_handle = (
        lambda job, simulation_id, poll_interval, fetch, check=False, backend=None: (
            SimpleNamespace(
                job=job,
                simulation_id=simulation_id,
                poll_interval=poll_interval,
                fetch=fetch,
                check=check,
                backend=backend,
            )
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


def test_graphql_submit_defaults_to_the_managed_site():
    site = make_graphql_site()
    site.submit(FakeJob())
    submitted = site.graphql_client.submit_calls[0]
    assert submitted["execution_site_id"] == "managed-slurm"
    assert submitted["execution_resources"] == {
        "nodes": 1,
        "mpiRanks": 1,
        "wallTimeSeconds": 3600,
    }
    assert "vcpu" not in submitted and "memory" not in submitted
    assert submitted["project_name"] == "project-a"
    assert submitted["project_display_name"] == "Project A"
    assert submitted["simulation_name"] == "model"
    assert submitted["simulation_job_name"] == "demo-job"


def test_graphql_submit_uses_only_the_named_managed_slurm_shape():
    site = make_graphql_site()
    site.execution_profile = ManagedExecutionProfile.from_mapping(
        {
            "execution_site_id": "managed-slurm",
            "execution_resources": {
                "nodes": 2,
                "mpi_ranks": 8,
                "wall_time_seconds": 1800,
            },
        }
    )
    run = site.submit(FakeJob())
    submitted = site.graphql_client.submit_calls[0]
    assert submitted["execution_resources"] == {
        "nodes": 2,
        "mpiRanks": 8,
        "wallTimeSeconds": 1800,
    }
    assert submitted["execution_site_id"] == "managed-slurm"
    assert run.backend["executionSiteId"] == "managed-slurm"
    assert run.backend["requestedResources"] == submitted["execution_resources"]


def test_graphql_submit_supports_right_sized_single_node_profile():
    site = make_graphql_site()
    site.execution_profile = ManagedExecutionProfile.from_mapping(
        {
            "execution_resources": {
                "nodes": 1,
                "mpi_ranks": 1,
                "wall_time_seconds": 1800,
                "cpu": 8,
                "memory_mib": 16384,
            }
        }
    )
    site.submit(FakeJob())
    assert site.graphql_client.submit_calls[0]["execution_resources"] == {
        "nodes": 1,
        "mpiRanks": 1,
        "wallTimeSeconds": 1800,
        "cpu": 8,
        "memoryMiB": 16384,
    }


def test_submit_rejects_managed_execution_overrides():
    site = make_graphql_site()

    with pytest.raises(ValueError, match="named site.toml profile"):
        site.submit(FakeJob(), nodes=2)


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


@pytest.mark.parametrize(
    "billing_status", ["CAPTURE_PENDING", "RECONCILIATION_REQUIRED", "CAPTURED"]
)
def test_poll_run_keeps_billing_separate_from_success(billing_status):
    site = AWSSite.__new__(AWSSite)
    billing = {
        "creditSettlementMode": "END_OF_RUN_CAPTURE_V1",
        "creditSettlementStatus": billing_status,
        "creditSettlementOperationId": "synthetic-operation",
        "creditSettlementAmount": "1.23",
    }
    site.graphql_client = SimpleNamespace(
        get_simulation_status_details=lambda _: {"status": "SUCCEEDED", **billing}
    )
    run = SimpleNamespace(id="simulation-1", backend={})
    assert site._poll_run(run).state == "completed"
    assert run.backend == billing


@pytest.mark.parametrize(
    "status",
    [
        "SUCCEEDED",
        "COMPLETED",
        "FAILED",
        "CANCELED",
        "CANCELLED",
        "completed",
        "cancelled",
    ],
)
def test_cancel_job_treats_terminal_states_as_idempotent(status):
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = SimpleNamespace(
        get_simulation_status=lambda simulation_id: status
    )

    assert site.cancel_job("private-simulation-id") is None


@pytest.mark.parametrize("status", ["PENDING", "RUNNING"])
def test_cancel_job_requests_cloud_cancellation(status):
    calls = []
    site = AWSSite.__new__(AWSSite)
    site.graphql_client = SimpleNamespace(
        get_simulation_status=lambda simulation_id: status,
        cancel_simulation=calls.append,
    )

    assert site.cancel_job("simulation-1") is None
    assert calls == ["simulation-1"]


def test_cancel_job_sanitizes_status_lookup_failure():
    site = AWSSite.__new__(AWSSite)

    def fail_status(simulation_id):
        raise RuntimeError(
            "token=private-token account=private-account object=private/key"
        )

    site.graphql_client = SimpleNamespace(get_simulation_status=fail_status)

    with pytest.raises(
        RuntimeError, match="Cloud access and configuration"
    ) as exc_info:
        site.cancel_job("private-simulation-id")

    diagnostic = str(exc_info.value)
    for secret in ("private-token", "private-account", "private/key"):
        assert secret not in diagnostic


def test_graphql_submit_never_checks_or_provisions_compute():
    site = make_graphql_site()

    def unexpected(*args, **kwargs):
        pytest.fail("Submission must not inspect or provision compute")

    site.graphql_client.get_compute_provisioning_mode = unexpected
    site.graphql_client._check_compute_stack_exists = unexpected
    site.graphql_client.deploy_compute_stack = unexpected
    site.submit(FakeJob())
    assert len(site.graphql_client.submit_calls) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"vcpu": 8},
        {"memory": 16384},
        {"execution_site_id": "managed-batch"},
        {"execution_backend": "batch"},
    ],
)
def test_graphql_submit_rejects_resource_overrides(options):
    site = make_graphql_site()
    with pytest.raises(ValueError, match="named site.toml profile"):
        site.submit(FakeJob(), **options)
    assert site.graphql_client.submit_calls == []


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


def test_connectivity_queries_authenticated_api():
    site = make_graphql_site()

    assert site.test_api_connectivity() is True


def test_connectivity_returns_false_without_graphql_authentication():
    site = make_graphql_site()
    site.graphql_client = None

    assert site.test_api_connectivity() is False


def test_connectivity_returns_false_when_api_probe_fails():
    site = make_graphql_site()

    def fail_api_probe(*args):
        raise RuntimeError("request timed out")

    site.graphql_client.execute = fail_api_probe

    assert site.test_api_connectivity() is False


@pytest.mark.parametrize("initial_status", ["PENDING", "RUNNING"])
@pytest.mark.parametrize(
    "final_status",
    ["SUCCEEDED", "completed", "FAILED", "CANCELED", "cancelled", "RUNNING", None],
)
def test_cancel_job_reconciles_race_once(initial_status, final_status):
    from frequensolve.orchestrator.sites.aws.graphql_client import CloudAPIError

    calls = []
    statuses = iter([initial_status, final_status])
    rejection = CloudAPIError("Cloud rejected cancellation")

    def status(simulation_id):
        calls.append(("status", simulation_id))
        value = next(statuses)
        if value is None:
            raise RuntimeError("private-status-error")
        return value

    def cancel(simulation_id):
        calls.append(("cancel", simulation_id))
        raise rejection

    site = AWSSite.__new__(AWSSite)
    site.graphql_client = SimpleNamespace(
        get_simulation_status=status, cancel_simulation=cancel
    )
    if final_status in {"RUNNING", None}:
        with pytest.raises(CloudAPIError) as error:
            site.cancel_job("simulation-1")
        assert error.value is rejection
    else:
        assert site.cancel_job("simulation-1") is None
    assert calls == [
        ("status", "simulation-1"),
        ("cancel", "simulation-1"),
        ("status", "simulation-1"),
    ]


@pytest.mark.parametrize("location", ["project-images", "outside-project"])
def test_invalid_imaging_path_is_rejected_before_remote_admission(tmp_path, location):
    from frequensolve.simulation.jobs import ImagingJob

    site = make_graphql_site()
    job = object.__new__(ImagingJob)
    job.__dict__.update(FakeJob().__dict__)
    job.name = "demo-job"
    job.simulation.project_path = tmp_path / "project-a"
    job.save_path = (
        job.project_path / "images"
        if location == "project-images"
        else tmp_path / "elsewhere"
    )
    job.is_run_current = lambda: False
    job.save_for_remote = lambda *_: ("local-job.json", "project-a/jobs/job.json")

    with pytest.raises(RuntimeError, match="imaging output path.*inside"):
        site.submit(job, validate=False)
    assert site.graphql_client.submit_calls == []
    assert job._job_id is None


def test_submission_freezes_result_state_before_the_remote_call():
    site = make_graphql_site()
    job = FakeJob()
    job.simulation.extra = {"coordinates": [1.0, 2.0]}
    submit = site.graphql_client.submit_job

    def mutate_after_admission(**kwargs):
        job.simulation.extra["coordinates"][0] = 99.0
        return submit(**kwargs)

    site.graphql_client.submit_job = mutate_after_admission
    run = site.submit(job)
    assert run.job.simulation.extra == {"coordinates": [1.0, 2.0]}
    assert job.simulation.extra == {"coordinates": [99.0, 2.0]}
    assert run.job.simulation._project is job.simulation._project


def test_cpu_sharing_is_per_submission_and_does_not_mutate_site():
    site = make_graphql_site()
    enabled = site.submit(FakeJob(), allow_cpu_sharing=True)
    site.submit(FakeJob())
    site.submit(FakeJob(), allow_cpu_sharing=False)
    calls = site.graphql_client.submit_calls
    assert calls[0]["execution_resources"]["allowCpuSharing"] is True
    assert enabled.backend["requestedResources"]["allowCpuSharing"] is True
    assert all(
        "allowCpuSharing" not in call["execution_resources"] for call in calls[1:]
    )
    assert (
        "allowCpuSharing"
        not in site.execution_profile.graphql_arguments()["execution_resources"]
    )


@pytest.mark.parametrize("value", [1, 2, "true", None])
def test_cpu_sharing_rejects_nonboolean_before_submission(value):
    site = make_graphql_site()
    with pytest.raises(ValueError, match="boolean"):
        site.submit(FakeJob(), allow_cpu_sharing=value)
    assert not site.graphql_client.submit_calls


def test_cpu_sharing_rejects_distributed_profile():
    site = make_graphql_site()
    site.execution_profile = ManagedExecutionProfile.from_mapping(
        {"execution_resources": {"nodes": 2, "mpi_ranks": 2, "wall_time_seconds": 600}}
    )
    with pytest.raises(ValueError, match="single-node"):
        site.submit(FakeJob(), allow_cpu_sharing=True)
    assert not site.graphql_client.submit_calls
