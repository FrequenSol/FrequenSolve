import pytest

from frequensolve.orchestrator.sites.aws.execution_profile import (
    ManagedExecutionProfile,
    ManagedExecutionProfileError,
)
from frequensolve.orchestrator.sites.execution import ExecutionDetails


def test_named_site_uses_portable_submission_arguments():
    profile = ManagedExecutionProfile.from_mapping(
        {"execution_site_id": "managed-slurm"}
    )
    assert profile.graphql_arguments() == {
        "execution_site_id": "managed-slurm",
        "execution_resources": {"nodes": 1, "mpiRanks": 1, "wallTimeSeconds": 3600},
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"execution_site_id": []},
        {"execution_site_id": "other"},
        {"execution_site_id": "managed-slurm", "execution_resources": []},
        {"execution_site_id": "managed-slurm", "slurm_partition": "cpu-single"},
        {"execution_resources": {"nodes": 1}},
    ],
)
def test_invalid_profiles_have_actionable_errors(value):
    with pytest.raises(ManagedExecutionProfileError):
        ManagedExecutionProfile.from_mapping(value)


def test_legacy_batch_profile_and_neutral_result():
    assert ManagedExecutionProfile.from_mapping({}).graphql_arguments() == {}
    for site in ["managed-batch", "managed-slurm"]:
        details = ExecutionDetails.from_mapping(
            {
                "executionSiteId": site,
                "providerJobId": "opaque-id",
                "executionState": "succeeded",
            }
        )
        assert details.execution_site_id == site
        assert details.provider_job_id == "opaque-id"
        assert details.state == "succeeded"


def test_old_graphql_deployment_retains_slurm_identity_on_site_field_fallback():
    from unittest.mock import Mock

    from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient

    client = GraphQLClient.__new__(GraphQLClient)
    client.execute = Mock(
        side_effect=[
            RuntimeError("Cannot query field executionSiteId"),
            {
                "getSimulation": {
                    "id": "test",
                    "status": "SUCCEEDED",
                    "executionBackend": "SLURM",
                    "providerAttemptId": "123",
                }
            },
        ]
    )
    details = client.get_simulation_status_details("test")
    assert details["executionSiteId"] == "managed-slurm"
    assert details["providerJobId"] == "123"
    assert details["executionState"] == "succeeded"
    assert client.execute.call_count == 2


def test_named_site_does_not_fall_back_to_another_backend():
    from unittest.mock import Mock

    from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient

    client = GraphQLClient.__new__(GraphQLClient)
    client.execute = Mock(side_effect=RuntimeError("Unknown argument executionSiteId"))
    with pytest.raises(RuntimeError, match="does not support named execution sites"):
        client.submit_job(
            job_file_s3_key="private/test/job.json", execution_site_id="managed-slurm"
        )
    assert client.execute.call_count == 1


@pytest.mark.parametrize(
    "status,expected", [("COMPLETED", "succeeded"), ("CANCELLED", "canceled")]
)
def test_normalized_state_covers_supported_status_aliases(status, expected):
    from unittest.mock import Mock

    from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient

    client = GraphQLClient.__new__(GraphQLClient)
    client.execute = Mock(
        return_value={"getSimulation": {"id": "test", "status": status}}
    )
    assert client.get_simulation_status_details("test")["executionState"] == expected
    client.execute.return_value["getSimulation"]["executionState"] = "running"
    assert client.get_simulation_status_details("test")["executionState"] == "running"
