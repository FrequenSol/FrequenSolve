"""Tests for GraphQL stack mutation environment handling."""

from typing import Any, Dict, Optional

from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient


class CapturingGraphQLClient(GraphQLClient):
    def __init__(self):
        super().__init__("https://example.invalid/graphql", auth=object())
        self.last_query = ""
        self.last_variables: Optional[Dict[str, Any]] = None

    def execute(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.last_query = query
        self.last_variables = variables
        if "deployStorage" in query:
            return {
                "deployStorage": {
                    "stackId": "stack-1",
                    "stackName": "storage-stack",
                    "status": "CREATE_IN_PROGRESS",
                    "outputs": {},
                    "error": None,
                }
            }
        if "deployComputeInfrastructure" in query:
            return {
                "deployComputeInfrastructure": {
                    "stackId": "stack-2",
                    "stackName": "compute-stack",
                    "status": "CREATE_IN_PROGRESS",
                    "outputs": {},
                    "error": None,
                }
            }
        if "submitJob" in query:
            return {
                "submitJob": {
                    "simulationId": "simulation-1",
                    "batchJobId": "batch-1",
                    "status": "PENDING",
                }
            }
        raise AssertionError(f"unexpected query: {query}")


class AccountAuth:
    def get_account_id(self):
        return "account-123"


class StackQueryCapturingGraphQLClient(GraphQLClient):
    def __init__(self):
        super().__init__("https://example.invalid/graphql", auth=AccountAuth())
        self.last_query = ""
        self.last_variables: Optional[Dict[str, Any]] = None

    def execute(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.last_query = query
        self.last_variables = variables
        return {
            "listStacks": {
                "items": [
                    {
                        "stackId": "compute-stack",
                        "outputs": "{}",
                        "status": "CREATE_COMPLETE",
                        "createdAt": "2026-05-24T00:00:00Z",
                    }
                ]
            }
        }


def test_check_compute_stack_exists_filters_by_account_id():
    client = StackQueryCapturingGraphQLClient()

    assert client._check_compute_stack_exists() is True

    assert "listStacks(filter: $filter)" in client.last_query
    assert client.last_variables == {
        "filter": {
            "stackType": {"eq": "compute"},
            "accountId": {"eq": "account-123"},
            "or": [
                {"status": {"eq": "CREATE_COMPLETE"}},
                {"status": {"eq": "UPDATE_COMPLETE"}},
                {"status": {"eq": "UPDATE_ROLLBACK_COMPLETE"}},
            ],
        }
    }


def test_deploy_storage_stack_does_not_send_environment_argument():
    client = CapturingGraphQLClient()

    client.deploy_storage_stack()

    assert "$environment" not in client.last_query
    assert "environment:" not in client.last_query
    assert client.last_variables in (None, {})


def test_deploy_compute_stack_does_not_send_environment_argument():
    client = CapturingGraphQLClient()

    client.deploy_compute_stack()

    assert "$environment" not in client.last_query
    assert "environment:" not in client.last_query
    assert client.last_variables in (None, {})


def test_legacy_environment_argument_is_accepted_but_not_sent():
    client = CapturingGraphQLClient()

    client.deploy_storage_stack("staging")
    assert "$environment" not in client.last_query
    assert "environment:" not in client.last_query
    assert client.last_variables in (None, {})

    client.deploy_compute_stack("staging")
    assert "$environment" not in client.last_query
    assert "environment:" not in client.last_query
    assert client.last_variables in (None, {})


def test_submit_job_can_send_status_email_override_and_force_run():
    client = CapturingGraphQLClient()

    result = client.submit_job(
        "project/jobs/job.json",
        send_simulation_status_email=True,
        force_run=True,
    )

    assert result["simulationId"] == "simulation-1"
    assert "sendSimulationStatusEmail: $sendSimulationStatusEmail" in client.last_query
    assert "forceRun: $forceRun" in client.last_query
    assert client.last_variables == {
        "jobFileS3Key": "project/jobs/job.json",
        "sendSimulationStatusEmail": True,
        "forceRun": True,
    }
