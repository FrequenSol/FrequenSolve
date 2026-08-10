"""Tests for GraphQL stack mutation environment handling."""

from typing import Any, Dict, Optional

import pytest

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


class CapabilityGraphQLClient(GraphQLClient):
    def __init__(self, query_fields=(), mutation_fields=()):
        super().__init__("https://example.invalid/graphql", auth=object())
        self.query_fields = query_fields
        self.mutation_fields = mutation_fields

    def execute(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        assert "__type" in query
        assert variables is None
        return {
            "queryType": {"fields": [{"name": name} for name in self.query_fields]},
            "mutationType": {
                "fields": [{"name": name} for name in self.mutation_fields]
            },
        }


class SimulationStatusGraphQLClient(GraphQLClient):
    def __init__(self, *, legacy=False):
        super().__init__("https://example.invalid/graphql", auth=object())
        self.legacy = legacy
        self.queries = []

    def execute(self, query, variables=None):
        self.queries.append(query)
        if self.legacy and "failureMessage" in query:
            raise RuntimeError(
                "GraphQL errors: Cannot query field 'failureMessage' on type 'Simulation'"
            )
        return {
            "getSimulation": {
                "id": variables["id"],
                "status": "FAILED",
                **(
                    {}
                    if self.legacy
                    else {
                        "failureCode": "SCU_BALANCE_INSUFFICIENT",
                        "failureMessage": "This simulation needs more SCUs.",
                    }
                ),
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


def test_compute_provisioning_mode_detects_shared_runtime_catalog():
    client = CapabilityGraphQLClient(
        query_fields=("fetchAvailableComputeRuntimes",),
        mutation_fields=("deployComputeInfrastructure",),
    )

    assert client.get_compute_provisioning_mode() == "shared"


def test_compute_provisioning_mode_detects_legacy_per_user_mutation():
    client = CapabilityGraphQLClient(
        mutation_fields=("deployComputeInfrastructure",),
    )

    assert client.get_compute_provisioning_mode() == "per-user"


def test_compute_provisioning_mode_rejects_unknown_contract():
    client = CapabilityGraphQLClient(query_fields=("listStacks",))

    with pytest.raises(RuntimeError, match="does not expose a supported compute"):
        client.get_compute_provisioning_mode()


def test_simulation_status_details_include_customer_safe_failure_message():
    client = SimulationStatusGraphQLClient()

    assert client.get_simulation_status_details("simulation-1") == {
        "id": "simulation-1",
        "status": "FAILED",
        "failureCode": "SCU_BALANCE_INSUFFICIENT",
        "failureMessage": "This simulation needs more SCUs.",
    }
    assert len(client.queries) == 1


def test_simulation_status_details_fall_back_for_older_cloud_schemas():
    client = SimulationStatusGraphQLClient(legacy=True)

    assert client.get_simulation_status("simulation-legacy") == "FAILED"
    assert len(client.queries) == 2


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


def test_submit_job_can_send_status_email_override_and_fresh_run():
    client = CapturingGraphQLClient()

    result = client.submit_job(
        "project/jobs/job.json",
        send_simulation_status_email=True,
        fresh=True,
    )

    assert result["simulationId"] == "simulation-1"
    assert "sendSimulationStatusEmail: $sendSimulationStatusEmail" in client.last_query
    assert "forceRun: $forceRun" in client.last_query
    assert client.last_variables == {
        "jobFileS3Key": "project/jobs/job.json",
        "sendSimulationStatusEmail": True,
        "forceRun": True,
    }


def test_submit_job_explains_when_cloud_does_not_support_fresh_runs(monkeypatch):
    client = CapturingGraphQLClient()

    def reject_force_run(query, variables=None):
        raise RuntimeError(
            "GraphQL errors: Unknown argument 'forceRun' on field "
            "'Mutation.submitJob'."
        )

    monkeypatch.setattr(client, "execute", reject_force_run)

    with pytest.raises(RuntimeError, match="Submit without force=True"):
        client.submit_job("project/jobs/job.json", fresh=True)
