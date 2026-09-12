"""Offline SDK boundary; optional exchange uses actual Cloud-produced responses."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from frequensolve.orchestrator.sites.aws.execution_profile import (
    ManagedExecutionProfile,
)
from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient
from tests.test_awssite_submit_resources import FakeJob, make_graphql_site

FIXTURES = Path(__file__).parent / "fixtures/execution-site"
CASES = json.loads((FIXTURES / "cases.v1.json").read_text())


def profile(value):
    names = {
        "mpiRanks": "mpi_ranks",
        "wallTimeSeconds": "wall_time_seconds",
        "memoryMiB": "memory_mib",
    }
    return ManagedExecutionProfile.from_mapping(
        {
            "execution_site_id": "managed-slurm",
            "execution_resources": {names.get(k, k): v for k, v in value.items()},
        }
    )


@pytest.mark.parametrize("case", CASES["resources"], ids=lambda c: c["name"])
def test_shared_resource_contract(case):
    if not case["valid"]:
        with pytest.raises(ValueError):
            profile(case["value"])
        return
    assert (
        profile(case["value"]).graphql_arguments()["execution_resources"]
        == case["value"]
    )


@pytest.mark.parametrize("case", CASES["states"], ids=lambda c: c["status"])
def test_submission_and_polling_agree(case):
    site = make_graphql_site()
    site.execution_profile = profile(CASES["resources"][1]["value"])
    site.graphql_client.submit_job = Mock(
        return_value={
            "simulationId": "simulation-1",
            "status": "PENDING",
            "executionState": case["state"],
            "providerJobId": "opaque-provider",
        }
    )
    handle = site.submit(FakeJob())
    assert handle.backend["executionState"] == case["state"]
    assert handle.backend["requestedResources"] == CASES["resources"][1]["value"]
    assert handle.backend["providerJobId"] == "opaque-provider"


def test_partial_deployment_retains_supported_identity_fields():
    client = GraphQLClient.__new__(GraphQLClient)
    client.execute = Mock(
        side_effect=[
            RuntimeError('Cannot query field "requestedResources"'),
            RuntimeError('Cannot query field "allocatedResources"'),
            {
                "getSimulation": {
                    "id": "test",
                    "status": "RUNNING",
                    "executionSiteId": "managed-slurm",
                    "logicalAttemptId": "test:7",
                    "providerJobId": "opaque",
                    "executionState": "running",
                }
            },
        ]
    )
    result = client.get_simulation_status_details("test")
    assert result["logicalAttemptId"] == "test:7"
    assert result["providerJobId"] == "opaque"
    assert result["executionSiteId"] == "managed-slurm"
    final_query = client.execute.call_args.args[0]
    assert "requestedResources" not in final_query
    assert "allocatedResources" not in final_query
    assert "logicalAttemptId" in final_query


def test_transport_errors_do_not_trigger_schema_downgrades():
    client = GraphQLClient.__new__(GraphQLClient)
    client.execute = Mock(side_effect=RuntimeError("timeout reading executionSiteId"))
    with pytest.raises(RuntimeError, match="timeout"):
        client.get_simulation_status_details("test")
    assert client.execute.call_count == 1


def test_producer_consumer_exchange():
    requests = []
    for case in CASES["resources"]:
        if not case["valid"]:
            continue
        client = GraphQLClient.__new__(GraphQLClient)
        client.execute = Mock(
            return_value={"submitJob": {"simulationId": "test", "status": "PENDING"}}
        )
        client.submit_job(
            job_file_s3_key="private/test/job.json",
            **profile(case["value"]).graphql_arguments(),
        )
        variables = client.execute.call_args.args[1]
        assert json.loads(variables["executionResources"]) == case["value"]
        requests.append({"name": case["name"], "variables": variables})
    exchange = os.environ.get("EXECUTION_CONTRACT_EXCHANGE")
    if exchange:
        directory = Path(exchange)
        (directory / "sdk-requests.json").write_text(json.dumps(requests))
        responses = directory / "cloud-responses.json"
        if responses.exists():
            for response in json.loads(responses.read_text()):
                client = GraphQLClient.__new__(GraphQLClient)
                client.execute = Mock(
                    return_value={"getSimulation": response["record"]}
                )
                actual = client.get_simulation_status_details(response["record"]["id"])
                for key, value in response["expected"].items():
                    assert actual[key] == value
