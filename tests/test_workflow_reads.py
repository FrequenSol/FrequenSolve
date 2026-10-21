"""Observation uncertainty never becomes absence or mutation recovery."""

import json

import anyio
import pytest
from mcp import Client

from frequensolve.workflow import CONTRACT
from frequensolve.workflow.auth import WorkflowError
from frequensolve.workflow.client import WorkflowClient
from frequensolve.workflow.server import build_server
from tests.test_workflow import auth, tokens

READS = (
    "discover",
    "plan_get",
    "plan_list",
    "plan_results",
    "plan_files",
    "credit_history",
)


@pytest.mark.parametrize("operation", READS)
@pytest.mark.parametrize("failure", ("refused", "malformed", "transport"))
def test_unconfirmed_read_is_unknown_safe_and_retries_only_that_read(
    tmp_path, operation, failure
):
    connection = auth(
        tmp_path, lambda *_args, **_kwargs: pytest.fail("No login expected")
    )
    connection.save(tokens(connection.connection))
    calls = []

    def request(_url, payload, **_kwargs):
        calls.append(payload)
        if failure == "transport":
            raise RuntimeError("synthetic-sensitive-diagnostic")
        if failure == "malformed":
            return []
        return {
            "version": CONTRACT,
            "ok": False,
            "error": "synthetic-sensitive-diagnostic",
        }

    client = WorkflowClient(connection, request=request)
    with pytest.raises(WorkflowError) as error:
        client.call(operation, {"plan_id": "fake-plan"})
    assert getattr(error.value, "failure_code", None) == "read_unconfirmed"
    assert error.value.operation == operation
    assert "synthetic-sensitive" not in str(error.value)
    assert calls == [
        {
            "version": CONTRACT,
            "operation": operation,
            "arguments": {"plan_id": "fake-plan"},
        }
    ]


@pytest.mark.parametrize("operation", READS)
def test_successful_read_keeps_the_authoritative_response(tmp_path, operation):
    connection = auth(
        tmp_path, lambda *_args, **_kwargs: pytest.fail("No login expected")
    )
    connection.save(tokens(connection.connection))
    response = {"available": False, "observation": "synthetic-readback"}
    client = WorkflowClient(
        connection,
        request=lambda *_args, **_kwargs: {
            "version": CONTRACT,
            "ok": True,
            "value": response,
        },
    )
    assert client.call(operation, {}) is response


@pytest.mark.parametrize(
    "operation",
    (
        "bind_project",
        "upload_begin",
        "upload_ticket",
        "upload_finalize",
        "plan_create",
        "plan_revise",
        "plan_validate",
        "plan_quote",
        "native_review",
        "native_approve",
        "result_ticket",
    ),
)
def test_preparation_quote_ticket_and_paid_requests_never_use_read_retry_classification(
    tmp_path, operation
):
    connection = auth(
        tmp_path, lambda *_args, **_kwargs: pytest.fail("No login expected")
    )
    connection.save(tokens(connection.connection))
    client = WorkflowClient(
        connection, request=lambda *_args, **_kwargs: {"version": CONTRACT, "ok": False}
    )
    with pytest.raises(WorkflowError) as error:
        client.call(operation, {})
    assert getattr(error.value, "failure_code", None) != "read_unconfirmed"


@pytest.mark.parametrize(
    "tool,arguments,operation",
    [
        (
            "workflow_plan",
            {
                "action": "get",
                "root_id": "pilot",
                "project_key": "fake-project",
                "plan_id": "fake-plan",
            },
            "plan_get",
        ),
        (
            "workflow_results",
            {
                "action": "list",
                "root_id": "pilot",
                "project_key": "fake-project",
                "plan_id": "fake-plan",
                "execution_id": "fake-execution",
            },
            "plan_results",
        ),
    ],
)
def test_actual_mcp_read_failure_has_safe_observation_retry_and_no_mutation(
    tmp_path, tool, arguments, operation
):
    calls = []

    class FailedRead:
        def call(self, name, arguments):
            calls.append(name)
            raise TimeoutError("synthetic-sensitive-diagnostic")

        def journal(self, *_args):
            pytest.fail("A read must not create or recover a mutation intent")

    async def run():
        async with Client(
            build_server(
                FailedRead(), {"pilot": tmp_path.resolve()}, "http://127.0.0.1:5173"
            ),
            cache=None,
        ) as client:
            reply = (
                await client.call_tool(tool, {"request": arguments})
            ).structured_content
            assert not reply["ok"]
            payload = json.loads(reply["payload"])
            assert payload == {
                "failure_code": "read_unconfirmed",
                "operation": operation,
                "read_outcome": "unknown",
                "next_action": "retry_same_read_with_same_saved_ids",
                "paid_request_sent": False,
            }
            assert "synthetic-sensitive" not in json.dumps(reply)
            assert calls == [operation]

    anyio.run(run)
