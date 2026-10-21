"""Explicit native paid authority and retry identity, with no live paid calls."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import anyio
import pytest
from mcp import Client
from mcp.types import ElicitResult

from frequensolve.workflow.approval import (
    VERSION,
    PaidApproval,
    PaidApprovalInputError,
    confirm_paid,
    envelope,
)
from frequensolve.workflow.auth import Connection, WorkflowAuth, WorkflowError
from frequensolve.workflow.client import WorkflowClient, create_project
from frequensolve.workflow.journal import IntentJournal
from frequensolve.workflow.server import APPROVAL_TOOL, build_server
from frequensolve.workflow.storage import PrivateStore


def review(stage="planning", ids=None):
    ids = ids or {
        "project_key": "fake-project",
        "project_id": "fake-project-id",
        "plan_id": "fake-plan",
    }
    snapshot = {
        "version": VERSION,
        **ids,
        "stage": stage,
        "revision": 2,
        "input_digest": "sha256:" + "1" * 64,
        "scientific_inputs": {"physics": "acoustic", "tolerance": 1e-6},
        "compute_review": {
            "execution_site_id": "fake-cluster",
            "planning_evaluations": [{"nodes": 1}],
        },
        "plan": {"rows": [{"frequency": "2"}]} if stage == "execution" else None,
        "quote": {
            "purpose": stage,
            "maximum_credits": "10.25",
            "fingerprint": "sha256:" + "2" * 64,
        },
    }
    return envelope(
        {"proposal": snapshot, "proposal_fingerprint": "sha256:" + "3" * 64}, ids, stage
    )


def client_response(response, schema):
    """Supply only the current field in this synthetic multi-prompt client."""
    content = response.get("content")
    if response.get("action") != "accept" or not isinstance(content, dict):
        return response
    if "choice" not in content or set(content) - {"choice", "maximum_credits", "note"}:
        return response
    return {
        "action": "accept",
        "content": {
            key: content[key] for key in schema["properties"] if key in content
        },
    }


def approval(path, user="fake-user"):
    return PaidApproval(IntentJournal(PrivateStore(path), {"subject": user}))


def accepted(arguments):
    return {
        "head": {"plan_id": arguments["plan_id"], "status": "planning"},
        "native_approval": {
            "accepted": True,
            "evidence_id": arguments["evidence_id"],
            "proposal_fingerprint": arguments["proposal_fingerprint"],
            "human_approval_verified": False,
            "evidence_origin": "native_client_reported",
        },
    }


def test_recover_absent_intent_is_explicit_and_never_sends(tmp_path):
    def never_send(*_):
        raise AssertionError("Absent evidence must never authorize paid work")

    result = approval(tmp_path).recover("first-use", never_send)
    assert result["state"] == "absent"
    assert not result["backend_authorized"]
    assert not result["human_approval_verified"]
    assert approval(tmp_path).lookup("first-use") is None
    assert not list(tmp_path.glob("*.json"))


def test_recovery_does_not_treat_foreign_or_unreadable_evidence_as_absent(tmp_path):
    saved, _ = approval(tmp_path).begin("existing", review())

    def never_send(*_):
        raise AssertionError("Invalid evidence must never authorize paid work")

    with pytest.raises(ValueError, match="unavailable for this connection"):
        approval(tmp_path, "foreign-user").recover("existing", never_send)
    record = tmp_path / (approval(tmp_path).key("existing") + ".json")
    record.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        approval(tmp_path).recover("existing", never_send)
    assert saved["state"] == "awaiting_input"


@pytest.mark.parametrize("stage", ["planning", "validation", "execution"])
def test_confirmed_exact_stage_uses_one_saved_request_then_recovers_without_form(
    tmp_path, stage
):
    current = review(stage)
    prompts, calls = [], []

    async def fetch():
        return current

    async def elicit(message, schema):
        prompts.append((message, schema))
        assert "PAID " + stage.upper() in message and "10.25" in message
        if "choice" in schema["properties"]:
            assert set(schema["properties"]) == {"choice"}
            assert "default" not in schema["properties"]["choice"]
        else:
            assert set(schema["properties"]) == {"maximum_credits"}
            assert schema["required"] == ["maximum_credits"]
            assert "10.25" in schema["properties"]["maximum_credits"]["description"]
            assert (
                "do not repeat approve_paid_stage"
                in schema["properties"]["maximum_credits"]["description"]
            )
        # Present the complete proposal once, then a short bound limit prompt.
        if "choice" in schema["properties"]:
            assert schema["properties"]["choice"]["enum"] == [
                "Cancel",
                "Request changes",
                "Approve this stage",
            ]
            assert "Exact proposal details" in message
            assert "acoustic" in message and "1e-06" in message
            assert current["proposal"]["input_digest"] in message
            assert current["proposal"]["quote"]["fingerprint"] in message
        else:
            assert current["proposal_fingerprint"] in message
            assert len(message) < 1000
        return client_response(
            {
                "action": "accept",
                "content": {
                    "choice": "Approve this stage",
                    "maximum_credits": "10.25",
                    "note": "No note",
                },
            },
            schema,
        )

    def send(operation, arguments):
        assert operation == "native_approve"
        # Request intent and exact bytes are already durable before this call.
        pending = json.loads(
            (
                tmp_path
                / ("intent-" + hashlib.sha256(b"native-paid:one").hexdigest() + ".json")
            ).read_text()
        )
        assert pending["idempotency_key"] == arguments["idempotency_key"]
        assert pending["status"] == "pending"
        calls.append(deepcopy(arguments))
        return accepted(arguments)

    result = anyio.run(
        confirm_paid,
        approval(tmp_path),
        "one",
        current,
        current["proposal_fingerprint"],
        fetch,
        elicit,
        send,
    )
    assert result["state"] == "backend_accepted" and result["backend_authorized"]
    assert result["note"] == ""
    assert not result["human_approval_verified"]
    assert result["evidence_origin"] == "native_client_reported"
    assert approval(tmp_path).recover("one", send) == result
    assert len(calls) == 1 and len(prompts) == 2
    assert calls[0]["stage"] == stage and calls[0]["expected_revision"] == 2
    with pytest.raises(ValueError):
        approval(tmp_path, "foreign-user").read("one")


@pytest.mark.parametrize("maximum", ["approve_paid_stage", "", None])
def test_wrong_credit_field_reports_safe_actionable_failure(tmp_path, maximum):
    current = review()

    async def fetch():
        return current

    async def elicit(message, schema):
        return client_response(
            {
                "action": "accept",
                "content": {"choice": "approve_paid_stage", "maximum_credits": maximum},
            },
            schema,
        )

    def never_send(*_):
        raise AssertionError("An incorrect Credit maximum cannot authorize work")

    with pytest.raises(PaidApprovalInputError, match="enter exactly 10.25"):
        anyio.run(
            confirm_paid,
            approval(tmp_path),
            "wrong-field",
            current,
            current["proposal_fingerprint"],
            fetch,
            elicit,
            never_send,
        )
    result = approval(tmp_path).recover("wrong-field", never_send)
    assert result["state"] == "incomplete"
    assert result["failure_code"] == "credit_limit_not_confirmed"
    assert not result["backend_authorized"]


@pytest.mark.parametrize("action", ["cancel", "decline", "timeout"])
def test_decision_alone_never_authorizes_when_credit_prompt_is_interrupted(
    tmp_path, action
):
    current, fields = review(), []

    async def fetch():
        return current

    async def elicit(message, schema):
        fields.append(list(schema["properties"]))
        if "choice" in schema["properties"]:
            return {"action": "accept", "content": {"choice": "approve_paid_stage"}}
        if action == "timeout":
            raise TimeoutError()
        return {"action": action, "content": None}

    def never_send(*_):
        raise AssertionError("A stage decision alone cannot authorize paid work")

    if action == "timeout":
        with pytest.raises(TimeoutError):
            anyio.run(
                confirm_paid,
                approval(tmp_path),
                "second",
                current,
                current["proposal_fingerprint"],
                fetch,
                elicit,
                never_send,
            )
    else:
        anyio.run(
            confirm_paid,
            approval(tmp_path),
            "second",
            current,
            current["proposal_fingerprint"],
            fetch,
            elicit,
            never_send,
        )
    result = approval(tmp_path).recover("second", never_send)
    assert fields == [["choice"], ["maximum_credits"]]
    assert (
        result["state"]
        == {"cancel": "canceled", "decline": "declined", "timeout": "incomplete"}[
            action
        ]
    )
    assert not result["backend_authorized"]


@pytest.mark.parametrize(
    "response,state",
    [
        ({"action": "accept", "content": {"choice": "cancel"}}, "canceled"),
        ({"action": "cancel", "content": None}, "canceled"),
        ({"action": "decline", "content": None}, "declined"),
        (
            {
                "action": "accept",
                "content": {"choice": "request_changes", "note": "More memory"},
            },
            "changes_requested",
        ),
        (
            {
                "action": "accept",
                "content": {"choice": "approve_paid_stage", "maximum_credits": "11"},
            },
            "incomplete",
        ),
        (
            {"action": "accept", "content": {"choice": "confirm_rehearsal"}},
            "incomplete",
        ),
        ({"action": "accept", "content": {"approved": True}}, "incomplete"),
        ({"action": "accept", "content": {"choice": "request_changes"}}, "incomplete"),
    ],
)
def test_unconfirmed_and_rehearsal_responses_never_send_paid_mutations(
    tmp_path, response, state
):
    current = review()

    async def fetch():
        return current

    async def elicit(message, schema):
        return client_response(response, schema)

    def send(*_):
        raise AssertionError("No paid mutation is permitted")

    async def run():
        try:
            await confirm_paid(
                approval(tmp_path),
                "no-paid",
                current,
                current["proposal_fingerprint"],
                fetch,
                elicit,
                send,
            )
        except ValueError:
            assert state == "incomplete"

    anyio.run(run)
    result = approval(tmp_path).recover("no-paid", send)
    assert result["state"] == state and not result["backend_authorized"]


def test_lost_response_recovers_original_paid_identity_after_settlement(tmp_path):
    current, calls = review(), []

    async def fetch():
        return current

    async def elicit(message, schema):
        return client_response(
            {
                "action": "accept",
                "content": {"choice": "approve_paid_stage", "maximum_credits": "10.25"},
            },
            schema,
        )

    def lost(operation, args):
        calls.append(deepcopy(args))
        raise WorkflowError("response lost")

    with pytest.raises(WorkflowError):
        anyio.run(
            confirm_paid,
            approval(tmp_path),
            "retry",
            current,
            current["proposal_fingerprint"],
            fetch,
            elicit,
            lost,
        )
    assert approval(tmp_path).read("retry")["state"] == "sending"

    def recovered(operation, args):
        calls.append(deepcopy(args))
        return accepted(args)

    result = approval(tmp_path).recover("retry", recovered)
    assert result["state"] == "backend_accepted" and calls[0] == calls[1]
    # The server's original receipt, not a refreshed plan/quote, is authoritative.
    assert approval(tmp_path).recover("retry", recovered) == result
    assert len(calls) == 2


def test_concurrent_recovery_and_pending_prompt_preserve_one_identity(tmp_path):
    current = review()
    saved, created = approval(tmp_path).begin("concurrent", current)
    assert created
    again, created = approval(tmp_path).begin("concurrent", current)
    assert not created and again == saved
    calls = []

    def send(operation, args):
        calls.append(deepcopy(args))
        return accepted(args)

    # An interrupted prompt never becomes authorization on recovery.
    assert approval(tmp_path).recover("concurrent", send)["state"] == "awaiting_input"
    assert not calls
    approval(tmp_path).transition(
        "concurrent", saved["evidence_id"], ("awaiting_input",), "client_confirmed"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: approval(tmp_path).recover("concurrent", send), range(2))
        )
    assert results[0] == results[1] and results[0]["state"] == "backend_accepted"
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["changed_during", "timeout", "unsupported"])
def test_stale_or_interrupted_form_fails_without_paid_authority(tmp_path, failure):
    current = review()

    async def fetch():
        changed = deepcopy(current)
        changed["proposal_fingerprint"] = "sha256:" + "4" * 64
        return changed

    async def elicit(message, schema):
        if failure == "unsupported":
            raise ValueError("unsupported")
        if failure == "timeout":
            await anyio.sleep(1)
        return client_response(
            {
                "action": "accept",
                "content": {"choice": "approve_paid_stage", "maximum_credits": "10.25"},
            },
            schema,
        )

    def send(*_):
        raise AssertionError("No paid mutation is permitted")

    async def run():
        try:
            await confirm_paid(
                approval(tmp_path),
                "failed",
                current,
                current["proposal_fingerprint"],
                fetch,
                elicit,
                send,
                timeout=0.01,
            )
        except (ValueError, TimeoutError):
            assert failure != "changed_during"

    anyio.run(run)
    result = approval(tmp_path).recover("failed", send)
    assert result["state"] == ("stale" if failure == "changed_during" else "incomplete")
    assert not result["backend_authorized"]


def test_old_profile_cannot_make_paid_requests_or_expose_paid_tool(tmp_path):
    connection = Connection(
        "https://example.test", "https://example.test/cloud", "fake-client"
    )
    assert "native_approval" not in connection.binding
    native = Connection(
        connection.issuer,
        connection.resource,
        "fake-native-client",
        native_approval=True,
    )
    assert native.scopes - connection.scopes == {
        connection.resource + "/workflow-approve"
    }
    client = WorkflowClient(
        WorkflowAuth(connection, PrivateStore(tmp_path / "cache"), "fake")
    )
    with pytest.raises(WorkflowError, match="separate explicitly consented"):
        client.call("native_approve", {})
    with pytest.raises(ValueError, match="separate explicitly consented"):
        build_server(
            client,
            {"pilot": tmp_path.resolve()},
            "http://127.0.0.1:5173",
            native_approval=True,
        )


def test_real_mcp_paid_form_has_no_model_approval_argument(tmp_path):
    project = create_project(tmp_path.resolve(), "fresh", "Fake paid approval")
    ids = {
        "project_key": project["project_key"],
        "project_id": project["project_id"],
        "plan_id": "fake-plan",
    }
    current, calls, forms = review(ids=ids), [], []

    class Adapter:
        auth = SimpleNamespace(connection=SimpleNamespace(native_approval=True))

        def call(self, operation, args):
            calls.append((operation, args))
            if operation == "native_review":
                return current
            return accepted(args)

        def journal(self, _):
            return IntentJournal(
                PrivateStore(tmp_path / "journal"), {"subject": "fake-user"}
            )

    async def on_elicit(context, params):
        forms.append(params)
        response = client_response(
            {
                "action": "accept",
                "content": {"choice": "approve_paid_stage", "maximum_credits": "10.25"},
            },
            params.requested_schema,
        )
        return ElicitResult(**response)

    async def run():
        server = build_server(
            Adapter(),
            {"pilot": tmp_path.resolve()},
            "http://127.0.0.1:5173",
            native_approval=True,
        )
        async with Client(
            server, mode="legacy", elicitation_callback=on_elicit
        ) as client:
            request = {
                "action": "request_input",
                "root_id": "pilot",
                "local_project": "fresh",
                "plan_id": "fake-plan",
                "stage": "planning",
                "intent": "mcp-paid",
                "expected_fingerprint": current["proposal_fingerprint"],
            }
            invalid = await client.call_tool(
                APPROVAL_TOOL, {"request": {**request, "approved": True}}
            )
            assert invalid.is_error and not calls
            initial = await client.call_tool(
                APPROVAL_TOOL,
                {"request": {**request, "action": "recover"}},
            )
            assert initial.structured_content["ok"]
            absent = json.loads(initial.structured_content["payload"])
            assert absent["state"] == "absent"
            assert not absent["backend_authorized"]
            assert not calls and not forms
            preview = await client.call_tool(
                APPROVAL_TOOL,
                {"request": {**request, "action": "preview"}},
            )
            assert preview.structured_content["ok"]
            assert (
                json.loads(preview.structured_content["payload"])[
                    "proposal_fingerprint"
                ]
                == request["expected_fingerprint"]
            )
            assert len(calls) == 1 and not forms
            result = await client.call_tool(APPROVAL_TOOL, {"request": request})
            value = json.loads(result.structured_content["payload"])
            assert (
                value["state"] == "backend_accepted"
                and not value["human_approval_verified"]
            )
            assert len(forms) == 2
            assert [op for op, _ in calls].count("native_approve") == 1
            before = len(calls)
            repeat = await client.call_tool(APPROVAL_TOOL, {"request": request})
            assert json.loads(repeat.structured_content["payload"]) == value
            assert len(calls) == before and len(forms) == 2
            wrong = await client.call_tool(
                APPROVAL_TOOL,
                {
                    "request": {
                        **request,
                        "action": "recover",
                        "plan_id": "foreign-plan",
                    }
                },
            )
            assert not wrong.structured_content["ok"] and len(calls) == before

    anyio.run(run)
