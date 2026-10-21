"""Exercise the new restricted profile through the official MCP client."""

import json

import anyio
from mcp import Client

from frequensolve.workflow.server import RESOURCE, TOOLS, build_server


def test_study_save_recovers_exact_intent_across_fresh_clients(tmp_path):
    from frequensolve.workflow.client import create_project
    from frequensolve.workflow.journal import IntentJournal
    from frequensolve.workflow.storage import PrivateStore

    root = tmp_path.resolve()
    create_project(root, "study", "A customer study")
    journal = IntentJournal(PrivateStore(root / "journal"), {"project": "fake-project"})
    calls = []

    class ClientTransport:
        def journal(self, _path):
            return journal

        def call(self, operation, payload):
            calls.append((operation, payload))
            if operation == "study_get":
                return {"study": None}
            assert operation == "study_save"
            if len([x for x in calls if x[0] == "study_save"]) == 1:
                raise RuntimeError("Response lost after server committed")
            return {"study": {"revision": 1, **payload["content"]}}

    async def attempt():
        server = build_server(
            ClientTransport(), {"workspace": root}, "https://example.test"
        )
        async with Client(server, cache=None) as session:
            result = await session.call_tool(
                "workflow_study",
                {
                    "request": {
                        "action": "save",
                        "root_id": "workspace",
                        "local_project": "study",
                        "plan_id": "fake-plan",
                        "title": "My study",
                        "question": "How does pressure vary?",
                        "success_criterion": "Inspect finite receivers.",
                    }
                },
            )
            return result.structured_content

    assert not anyio.run(attempt)["ok"]
    recovered = anyio.run(attempt)
    assert (
        recovered["ok"]
        and json.loads(recovered["payload"])["paid_request_sent"] is False
    )
    mutations = [payload for op, payload in calls if op == "study_save"]
    assert mutations[0] == mutations[1] and mutations[0]["idempotency_key"]
    assert [op for op, _ in calls] == ["study_get", "study_save", "study_save"]


def test_workflow_profile_protocol_and_local_authoring(tmp_path):
    calls = []

    class RestrictedClient:
        def call(self, operation, arguments):
            calls.append((operation, arguments))
            return {"capacity": "unavailable", "paid_authority": False}

    async def exercise():
        root = tmp_path.resolve()
        server = build_server(
            RestrictedClient(), {"pilot": root}, "http://127.0.0.1:5173"
        )
        async with Client(server, cache=None) as session:
            assert {tool.name for tool in (await session.list_tools()).tools} == set(
                TOOLS
            )
            assert {
                str(item.uri) for item in (await session.list_resources()).resources
            } == {RESOURCE}
            discovered = await session.call_tool("workflow_discover", {"request": {}})
            assert (
                json.loads(discovered.structured_content["payload"])["capacity"]
                == "unavailable"
            )
            created = await session.call_tool(
                "workflow_create_project",
                {
                    "request": {
                        "root_id": "pilot",
                        "relative_path": "fresh",
                        "display_name": "Protocol fixture",
                    }
                },
            )
            assert created.structured_content["ok"]
            identity = json.loads(
                (root / "fresh/.frequensolve/project-identity.json").read_text()
            )
            assert identity["project_id"]
            assert (root / "fresh/jobs/acoustic_3d/frequency/frequency.json").is_file()
            collision = await session.call_tool(
                "workflow_create_project",
                {
                    "request": {
                        "root_id": "pilot",
                        "relative_path": "fresh",
                        "display_name": "Must preserve",
                    }
                },
            )
            assert not collision.structured_content["ok"]
            assert (
                json.loads(
                    (root / "fresh/.frequensolve/project-identity.json").read_text()
                )
                == identity
            )
            for action in ("execute", "authorize", "raw_graphql"):
                denied = await session.call_tool(
                    "workflow_plan", {"request": {"action": action, "root_id": "pilot"}}
                )
                assert denied.is_error
            forged = await session.call_tool(
                "workflow_discover", {"request": {"account_id": "foreign"}}
            )
            assert forged.is_error
            missing_project = await session.call_tool(
                "workflow_plan",
                {
                    "request": {
                        "action": "get",
                        "root_id": "pilot",
                        "plan_id": "fake-plan",
                    }
                },
            )
            assert missing_project.is_error
            assert calls == [("discover", {})]

    anyio.run(exercise)


def test_repeated_creation_reconciles_original_remote_preparation(tmp_path):
    from frequensolve.workflow.client import create_project
    from frequensolve.workflow.journal import IntentJournal
    from frequensolve.workflow.storage import PrivateStore

    root = tmp_path.resolve()
    project = create_project(root, "fresh", "Protocol recovery fixture")
    journal = IntentJournal(
        PrivateStore(root / "journal"), {"project": project["project_id"]}
    )
    arguments = {"project_key": project["project_key"], "cluster_id": "fake-cluster"}
    journal.perform(
        "create", "plan_create", arguments, lambda *_: {"status": "preparing"}
    )
    original_key = journal.inspect("create")["idempotency_key"]
    calls = []

    class RecoveryClient:
        def journal(self, path):
            assert path == root / "fresh"
            return journal

        def call(self, operation, payload):
            calls.append((operation, payload))
            return {"status": "quoted", "paid_authority": False}

    async def exercise():
        server = build_server(
            RecoveryClient(), {"pilot": root}, "http://127.0.0.1:5173"
        )
        async with Client(server, cache=None) as session:
            request = {
                "action": "create",
                "root_id": "pilot",
                "local_project": "fresh",
                "intent": "create",
                "cluster_id": "fake-cluster",
            }
            result = await session.call_tool("workflow_plan", {"request": request})
            assert result.structured_content["ok"]
            assert (
                json.loads(result.structured_content["payload"])["status"] == "quoted"
            )
            changed = await session.call_tool(
                "workflow_plan", {"request": {**request, "cluster_id": "changed"}}
            )
            assert not changed.structured_content["ok"]

    anyio.run(exercise)
    assert calls == [("plan_create", {**arguments, "idempotency_key": original_key})]


def test_credit_scan_resumes_pages_and_counts_each_operation_once(tmp_path):
    from types import SimpleNamespace

    from frequensolve.workflow.server import credit_scan

    def row(operation, phase, captured=None):
        return {
            "operation_id": operation,
            "plan_id": "fake-plan",
            "purpose": "planning",
            "phase": phase,
            "observed_at": "2026-01-01T00:00:00Z",
            "captured_credits": captured,
            "reconciliation": "settled" if captured else "not_authorized",
        }

    class HistoryClient:
        auth = SimpleNamespace(
            credentials=lambda: {"subject": "fake-user", "account_id": "fake-account"},
            connection=SimpleNamespace(resource="https://example.test/cloud"),
        )

        def __init__(self):
            self.calls = []

        def call(self, operation, arguments):
            assert operation == "credit_history"
            self.calls.append(arguments)
            page = int(arguments.get("cursor", 0))
            observations = [row("fake-planning", "settled", "3.00")]
            if page == 1:
                observations += [row("fake-planning", "quoted")]
            if page == 5:
                observations += [row("fake-validation", "settled", "2.00")]
            return {
                "credit_history": {
                    "observations": observations,
                    "cursor": str(page + 1) if page < 5 else None,
                }
            }

    client = HistoryClient()
    arguments = {"project_key": "fake-project", "plan_id": "fake-plan"}
    partial = credit_scan(client, tmp_path.resolve(), arguments)
    assert not partial["history_complete"] and partial["captured_credits"] is None
    complete = credit_scan(client, tmp_path.resolve(), arguments)
    assert complete["history_complete"] and complete["captured_credits"] == "5.00"
    assert len(complete["observations"]) == 2 and len(client.calls) == 6
