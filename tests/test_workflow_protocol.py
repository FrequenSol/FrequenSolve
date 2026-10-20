"""Exercise the new restricted profile through the official MCP client."""

import json

import anyio
from mcp import Client

from frequensolve.workflow.server import RESOURCE, TOOLS, build_server


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
            assert calls == [("discover", {})]

    anyio.run(exercise)


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
