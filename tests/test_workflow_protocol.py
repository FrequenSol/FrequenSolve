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
