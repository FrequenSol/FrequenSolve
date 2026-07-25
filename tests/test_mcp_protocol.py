"""Official in-memory MCP protocol tests for the simulation assistant."""

from __future__ import annotations

import builtins
import json
import math
from typing import Any, Mapping

import anyio
import pytest
from click.testing import CliRunner
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

import frequensolve.mcp_server.cli as cli_module
import frequensolve.mcp_server.server as server_module
from frequensolve.mcp_server.cli import main
from frequensolve.mcp_server.server import MCP_SERVER_VERSION, build_server

TOOL_NAMES = {
    "find_vetted_example",
    "create_simulation_draft",
    "validate_simulation_setup",
    "render_starter_python",
    "inspect_simulation_artifact",
    "preview_simulation",
    "explain_validation",
}
PROMPT_NAMES = {
    "start_2d_acoustic",
    "review_simulation_setup",
    "prepare_simulation_run",
    "debug_validation",
}
RESOURCE_NAMES = {
    "identity",
    "contracts",
    "catalog",
    "public-api",
    "physics",
    "authoring-rules",
    "validation-codes",
    "examples",
    "glossary",
    "allowed-roots",
}


def _run(function):
    return anyio.run(function)


def _assert_closed_object_schemas(schema: Mapping[str, Any]) -> None:
    stack: list[Any] = [schema]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if current.get("type") == "object":
                assert current.get("additionalProperties") is False
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _draft() -> dict[str, Any]:
    return {
        "schema": "frequensolve-simulation-draft/v1",
        "scenario_id": "known-small-2d-acoustic",
        "project_name": "project",
        "simulation_name": "simulation",
        "job_name": "frequency_10hz",
        "physics": "acoustic",
        "dimension": 2,
        "frequency_hz": 10.0,
        "receiver_count": 5,
    }


def test_official_client_initializes_and_lists_the_finite_closed_surface():
    async def check() -> None:
        server = build_server()
        initialization = server._mcp_server.create_initialization_options()
        assert initialization.server_version == MCP_SERVER_VERSION
        assert initialization.instructions

        async with create_connected_server_and_client_session(server) as session:
            await session.send_ping()
            tools = (await session.list_tools()).tools
            resources = (await session.list_resources()).resources
            prompts = (await session.list_prompts()).prompts

            assert {tool.name for tool in tools} == TOOL_NAMES
            assert {
                str(resource.uri).rsplit("/", 1)[-1] for resource in resources
            } == RESOURCE_NAMES
            assert {prompt.name for prompt in prompts} == PROMPT_NAMES
            for tool in tools:
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is True
                assert tool.annotations.destructiveHint is False
                assert tool.annotations.idempotentHint is True
                assert tool.annotations.openWorldHint is False
                _assert_closed_object_schemas(tool.inputSchema)
                assert tool.outputSchema is not None
                _assert_closed_object_schemas(tool.outputSchema)

            create_tool = next(
                tool for tool in tools if tool.name == "create_simulation_draft"
            )
            frequency_schema = create_tool.inputSchema["$defs"]["CreateDraftRequest"][
                "properties"
            ]["frequency_hz"]
            assert frequency_schema["minimum"] == 10.0
            assert frequency_schema["maximum"] == 10.0

            for prompt_name in PROMPT_NAMES:
                rendered = await session.get_prompt(prompt_name)
                assert rendered.messages
            identity = await session.read_resource(
                AnyUrl("frequensolve://simulation-assistant/identity")
            )
            payload = json.loads(identity.contents[0].text)
            assert payload["mcp_contract"] == ("frequensolve-simulation-assistant/v1")
            assert payload["package"]["version"]
            assert payload["package"]["full_revisionid"]

    _run(check)


def test_canonical_2d_acoustic_agent_flow_is_deterministic_and_read_only():
    async def check() -> None:
        server = build_server()
        async with create_connected_server_and_client_session(server) as session:
            create_arguments = {
                "request": {
                    "project_name": "starter-project",
                    "simulation_name": "starter_simulation",
                    "frequency_hz": 10.0,
                    "receiver_count": 9,
                }
            }
            first = await session.call_tool(
                "create_simulation_draft",
                create_arguments,
            )
            second = await session.call_tool(
                "create_simulation_draft",
                create_arguments,
            )
            assert first.isError is False
            assert first.structuredContent == second.structuredContent
            assert first.structuredContent is not None
            assert first.structuredContent["identity"]["mcp_server_version"] == (
                MCP_SERVER_VERSION
            )
            draft = json.loads(first.structuredContent["payload"])
            assert draft["job_name"] == "frequency_10hz"

            source = {"kind": "draft", "draft": draft}
            validation = await session.call_tool(
                "validate_simulation_setup",
                {"request": {"source": source}},
            )
            preview = await session.call_tool(
                "preview_simulation",
                {"request": {"source": source}},
            )
            rendering = await session.call_tool(
                "render_starter_python",
                {"request": {"draft": draft}},
            )
            example = await session.call_tool(
                "find_vetted_example",
                {"request": {"query": "2D acoustic"}},
            )
            explanation = await session.call_tool(
                "explain_validation",
                {"request": {"codes": ["field.unsupported"]}},
            )

            for result in (
                validation,
                preview,
                rendering,
                example,
                explanation,
            ):
                assert result.isError is False
                assert result.structuredContent is not None
                assert result.structuredContent["ok"] is True

            validated = json.loads(validation.structuredContent["payload"])
            previewed = json.loads(preview.structuredContent["payload"])
            rendered = rendering.structuredContent["payload"]
            explained = json.loads(explanation.structuredContent["payload"])
            assert validated == {
                "diagnostics": [],
                "draft_contract": "frequensolve-simulation-draft/v1",
                "error_count": 0,
                "valid": True,
                "warning_count": 0,
            }
            assert previewed["frequencies_hz"] == [10.0]
            assert previewed["task_count"] == 1
            assert previewed["receiver_count"] == 9
            assert explained["items"][0]["code"] == "field.unsupported"
            compile(rendered, "<mcp-starter>", "exec")
            for forbidden in (
                ".save(",
                ".submit(",
                ".run(",
                ".dry_run(",
                "subprocess",
                "__import__",
                "importlib",
            ):
                assert forbidden not in rendered

    _run(check)


def test_protocol_errors_are_stable_strict_and_do_not_reflect_input():
    async def check() -> None:
        server = build_server()
        sensitive = "DO_NOT_REFLECT_PRIVATE_VALUE"
        invalid_calls = (
            (
                "create_simulation_draft",
                {"request": {"receiver_count": True}},
            ),
            (
                "create_simulation_draft",
                {"request": {"frequency_hz": math.nan}},
            ),
            (
                "create_simulation_draft",
                {"request": {"frequency_hz": 8.25}},
            ),
            (
                "create_simulation_draft",
                {"request": {"frequency_hz": 1 << 300_000}},
            ),
            (
                "create_simulation_draft",
                {"request": {"units": {"length": sensitive}}},
            ),
            (
                "validate_simulation_setup",
                {
                    "request": {
                        "source": {
                            "kind": "draft",
                            "draft": {**_draft(), "physics": sensitive},
                        }
                    }
                },
            ),
            (
                "inspect_simulation_artifact",
                {
                    "request": {
                        "artifact": {
                            "root_id": "project",
                            "relative_path": "../" + sensitive + ".json",
                        }
                    }
                },
            ),
            (
                "find_vetted_example",
                {"request": {"query": sensitive * 6_000}},
            ),
        )
        async with create_connected_server_and_client_session(server) as session:
            for name, arguments in invalid_calls:
                result = await session.call_tool(name, arguments)
                assert result.isError is True
                text = result.content[0].text
                assert text == (
                    "Invalid tool arguments; use the published input schema."
                )
                assert sensitive not in text

            unknown = await session.call_tool(
                "unknown_private_tool",
                {"private": sensitive},
            )
            assert unknown.isError is True
            assert unknown.content[0].text == "Unknown tool; use tools/list."
            assert sensitive not in unknown.content[0].text

            with pytest.raises(Exception) as resource_error:
                await session.read_resource(
                    AnyUrl(
                        "frequensolve://simulation-assistant/"
                        "unknown-private-resource"
                    )
                )
            assert "unknown-private-resource" not in str(resource_error.value)

            with pytest.raises(Exception) as prompt_error:
                await session.get_prompt("unknown-private-prompt")
            assert "unknown-private-prompt" not in str(prompt_error.value)

    _run(check)


def test_tool_results_redact_sensitive_fields_paths_and_prohibited_uris(
    monkeypatch,
):
    async def check() -> None:
        server = build_server()
        monkeypatch.setattr(
            server_module.core,
            "find_vetted_example",
            lambda _query: {
                "apiToken": "DO_NOT_RETURN_TOKEN",
                "nested": {
                    "password": "DO_NOT_RETURN_PASSWORD",
                    "authorization": "Bearer abcdefghijklmnop",
                    "path": "/Users/private/customer/input.json",
                    "blob": "s3://private-bucket/private/key",
                    "embedded": (
                        "See s3://embedded-bucket/private/key and "
                        "file:///Users/private/embedded.json"
                    ),
                    "short_paths": "root=/etc temporary=/tmp",
                    "documentation": "https://example.com/public/guide",
                },
            },
        )
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "find_vetted_example",
                {"request": {"query": "safe"}},
            )
            assert result.structuredContent is not None
            payload = result.structuredContent["payload"]
            assert "DO_NOT_RETURN" not in payload
            assert "/Users/private" not in payload
            assert "s3://private-bucket" not in payload
            assert "s3://embedded-bucket" not in payload
            assert "file:///Users/private" not in payload
            assert "root=/etc" not in payload
            assert "temporary=/tmp" not in payload
            assert payload.count("<redacted-secret>") == 3
            assert "<redacted-path>" in payload
            assert "<redacted-uri>" in payload
            assert "https://example.com/public/guide" in payload

            monkeypatch.setattr(
                server_module.core,
                "find_vetted_example",
                lambda query: {"query": query},
            )
            for secret_query in (
                "OPENAI_API_KEY=sk-proj-1234567890abcdef",
                "AWS_SECRET_ACCESS_KEY=1234567890abcdef",
                "credential sk-proj-1234567890abcdef",
            ):
                redacted = await session.call_tool(
                    "find_vetted_example",
                    {"request": {"query": secret_query}},
                )
                assert redacted.structuredContent is not None
                redacted_payload = redacted.structuredContent["payload"]
                assert "1234567890abcdef" not in redacted_payload
                assert "<redacted-secret>" in redacted_payload

            monkeypatch.setattr(
                server_module.core,
                "find_vetted_example",
                lambda _query: {"oversized": "x" * 140_000},
            )
            oversized = await session.call_tool(
                "find_vetted_example",
                {"request": {"query": "safe"}},
            )
            assert oversized.structuredContent is not None
            assert oversized.structuredContent["ok"] is False
            assert oversized.structuredContent["diagnostics"][0]["code"] == (
                "mcp.output.too_large"
            )

    _run(check)


def test_worker_timeout_is_reported_as_a_stable_tool_diagnostic(monkeypatch):
    async def never_returns(*_args, **_kwargs):
        await anyio.sleep_forever()

    async def check() -> None:
        monkeypatch.setattr(
            server_module.anyio.to_process,
            "run_sync",
            never_returns,
        )
        server = build_server(operation_timeout_seconds=1)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "validate_simulation_setup",
                {
                    "request": {
                        "source": {
                            "kind": "draft",
                            "draft": _draft(),
                        }
                    }
                },
            )
            assert result.isError is False
            assert result.structuredContent is not None
            assert result.structuredContent["ok"] is False
            assert result.structuredContent["diagnostics"][0]["code"] == (
                "mcp.execution.timeout"
            )

    _run(check)


def test_client_cancellation_stops_the_worker_and_keeps_the_session_usable(
    monkeypatch,
):
    async def check() -> None:
        started = anyio.Event()
        stopped = anyio.Event()
        original = server_module.anyio.to_process.run_sync

        async def never_returns(*_args, **_kwargs):
            started.set()
            try:
                await anyio.sleep_forever()
            finally:
                stopped.set()

        monkeypatch.setattr(
            server_module.anyio.to_process,
            "run_sync",
            never_returns,
        )
        server = build_server(operation_timeout_seconds=60)
        cancellation_errors: list[str] = []
        async with create_connected_server_and_client_session(server) as session:
            request_id = session._request_id
            async with anyio.create_task_group() as task_group:

                async def call_tool() -> None:
                    try:
                        await session.call_tool(
                            "validate_simulation_setup",
                            {
                                "request": {
                                    "source": {
                                        "kind": "draft",
                                        "draft": _draft(),
                                    }
                                }
                            },
                        )
                    except Exception as exc:
                        cancellation_errors.append(str(exc))

                task_group.start_soon(call_tool)
                await started.wait()
                assert session._request_id == request_id + 1
                await session.send_notification(
                    types.ClientNotification(
                        types.CancelledNotification(
                            params=types.CancelledNotificationParams(
                                requestId=request_id,
                                reason="MCP protocol cancellation test",
                            )
                        )
                    )
                )
                with anyio.fail_after(1):
                    await stopped.wait()

            assert cancellation_errors == ["Request cancelled"]
            monkeypatch.setattr(
                server_module.anyio.to_process,
                "run_sync",
                original,
            )
            await session.send_ping()

    _run(check)


def test_two_official_clients_can_use_one_server_concurrently():
    async def check() -> None:
        server = build_server(max_concurrency=2)
        results: list[str] = []

        async def use_client(name: str) -> None:
            async with create_connected_server_and_client_session(server) as session:
                response = await session.call_tool(
                    "create_simulation_draft",
                    {"request": {"project_name": name}},
                )
                assert response.structuredContent is not None
                payload = json.loads(response.structuredContent["payload"])
                results.append(payload["project_name"])

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(use_client, "client-one")
            task_group.start_soon(use_client, "client-two")

        assert sorted(results) == ["client-one", "client-two"]

    _run(check)


@pytest.mark.parametrize(
    ("timeout", "concurrency"),
    [
        (0, 1),
        (61, 1),
        (math.nan, 1),
        (math.inf, 1),
        (15, 0),
        (15, 5),
        (15, True),
    ],
)
def test_programmatic_server_limits_match_the_cli(timeout, concurrency):
    with pytest.raises(ValueError):
        build_server(
            operation_timeout_seconds=timeout,
            max_concurrency=concurrency,
        )


def test_cli_doctor_uses_the_official_in_memory_client():
    result = CliRunner().invoke(main, ["doctor"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "allowed_root_ids": [],
        "ok": True,
        "prompts": 4,
        "resources": 10,
        "tools": 7,
        "transport": "in-memory",
    }


def test_cli_missing_mcp_extra_shows_the_install_command(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "frequensolve.mcp_server._sdk_v1":
            raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = CliRunner().invoke(main, ["doctor"])

    assert result.exit_code == 1
    assert result.output == (
        "Error: Install MCP support with: " "pip install 'frequensolve[mcp]'\n"
    )
    assert "Traceback" not in result.output


def test_cli_rejects_broad_roots_and_hides_startup_exceptions(monkeypatch):
    runner = CliRunner()
    broad = runner.invoke(main, ["doctor", "--allow-root", "all=/"])

    assert broad.exit_code == 1
    assert broad.output == (
        "Error: Choose a narrower non-sensitive allowed root directory.\n"
    )
    assert "Traceback" not in broad.output

    unknown_user = runner.invoke(
        main,
        [
            "doctor",
            "--allow-root",
            "project=~definitely_no_such_frequensolve_user",
        ],
    )
    assert unknown_user.exit_code == 1
    assert unknown_user.output == (
        "Error: Each allowed root must be an existing absolute directory.\n"
    )
    assert "Traceback" not in unknown_user.output

    def fail_startup(*_args, **_kwargs):
        raise RuntimeError("private failure at /Users/customer/private")

    monkeypatch.setattr(cli_module, "_build_server", fail_startup)
    failed = runner.invoke(main, ["doctor"])

    assert failed.exit_code == 1
    assert failed.output == "Error: The MCP startup check failed safely.\n"
    assert "customer" not in failed.output
    assert "Traceback" not in failed.output
