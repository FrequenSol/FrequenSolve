"""Returning customers can look up saved work without an editing capability."""

import hashlib
import json
from copy import deepcopy

import anyio
import pytest
from mcp import Client

from frequensolve.workflow.client import create_project
from frequensolve.workflow.server import OBSERVATION_TOOLS, build_server

CURSOR = {
    "phase": "preparations",
    "key": {"scope": "fake-preparations", "id": "fake-page-key"},
}
ROW = {
    "plan_id": "fake-plan",
    "revision": 3,
    "status": "executed",
    "execution_id": "fake-run",
    "updated_at": "2026-01-02T03:04:05Z",
}


def test_read_only_lookup_preserves_scope_pagination_and_files(tmp_path):
    root = tmp_path.resolve()
    project = create_project(root, "study", "Acoustic study")
    calls = []

    class Reader:
        def call(self, operation, arguments):
            assert operation == "plan_list"
            calls.append(deepcopy(arguments))
            if arguments.get("cursor"):
                return {"items": [], "cursor": None}
            # Execution admission can be disabled while owned history is readable.
            return {"available": False, "items": [ROW], "cursor": CURSOR}

        def prepare(self, *_):
            pytest.fail("A lookup must not prepare or upload")

        def journal(self, *_):
            pytest.fail("A lookup must not recover a mutation")

    def files():
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        }

    before = files()

    async def run():
        async with Client(
            build_server(Reader(), {"root": root}, "https://example.test"), cache=None
        ) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            for name in OBSERVATION_TOOLS:
                annotation = tools[name].annotations
                assert annotation.read_only_hint and annotation.idempotent_hint
                assert not annotation.destructive_hint
            request = {"root_id": "root", "local_project": "study"}
            reply = await client.call_tool("workflow_saved_plans", {"request": request})
            assert reply.structured_content["ok"] and not reply.is_error
            result = json.loads(reply.structured_content["payload"])
            assert result["items"] == [ROW] and result["cursor"] == CURSOR
            assert result["available"] is False and not result["paid_request_sent"]
            assert result["next_action"] == "open_saved_plan"
            text = reply.content[0].text
            assert "executed" in text and "2026-01-02T03:04:05Z" in text
            assert "does not establish successful science" in text
            assert "No saved plans" not in text and "fake-plan" not in text
            page = await client.call_tool(
                "workflow_saved_plans",
                {"request": {**request, "cursor": result["cursor"]}},
            )
            assert page.structured_content["ok"]
            assert "final page has no additional saved plans" in page.content[0].text
            assert "No saved plans are recorded" not in page.content[0].text
            assert (
                json.loads(page.structured_content["payload"])["next_action"]
                == "finish_saved_plan_lookup"
            )
            for extra in (
                {"action": "create"},
                {"account_id": "foreign"},
                {"user_id": "foreign"},
                {"project_key": "foreign"},
                {"project_id": "foreign"},
                {"intent": "new-work"},
                {"maximum_credits": "100"},
                {"cursor": {**CURSOR, "phase": "execute"}},
                {"cursor": {**CURSOR, "key": {**CURSOR["key"], "user_id": "foreign"}}},
            ):
                denied = await client.call_tool(
                    "workflow_saved_plans", {"request": {**request, **extra}}
                )
                assert denied.is_error

    anyio.run(run)
    scope = {k: project[k] for k in ("project_id", "project_key")}
    assert calls == [scope, {**scope, "cursor": CURSOR}]
    assert files() == before
    assert not (root / "study/.frequensolve/workflow").exists()


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "next_page",
        "timeout",
        "missing_items",
        "missing_cursor",
        "bad_cursor",
        "bad_row",
        "too_many",
    ],
)
def test_empty_history_is_distinct_from_unconfirmed_history(tmp_path, case):
    root = tmp_path.resolve()
    create_project(root, "study", "Acoustic study")
    calls = []

    class Reader:
        def call(self, operation, arguments):
            calls.append(operation)
            assert operation == "plan_list"
            if case == "timeout":
                raise TimeoutError("sensitive-provider-detail")
            return {
                "empty": {"items": [], "cursor": None},
                "next_page": {"items": [], "cursor": CURSOR},
                "missing_items": {"available": False},
                "missing_cursor": {"items": []},
                "bad_cursor": {"items": [], "cursor": {"phase": "raw-operation"}},
                "bad_row": {"items": [{**ROW, "status": False}], "cursor": None},
                "too_many": {"items": [ROW] * 101, "cursor": None},
            }[case]

    async def run():
        async with Client(
            build_server(Reader(), {"root": root}, "https://example.test"), cache=None
        ) as client:
            return await client.call_tool(
                "workflow_saved_plans",
                {"request": {"root_id": "root", "local_project": "study"}},
            )

    reply = anyio.run(run)
    payload = json.loads(reply.structured_content["payload"])
    if case in ("empty", "next_page"):
        assert reply.structured_content["ok"]
        assert payload["next_action"] == (
            "read_next_saved_plan_page"
            if case == "next_page"
            else "choose_whether_to_prepare_study"
        )
        if case == "next_page":
            assert "No saved plans are recorded" not in reply.content[0].text
    else:
        assert not reply.structured_content["ok"]
        assert payload["read_outcome"] == "unknown"
        assert payload["next_action"] == "retry_same_read_with_same_saved_ids"
        assert "sensitive-provider-detail" not in json.dumps(reply.structured_content)
        assert "No saved plans are recorded" not in reply.content[0].text
    assert not payload["paid_request_sent"]
    assert calls == ["plan_list"]
