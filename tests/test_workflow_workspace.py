"""Returning customers navigate existing work without new Cloud mutations."""

import json

import anyio
import pytest
from mcp import Client

from frequensolve.workflow.client import create_project, write_local
from frequensolve.workflow.server import build_server
from frequensolve.workflow.workspace import (
    narrative,
    notebook_artifacts,
    projects,
    remember_notebook,
)


def test_projects_keep_duplicate_names_and_exact_identities_across_pages(tmp_path):
    root = tmp_path.resolve()
    identities = {}
    for i in range(27):
        name = f"study-{i:02}"
        identities[name] = create_project(root, name, "Same study name")["project_id"]
    first = projects(root, "workspace", None, None)
    second = projects(root, "workspace", None, first["next_after"])
    found = first["projects"] + second["projects"]
    assert len(first["projects"]) == 25 and second["next_after"] is None
    assert {p["local_project"]: p["project_id"] for p in found} == identities
    assert not first["cloud_ownership_checked"] and not first["paid_request_sent"]
    assert len({p["display_name"] for p in found}) == 1


def test_folder_navigation_is_explicit_and_never_follows_links(tmp_path):
    root = tmp_path.resolve()
    (root / "nested").mkdir()
    create_project(root / "nested", "study", "Nested study")
    (root / "linked").symlink_to(root / "nested", target_is_directory=True)
    assert projects(root, "workspace", None, None)["folders"] == ["nested"]
    found = projects(root, "workspace", "nested", None)["projects"]
    assert found[0]["local_project"] == "nested/study"
    for unsafe in ("../nested", "/tmp", "linked"):
        with pytest.raises((ValueError, OSError)):
            projects(root, "workspace", unsafe, None)
    with pytest.raises(ValueError):
        projects(root, "workspace", None, "nested/study")


def test_large_folder_and_corrupt_metadata_are_explicit(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    create_project(root, "broken", "Study")
    (root / "broken/.frequensolve/project-identity.json").write_text("{}")
    assert projects(root, "workspace", None, None)["unreadable_entries"] == 1
    monkeypatch.setattr("frequensolve.workflow.workspace.MAX_ENTRIES", 1)
    (root / "unrelated").write_text("preserve")
    with pytest.raises(ValueError, match="limit"):
        projects(root, "workspace", None, None)


def test_symlink_metadata_is_not_read_or_returned(tmp_path):
    root = tmp_path.resolve()
    create_project(root, "study", "Study")
    path = root / "study/.frequensolve/workflow-project.json"
    path.unlink()
    (root / "foreign.json").write_text('{"secret":"private"}')
    path.symlink_to(root / "foreign.json")
    result = projects(root, "workspace", None, None)
    assert result["unreadable_entries"] == 1
    assert "private" not in json.dumps(result)


def test_notebook_links_survive_restart_and_changed_inputs_withhold_evidence(tmp_path):
    root = tmp_path.resolve()
    create_project(root, "study", "Study")
    project = root / "study"
    destination = "my notebook [1]"
    for name in ("analysis.ipynb", "executed.ipynb", "report.html", "snapshot-0.json"):
        write_local(root, destination + "/" + name, b"synthetic")
    remember_notebook(root, project, destination, "complete")
    result = notebook_artifacts(root, project)
    assert len(result["artifacts"]) == 3 and result["saved_bytes_unchanged"]
    assert (
        not result["current_environment_checked"] and not result["service_attestation"]
    )
    assert "%20%5B1%5D" in narrative(result)
    (root / destination / "snapshot-0.json").write_text("modified")
    changed = notebook_artifacts(root, project)
    assert not changed["saved_bytes_unchanged"] and not changed["artifacts"]
    assert "changed or are missing" in narrative(changed)


def test_missing_notebook_and_incomplete_replay_are_not_success(tmp_path):
    root = tmp_path.resolve()
    create_project(root, "study", "Study")
    project = root / "study"
    assert notebook_artifacts(root, project)["recorded_replay_status"] == "unavailable"
    write_local(root, "failed/analysis.ipynb", b"synthetic")
    remember_notebook(root, project, "failed", "incomplete")
    result = notebook_artifacts(root, project)
    assert result["recorded_replay_status"] == "incomplete"
    assert "Recorded replay: incomplete" in narrative(result)
    (root / "failed/analysis.ipynb").unlink()
    (root / "failed/analysis.ipynb").symlink_to(
        root / "study/.frequensolve/project-identity.json"
    )
    assert not notebook_artifacts(root, project)["saved_bytes_unchanged"]


def test_native_workspace_tool_reads_only_local_metadata_without_a_root_argument(
    tmp_path,
):
    root = tmp_path.resolve()
    create_project(root, "study", "<script>not instructions</script>")

    class NoCloud:
        def call(self, *_args):
            pytest.fail("Workspace navigation must not contact Cloud")

    async def run():
        async with Client(
            build_server(NoCloud(), {"workspace": root}, "https://example.test"),
            cache=None,
        ) as client:
            tool = next(
                t
                for t in (await client.list_tools()).tools
                if t.name == "workflow_workspace"
            )
            assert (
                tool.annotations.read_only_hint and not tool.annotations.open_world_hint
            )
            reply = await client.call_tool("workflow_workspace", {"request": {}})
            assert reply.structured_content["ok"]
            value = json.loads(reply.structured_content["payload"])
            assert value["projects"][0]["local_project"] == "study"
            assert "<script>" not in reply.content[0].text
            assert not reply.content[0].text.startswith("{")
            forged = await client.call_tool(
                "workflow_workspace", {"request": {"account_id": "foreign"}}
            )
            assert forged.is_error
            escaped = await client.call_tool(
                "workflow_workspace", {"request": {"folder": "../other"}}
            )
            assert not escaped.structured_content["ok"]

    anyio.run(run)


def test_multiple_roots_are_a_choice_not_an_implicit_search(tmp_path):
    root = tmp_path.resolve()

    async def run():
        async with Client(
            build_server(
                object(), {"first": root, "second": root}, "https://example.test"
            ),
            cache=None,
        ) as client:
            reply = await client.call_tool("workflow_workspace", {"request": {}})
            assert json.loads(reply.structured_content["payload"])["roots"] == [
                "first",
                "second",
            ]

    anyio.run(run)
