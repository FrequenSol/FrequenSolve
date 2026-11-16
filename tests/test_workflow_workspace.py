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


def test_empty_page_does_not_hide_a_project_on_a_later_page(tmp_path):
    root = tmp_path.resolve()
    for i in range(30):
        (root / f"folder-{i:02}").mkdir()
    identity = create_project(root, "zz-retained-study", "Retained acoustic study")
    first = projects(root, "workspace", None, None)
    assert not first["projects"] and first["next_after"] == "folder-24"
    assert first["after"] is None and first["entries_checked"] == 25
    assert first["search_scope"] == "direct_local_subfolders"
    message = narrative(first)
    assert "No curated projects were found on this page." in message
    assert "Continue to the next page" in message
    assert "Other folders and saved Cloud work have not been searched." in message
    second = projects(root, "workspace", None, first["next_after"])
    assert second["after"] == "folder-24" and second["entries_checked"] == 6
    assert second["next_after"] is None
    assert second["projects"][0]["project_id"] == identity["project_id"]
    assert "Retained acoustic study" in narrative(second)


def test_empty_final_page_retains_earlier_project_results(tmp_path):
    root = tmp_path.resolve()
    create_project(root, "a-study", "Earlier study")
    for i in range(25):
        (root / f"folder-{i:02}").mkdir()
    first = projects(root, "workspace", None, None)
    assert len(first["projects"]) == 1 and first["next_after"]
    final = projects(root, "workspace", None, first["next_after"])
    assert not final["projects"] and final["next_after"] is None
    assert final["entries_checked"] == 1
    message = narrative(final)
    assert "No curated projects were found on this final page." in message
    assert "Keep projects found on earlier pages." in message
    assert "Other folders and saved Cloud work have not been searched." in message


@pytest.mark.parametrize("with_local_file", [False, True])
def test_empty_local_scan_has_bounded_scope_and_existing_work_recovery(
    tmp_path, with_local_file
):
    root = tmp_path.resolve()
    if with_local_file:
        (root / "unrelated.txt").write_text("preserve")
    value = projects(root, "workspace", None, None)
    assert value["entries_checked"] == 0 and value["next_after"] is None
    assert not value["projects"] and not value["folders"]
    message = narrative(value)
    assert "No curated projects or inspectable subfolders" in message
    assert "directly in this local folder" in message
    assert "Other folders and saved Cloud work have not been searched." in message
    assert "existing parent folder within an allowed workspace" in message
    assert "restore its local project files" in message


@pytest.mark.parametrize("identity_file", ["missing", "invalid"])
def test_incomplete_project_metadata_remains_unconfirmed(tmp_path, identity_file):
    root = tmp_path.resolve()
    create_project(root, "retained-study", "Retained study")
    identity = root / "retained-study/.frequensolve/project-identity.json"
    if identity_file == "missing":
        identity.unlink()
    else:
        identity.write_text("{}")
    value = projects(root, "workspace", None, None)
    assert value["unreadable_entries"] == 1
    assert not value["projects"] and not value["folders"]
    message = narrative(value)
    assert "readable entries" in message and "remain unconfirmed" in message
    assert "Other folders and saved Cloud work have not been searched." in message


def test_unreadable_metadata_does_not_imply_an_empty_folder(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "retained-study").mkdir()

    def denied(*_args):
        raise PermissionError("synthetic permission failure")

    monkeypatch.setattr("frequensolve.workflow.workspace.read_local", denied)
    value = projects(root, "workspace", None, None)
    assert value["unreadable_entries"] == 1
    assert "remain unconfirmed" in narrative(value)
    assert "synthetic permission" not in narrative(value)


def test_folder_navigation_is_explicit_and_never_follows_links(tmp_path):
    root = tmp_path.resolve()
    (root / "nested").mkdir()
    create_project(root / "nested", "study", "Nested study")
    (root / "linked").symlink_to(root / "nested", target_is_directory=True)
    first = projects(root, "workspace", None, None)
    assert first["folders"] == ["nested"] and not first["projects"]
    assert "Choose a folder to inspect explicitly:\n- nested" in narrative(first)
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


def test_fresh_mcp_client_continues_discovery_without_cloud_or_mutation(tmp_path):
    root = tmp_path.resolve()
    for i in range(25):
        (root / f"folder-{i:02}").mkdir()
    identity = create_project(root, "zz-study", "Existing study")

    def snapshot():
        return {
            str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot()

    class NoCloud:
        def call(self, *_args):
            pytest.fail("Local discovery must not contact Cloud")

    async def page(request):
        async with Client(
            build_server(NoCloud(), {"workspace": root}, "https://example.test"),
            cache=None,
        ) as client:
            reply = await client.call_tool("workflow_workspace", {"request": request})
            assert reply.structured_content["ok"]
            return (
                json.loads(reply.structured_content["payload"]),
                reply.content[0].text,
            )

    first, first_text = anyio.run(page, {})
    assert not first["projects"] and "on this page" in first_text
    final, final_text = anyio.run(page, {"after": first["next_after"]})
    assert final["projects"][0]["project_id"] == identity["project_id"]
    assert "Existing study" in final_text and final["next_after"] is None
    assert not final["paid_request_sent"] and not final["cloud_ownership_checked"]
    assert snapshot() == before
    assert not (root / "zz-study/.frequensolve/workflow").exists()
