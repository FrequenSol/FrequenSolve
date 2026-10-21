"""Registered history survives newer copies without new execution or authority."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import anyio
import pytest
from mcp import Client

from frequensolve.workflow import notebook_catalog as catalog
from frequensolve.workflow.client import create_project, write_local
from frequensolve.workflow.notebook_binding import context_binding
from frequensolve.workflow.notebook_recovery import recover
from frequensolve.workflow.server import build_server
from frequensolve.workflow.storage import canonical
from frequensolve.workflow.workspace import (
    narrative,
    notebook_artifacts,
    notebook_history,
    remember_notebook,
)
from tests.test_workflow_notebook_recovery import manifest, state


def setup(root):
    created = create_project(root, "project", "Saved acoustic study")
    value = state()
    value["plan"]["head"]["owner"] = {
        k: created[k] for k in ("project_id", "project_key")
    }
    value["project_identity"] = {
        **value["plan"]["head"]["owner"],
        "plan_id": "fake-plan",
    }
    return root / "project", value


def register(
    root, project, value, destination, indices, *, status="complete", bound=True
):
    """Explicit saved-byte fixtures; these do not claim real kernel evidence."""
    for name in ("analysis.ipynb", "executed.ipynb", "report.html", "snapshot.json"):
        write_local(
            root, destination + "/" + name, (destination + ":synthetic").encode()
        )
    binding = context_binding(
        value["plan"], value["study"], indices, result_digest="sha256:" + "3" * 64
    )
    remember_notebook(
        root,
        project,
        destination,
        status,
        binding=binding if bound else None,
        label=value["study"]["title"],
    )
    return notebook_artifacts(root, project)["notebook_id"]


def snapshot(root):
    return {
        str(p.relative_to(root)): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def test_earlier_context_survives_new_registration_and_reads_do_not_write(tmp_path):
    root = tmp_path.resolve()
    project, value = setup(root)
    earlier = register(root, project, value, "earlier-2hz", [0])
    latest = register(root, project, value, "later-20hz", [1])
    before = snapshot(root)
    history = notebook_history(project)
    assert {item["notebook_id"] for item in history["notebooks"]} == {earlier, latest}
    assert history["history_complete"] and not history["next_after"]
    calls = []

    def fetch(arguments):
        calls.append(arguments)
        return manifest(value)

    recovered = recover(root, project, value, [0], None, fetch)
    assert recovered["notebook_id"] == earlier
    assert recovered["context_match"]["status"] == "match"
    assert all("earlier-2hz" in item["path"] for item in recovered["artifacts"])
    assert len(calls) == 1 and calls[0]["execution_id"] == "fake-run"
    assert notebook_artifacts(root, project)["notebook_id"] == latest
    assert snapshot(root) == before


def test_multiple_matching_copies_require_explicit_choice_before_file_checks(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    project, value = setup(root)
    first = register(root, project, value, "original", [0])
    second = register(root, project, value, "another-copy", [0])
    real_check = notebook_artifacts
    monkeypatch.setattr(
        "frequensolve.workflow.notebook_recovery.notebook_artifacts",
        lambda *_: pytest.fail("Choice discovery must not hash every bundle"),
    )
    choice = recover(root, project, value, [0], None, lambda _: manifest(value))
    assert choice["selection_required"] and not choice["artifacts"]
    assert {item["notebook_id"] for item in choice["candidates"]} == {first, second}
    assert all(not item["saved_bytes_checked"] for item in choice["candidates"])
    monkeypatch.setattr(
        "frequensolve.workflow.notebook_recovery.notebook_artifacts", real_check
    )
    selected = recover(root, project, value, [0], first, lambda _: manifest(value))
    assert (
        selected["notebook_id"] == first
        and selected["context_match"]["status"] == "match"
    )
    assert all("original" in item["path"] for item in selected["artifacts"])


@pytest.mark.parametrize(
    "case",
    [
        "changed",
        "missing",
        "incomplete",
        "unbound",
        "wrong_project",
        "wrong_run",
        "wrong_results",
        "manifest_unknown",
    ],
)
def test_selected_history_keeps_integrity_context_and_uncertainty_guards(
    tmp_path, case
):
    root = tmp_path.resolve()
    project, value = setup(root)
    saved = deepcopy(value)
    if case == "wrong_project":
        saved["plan"]["head"]["owner"]["project_id"] = "foreign-project"
    if case == "wrong_run":
        saved["plan"]["head"]["execution_id"] = "different-run"
    selected = register(
        root,
        project,
        saved,
        "earlier",
        [0],
        status="incomplete" if case == "incomplete" else "complete",
        bound=case != "unbound",
    )
    register(root, project, value, "later", [1])
    if case == "changed":
        (root / "earlier/snapshot.json").write_text("changed")
    if case == "missing":
        (root / "earlier/analysis.ipynb").unlink()
    calls = []

    def fetch(_):
        calls.append("manifest")
        if case == "manifest_unknown":
            raise TimeoutError("private detail")
        response = manifest(value)
        if case == "wrong_results":
            response["results"]["digest"] = "sha256:" + "4" * 64
        return response

    result = recover(root, project, value, [0], selected, fetch)
    if case == "incomplete":
        assert [item["name"] for item in result["artifacts"]] == ["analysis.ipynb"]
    else:
        assert not result["artifacts"]
    assert not result["paid_request_sent"] and not result["service_attestation"]
    if case in ("changed", "missing", "unbound", "wrong_project", "wrong_run"):
        assert not calls
    if case == "manifest_unknown":
        assert result["context_match"]["status"] == "unconfirmed"
    assert "private detail" not in json.dumps(result)


def test_history_pages_keep_all_entries_and_cursor_scoped_reads(tmp_path):
    root = tmp_path.resolve()
    project, value = setup(root)
    ids = {
        register(root, project, value, f"notebook-{i:02}", [i % 2]) for i in range(30)
    }
    before = snapshot(root)
    latest = notebook_artifacts(root, project)["notebook_id"]
    first = notebook_history(project)
    final = notebook_history(project, first["next_after"])
    assert first["latest_notebook_id"] == final["latest_notebook_id"] == latest
    assert [
        item["notebook_id"]
        for item in first["notebooks"] + final["notebooks"]
        if item["is_latest_registration"]
    ] == [latest]
    assert len(first["notebooks"]) == 25 and len(final["notebooks"]) == 5
    assert not final["next_after"] and final["after"] == first["next_after"]
    assert {
        item["notebook_id"] for item in first["notebooks"] + final["notebooks"]
    } == ids
    assert snapshot(root) == before
    for invalid in ("../tokens", "/tmp", "not-a-returned-cursor"):
        with pytest.raises(ValueError):
            notebook_history(project, invalid)


def test_latest_registration_uses_saved_pointer_not_catalog_order(tmp_path):
    root = tmp_path.resolve()
    project, value = setup(root)
    ids = [register(root, project, value, f"saved-copy-{i}", [0]) for i in range(3)]
    # A valid saved bookmark deliberately points to the middle catalog identity.
    selected = sorted(ids)[1]
    saved = catalog.get(project, selected)["bookmark"]
    remember_notebook(
        root,
        project,
        saved["destination"],
        saved["recorded_replay_status"],
        binding=saved["binding"],
        label=saved["label"],
    )
    before = snapshot(root)
    history = notebook_history(project)
    assert history["latest_notebook_id"] == selected
    assert [
        item["notebook_id"]
        for item in history["notebooks"]
        if item["is_latest_registration"]
    ] == [selected]
    assert "Latest registered" in narrative(history)
    current = notebook_artifacts(root, project)
    assert current["notebook_id"] == selected
    assert current["selection_origin"] == "latest_registered"
    assert current["saved_bytes_unchanged"]
    explicit = notebook_artifacts(root, project, ids[0])
    assert explicit["selection_origin"] == "explicit_copy"
    assert snapshot(root) == before


@pytest.mark.parametrize("case", ["missing", "invalid_json", "schema", "symlink"])
def test_unknown_latest_never_promotes_an_archived_copy(tmp_path, case):
    root = tmp_path.resolve()
    project, value = setup(root)
    first = register(root, project, value, "older", [0])
    second = register(root, project, value, "newer", [0])
    path = project / ".frequensolve/workflow/latest-notebook.json"
    path.unlink()
    if case == "invalid_json":
        path.write_text("not JSON")
    elif case == "schema":
        path.write_text('{"destination":"newer","files":[]}')
    elif case == "symlink":
        (root / "private.json").write_text('{"private":"do not read"}')
        path.symlink_to(root / "private.json")
    before = snapshot(root)
    history = notebook_history(project)
    assert history["latest_notebook_id"] is None
    assert not any(item["is_latest_registration"] for item in history["notebooks"])
    assert {item["notebook_id"] for item in history["notebooks"]} == {first, second}
    assert history["unreadable_entries"] == (0 if case == "missing" else 1)
    assert "could not be confirmed" in narrative(history)
    assert snapshot(root) == before


def test_changed_latest_files_do_not_fall_back_to_an_older_copy(tmp_path):
    root = tmp_path.resolve()
    project, value = setup(root)
    register(root, project, value, "older", [0])
    latest = register(root, project, value, "latest", [0])
    (root / "latest/analysis.ipynb").write_text("changed")
    before = snapshot(root)
    result = notebook_artifacts(root, project)
    assert result["notebook_id"] == latest
    assert result["selection_origin"] == "latest_registered"
    assert not result["saved_bytes_unchanged"] and not result["artifacts"]
    assert snapshot(root) == before


def test_mcp_latest_read_needs_no_copy_id_cloud_or_kernel(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    project, value = setup(root)
    register(root, project, value, "older", [0, 1])
    latest = register(root, project, value, "latest", [0, 1])

    class Reader:
        def call(self, *_):
            pytest.fail("Local bookmark recovery called Cloud")

        def journal(self, *_):
            pytest.fail("Read created a mutation journal")

    monkeypatch.setattr(
        "frequensolve.workflow.study.replay_study_notebook",
        lambda *_: pytest.fail("Read ran a kernel"),
    )
    before = snapshot(root)

    async def read():
        async with Client(
            build_server(Reader(), {"root": root}, "https://example.test"), cache=None
        ) as client:
            reply = await client.call_tool(
                "workflow_workspace",
                {"request": {"action": "artifacts", "local_project": "project"}},
            )
            assert reply.structured_content["ok"]
            return (
                json.loads(reply.structured_content["payload"]),
                reply.content[0].text,
            )

    result, text = anyio.run(read)
    assert result["notebook_id"] == latest and result["saved_bytes_unchanged"]
    assert result["selection_origin"] == "latest_registered"
    assert all("/latest/" in item["path"] for item in result["artifacts"])
    assert "latest registered copy" in text and "Editable notebook" in text
    assert not result["paid_request_sent"] and not result["service_attestation"]
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "case", ["symlink", "traversal", "oversized", "changed_identity"]
)
def test_bad_catalog_entries_remain_unknown_and_do_not_hide_safe_choices(
    tmp_path, case
):
    root = tmp_path.resolve()
    project, value = setup(root)
    earlier = register(root, project, value, "earlier", [0])
    latest = register(root, project, value, "later", [1])
    path = project / f".frequensolve/workflow/notebook-{earlier}.json"
    if case == "symlink":
        path.unlink()
        (root / "private.json").write_text('{"private":"must not be read"}')
        path.symlink_to(root / "private.json")
    elif case == "oversized":
        path.write_text("x" * (catalog.MAX_METADATA_BYTES + 129))
    else:
        payload = json.loads(path.read_text())
        payload["bookmark"]["destination"] = (
            "../escape" if case == "traversal" else "other"
        )
        if case == "traversal":
            path.unlink()
            path = path.with_name(
                "notebook-" + catalog.bookmark_id(payload["bookmark"]) + ".json"
            )
        path.write_text(json.dumps(payload))
    history = notebook_history(project)
    assert not history["history_complete"] and history["unreadable_entries"] == 1
    assert [item["notebook_id"] for item in history["notebooks"]] == [latest]
    assert "must not be read" not in json.dumps(history)
    result = recover(
        root,
        project,
        value,
        [0],
        None,
        lambda _: pytest.fail(
            "Unconfirmed history must not inspect result identity yet"
        ),
    )
    assert result["history_incomplete"] and not result["artifacts"]


def test_automatic_bounds_are_unknown_and_selected_reads_remain_bounded(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    project, value = setup(root)
    ids = [register(root, project, value, f"copy-{i}", [0]) for i in range(3)]
    monkeypatch.setattr(catalog, "MAX_AUTOMATIC_RECORDS", 2)
    result = recover(
        root,
        project,
        value,
        [0],
        None,
        lambda _: pytest.fail("Incomplete lookup must not infer uniqueness"),
    )
    assert result["history_incomplete"] and not result["artifacts"]
    selected = recover(root, project, value, [0], ids[0], lambda _: manifest(value))
    assert selected["context_match"]["status"] == "match"
    monkeypatch.setattr(catalog, "MAX_DIRECTORY_ENTRIES", 1)
    with pytest.raises(ValueError, match="directory limit"):
        notebook_history(project)


def test_concurrent_registrations_keep_each_bookmark_once(tmp_path):
    root = tmp_path.resolve()
    project, value = setup(root)
    for folder in ["one", "two"]:
        write_local(root, folder + "/analysis.ipynb", b"synthetic")
    binding = context_binding(
        value["plan"], value["study"], [0], result_digest="sha256:" + "3" * 64
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                remember_notebook, root, project, folder, "complete", binding=binding
            )
            for folder in ["one", "two"]
        ]
        for future in futures:
            future.result()
    records = catalog.candidates(project)
    assert records["history_complete"] and len(records["records"]) == 2
    assert {entry["bookmark"]["destination"] for entry in records["records"]} == {
        "one",
        "two",
    }


def test_legacy_latest_is_archived_when_a_new_notebook_is_registered(tmp_path):
    root = tmp_path.resolve()
    project, value = setup(root)
    write_local(root, "legacy/analysis.ipynb", b"legacy synthetic")
    saved = {
        "destination": "legacy",
        "recorded_replay_status": "complete",
        "files": [
            {
                "name": "analysis.ipynb",
                "bytes": 16,
                "sha256": hashlib.sha256(b"legacy synthetic").hexdigest(),
            }
        ],
    }
    write_local(
        project, ".frequensolve/workflow/latest-notebook.json", canonical(saved)
    )
    legacy = catalog.bookmark_id(saved)
    register(root, project, value, "new", [1])
    assert legacy in {
        entry["notebook_id"] for entry in notebook_history(project)["notebooks"]
    }
    result = notebook_artifacts(root, project, legacy)
    assert result["binding"] is None and result["artifacts"][0]["path"].endswith(
        "legacy/analysis.ipynb"
    )


def test_fresh_mcp_clients_reopen_earlier_context_without_ids_from_customer(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    project, value = setup(root)
    earlier = register(root, project, value, "earlier-2hz", [0])
    register(root, project, value, "later-20hz", [1])
    calls = []

    class Reader:
        def call(self, operation, _args):
            calls.append(operation)
            if operation == "plan_get":
                return value["plan"]
            if operation == "study_get":
                return {"study": value["study"]}
            assert operation == "plan_results"
            return manifest(value)

        def journal(self, *_):
            pytest.fail("Read created a Cloud mutation journal")

    monkeypatch.setattr(
        "frequensolve.workflow.server.credit_scan", lambda *_: value["credits"]
    )
    monkeypatch.setattr(
        "frequensolve.workflow.study.replay_study_notebook",
        lambda *_: pytest.fail("Read ran a kernel"),
    )
    before = snapshot(root)

    async def read(tool, request):
        async with Client(
            build_server(Reader(), {"root": root}, "https://example.test"), cache=None
        ) as client:
            reply = await client.call_tool(tool, {"request": request})
            assert reply.structured_content["ok"]
            return (
                json.loads(reply.structured_content["payload"]),
                reply.content[0].text,
            )

    listed, text = anyio.run(
        read, "workflow_workspace", {"action": "notebooks", "local_project": "project"}
    )
    assert len(listed["notebooks"]) == 2 and "earlier-2hz" in text
    request = {
        "root_id": "root",
        "local_project": "project",
        "plan_id": "fake-plan",
        "chapter": "reproduce",
        "frequency_indices": [0],
    }
    first, text = anyio.run(read, "workflow_present", request)
    again, _ = anyio.run(read, "workflow_present", request)
    assert first == again and first["notebook"]["notebook_id"] == earlier
    assert (
        first["next_action"] == "open_saved_notebook"
        and "Open editable notebook" in text
    )
    assert set(calls) == {"plan_get", "study_get", "plan_results"}
    assert snapshot(root) == before


def test_mcp_choice_selection_and_invalid_ids_stay_read_only(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    project, value = setup(root)
    first = register(root, project, value, "first-copy", [0])
    register(root, project, value, "second-copy", [0])
    register(root, project, value, "later-20hz", [1])
    calls = []

    class Reader:
        def call(self, operation, _args):
            calls.append(operation)
            if operation == "plan_get":
                return value["plan"]
            if operation == "study_get":
                return {"study": value["study"]}
            assert operation == "plan_results"
            return manifest(value)

        def journal(self, *_):
            pytest.fail("Read recovered a mutation")

    monkeypatch.setattr(
        "frequensolve.workflow.server.credit_scan", lambda *_: value["credits"]
    )
    before = snapshot(root)

    async def run():
        async with Client(
            build_server(Reader(), {"root": root}, "https://example.test"), cache=None
        ) as client:
            request = {
                "root_id": "root",
                "local_project": "project",
                "plan_id": "fake-plan",
                "chapter": "reproduce",
                "frequency_indices": [0],
            }
            reply = await client.call_tool("workflow_present", {"request": request})
            choice = json.loads(reply.structured_content["payload"])
            assert choice["next_action"] == "choose_saved_notebook"
            assert (
                not choice["notebook"]["artifacts"]
                and "Choose a copy" in reply.content[0].text
            )
            assert (
                "fake-plan" not in reply.content[0].text
                and "fake-run" not in reply.content[0].text
            )
            picked = await client.call_tool(
                "workflow_present", {"request": {**request, "notebook_id": first}}
            )
            book = json.loads(picked.structured_content["payload"])["notebook"]
            assert (
                book["notebook_id"] == first
                and book["context_match"]["status"] == "match"
            )
            local = await client.call_tool(
                "workflow_workspace",
                {
                    "request": {
                        "action": "artifacts",
                        "local_project": "project",
                        "notebook_id": first,
                    }
                },
            )
            assert (
                json.loads(local.structured_content["payload"])["notebook_id"] == first
            )
            for changes in [
                {"notebook_id": "../tokens"},
                {"notebook_id": first, "chapter": "model"},
            ]:
                invalid = await client.call_tool(
                    "workflow_present", {"request": {**request, **changes}}
                )
                assert invalid.is_error

    anyio.run(run)
    assert (
        set(calls) == {"plan_get", "study_get", "plan_results"}
        and snapshot(root) == before
    )
