"""A returning customer's notebook must belong to the exact selected study."""

import json
from copy import deepcopy

import anyio
import pytest
from mcp import Client

from frequensolve.workflow.client import create_project, write_local
from frequensolve.workflow.notebook_binding import (
    context_binding,
    current_result_digest,
    match,
    validate,
)
from frequensolve.workflow.server import build_server
from frequensolve.workflow.workspace import notebook_artifacts, remember_notebook
from tests.test_workflow_presentation import view
from tests.test_workflow_study import data


def state():
    value = view()
    value["plan"]["head"]["owner"] = {
        "project_id": "fake-project-id",
        "project_key": "fake-project",
    }
    return value


def binding(value=None):
    value = value or state()
    return context_binding(
        value["plan"], value["study"], [0, 1], result_digest="sha256:" + "3" * 64
    )


def manifest(value):
    head = value["plan"]["head"]
    return {
        "results": {
            "plan_id": head["plan_id"],
            "execution_id": head["execution_id"],
            "plan_revision": head["revision"],
            "plan_digest": value["plan"]["plan"]["digest"],
            "digest": "sha256:" + "3" * 64,
        }
    }


def files(root, project, context, status="complete", *, bound=True):
    for name in (
        "analysis.ipynb",
        "executed.ipynb",
        "report.html",
        "study.json",
        "snapshot-0.json",
        "replay.json",
    ):
        write_local(root, "notebook/" + name, b"synthetic fixture")
    remember_notebook(
        root, project, "notebook", status, binding=binding(context) if bound else None
    )


def test_binding_is_an_exact_copied_context_not_an_execution_attestation():
    value = state()
    original = deepcopy(value)
    saved = binding(value)
    assert saved["plan_id"] == "fake-plan" and saved["plan_revision"] == 3
    assert saved["execution_id"] == "fake-run"
    assert saved["frequency_indices"] == [0, 1]
    assert saved["frequency_coordinates"] == [["2", "0"], ["20", "0"]]
    value["plan"]["scientific_inputs"]["frequencies"][0][0] = "19"
    assert saved["frequency_coordinates"][0][0] == "2"
    assert match(saved, binding(original))["status"] == "match"


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("project_id", "other-project"),
        ("project_key", "other-prefix"),
        ("plan_id", "other-plan"),
        ("plan_revision", 4),
        ("execution_id", "other-run"),
        ("input_digest", "sha256:" + "4" * 64),
        ("plan_digest", "sha256:" + "4" * 64),
        ("result_digest", "sha256:" + "4" * 64),
        ("study_digest", "sha256:" + "4" * 64),
        ("frequency_indices", [1, 0]),
        ("frequency_coordinates", [["2", "0"], ["20", "1"]]),
    ],
)
def test_each_identity_dimension_can_invalidate_the_current_notebook(
    field, replacement
):
    saved = binding()
    expected = deepcopy(saved)
    expected[field] = replacement
    result = match(saved, expected)
    assert result["status"] == "mismatch" and field in result["mismatch_fields"]


def test_unknown_results_old_bookmarks_and_unsupported_versions_remain_unknown():
    saved = binding()
    expected = deepcopy(saved)
    expected["result_digest"] = None
    assert match(saved, expected)["status"] == "unconfirmed"
    assert match(saved, None)["status"] == "unconfirmed"
    assert match(None, saved)["status"] == "unbound"
    old = {**saved, "version": "future-version"}
    assert match(old, expected)["status"] == "unbound"
    with pytest.raises(ValueError):
        validate(expected)
    malformed = deepcopy(saved)
    malformed["frequency_coordinates"][0][0] = float("nan")
    with pytest.raises(ValueError):
        validate(malformed)
    malformed = deepcopy(saved)
    malformed["frequency_coordinates"][0][0] = 10**1000
    with pytest.raises(ValueError, match="frequency coordinates"):
        validate(malformed)
    assert match(malformed, saved)["status"] == "unbound"


@pytest.mark.parametrize(
    "field", ["plan_id", "execution_id", "plan_revision", "plan_digest", "digest"]
)
def test_result_metadata_must_match_the_current_exact_run(field):
    value = state()
    response = manifest(value)
    assert current_result_digest(response, value["plan"]) == "sha256:" + "3" * 64
    response["results"][field] = "different"
    assert current_result_digest(response, value["plan"]) is None


@pytest.mark.parametrize(
    "case",
    [
        "match",
        "missing",
        "legacy",
        "changed",
        "incomplete",
        "other_run",
        "other_input",
        "other_results",
        "other_question",
        "frequency",
        "manifest_timeout",
        "corrupt",
    ],
)
def test_reproduce_reopens_only_matching_notebooks_without_writes_or_kernel(
    tmp_path, monkeypatch, case
):
    root = tmp_path.resolve()
    created = create_project(root, "study", "Acoustic study")
    value = state()
    value["plan"]["head"]["owner"] = {
        k: created[k] for k in ("project_id", "project_key")
    }
    if case != "missing":
        files(
            root,
            root / "study",
            value,
            "incomplete" if case == "incomplete" else "complete",
            bound=case != "legacy",
        )
    if case == "changed":
        (root / "notebook/snapshot-0.json").write_text("changed")
    if case == "other_run":
        value["plan"]["head"]["execution_id"] = "different-run"
    if case == "other_input":
        value["plan"]["input_digest"] = "sha256:" + "5" * 64
    if case == "other_question":
        value["study"]["question"] = "A different scientific question?"
    if case == "corrupt":
        (root / "study/.frequensolve/workflow/latest-notebook.json").write_text(
            "{broken"
        )
    response = manifest(value)
    if case == "other_results":
        response["results"]["digest"] = "sha256:" + "5" * 64
    calls = []

    class Reader:
        def call(self, operation, args):
            calls.append(operation)
            if operation == "plan_get":
                return value["plan"]
            if operation == "study_get":
                return {"study": value["study"]}
            assert operation == "plan_results", "Reopening tried a mutation"
            if case == "manifest_timeout":
                raise TimeoutError("sensitive provider detail")
            return response

        def journal(self, *_):
            pytest.fail("A read must not recover a mutation")

    monkeypatch.setattr(
        "frequensolve.workflow.server.credit_scan", lambda *_: value["credits"]
    )
    monkeypatch.setattr(
        "frequensolve.workflow.study.replay_study_notebook",
        lambda *_: pytest.fail("Reopening ran a kernel"),
    )

    async def read():
        async with Client(
            build_server(Reader(), {"root": root}, "https://example.test"), cache=None
        ) as client:
            reply = await client.call_tool(
                "workflow_present",
                {
                    "request": {
                        "root_id": "root",
                        "local_project": "study",
                        "plan_id": "fake-plan",
                        "chapter": "reproduce",
                        "frequency_indices": [1] if case == "frequency" else [0, 1],
                    }
                },
            )
            assert reply.structured_content["ok"]
            return (
                json.loads(reply.structured_content["payload"]),
                reply.content[0].text,
            )

    result, text = anyio.run(read)
    again, _ = anyio.run(read)
    assert again == result
    assert not result["paid_request_sent"]
    assert "sensitive provider detail" not in text
    book = result["notebook"]
    if case == "match":
        assert result["next_action"] == "open_saved_notebook"
        assert len(book["artifacts"]) == 3
        assert "Open editable notebook" in text and "Open static report" in text
        assert "Recorded local replay: **complete**" in text
        assert (
            not book["current_environment_checked"] and not book["service_attestation"]
        )
    elif case == "incomplete":
        assert result["next_action"] == "inspect_incomplete_notebook"
        assert [a["name"] for a in book["artifacts"]] == ["analysis.ipynb"]
        assert "Recorded replay: **incomplete**" in text
        assert "Open static report" not in text
    elif case == "missing":
        assert result["next_action"] == "create_or_verify_analysis_notebook"
    elif case == "corrupt":
        assert not book["artifacts"] and book["history_incomplete"]
        assert result["next_action"] == "choose_saved_notebook"
        assert "history could not be checked completely" in text
    else:
        assert not book["artifacts"]
        assert result["next_action"] == "inspect_notebook_context"
        assert "Open editable notebook" not in text
    if case in ("manifest_timeout", "corrupt"):
        assert book["context_match"]["status"] == "unconfirmed"
    if case in (
        "missing",
        "legacy",
        "changed",
        "other_run",
        "other_input",
        "other_question",
        "frequency",
        "corrupt",
    ):
        assert "plan_results" not in calls
    if case == "legacy":
        assert len(notebook_artifacts(root, root / "study")["artifacts"]) == 3


@pytest.mark.parametrize(
    "case", ["complete", "incomplete", "wrong_owner", "wrong_plan", "mixed_results"]
)
def test_notebook_creation_saves_the_verified_context_for_a_fresh_client(
    tmp_path, monkeypatch, case
):
    """Exercise generation/bookmark wiring; the kernel is a deterministic fixture."""
    root = tmp_path.resolve()
    created = create_project(root, "study", "Acoustic study")
    value = state()
    value["plan"]["head"]["owner"] = {
        k: created[k] for k in ("project_id", "project_key")
    }
    if case == "wrong_owner":
        value["plan"]["head"]["owner"]["project_id"] = "other-project"
    if case == "wrong_plan":
        value["plan"]["head"]["plan_id"] = "other-plan"
    _, snapshots, _ = data()
    for index, snapshot in enumerate(snapshots):
        snapshot["evidence"]["result_digest"] = (
            "sha256:" + ("4" if case == "mixed_results" and index else "3") * 64
        )
    calls, generated, replays = [], [], []

    class Reader:
        def call(self, operation, arguments):
            calls.append(operation)
            if operation == "plan_get":
                return value["plan"]
            if operation == "study_get":
                return {"study": value["study"]}
            assert operation == "plan_results", "Notebook creation tried a mutation"
            return manifest(value)

        def journal(self, *_):
            pytest.fail("Local analysis must not journal a Cloud mutation")

    def create(path, context, values, references):
        assert context["plan"] == value["plan"]
        assert values == snapshots and references == [None, None]
        generated.append(context)
        path.mkdir()
        (path / "analysis.ipynb").write_text("synthetic fixture")
        (path / "study.json").write_text(json.dumps(context))
        return path / "analysis.ipynb"

    def replay(path):
        replays.append(path)
        (path.parent / "executed.ipynb").write_text("synthetic fixture")
        (path.parent / "replay.json").write_text(
            json.dumps({"status": case, "origin": "synthetic_test"})
        )
        if case == "incomplete":
            raise RuntimeError("synthetic interrupted kernel")
        (path.parent / "report.html").write_text("synthetic fixture")
        (path.parent / "report.json").write_text('{"origin":"synthetic_test"}')
        return {"status": "complete"}

    monkeypatch.setattr(
        "frequensolve.workflow.server.credit_scan", lambda *_: value["credits"]
    )
    monkeypatch.setattr(
        "frequensolve.workflow.server.snapshot_from_bundle",
        lambda root, path, index: deepcopy(snapshots[index]),
    )
    monkeypatch.setattr("frequensolve.workflow.study.create_study_notebook", create)
    monkeypatch.setattr("frequensolve.workflow.study.replay_study_notebook", replay)

    async def call(tool, request):
        async with Client(
            build_server(Reader(), {"root": root}, "https://example.test"), cache=None
        ) as client:
            return await client.call_tool(tool, {"request": request})

    request = {
        "root_id": "root",
        "local_project": "study",
        "plan_id": "fake-plan",
    }
    result = anyio.run(
        call,
        "workflow_study",
        {
            **request,
            "action": "notebook",
            "bundle": "synthetic.zip",
            "destination": "notebook",
        },
    )
    valid = case in ("complete", "incomplete")
    assert bool(generated) == bool(replays) == valid
    assert result.structured_content["ok"] == (case == "complete")
    assert (root / "notebook").exists() == valid
    if not valid:
        assert not (root / "study/.frequensolve/workflow/latest-notebook.json").exists()
        return
    saved = notebook_artifacts(root, root / "study")
    assert saved["binding"] == binding(value)
    assert saved["recorded_replay_status"] == case
    # A new server session uses the stored identities and does not replay again.
    reopened = anyio.run(call, "workflow_present", {**request, "chapter": "reproduce"})
    payload = json.loads(reopened.structured_content["payload"])
    assert payload["notebook"]["context_match"]["status"] == "match"
    assert payload["next_action"] == (
        "open_saved_notebook" if case == "complete" else "inspect_incomplete_notebook"
    )
    assert len(replays) == 1
    assert calls == ["plan_get", "study_get", "plan_get", "study_get", "plan_results"]
