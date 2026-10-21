"""Find unfinished local analysis without old client context or new execution."""

import hashlib
import json
import os
from urllib.parse import urlencode

import anyio
import pytest
from mcp import Client

from frequensolve.workflow import notebook_navigation as navigation
from frequensolve.workflow import study
from frequensolve.workflow.client import create_project, write_local
from frequensolve.workflow.notebook_attempt import MAX_RECORD_BYTES, perform
from frequensolve.workflow.notebook_binding import context_binding
from frequensolve.workflow.server import StudyRequest, build_server
from frequensolve.workflow.storage import PrivateStore, canonical
from frequensolve.workflow.workspace import remember_notebook
from tests.test_workflow_notebook_recovery import state


def snapshot(root):
    return {
        str(p.relative_to(root)): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def saved_work(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    identity = create_project(root, "project", "A returning customer's study")
    project = root / "project"
    owner = {k: identity[k] for k in ("project_id", "project_key")}
    value = state()
    value["plan"]["head"]["owner"] = owner

    def create_attempt(
        folder,
        status="pending",
        *,
        root_id="workspace",
        indices=None,
        phase="replaying",
    ):
        request = StudyRequest(
            action="notebook",
            root_id=root_id,
            local_project="project",
            plan_id="fake-plan",
            frequency_indices=indices or [0, 1],
            bundle="synthetic.zip",
            destination=folder,
        ).model_dump(exclude_none=True)
        binding = context_binding(
            value["plan"],
            value["study"],
            request["frequency_indices"],
            result_digest="sha256:" + "3" * 64,
        )
        desired = {**request, "project_identity": owner}

        def generate(mark):
            if phase == "preparing":
                raise SystemExit("Synthetic interruption before context")
            mark("context", binding)
            for name in ("analysis.ipynb", "study.json"):
                write_local(
                    root,
                    folder + "/" + name,
                    b"Explicit synthetic saved bytes; no real kernel",
                )
            mark("authored")
            mark("replaying")
            if status == "pending":
                raise SystemExit("Synthetic interruption before kernel")
            for name in ("executed.ipynb", "report.html", "report.json", "replay.json"):
                write_local(root, folder + "/" + name, b"Synthetic output fixture")
            book = remember_notebook(
                root,
                project,
                folder,
                "complete",
                binding=binding,
                label="Synthetic fixture",
            )
            mark("registered", notebook_id=book)
            return {
                "relative_path": folder + "/analysis.ipynb",
                "notebook_path": str(root / folder / "analysis.ipynb"),
                "report_path": folder + "/report.html",
                "replay_status": "complete",
                "binding": binding,
                "notebook_id": book,
                "study_url": "https://example.test/study?"
                + urlencode(
                    {"project_key": owner["project_key"], "plan_id": "fake-plan"}
                ),
                "assurance": "synthetic fixture",
                "paid_request_sent": False,
            }

        if status == "pending":
            with pytest.raises(SystemExit):
                perform(root, project, desired, generate)
        else:
            perform(root, project, desired, generate)
        key = hashlib.sha256(canonical({"destination": folder})).hexdigest()
        return key, request

    class NoCloud:
        def call(self, *_):
            pytest.fail("Local attempt navigation called Cloud")

        def journal(self, *_):
            pytest.fail("Local attempt navigation opened a mutation journal")

    def inspect(request):
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
                    tool.annotations.read_only_hint
                    and tool.annotations.idempotent_hint
                    and not tool.annotations.open_world_hint
                )
                return await client.call_tool(
                    "workflow_workspace", {"request": request}
                )

        with monkeypatch.context() as guard:
            guard.setattr(
                PrivateStore,
                "locked",
                lambda *_args, **_kwargs: pytest.fail(
                    "Read created or acquired a writer lock"
                ),
            )
            guard.setattr(
                study,
                "replay_study_notebook",
                lambda *_: pytest.fail("Read ran a kernel"),
            )
            return anyio.run(run)

    return root, project, create_attempt, inspect


def decoded(reply):
    assert not reply.is_error and reply.structured_content["ok"]
    return json.loads(reply.structured_content["payload"])


def request(action, **kwargs):
    return {
        "action": action,
        "root_id": "workspace",
        "local_project": "project",
        **kwargs,
    }


def test_fresh_client_finds_unregistered_attempt_from_named_project(saved_work):
    root, _, create, inspect = saved_work
    key, original = create("interrupted-analysis")
    before = snapshot(root)
    projects = decoded(inspect({}))["projects"]
    assert (
        len(projects) == 1
        and projects[0]["display_name"] == "A returning customer's study"
    )
    base = {k: projects[0][k] for k in ("root_id", "local_project")}
    listed = decoded(inspect({**base, "action": "attempts"}))
    assert listed["history_complete"] and len(listed["notebook_attempts"]) == 1
    selection = listed["notebook_attempts"][0]
    assert selection["attempt_key"] == key and not selection["saved_bytes_checked"]
    assert (
        selection["recorded_status"] == "pending"
        and selection["frequency_label"] == "2 Hz, 20 Hz"
    )
    reply = inspect(
        {**base, "action": "attempt", "attempt_key": selection["attempt_key"]}
    )
    attempt = decoded(reply)["notebook_attempt"]
    assert attempt["resume_request"] == original
    assert (
        attempt["recorded_status"] == "pending"
        and attempt["generation_status"] == "unconfirmed"
    )
    assert (
        attempt["current_process_state"] == "unconfirmed"
        and attempt["kernel_status"] == "unknown"
    )
    assert (
        attempt["editable_notebook_unchanged"]
        and attempt["read_only"]
        and not attempt["new_kernel_started"]
    )
    assert "Inspect the saved editable notebook" in reply.content[0].text
    assert "Current process and kernel state are unconfirmed" in reply.content[0].text
    assert (
        "Local analysis requested" in reply.content[0].text
        and "Running local analysis" not in reply.content[0].text
    )
    assert all(
        internal not in reply.content[0].text
        for internal in (key, attempt["attempt_id"], "fake-plan", "fake-run")
    )
    assert decoded(inspect(request("notebooks")))["notebooks"] == []
    assert snapshot(root) == before


def test_completed_attempt_inspection_checks_saved_files_without_current_study_claim(
    saved_work,
):
    root, _, create, inspect = saved_work
    key, original = create("complete-analysis", "complete")
    before = snapshot(root)
    reply = inspect(request("attempt", attempt_key=key))
    attempt = decoded(reply)["notebook_attempt"]
    assert (
        attempt["generation_status"] == "complete" and attempt["saved_bytes_unchanged"]
    )
    assert attempt["resume_request"] == original and "study_url" not in attempt
    assert not any(
        attempt[k]
        for k in (
            "new_kernel_started",
            "paid_request_sent",
            "current_study_checked",
            "cloud_ownership_checked",
            "current_environment_checked",
            "service_attestation",
        )
    )
    assert attempt["evidence_origin"] == "local_client_reported"
    assert "Open static report" in reply.content[0].text
    assert snapshot(root) == before


@pytest.mark.parametrize("file", ["analysis.ipynb", "study.json", "report.html"])
def test_changed_or_missing_bundle_cannot_be_recovered_as_current_success(
    saved_work, file
):
    root, _, create, inspect = saved_work
    key, _ = create("changed-analysis", "complete")
    (root / "changed-analysis" / file).unlink()
    before = snapshot(root)
    reply = inspect(request("attempt", attempt_key=key))
    attempt = decoded(reply)["notebook_attempt"]
    assert attempt["generation_status"] != "complete" and "report_path" not in attempt
    assert attempt["recorded_status"] == "complete" and attempt["read_only"]
    assert "Open static report" not in reply.content[0].text
    assert bool(attempt.get("editable_notebook_unchanged")) == (
        file != "analysis.ipynb"
    )
    assert snapshot(root) == before


def test_pages_preserve_choices_and_uncertainty_across_concurrent_additions(saved_work):
    root, _, create, inspect = saved_work
    ids = {create(f"saved-{i:02}")[0] for i in range(30)}
    before = snapshot(root)
    first = decoded(inspect(request("attempts")))
    assert len(first["notebook_attempts"]) == 25 and first["next_after"]
    assert snapshot(root) == before
    folder = next(
        f"added-{i}"
        for i in range(100)
        if hashlib.sha256(canonical({"destination": f"added-{i}"})).hexdigest()
        < first["next_after"]
    )
    added, _ = create(folder)
    before = snapshot(root)
    final = decoded(inspect(request("attempts", after=first["next_after"])))
    refresh = decoded(inspect(request("attempts")))
    retained = (
        first["notebook_attempts"]
        + final["notebook_attempts"]
        + refresh["notebook_attempts"]
    )
    assert {item["attempt_key"] for item in retained} == ids | {added}
    assert not first["history_complete"] and not final["history_complete"]
    assert not final["next_after"] and final["after"] == first["next_after"]
    assert "Records may change between pages" in final["pagination"]
    assert snapshot(root) == before


def test_empty_unreadable_page_does_not_hide_a_later_valid_attempt(saved_work):
    root, project, create, inspect = saved_work
    keys = sorted(create(f"entry-{i:02}")[0] for i in range(26))
    for key in keys[:25]:
        path = project / f".frequensolve/workflow/notebook-attempt-{key}.json"
        saved = json.loads(path.read_text())
        saved["unsupported_field"] = "unreadable local entry"
        path.write_text(json.dumps(saved))
    before = snapshot(root)
    first_reply = inspect(request("attempts"))
    first = decoded(first_reply)
    assert not first["notebook_attempts"] and first["unreadable_entries"] == 25
    assert first["next_after"] and not first["history_complete"]
    assert "history remains unconfirmed" in first_reply.content[0].text
    final = decoded(inspect(request("attempts", after=first["next_after"])))
    assert [item["attempt_key"] for item in final["notebook_attempts"]] == [keys[-1]]
    assert not final["next_after"] and not final["history_complete"]
    assert snapshot(root) == before


def test_pre_context_checkpoint_does_not_invent_frequency_or_artifact_evidence(
    saved_work,
):
    root, _, create, inspect = saved_work
    key, original = create("early-interruption", phase="preparing")
    before = snapshot(root)
    page = decoded(inspect(request("attempts")))
    assert page["notebook_attempts"][0]["frequency_label"] == "Inputs not yet bound"
    attempt = decoded(inspect(request("attempt", attempt_key=key)))["notebook_attempt"]
    assert (
        attempt["last_recorded_phase"] == "preparing"
        and attempt["resume_request"] == original
    )
    assert "notebook_path" not in attempt and "binding" not in attempt
    assert not attempt["current_study_checked"] and not attempt["new_kernel_started"]
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "case",
    [
        "extra_field",
        "request_field",
        "foreign_owner",
        "foreign_binding",
        "frequency_binding",
        "wrong_key",
        "traversal",
        "oversized",
        "symlink",
        "fifo",
    ],
)
def test_invalid_or_foreign_attempts_remain_unknown_without_exposing_contents(
    saved_work, case
):
    root, project, create, inspect = saved_work
    key, _ = create("safe-analysis")
    path = project / f".frequensolve/workflow/notebook-attempt-{key}.json"
    if case == "oversized":
        path.write_text("x" * (MAX_RECORD_BYTES + 1))
    elif case == "symlink":
        (root / "private.json").write_text(
            '{"private":"secret fixture must not be exposed"}'
        )
        path.unlink()
        path.symlink_to(root / "private.json")
    elif case == "fifo":
        path.unlink()
        os.mkfifo(path, mode=0o600)
    else:
        saved = json.loads(path.read_text())
        if case == "extra_field":
            saved["access_token"] = "secret fixture must not be exposed"
        if case == "request_field":
            saved["request"]["account_id"] = "secret fixture must not be exposed"
        if case == "foreign_owner":
            saved["request"]["project_identity"]["project_id"] = "other-project"
        if case == "foreign_binding":
            saved["binding"]["project_id"] = "other-project"
        if case == "frequency_binding":
            saved["request"]["frequency_indices"] = [0]
        if case == "traversal":
            saved["request"]["destination"] = "../escape"
        if case == "wrong_key":
            path.unlink()
            key = "0" * 64
            path = path.with_name("notebook-attempt-" + key + ".json")
        path.write_text(json.dumps(saved))
    before = snapshot(root)
    page = decoded(inspect(request("attempts")))
    assert (
        not page["notebook_attempts"]
        and page["unreadable_entries"] == 1
        and not page["history_complete"]
    )
    reply = inspect(request("attempt", attempt_key=key))
    assert not reply.structured_content["ok"]
    assert "secret fixture" not in json.dumps([page, reply.structured_content])
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "case", ["unknown_response_field", "signed_url", "foreign_host"]
)
def test_record_links_and_unknown_response_fields_are_not_reflected(saved_work, case):
    root, project, create, inspect = saved_work
    key, _ = create("complete-analysis", "complete")
    path = project / f".frequensolve/workflow/notebook-attempt-{key}.json"
    saved = json.loads(path.read_text())
    if case == "unknown_response_field":
        saved["response"]["access_token"] = "secret fixture"
    elif case == "signed_url":
        saved["response"]["study_url"] += "&X-Amz-Signature=secret-fixture"
    else:
        saved["response"]["study_url"] = saved["response"]["study_url"].replace(
            "example.test", "untrusted.invalid"
        )
    path.write_text(json.dumps(saved))
    before = snapshot(root)
    reply = inspect(request("attempt", attempt_key=key))
    value = decoded(reply)
    assert "study_url" not in value["notebook_attempt"]
    assert not any(
        token in json.dumps(value) + reply.content[0].text
        for token in ("secret fixture", "X-Amz-Signature", "untrusted.invalid")
    )
    if case != "foreign_host":
        assert value["notebook_attempt"]["generation_status"] != "complete"
    assert snapshot(root) == before


@pytest.mark.parametrize("context", ["root_name", "project_folder"])
def test_moved_context_is_not_silently_rewritten(saved_work, context):
    root, project, create, inspect = saved_work
    key, original = create(
        "complete-analysis",
        "complete",
        root_id="old-root" if context == "root_name" else "workspace",
    )
    selected_project = "project"
    if context == "project_folder":
        target = project.with_name("renamed-project")
        project.rename(target)
        project = target
        selected_project = target.name
    before = snapshot(root)
    page = decoded(inspect(request("attempts", local_project=selected_project)))
    assert page["notebook_attempts"][0]["request_context"] == "moved_context"
    reply = inspect(request("attempt", attempt_key=key, local_project=selected_project))
    attempt = decoded(reply)["notebook_attempt"]
    assert attempt["resume_request"] is None and attempt["saved_bytes_unchanged"]
    assert "identities have not been rewritten" in reply.content[0].text
    path = project / f".frequensolve/workflow/notebook-attempt-{key}.json"
    assert json.loads(path.read_text())["request"]["root_id"] == original["root_id"]
    assert (
        json.loads(path.read_text())["request"]["local_project"]
        == original["local_project"]
    )
    assert snapshot(root) == before


def test_metadata_limit_does_not_prevent_one_known_selection_read(
    saved_work, monkeypatch
):
    root, _, create, inspect = saved_work
    key, _ = create("known-analysis")
    before = snapshot(root)
    monkeypatch.setattr(navigation, "MAX_DIRECTORY_ENTRIES", 1)
    page = inspect(request("attempts"))
    assert not page.structured_content["ok"]
    selected = decoded(inspect(request("attempt", attempt_key=key)))
    assert selected["notebook_attempt"]["attempt_key"] == key
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "args",
    [
        request("attempt"),
        request("attempt", attempt_key="../tokens"),
        request("attempt", attempt_key="0" * 64, after=""),
        request("attempts", attempt_key="0" * 64),
        request("attempt", attempt_key="0" * 64, notebook_id="1" * 64),
        request("notebooks", attempt_key="0" * 64),
    ],
)
def test_selection_and_cursor_roles_are_enforced_before_reading(saved_work, args):
    root, _, _, inspect = saved_work
    before = snapshot(root)
    assert inspect(args).is_error
    assert snapshot(root) == before
