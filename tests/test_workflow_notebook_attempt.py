"""Local generation retries recover identity; they never replay uncertain work."""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import anyio
import pytest
from mcp import Client

from frequensolve.workflow import notebook_attempt, server, study
from frequensolve.workflow.client import create_project, write_local
from frequensolve.workflow.notebook_binding import context_binding
from frequensolve.workflow.workspace import notebook_history
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
def generation(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    identity = create_project(root, "project", "A customer study")
    project = root / "project"
    value = state()
    value["plan"]["head"]["owner"] = {
        k: identity[k] for k in ("project_id", "project_key")
    }
    control = {
        "failure": None,
        "reads": [],
        "kernels": [],
        "messages": [],
        "started": None,
        "release": None,
    }

    def attempts():
        return list(
            (project / ".frequensolve/workflow").glob("notebook-attempt-*.json")
        )

    class Reader:
        def call(self, operation, _arguments):
            control["reads"].append(operation)
            # Durable intent precedes even the read needed to establish context.
            pending = json.loads(attempts()[0].read_text())
            assert pending["status"] == "pending" and pending["phase"] == "preparing"
            assert (
                pending["request"]["project_identity"]["project_id"]
                == identity["project_id"]
            )
            if control["failure"] == "before_context":
                raise TimeoutError("private connection details")
            if operation == "plan_get":
                return value["plan"]
            if operation == "study_get":
                return {"study": value["study"]}
            pytest.fail("Generation requested a mutation or transfer")

        def journal(self, *_):
            pytest.fail("Local generation requested a Cloud mutation journal")

    def create(destination, *_):
        if control["failure"] == "authoring":
            raise RuntimeError("private authoring details")
        destination.mkdir()
        for name in ("analysis.ipynb", "study.json", "snapshot-0.json"):
            (destination / name).write_text("Explicit synthetic saved-byte fixture")
        return destination / "analysis.ipynb"

    def replay(path):
        control["kernels"].append(str(path))
        if control["started"]:
            control["started"].set()
            assert control["release"].wait(5)
        (path.parent / "executed.ipynb").write_text("Synthetic executed bytes")
        if control["failure"] == "kernel":
            raise RuntimeError("private kernel details")
        (path.parent / "report.html").write_text("Synthetic report")
        (path.parent / "report.json").write_text('{"origin":"synthetic_fixture"}')
        (path.parent / "replay.json").write_text(
            '{"status":"complete","origin":"synthetic_fixture"}'
        )
        return {"status": "complete", "origin": "synthetic_fixture"}

    real_write = server.write_local

    def publish(allowed, name, data):
        if control["failure"] == "publication" and name == "saved-analysis/report.html":
            raise OSError("private publication details")
        return real_write(allowed, name, data)

    monkeypatch.setattr(server, "credit_scan", lambda *_: value["credits"])
    monkeypatch.setattr(
        server,
        "snapshot_from_bundle",
        lambda *_: {"evidence": {"result_digest": "sha256:" + "3" * 64}},
    )
    monkeypatch.setattr(study, "create_study_notebook", create)
    monkeypatch.setattr(study, "replay_study_notebook", replay)
    monkeypatch.setattr(server, "write_local", publish)
    request = {
        "action": "notebook",
        "root_id": "workspace",
        "local_project": "project",
        "plan_id": "fake-plan",
        "frequency_indices": [0],
        "bundle": "synthetic.zip",
        "destination": "saved-analysis",
    }

    def invoke(changes=None):
        async def run():
            async with Client(
                server.build_server(
                    Reader(), {"workspace": root}, "https://example.test"
                ),
                cache=None,
            ) as client:
                reply = await client.call_tool(
                    "workflow_study", {"request": {**request, **(changes or {})}}
                )
                control["messages"].append(reply.content[0].text)
                return reply.structured_content

        return anyio.run(run)

    return root, project, value, control, request, attempts, invoke


def test_lost_completed_response_recovers_exact_receipt_across_fresh_clients(
    generation,
):
    root, project, _, control, _, attempts, invoke = generation
    first = invoke()
    assert first["ok"]
    original = json.loads(first["payload"])
    assert original["generation_status"] == "complete" and not original["recovered"]
    assert original["new_kernel_started"] and original["saved_bytes_unchanged"]
    assert "Your analysis notebook is ready" in control["messages"][-1]
    assert (
        "Open editable notebook" in control["messages"][-1]
        and "Open static report" in control["messages"][-1]
    )
    before = snapshot(root)
    reads = list(control["reads"])
    recovered = invoke()
    assert recovered["ok"]
    saved = json.loads(recovered["payload"])
    assert saved["attempt_id"] == original["attempt_id"]
    assert saved["notebook_id"] == original["notebook_id"]
    assert saved["binding"] == original["binding"]
    assert saved["recorded_completed_at"] == original["recorded_completed_at"]
    assert saved["relative_path"] == original["relative_path"]
    assert saved["recovered"] and not saved["new_kernel_started"]
    assert not saved["current_environment_checked"] and not saved["service_attestation"]
    assert control["reads"] == reads and len(control["kernels"]) == 1
    assert len(attempts()) == 1 and snapshot(root) == before
    assert len(notebook_history(project)["notebooks"]) == 1
    message = control["messages"][-1]
    assert (
        "Recovered your saved analysis notebook" in message
        and "did not run another kernel" in message
    )
    assert all(
        internal not in message
        for internal in (
            saved["attempt_id"],
            saved["notebook_id"],
            "fake-plan",
            "fake-run",
        )
    )


@pytest.mark.parametrize("change", ["edited", "missing"])
def test_completed_history_is_not_current_success_after_saved_bytes_change(
    generation, change
):
    root, _, _, control, _, _, invoke = generation
    original = json.loads(invoke()["payload"])
    path = root / "saved-analysis/snapshot-0.json"
    if change == "edited":
        path.write_bytes(b"Changed input")
    else:
        path.unlink()
    before = snapshot(root)
    reads = list(control["reads"])
    reply = invoke()
    evidence = json.loads(reply["payload"])
    assert not reply["ok"] and evidence["original_generation_status"] == "complete"
    assert (
        evidence["replay_status"] == "unconfirmed"
        and evidence["attempt_id"] == original["attempt_id"]
    )
    assert "report_path" not in evidence and not evidence["new_kernel_started"]
    assert (
        control["reads"] == reads
        and len(control["kernels"]) == 1
        and snapshot(root) == before
    )


def test_concurrent_duplicate_observes_active_attempt_without_waiting_or_replaying(
    generation,
):
    _, _, _, control, _, _, invoke = generation
    control["started"], control["release"] = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke)
        try:
            assert control["started"].wait(5)
            second = pool.submit(invoke).result(timeout=3)
            active = json.loads(second["payload"])
            assert not second["ok"] and active["generation_status"] == "running"
            assert active["last_recorded_phase"] == "replaying"
            assert (
                not active["new_kernel_started"]
                and active["editable_notebook_unchanged"]
            )
            assert (
                "still in progress" in control["messages"][-1]
                and "Open static report" not in control["messages"][-1]
            )
        finally:
            control["release"].set()
        completed = first.result(timeout=5)
    assert completed["ok"] and len(control["kernels"]) == 1
    assert json.loads(completed["payload"])["attempt_id"] == active["attempt_id"]


@pytest.mark.parametrize(
    "failure", ["before_context", "authoring", "kernel", "publication", "registration"]
)
def test_failures_keep_same_incomplete_attempt_and_never_automatically_retry(
    generation, failure, monkeypatch
):
    root, project, _, control, _, attempts, invoke = generation
    control["failure"] = failure
    if failure == "registration":
        from frequensolve.workflow import workspace

        real_remember = workspace.remember_notebook

        def uncertain_registration(*args, **kwargs):
            real_remember(*args, **kwargs)
            raise RuntimeError("Lost registration response; private details")

        monkeypatch.setattr(workspace, "remember_notebook", uncertain_registration)
    first = invoke()
    assert not first["ok"]
    initial = json.loads(first["payload"])
    assert initial["generation_status"] == "incomplete"
    assert not initial["recovered"] and not initial["paid_request_sent"]
    before = snapshot(root)
    reads, kernels = list(control["reads"]), list(control["kernels"])
    again = invoke()
    observed = json.loads(again["payload"])
    assert not again["ok"] and observed["attempt_id"] == initial["attempt_id"]
    assert observed["recovered"] and not observed["new_kernel_started"]
    assert observed["kernel_status"] == "unknown" and "report_path" not in observed
    assert control["reads"] == reads and control["kernels"] == kernels
    assert snapshot(root) == before and len(attempts()) == 1
    assert not any(
        message in json.dumps([first, again])
        for message in (
            "private connection details",
            "private authoring details",
            "private kernel details",
            "private publication details",
            "Lost registration response; private details",
        )
    )
    if failure in ("kernel", "publication", "registration"):
        assert observed["editable_notebook_unchanged"] and observed[
            "relative_path"
        ].endswith("analysis.ipynb")
    else:
        assert "notebook_path" not in observed
    if failure == "registration":
        assert notebook_history(project)["notebooks"]


@pytest.mark.parametrize(
    "changes",
    [
        {"frequency_indices": [1]},
        {"bundle": "changed.zip"},
        {"plan_id": "different-plan"},
        {"question": "A different request"},
    ],
)
def test_changed_request_cannot_reuse_or_overwrite_generation(generation, changes):
    root, _, _, control, _, _, invoke = generation
    assert invoke()["ok"]
    before, reads = snapshot(root), list(control["reads"])
    reply = invoke(changes)
    assert (
        not reply["ok"]
        and json.loads(reply["payload"])["generation_status"] == "unconfirmed"
    )
    assert (
        control["reads"] == reads
        and len(control["kernels"]) == 1
        and snapshot(root) == before
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "json",
        "oversized",
        "foreign_request",
        "receipt_path",
        "paid_claim",
        "symlink",
        "fifo",
    ],
)
def test_corrupt_records_never_become_receipts_or_replace_saved_files(
    generation, corruption
):
    root, _, _, control, _, attempts, invoke = generation
    assert invoke()["ok"]
    path = attempts()[0]
    if corruption == "json":
        path.write_text("not JSON")
    elif corruption == "oversized":
        path.write_text("x" * (notebook_attempt.MAX_RECORD_BYTES + 1))
    elif corruption == "symlink":
        secret = root / "private.json"
        secret.write_text('{"private":"must not be exposed"}')
        path.unlink()
        path.symlink_to(secret)
    elif corruption == "fifo":
        path.unlink()
        os.mkfifo(path, mode=0o600)
    else:
        value = json.loads(path.read_text())
        if corruption == "foreign_request":
            value["request"]["project_identity"]["project_id"] = "another-project"
        if corruption == "receipt_path":
            value["response"]["relative_path"] = "../escape/analysis.ipynb"
        if corruption == "paid_claim":
            value["response"]["paid_request_sent"] = True
        path.write_text(json.dumps(value))
    before, reads = snapshot(root), list(control["reads"])
    reply = invoke()
    assert not reply["ok"] and "report_path" not in json.loads(reply["payload"])
    assert "must not be exposed" not in json.dumps(reply)
    assert (
        control["reads"] == reads
        and len(control["kernels"]) == 1
        and snapshot(root) == before
    )


@pytest.mark.parametrize(
    "destination", ["../escape", "/absolute", "link/analysis", "saved-analysis"]
)
def test_unsafe_or_existing_destinations_preserve_original_files(
    generation, destination
):
    root, _, _, control, _, _, invoke = generation
    (root / "saved-analysis").mkdir()
    original = root / "saved-analysis/analysis.ipynb"
    original.write_bytes(b"Keep existing customer work")
    (root / "link").symlink_to(root / "saved-analysis", target_is_directory=True)
    reply = invoke({"destination": destination})
    assert not reply["ok"] and not control["kernels"]
    assert original.read_bytes() == b"Keep existing customer work"


def test_generation_handoff_escapes_customer_folder_names(generation):
    from urllib.parse import quote

    root, _, _, control, _, _, invoke = generation
    folder = "analysis ](unsafe) résumé"
    reply = invoke({"destination": folder})
    assert reply["ok"]
    message = control["messages"][-1]
    assert quote(str(root / folder / "analysis.ipynb"), safe="/") in message
    assert "](unsafe)" not in message and "fake-plan" not in message


@pytest.mark.parametrize(
    "changes",
    [{"frequency_indices": [2]}, {"bundle": "../escape.zip"}, {"destination": None}],
)
def test_invalid_request_does_not_create_a_generation_record(generation, changes):
    _, _, _, control, _, attempts, invoke = generation
    reply = invoke(changes)
    assert (
        not reply["ok"]
        and not attempts()
        and not control["reads"]
        and not control["kernels"]
    )


def test_replaced_local_project_identity_cannot_recover_another_projects_receipt(
    generation,
):
    root, project, _, control, _, _, invoke = generation
    assert invoke()["ok"]
    create_project(root, "other-project", "Another study")
    identity_path = project / ".frequensolve/project-identity.json"
    identity_path.write_bytes(
        (root / "other-project/.frequensolve/project-identity.json").read_bytes()
    )
    before, reads = snapshot(root), list(control["reads"])
    reply = invoke()
    assert not reply["ok"] and not json.loads(reply["payload"])["new_kernel_started"]
    assert (
        control["reads"] == reads
        and len(control["kernels"]) == 1
        and snapshot(root) == before
    )


def test_fatal_interruption_checkpoint_is_recoverable_without_resuming_kernel(tmp_path):
    root = tmp_path.resolve()
    owner = create_project(root, "project", "Interrupted synthetic study")
    project = root / "project"
    desired = {"destination": "interrupted"}
    value = state()
    value["plan"]["head"]["owner"] = {
        k: owner[k] for k in ("project_id", "project_key")
    }
    binding = context_binding(
        value["plan"], value["study"], [0], result_digest="sha256:" + "3" * 64
    )

    def interrupted(mark):
        mark("context", binding)
        write_local(root, "interrupted/analysis.ipynb", b"Saved authoring, no kernel")
        mark("authored")
        raise SystemExit("Synthetic process interruption")

    with pytest.raises(SystemExit):
        notebook_attempt.perform(root, project, desired, interrupted)
    before = snapshot(root)
    with pytest.raises(notebook_attempt.NotebookAttemptIncomplete) as caught:
        notebook_attempt.perform(
            root,
            project,
            desired,
            lambda *_: pytest.fail("Recovery restarted generation"),
        )
    evidence = caught.value.evidence
    assert (
        evidence["original_generation_status"] == "pending"
        and evidence["last_recorded_phase"] == "authored"
    )
    assert (
        evidence["generation_status"] == "incomplete"
        and evidence["kernel_status"] == "unknown"
    )
    assert (
        evidence["editable_notebook_unchanged"] and not evidence["new_kernel_started"]
    )
    assert snapshot(root) == before
