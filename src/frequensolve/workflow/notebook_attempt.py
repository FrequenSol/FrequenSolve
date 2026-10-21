"""Durable local generation intent; recovery never starts another kernel."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, quote, urlsplit
from uuid import uuid4

from .client import read_local, relative
from .notebook_binding import validate as validate_binding
from .notebook_catalog import get as registered_notebook
from .storage import PrivateStore, canonical
from .workspace import notebook_artifacts

VERSION = "notebook-attempt.v1"
MAX_RECORD_BYTES = 65536
PHASES = {
    "preparing",
    "context",
    "directory_created",
    "authored",
    "replaying",
    "publishing",
    "registered",
    "complete",
}
RESPONSE_FIELDS = {
    "relative_path",
    "notebook_path",
    "report_path",
    "replay_status",
    "binding",
    "notebook_id",
    "study_url",
    "assurance",
    "paid_request_sent",
}


class NotebookAttemptIncomplete(Exception):
    def __init__(self, evidence: dict) -> None:
        self.evidence = evidence


def narrative(value: dict) -> str:
    """Customer content uses verified links; operational identities stay structured."""
    status = value.get("generation_status")
    if status == "complete":
        lines = [
            (
                "Recovered your saved analysis notebook."
                if value["recovered"]
                else "Your analysis notebook is ready."
            ),
            "Recorded local replay: **complete**. The saved files match the original receipt.",
        ]
        notebook = Path(value["notebook_path"])
        for label, path in (
            ("Open editable notebook", notebook),
            ("Open notebook with recorded outputs", notebook.parent / "executed.ipynb"),
            ("Open static report", notebook.parent / "report.html"),
        ):
            lines.append(f"- [{label}]({quote(str(path), safe='/')})")
        lines.append(
            "Reopening did not run another kernel or recheck the current environment. This is recorded local execution evidence."
            if value["recovered"]
            else "The notebook contains local analysis only. Its execution evidence records the code, inputs and environment used."
        )
        return "\n\n".join(lines)
    lines = [
        (
            "Your notebook creation is still in progress."
            if status == "running"
            else (
                "The original replay completed, but its saved bundle cannot currently be verified."
                if value.get("original_generation_status") == "complete"
                else "This notebook attempt has no confirmed completed receipt."
            )
        )
    ]
    phases = {
        "preparing": "checking the saved study",
        "context": "binding the saved inputs",
        "directory_created": "preparing local files",
        "authored": "notebook authored",
        "replaying": "local analysis requested",
        "publishing": "saving outputs",
        "registered": "registering the saved notebook",
        "complete": "completed local replay",
    }
    if value.get("last_recorded_phase") in phases:
        lines.append(
            "Last recorded step: " + phases[value["last_recorded_phase"]] + "."
        )
    if value.get("editable_notebook_unchanged"):
        lines.append(
            f"[Inspect the saved editable notebook]({quote(value['notebook_path'], safe='/')})"
        )
        lines.append(
            "Only the editable notebook file has been checked here; its inputs or outputs may be incomplete."
        )
    lines.append(
        "Keep this attempt and recover the same request to check again. Recovery does not start another kernel or overwrite files."
    )
    return "\n\n".join(lines)


def _validate(saved: dict, desired: dict) -> None:
    required = {
        "version",
        "request",
        "attempt_id",
        "status",
        "phase",
        "created_at",
        "updated_at",
        "producer",
    }
    if (
        not isinstance(saved, dict)
        or not required
        <= set(saved)
        <= required | {"binding", "editable", "notebook_id", "response"}
        or len(canonical(saved)) > MAX_RECORD_BYTES
        or saved["version"] != VERSION
        or canonical(saved["request"]) != canonical(desired)
        or not isinstance(saved["attempt_id"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", saved["attempt_id"])
        or saved["status"] not in ("pending", "incomplete", "complete")
        or saved["phase"] not in PHASES
    ):
        raise ValueError(
            "The saved notebook attempt is unavailable or belongs to a different request"
        )
    for key in ("created_at", "updated_at"):
        if (
            not isinstance(saved[key], str)
            or len(saved[key]) > 40
            or datetime.fromisoformat(saved[key]).utcoffset() is None
        ):
            raise ValueError("Invalid notebook checkpoint time")
    producer = saved["producer"]
    if (
        not isinstance(producer, dict)
        or set(producer) != {"package_version", "source_revision", "source_dirty"}
        or not isinstance(producer["package_version"], str)
        or not 1 <= len(producer["package_version"]) <= 128
        or (
            producer["source_revision"] is not None
            and not re.fullmatch(r"[0-9a-f]{40}", producer["source_revision"])
        )
        or (
            producer["source_dirty"] is not None
            and type(producer["source_dirty"]) is not bool
        )
    ):
        raise ValueError("Invalid notebook producer identity")
    if "binding" in saved:
        validate_binding(saved["binding"])
    if saved["status"] == "complete" and not {
        "binding",
        "notebook_id",
        "response",
    } <= set(saved):
        raise ValueError("Incomplete generation receipt")
    if "notebook_id" in saved and not re.fullmatch(
        r"[0-9a-f]{64}", saved["notebook_id"]
    ):
        raise ValueError("Invalid notebook selection")
    if "editable" in saved:
        item = saved["editable"]
        if (
            not isinstance(item, dict)
            or set(item) != {"bytes", "sha256"}
            or type(item["bytes"]) is not int
            or not 0 <= item["bytes"] <= 16 * 1024 * 1024
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid editable input identity")


def _evidence(root: Path, saved: dict, *, active: bool = False) -> dict:
    result = {
        "attempt_id": saved["attempt_id"],
        "generation_status": "running" if active else "incomplete",
        "last_recorded_phase": saved["phase"],
        "last_recorded_at": saved["updated_at"],
        "producer": saved["producer"],
        "original_generation_status": saved["status"],
        "replay_status": (
            "unconfirmed" if active or saved["status"] == "complete" else "incomplete"
        ),
        "recovered": True,
        "kernel_status": "unknown",
        "new_kernel_started": False,
        "paid_request_sent": False,
        "current_environment_checked": False,
        "service_attestation": False,
        "evidence_origin": "local_client_reported",
        "next_action": (
            "observe_same_notebook_attempt"
            if active
            else "inspect_saved_notebook_attempt"
        ),
    }
    if "editable" in saved:
        name = relative(saved["request"]["destination"]) + "/analysis.ipynb"
        try:
            data = read_local(root, name)
            unchanged = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            } == saved["editable"]
        except (OSError, ValueError):
            unchanged = False
        result["editable_notebook_unchanged"] = unchanged
        if unchanged:
            result.update(
                relative_path=name,
                notebook_path=str(root / name),
                artifact_scope="editable notebook only; input/output bundle completeness unconfirmed",
            )
    return result


def _completed(root: Path, project: Path, saved: dict) -> dict:
    response = saved.get("response")
    destination = relative(saved["request"]["destination"])
    if (
        not isinstance(response, dict)
        or set(response) != RESPONSE_FIELDS
        or response["relative_path"] != destination + "/analysis.ipynb"
        or response["notebook_path"] != str(root / destination / "analysis.ipynb")
        or response["report_path"] != destination + "/report.html"
        or response["replay_status"] != "complete"
        or response["paid_request_sent"] is not False
        or response["binding"] != saved.get("binding")
        or response["notebook_id"] != saved.get("notebook_id")
    ):
        raise ValueError("Invalid completed notebook receipt")
    location = urlsplit(response["study_url"])
    if (
        location.scheme not in ("http", "https")
        or not location.netloc
        or location.username is not None
        or location.password is not None
        or location.fragment
        or not location.path.endswith("/study")
        or sorted(parse_qsl(location.query))
        != sorted(
            [
                ("project_key", saved["binding"]["project_key"]),
                ("plan_id", saved["binding"]["plan_id"]),
            ]
        )
    ):
        raise ValueError("Invalid study receipt link")
    try:
        files = registered_notebook(project, saved["notebook_id"])["bookmark"]["files"]
        book = notebook_artifacts(root, project, saved["notebook_id"])
    except (OSError, ValueError, TypeError, KeyError):
        raise NotebookAttemptIncomplete(_evidence(root, saved)) from None
    if (
        not book.get("saved_bytes_unchanged")
        or book.get("recorded_replay_status") != "complete"
        or book.get("binding") != saved["binding"]
        or not {
            "analysis.ipynb",
            "executed.ipynb",
            "report.html",
            "report.json",
            "replay.json",
        }
        <= {item["name"] for item in files}
    ):
        raise NotebookAttemptIncomplete(_evidence(root, saved))
    return {
        **response,
        "attempt_id": saved["attempt_id"],
        "generation_status": "complete",
        "last_recorded_phase": "complete",
        "recorded_completed_at": saved["updated_at"],
        "attempt_created_at": saved["created_at"],
        "producer": saved["producer"],
        "recovered": True,
        "saved_bytes_unchanged": True,
        "new_kernel_started": False,
        "current_environment_checked": False,
        "service_attestation": False,
        "evidence_origin": "local_client_reported",
        "assurance": "original local replay receipt; saved files checked, no new environment check or service attestation",
    }


def perform(
    root: Path, project: Path, desired: dict, generate: Callable[..., dict]
) -> dict:
    destination = relative(desired["destination"])
    if len(canonical(desired)) > 8192:
        raise ValueError("Notebook request exceeds the local recovery limit")
    name = (
        "notebook-attempt-"
        + hashlib.sha256(canonical({"destination": destination})).hexdigest()
    )
    store = PrivateStore(project / ".frequensolve/workflow")
    try:
        with store.locked(name, blocking=False) as record:
            previous = record.read()
            if previous is not None:
                _validate(previous, desired)
                if previous["status"] == "complete":
                    return _completed(root, project, previous)
                raise NotebookAttemptIncomplete(_evidence(root, previous))
            from frequensolve._version import get_versions

            version = get_versions()
            saved: dict = {
                "version": VERSION,
                "request": desired,
                "attempt_id": uuid4().hex,
                "status": "pending",
                "phase": "preparing",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "producer": {
                    "package_version": version["version"],
                    "source_revision": version["full-revisionid"],
                    "source_dirty": version["dirty"],
                },
            }
            _validate(saved, desired)
            record.write(saved)

            def mark(
                phase: str, binding: dict | None = None, notebook_id: str | None = None
            ) -> None:
                saved["phase"] = phase
                saved["updated_at"] = datetime.now(timezone.utc).isoformat()
                if binding is not None:
                    saved["binding"] = binding
                if notebook_id is not None:
                    saved["notebook_id"] = notebook_id
                if phase == "authored":
                    data = read_local(root, destination + "/analysis.ipynb")
                    saved["editable"] = {
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                _validate(saved, desired)
                record.write(saved)

            try:
                response = generate(mark)
                completed = {
                    **saved,
                    "status": "complete",
                    "phase": "complete",
                    "response": response,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                _validate(completed, desired)
                # Verify publication before writing a completed generation receipt.
                value = _completed(root, project, completed)
                record.write(completed)
                return {**value, "recovered": False, "new_kernel_started": True}
            except Exception:
                saved.pop("response", None)
                saved["status"] = "incomplete"
                saved["updated_at"] = datetime.now(timezone.utc).isoformat()
                record.write(saved)
                raise NotebookAttemptIncomplete(
                    {
                        **_evidence(root, saved),
                        "recovered": False,
                        "new_kernel_started": (
                            None
                            if saved["phase"]
                            in ("replaying", "publishing", "registered")
                            else False
                        ),
                    }
                ) from None
    except BlockingIOError:
        # The other writer owns the attempt. Read only its atomic last checkpoint.
        observed = json.loads(
            read_local(
                project, ".frequensolve/workflow/" + name + ".json", MAX_RECORD_BYTES
            )
        )
        _validate(observed, desired)
        raise NotebookAttemptIncomplete(
            _evidence(root, observed, active=True)
        ) from None
