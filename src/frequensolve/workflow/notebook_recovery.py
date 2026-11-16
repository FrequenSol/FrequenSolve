"""Choose registered local evidence without running analysis or paid work."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import notebook_catalog as catalog
from .notebook_binding import context_binding, current_result_digest, match
from .workspace import notebook_artifacts


def _unknown(**extra: object) -> dict:
    return {
        "artifacts": [],
        "binding": None,
        "recorded_replay_status": "unconfirmed",
        "read_outcome": "unconfirmed",
        "paid_request_sent": False,
        "current_environment_checked": False,
        "service_attestation": False,
        **extra,
    }


def _choices(entries: list[dict], **extra: object) -> dict:
    return _unknown(
        candidates=[catalog.summary(entry) for entry in entries[: catalog.PAGE_SIZE]],
        additional_candidates=len(entries) > catalog.PAGE_SIZE,
        **extra,
    )


def recover(
    root: Path,
    project: Path,
    view: dict,
    indices: list[int],
    selected: str | None,
    fetch_manifest: Callable[[dict], dict],
) -> dict:
    expected = None
    try:
        expected = context_binding(
            view["plan"], view.get("study") or {}, indices, result_digest=None
        )
        reference = view["project_identity"]
        if any(
            expected[key] != reference[key]
            for key in ("project_id", "project_key", "plan_id")
        ):
            expected = None
    except (ValueError, TypeError, KeyError, IndexError):
        expected = None

    collection: dict
    try:
        collection = (
            {"records": [catalog.get(project, selected)], "history_complete": True}
            if selected is not None
            else catalog.candidates(project)
        )
    except Exception:
        collection = {"records": [], "history_complete": False}

    def possible(entry: dict) -> bool:
        binding = entry["bookmark"].get("binding")
        return (
            expected is not None
            and binding is not None
            and match(binding, expected)["status"] not in ("mismatch", "unbound")
        )

    potential = [entry for entry in collection["records"] if possible(entry)]

    def refresh_result() -> None:
        assert expected is not None
        try:
            response = fetch_manifest(
                {
                    "project_id": expected["project_id"],
                    "project_key": expected["project_key"],
                    "plan_id": expected["plan_id"],
                    "execution_id": expected["execution_id"],
                }
            )
            expected["result_digest"] = current_result_digest(response, view["plan"])
        except Exception:
            expected["result_digest"] = None

    if not collection["history_complete"]:
        book = _choices(
            potential,
            history_incomplete=True,
        )
    elif len(potential) > 1:
        # Metadata first; don't hash every bundle merely to ask which copy is wanted.
        refresh_result()
        matching = [
            entry
            for entry in potential
            if match(entry["bookmark"].get("binding"), expected)["status"] == "match"
        ]
        if len(matching) > 1:
            book = _choices(
                matching,
                selection_required=True,
            )
        elif expected is not None and expected["result_digest"] is None:
            book = _unknown(result_observation="unconfirmed")
        else:
            try:
                book = notebook_artifacts(
                    root, project, matching[0]["bookmark_id"] if matching else None
                )
            except Exception:
                book = _unknown()
    else:
        try:
            key = selected or (potential[0]["bookmark_id"] if potential else None)
            book = notebook_artifacts(root, project, key)
        except Exception:
            book = _unknown()
        if possible({"bookmark": book}) and book.get("saved_bytes_unchanged"):
            refresh_result()
            if expected is not None and expected["result_digest"] is None:
                book["result_observation"] = "unconfirmed"

    book["context_match"] = (
        {"status": "unconfirmed", "mismatch_fields": []}
        if book.get("read_outcome") == "unconfirmed"
        else match(book.get("binding"), expected)
    )
    book["recoverable_in_workspace"] = bool(book["artifacts"])
    if book["context_match"]["status"] != "match":
        book["artifacts"] = []
    elif book.get("recorded_replay_status") != "complete":
        book["artifacts"] = [
            item for item in book["artifacts"] if item["name"] == "analysis.ipynb"
        ]
    return book
