"""Bounded native review rehearsal. Client responses never authorize paid work."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

import anyio

from .journal import IntentJournal
from .review_display import render
from .storage import canonical

VERSION = "native-workflow-review-rehearsal.v1"
MAXIMUM_BYTES = 64000
FORM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "string",
            "title": "Review decision (no paid authorization)",
            "enum": ["Cancel", "Request changes", "Confirm rehearsal"],
        },
    },
    "required": ["choice"],
}


def digest(value: dict) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def proposal(value: dict, project_key: str, plan_id: str) -> dict:
    """Select customer review data from an owned authoritative read only."""
    head, quote = value.get("head", {}), value.get("quote", {})
    science, compute = value.get("scientific_inputs"), value.get("compute_review")
    plan = value.get("plan")
    if (
        head.get("plan_id") != plan_id
        or head.get("owner", {}).get("project_key") != project_key
        or type(head.get("revision")) is not int
        or head["revision"] < 0
        or not isinstance(science, dict)
        or not science
        or not isinstance(value.get("input_digest"), str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", value["input_digest"])
        or quote.get("purpose") not in ("planning", "validation", "execution")
        or not isinstance(quote.get("fingerprint"), str)
        or not isinstance(quote.get("maximum_credits"), str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", quote["fingerprint"])
        or not re.fullmatch(r"(0|[1-9]\d{0,17})(\.\d{1,6})?", quote["maximum_credits"])
    ):
        raise ValueError("A complete owned scientific and Credit proposal is required")
    if quote["purpose"] in ("planning", "validation") and (
        not isinstance(compute, dict)
        or not compute.get("execution_site_id")
        or not compute.get("planning_evaluations")
    ):
        raise ValueError("Sizing target and resource bounds are required")
    if quote["purpose"] == "execution" and (
        not isinstance(plan, dict) or not plan.get("context") or not plan.get("rows")
    ):
        raise ValueError("Execution requires the exact saved resource revision")
    result = deepcopy(
        {
            "version": VERSION,
            "project_key": project_key,
            "plan_id": plan_id,
            "revision": head["revision"],
            "status": head.get("status"),
            "operation_id": head.get("operation_id"),
            "input_digest": value["input_digest"],
            "scientific_inputs": science,
            "compute_review": compute,
            "plan": plan,
            "quote": quote,
            "charging": (
                "Allocated sizing work, including failed or canceled evaluations, "
                "can consume Credits. No solver execution minimum applies."
                if quote["purpose"] != "execution"
                else "Allocated execution can consume Credits, including its quoted "
                "minimum and failure/cancellation policy. Inspect quote.pricing."
            ),
            "cancellation": "Cancellation does not imply immediate termination or refund.",
        }
    )
    if len(canonical(result)) > MAXIMUM_BYTES:
        raise ValueError("Review exceeds the bounded native display limit")
    return result


def display(snapshot: dict) -> str:
    """The host receives the same complete snapshot returned by preview."""
    return render(snapshot, digest(snapshot), paid=False)


class ReviewEvidence:
    """A local rehearsal journal, separate from authoritative paid receipts."""

    def __init__(self, journal: IntentJournal):
        self.store = journal.store
        self.binding = digest(journal.binding)

    def key(self, intent: str) -> str:
        if not intent or len(intent) > 100:
            raise ValueError("Use a bounded review intent")
        return "review-" + hashlib.sha256(intent.encode()).hexdigest()

    def begin(self, intent: str, snapshot: dict) -> tuple[dict, bool]:
        with self.store.locked(self.key(intent)) as record:
            saved = record.read()
            fingerprint = digest(snapshot)
            if saved:
                if (
                    saved.get("binding") != self.binding
                    or saved.get("proposal_fingerprint") != fingerprint
                ):
                    raise ValueError(
                        "Review intent belongs to another proposal or connection"
                    )
                return saved, False
            saved = {
                "version": VERSION,
                "evidence_id": uuid4().hex,
                "binding": self.binding,
                "proposal_fingerprint": fingerprint,
                "proposal": snapshot,
                "state": "awaiting_input",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "evidence_origin": "client_reported_unattested",
                "human_approval_verified": False,
                "paid_authority": False,
                "backend_authorized": False,
            }
            record.write(saved)
            return saved, True

    def finish(self, intent: str, evidence_id: str, state: str, note: str = "") -> dict:
        with self.store.locked(self.key(intent)) as record:
            saved = record.read()
            if (
                not saved
                or saved.get("binding") != self.binding
                or saved.get("evidence_id") != evidence_id
                or saved.get("state") != "awaiting_input"
            ):
                raise ValueError("Review evidence is no longer current")
            saved.update(
                state=state,
                note=note,
                observed_at=datetime.now(timezone.utc).isoformat(),
            )
            record.write(saved)
            return saved

    def recover(self, intent: str) -> dict:
        with self.store.locked(self.key(intent)) as record:
            saved = record.read()
            if not saved or saved.get("binding") != self.binding:
                raise ValueError("Review evidence is unavailable for this connection")
            return saved


async def request_input(
    evidence: ReviewEvidence,
    intent: str,
    snapshot: dict,
    expected_fingerprint: str,
    fetch: Callable[[], Awaitable[dict]],
    elicit: Callable[[str, dict], Awaitable[dict]],
    *,
    timeout: float = 180,
) -> dict:
    if digest(snapshot) != expected_fingerprint:
        raise ValueError("Proposal changed; preview and review the new proposal")
    saved, created = await anyio.to_thread.run_sync(evidence.begin, intent, snapshot)
    if not created:
        return saved
    state, note = "incomplete", ""
    try:
        with anyio.fail_after(timeout):
            response = await elicit(display(snapshot), deepcopy(FORM))
            action, content = response.get("action"), response.get("content")
            if action in ("decline", "cancel") and content is None:
                state = "declined" if action == "decline" else "canceled"
            elif action == "accept" and isinstance(content, dict):
                choice, note = content.get("choice"), content.get("note", "")
                if isinstance(choice, str):
                    choice = {
                        "Cancel": "cancel",
                        "Request changes": "request_changes",
                        "Confirm rehearsal": "confirm_rehearsal",
                    }.get(choice, choice)
                if (
                    set(content) - {"choice", "note"}
                    or not isinstance(choice, str)
                    or choice not in ("cancel", "request_changes", "confirm_rehearsal")
                    or not isinstance(note, str)
                    or len(note) > 1024
                ):
                    raise ValueError("Invalid client response")
                state = {
                    "confirm_rehearsal": "client_confirmed_rehearsal",
                    "request_changes": "changes_requested",
                    "cancel": "canceled",
                }[choice]
                if choice == "request_changes" and not note.strip():
                    response = await elicit(
                        "NO-SPEND REVIEW REHEARSAL · Requested resource changes\nDescribe the change you want. This does not authorize paid work.",
                        {
                            "type": "object",
                            "properties": {
                                "note": {
                                    "type": "string",
                                    "title": "Requested changes",
                                    "minLength": 1,
                                    "maxLength": 1024,
                                }
                            },
                            "required": ["note"],
                        },
                    )
                    action, content = response.get("action"), response.get("content")
                    if action in ("decline", "cancel") and content is None:
                        state, note = (
                            ("declined" if action == "decline" else "canceled"),
                            "",
                        )
                    elif (
                        action == "accept"
                        and isinstance(content, dict)
                        and set(content) == {"note"}
                        and isinstance(content["note"], str)
                        and 0 < len(content["note"]) <= 1024
                        and content["note"].strip()
                    ):
                        note = content["note"]
                    else:
                        raise ValueError("Describe the requested change")
            else:
                raise ValueError("Invalid client response")
            if digest(await fetch()) != expected_fingerprint:
                state, note = "stale", ""
    except Exception:
        state, note = "incomplete", ""
    except BaseException:
        state, note = "incomplete", ""
        raise
    finally:
        # Cancellation/interruption also retains incomplete evidence. No retry
        # implicitly reopens the prompt or interprets silence as acceptance.
        with anyio.CancelScope(shield=True):
            result = await anyio.to_thread.run_sync(
                evidence.finish, intent, saved["evidence_id"], state, note
            )
    return result
