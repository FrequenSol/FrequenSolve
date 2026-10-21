"""Client-mediated paid-stage forms; Cloud receipts remain authoritative."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

import anyio

from .journal import IntentJournal
from .review import digest
from .storage import canonical

VERSION = "native-workflow-paid-review.v1"
FORM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "string",
            "title": "Paid-stage decision",
            "description": "Choose one action for this paid stage.",
            "enum": ["Cancel", "Request changes", "Approve this stage"],
        },
    },
    "required": ["choice"],
}


class PaidApprovalInputError(ValueError):
    """A safe form diagnostic created locally, never from a transport response."""


def form_answer(response: dict, field: str) -> tuple[str, Any]:
    action, content = response.get("action"), response.get("content")
    if action in ("cancel", "decline") and content is None:
        return ("canceled" if action == "cancel" else "declined"), None
    if action == "accept" and isinstance(content, dict) and set(content) == {field}:
        return "accepted", content[field]
    raise PaidApprovalInputError(
        "The client did not return the requested form field. No paid request was sent."
    )


def envelope(value: dict, ids: dict, stage: str) -> dict:
    snapshot, fingerprint = value.get("proposal"), value.get("proposal_fingerprint")
    if (
        not isinstance(snapshot, dict)
        or not isinstance(fingerprint, str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", fingerprint)
    ):
        raise ValueError("Cloud did not return an authoritative review")
    quote = snapshot.get("quote", {})
    if (
        snapshot.get("version") != VERSION
        or snapshot.get("stage") != stage
        or any(snapshot.get(k) != v for k, v in ids.items())
        or type(snapshot.get("revision")) is not int
        or snapshot["revision"] < 0
        or not isinstance(quote, dict)
        or quote.get("purpose") != stage
        or not isinstance(quote.get("maximum_credits"), str)
        or not re.fullmatch(r"(0|[1-9]\d{0,17})(\.\d{1,6})?", quote["maximum_credits"])
        or not isinstance(quote.get("fingerprint"), str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", quote["fingerprint"])
        or not snapshot.get("input_digest")
        or not snapshot.get("scientific_inputs")
        or stage == "execution"
        and not snapshot.get("plan")
        or stage != "execution"
        and not snapshot.get("compute_review")
        or len(canonical(snapshot)) > 64000
    ):
        raise ValueError("The complete exact paid-stage proposal is required")
    # The service generates the fingerprint. Preserve a separate local digest
    # rather than pretending Python/JavaScript float serialization is identical.
    return {
        "proposal": deepcopy(snapshot),
        "proposal_fingerprint": fingerprint,
        "local_digest": digest(snapshot),
    }


class PaidApproval:
    def __init__(self, journal: IntentJournal):
        self.journal, self.store, self.binding = (
            journal,
            journal.store,
            digest(journal.binding),
        )

    def key(self, intent: str) -> str:
        if not intent or len(intent) > 100:
            raise ValueError("Retain a bounded native approval intent")
        return "native-paid-" + hashlib.sha256(intent.encode()).hexdigest()

    def lookup(self, intent: str) -> dict | None:
        with self.store.locked(self.key(intent)) as record:
            value = record.read()
        if value and value.get("binding") != self.binding:
            raise ValueError(
                "Paid approval evidence is unavailable for this connection"
            )
        return value

    def read(self, intent: str) -> dict:
        value = self.lookup(intent)
        if value is None:
            raise ValueError(
                "Paid approval evidence is unavailable for this connection"
            )
        return value

    def begin(self, intent: str, current: dict) -> tuple[dict, bool]:
        with self.store.locked(self.key(intent)) as record:
            saved = record.read()
            if saved:
                if (
                    saved.get("binding") != self.binding
                    or saved.get("review") != current
                ):
                    raise ValueError(
                        "Intent already belongs to another proposal or connection"
                    )
                return saved, False
            snapshot, quote = current["proposal"], current["proposal"]["quote"]
            evidence_id = uuid4().hex
            request = {
                k: snapshot[k]
                for k in ("project_key", "project_id", "plan_id", "stage")
            }
            request.update(
                expected_revision=snapshot["revision"],
                proposal_fingerprint=current["proposal_fingerprint"],
                accepted_quote_fingerprint=quote["fingerprint"],
                maximum_credits=quote["maximum_credits"],
                evidence_id=evidence_id,
            )
            saved = {
                "version": VERSION,
                "binding": self.binding,
                "evidence_id": evidence_id,
                "review": current,
                "request": request,
                "state": "awaiting_input",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "evidence_origin": "native_client_reported",
                "human_approval_verified": False,
                "backend_authorized": False,
            }
            record.write(saved)
            return saved, True

    def transition(
        self,
        intent: str,
        evidence_id: str,
        allowed: tuple[str, ...],
        state: str,
        **updates: Any,
    ) -> dict:
        with self.store.locked(self.key(intent)) as record:
            saved = record.read()
            if (
                not saved
                or saved.get("binding") != self.binding
                or saved.get("evidence_id") != evidence_id
                or saved.get("state") not in allowed
            ):
                raise ValueError("Paid review state changed; recover the same intent")
            saved.update(
                state=state,
                observed_at=datetime.now(timezone.utc).isoformat(),
                **updates,
            )
            record.write(saved)
            return saved

    def recover(self, intent: str, send: Callable[[str, dict], dict]) -> dict:
        saved = self.lookup(intent)
        if saved is None:
            # First use is a normal recovery outcome, not failed confirmation.
            # Do not create evidence or a paid request until the client form.
            return {
                "version": VERSION,
                "state": "absent",
                "backend_authorized": False,
                "human_approval_verified": False,
                "next_action": "Preview the same stage, then request_input with this same intent and the exact proposal fingerprint.",
            }
        if saved["state"] not in ("client_confirmed", "sending"):
            return saved
        try:
            self.transition(
                intent, saved["evidence_id"], ("client_confirmed", "sending"), "sending"
            )
        except ValueError:
            current = self.read(intent)
            if current["state"] == "backend_accepted":
                return current
            raise
        # This stores the exact request and one idempotency key before transport.
        # A lost response retries only this request; the service owns paid state.
        result = self.journal.perform(
            "native-paid:" + intent, "native_approve", saved["request"], send
        )
        receipt = result.get("native_approval", {})
        if (
            not isinstance(receipt, dict)
            or receipt.get("accepted") is not True
            or receipt.get("evidence_id") != saved["evidence_id"]
            or receipt.get("proposal_fingerprint")
            != saved["review"]["proposal_fingerprint"]
            or receipt.get("human_approval_verified") is not False
            or receipt.get("evidence_origin") != "native_client_reported"
        ):
            raise ValueError("Cloud did not confirm the existing paid authorization")
        # Concurrent recoveries share the same domain receipt and may finish first.
        current = self.read(intent)
        if current["state"] == "backend_accepted":
            return current
        try:
            return self.transition(
                intent,
                saved["evidence_id"],
                ("sending",),
                "backend_accepted",
                backend_authorized=True,
                response=result,
            )
        except ValueError:
            current = self.read(intent)
            if current["state"] == "backend_accepted":
                return current
            raise


async def confirm_paid(
    approval: PaidApproval,
    intent: str,
    current: dict,
    expected_fingerprint: str,
    fetch: Callable[[], Awaitable[dict]],
    elicit: Callable[[str, dict], Awaitable[dict]],
    send: Callable[[str, dict], dict],
    *,
    timeout: float = 180,
) -> dict:
    if current["proposal_fingerprint"] != expected_fingerprint:
        raise ValueError("Proposal changed; preview the exact paid stage again")
    saved, created = await anyio.to_thread.run_sync(approval.begin, intent, current)
    if not created:
        return saved
    snapshot = current["proposal"]
    state, note = "incomplete", ""
    failure_code = "client_form_failed"
    try:
        with anyio.fail_after(timeout):
            maximum = snapshot["quote"]["maximum_credits"]
            message = (
                f"PAID {snapshot['stage'].upper()} APPROVAL. This can start paid compute.\n"
                "Review the complete authoritative proposal below. To approve only this stage, choose Approve this stage. A second prompt asks for "
                f"exactly {maximum} Credits (enter only the number). Cancellation does not imply immediate termination or refund. "
                "Allocated failed/canceled work may be charged; execution minimum/failure policy is in quote.pricing. "
                "The client reports your response; this is not independent proof that a human answered.\n"
                + json.dumps(
                    snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                + "\nCloud proposal fingerprint: "
                + current["proposal_fingerprint"]
            )
            state, choice = form_answer(await elicit(message, deepcopy(FORM)), "choice")
            failure_code = "invalid_form_response"
            if state == "accepted":
                # Older saved clients can still return the original protocol labels.
                choice = (
                    {
                        "Cancel": "cancel",
                        "Request changes": "request_changes",
                        "Approve this stage": "approve_paid_stage",
                    }.get(choice, choice)
                    if isinstance(choice, str)
                    else None
                )
                if choice not in ("cancel", "request_changes", "approve_paid_stage"):
                    raise PaidApprovalInputError(
                        "The paid-stage decision was invalid. No paid request was sent."
                    )
                if choice == "cancel":
                    state = "canceled"
                else:
                    field = (
                        "maximum_credits" if choice == "approve_paid_stage" else "note"
                    )
                    description = (
                        f"Credit maximum: type exactly {maximum} to approve up to {maximum} Credits for {snapshot['stage']}. Enter the number only; do not repeat approve_paid_stage."
                        if field == "maximum_credits"
                        else "Describe the requested changes. This does not authorize paid work."
                    )
                    detail_form = {
                        "type": "object",
                        "properties": {
                            field: {
                                "type": "string",
                                "title": (
                                    "Credit maximum"
                                    if field == "maximum_credits"
                                    else "Requested changes"
                                ),
                                "description": description,
                                "minLength": 1,
                                "maxLength": 25 if field == "maximum_credits" else 1024,
                            }
                        },
                        "required": [field],
                    }
                    failure_code = (
                        "credit_limit_not_confirmed"
                        if field == "maximum_credits"
                        else "missing_change_note"
                    )
                    state, value = form_answer(
                        await elicit(
                            f"PAID {snapshot['stage'].upper()} · Confirm the Credit limit\n"
                            f"{description}\nThe exact proposal you just reviewed remains bound to revision {snapshot['revision']}. "
                            f"Proposal fingerprint: {current['proposal_fingerprint']}. No other stage is approved.",
                            detail_form,
                        ),
                        field,
                    )
                    if state == "accepted":
                        if field == "maximum_credits":
                            if value != maximum:
                                raise PaidApprovalInputError(
                                    f"The Credit maximum was not confirmed: enter exactly {maximum} in the Credit maximum field, not approve_paid_stage. No paid request was sent."
                                )
                            state = "client_confirmed"
                        else:
                            if (
                                not isinstance(value, str)
                                or not value.strip()
                                or len(value) > 1024
                            ):
                                raise PaidApprovalInputError(
                                    "Describe the requested change. No paid request was sent."
                                )
                            state, note = "changes_requested", value
            failure_code = "proposal_recheck_failed"
            if state == "client_confirmed" and await fetch() != current:
                state = "stale"
    except BaseException:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(
                lambda: approval.transition(
                    intent,
                    saved["evidence_id"],
                    ("awaiting_input",),
                    "incomplete",
                    failure_code=failure_code,
                )
            )
        raise
    saved = await anyio.to_thread.run_sync(
        lambda: approval.transition(
            intent, saved["evidence_id"], ("awaiting_input",), state, note=note
        )
    )
    if state == "client_confirmed":
        return await anyio.to_thread.run_sync(approval.recover, intent, send)
    return saved
