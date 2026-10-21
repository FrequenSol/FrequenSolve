"""Persist intent before transport; the remote service owns operation state."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Callable
from uuid import uuid4

from .storage import PrivateStore, canonical


class IntentJournal:
    def __init__(self, store: PrivateStore, binding: dict):
        self.store, self.binding = store, deepcopy(binding)

    def perform(
        self,
        intent: str,
        operation: str,
        arguments: dict,
        send: Callable[[str, dict], dict],
        *,
        reconcile: bool = False,
    ) -> dict:
        """An intentional repeat needs a new intent; retries retain exact bytes."""
        request = {"operation": operation, "arguments": deepcopy(arguments)}
        identity = hashlib.sha256(canonical(self.binding)).hexdigest()
        with self.store.locked(
            "intent-" + hashlib.sha256(intent.encode()).hexdigest()
        ) as record:
            saved = record.read()
            if saved is None:
                saved = {
                    "version": 1,
                    "binding": identity,
                    "request": request,
                    "idempotency_key": uuid4().hex,
                    "status": "pending",
                }
                record.write(saved)
            if saved.get("binding") != identity or saved.get("request") != request:
                raise ValueError(
                    "Intent already belongs to a different request or connection"
                )
            if saved.get("status") == "complete" and not reconcile:
                return deepcopy(saved["response"])
            # Reconcile accepted preparation with the authoritative service,
            # retaining the exact original bytes and request identity.
            saved["status"] = "pending"
            record.write(saved)
            payload = {**arguments, "idempotency_key": saved["idempotency_key"]}
            # A transport error deliberately leaves the original pending intent.
            response = send(operation, payload)
            saved.update(status="complete", response=response)
            record.write(saved)
            return deepcopy(response)

    def inspect(self, intent: str) -> dict | None:
        with self.store.locked(
            "intent-" + hashlib.sha256(intent.encode()).hexdigest()
        ) as record:
            return record.read()
