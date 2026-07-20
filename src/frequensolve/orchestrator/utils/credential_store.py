"""Secure credential storage adapters for execution sites."""

from __future__ import annotations

from typing import Protocol

__all__ = [
    "CredentialStore",
    "CredentialStoreError",
    "KeyringCredentialStore",
]


class CredentialStoreError(RuntimeError):
    """Raised when a secure credential backend cannot be used."""


class CredentialStore(Protocol):
    """Minimal interface used to retrieve and persist site secrets."""

    def get_secret(self, key: str) -> str | None:
        """Return the secret stored for *key*, or ``None`` when absent."""

    def set_secret(self, key: str, value: str) -> None:
        """Persist *value* securely for *key*."""


class KeyringCredentialStore:
    """Store HPC secrets in the operating system's credential manager."""

    service_name = "frequensolve.hpc"

    def __init__(self) -> None:
        try:
            import keyring
        except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
            raise CredentialStoreError(
                "The keyring package is not installed; install frequensolve[hpc]"
            ) from exc
        self._keyring = keyring

    def get_secret(self, key: str) -> str | None:
        """Read a secret from the active OS keyring backend."""

        try:
            return self._keyring.get_password(self.service_name, key)
        except Exception as exc:
            raise CredentialStoreError(
                "The operating system credential store is unavailable"
            ) from exc

    def set_secret(self, key: str, value: str) -> None:
        """Write a secret to the active OS keyring backend."""

        try:
            self._keyring.set_password(self.service_name, key, value)
        except Exception as exc:
            raise CredentialStoreError(
                "The operating system credential store is unavailable"
            ) from exc
