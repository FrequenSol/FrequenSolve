"""Credential helpers for SSH/HPC and cloud-backed execution sites."""

import getpass
import os
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

from frequensolve._optional import optional_dependency_error

try:
    from paramiko import (
        PasswordRequiredException,
        PKey,
    )
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "HPC credentials",
        extra="hpc",
        dependencies=("paramiko",),
        error=exc,
    ) from exc

from frequensolve.orchestrator.utils.credential_store import (
    CredentialStore,
    CredentialStoreError,
    KeyringCredentialStore,
)

__all__ = ["Credentials", "CloudCredentials"]


# ----------------------------------
# Login Credentials
# ----------------------------------
class Credentials:
    """Credentials for SSH-backed HPC sites.

    Subclasses configure the environment variable names used for the username,
    password, and SSH key passphrase. Missing values are requested
    interactively.
    """

    user_env: str = "HPC_USERNAME"
    pw_env: str = "HPC_PASSWORD"
    ssh_key_env: str = "SSH_PASSPHRASE"

    def __init__(
        self,
        *,
        username: Optional[str] = None,
        credential: Optional[str] = None,
        ssh_key: Optional[Union[str, Path]] = None,
        credential_store: Optional[CredentialStore] = None,
    ) -> None:
        self._configured_username = username
        self.credential = credential or type(self).__name__
        self.ssh_key_path = Path(ssh_key).expanduser() if ssh_key else None
        self._pending_secrets: dict[str, str] = {}
        self.credential_store: Optional[CredentialStore] = None
        if credential_store is not None:
            self.credential_store = credential_store
        else:
            try:
                self.credential_store = KeyringCredentialStore()
            except CredentialStoreError as exc:
                warnings.warn(
                    f"{exc}; credentials will be prompted for each session",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.credential_store = None

    @cached_property
    def username(self) -> str:
        """Return the SSH username from the environment or an interactive prompt."""

        user = self._configured_username or os.getenv(self.user_env)
        if user is None or user == "":
            user = input("HPC username: ")
        return user

    @cached_property
    def password(self) -> str:
        """Return the SSH password from the environment or an interactive prompt."""

        pw = os.getenv(self.pw_env)
        if pw:
            return pw
        pw = self._stored_secret("password")
        if not pw:
            pw = getpass.getpass("HPC password: ")
            self._pending_secrets["password"] = pw
        return pw

    @cached_property
    def ssh_key(self) -> PKey:
        """Load the configured private key for SSH authentication.

        Returns:
            Paramiko ``PKey`` loaded from ``ssh_key`` or ``~/.ssh/id_rsa``.
        """

        filename = self.ssh_key_path or Path("~/.ssh/id_rsa").expanduser()
        try:
            return PKey.from_path(filename)
        except PasswordRequiredException:
            passphrase = self._ssh_passphrase
            try:
                return PKey.from_path(filename, passphrase=passphrase)
            except Exception:
                self._pending_secrets.pop("ssh-passphrase", None)
                raise

    @cached_property
    def _ssh_passphrase(self) -> str:
        passphrase = os.getenv(self.ssh_key_env)
        if passphrase:
            return passphrase
        passphrase = self._stored_secret("ssh-passphrase")
        if not passphrase:
            passphrase = getpass.getpass("SSH key passphrase: ")
            self._pending_secrets["ssh-passphrase"] = passphrase
        return passphrase

    def _secret_key(self, kind: str) -> str:
        return f"{self.credential}:{self.username}:{kind}"

    def _stored_secret(self, kind: str) -> Optional[str]:
        if self.credential_store is None:
            return None
        try:
            return self.credential_store.get_secret(self._secret_key(kind))
        except CredentialStoreError as exc:
            warnings.warn(
                f"{exc}; credentials will be prompted for this session",
                RuntimeWarning,
                stacklevel=2,
            )
            self.credential_store = None
            return None

    def persist_pending(self) -> None:
        """Persist secrets entered this session after successful authentication."""

        pending, self._pending_secrets = self._pending_secrets, {}
        if not pending or self.credential_store is None:
            return
        try:
            for kind, value in pending.items():
                self.credential_store.set_secret(self._secret_key(kind), value)
        except CredentialStoreError as exc:
            warnings.warn(
                f"{exc}; credentials were not saved",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def duo_code(self) -> str:
        """Prompt for and return a site two-factor authentication code."""

        return getpass.getpass("Site 2FA code: ")

    def __str__(self) -> str:
        """Don't print credentials."""
        return ""

    def __repr__(self) -> str:
        """Don't print credentials."""
        return ""


# ----------------------------------
# Cloud Credentials
# ----------------------------------
@dataclass
class CloudCredentials:
    """Cloud-specific credentials.

    Attributes:
        access_key: AWS access key ID.
        secret_key: AWS secret access key.
    """

    key_env: str
    secret_env: str

    @cached_property
    def access_key(self) -> Optional[str]:
        """Retrieve AWS access key ID from the configured environment variable."""
        return os.getenv(self.key_env)

    @cached_property
    def secret_key(self) -> Optional[str]:
        """Retrieve AWS secret access key from the configured environment variable."""
        return os.getenv(self.secret_env)
