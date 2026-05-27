import getpass
import os
from dataclasses import dataclass
from functools import cached_property

from frequensolve._optional import optional_dependency_error

try:
    from dotenv import load_dotenv
    from paramiko import (
        PasswordRequiredException,
        RSAKey,
    )
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "HPC credentials",
        extra="hpc",
        dependencies=("paramiko", "python-dotenv"),
        error=exc,
    ) from exc

__all__ = ["Credentials", "CloudCredentials"]


# ----------------------------------
# Login Credentials
# ----------------------------------
@dataclass
class Credentials:
    """Credentials for SSH-backed HPC sites.

    Subclasses configure the environment variable names used for the username,
    password, and SSH key passphrase. Missing values are requested
    interactively.
    """

    user_env: str
    pw_env: str
    ssh_key_env: str

    def __init__(self):
        load_dotenv()

    @cached_property
    def username(self):
        """Return the SSH username from the environment or an interactive prompt."""

        user = os.getenv(self.user_env)
        if user is None or user == "":
            print(
                f"Avoid providing this each time by adding the {self.user_env} to FrequenSolve/.env"
            )
            user = input("TACC Username:")
        return user

    @cached_property
    def password(self):
        """Return the SSH password from the environment or an interactive prompt."""

        pw = os.getenv(self.pw_env)
        if pw is None or pw == "":
            print(
                f"Avoid providing this each time by adding the {self.pw_env} to FrequenSolve/.env"
            )
            pw = input("TACC Password:")
        return pw

    @cached_property
    def ssh_key(self):
        """Load the default RSA private key for SSH authentication.

        Returns:
            Paramiko ``RSAKey`` loaded from ``~/.ssh/id_rsa``.
        """

        filename = os.path.expanduser("~/.ssh/id_rsa")
        try:
            return RSAKey.from_private_key_file(filename)
        except PasswordRequiredException:
            passphrase = self._ssh_passphrase
            return RSAKey.from_private_key_file(filename, password=passphrase)

    @cached_property
    def _ssh_passphrase(self):
        passphrase = os.getenv(self.ssh_key_env)
        if passphrase is None or passphrase == "":
            print(
                f"Avoid providing this each time by adding the {self.ssh_key_env} to FrequenSolve/.env"
            )
            passphrase = getpass.getpass("SSH key passphrase: ")
        return passphrase

    @property
    def duo_code(self):
        """Prompt for and return a site two-factor authentication code."""

        return input("Site 2FA Code:")

    def __str__(self):
        """Don't print credentials."""
        return ""

    def __repr__(self):
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
    def access_key(self):
        """Retrieve AWS access key ID from the configured environment variable."""
        return os.getenv(self.key_env)

    @cached_property
    def secret_key(self):
        """Retrieve AWS secret access key from the configured environment variable."""
        return os.getenv(self.secret_env)
