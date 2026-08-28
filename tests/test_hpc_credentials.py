import pytest

from frequensolve.orchestrator.sites.hpc import auth
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.hpc.stampede3 import TACCLoginCredentials
from frequensolve.orchestrator.utils.credentials import Credentials

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


class MemoryCredentialStore:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get_secret(self, key):
        return self.values.get(key)

    def set_secret(self, key, value):
        self.values[key] = value


def test_tacc_credentials_use_established_environment_names(monkeypatch):
    monkeypatch.setenv("TACC_USERNAME", "tacc-user")
    monkeypatch.setenv("TACC_PASSWORD", "tacc-password")
    monkeypatch.setenv("HPC_USERNAME", "generic-user")
    monkeypatch.setenv("HPC_PASSWORD", "generic-password")
    credentials = TACCLoginCredentials(credential_store=MemoryCredentialStore())

    assert credentials.username == "tacc-user"
    assert credentials.password == "tacc-password"


def test_configured_username_and_keyring_password_precede_environment(
    monkeypatch,
):
    store = MemoryCredentialStore({"cluster:config-user:password": "stored-password"})
    monkeypatch.setenv("HPC_USERNAME", "environment-user")
    monkeypatch.delenv("HPC_PASSWORD", raising=False)
    monkeypatch.setattr(
        "frequensolve.orchestrator.utils.credentials.getpass.getpass",
        lambda prompt: pytest.fail("password prompt should not be used"),
    )

    credentials = Credentials(
        username="config-user",
        credential="cluster",
        credential_store=store,
    )

    assert credentials.username == "config-user"
    assert credentials.password == "stored-password"


def test_prompted_password_is_saved_only_after_authentication_success(monkeypatch):
    store = MemoryCredentialStore()
    monkeypatch.delenv("HPC_PASSWORD", raising=False)
    monkeypatch.setattr(
        "frequensolve.orchestrator.utils.credentials.getpass.getpass",
        lambda prompt: "entered-password",
    )
    credentials = Credentials(
        username="user",
        credential="cluster",
        credential_store=store,
    )

    assert credentials.password == "entered-password"
    assert store.values == {}

    credentials.persist_pending()

    assert store.values == {"cluster:user:password": "entered-password"}


def test_environment_password_remains_an_unsaved_automation_override(monkeypatch):
    store = MemoryCredentialStore({"cluster:user:password": "stored-password"})
    monkeypatch.setenv("HPC_PASSWORD", "environment-password")
    credentials = Credentials(
        username="user",
        credential="cluster",
        credential_store=store,
    )

    assert credentials.password == "environment-password"
    credentials.persist_pending()
    assert store.values == {"cluster:user:password": "stored-password"}


def test_prompted_ssh_passphrase_is_saved_only_after_authentication(monkeypatch):
    store = MemoryCredentialStore()
    monkeypatch.delenv("SSH_PASSPHRASE", raising=False)
    monkeypatch.setattr(
        "frequensolve.orchestrator.utils.credentials.getpass.getpass",
        lambda prompt: "entered-passphrase",
    )
    credentials = Credentials(
        username="user",
        credential="cluster",
        credential_store=store,
    )

    assert credentials._ssh_passphrase == "entered-passphrase"
    assert store.values == {}

    credentials.persist_pending()

    assert store.values == {"cluster:user:ssh-passphrase": "entered-passphrase"}


def test_two_factor_code_is_hidden_and_never_saved(monkeypatch):
    store = MemoryCredentialStore()
    monkeypatch.setattr(
        "frequensolve.orchestrator.utils.credentials.getpass.getpass",
        lambda prompt: "123456",
    )
    credentials = Credentials(
        username="user",
        credential="cluster",
        credential_store=store,
    )

    assert credentials.duo_code == "123456"
    credentials.persist_pending()
    assert store.values == {}


def test_site_persists_pending_credentials_after_authenticator_returns():
    class FakeCredentials:
        persisted = False

        def persist_pending(self):
            self.persisted = True

    class FakeAuthenticator:
        def authenticate(self, host):
            return "client"

    site = object.__new__(SlurmSite)
    site.credentials = FakeCredentials()
    site._authenticator = FakeAuthenticator()

    assert SlurmSite.authenticate(site, "login.example.edu") == "client"
    assert site.credentials.persisted is True


def test_site_does_not_persist_credentials_after_failed_authentication():
    class FakeCredentials:
        persisted = False

        def persist_pending(self):
            self.persisted = True

    class FakeAuthenticator:
        def authenticate(self, host):
            raise RuntimeError("authentication failed")

    site = object.__new__(SlurmSite)
    site.credentials = FakeCredentials()
    site._authenticator = FakeAuthenticator()

    with pytest.raises(RuntimeError, match="authentication failed"):
        SlurmSite.authenticate(site, "login.example.edu")
    assert site.credentials.persisted is False


def test_host_key_verification_rejects_unknown_hosts(monkeypatch):
    class FakeHostKeys:
        def load(self, path):
            pass

        def lookup(self, host):
            return None

    class FakeTransport:
        def get_remote_server_key(self):
            return type("Key", (), {"get_name": lambda self: "ssh-ed25519"})()

    monkeypatch.setattr(auth, "HostKeys", FakeHostKeys)

    with pytest.raises(auth.SSHException, match="unknown or does not match"):
        auth._verify_server_host_key(FakeTransport(), "login.example.edu")


def test_host_key_verification_accepts_matching_known_host(monkeypatch):
    server_key = type("Key", (), {"get_name": lambda self: "ssh-ed25519"})()

    class FakeHostKeys:
        def load(self, path):
            pass

        def lookup(self, host):
            return {"ssh-ed25519": server_key}

    class FakeTransport:
        def get_remote_server_key(self):
            return server_key

    monkeypatch.setattr(auth, "HostKeys", FakeHostKeys)

    auth._verify_server_host_key(FakeTransport(), "login.example.edu")


def test_host_key_verification_derives_nonstandard_port_lookup_name(
    monkeypatch, tmp_path
):
    server_key = type("Key", (), {"get_name": lambda self: "ssh-ed25519"})()
    loaded = []
    lookups = []

    class FakeHostKeys:
        def load(self, path):
            loaded.append(path)

        def lookup(self, host):
            lookups.append(host)
            return {"ssh-ed25519": server_key}

    class FakeTransport:
        def get_remote_server_key(self):
            return server_key

    monkeypatch.setattr(auth, "HostKeys", FakeHostKeys)
    known_hosts = tmp_path / "approved_known_hosts"

    auth._verify_server_host_key(
        FakeTransport(),
        "127.0.0.1",
        port=50222,
        known_hosts_file=known_hosts,
    )

    assert loaded == [str(known_hosts)]
    assert lookups == ["[127.0.0.1]:50222"]
