import asyncio
import os
import shutil
import socket
import stat
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.base import JobStatus, RunHandle
from frequensolve.orchestrator.sites.hpc import auth
from frequensolve.orchestrator.sites.hpc.auth import SlurmAuthenticator
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.hpc.slurm_helpers import normalize_slurm_state
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager
from frequensolve.orchestrator.utils import ssh as ssh_module
from frequensolve.orchestrator.utils.ssh import SSHProxy

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


class Stream:
    def __init__(self, text="", *, exit_status=None):
        self.text = text
        if exit_status is not None:
            self.channel = SimpleNamespace(
                recv_exit_status=lambda: exit_status,
            )

    def read(self):
        return self.text.encode()


def _auth_site():
    return SimpleNamespace(
        credentials=SimpleNamespace(
            username="scientist",
            password="private-password",
            duo_code="private-token",
            ssh_key=object(),
        )
    )


def test_login_connection_timeout_is_bounded_and_sanitized(monkeypatch):
    def time_out(address, timeout):
        assert address == ("login.example.edu", 22)
        assert timeout == auth.SSH_CONNECT_TIMEOUT_SECONDS
        raise socket.timeout("private endpoint detail")

    monkeypatch.setattr(auth.socket, "create_connection", time_out)

    with pytest.raises(TimeoutError, match="timed out.*15 seconds") as exc_info:
        SlurmAuthenticator(_auth_site())._interactive_authentication(
            "login.example.edu"
        )

    assert "private endpoint detail" not in str(exc_info.value)


def test_failed_control_socket_probes_exhaust_before_one_interactive_attempt(
    monkeypatch,
):
    probes = []
    site = _auth_site()
    site.config = SimpleNamespace(hostname="login.example.edu")
    site.default_host = None
    authenticator = SlurmAuthenticator(site)
    interactive = []

    def probe(argv, **kwargs):
        probes.append(argv)
        if len(probes) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=255)

    monkeypatch.setattr(auth.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        auth.glob,
        "glob",
        lambda pattern: ["/tmp/control-one", "/tmp/control-two"],
    )
    monkeypatch.setattr(auth.subprocess, "run", probe)
    monkeypatch.setattr(
        authenticator,
        "_interactive_authentication",
        lambda host: interactive.append(host) or "interactive-client",
    )

    assert authenticator.authenticate() == "interactive-client"
    assert len(probes) == 2
    assert interactive == ["login.example.edu"]


def test_authentication_exhaustion_closes_transport_and_hides_prompts(monkeypatch):
    transport_instances = []

    class FakeTransport:
        def __init__(self, sock):
            self.closed = False
            self.responses = None
            transport_instances.append(self)

        def start_client(self, timeout):
            assert timeout == auth.SSH_CONNECT_TIMEOUT_SECONDS
            pass

        def is_authenticated(self):
            return False

        def auth_publickey(self, username, key):
            pass

        def auth_interactive(self, username, handler):
            self.responses = handler(
                "title",
                "instructions",
                [("Password: ", False), ("Verification code: ", False)],
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        auth.socket, "create_connection", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(auth, "Transport", FakeTransport)
    monkeypatch.setattr(
        auth,
        "_verify_server_host_key",
        lambda transport, host, **kwargs: None,
    )
    monkeypatch.setattr(
        "paramiko.agent.Agent",
        lambda: SimpleNamespace(get_keys=lambda: []),
    )

    with pytest.raises(
        auth.AuthenticationException, match="Authentication failed"
    ) as exc_info:
        SlurmAuthenticator(_auth_site())._interactive_authentication(
            "login.example.edu"
        )

    transport = transport_instances[0]
    assert transport.closed is True
    assert transport.responses == ["private-password", "private-token"]
    assert "private-password" not in str(exc_info.value)
    assert "private-token" not in str(exc_info.value)


def test_transport_setup_failure_closes_socket_and_is_sanitized(monkeypatch):
    sock = SimpleNamespace(closed=False)
    sock.close = lambda: setattr(sock, "closed", True)
    monkeypatch.setattr(auth.socket, "create_connection", lambda *args, **kwargs: sock)
    monkeypatch.setattr(
        auth,
        "Transport",
        lambda value: (_ for _ in ()).throw(
            auth.SSHException("private handshake detail")
        ),
    )

    with pytest.raises(RuntimeError, match="SSH transport setup failed") as exc_info:
        SlurmAuthenticator(_auth_site())._interactive_authentication(
            "login.example.edu"
        )

    assert sock.closed is True
    assert "private handshake detail" not in str(exc_info.value)


def test_explicit_known_hosts_binding_skips_control_socket_reuse(monkeypatch, tmp_path):
    site = _auth_site()
    site.config = SimpleNamespace(
        hostname="127.0.0.1",
        ssh_port=50222,
        known_hosts_file=tmp_path / "known_hosts",
        known_hosts_name="[127.0.0.1]:50222",
    )
    site.default_host = None
    authenticator = SlurmAuthenticator(site)
    interactive = []

    monkeypatch.setattr(auth.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        auth.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "explicit host-key policy must bypass unrelated control sockets"
        ),
    )
    monkeypatch.setattr(
        authenticator,
        "_interactive_authentication",
        lambda host: interactive.append(host) or "interactive-client",
    )

    assert authenticator.authenticate() == "interactive-client"
    assert interactive == ["127.0.0.1"]


def test_explicit_ssh_endpoint_is_bound_to_configured_host_key(monkeypatch, tmp_path):
    known_hosts_file = tmp_path / "known_hosts"
    site = _auth_site()
    site.config = SimpleNamespace(
        ssh_port=50222,
        known_hosts_file=known_hosts_file,
        known_hosts_name="[127.0.0.1]:50222",
    )
    connections = []
    verifications = []

    class FakeTransport:
        def __init__(self, sock):
            self.authenticated = False

        def start_client(self, timeout):
            assert timeout == auth.SSH_CONNECT_TIMEOUT_SECONDS

        def auth_publickey(self, username, key):
            self.authenticated = True

        def is_authenticated(self):
            return self.authenticated

        def set_keepalive(self, seconds):
            assert seconds == 120

        def close(self):
            pass

    monkeypatch.setattr(
        auth.socket,
        "create_connection",
        lambda address, timeout: connections.append((address, timeout)) or object(),
    )
    monkeypatch.setattr(auth, "Transport", FakeTransport)
    monkeypatch.setattr(
        auth,
        "_verify_server_host_key",
        lambda transport, host, **kwargs: verifications.append((host, kwargs)),
    )
    monkeypatch.setattr(
        "paramiko.agent.Agent",
        lambda: SimpleNamespace(get_keys=lambda: [object()]),
    )

    SlurmAuthenticator(site)._interactive_authentication("127.0.0.1")

    assert connections == [(("127.0.0.1", 50222), auth.SSH_CONNECT_TIMEOUT_SECONDS)]
    assert verifications == [
        (
            "127.0.0.1",
            {
                "known_hosts_file": known_hosts_file,
                "known_hosts_name": "[127.0.0.1]:50222",
            },
        )
    ]


def test_failed_compute_connection_closes_channel_and_client(monkeypatch):
    channel = SimpleNamespace(closed=False)
    channel.close = lambda: setattr(channel, "closed", True)
    tunnel = SimpleNamespace(open_channel=lambda *args: channel)
    site = SimpleNamespace(
        _login_client=SimpleNamespace(
            is_proxy=lambda: False,
            get_transport=lambda: tunnel,
        ),
        credentials=SimpleNamespace(username="scientist"),
    )
    authenticator = SlurmAuthenticator(site)
    monkeypatch.setattr(authenticator, "get_job_host", lambda job_id: "compute-1")

    class FakeClient:
        def __init__(self):
            self.closed = False

        def load_system_host_keys(self):
            pass

        def connect(self, *args, **kwargs):
            assert kwargs["timeout"] == auth.SSH_CONNECT_TIMEOUT_SECONDS
            assert kwargs["banner_timeout"] == auth.SSH_CONNECT_TIMEOUT_SECONDS
            assert kwargs["auth_timeout"] == auth.SSH_CONNECT_TIMEOUT_SECONDS
            raise auth.SSHException("private provider detail")

        def close(self):
            self.closed = True

    client = FakeClient()
    monkeypatch.setattr(auth, "SSHClient", lambda: client)

    with pytest.raises(
        RuntimeError, match="SSH compute-node connection failed"
    ) as exc_info:
        authenticator.connect_to_job_host(123)

    assert client.closed is True
    assert channel.closed is True
    assert "private provider detail" not in str(exc_info.value)


def test_compute_tunnel_failure_is_sanitized(monkeypatch):
    tunnel = SimpleNamespace(
        open_channel=lambda *args: (_ for _ in ()).throw(
            OSError("private tunnel detail")
        )
    )
    site = SimpleNamespace(
        _login_client=SimpleNamespace(
            is_proxy=lambda: False,
            get_transport=lambda: tunnel,
        ),
        credentials=SimpleNamespace(username="scientist"),
    )
    authenticator = SlurmAuthenticator(site)
    monkeypatch.setattr(authenticator, "get_job_host", lambda job_id: "compute-1")

    with pytest.raises(
        RuntimeError, match="SSH compute-node tunnel failed"
    ) as exc_info:
        authenticator.connect_to_job_host(123)

    assert "private tunnel detail" not in str(exc_info.value)


def test_compute_hostname_is_validated_before_connection_use():
    responses = iter(["R", "compute-1; command"])
    site = SimpleNamespace(run_login=lambda command: next(responses))

    with pytest.raises(RuntimeError, match="invalid compute-node hostname"):
        SlurmAuthenticator(site).get_job_host(123)


def _sftp_site(sftp):
    return SimpleNamespace(
        _login_client=SimpleNamespace(is_proxy=lambda: False),
        login_client=SimpleNamespace(open_sftp=lambda: sftp),
        transfer_method="sftp",
        run_login=lambda command: "",
    )


def test_sftp_upload_size_mismatch_removes_only_temporary_object(tmp_path):
    local = tmp_path / "input.bin"
    local.write_bytes(b"payload")

    class FakeSFTP:
        def __init__(self):
            self.uploaded = []
            self.removed = []
            self.published = []

        def put(self, source, target):
            self.uploaded.append(target)

        def stat(self, target):
            if target == "/work/input.bin":
                raise FileNotFoundError(target)
            return SimpleNamespace(st_size=999)

        def chmod(self, target, mode):
            pass

        def posix_rename(self, source, target):
            self.published.append((source, target))

        def remove(self, target):
            self.removed.append(target)

        def close(self):
            pass

    sftp = FakeSFTP()

    with pytest.raises(RuntimeError, match="size verification failed"):
        SlurmTransferManager(_sftp_site(sftp)).put(local, "/work/input.bin")

    assert len(sftp.uploaded) == 1
    assert sftp.removed == sftp.uploaded
    assert sftp.published == []


def test_sftp_provider_failure_is_sanitized_and_cleans_partial_upload(tmp_path):
    local = tmp_path / "input.bin"
    local.write_bytes(b"payload")

    class FakeSFTP:
        def __init__(self):
            self.removed = []

        def stat(self, target):
            if target == "/work/input.bin":
                raise FileNotFoundError(target)
            return SimpleNamespace(st_size=7)

        def put(self, source, target):
            raise OSError("private remote path and token")

        def remove(self, target):
            self.removed.append(target)

        def close(self):
            pass

    sftp = FakeSFTP()

    with pytest.raises(
        RuntimeError, match=r"HPC upload failed \(OSError\)"
    ) as exc_info:
        SlurmTransferManager(_sftp_site(sftp)).put(local, "/work/input.bin")

    assert len(sftp.removed) == 1
    assert "private remote path" not in str(exc_info.value)
    assert "token" not in str(exc_info.value)


def test_sftp_upload_falls_back_to_standard_rename_for_new_destination(tmp_path):
    local = tmp_path / "run.sh"
    local.write_bytes(b"#!/bin/sh\n")
    local.chmod(0o750)
    calls = []

    class FakeSFTP:
        def stat(self, target):
            if target == "/work/run.sh":
                raise FileNotFoundError(target)
            return SimpleNamespace(st_size=local.stat().st_size)

        def put(self, source, target):
            calls.append(("put", source, target))

        def chmod(self, target, mode):
            calls.append(("chmod", target, mode))

        def posix_rename(self, source, target):
            raise OSError("unsupported extension")

        def rename(self, source, target):
            calls.append(("rename", source, target))

        def remove(self, target):
            calls.append(("remove", target))

        def close(self):
            pass

    SlurmTransferManager(_sftp_site(FakeSFTP())).put(local, "/work/run.sh")

    assert any(call[0] == "rename" for call in calls)
    assert next(call for call in calls if call[0] == "chmod")[2] == 0o750
    assert not any(call[0] == "remove" for call in calls)


def test_sftp_replacement_preserves_remote_mode(tmp_path):
    local = tmp_path / "run.sh"
    local.write_bytes(b"#!/bin/sh\n")
    local.chmod(0o600)
    modes = []

    class FakeSFTP:
        def stat(self, target):
            if target == "/work/run.sh":
                return SimpleNamespace(st_mode=0o100775, st_size=999)
            return SimpleNamespace(st_mode=0o100600, st_size=local.stat().st_size)

        def put(self, source, target):
            pass

        def chmod(self, target, mode):
            modes.append(mode)

        def posix_rename(self, source, target):
            pass

        def remove(self, target):
            pass

        def close(self):
            pass

    SlurmTransferManager(_sftp_site(FakeSFTP())).put(local, "/work/run.sh")

    assert modes == [0o775]


def test_sftp_replacement_requires_atomic_server_support(tmp_path):
    local = tmp_path / "run.sh"
    local.write_bytes(b"#!/bin/sh\n")
    removed = []

    class FakeSFTP:
        def stat(self, target):
            if target == "/work/run.sh":
                return SimpleNamespace(st_mode=0o100755, st_size=999)
            return SimpleNamespace(st_mode=0o100600, st_size=local.stat().st_size)

        def put(self, source, target):
            pass

        def chmod(self, target, mode):
            pass

        def posix_rename(self, source, target):
            raise OSError("unsupported extension")

        def rename(self, source, target):
            raise AssertionError("non-atomic replacement must not be attempted")

        def remove(self, target):
            removed.append(target)

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="cannot atomically replace"):
        SlurmTransferManager(_sftp_site(FakeSFTP())).put(local, "/work/run.sh")

    assert len(removed) == 1


def test_sftp_download_mismatch_preserves_existing_destination(tmp_path):
    destination = tmp_path / "result.bin"
    destination.write_bytes(b"existing")

    class FakeSFTP:
        def stat(self, target):
            return SimpleNamespace(st_mode=0o100644, st_size=999)

        def get(self, source, target):
            Path(target).write_bytes(b"partial")

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="size verification failed"):
        SlurmTransferManager(_sftp_site(FakeSFTP())).get(
            "/work/result.bin", destination
        )

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.glob("*.partial")) == []


def test_sftp_download_preserves_existing_destination_mode(tmp_path):
    destination = tmp_path / "result.bin"
    destination.write_bytes(b"existing")
    destination.chmod(0o640)

    class FakeSFTP:
        def stat(self, target):
            return SimpleNamespace(st_mode=0o100755, st_size=3)

        def get(self, source, target):
            Path(target).write_bytes(b"new")

        def close(self):
            pass

    SlurmTransferManager(_sftp_site(FakeSFTP())).get("/work/result.bin", destination)

    assert destination.read_bytes() == b"new"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_sftp_download_new_destination_uses_local_creation_mode(tmp_path):
    destination = tmp_path / "result.bin"
    expected = tmp_path / "expected.bin"
    expected.touch()
    expected_mode = stat.S_IMODE(expected.stat().st_mode)
    expected.unlink()

    class FakeSFTP:
        def stat(self, target):
            return SimpleNamespace(st_mode=0o100750, st_size=3)

        def get(self, source, target):
            Path(target).write_bytes(b"new")

        def close(self):
            pass

    SlurmTransferManager(_sftp_site(FakeSFTP())).get("/work/result.bin", destination)

    assert stat.S_IMODE(destination.stat().st_mode) == expected_mode


def test_directory_download_failure_preserves_existing_tree_and_cleans_remote(
    tmp_path,
):
    destination = tmp_path / "project"
    destination.mkdir()
    (destination / "existing.txt").write_text("existing")
    commands = []
    archive_size = None

    class FakeSFTP:
        def stat(self, target):
            if target == "/work/project":
                return SimpleNamespace(st_mode=0o040755)
            return SimpleNamespace(st_mode=0o100644, st_size=archive_size)

        def get(self, source, target):
            nonlocal archive_size
            Path(target).write_bytes(b"not a tar archive")
            archive_size = Path(target).stat().st_size

        def close(self):
            pass

    local_tmp = tmp_path / "transfer-tmp"
    config = tmp_path / "site.toml"
    config.write_text(f'[host]\ntmp_dir = "{local_tmp.as_posix()}"\n')
    site = _sftp_site(FakeSFTP())
    site._site_config_path = config
    site.remote_tmp_dir = Path("/remote/tmp")
    site.run_login = lambda command: commands.append(command) or ""
    site.run_login_cmd = lambda command: (
        commands.append(command)
        or (
            None,
            Stream(),
            Stream(),
        )
    )

    with pytest.raises(RuntimeError, match=r"HPC download failed \(ReadError\)"):
        SlurmTransferManager(site).get("/work/project", destination)

    assert (destination / "existing.txt").read_text() == "existing"
    assert any(command.startswith("rm -f /remote/tmp/") for command in commands)
    assert list(local_tmp.glob("*.tar.gz")) == []


def test_directory_download_nonzero_exit_never_publishes_valid_archive(tmp_path):
    destination = tmp_path / "project"
    destination.mkdir()
    (destination / "existing.txt").write_text("existing")
    downloads = []
    commands = []
    archive_size = None

    class FakeSFTP:
        def stat(self, target):
            if target == "/work/project":
                return SimpleNamespace(st_mode=0o040755)
            return SimpleNamespace(st_mode=0o100644, st_size=archive_size)

        def get(self, source, target):
            nonlocal archive_size
            downloads.append((source, target))
            payload = tmp_path / "new.txt"
            payload.write_text("new")
            with tarfile.open(target, "w:gz") as tar:
                tar.add(payload, arcname="new.txt")
            archive_size = Path(target).stat().st_size

        def close(self):
            pass

    site = _sftp_site(FakeSFTP())
    site.remote_tmp_dir = Path("/remote/tmp")
    site.run_login = lambda command: commands.append(command) or ""
    site.run_login_cmd = lambda command: (
        commands.append(command)
        or (
            None,
            Stream(exit_status=1),
            Stream(),
        )
    )

    with pytest.raises(RuntimeError, match="Remote directory archive failed"):
        SlurmTransferManager(site).get("/work/project", destination)

    assert downloads == []
    assert (destination / "existing.txt").read_text() == "existing"
    assert any(command.startswith("rm -f /remote/tmp/") for command in commands)


def test_directory_download_resolves_only_symlinked_remote_root(tmp_path):
    destination = tmp_path / "project-link"
    commands = []
    archive_size = None

    class FakeSFTP:
        def stat(self, target):
            if target == "/work/project-link":
                return SimpleNamespace(st_mode=0o040755)
            return SimpleNamespace(st_mode=0o100644, st_size=archive_size)

        def get(self, source, target):
            nonlocal archive_size
            payload = tmp_path / "new.txt"
            payload.write_text("new")
            with tarfile.open(target, "w:gz") as tar:
                tar.add(payload, arcname="new.txt")
            archive_size = Path(target).stat().st_size

        def close(self):
            pass

    local_tmp = tmp_path / "transfer-tmp"
    config = tmp_path / "site.toml"
    config.write_text(f'[host]\ntmp_dir = "{local_tmp.as_posix()}"\n')
    site = _sftp_site(FakeSFTP())
    site._site_config_path = config
    site.remote_tmp_dir = Path("/remote/tmp")
    site.run_login = lambda command: commands.append(command) or ""
    site.run_login_cmd = lambda command: (
        commands.append(command)
        or (
            None,
            Stream(),
            Stream(),
        )
    )

    SlurmTransferManager(site).get("/work/project-link", destination)

    assert (destination / "new.txt").read_text() == "new"
    archive_command = next(command for command in commands if "tar czf" in command)
    assert "readlink -f -- /work/project-link" in archive_command
    assert '-C "$resolved_dir" .' in archive_command
    assert "--dereference" not in archive_command
    assert any(command.startswith("rm -f /remote/tmp/") for command in commands)
    assert list(local_tmp.glob("*.tar.gz")) == []


def test_directory_upload_failure_cleans_remote_archive_and_hides_stderr(tmp_path):
    source = tmp_path / "project"
    source.mkdir()
    (source / "input.txt").write_text("input")
    commands = []
    uploaded_source = None

    class FakeSFTP:
        def put(self, source_path, target):
            nonlocal uploaded_source
            uploaded_source = Path(source_path)

        def stat(self, target):
            if target.startswith("/remote/tmp/frequensolve-") and not target.endswith(
                ".partial"
            ):
                raise FileNotFoundError(target)
            return SimpleNamespace(st_size=uploaded_source.stat().st_size)

        def chmod(self, target, mode):
            pass

        def posix_rename(self, source_path, target):
            pass

        def remove(self, target):
            pass

        def close(self):
            pass

    local_tmp = tmp_path / "transfer-tmp"
    config = tmp_path / "site.toml"
    config.write_text(f'[host]\ntmp_dir = "{local_tmp.as_posix()}"\n')
    site = _sftp_site(FakeSFTP())
    site._site_config_path = config
    site.remote_tmp_dir = Path("/remote/tmp")
    site.run_login = lambda command: commands.append(command) or ""
    site.run_login_cmd = lambda command: (
        commands.append(command)
        or (
            None,
            Stream(),
            Stream("private extraction detail"),
        )
    )

    with pytest.raises(
        RuntimeError, match="Remote directory extraction failed"
    ) as exc_info:
        SlurmTransferManager(site).put(source, "/work/project")

    assert "private extraction detail" not in str(exc_info.value)
    assert any(command.startswith("rm -f /remote/tmp/") for command in commands)


def test_directory_upload_tar_failure_preserves_existing_remote_tree(tmp_path):
    source = tmp_path / "local" / "project"
    source.mkdir(parents=True)
    (source / "new.txt").write_text("new")

    remote_parent = tmp_path / "remote"
    destination = remote_parent / "project"
    destination.mkdir(parents=True)
    (destination / "existing.txt").write_text("existing")
    remote_tmp = tmp_path / "remote-tmp"
    corrupt_archive = True

    class LocalSFTP:
        def put(self, source_path, target):
            shutil.copyfile(source_path, target)

        def stat(self, target):
            return os.stat(target)

        def chmod(self, target, mode):
            os.chmod(target, mode)

        def posix_rename(self, source_path, target):
            os.replace(source_path, target)
            if corrupt_archive and Path(target).parent == remote_tmp:
                Path(target).write_bytes(b"corrupt archive")

        def remove(self, target):
            Path(target).unlink(missing_ok=True)

        def close(self):
            pass

    def run(command):
        result = subprocess.run(command, shell=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("remote command failed")
        return result.stdout.decode().strip()

    def run_cmd(command):
        result = subprocess.run(command, shell=True, capture_output=True, check=False)
        return (
            None,
            Stream(result.stdout.decode(), exit_status=result.returncode),
            Stream(result.stderr.decode()),
        )

    site = _sftp_site(LocalSFTP())
    site.remote_tmp_dir = remote_tmp
    site.run_login = run
    site.run_login_cmd = run_cmd

    manager = SlurmTransferManager(site)
    with pytest.raises(RuntimeError, match="Remote directory extraction failed"):
        manager.put(source, destination)

    assert (destination / "existing.txt").read_text() == "existing"
    assert not (destination / "new.txt").exists()
    assert list(remote_parent.glob(".project.frequensolve-*")) == []
    assert list(remote_tmp.glob("frequensolve-*")) == []

    corrupt_archive = False
    manager.put(source, destination)

    assert (destination / "new.txt").read_text() == "new"
    assert not (destination / "existing.txt").exists()
    assert list(remote_parent.glob(".project.frequensolve-*")) == []
    assert list(remote_tmp.glob("frequensolve-*")) == []


@pytest.mark.parametrize(
    "remote_path",
    ["/work/../private", "/work/bad\nname", r"C:\work\file"],
)
def test_transfer_rejects_unsafe_remote_paths_without_remote_calls(
    tmp_path, remote_path
):
    commands = []
    site = _sftp_site(SimpleNamespace())
    site.run_login = lambda command: commands.append(command)
    local = tmp_path / "input.bin"
    local.write_bytes(b"payload")

    with pytest.raises(ValueError, match="traversal-free"):
        SlurmTransferManager(site).put(local, remote_path)

    assert commands == []


def test_transfer_accepts_traversal_free_relative_remote_path(tmp_path):
    local = tmp_path / "output.h5"
    local.write_bytes(b"payload")
    commands = []
    published = []

    class FakeSFTP:
        def stat(self, target):
            if target == "results/output.h5":
                raise FileNotFoundError(target)
            return SimpleNamespace(st_size=local.stat().st_size)

        def put(self, source, target):
            pass

        def chmod(self, target, mode):
            pass

        def posix_rename(self, source, target):
            published.append((source, target))

        def remove(self, target):
            pass

        def close(self):
            pass

    site = _sftp_site(FakeSFTP())
    site.run_login = lambda command: commands.append(command) or ""

    SlurmTransferManager(site).put(local, "results/output.h5")

    assert commands == ["mkdir -p -- results"]
    assert len(published) == 1
    assert published[0][1] == "results/output.h5"


def test_download_accepts_traversal_free_relative_remote_path(tmp_path):
    destination = tmp_path / "output.h5"

    class FakeSFTP:
        def stat(self, target):
            assert target == "results/output.h5"
            return SimpleNamespace(st_mode=0o100664, st_size=7)

        def get(self, source, target):
            Path(target).write_bytes(b"payload")

        def close(self):
            pass

    SlurmTransferManager(_sftp_site(FakeSFTP())).get("results/output.h5", destination)

    assert destination.read_bytes() == b"payload"


def test_rsync_failure_hides_provider_output(monkeypatch):
    site = SimpleNamespace(
        _login_client=SimpleNamespace(
            is_proxy=lambda: True,
            get_proxy_details=lambda: ("/tmp/control.sock", "scientist"),
        ),
        transfer_method="rsync",
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=23,
            stderr="private provider output",
        ),
    )

    with pytest.raises(RuntimeError, match="status 23") as exc_info:
        SlurmTransferManager(site)._run_rsync(
            "/tmp/source", "scientist@login.example.edu:/work/target"
        )

    assert "private provider output" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("username", "hostname"),
    [
        ("scientist;command", "login.example.edu"),
        ("scientist", "login.example.edu\ncommand"),
    ],
)
def test_rsync_remote_spec_rejects_unsafe_endpoint_values(username, hostname):
    site = SimpleNamespace(
        credentials=SimpleNamespace(username=username),
        config=SimpleNamespace(hostname=hostname),
    )

    with pytest.raises(ValueError, match="unsafe characters"):
        SlurmTransferManager(site)._remote_spec("/work/result")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PD", "pending"),
        ("CF", "pending"),
        ("CONFIGURING", "pending"),
        ("R", "running"),
        ("COMPLETING", "running"),
        ("CD", "complete"),
        ("FAILED", "failed"),
        ("NODE_FAIL", "failed"),
        ("PREEMPTED", "failed"),
        ("TO", "timeout"),
        ("CANCELLED by 123", "cancelled"),
        ("future_state", "unknown"),
    ],
)
def test_scheduler_state_contract(raw, expected):
    assert normalize_slurm_state(raw) == expected


def test_status_falls_back_to_accounting_and_uses_first_known_state():
    site = object.__new__(SlurmSite)
    responses = iter(["", "PREEMPTED\nCOMPLETED\n"])
    commands = []
    site.run_login = lambda command: commands.append(command) or next(responses)

    assert site.update_status("123") == "failed"
    assert commands == [
        "squeue -j 123 -h -o %t",
        "sacct -j 123 -n -o State%20",
    ]


def test_status_rejects_command_injection_before_scheduler_call():
    site = object.__new__(SlurmSite)
    calls = []
    site.run_login = lambda command: calls.append(command) or "R"

    with pytest.raises(ValueError, match="job id must be numeric"):
        site.update_status("123; touch /tmp/private")

    assert calls == []


def test_cancel_is_idempotent_and_sanitizes_scheduler_failure():
    site = object.__new__(SlurmSite)
    calls = []
    site.run_login_cmd = lambda command: (
        calls.append(command)
        or (
            None,
            Stream(),
            Stream(),
        )
    )

    assert site.cancel_job("123") is True
    assert site.cancel_job("123") is False
    assert calls == ["scancel 123"]

    failing = object.__new__(SlurmSite)
    failing.run_login_cmd = lambda command: (
        None,
        Stream(),
        Stream("private scheduler detail"),
    )
    with pytest.raises(RuntimeError, match="SLURM cancellation failed") as exc_info:
        failing.cancel_job("124")
    assert "private scheduler detail" not in str(exc_info.value)


def test_cancel_nonzero_exit_with_empty_stderr_can_be_retried():
    site = object.__new__(SlurmSite)
    calls = []
    exit_statuses = iter([1, 0])

    def run_login_cmd(command):
        calls.append(command)
        exit_status = next(exit_statuses)
        return None, Stream(exit_status=exit_status), Stream()

    site.run_login_cmd = run_login_cmd

    with pytest.raises(RuntimeError, match="SLURM cancellation failed"):
        site.cancel_job("125")

    assert site.cancel_job("125") is True
    assert calls == ["scancel 125", "scancel 125"]


def test_cancel_observes_ssh_proxy_exit_status_with_empty_stderr(monkeypatch):
    proxy = SSHProxy("/tmp/fake-control", "scientist", "login.example.edu")
    monkeypatch.setattr(
        proxy,
        "_exec_on_login",
        lambda command, term=False, timeout=None: SimpleNamespace(
            stdout=b"",
            stderr=b"",
            returncode=1,
        ),
    )
    site = object.__new__(SlurmSite)
    site.run_login_cmd = proxy.exec_command

    with pytest.raises(RuntimeError, match="SLURM cancellation failed"):
        site.cancel_job("126")


def test_ssh_proxy_sftp_reuses_verified_control_socket(monkeypatch, tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "ssh":
            return SimpleNamespace(returncode=0, stdout="81a4 3\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ssh_module.subprocess, "run", run)
    proxy = SSHProxy("/tmp/verified-control", "scientist", "login.example.edu")
    sftp = proxy.open_sftp()
    local = tmp_path / "input.bin"
    local.write_bytes(b"abc")
    fetched = tmp_path / "output.bin"

    attributes = sftp.stat("/remote/input.bin")
    sftp.put(str(local), "/remote/.input.partial")
    sftp.chmod("/remote/.input.partial", 0o640)
    sftp.posix_rename("/remote/.input.partial", "/remote/input.bin")
    sftp.get("/remote/input.bin", str(fetched))
    sftp.remove("/remote/input.bin")
    sftp.close()

    assert attributes.st_mode == 0o100644
    assert attributes.st_size == 3
    assert [call[0][0] for call in calls] == [
        "ssh",
        "sftp",
        "sftp",
        "sftp",
        "sftp",
        "sftp",
    ]
    for argv, kwargs in calls:
        assert "ControlPath=/tmp/verified-control" in argv
        assert kwargs["timeout"] == 120
        assert "private-token" not in str(kwargs)


def test_slurm_wait_timeout_cancels_before_returning_timeout_result():
    site = object.__new__(SlurmSite)
    site.config = SimpleNamespace(poll_interval=0.0)
    site._poll_run = lambda run: JobStatus(state="running", job_id=run.id)
    cancelled = []
    finalized = []
    site.cancel_job = lambda job_id: cancelled.append(job_id) or True
    site._finalize_run_record = lambda run, status: finalized.append(status.state)
    run = site.handle(SimpleNamespace(), job_id="127", mode="batch")

    result = run.wait(timeout=0, poll_interval=0, check=False)

    assert result.status.state == "timeout"
    assert cancelled == ["127"]
    assert finalized == ["timeout"]


def test_slurm_wait_timeout_surfaces_sanitized_cancellation_failure():
    site = object.__new__(SlurmSite)
    site.config = SimpleNamespace(poll_interval=0.0)
    site._poll_run = lambda run: JobStatus(state="running", job_id=run.id)
    site.cancel_job = lambda job_id: (_ for _ in ()).throw(
        RuntimeError("SLURM cancellation failed")
    )
    run = site.handle(SimpleNamespace(), job_id="128", mode="batch")

    with pytest.raises(RuntimeError, match="^SLURM cancellation failed$"):
        run.wait(timeout=0, poll_interval=0, check=False)


def test_attached_slurm_async_timeout_requests_cancellation():
    site = object.__new__(SlurmSite)
    cancelled = []
    site.cancel_job = lambda job_id: cancelled.append(job_id) or True
    site._finalize_run_record = lambda run, status: None

    async def wait_for_timeout():
        future = asyncio.get_running_loop().create_future()
        run = RunHandle(site=site, job=SimpleNamespace(), id="129", mode="attached")
        run.backend["future"] = future
        return await site._wait_attached_run_async(run, timeout=0)

    result = asyncio.run(wait_for_timeout())

    assert result.status.state == "timeout"
    assert cancelled == ["129"]


def test_attached_slurm_sync_deadline_requests_cancellation():
    site = object.__new__(SlurmSite)
    site._is_notebook = False
    cancelled = []
    site.cancel_job = lambda job_id: cancelled.append(job_id) or True
    site._finalize_run_record = lambda run, status: None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        run = RunHandle(site=site, job=SimpleNamespace(), id="130", mode="attached")
        run.backend["future"] = loop.create_future()
        result = site._wait_attached_run(run, timeout=0)
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    assert result.status.state == "timeout"
    assert cancelled == ["130"]


def test_attached_slurm_sync_task_timeout_propagates_without_cancellation():
    site = object.__new__(SlurmSite)
    site._is_notebook = False
    cancelled = []
    site.cancel_job = lambda job_id: cancelled.append(job_id) or True
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        future = loop.create_future()
        future.set_exception(TimeoutError("task-level timeout"))
        run = RunHandle(site=site, job=SimpleNamespace(), id="131", mode="attached")
        run.backend["future"] = future
        with pytest.raises(TimeoutError, match="task-level timeout"):
            site._wait_attached_run(run, timeout=1)
        assert site._poll_attached_run(run).state == "failed"
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    assert cancelled == []


def test_attached_slurm_async_task_timeout_propagates_without_cancellation():
    site = object.__new__(SlurmSite)
    cancelled = []
    site.cancel_job = lambda job_id: cancelled.append(job_id) or True

    async def wait_for_task_failure():
        future = asyncio.get_running_loop().create_future()
        future.set_exception(TimeoutError("task-level timeout"))
        run = RunHandle(site=site, job=SimpleNamespace(), id="132", mode="attached")
        run.backend["future"] = future
        with pytest.raises(TimeoutError, match="task-level timeout"):
            await site._wait_attached_run_async(run, timeout=1)
        assert site._poll_attached_run(run).state == "failed"

    asyncio.run(wait_for_task_failure())

    assert cancelled == []
