import socket
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.hpc import auth
from frequensolve.orchestrator.sites.hpc.auth import SlurmAuthenticator
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.hpc.slurm_helpers import normalize_slurm_state
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


class Stream:
    def __init__(self, text=""):
        self.text = text

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
    monkeypatch.setattr(auth, "_verify_server_host_key", lambda transport, host: None)
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
            return SimpleNamespace(st_size=999)

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
            payload = tmp_path / "unexpected"
            payload.mkdir(exist_ok=True)
            (payload / "new.txt").write_text("new")
            with tarfile.open(target, "w:gz") as tar:
                tar.add(payload, arcname="unexpected")
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

    with pytest.raises(RuntimeError, match="did not contain its root"):
        SlurmTransferManager(site).get("/work/project", destination)

    assert (destination / "existing.txt").read_text() == "existing"
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
            return SimpleNamespace(st_size=uploaded_source.stat().st_size)

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


@pytest.mark.parametrize(
    "remote_path",
    ["relative/path", "/work/../private", "/work/bad\nname", r"C:\work\file"],
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
