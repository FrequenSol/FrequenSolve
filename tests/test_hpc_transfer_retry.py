"""Stalled or disconnected HPC transfers retry over a live connection."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.config_file import _slurm_site_kwargs
from frequensolve.orchestrator.sites.hpc import SlurmSite, SlurmSiteConfig
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc import transfer as transfer_module
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager
from frequensolve.orchestrator.utils import ssh as ssh_module
from frequensolve.orchestrator.utils.ssh import SSHProxy

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


class FlakySFTP:
    """SFTP double whose first ``failures`` uploads/downloads raise ``error``."""

    def __init__(self, failures, error=TimeoutError("stalled")):
        self.failures = failures
        self.error = error
        self.calls = 0
        self.published = []
        self.removed = []
        self.sizes = {}

    def _maybe_fail(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error

    def put(self, source, target):
        self._maybe_fail()
        self.sizes[target] = Path(source).stat().st_size

    def get(self, source, target):
        self._maybe_fail()
        Path(target).write_bytes(b"payload")
        self.sizes[source] = 7

    def stat(self, target):
        if target not in self.sizes:
            raise FileNotFoundError(target)
        return SimpleNamespace(st_size=self.sizes[target], st_mode=0o100644)

    def chmod(self, target, mode):
        pass

    def posix_rename(self, source, target):
        self.published.append(target)

    def remove(self, target):
        self.removed.append(target)

    def close(self):
        pass


def _site(sftp, *, attempts=3, stall=120.0, recover=lambda: True, method="sftp"):
    recoveries = []

    def recovery():
        recoveries.append(True)
        return recover()

    site = SimpleNamespace(
        _login_client=SimpleNamespace(
            is_proxy=lambda: method == "rsync",
            get_proxy_details=lambda: ("/tmp/control.sock", "user"),
        ),
        login_client=SimpleNamespace(open_sftp=lambda: sftp),
        transfer_method=method,
        run_login=lambda command: "",
        config=SimpleNamespace(
            hostname="login.example.edu",
            transfer_attempts=attempts,
            transfer_stall_timeout=stall,
        ),
        credentials=SimpleNamespace(username="user"),
        _recover_login_connection=recovery,
    )
    return site, recoveries


@pytest.fixture
def sleeps(monkeypatch):
    delays = []
    monkeypatch.setattr(transfer_module, "_sleep", delays.append)
    return delays


def test_stalled_upload_is_retried_after_checking_the_connection(tmp_path, sleeps):
    local = tmp_path / "input.bin"
    local.write_bytes(b"payload")
    sftp = FlakySFTP(failures=1)
    site, recoveries = _site(sftp)

    SlurmTransferManager(site).put(local, "/work/input.bin")

    assert sftp.published == ["/work/input.bin"]
    assert len(sftp.removed) == 1  # the stalled attempt's partial upload
    assert recoveries == [True]
    assert sleeps == [2.0]


def test_stalled_download_fails_after_bounded_attempts(tmp_path, sleeps):
    sftp = FlakySFTP(failures=10)
    site, recoveries = _site(sftp, stall=300.0)

    with pytest.raises(RuntimeError) as caught:
        SlurmTransferManager(site).get("/work/output.bin", tmp_path / "output.bin")

    message = str(caught.value)
    assert message.startswith("HPC download failed after 3 attempts")
    assert "300 s (transfer_stall_timeout)" in message
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert sftp.calls == 3
    assert sleeps == [2.0, 4.0]
    assert len(recoveries) == 2
    assert not (tmp_path / "output.bin").exists()


def test_lost_connection_without_a_live_replacement_is_not_retried(tmp_path, sleeps):
    local = tmp_path / "input.bin"
    local.write_bytes(b"payload")
    sftp = FlakySFTP(failures=10, error=ConnectionError("private detail"))
    site, recoveries = _site(sftp, recover=lambda: False)

    with pytest.raises(RuntimeError) as caught:
        SlurmTransferManager(site).put(local, "/work/input.bin")

    message = str(caught.value)
    assert "after 1 attempt (ConnectionError)" in message
    assert "frequensolve site connect" in message
    assert "private detail" not in message
    assert sftp.calls == 1 and recoveries == [True] and sleeps == []


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), OSError("quota exceeded")]
)
def test_file_errors_are_not_retried(tmp_path, sleeps, error):
    local = tmp_path / "input.bin"
    local.write_bytes(b"payload")
    sftp = FlakySFTP(failures=10, error=error)
    site, recoveries = _site(sftp)

    with pytest.raises(
        RuntimeError, match=rf"HPC upload failed \({type(error).__name__}\)$"
    ):
        SlurmTransferManager(site).put(local, "/work/input.bin")

    assert sftp.calls == 1 and recoveries == [] and sleeps == []


def test_one_attempt_disables_retries(tmp_path, sleeps):
    local = tmp_path / "input.bin"
    local.write_bytes(b"payload")
    sftp = FlakySFTP(failures=1)
    site, recoveries = _site(sftp, attempts=1)

    with pytest.raises(RuntimeError, match="after 1 attempt"):
        SlurmTransferManager(site).put(local, "/work/input.bin")
    assert sftp.calls == 1 and recoveries == [] and sleeps == []


def test_rsync_stall_is_bounded_retried_and_resumes_partial_files(
    monkeypatch, tmp_path, sleeps
):
    statuses = iter([30, 255, 0])
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=next(statuses), stdout="", stderr="")

    monkeypatch.setattr(transfer_module.subprocess, "run", fake_run)
    monkeypatch.setattr(transfer_module.logger, "isEnabledFor", lambda level: False)
    local = tmp_path / "model.h5"
    local.write_bytes(b"payload")
    site, recoveries = _site(None, stall=90.5, method="rsync")

    SlurmTransferManager(site).put(local, "/work/model.h5")

    assert len(calls) == 3 and sleeps == [2.0, 4.0] and len(recoveries) == 2
    for argv in calls:
        # rsync's I/O timeout stops a stall; --partial lets the retry resume.
        assert "--timeout=91" in argv
        assert "--partial" in argv


def test_rsync_file_errors_are_not_retried(monkeypatch, tmp_path, sleeps):
    calls = []
    monkeypatch.setattr(
        transfer_module.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv)
        or SimpleNamespace(returncode=23, stdout="", stderr=""),
    )
    local = tmp_path / "model.h5"
    local.write_bytes(b"payload")
    site, _ = _site(None, method="rsync")

    with pytest.raises(RuntimeError, match="rsync transfer failed with status 23"):
        SlurmTransferManager(site).put(local, "/work/model.h5")
    assert len(calls) == 1 and sleeps == []


def test_sftp_transfers_use_the_configured_stall_timeout(monkeypatch, tmp_path):
    seen = {}

    def run_with_progress(argv, stdin_text, progress, *, idle_timeout, poll_interval):
        seen.update(idle=idle_timeout, poll=poll_interval)
        return 0, ""

    monkeypatch.setattr(ssh_module, "_run_with_progress", run_with_progress)
    proxy = SSHProxy("/tmp/control.sock", "user", "login", command_timeout=120)
    site, _ = _site(None, stall=600.0)
    site.login_client = proxy

    sftp = SlurmTransferManager(site)._open_sftp()
    sftp.put(str(tmp_path / "a"), "/work/a")

    assert seen == {"idle": 600.0, "poll": 5.0}
    # Metadata commands keep the proxy's own command timeout.
    assert sftp.command_timeout == 120
    assert proxy.open_sftp().stall_timeout == 120


def test_paramiko_sftp_channel_times_out_on_stall():
    timeouts = []
    channel = SimpleNamespace(settimeout=timeouts.append)
    sftp = SimpleNamespace(get_channel=lambda: channel)
    site, _ = _site(sftp, stall=45.0)

    assert SlurmTransferManager(site)._open_sftp() is sftp
    assert timeouts == [45.0]


def test_sftp_connection_failures_are_connection_errors(monkeypatch, tmp_path):
    def run_with_progress(*args, **kwargs):
        return 255, "client_loop: send disconnect: Broken pipe\n"

    monkeypatch.setattr(ssh_module, "_run_with_progress", run_with_progress)
    sftp = SSHProxy("/tmp/control.sock", "user", "login").open_sftp()

    with pytest.raises(ConnectionError):
        sftp.get("/work/a", str(tmp_path / "a"))


def test_transfer_settings_are_validated_and_read_from_site_toml():
    config = SlurmSiteConfig(hostname="login.example.edu")
    assert config.transfer_stall_timeout == 120.0
    assert config.transfer_attempts == 3
    for bad in ({"transfer_stall_timeout": 0}, {"transfer_attempts": 0}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            SlurmSiteConfig(hostname="login.example.edu", **bad)
    kwargs = _slurm_site_kwargs(
        {
            "hostname": "login.example.edu",
            "transfer_stall_timeout": 300,
            "transfer_attempts": 5,
        }
    )
    assert kwargs["config"].transfer_stall_timeout == 300.0
    assert kwargs["config"].transfer_attempts == 5
    assert "transfer_attempts" not in kwargs


def _proxy_site(monkeypatch, *, current_works, replacement):
    monkeypatch.setattr(hpc, "SSHClientClass", lambda client: ("wrapped", client))
    site = object.__new__(SlurmSite)
    site.config = SimpleNamespace(hostname="login.example.edu")
    proxy = SSHProxy("/home/u/.ssh/control/old", "user", "login.example.edu")
    site._login_client = SimpleNamespace(
        is_proxy=lambda: True,
        get_proxy_details=lambda: (proxy.control_path, "user"),
        client=proxy,
    )
    searched = []
    site._authenticator = SimpleNamespace(
        control_socket_works=lambda path, host: current_works,
        find_control_socket=lambda host, exclude=(): searched.append(exclude)
        or replacement,
    )
    return site, searched


def test_recovery_keeps_a_live_control_socket(monkeypatch):
    site, searched = _proxy_site(monkeypatch, current_works=True, replacement=None)
    original = site._login_client
    assert site._recover_login_connection() is True
    assert site._login_client is original and searched == []


def test_recovery_switches_to_another_live_control_socket(monkeypatch):
    replacement = SSHProxy("/home/u/.ssh/control/new", "user", "login.example.edu")
    site, searched = _proxy_site(
        monkeypatch, current_works=False, replacement=replacement
    )
    assert site._recover_login_connection() is True
    assert site._login_client == ("wrapped", replacement)
    assert searched == [("/home/u/.ssh/control/old",)]


def test_recovery_reports_a_dead_connection(monkeypatch):
    site, _ = _proxy_site(monkeypatch, current_works=False, replacement=None)
    assert site._recover_login_connection() is False

    paramiko_site = object.__new__(SlurmSite)
    for active in (True, False):
        paramiko_site._login_client = SimpleNamespace(
            is_proxy=lambda: False,
            get_transport=lambda active=active: SimpleNamespace(
                is_active=lambda: active
            ),
        )
        assert paramiko_site._recover_login_connection() is active


def test_proxy_status_255_is_a_retryable_transport_error(monkeypatch):
    monkeypatch.setattr(
        ssh_module.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(returncode=255, stdout=b"", stderr=b""),
    )
    proxy = SSHProxy("/tmp/control.sock", "user", "login")
    with pytest.raises(ssh_module.SSHTransportError) as caught:
        proxy.exec_command("true")
    assert isinstance(caught.value, RuntimeError)
    assert transfer_module._is_transient(caught.value)
    assert transfer_module._is_transient(subprocess.TimeoutExpired("sftp", 1))
    assert not transfer_module._is_transient(FileNotFoundError("missing"))
