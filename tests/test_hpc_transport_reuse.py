import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.hpc import auth
from frequensolve.orchestrator.sites.hpc.auth import SlurmAuthenticator
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager
from frequensolve.orchestrator.utils.ssh import (
    SSH_COMMAND_TIMEOUT_SECONDS,
    SSHProxy,
)


def _result(*, stdout=b"", stderr=b"", returncode=0):
    return SimpleNamespace(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


def _stream(text=""):
    return SimpleNamespace(read=lambda: text.encode())


def _host_config(tmp_path, tmp_dir):
    config_path = tmp_path / "site.toml"
    config_path.write_text(f'[host]\ntmp_dir = "{Path(tmp_dir).as_posix()}"\n')
    return config_path


def test_control_socket_probe_is_noninteractive_and_bounded(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _result(stdout="Connection test\n")

    site = SimpleNamespace(
        config=SimpleNamespace(hostname="login.example.edu"),
        default_host=None,
        credentials=SimpleNamespace(username="user"),
    )
    monkeypatch.setattr(auth.os.path, "exists", lambda path: True)
    monkeypatch.setattr(auth.glob, "glob", lambda pattern: ["/tmp/control.sock"])
    monkeypatch.setattr(auth.subprocess, "run", fake_run)

    client = SlurmAuthenticator(site).authenticate()

    assert isinstance(client, SSHProxy)
    argv, kwargs = calls[0]
    assert "BatchMode=yes" in argv
    assert "ControlPath=/tmp/control.sock" in argv
    assert kwargs["timeout"] > 0


def test_ssh_proxy_reuses_control_socket_for_login_commands(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _result(stdout=b"ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    proxy = SSHProxy("/tmp/control.sock", "user", "login.example.edu")

    result = proxy._exec_on_login("echo ok")

    assert result.stdout == b"ok\n"
    argv, kwargs = calls[0]
    assert "BatchMode=yes" in argv
    assert "ControlPath=/tmp/control.sock" in argv
    assert kwargs["timeout"] == SSH_COMMAND_TIMEOUT_SECONDS


def test_ssh_proxy_reports_command_timeout(monkeypatch):
    def time_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", time_out)
    proxy = SSHProxy(
        "/tmp/control.sock",
        "user",
        "login.example.edu",
        command_timeout=3,
    )

    with pytest.raises(TimeoutError, match="login.example.edu.*3 seconds"):
        proxy._exec_on_login("squeue")


def test_ssh_proxy_reports_expired_control_socket(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: _result(
            stderr=b"Control socket connect: Connection refused\n",
            returncode=255,
        ),
    )
    proxy = SSHProxy("/tmp/control.sock", "user", "login.example.edu")

    with pytest.raises(RuntimeError, match="Control socket connect"):
        proxy._exec_on_login("squeue")


def test_rsync_reuses_authenticated_control_socket(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _result()

    login_client = SimpleNamespace(
        is_proxy=lambda: True,
        get_proxy_details=lambda: ("/tmp/control.sock", "user"),
    )
    site = SimpleNamespace(
        _login_client=login_client,
        transfer_method="rsync",
        verbose=False,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    SlurmTransferManager(site)._run_rsync(
        "/tmp/source",
        "user@login.example.edu:/work/target",
    )

    argv, _ = calls[0]
    ssh_command = argv[argv.index("-e") + 1]
    assert "BatchMode=yes" in ssh_command
    assert "ControlPath=/tmp/control.sock" in ssh_command


def test_debug_rsync_streams_file_names_and_progress(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _result()

    login_client = SimpleNamespace(
        is_proxy=lambda: True,
        get_proxy_details=lambda: ("/tmp/control.sock", "user"),
    )
    site = SimpleNamespace(
        _login_client=login_client,
        transfer_method="rsync",
        verbose=False,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "frequensolve.orchestrator.sites.hpc.transfer.logger.isEnabledFor",
        lambda level: True,
    )

    SlurmTransferManager(site)._run_rsync(
        "/tmp/source",
        "user@login.example.edu:/work/target",
    )

    argv, kwargs = calls[0]
    assert "-avzP" in argv
    assert "capture_output" not in kwargs
    assert kwargs["text"] is True


def test_info_rsync_is_quiet_even_when_site_is_verbose(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _result()

    login_client = SimpleNamespace(
        is_proxy=lambda: True,
        get_proxy_details=lambda: ("/tmp/control.sock", "user"),
    )
    site = SimpleNamespace(
        _login_client=login_client,
        transfer_method="rsync",
        verbose=True,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "frequensolve.orchestrator.sites.hpc.transfer.logger.isEnabledFor",
        lambda level: False,
    )

    SlurmTransferManager(site)._run_rsync(
        "/tmp/source",
        "user@login.example.edu:/work/target",
    )

    argv, kwargs = calls[0]
    assert "-az" in argv
    assert "--partial" in argv
    assert "-avzP" not in argv
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_rsync_configuration_uses_authenticated_sftp_without_control_socket(
    monkeypatch,
    tmp_path,
):
    transferred = []

    class FakeSFTP:
        def put(self, source, target):
            transferred.append((source, target))

        def close(self):
            pass

    login = SimpleNamespace(open_sftp=lambda: FakeSFTP())
    site = SimpleNamespace(
        _login_client=SimpleNamespace(is_proxy=lambda: False),
        login_client=login,
        transfer_method="rsync",
        verbose=False,
        run_login=lambda command: "",
    )
    local_file = tmp_path / "input.json"
    local_file.write_text("{}")

    SlurmTransferManager(site).put(local_file, Path("/work/input.json"))

    assert transferred == [(str(local_file), "/work/input.json")]


def test_sftp_directory_put_uses_configured_tmp_dirs(tmp_path):
    local_tmp = tmp_path / "local-tmp"
    local_dir = tmp_path / "payload"
    local_dir.mkdir()
    (local_dir / "input.json").write_text("{}")
    transfers = []
    commands = []

    class FakeSFTP:
        def put(self, source, target):
            transfers.append((Path(source), target))

        def close(self):
            pass

    login = SimpleNamespace(open_sftp=lambda: FakeSFTP())
    site = SimpleNamespace(
        _login_client=SimpleNamespace(is_proxy=lambda: False),
        login_client=login,
        transfer_method="sftp",
        verbose=False,
        remote_tmp_dir=Path("/remote/tmp/frequensolve"),
        _site_config_path=_host_config(tmp_path, local_tmp),
        run_login=lambda command: commands.append(command) or "",
        run_login_cmd=lambda command: commands.append(command)
        or (None, _stream(), _stream()),
    )

    SlurmTransferManager(site).put(local_dir, Path("/work/project"))

    assert transfers[0][0].parent == local_tmp
    assert transfers[0][1].startswith("/remote/tmp/frequensolve/")
    assert commands[0] == "mkdir -p /work"
    assert "mkdir -p /remote/tmp/frequensolve" in commands
    assert any("tar xzf /remote/tmp/frequensolve/" in command for command in commands)
    assert not (tmp_path / "project.tar.gz").exists()


def test_sftp_directory_get_uses_configured_tmp_dirs(tmp_path):
    local_tmp = tmp_path / "local-tmp"
    payload = tmp_path / "remote-payload"
    payload.mkdir()
    (payload / "output.json").write_text("{}")
    downloads = []
    commands = []

    class FakeSFTP:
        def stat(self, path):
            return SimpleNamespace(st_mode=0o040755)

        def get(self, remote, local):
            downloads.append((remote, Path(local)))
            with tarfile.open(local, "w:gz") as tar:
                tar.add(payload, arcname="project")

        def close(self):
            pass

    login = SimpleNamespace(open_sftp=lambda: FakeSFTP())
    site = SimpleNamespace(
        _login_client=SimpleNamespace(is_proxy=lambda: False),
        login_client=login,
        transfer_method="sftp",
        verbose=False,
        remote_tmp_dir=Path("/remote/tmp/frequensolve"),
        _site_config_path=_host_config(tmp_path, local_tmp),
        run_login=lambda command: commands.append(command) or "",
        run_login_cmd=lambda command: commands.append(command)
        or (None, _stream(), _stream()),
    )

    SlurmTransferManager(site).get(Path("/work/project"), tmp_path / "project")

    assert downloads[0][0].startswith("/remote/tmp/frequensolve/")
    assert downloads[0][1].parent == local_tmp
    assert (tmp_path / "project" / "output.json").read_text() == "{}"
    assert any("tar czf /remote/tmp/frequensolve/" in command for command in commands)
    assert any(
        command.startswith("rm -f /remote/tmp/frequensolve/") for command in commands
    )
    assert not (tmp_path / "project.tar.gz").exists()


def test_unknown_transfer_method_fails_before_starting_subprocess():
    site = SimpleNamespace(
        _login_client=SimpleNamespace(is_proxy=lambda: True),
        transfer_method="scp",
        verbose=False,
    )

    with pytest.raises(ValueError, match="transfer_method"):
        SlurmTransferManager(site)._uses_sftp()
