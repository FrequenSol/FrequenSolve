import os
import pty
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

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


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


def test_ssh_proxy_interactive_shell_preserves_binary_streams(monkeypatch):
    popen_calls = []
    master_file = SimpleNamespace(flush=lambda: None)
    process = SimpleNamespace(stdin=None, stdout=None)

    def fake_popen(argv, **kwargs):
        popen_calls.append((argv, kwargs))
        return process

    monkeypatch.setattr(pty, "openpty", lambda: (10, 11))
    monkeypatch.setattr(os, "close", lambda fd: None)
    monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: master_file)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    proxy = SSHProxy("/tmp/control.sock", "user", "login.example.edu")

    assert proxy.invoke_shell() is process
    _, kwargs = popen_calls[0]
    assert "text" not in kwargs
    assert "universal_newlines" not in kwargs
    assert process.stdin is master_file
    assert process.stdout is master_file


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


def test_ssh_proxy_honors_per_command_timeout(monkeypatch):
    calls = []

    def time_out(argv, **kwargs):
        calls.append((argv, kwargs))
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", time_out)
    proxy = SSHProxy(
        "/tmp/control.sock",
        "user",
        "login.example.edu",
        command_timeout=120,
    )

    with pytest.raises(TimeoutError, match="login.example.edu.*15.0 seconds"):
        proxy.exec_command("FS_seismic --identity-json", timeout=15.0)

    assert calls[0][1]["timeout"] == 15.0


def test_ssh_proxy_reports_expired_control_socket(monkeypatch):
    private_detail = "Control socket connect: private token"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: _result(
            stderr=f"{private_detail}\n".encode(),
            returncode=255,
        ),
    )
    proxy = SSHProxy("/tmp/control.sock", "user", "login.example.edu")

    with pytest.raises(RuntimeError, match="status 255") as exc_info:
        proxy._exec_on_login("squeue")

    assert private_detail not in str(exc_info.value)


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
    published = []

    class FakeSFTP:
        def put(self, source, target):
            transferred.append((source, target))

        def stat(self, target):
            if target == "/work/input.json":
                raise FileNotFoundError(target)
            return SimpleNamespace(st_size=Path(transferred[-1][0]).stat().st_size)

        def chmod(self, target, mode):
            pass

        def posix_rename(self, source, target):
            published.append((source, target))

        def remove(self, target):
            pass

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

    assert transferred[0][0] == str(local_file)
    assert transferred[0][1].startswith("/work/.input.json.frequensolve-")
    assert published == [(transferred[0][1], "/work/input.json")]


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

        def stat(self, target):
            if not target.split("/")[-1].startswith("."):
                raise FileNotFoundError(target)
            return SimpleNamespace(st_size=transfers[-1][0].stat().st_size)

        def chmod(self, target, mode):
            pass

        def posix_rename(self, source, target):
            pass

        def remove(self, target):
            pass

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
        run_login_cmd=lambda command: (
            commands.append(command) or (None, _stream(), _stream())
        ),
    )

    SlurmTransferManager(site).put(local_dir, Path("/work/project"))

    assert transfers[0][0].parent == local_tmp
    assert transfers[0][1].startswith("/remote/tmp/frequensolve/")
    assert commands[0] == "mkdir -p -- /work"
    assert "mkdir -p /remote/tmp/frequensolve" in commands
    publish_command = next(command for command in commands if "tar xzf" in command)
    assert 'tar xzf "$archive" -C "$staging"' in publish_command
    assert 'mv -- "$destination" "$backup"' in publish_command
    assert 'mv -- "$staging/$entry" "$destination"' in publish_command
    assert 'mv -- "$backup" "$destination" || true' in publish_command
    assert 'rm -rf -- "$backup"' in publish_command
    assert "trap cleanup EXIT HUP INT TERM" in publish_command
    assert not (tmp_path / "project.tar.gz").exists()


def test_sftp_directory_get_uses_configured_tmp_dirs(tmp_path):
    local_tmp = tmp_path / "local-tmp"
    payload = tmp_path / "remote-payload"
    payload.mkdir()
    (payload / "output.json").write_text("{}")
    downloads = []
    commands = []
    archive_size = None

    class FakeSFTP:
        def stat(self, path):
            if path == "/work/project":
                return SimpleNamespace(st_mode=0o040755)
            return SimpleNamespace(st_mode=0o100644, st_size=archive_size)

        def get(self, remote, local):
            nonlocal archive_size
            downloads.append((remote, Path(local)))
            with tarfile.open(local, "w:gz") as tar:
                tar.add(payload / "output.json", arcname="output.json")
            archive_size = Path(local).stat().st_size

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
        run_login_cmd=lambda command: (
            commands.append(command) or (None, _stream(), _stream())
        ),
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
