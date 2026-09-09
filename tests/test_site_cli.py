import subprocess
from pathlib import Path
from types import SimpleNamespace

import toml
from click.testing import CliRunner

from frequensolve.commands import site as site_commands
from frequensolve.commands.cli import main
from frequensolve.orchestrator.sites.config_file import (
    STARTER_SITE_CONFIG,
    _resolve_site_preset,
)


def _configure(runner, storage_root, *extra):
    return runner.invoke(
        main,
        [
            "site",
            "configure",
            "stampede3",
            "--account",
            "TG-TEST",
            "--solver",
            "/work/shared/FS_seismic",
            *extra,
        ],
        input="student\n",
        env={"FREQUENSOLVE_HOME": str(storage_root)},
    )


def test_configure_stampede3_writes_minimal_profile(tmp_path):
    runner = CliRunner()

    result = _configure(runner, tmp_path)

    assert result.exit_code == 0, result.output
    config_path = tmp_path / "site.toml"
    config = toml.loads(config_path.read_text())
    profile = config["sites"]["stampede3"]
    assert config["default"] == "stampede3"
    assert profile == {
        "type": "slurm",
        "preset": "stampede3",
        "username": "student",
        "solver": "/work/shared/FS_seismic",
        "verbose": True,
        "run_config": {
            "account": "TG-TEST",
            "nodes": 1,
            "duration": "00:30:00",
            "ranks_per_node": 2,
            "ranks_per_task": 1,
        },
    }
    assert "credential" not in profile
    assert "ssh_key" not in profile
    assert "modules" not in profile
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert "frequensolve site connect" in result.output

    resolved = _resolve_site_preset(profile)
    assert resolved["modules"] == [
        "intel/25.1",
        "impi/21.15",
        "petsc/3.23",
        "phdf5",
    ]


def test_configure_stampede3_replaces_only_unmodified_starter(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "site.toml"
    config_path.write_text(STARTER_SITE_CONFIG)

    result = _configure(runner, tmp_path)

    assert result.exit_code == 0, result.output
    assert "Replaced the unmodified starter config" in result.output

    config_path.write_text('default = "custom"\n')
    result = _configure(runner, tmp_path)

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert config_path.read_text() == 'default = "custom"\n'


def test_configure_stampede3_validates_remote_solver_path(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "site",
            "configure",
            "stampede3",
            "--username",
            "student",
            "--account",
            "TG-TEST",
            "--solver",
            "relative/FS_seismic",
        ],
        env={"FREQUENSOLVE_HOME": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "remote solver path must be absolute" in result.output
    assert not (tmp_path / "site.toml").exists()


def test_connect_creates_and_verifies_shared_ssh_connection(monkeypatch, tmp_path):
    storage_root = tmp_path / "config"
    runner = CliRunner()
    configured = _configure(runner, storage_root)
    assert configured.exit_code == 0, configured.output

    calls = []
    results = iter(
        [
            subprocess.CompletedProcess([], 255),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
        ]
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(results)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(site_commands.subprocess, "run", fake_run)

    result = runner.invoke(
        main,
        ["site", "connect"],
        env={"FREQUENSOLVE_HOME": str(storage_root)},
    )

    assert result.exit_code == 0, result.output
    assert "respond to any SSH password or MFA prompts" in result.output
    assert "Connected to stampede3.tacc.utexas.edu" in result.output
    assert "scripts and transfers will reuse this connection" in result.output
    start_command, start_kwargs = calls[1]
    assert start_command[0:2] == ["/usr/bin/ssh", "-MNf"]
    assert "ControlMaster=yes" in start_command
    assert "ControlPersist=8h" in start_command
    assert start_command[-1] == "student@stampede3.tacc.utexas.edu"
    assert "-i" not in start_command
    assert start_kwargs == {"check": False}


def test_connect_reuses_existing_shared_connection(monkeypatch, tmp_path):
    storage_root = tmp_path / "config"
    runner = CliRunner()
    configured = _configure(runner, storage_root, "--username", "student")
    assert configured.exit_code == 0, configured.output

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(site_commands.subprocess, "run", fake_run)

    result = runner.invoke(
        main,
        ["site", "connect"],
        env={"FREQUENSOLVE_HOME": str(storage_root)},
    )

    assert result.exit_code == 0, result.output
    assert "already available" in result.output
    assert len(calls) == 2
    assert calls[1][0][-1] == "true"
    assert "BatchMode=yes" in calls[1][0]
    assert calls[1][1]["timeout"] == site_commands.SSH_CONNECTION_PROBE_TIMEOUT_SECONDS


def test_connect_replaces_master_when_remote_probe_times_out(monkeypatch, tmp_path):
    storage_root = tmp_path / "config"
    runner = CliRunner()
    configured = _configure(runner, storage_root, "--username", "student")
    assert configured.exit_code == 0, configured.output

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    control_path = site_commands._control_socket_path(
        "student", "stampede3.tacc.utexas.edu"
    )
    control_path.parent.mkdir(parents=True)
    control_path.touch()
    calls = []
    probe_timed_out = False

    def fake_run(command, **kwargs):
        nonlocal probe_timed_out
        calls.append((command, kwargs))
        if command[-1] == "true" and not probe_timed_out:
            probe_timed_out = True
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(site_commands.subprocess, "run", fake_run)

    result = runner.invoke(
        main,
        ["site", "connect"],
        env={"FREQUENSOLVE_HOME": str(storage_root)},
    )

    assert result.exit_code == 0, result.output
    assert "already available" not in result.output
    assert "Connected to stampede3.tacc.utexas.edu" in result.output
    assert any("exit" in command for command, _ in calls)
    assert not control_path.exists()


def test_connect_removes_socket_when_new_connection_probe_times_out(
    monkeypatch, tmp_path
):
    storage_root = tmp_path / "config"
    runner = CliRunner()
    configured = _configure(runner, storage_root, "--username", "student")
    assert configured.exit_code == 0, configured.output

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    control_path = site_commands._control_socket_path(
        "student", "stampede3.tacc.utexas.edu"
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 255)
        if command[0:2] == ["/usr/bin/ssh", "-MNf"]:
            control_path.touch()
            return subprocess.CompletedProcess(command, 0)
        if command[-1] == "true":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(site_commands.subprocess, "run", fake_run)

    result = runner.invoke(
        main,
        ["site", "connect"],
        env={"FREQUENSOLVE_HOME": str(storage_root)},
    )

    assert result.exit_code != 0
    assert "shared connection could not be verified" in result.output
    assert any("exit" in command for command, _ in calls)
    assert not control_path.exists()


def test_connect_restricts_existing_control_directory(monkeypatch, tmp_path):
    storage_root = tmp_path / "config"
    runner = CliRunner()
    configured = _configure(runner, storage_root, "--username", "student")
    assert configured.exit_code == 0, configured.output

    control_dir = tmp_path / "home" / ".ssh" / "control"
    control_dir.mkdir(parents=True)
    control_dir.chmod(0o755)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(
        site_commands.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    result = runner.invoke(
        main,
        ["site", "connect"],
        env={"FREQUENSOLVE_HOME": str(storage_root)},
    )

    assert result.exit_code == 0, result.output
    assert control_dir.stat().st_mode & 0o777 == 0o700


def test_connect_supports_generic_slurm_profile(monkeypatch, tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
default = "research-cluster"

[sites.research-cluster]
type = "slurm"
hostname = "login.example.edu"
username = "researcher"
solver = "/shared/FS_seismic"
""".strip()
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(site_commands.subprocess, "run", fake_run)

    result = CliRunner().invoke(
        main,
        ["site", "connect", "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert "SSH connection to login.example.edu is already available" in result.output
    assert calls[0][0][-1] == "researcher@login.example.edu"


def test_connections_lists_active_stale_and_closed_profiles(monkeypatch, tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
default = "active"

[sites.active]
type = "slurm"
hostname = "active.example.edu"
username = "researcher"

[sites.stale]
type = "slurm"
hostname = "stale.example.edu"
username = "researcher"

[sites.closed]
type = "slurm"
hostname = "closed.example.edu"
username = "researcher"

[sites.local]
type = "local"
solver = "/usr/bin/true"
""".strip()
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    active_path = site_commands._control_socket_path("researcher", "active.example.edu")
    stale_path = site_commands._control_socket_path("researcher", "stale.example.edu")
    active_path.parent.mkdir(parents=True)
    active_path.touch()
    stale_path.touch()
    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(
        site_commands,
        "_connection_is_live",
        lambda ssh, connection: connection.hostname == "active.example.edu",
    )

    result = CliRunner().invoke(
        main,
        ["site", "connections", "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    rows = {line.split()[0]: line.split()[1] for line in result.output.splitlines()[1:]}
    assert rows == {"active": "active", "stale": "stale", "closed": "closed"}
    assert "local" not in result.output
    assert str(active_path) in result.output


def test_disconnect_closes_master_and_removes_socket(monkeypatch, tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
default = "research"

[sites.research]
type = "slurm"
hostname = "login.example.edu"
username = "researcher"
""".strip()
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    control_path = site_commands._control_socket_path("researcher", "login.example.edu")
    control_path.parent.mkdir(parents=True)
    control_path.touch()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(site_commands.subprocess, "run", fake_run)

    result = CliRunner().invoke(
        main,
        ["site", "disconnect", "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Closed SSH connection to login.example.edu" in result.output
    assert calls[0][0][-2:] == ["exit", "researcher@login.example.edu"]
    assert calls[0][1]["timeout"] > 0
    assert not control_path.exists()


def test_disconnect_all_closes_each_configured_connection(monkeypatch, tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
default = "first"

[sites.first]
type = "slurm"
hostname = "first.example.edu"
username = "researcher"

[sites.second]
type = "slurm"
hostname = "second.example.edu"
username = "researcher"
""".strip()
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    paths = [
        site_commands._control_socket_path("researcher", "first.example.edu"),
        site_commands._control_socket_path("researcher", "second.example.edu"),
    ]
    paths[0].parent.mkdir(parents=True)
    for path in paths:
        path.touch()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(site_commands.shutil, "which", lambda command: "/usr/bin/ssh")
    monkeypatch.setattr(site_commands.subprocess, "run", fake_run)

    result = CliRunner().invoke(
        main,
        ["site", "disconnect", "--all", "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert "first.example.edu" in result.output
    assert "second.example.edu" in result.output
    assert len(calls) == 2
    assert all(not path.exists() for path in paths)


def test_check_reports_resolved_site_defaults_and_closes(monkeypatch):
    class FakeConfig:
        hostname = "stampede3.tacc.utexas.edu"

    class FakeSite:
        config = FakeConfig()
        executable = "/work/shared/FS_seismic"
        work_dir = Path("/work/student/frequensolve")
        modules = ["intel/25.1", "impi/21.15", "petsc/3.23", "phdf5"]
        closed = False

        def run_login(self, command):
            assert "test -x /work/shared/FS_seismic" in command
            assert "module load intel/25.1" in command
            assert "module load impi/21.15" in command
            assert "module load petsc/3.23" in command
            assert "module load phdf5" in command
            return "module setup output\nfrequensolve-solver-ready"

        def check_frequensolver_compatibility(self):
            return SimpleNamespace(
                confirmed=True,
                status="compatible",
                message=("FrequenSolve 0.3.0 matches preferred Solver v0.1.0."),
            )

        def close(self):
            self.closed = True

    fake_site = FakeSite()
    monkeypatch.setattr(site_commands, "_create_site", lambda **kwargs: fake_site)

    result = CliRunner().invoke(main, ["site", "check"])

    assert result.exit_code == 0, result.output
    assert "Site profile is ready: stampede3.tacc.utexas.edu" in result.output
    assert "Remote work directory: /work/student/frequensolve" in result.output
    assert "intel/25.1, impi/21.15, petsc/3.23, phdf5" in result.output
    assert "matches preferred Solver" in result.output
    assert fake_site.closed is True


def test_check_fails_when_modules_or_remote_solver_are_unavailable(monkeypatch):
    class FakeSite:
        executable = "/work/shared/FS_seismic"
        modules = []
        closed = False

        def run_login(self, command):
            return ""

        def close(self):
            self.closed = True

    fake_site = FakeSite()
    monkeypatch.setattr(site_commands, "_create_site", lambda **kwargs: fake_site)

    result = CliRunner().invoke(main, ["site", "check"])

    assert result.exit_code != 0
    assert "modules could not be loaded or the solver is missing" in result.output
    assert fake_site.closed is True


def test_empty_connection_commands_do_not_require_openssh(monkeypatch, tmp_path):
    config = tmp_path / "site.toml"
    config.write_text('default = "local"\n[sites.local]\ntype = "local"\n')
    monkeypatch.setattr(site_commands.shutil, "which", lambda _: None)
    for command in (["connections"], ["disconnect", "--all"]):
        result = CliRunner().invoke(main, ["site", *command, "--config", str(config)])
        assert result.exit_code == 0, result.output
        assert "No " in result.output
