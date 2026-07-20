"""Commands for configuring and connecting to execution sites."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

import click
import toml

from frequensolve.orchestrator.sites.config_file import (
    STARTER_SITE_CONFIG,
    _normalize_site_type,
    _resolve_site_preset,
    _site_config_table,
    load_site_config,
    site_config_path,
)

STAMPEDE3_DEFAULT_DURATION = "00:30:00"
STAMPEDE3_DEFAULT_RANKS_PER_NODE = 2
SSH_CONTROL_PERSIST = "8h"


@click.group()
def site() -> None:
    """Configure and connect to FrequenSolve execution sites."""


@site.command("configure")
@click.argument("preset", type=click.Choice(["stampede3"], case_sensitive=False))
@click.option(
    "--username",
    envvar="TACC_USERNAME",
    prompt="TACC username",
    help="TACC login name (or set TACC_USERNAME).",
)
@click.option("--account", required=True, help="TACC allocation/project id.")
@click.option("--solver", required=True, help="Remote FS_seismic executable path.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Site config path (defaults to ~/.frequensolve/site.toml).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing site config instead of refusing to overwrite it.",
)
def configure(
    preset: str,
    username: str,
    account: str,
    solver: str,
    config_path: Optional[Path],
    force: bool,
) -> None:
    """Create a ready-to-use site profile from a built-in PRESET."""

    if preset.lower() != "stampede3":  # pragma: no cover - guarded by Click
        raise click.ClickException(f"Unsupported site preset: {preset}")
    target = site_config_path(config_path)
    replaced_starter = _write_stampede3_config(
        target,
        username=username,
        account=account,
        solver=solver,
        force=force,
    )
    action = "Replaced the unmodified starter config" if replaced_starter else "Wrote"
    click.echo(f"{action} {target}")
    click.echo("Next, authenticate once with: frequensolve site connect")


@site.command("connect")
@click.option(
    "--profile", help="Configured site profile (defaults to site.toml default)."
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Site config path (defaults to ~/.frequensolve/site.toml).",
)
def connect(profile: Optional[str], config_path: Optional[Path]) -> None:
    """Authenticate once and share the SSH connection with FrequenSolve."""

    resolved_path = site_config_path(config_path)
    try:
        settings = _ssh_settings(resolved_path, profile=profile)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    ssh = shutil.which("ssh")
    if ssh is None:
        raise click.ClickException("OpenSSH is required, but 'ssh' was not found.")

    username = str(settings["username"])
    hostname = str(settings["hostname"])
    target = f"{username}@{hostname}"
    control_path = _control_socket_path(username, hostname)
    _ensure_control_directory(control_path.parent)

    check_command = [ssh, "-q", "-S", str(control_path), "-O", "check", target]
    if _run_control_check(check_command):
        click.echo(f"SSH connection to {hostname} is already available.")
        return
    if control_path.exists():
        control_path.unlink()

    command = [
        ssh,
        "-MNf",
        "-o",
        "ControlMaster=yes",
        "-o",
        f"ControlPersist={SSH_CONTROL_PERSIST}",
        "-o",
        f"ControlPath={control_path}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    ssh_key = settings.get("ssh_key")
    if ssh_key:
        key_path = Path(str(ssh_key)).expanduser()
        if not key_path.is_file():
            raise click.ClickException(f"Configured SSH key does not exist: {key_path}")
        command.extend(["-i", str(key_path)])
    command.append(target)

    click.echo(
        f"Authenticating with {hostname}; respond to any SSH password or MFA prompts."
    )
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise click.ClickException(
            f"SSH authentication failed with exit code {result.returncode}."
        )
    if not _run_control_check(check_command):
        raise click.ClickException(
            "SSH authenticated but the shared connection could not be verified."
        )
    click.echo(
        f"Connected to {hostname}. FrequenSolve scripts and transfers will reuse "
        f"this connection for up to {SSH_CONTROL_PERSIST}."
    )


@site.command("check")
@click.option(
    "--profile", help="Configured site profile (defaults to site.toml default)."
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Site config path (defaults to ~/.frequensolve/site.toml).",
)
def check(profile: Optional[str], config_path: Optional[Path]) -> None:
    """Verify an SSH-backed site and its configured solver."""

    execution_site = None
    try:
        execution_site = _create_site(config_path=config_path, profile=profile)
        solver = str(execution_site.executable)
        marker = "frequensolve-solver-ready"
        setup = [
            f"module load {shlex.quote(str(module))}"
            for module in execution_site.modules
        ]
        output = execution_site.run_login(
            "set -e\n"
            + "\n".join(
                [
                    *setup,
                    f"test -x {shlex.quote(solver)}",
                    f"printf '%s\\n' {shlex.quote(marker)}",
                ]
            )
        )
        if marker not in output.splitlines():
            raise click.ClickException(
                "The configured modules could not be loaded or the solver is "
                f"missing or not executable: {solver}"
            )
        hostname = getattr(execution_site.config, "hostname", "remote site")
        click.echo(f"Site profile is ready: {hostname}")
        click.echo(f"Remote work directory: {execution_site.work_dir}")
        click.echo(f"Solver: {solver}")
        modules = ", ".join(execution_site.modules) or "none"
        click.echo(f"Solver modules: {modules}")
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Site check failed: {exc}") from exc
    finally:
        if execution_site is not None:
            execution_site.close()


def _write_stampede3_config(
    path: Path,
    *,
    username: str,
    account: str,
    solver: str,
    force: bool,
) -> bool:
    """Write a minimal Stampede3 profile and return whether it replaced a starter."""

    username = _single_line_value(username, name="username")
    account = _single_line_value(account, name="account")
    solver = _single_line_value(solver, name="solver")
    if not PurePosixPath(solver).is_absolute():
        raise click.ClickException("The remote solver path must be absolute.")

    replaced_starter = False
    if path.exists():
        existing = path.read_text()
        replaced_starter = existing == STARTER_SITE_CONFIG
        if not force and not replaced_starter:
            raise click.ClickException(
                f"Site config already exists at {path}. Use --force to replace it."
            )

    document = {
        "default": "stampede3",
        "sites": {
            "stampede3": {
                "type": "slurm",
                "preset": "stampede3",
                "username": username,
                "solver": solver,
                "verbose": True,
                "run_config": {
                    "account": account,
                    "nodes": 1,
                    "duration": STAMPEDE3_DEFAULT_DURATION,
                    "ranks_per_node": STAMPEDE3_DEFAULT_RANKS_PER_NODE,
                    "ranks_per_task": 1,
                },
            }
        },
    }
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        "# Generated by `frequensolve site configure stampede3`.\n"
        + toml.dumps(document)
    )
    path.chmod(0o600)
    return replaced_starter


def _single_line_value(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise click.ClickException(f"{name} must be a non-empty single-line value.")
    return normalized


def _ssh_settings(path: Path, *, profile: Optional[str]) -> Mapping[str, Any]:
    config = load_site_config(path)
    values = _resolve_site_preset(_site_config_table(config, profile=profile))
    site_type = values.get("type")
    if not isinstance(site_type, str) or _normalize_site_type(site_type) not in {
        "slurm",
        "slurmsite",
        "stampede3",
        "stampede3site",
        "tacc",
    }:
        raise ValueError("The selected site profile is not an SSH-backed SLURM site.")
    hostname = values.get("hostname")
    username = (
        values.get("username")
        or os.getenv("HPC_USERNAME")
        or os.getenv("TACC_USERNAME")
    )
    if not isinstance(hostname, str) or not hostname.strip():
        raise ValueError("The selected site profile has no login hostname.")
    if not isinstance(username, str) or not username.strip():
        raise ValueError("The selected site profile has no SSH username.")
    return {**values, "hostname": hostname.strip(), "username": username.strip()}


def _control_socket_path(username: str, hostname: str) -> Path:
    digest = hashlib.sha256(f"{username}@{hostname}".encode()).hexdigest()[:16]
    return Path("~/.ssh/control").expanduser() / f"frequensolve-{digest}"


def _ensure_control_directory(path: Path) -> None:
    """Create the control-socket directory and keep it private to the user."""

    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _run_control_check(command: list[str]) -> bool:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _create_site(*, config_path: Optional[Path], profile: Optional[str]):
    from frequensolve.orchestrator.sites.config_file import Site

    return Site(config_path=config_path, profile=profile)
