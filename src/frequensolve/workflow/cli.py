"""Explicit restricted login and local MCP entrypoint."""

from __future__ import annotations

import json
import re
import webbrowser
from pathlib import Path

import click

from .auth import Connection, WorkflowAuth, WorkflowError
from .storage import PrivateStore


def profile_auth(profile: str, cache: Path | None = None) -> WorkflowAuth:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", profile):
        raise click.ClickException("Use a safe explicit workflow profile name")
    store = PrivateStore(cache or Path.home() / ".frequensolve" / "workflow")
    with store.locked(profile + "-connection") as record:
        value = record.read()
    if not value:
        raise WorkflowError("Run frequensolve-workflow login for this explicit profile")
    return WorkflowAuth(Connection(**value), store, profile + "-tokens")


@click.group()
def main() -> None:
    """Prepare, inspect and analyze scientific work with restricted Cloud access."""


@main.command()
@click.option("--profile", required=True)
@click.option("--issuer", required=True)
@click.option("--client-id", required=True)
def login(profile: str, issuer: str, client_id: str) -> None:
    """Approve a restricted connection in the browser. Never accepts passwords."""
    try:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", profile):
            raise ValueError()
        connection = Connection(
            issuer.rstrip("/"), issuer.rstrip("/") + "/cloud", client_id
        )
        store = PrivateStore(Path.home() / ".frequensolve" / "workflow")
        with store.locked(profile + "-connection") as record:
            record.write(connection.binding)
        WorkflowAuth(connection, store, profile + "-tokens").login(webbrowser.open)
        click.echo("Restricted workflow connection approved.")
    except (ValueError, OSError, WorkflowError) as error:
        raise click.ClickException(
            str(error)
            if isinstance(error, WorkflowError)
            else "Workflow login could not start safely"
        ) from None


@main.command()
@click.option("--profile", required=True)
def status(profile: str) -> None:
    """Check the restricted connection without inspecting any SDK credential cache."""
    try:
        auth = profile_auth(profile)
        credentials = auth.credentials()
        click.echo(
            json.dumps(
                {
                    "connected": True,
                    "issuer": auth.connection.issuer,
                    "scopes": sorted(auth.connection.scopes),
                    "paid_authority": False,
                    "expires_at": credentials["expires_at"],
                }
            )
        )
    except WorkflowError as error:
        raise click.ClickException(str(error)) from None


@main.command()
@click.option("--profile", required=True)
def logout(profile: str) -> None:
    """Revoke this adapter's tokens. Cloud Settings can disconnect all grants."""
    try:
        profile_auth(profile).logout()
        click.echo("Restricted workflow connection removed.")
    except WorkflowError as error:
        raise click.ClickException(str(error)) from None


@main.command()
@click.option("--profile", required=True)
@click.option(
    "--allow-root", "root_specs", multiple=True, required=True, metavar="ID=PATH"
)
@click.option(
    "--cloud-url",
    required=True,
    help="Exact Sandbox product origin for human review links.",
)
@click.option(
    "--interactive-review",
    is_flag=True,
    help="Enable no-spend native MCP review rehearsal; grants no paid authority.",
)
def serve(
    profile: str, root_specs: tuple[str, ...], cloud_url: str, interactive_review: bool
) -> None:
    """Serve the separate scientific workflow profile over stdio."""
    from frequensolve.mcp_server.cli import _parse_roots

    from .client import WorkflowClient
    from .server import build_server

    try:
        server = build_server(
            WorkflowClient(profile_auth(profile)),
            _parse_roots(root_specs),
            cloud_url,
            interactive_review=interactive_review,
        )
        server.run_stdio()
    except Exception:
        raise click.ClickException(
            "Workflow server stopped safely; check the explicit profile and allowed roots"
        ) from None


if __name__ == "__main__":
    main()
