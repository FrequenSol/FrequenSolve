"""Command line entry points for the optional local MCP server."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import click

if TYPE_CHECKING:
    from frequensolve.mcp_server._sdk_v1 import SafeFastMCP

_ROOT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


@click.group()
@click.version_option(package_name="frequensolve")
def main() -> None:
    """Serve or verify the FrequenSolve simulation-assistant MCP."""


@main.command("serve")
@click.option(
    "--allow-root",
    "root_specs",
    multiple=True,
    metavar="ID=PATH",
    help="Allow one existing directory under a non-sensitive root ID.",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=click.FloatRange(min=1.0, max=60.0),
    default=15.0,
    show_default=True,
)
@click.option(
    "--max-concurrency",
    type=click.IntRange(min=1, max=4),
    default=2,
    show_default=True,
)
@click.option(
    "--cloud-profile",
    type=str,
    metavar="NAME",
    help="Use one existing aws/cloud profile from site.toml for read-only tools.",
)
def serve(
    root_specs: tuple[str, ...],
    timeout_seconds: float,
    max_concurrency: int,
    cloud_profile: str | None,
) -> None:
    """Run the local server over STDIO."""

    roots = _parse_roots(root_specs)
    server = _build_server(
        roots,
        cloud_profile=cloud_profile,
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
    )
    try:
        server.run("stdio")
    except Exception:
        raise click.ClickException("The MCP server stopped unexpectedly.") from None


@main.command("doctor")
@click.option(
    "--allow-root",
    "root_specs",
    multiple=True,
    metavar="ID=PATH",
    help="Verify startup with one existing allowed directory.",
)
@click.option(
    "--cloud-profile",
    type=str,
    metavar="NAME",
    help="Verify the surface with one configured aws/cloud site profile.",
)
def doctor(root_specs: tuple[str, ...], cloud_profile: str | None) -> None:
    """Run an in-memory MCP initialization and surface check."""

    roots = _parse_roots(root_specs)
    try:
        from frequensolve.mcp_server._sdk_v1 import run_in_memory_doctor

        server = _build_server(roots, cloud_profile=cloud_profile)
        result = run_in_memory_doctor(server)
    except click.ClickException:
        raise
    except ModuleNotFoundError as exc:
        if _is_optional_mcp_import(exc.name):
            raise click.ClickException(
                "Install MCP support with: pip install 'frequensolve[mcp]'"
            ) from None
        raise click.ClickException("The MCP startup check failed safely.") from None
    except Exception:
        raise click.ClickException("The MCP startup check failed safely.") from None
    result["allowed_root_ids"] = sorted(roots)
    result["cloud_profile_selection"] = (
        "explicit" if cloud_profile is not None else "default"
    )
    click.echo(json.dumps(result, sort_keys=True, separators=(",", ":")))


def _build_server(
    roots: dict[str, Path],
    *,
    cloud_profile: str | None = None,
    timeout_seconds: float = 15.0,
    max_concurrency: int = 2,
) -> SafeFastMCP:
    try:
        from frequensolve.mcp_server.server import build_server
    except ModuleNotFoundError as exc:
        if _is_optional_mcp_import(exc.name):
            raise click.ClickException(
                "Install MCP support with: pip install 'frequensolve[mcp]'"
            ) from None
        raise
    try:
        return build_server(
            allowed_roots=roots,
            cloud_profile=cloud_profile,
            operation_timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
        )
    except (ValueError, RuntimeError):
        raise click.ClickException("The MCP server configuration is invalid.") from None


def _parse_roots(specifications: Iterable[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for specification in specifications:
        if specification.count("=") != 1:
            raise click.ClickException("Each allowed root must use the form ID=PATH.")
        root_id, raw_path = specification.split("=", 1)
        if not _ROOT_ID_RE.fullmatch(root_id) or root_id in roots:
            raise click.ClickException(
                "Allowed root IDs must be unique lower-case safe names."
            )
        if not raw_path or "\x00" in raw_path or "\r" in raw_path or "\n" in raw_path:
            raise click.ClickException("An allowed root path is invalid.")
        try:
            path = Path(raw_path).expanduser()
            if not path.is_absolute() or not path.is_dir() or path.is_symlink():
                raise click.ClickException(
                    "Each allowed root must be an existing absolute directory."
                )
            resolved = path.resolve(strict=True)
            if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
                raise click.ClickException(
                    "Choose a narrower non-sensitive allowed root directory."
                )
        except click.ClickException:
            raise
        except (OSError, RuntimeError):
            raise click.ClickException(
                "Each allowed root must be an existing absolute directory."
            ) from None
        roots[root_id] = resolved
    return roots


def _is_optional_mcp_import(name: str | None) -> bool:
    return bool(
        name
        and any(
            name == package or name.startswith(package + ".")
            for package in ("anyio", "mcp", "pydantic")
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
