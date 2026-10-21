"""Generate a standalone client configuration without changing shared settings."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .storage import directory


def client_config(
    profile: str,
    roots: dict[str, Path],
    cloud_url: str,
    *,
    interactive_review: bool = False,
    native_approval: bool = False,
) -> str:
    """Only the configured adapter is declared; never embed tokens or identities."""
    from .server import APPROVAL_TOOL, REVIEW_TOOL, TOOLS

    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", profile) or not roots:
        raise ValueError("Choose an explicit profile and workspace")
    origin = urlsplit(cloud_url)
    if (
        origin.scheme not in ("https", "http")
        or not origin.hostname
        or origin.username
        or origin.password
        or origin.path not in ("", "/")
        or origin.query
        or origin.fragment
        or (
            origin.scheme == "http"
            and origin.hostname not in ("localhost", "127.0.0.1")
        )
    ):
        raise ValueError("Choose an exact HTTPS or local Sandbox product origin")
    args = ["-m", "frequensolve.workflow.cli", "serve", "--profile", profile]
    for root_id, path in sorted(roots.items()):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", root_id):
            raise ValueError("Use a safe workspace ID")
        with directory(path):
            pass
        args.extend(["--allow-root", root_id + "=" + str(path.absolute())])
    args.extend(["--cloud-url", cloud_url.rstrip("/")])
    tools = list(TOOLS)
    if interactive_review:
        args.append("--interactive-review")
        tools.append(REVIEW_TOOL)
    if native_approval:
        args.append("--native-approval")
        tools.append(APPROVAL_TOOL)
    # JSON basic strings and arrays are valid TOML; these are data, never shell code.
    return (
        "# FrequenSolve scientific workflow. Add this server to your Codex MCP settings.\n"
        "# Existing settings and credentials have not been changed.\n"
        "# Native paid approval requires separate connection consent.\n"
        "[mcp_servers.frequensolve-workflow]\n"
        f"command = {json.dumps(sys.executable, ensure_ascii=False)}\n"
        f"args = {json.dumps(args, ensure_ascii=False)}\n"
        f"enabled_tools = {json.dumps(tools)}\n"
        "required = true\nstartup_timeout_sec = 30\ntool_timeout_sec = 300\n"
        '\n[mcp_servers.frequensolve-workflow.tools.workflow_workspace]\napproval_mode = "approve"\n'
        '\n[mcp_servers.frequensolve-workflow.tools.workflow_present]\napproval_mode = "approve"\n'
        '\n[mcp_servers.frequensolve-workflow.tools.workflow_discover]\napproval_mode = "approve"\n'
    )
