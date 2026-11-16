"""Installed-client configuration remains scoped, parseable and non-destructive."""

import json
import sys

import pytest
from click.testing import CliRunner

from frequensolve.workflow.cli import main
from frequensolve.workflow.codex import client_config
from frequensolve.workflow.server import OBSERVATION_TOOLS, TOOLS

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def test_config_round_trip_for_unicode_and_shell_metacharacters(tmp_path):
    root = tmp_path.resolve() / 'study space 🔬 "`$(touch never)`'
    root.mkdir()
    config = tomllib.loads(
        client_config("sandbox", {"study": root}, "https://example.test/")
    )
    server = config["mcp_servers"]["frequensolve-workflow"]
    assert set(config) == {"mcp_servers"}
    assert server["command"] == sys.executable
    assert server["args"] == [
        "-m",
        "frequensolve.workflow.cli",
        "serve",
        "--profile",
        "sandbox",
        "--allow-root",
        "study=" + str(root),
        "--cloud-url",
        "https://example.test",
    ]
    assert server["enabled_tools"] == list(TOOLS)
    assert set(server["tools"]) == set(OBSERVATION_TOOLS)
    assert all(
        settings == {"approval_mode": "approve"}
        for settings in server["tools"].values()
    )
    assert "workflow_approve" not in json.dumps(config)


def test_paid_and_rehearsal_tools_require_explicit_configuration_options(tmp_path):
    config = tomllib.loads(
        client_config(
            "native",
            {"study": tmp_path.resolve()},
            "http://localhost:5173",
            native_approval=True,
            interactive_review=True,
        )
    )
    server = config["mcp_servers"]["frequensolve-workflow"]
    assert "workflow_approve" in server["enabled_tools"]
    assert "--native-approval" in server["args"]
    assert "workflow_review" in server["enabled_tools"]
    assert "--interactive-review" in server["args"]
    assert server["tool_timeout_sec"] >= 240


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test",
        "https://user:password@example.test",
        "https://example.test/?secret=x",
        "https://example.test/path",
        "https://example.test/#secret",
    ],
)
def test_config_rejects_non_origin_and_credential_urls(tmp_path, url):
    with pytest.raises(ValueError):
        client_config("sandbox", {"study": tmp_path.resolve()}, url)


def test_cli_creates_only_new_file_and_preserves_existing_configuration(tmp_path):
    root = tmp_path.resolve()
    output = root / "client.toml"
    args = [
        "codex-config",
        "--profile",
        "sandbox",
        "--allow-root",
        "study=" + str(root),
        "--cloud-url",
        "https://example.test",
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    original = output.read_bytes()
    assert (output.stat().st_mode & 0o777) == 0o600
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0 and "preserved" in result.output
    assert output.read_bytes() == original
    assert not (root / ".codex").exists()


def test_config_rejects_symlink_workspace(tmp_path):
    root = tmp_path.resolve()
    (root / "linked").symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        client_config("sandbox", {"study": root / "linked"}, "https://example.test")
