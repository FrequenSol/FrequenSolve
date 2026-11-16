"""CLI support diagnostics without importing or starting an MCP server."""

import builtins
import json
import sys
from types import ModuleType

import pytest
from click.testing import CliRunner

from frequensolve import _cli_support, _platform
from frequensolve.mcp_server import cli


@pytest.fixture(params=["win32", "linux", "darwin"])
def guidance(request, monkeypatch):
    text = _platform.WINDOWS_GUIDANCE if request.param == "win32" else ""
    monkeypatch.setattr(_cli_support, "platform_support_guidance", lambda: text)
    return text


@pytest.mark.parametrize(
    ("args", "status"),
    [
        (["--help"], 0),
        (["--version"], 0),
        ([], 2),
        (["unknown-command"], 2),
        (["doctor", "--help"], 0),
        (["serve", "--help"], 0),
        (["serve", "--timeout", "invalid"], 2),
    ],
)
def test_mcp_eager_options_and_parse_errors(args, status, guidance):
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == status, result.output
    assert _platform.WINDOWS_GUIDANCE not in result.stdout
    assert result.stderr.count(_platform.WINDOWS_GUIDANCE) == bool(guidance)


@pytest.mark.parametrize("command", ["doctor", "serve"])
def test_mcp_commands_preserve_protocol_output(command, guidance, monkeypatch):
    events = []

    class Server:
        def run_stdio(self):
            events.append("serve")

    def doctor(server):
        assert isinstance(server, Server)
        events.append("doctor")
        return {"ok": True}

    sdk = ModuleType("frequensolve.mcp_server._sdk_v2")
    sdk.run_in_memory_doctor = doctor
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    monkeypatch.setattr(cli, "_build_server", lambda *a, **k: Server())
    result = CliRunner().invoke(cli.main, [command])
    assert result.exit_code == 0, result.output
    assert events == [command]
    assert result.stderr == (guidance + "\n" if guidance else "")
    if command == "doctor":
        assert json.loads(result.stdout)["ok"] is True
    else:
        assert result.stdout == ""


@pytest.mark.parametrize("command", ["doctor", "serve"])
def test_missing_extra_stays_sanitized_and_on_stderr(command, guidance, monkeypatch):
    original_import = builtins.__import__

    def missing_extra(name, *args, **kwargs):
        if name in {
            "frequensolve.mcp_server._sdk_v2",
            "frequensolve.mcp_server.server",
        }:
            raise ModuleNotFoundError("private dependency path", name="mcp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_extra)
    result = CliRunner().invoke(cli.main, [command])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        (guidance + "\n" if guidance else "")
        + "Error: Install MCP support with: pip install 'frequensolve[mcp]'\n"
    )


def test_resilient_parsing_does_not_emit_diagnostics(monkeypatch, capsys):
    monkeypatch.setattr(
        _cli_support, "platform_support_guidance", lambda: _platform.WINDOWS_GUIDANCE
    )
    with cli.main.make_context("frequensolve-mcp", [], resilient_parsing=True):
        pass
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
