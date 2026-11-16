import click
import pytest
from click.testing import CliRunner

from frequensolve import _platform
from frequensolve._optional import optional_dependency_error
from frequensolve.commands.cli import main


@pytest.mark.parametrize("host", ["linux", "darwin"])
def test_supported_hosts_keep_optional_dependency_diagnostic(host, monkeypatch):
    monkeypatch.setattr(_platform.sys, "platform", host)
    error = optional_dependency_error(
        "Example", extra="cloud", error=ImportError("missing")
    )
    assert "pip install frequensolve[cloud]" in str(error)
    assert "Windows" not in str(error)
    assert _platform.platform_support_guidance() == ""


def test_native_windows_optional_error_explains_supported_host(monkeypatch):
    monkeypatch.setattr(_platform.sys, "platform", "win32")
    error = optional_dependency_error(
        "Example", extra="cloud", error=ImportError("missing")
    )
    assert "Native Windows is unsupported" in str(error)
    assert "WSL2" in str(error)
    assert "local Python environment" in str(error)


@pytest.mark.parametrize("host", ["win32", "linux", "darwin"])
def test_cli_guidance_does_not_block_command_execution(host, monkeypatch):
    # Replace only the policy provider; changing sys.platform while Click imports
    # terminal helpers would test this host's unavailable Windows libraries.
    monkeypatch.setattr(
        "frequensolve._cli_support.platform_support_guidance",
        lambda: _platform.WINDOWS_GUIDANCE if host == "win32" else "",
    )
    ran = []

    @click.command()
    def probe():
        ran.append(True)

    monkeypatch.setitem(main.commands, "probe", probe)
    result = CliRunner().invoke(main, ["probe"])
    assert result.exit_code == 0, result.output
    assert ran == [True]
    assert ("Native Windows is unsupported" in result.output) == (host == "win32")


@pytest.mark.parametrize(
    ("args", "exit_code"),
    [(["--help"], 0), (["--version"], 0), ([], 2), (["unknown-command"], 2)],
)
def test_windows_guidance_precedes_eager_options_and_parse_errors(
    args, exit_code, monkeypatch
):
    monkeypatch.setattr(
        "frequensolve._cli_support.platform_support_guidance",
        lambda: _platform.WINDOWS_GUIDANCE,
    )
    result = CliRunner().invoke(main, args)
    assert result.exit_code == exit_code, result.output
    assert result.output.startswith(_platform.WINDOWS_GUIDANCE)
    assert result.output.count(_platform.WINDOWS_GUIDANCE) == 1
