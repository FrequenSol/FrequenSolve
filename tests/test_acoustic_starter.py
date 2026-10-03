"""Executable checks for the first-run tutorial and non-executing config check."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from frequensolve.commands.cli import main
from frequensolve.mcp_server.core import create_simulation_draft, preview_simulation


@pytest.fixture
def starter():
    path = (
        Path(__file__).parents[1] / "tutorials/00_getting_started/acoustic_starter.py"
    )
    spec = importlib.util.spec_from_file_location("acoustic_starter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_starter_preparation_matches_mcp_and_preserves_existing_work(
    starter, tmp_path, monkeypatch
):
    def forbid_site(*args, **kwargs):
        pytest.fail("Preparation must not instantiate an execution site")

    monkeypatch.setattr(starter.fs, "Site", forbid_site)
    directory = tmp_path / "starter"
    project, job = starter.prepare(directory)
    expected = preview_simulation(create_simulation_draft())
    assert job.f_list == expected["frequencies_hz"]
    assert job.simulation.dimension == 2
    assert job.simulation.physics == "acoustic"
    assert (
        job.simulation.acquisition.receiver_groups[0].size == expected["receiver_count"]
    )
    assert project.path == directory
    assert list(directory.rglob("*.json"))
    saved = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="preserved"):
        starter.prepare(directory)
    assert saved == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_submit_uses_prepared_job_and_keeps_failure_diagnostics(
    starter, tmp_path, monkeypatch
):
    directory = tmp_path / "starter"
    starter.prepare(directory)
    calls = []
    result = SimpleNamespace(status="FAILED", successful=False)
    run = SimpleNamespace(id="fake-run", wait=lambda **kwargs: result)
    monkeypatch.setattr(
        starter.fs,
        "Site",
        lambda **kwargs: SimpleNamespace(
            submit=lambda job, **opts: (calls.append((kwargs, job.name, opts)) or run)
        ),
    )
    assert starter.submit(directory, "cloud") is result
    assert calls[0][0] == {"profile": "cloud", "interactive": True}
    assert calls[0][1] == "frequency_10hz"


def test_site_check_missing_configuration_has_no_side_effects(tmp_path):
    path = tmp_path / "missing.toml"
    result = CliRunner().invoke(
        main, ["site", "check", "--local", "--config", str(path)]
    )
    assert result.exit_code != 0
    assert "Copy the configuration" in result.output
    assert not path.exists()


@pytest.mark.parametrize(
    "domain", ["app.example.test", "localhost:5173", "http://127.0.0.1:5173"]
)
def test_site_check_validates_configuration_without_authentication(
    tmp_path, domain, monkeypatch
):
    import frequensolve.orchestrator.sites.config_file as configuration

    monkeypatch.setattr(
        configuration,
        "_resolve_site_class",
        lambda *args: pytest.fail("No site creation"),
    )
    path = tmp_path / "site.toml"
    path.write_text(
        f'default = "cloud"\n[sites.cloud]\ntype = "aws"\ndomain = "{domain}"\nexecution_site_id = "managed-slurm"\n'
    )
    original = path.read_bytes()
    result = CliRunner().invoke(
        main, ["site", "check", "--local", "--config", str(path), "--profile", "cloud"]
    )
    assert result.exit_code == 0, result.output
    assert "No login, network request or solver run" in result.output
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "extra",
    [
        'domain = "https://user:password@example.test"',
        'domain = "http://example.test"',
        'domain = "https://example.test/?token=secret"',
        'domain = "app.example.test"\nexecution_site_id = "Upper"',
    ],
)
def test_site_check_rejects_unsafe_or_invalid_selection(tmp_path, extra):
    path = tmp_path / "site.toml"
    path.write_text(f'default = "cloud"\n[sites.cloud]\ntype = "aws"\n{extra}\n')
    result = CliRunner().invoke(
        main, ["site", "check", "--local", "--config", str(path)]
    )
    assert result.exit_code != 0
    assert "password" not in result.output
    assert "secret" not in result.output
