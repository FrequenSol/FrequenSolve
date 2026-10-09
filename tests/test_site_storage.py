"""Remote work-filesystem selection: $WORK/$SCRATCH expansion and ``storage``."""

import logging
from pathlib import Path

import pytest

from frequensolve.orchestrator.sites.config_file import Site
from frequensolve.orchestrator.sites.hpc import SlurmSiteConfig
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.curvature import SlurmCurvatureRunner
from frequensolve.orchestrator.sites.hpc.session import AllocationSession
from tests.test_slurm_site_refactor import (
    DummySlurmSite,
    DummySSHClientClass,
    DummyStream,
)

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

REMOTE_ENV = {
    "WORK": "/work2/01234/user/stampede3",
    "SCRATCH": "/scratch/01234/user",
    "PROJECT_SPACE": "/corral/project",
}


class EnvClient:
    """Login client that echoes remote variables and records each command."""

    def __init__(self):
        self.commands = []

    def exec_command(self, command, environment=None):
        self.commands.append(command)
        if command.startswith("echo $"):
            value = REMOTE_ENV.get(command[len("echo $") :], "")
            return None, DummyStream(value + "\n"), DummyStream("")
        return None, DummyStream(""), DummyStream("")

    def close(self):
        pass


class EnvSite(DummySlurmSite):
    def authenticate(self, host=None):
        self._env_client = EnvClient()
        return self._env_client


@pytest.fixture(autouse=True)
def dummy_ssh(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)


def echoes(site):
    return [c for c in site._env_client.commands if c.startswith("echo $")]


def test_default_storage_stays_on_work():
    site = EnvSite()
    assert site.storage == "work"
    assert site.work_dir == Path("/work2/01234/user/stampede3/frequensolve")
    assert site.scratch_dir is None
    assert site.remote_tmp_dir == Path("/tmp")


def test_scratch_storage_moves_every_work_directory_consumer(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=hpc.__name__):
        site = EnvSite(storage="scratch")
    work = Path("/scratch/01234/user/frequensolve")
    assert site.work_dir == work
    assert echoes(site) == ["echo $SCRATCH"]
    assert any(
        "Resolved $SCRATCH" in r.getMessage()
        and "/scratch/01234/user" in r.getMessage()
        for r in caplog.records
    )
    # Persistent-session records, heartbeats and status files follow work_dir.
    assert AllocationSession(site).root.parent == work / ".fs_sessions"
    # Curvature mirrors use scratch_dir when set, else work_dir.
    runner = SlurmCurvatureRunner(site)
    assert str(runner.remote_root(tmp_path)).startswith(str(work / "curvature"))
    assert site._remote_run_path(None) == work


def test_leading_remote_variables_are_resolved_once_and_cached():
    config = SlurmSiteConfig(
        hostname="login.example.edu", tmp_dir="${SCRATCH}/frequensolve/tmp"
    )
    site = EnvSite(
        work_dir="$SCRATCH/frequensolve",
        scratch_dir="$SCRATCH/frequensolve/high-io",
        config=config,
    )
    assert site.work_dir == Path("/scratch/01234/user/frequensolve")
    assert site.scratch_dir == Path("/scratch/01234/user/frequensolve/high-io")
    assert site.remote_tmp_dir == Path("/scratch/01234/user/frequensolve/tmp")
    assert site.remote_tmp_dir == Path("/scratch/01234/user/frequensolve/tmp")
    assert echoes(site) == ["echo $SCRATCH"]


def test_generic_and_bare_remote_variables():
    site = EnvSite(work_dir="$PROJECT_SPACE", scratch_dir="$WORK/")
    assert site.work_dir == Path("/corral/project")
    assert site.scratch_dir == Path("/work2/01234/user/stampede3")


def test_unset_remote_variable_fails_with_guidance():
    with pytest.raises(RuntimeError, match=r"Failed to determine \$UNSET_VAR"):
        EnvSite(work_dir="$UNSET_VAR/frequensolve")


@pytest.mark.parametrize(
    "value",
    ["$SCRATCH-old/fs", "/scratch/$USER/fs", "$SCRATCH/$USER", "${SCRATCH/fs"],
)
def test_malformed_variables_are_rejected_before_login(monkeypatch, value):
    def no_login(self, host=None):
        pytest.fail("bad paths must be rejected before authenticating")

    monkeypatch.setattr(EnvSite, "authenticate", no_login)
    with pytest.raises(ValueError, match="remote variable"):
        EnvSite(work_dir=value)


def test_storage_is_validated_and_exclusive_with_work_dir(monkeypatch):
    with pytest.raises(ValueError, match="storage must be 'work' or 'scratch'"):
        EnvSite(storage="SCRATCH")
    with pytest.raises(ValueError, match="storage selects the default work_dir"):
        EnvSite(storage="scratch", work_dir="/work2/user/frequensolve")


def test_site_toml_storage_and_variables(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SlurmSite", EnvSite)
    monkeypatch.setattr(
        "frequensolve.orchestrator.sites.SlurmSite", EnvSite, raising=False
    )
    config = tmp_path / "site.toml"
    config.write_text(
        """
default = "stampede3"

[sites.stampede3]
type = "slurm"
preset = "stampede3"
username = "user"
solver = "/work/shared/FS_seismic"
storage = "scratch"
tmp_dir = "$SCRATCH/frequensolve/tmp"

[sites.explicit]
type = "slurm"
preset = "stampede3"
username = "user"
solver = "/work/shared/FS_seismic"
work_dir = "$SCRATCH/runs"
"""
    )
    site = Site(config_path=config)
    assert site.storage == "scratch"
    assert site.work_dir == Path("/scratch/01234/user/frequensolve")
    assert site.remote_tmp_dir == Path("/scratch/01234/user/frequensolve/tmp")
    explicit = Site(config_path=config, profile="explicit")
    assert explicit.work_dir == Path("/scratch/01234/user/runs")
