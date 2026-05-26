import sys
from dataclasses import dataclass

import pytest

import frequensolve as fs
from frequensolve.orchestrator import sites


class FakeSite:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@dataclass
class FakeSlurmSiteConfig:
    hostname: str
    queue: str = "normal"
    mpi_wrapper: str = "srun"
    poll_interval: int = 5
    account: str = ""
    max_duration: str = "00-02:00:00"
    min_nodes: int = 1
    max_nodes: int = 1
    cores_per_node: int = 1
    memory_per_node: int = 0


@dataclass
class FakeSlurmRunConfig:
    queue: str | None = None
    nodes: int = 1
    duration: str | None = None
    procs_per_node: int | None = None
    procs_per_task: int | None = None
    account: str | None = None
    notify_on: str | None = None
    notify_email: str | None = None
    poll_interval: int | None = None
    run_path: str | None = None
    slurm_args: list[str] | None = None

    @classmethod
    def field_names(cls):
        return set(cls.__dataclass_fields__)


def install_fake_hpc_module(monkeypatch):
    module = type(sys)("frequensolve.orchestrator.sites.hpc")
    module.SlurmSiteConfig = FakeSlurmSiteConfig
    module.SlurmRunConfig = FakeSlurmRunConfig
    monkeypatch.setitem(sys.modules, "frequensolve.orchestrator.sites.hpc", module)


def test_site_factory_reads_default_config_from_env(monkeypatch, tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
[site]
type = "local"
shutdown_on_completion = true
verbose = true
dashboard_port = 8787
""".strip()
    )
    monkeypatch.setenv("FREQUENSOLVE_SITE_CONFIG", str(config_path))
    monkeypatch.setattr(sites, "LocalSite", FakeSite)

    site = sites.Site()

    assert isinstance(site, FakeSite)
    assert site.kwargs == {
        "shutdown_on_completion": True,
        "verbose": True,
        "dashboard_port": 8787,
    }


def test_site_factory_uses_stable_home_default_path(monkeypatch, tmp_path):
    monkeypatch.delenv("FREQUENSOLVE_SITE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert sites.site_config_path() == tmp_path / ".frequensolve" / "site.toml"


def test_site_factory_creates_starter_config_for_missing_default(monkeypatch, tmp_path):
    monkeypatch.delenv("FREQUENSOLVE_SITE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sites, "AWSSite", FakeSite)

    config_path = tmp_path / ".frequensolve" / "site.toml"

    with pytest.raises(FileNotFoundError, match="Created starter FrequenSolve"):
        sites.Site()

    starter = config_path.read_text()
    assert 'default = "cloud"' in starter
    assert "[sites.cloud]" in starter
    assert 'type = "aws"' in starter
    assert "[sites.local]" in starter
    assert "[sites.hpc]" in starter

    site = sites.Site()

    assert site.kwargs == {
        "domain": "app.frequensol.com",
        "interactive": True,
        "verbose": True,
    }


def test_site_factory_reads_named_profiles_and_routes_aws_lazily(monkeypatch, tmp_path):
    config_path = tmp_path / "sites.toml"
    config_path.write_text(
        """
default = "cloud"

[sites.local]
type = "local"
shutdown_on_completion = true

[sites.cloud]
type = "aws"
domain = "dev.frequensol.example"
interactive = true
verbose = true
""".strip()
    )
    monkeypatch.setattr(sites, "LocalSite", FakeSite)
    monkeypatch.setattr(sites, "AWSSite", FakeSite)

    default_site = sites.Site(config_path=config_path)
    local_site = sites.Site(config_path=config_path, profile="local", verbose=False)

    assert default_site.kwargs == {
        "domain": "dev.frequensol.example",
        "interactive": True,
        "verbose": True,
    }
    assert local_site.kwargs == {
        "shutdown_on_completion": True,
        "verbose": False,
    }


def test_site_factory_builds_slurm_config_objects(monkeypatch, tmp_path):
    install_fake_hpc_module(monkeypatch)
    config_path = tmp_path / "slurm-site.toml"
    config_path.write_text(
        """
[site]
type = "slurm"
rel_path = "projects/demo"
hostname = "login.example.edu"
queue = "debug"
account = "acct123"
nodes = 2
duration = "00:30:00"
procs_per_node = 4
verbose = true
""".strip()
    )
    monkeypatch.setattr(sites, "SlurmSite", FakeSite)

    site = sites.Site(config_path=config_path)

    assert site.kwargs["rel_path"] == "projects/demo"
    assert site.kwargs["default_queue"] == "debug"
    assert site.kwargs["verbose"] is True
    assert site.kwargs["config"] == FakeSlurmSiteConfig(
        hostname="login.example.edu",
        queue="debug",
        account="acct123",
    )
    assert site.kwargs["run_config"] == FakeSlurmRunConfig(
        queue="debug",
        nodes=2,
        duration="00:30:00",
        procs_per_node=4,
        account="acct123",
    )


def test_site_factory_normalizes_hpc_overrides(monkeypatch, tmp_path):
    install_fake_hpc_module(monkeypatch)
    config_path = tmp_path / "slurm-site.toml"
    config_path.write_text(
        """
[site]
type = "slurm"
rel_path = "projects/demo"
hostname = "login.example.edu"
queue = "debug"
account = "acct123"
nodes = 1
duration = "00:30:00"
""".strip()
    )
    monkeypatch.setattr(sites, "SlurmSite", FakeSite)

    site = sites.Site(
        config_path=config_path,
        nodes=3,
        duration="01:00:00",
        account="override-acct",
    )

    assert site.kwargs["config"] == FakeSlurmSiteConfig(
        hostname="login.example.edu",
        queue="debug",
        account="override-acct",
    )
    assert site.kwargs["run_config"] == FakeSlurmRunConfig(
        queue="debug",
        nodes=3,
        duration="01:00:00",
        account="override-acct",
    )


def test_site_factory_builds_stampede_run_config(monkeypatch, tmp_path):
    install_fake_hpc_module(monkeypatch)
    config_path = tmp_path / "stampede-site.toml"
    config_path.write_text(
        """
[site]
type = "stampede3"
rel_path = "projects/demo"
queue = "skx-dev"
nodes = 1
duration = "00:20:00"
transfer_method = "sftp"
""".strip()
    )
    monkeypatch.setattr(sites, "Stampede3Site", FakeSite)

    site = sites.Site(config_path=config_path)

    assert site.kwargs["rel_path"] == "projects/demo"
    assert site.kwargs["default_queue"] == "skx-dev"
    assert site.kwargs["transfer_method"] == "sftp"
    assert site.kwargs["run_config"] == FakeSlurmRunConfig(
        queue="skx-dev",
        nodes=1,
        duration="00:20:00",
    )
    assert "config" not in site.kwargs


def test_site_factory_is_exported_at_top_level():
    assert fs.Site is sites.Site
    assert "Site" in fs.__all__
    assert "Site" in dir(fs)


def test_site_factory_reports_missing_config_path(tmp_path):
    missing_path = tmp_path / "missing.toml"

    with pytest.raises(FileNotFoundError, match="FREQUENSOLVE_SITE_CONFIG"):
        sites.Site(config_path=missing_path)

    assert not missing_path.exists()


def test_site_factory_does_not_create_missing_env_config(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing-env.toml"
    monkeypatch.setenv("FREQUENSOLVE_SITE_CONFIG", str(missing_path))

    with pytest.raises(FileNotFoundError, match="No FrequenSolve site config"):
        sites.Site()

    assert not missing_path.exists()
