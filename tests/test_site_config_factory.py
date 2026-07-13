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
    scheduler: str = "SLURM"
    mpi_wrapper: str = "srun"
    poll_interval: int = 5
    account: str = ""
    max_duration: str = "00-02:00:00"
    min_nodes: int = 1
    max_nodes: int = 1
    cores_per_node: int = 1
    sockets_per_node: int = 1
    memory_per_node: int = 0
    gpus_per_node: int = 0
    partitions: dict | None = None


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
default = "local"

[sites.local]
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


def test_site_factory_local_runtime_overrides_preserve_profile_solver(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
default = "local"

[sites.local]
type = "local"
solver = "/configured/bin/fs3d_s"
shutdown_on_completion = false
n_workers = 4
threads_per_worker = 2
""".strip()
    )
    monkeypatch.setattr(sites, "LocalSite", FakeSite)

    site = sites.Site(
        config_path=config_path,
        profile="local",
        n_workers=1,
        threads_per_worker=16,
        shutdown_on_completion=True,
    )

    assert site.kwargs == {
        "solver": "/configured/bin/fs3d_s",
        "shutdown_on_completion": True,
        "n_workers": 1,
        "threads_per_worker": 16,
    }
    assert site._site_config_path == config_path
    assert site._site_profile == "local"


def test_site_factory_uses_stable_home_default_path(monkeypatch, tmp_path):
    monkeypatch.delenv("FREQUENSOLVE_SITE_CONFIG", raising=False)
    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert sites.site_config_path() == tmp_path / ".frequensolve" / "site.toml"


def test_site_factory_uses_frequensolve_home_override(monkeypatch, tmp_path):
    storage_root = tmp_path / "fs-user-storage"
    monkeypatch.delenv("FREQUENSOLVE_SITE_CONFIG", raising=False)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(storage_root))

    assert sites.site_config_path() == storage_root / "site.toml"


def test_site_factory_creates_starter_config_for_missing_default(monkeypatch, tmp_path):
    monkeypatch.delenv("FREQUENSOLVE_SITE_CONFIG", raising=False)
    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
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
    assert 'type = "slurm"' in starter
    assert 'solver = "/path/to/local/solver"' in starter
    assert "modules = []" in starter
    assert "[sites.hpc.environment]" in starter
    assert "[sites.hpc.run_config]" in starter

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

    selected_site = sites.Site(config_path=config_path)
    local_site = sites.Site(config_path=config_path, profile="local", verbose=False)

    assert selected_site.kwargs == {
        "domain": "dev.frequensol.example",
        "interactive": True,
        "verbose": True,
    }
    assert local_site.kwargs == {
        "shutdown_on_completion": True,
        "verbose": False,
    }


def test_site_factory_rejects_single_site_table(tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
[site]
type = "local"
shutdown_on_completion = true
""".strip()
    )

    with pytest.raises(ValueError, match=r"default.*\[sites\.<profile>\]"):
        sites.Site(config_path=config_path)


def test_site_factory_rejects_bare_site_table(tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
type = "local"
shutdown_on_completion = true
""".strip()
    )

    with pytest.raises(ValueError, match=r"default.*\[sites\.<profile>\]"):
        sites.Site(config_path=config_path)


def test_site_factory_requires_default_even_with_explicit_profile(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "sites.toml"
    config_path.write_text(
        """
[sites.local]
type = "local"
shutdown_on_completion = true
""".strip()
    )
    monkeypatch.setattr(sites, "LocalSite", FakeSite)

    with pytest.raises(ValueError, match="must set default"):
        sites.Site(config_path=config_path, profile="local")


def test_site_factory_rejects_default_site_alias(tmp_path):
    config_path = tmp_path / "sites.toml"
    config_path.write_text(
        """
default_site = "local"

[sites.local]
type = "local"
shutdown_on_completion = true
""".strip()
    )

    with pytest.raises(ValueError, match="default_site.*default"):
        sites.Site(config_path=config_path)


@pytest.mark.parametrize("key", ["backend", "class"])
def test_site_factory_rejects_site_type_key_aliases(key, tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        f"""
default = "local"

[sites.local]
{key} = "local"
shutdown_on_completion = true
""".strip()
    )

    with pytest.raises(ValueError, match=rf"{key}.*type"):
        sites.Site(config_path=config_path)


@pytest.mark.parametrize("key", ["name", "profile"])
def test_site_factory_rejects_site_metadata_keys(key, tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        f"""
default = "local"

[sites.local]
type = "local"
{key} = "local"
shutdown_on_completion = true
""".strip()
    )

    with pytest.raises(ValueError, match=rf"{key}.*\[sites\.local\]"):
        sites.Site(config_path=config_path)


def test_site_factory_rejects_solver_executable_alias(tmp_path):
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        """
default = "local"

[sites.local]
type = "local"
solver_executable = "/path/to/solver"
""".strip()
    )

    with pytest.raises(ValueError, match="solver_executable.*solver"):
        sites.Site(config_path=config_path)


def test_site_factory_builds_slurm_config_objects(monkeypatch, tmp_path):
    install_fake_hpc_module(monkeypatch)
    config_path = tmp_path / "slurm-site.toml"
    config_path.write_text(
        """
default = "cluster"

[sites.cluster]
type = "slurm"
rel_path = "projects/demo"
hostname = "login.example.edu"
username = "user"
credential = "primary-cluster"
ssh_key = "~/.ssh/id_ed25519"
solver = "/remote/bin/solver"
work_dir = "/scratch/user"
modules = ["gcc", "openmpi"]
default_partition = "debug"
account = "acct123"
nodes = 2
duration = "00:30:00"
procs_per_node = 4
verbose = true

[sites.cluster.environment]
OMP_NUM_THREADS = "2"
""".strip()
    )
    monkeypatch.setattr(sites, "SlurmSite", FakeSite)

    site = sites.Site(config_path=config_path)

    assert site.kwargs["rel_path"] == "projects/demo"
    assert site.kwargs["default_partition"] == "debug"
    assert site.kwargs["verbose"] is True
    assert site.kwargs["username"] == "user"
    assert site.kwargs["credential"] == "primary-cluster"
    assert site.kwargs["ssh_key"] == "~/.ssh/id_ed25519"
    assert site.kwargs["solver"] == "/remote/bin/solver"
    assert site.kwargs["work_dir"] == "/scratch/user"
    assert site.kwargs["modules"] == ["gcc", "openmpi"]
    assert site.kwargs["environment"] == {"OMP_NUM_THREADS": "2"}
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
default = "cluster"

[sites.cluster]
type = "slurm"
rel_path = "projects/demo"
hostname = "login.example.edu"
default_partition = "debug"
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
default = "stampede3"

[sites.stampede3]
type = "stampede3"
rel_path = "projects/demo"
queue = "icx"
nodes = 1
duration = "00:20:00"
transfer_method = "sftp"
username = "user"
credential = "tacc-primary"
solver = "/remote/bin/solver"
work_dir = "/scratch/user"
""".strip()
    )
    monkeypatch.setattr(sites, "SlurmSite", FakeSite)

    site = sites.Site(config_path=config_path)

    assert site.kwargs["rel_path"] == "projects/demo"
    assert site.kwargs["default_partition"] == "icx"
    assert site.kwargs["transfer_method"] == "sftp"
    assert site.kwargs["username"] == "user"
    assert site.kwargs["credential"] == "tacc-primary"
    assert site.kwargs["solver"] == "/remote/bin/solver"
    assert site.kwargs["work_dir"] == "/scratch/user"
    assert site.kwargs["run_config"] == FakeSlurmRunConfig(
        queue="icx",
        nodes=1,
        duration="00:20:00",
        poll_interval=5,
    )
    assert site.kwargs["config"].hostname == "stampede3.tacc.utexas.edu"
    assert site.kwargs["config"].mpi_wrapper == "ibrun"
    assert site.kwargs["config"].queue == "icx"
    assert site.kwargs["config"].partitions["spr"]["cores_per_node"] == 112


def test_site_factory_overlays_stampede_preset_and_partition(monkeypatch, tmp_path):
    install_fake_hpc_module(monkeypatch)
    config_path = tmp_path / "preset-site.toml"
    config_path.write_text(
        """
default = "tacc"

[sites.tacc]
type = "slurm"
preset = "stampede3"
rel_path = "projects/demo"
default_partition = "icx"
username = "user"

[sites.tacc.partitions.icx]
max_nodes = 8
""".strip()
    )
    monkeypatch.setattr(sites, "SlurmSite", FakeSite)

    site = sites.Site(config_path=config_path)

    assert site.kwargs["default_partition"] == "icx"
    assert site.kwargs["config"].hostname == "stampede3.tacc.utexas.edu"
    assert site.kwargs["config"].queue == "icx"
    assert site.kwargs["config"].partitions["icx"] == {
        "max_duration": "2-00:00:00",
        "min_nodes": 1,
        "max_nodes": 8,
        "cores_per_node": 80,
        "sockets_per_node": 2,
        "memory_per_node": 262144,
        "gpus_per_node": 0,
    }
    assert site.kwargs["run_config"].queue == "icx"


def test_site_factory_rejects_unknown_or_incompatible_presets(tmp_path):
    unknown = tmp_path / "unknown-preset.toml"
    unknown.write_text(
        """
default = "cluster"

[sites.cluster]
type = "slurm"
preset = "missing"
""".strip()
    )
    with pytest.raises(ValueError, match="Unknown FrequenSolve site preset.*missing"):
        sites.Site(config_path=unknown)

    incompatible = tmp_path / "incompatible-preset.toml"
    incompatible.write_text(
        """
default = "cloud"

[sites.cloud]
type = "aws"
preset = "stampede3"
""".strip()
    )
    with pytest.raises(ValueError, match="type 'aws'.*incompatible.*stampede3"):
        sites.Site(config_path=incompatible)


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
