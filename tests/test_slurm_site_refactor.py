from pathlib import Path

import pytest

from frequensolve.orchestrator.pool import PoolStatus
from frequensolve.orchestrator.sites import hpc
from frequensolve.orchestrator.sites.hpc import (
    SlurmLoginCredentials,
    SlurmSite,
    SlurmSiteConfig,
    _normalize_slurm_state,
    _parse_sbatch_job_id,
)
from frequensolve.orchestrator.sites.stampede3 import (
    Stampede3Config,
    Stampede3Site,
    TACCLoginCredentials,
)


class DummyStream:
    def __init__(self, text=""):
        self._text = text

    def read(self):
        return self._text.encode()


class DummyRawClient:
    def exec_command(self, command, environment=None):
        if command == "echo $WORK":
            return None, DummyStream("/remote/work\n"), DummyStream("")
        if command == "echo $HOSTNAME":
            return None, DummyStream("login\n"), DummyStream("")
        return None, DummyStream(""), DummyStream("")

    def close(self):
        pass


class DummySSHClientClass:
    def __init__(self, client):
        self.client = client
        self.hostname = "login"

    def close(self):
        self.client.close()

    def is_proxy(self):
        return False

    def get_transport(self):
        return None


class DummyCredentials(SlurmLoginCredentials):
    user_env = "DUMMY_HPC_USERNAME"
    pw_env = "DUMMY_HPC_PASSWORD"
    ssh_key_env = "DUMMY_SSH_PASSPHRASE"


class DummyConfig(SlurmSiteConfig):
    def __init__(self, queue="debug"):
        super().__init__(
            hostname="login.example.edu",
            queue=queue,
            mpi_wrapper="srun",
            max_nodes=4,
            cores_per_node=8,
            memory_per_node=1024,
        )


class DummySlurmSite(SlurmSite):
    site_name = "Dummy"
    credentials_cls = DummyCredentials
    config_cls = DummyConfig
    default_queue = "debug"
    work_dir_env = "DUMMY_HPC_WORK_DIR"
    default_solver_executable = "/remote/bin/FS"

    def authenticate(self, host=None):
        return DummyRawClient()


def test_generic_slurm_site_can_be_instantiated_without_site_specific_class(
    monkeypatch,
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")

    site = DummySlurmSite("project/run", default_queue="normal")

    assert site.config.queue == "normal"
    assert site.work_dir == Path("/scratch/user/project/run")
    assert site.executable == "/remote/bin/FS"
    assert site.mpi_cmd == "srun"
    assert site.config_for_queue("debug").queue == "debug"


def test_slurm_site_handles_missing_job_id_without_remote_cancel(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")

    site = DummySlurmSite("project/run")

    assert site.update_status() == "unknown"
    assert site.pool.status == "unknown"
    assert site.deprovision() is False


def test_slurm_state_and_sbatch_parsing_helpers():
    assert _normalize_slurm_state("R") == "running"
    assert _normalize_slurm_state("COMPLETED") == "complete"
    assert _normalize_slurm_state("CANCELLED by 1234") == "cancelled"
    assert _normalize_slurm_state("") == "unknown"
    assert _parse_sbatch_job_id("Submitted batch job 12345") == "12345"
    with pytest.raises(ValueError, match="failed to get job ID"):
        _parse_sbatch_job_id("no job id")


def test_submit_slurm_alias_delegates(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")

    def fake_submit(*args, **kwargs):
        return args, kwargs

    site.submit_SLURM = fake_submit

    assert site.submit_slurm("job", nodes=1) == (("job",), {"nodes": 1})


def test_slurm_site_config_validates_node_and_duration_requests():
    config = SlurmSiteConfig(
        hostname="login.example.edu",
        max_duration="00-01:00:00",
        min_nodes=1,
        max_nodes=2,
    )

    assert config.validate_request(1, 1, "00-00:30:00") == "00-00:30:00"
    assert config.validate_request(1, 1, "00-02:00:00") == "00-01:00:00"
    with pytest.raises(ValueError, match="Maximum number of nodes"):
        config.validate_request(3, 3, "00-00:30:00")


def test_pool_status_recognizes_slurm_terminal_states():
    for status in ("complete", "failed", "timeout", "cancelled"):
        assert PoolStatus(status=status).is_complete


def test_stampede3_site_is_specific_slurm_subclass():
    assert issubclass(Stampede3Site, SlurmSite)
    assert Stampede3Site.config_cls is Stampede3Config
    assert Stampede3Site.credentials_cls is TACCLoginCredentials
    assert Stampede3Site.work_dir_env == "STAMPEDE3_WORK_DIR"
