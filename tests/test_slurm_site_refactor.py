import asyncio
import logging
from pathlib import Path

import pytest

from frequensolve.orchestrator.pool import PoolStatus
from frequensolve.orchestrator.sites import hpc
from frequensolve.orchestrator.sites.base import BaseSite, JobStatus, RunHandle
from frequensolve.orchestrator.sites.hpc import (
    SlurmLoginCredentials,
    SlurmRunConfig,
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


class DummyJob:
    name = "dummy"
    n_tasks = 4
    _job_id = None

    def is_run_current(self):
        return False

    def write_run_state(self, status="completed", **extra):
        self.last_run_state = status

    def _remote_path(self, work_dir):
        return Path(work_dir) / "jobs" / self.name


class DummyBaseSite(BaseSite):
    def submit(self, job, **kwargs):
        self.submit_kwargs = kwargs
        return RunHandle(
            site=self,
            job=job,
            id="base",
            poll_interval=0,
            _status_fn=lambda run: JobStatus(state="completed", return_code=0),
        )

    def cancel_job(self, job_id: str) -> None:
        self.cancelled = job_id


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


def test_sbatch_stderr_with_job_id_is_not_logged_as_error(monkeypatch, caplog):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")
    monkeypatch.setattr(
        site,
        "run_login_cmd",
        lambda cmd: (
            None,
            DummyStream("Submitted batch job 12345\n"),
            DummyStream("module reload notice\n"),
        ),
    )

    with caplog.at_level(logging.DEBUG, logger=hpc.__name__):
        job_id = site._submit_sbatch("sbatch job.sh")

    assert job_id == "12345"
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert any("sbatch stderr" in record.message for record in caplog.records)


def test_legacy_slurm_public_methods_are_removed():
    assert not hasattr(SlurmSite, "submit_SLURM")
    assert not hasattr(SlurmSite, "submit_slurm")
    assert not hasattr(SlurmSite, "wait_completion")
    assert not hasattr(SlurmSite, "attach_to_existing_job")
    assert not hasattr(SlurmSite, "wait_provisioned")


def test_run_handle_wait_and_await():
    site = DummyBaseSite()
    job = DummyJob()
    states = iter(["pending", "running", "completed"])

    def poll(run):
        return JobStatus(state=next(states), return_code=0, job_id="1")

    run = RunHandle(site=site, job=job, id="1", poll_interval=0.0, _status_fn=poll)

    result = run.wait(timeout=1.0)

    assert result.successful
    assert result.status.state == "completed"

    async_states = iter(["pending", "completed"])
    async_run = RunHandle(
        site=site,
        job=job,
        id="2",
        poll_interval=0.0,
        _status_fn=lambda run: JobStatus(
            state=next(async_states), return_code=0, job_id="2"
        ),
    )

    async def wait_for_run():
        return await async_run

    async_result = asyncio.run(wait_for_run())

    assert async_result.successful
    assert async_result.status.state == "completed"


def test_run_handle_wait_prints_status_output_by_default(capsys):
    site = DummyBaseSite(verbose=False)
    run = site.submit(DummyJob())

    run.wait(timeout=1.0, poll_interval=0.0)

    captured = capsys.readouterr()
    assert "\033[38;5;244mDummyBaseSite: \033[38;5;40mcompleted\033[0m" in captured.out


def test_skipped_run_handle_prints_status_by_default(capsys):
    site = DummyBaseSite(verbose=False)

    RunHandle.skipped(site, DummyJob())

    captured = capsys.readouterr()
    assert (
        "\033[38;5;244mDummyBaseSite: \033[38;5;244mskipped\033[0m - Run is current"
        in captured.out
    )


def test_run_handle_cancel_delegates_to_site():
    site = DummyBaseSite()
    run = RunHandle(site=site, job=DummyJob(), id="123")

    run.cancel()

    assert site.cancelled == "123"


def test_site_run_separates_submit_and_wait_options():
    site = DummyBaseSite()

    result = site.run(DummyJob(), timeout=1.0, poll_interval=0.0, force=True)

    assert result.successful
    assert site.submit_kwargs == {"force": True}


def test_base_handle_requires_polling_support():
    class SiteWithoutPolling(BaseSite):
        def submit(self, job, **kwargs):
            raise NotImplementedError

        def cancel_job(self, job_id: str) -> None:
            pass

    with pytest.raises(NotImplementedError, match="cannot poll"):
        SiteWithoutPolling().handle(DummyJob(), job_id="1")


def test_slurm_site_accepts_run_config(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")

    run_config = SlurmRunConfig(queue="normal", nodes=2, duration="00-00:30:00")
    site = DummySlurmSite("project/run", run_config=run_config)

    assert site.run_config is run_config
    assert site.run_config.nodes == 2


def test_slurm_site_verbose_initialization(monkeypatch, capsys):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")

    DummySlurmSite("project/run", verbose=True)

    captured = capsys.readouterr()
    assert "Dummy initialized with work_dir: /scratch/user/project/run" in captured.out


def test_slurm_submit_auto_uses_batch_when_not_attached(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite(
        "project/run",
        run_config=SlurmRunConfig(queue="debug", nodes=2, duration="00-00:30:00"),
    )
    seen = {}

    def fake_submit(job, config, **kwargs):
        seen["config"] = config
        return "77"

    monkeypatch.setattr(site, "_submit_slurm_batch", fake_submit)

    run = site.submit(DummyJob())

    assert isinstance(run, RunHandle)
    assert run.id == "77"
    assert run.mode == "batch"
    assert seen["config"].nodes == 2
    assert seen["config"].queue == "debug"


def test_slurm_submit_overrides_site_run_config(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite(
        "project/run",
        run_config=SlurmRunConfig(queue="debug", nodes=1, duration="00-00:30:00"),
    )
    seen = {}

    def fake_submit(job, config, **kwargs):
        seen["config"] = config
        return "78"

    monkeypatch.setattr(site, "_submit_slurm_batch", fake_submit)

    site.submit(DummyJob(), nodes=3, queue="normal", duration="00-00:45:00")

    assert seen["config"].nodes == 3
    assert seen["config"].queue == "normal"
    assert seen["config"].duration == "00-00:45:00"


def test_slurm_sweep_scripts_run_solver_pack_after_tasks(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")
    site.pool.nproc = 4
    site.pool.ncore = 8

    batch_script = site._sweep_SLURM_script(
        n_tasks=4,
        n_nodes=1,
        stdout="/scratch/user/project/run/jobs/dummy/logs",
        duration="00-00:30:00",
        procs_per_node=4,
    )
    attached_script = site._sweep_script(DummyJob())
    attached_disabled_script = site._sweep_script(DummyJob(), pack=False)
    disabled_script = site._sweep_SLURM_script(
        n_tasks=4,
        n_nodes=1,
        stdout="/scratch/user/project/run/jobs/dummy/logs",
        duration="00-00:30:00",
        procs_per_node=4,
        pack=False,
    )

    assert '--pack >> "$dir_out/pack.log" 2>&1' in batch_script
    assert "--pack >> $dir_out/pack.log 2>&1" in attached_script
    assert "--pack" not in attached_disabled_script
    assert "--pack" not in disabled_script


def test_slurm_submit_attached_requires_active_allocation(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")

    with pytest.raises(RuntimeError, match="No active compute allocation"):
        site.submit(DummyJob(), mode="attached")


def test_slurm_submit_auto_uses_attached_when_provisioned(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")

    class DummyFuture:
        def done(self):
            return False

    monkeypatch.setattr(
        site, "_submit_attached", lambda job, procs_per_task=2: DummyFuture()
    )

    run = site.submit(DummyJob())

    assert run.mode == "attached"
    assert run.status().state == "running"


def test_slurm_submit_attached_can_disable_pack(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")
    seen = {}

    class DummyFuture:
        def done(self):
            return False

    def fake_submit(job, procs_per_task=2, *, pack=True):
        seen["pack"] = pack
        return DummyFuture()

    monkeypatch.setattr(site, "_submit_attached", fake_submit)

    site.submit(DummyJob(), pack=False)

    assert seen["pack"] is False


def test_slurm_allocation_attach_returns_awaitable_handle(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")
    states = iter(["pending", "running"])
    attached = {}

    monkeypatch.setattr(site, "update_status", lambda job_id=None: next(states))
    monkeypatch.setattr(
        site, "_attach_compute_client", lambda: attached.setdefault("ok", True)
    )

    allocation = site.attach_allocation("99")
    result = allocation.wait(poll_interval=0.0)

    assert allocation.mode == "allocation"
    assert result.successful
    assert result.status.raw["scheduler_state"] == "running"
    assert attached["ok"] is True


def test_slurm_attach_allocation_requires_explicit_id_when_multiple_jobs(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")
    monkeypatch.setattr(
        site,
        "_list_jobs",
        lambda: "101 node01 1 R 00:10\n102 node02 1 PD 00:20\n",
    )

    with pytest.raises(RuntimeError, match="pass job_id explicitly"):
        site.attach_allocation()


def test_slurm_context_manager_closes_clients(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")

    site.close()

    assert site._login_client is None
    assert site._compute_client is None


def test_slurm_provision_returns_allocation_handle(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")

    monkeypatch.setattr(site, "put", lambda local, remote: None)
    monkeypatch.setattr(site, "_submit_sbatch", lambda cmd: "44")
    monkeypatch.setattr(
        site, "_generate_provision_script", lambda *args, **kwargs: "script"
    )

    allocation = site.provision(nodes=1, tasks=1, duration="00-00:10:00")

    assert isinstance(allocation, RunHandle)
    assert allocation.id == "44"
    assert allocation.mode == "allocation"


def test_slurm_attached_run_is_awaitable(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")

    async def submit_and_wait():
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        monkeypatch.setattr(
            site, "_submit_attached", lambda job, procs_per_task=2: future
        )
        run = site.submit(DummyJob())
        return await run

    result = asyncio.run(submit_and_wait())

    assert result.successful
    assert result.status.state == "completed"


def test_slurm_attached_sync_wait_inside_event_loop_has_clear_error(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")

    async def call_sync_wait():
        future = asyncio.get_running_loop().create_future()
        run = RunHandle(site=site, job=DummyJob(), id="attached", mode="attached")
        run.backend["future"] = future
        with pytest.raises(RuntimeError, match="use 'await run' instead"):
            site._wait_attached_run(run)

    asyncio.run(call_sync_wait())


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
