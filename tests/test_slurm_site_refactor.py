import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import BaseSite, JobStatus, RunHandle
from frequensolve.orchestrator.sites.hpc import (
    SlurmLoginCredentials,
    SlurmRunConfig,
    SlurmSite,
    SlurmSiteConfig,
    _normalize_slurm_state,
    _parse_sbatch_job_id,
)
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.stampede3 import (
    Stampede3Config,
    Stampede3Site,
    TACCLoginCredentials,
)
from frequensolve.orchestrator.utils.pool import PoolStatus
from frequensolve.orchestrator.utils.progress import status_table_html, wait_all
from frequensolve.project.project import Project
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.outputs import WavefieldOutput


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


def test_run_handle_watch_finalizes_terminal_status():
    site = DummyBaseSite()
    job = DummyJob()
    states = iter(["running", "completed"])
    finalized = []

    def poll(run):
        return JobStatus(state=next(states), return_code=0, job_id="watch")

    def finalize(run, status):
        finalized.append(status.state)
        return run._make_result(
            JobStatus(
                state="completed",
                return_code=0,
                job_id=status.job_id,
                message="packed",
            )
        )

    run = RunHandle(
        site=site,
        job=job,
        id="watch",
        poll_interval=0.0,
        _status_fn=poll,
        _finalize_fn=finalize,
    )

    statuses = list(run.watch(timeout=1.0, poll_interval=0.0))

    assert [status.state for status in statuses] == ["running", "completed"]
    assert finalized == ["completed"]
    assert statuses[-1].message == "packed"
    assert run.wait().status.message == "packed"


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


def test_slurm_submit_force_run_passes_fresh_to_batch(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    seen = {}

    class CurrentJob(DummyJob):
        def is_run_current(self):
            return True

    def fake_submit(job, config, **kwargs):
        seen["fresh"] = kwargs.get("fresh")
        return "79"

    monkeypatch.setattr(site, "_submit_slurm_batch", fake_submit)

    run = site.submit(CurrentJob(), force_run=True)

    assert run.id == "79"
    assert seen["fresh"] is True


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


def test_job_save_for_remote_writes_remote_absolute_result_path(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    remote_project = Path("/scratch/user/project/run")

    local_job, remote_job = job.save_for_remote("Dummy", remote_project)
    payload = json.loads(Path(local_job).read_text())
    local_payload = json.loads(job.job_file.read_text())

    assert remote_job == remote_project / "jobs" / "simple" / "freq" / "freq.json"
    assert local_job != job.job_file
    assert local_payload["project_path"] == str(project.path)
    assert payload["project_path"] == str(remote_project)
    assert payload["simulation"] == str(
        remote_project / "simulations" / "simple" / "simple.json"
    )
    assert payload["result_path"] == str(
        remote_project / "jobs" / "simple" / "freq" / "results"
    )
    assert str(tmp_path) not in json.dumps(payload)


def test_project_transfer_keeps_mesh_paths_remote_safe(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="axisym", physics="acoustic", dimension=2)
    mesh_file = project.path / "simulations" / "axisym" / "mesh.gmp"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_file.write_text("mesh")
    sim.mesh = MeshManager(file=mesh_file, format="Gmsh")
    remote = Path("/scratch/user/copied_project")
    captured = {}

    class CaptureSite:
        work_dir = remote

        def put(self, local, target):
            local = Path(local)
            if local.is_dir():
                sim_json = local / "simulations" / "axisym" / "axisym.json"
                captured["simulation"] = json.loads(sim_json.read_text())
                captured["mesh_exists"] = (
                    local / "simulations" / "axisym" / "mesh.gmp"
                ).exists()

    sim_payload = json.loads(sim.save().read_text())

    assert sim_payload["Mesh"]["file"] == "simulations/axisym/mesh.gmp"

    project._transfer(CaptureSite())

    assert captured["mesh_exists"] is True
    assert captured["simulation"]["project_path"] == str(remote)
    assert captured["simulation"]["Mesh"]["file"] == "simulations/axisym/mesh.gmp"
    assert str(tmp_path) not in json.dumps(captured["simulation"])


def test_slurm_job_transfer_overwrites_remote_simulation_and_mesh(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="axisym", physics="acoustic", dimension=2)
    mesh_file = project.path / "simulations" / "axisym" / "mesh.gmp"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_file.write_text("mesh")
    sim.mesh = MeshManager(file=mesh_file, format="Gmsh")
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    puts = []
    captured = {}

    def fake_put(local, remote):
        local = Path(local)
        remote = Path(remote)
        puts.append((local, remote))
        if remote == site.work_dir / "simulations" / "axisym" / "axisym.json":
            captured["simulation"] = json.loads(local.read_text())

    monkeypatch.setattr(site, "put", fake_put)
    monkeypatch.setattr(site, "run_login", lambda cmd: "")

    site._transfer_SLURM_job("script", job)

    remote_paths = {remote for _local, remote in puts}
    assert site.work_dir / "simulations" / "axisym" / "axisym.json" in remote_paths
    assert site.work_dir / "simulations" / "axisym" / "mesh.gmp" in remote_paths
    assert site.work_dir / "jobs" / "axisym" / "freq" / "freq.json" in remote_paths
    assert captured["simulation"]["project_path"] == str(site.work_dir)
    assert captured["simulation"]["Mesh"]["file"] == "simulations/axisym/mesh.gmp"
    assert str(tmp_path) not in json.dumps(captured["simulation"])


def test_slurm_batch_maps_local_project_run_path_to_remote(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    seen = {}

    def fake_script(**kwargs):
        seen["script_kwargs"] = kwargs
        return "script"

    monkeypatch.setattr(site, "_sweep_SLURM_script", fake_script)
    monkeypatch.setattr(
        site,
        "_transfer_SLURM_job",
        lambda script, job: (
            Path("/scratch/user/project/run/sweep.slurm"),
            Path("job"),
        ),
    )

    def fake_submit_sbatch(cmd):
        seen["cmd"] = cmd
        return "88"

    monkeypatch.setattr(site, "_submit_sbatch", fake_submit_sbatch)
    monkeypatch.setattr(
        site, "_store_remote_run_records", lambda job, record=None: None
    )

    site._submit_slurm_batch(
        job,
        SlurmRunConfig(queue="debug", run_path=project.path),
    )

    assert seen["script_kwargs"]["run_path"] == site.work_dir
    assert seen["cmd"].startswith(f"mkdir -p {site.work_dir}/jobs/batch")
    record = job.latest_run(site="Dummy")
    assert record is not None
    assert record.scheduler_id == "88"
    assert record.job_file == site.work_dir / "jobs" / "simple" / "freq" / "freq.json"


def test_slurm_submit_ignores_local_current_without_remote_record(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    monkeypatch.setattr(job, "is_run_current", lambda: True)
    monkeypatch.setattr(site, "_sync_project", lambda project: None)
    monkeypatch.setattr(site, "_submit_slurm_batch", lambda job, config, **kwargs: "99")

    run = site.submit(job)

    assert run.id == "99"


def test_slurm_submit_reattaches_matching_inflight_run(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    job.record_site_run(
        site="Dummy",
        work_dir=site.work_dir,
        scheduler_id="77",
        status="running",
    )

    monkeypatch.setattr(site, "update_status", lambda job_id: "running")
    monkeypatch.setattr(site, "_read_scheduler_status", lambda run: None)
    monkeypatch.setattr(
        site,
        "_sync_project",
        lambda project: (_ for _ in ()).throw(AssertionError("synced project")),
    )
    monkeypatch.setattr(
        site,
        "_submit_slurm_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reran job")),
    )

    run = site.submit(job)

    assert run.id == "77"
    assert run.mode == "batch"
    assert run.check is False
    assert run.backend["reattached"] is True
    assert run.status().state == "running"
    assert job.latest_run(site="Dummy").status == "running"


def test_slurm_submit_reattach_compares_numpy_payload_values(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0, 20.0])
    job.save()
    record = job.record_site_run(
        site="Dummy",
        work_dir=site.work_dir,
        scheduler_id="77",
        status="running",
    )
    payload = dict(record.fingerprint_payload)
    payload["job"] = dict(payload["job"])
    payload["job"]["f_list"] = np.asarray(payload["job"]["f_list"])
    job.write_run_record(record.with_updates(fingerprint_payload=payload))

    monkeypatch.setattr(site, "update_status", lambda job_id: "running")
    monkeypatch.setattr(site, "_read_scheduler_status", lambda run: None)
    monkeypatch.setattr(
        site,
        "_sync_project",
        lambda project: (_ for _ in ()).throw(AssertionError("synced project")),
    )
    monkeypatch.setattr(
        site,
        "_submit_slurm_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reran job")),
    )

    run = site.submit(job)

    assert run.id == "77"
    assert run.backend["reattached"] is True


def test_slurm_submit_does_not_reattach_mismatched_inflight_run(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    job.record_site_run(
        site="Dummy",
        work_dir=site.work_dir,
        scheduler_id="77",
        status="running",
    )
    job.f_list = [20.0]
    seen = {"update_status": 0}

    def fake_update_status(job_id):
        seen["update_status"] += 1
        return "running"

    monkeypatch.setattr(site, "update_status", fake_update_status)
    monkeypatch.setattr(site, "_sync_project", lambda project: None)
    monkeypatch.setattr(site, "_submit_slurm_batch", lambda *args, **kwargs: "88")

    run = site.submit(job)

    assert run.id == "88"
    assert run.backend.get("reattached") is None
    assert seen["update_status"] == 0


def test_slurm_submit_does_not_reattach_mismatched_simulation_hash(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    record = job.record_site_run(
        site="Dummy",
        work_dir=site.work_dir,
        scheduler_id="77",
        status="running",
    )
    payload = dict(record.fingerprint_payload)
    payload["simulation"] = {"hash": "sha256:not-the-current-simulation"}
    job.write_run_record(record.with_updates(fingerprint_payload=payload))
    seen = {"update_status": 0}

    def fake_update_status(job_id):
        seen["update_status"] += 1
        return "running"

    monkeypatch.setattr(site, "update_status", fake_update_status)
    monkeypatch.setattr(site, "_sync_project", lambda project: None)
    monkeypatch.setattr(site, "_submit_slurm_batch", lambda *args, **kwargs: "88")

    run = site.submit(job)

    assert run.id == "88"
    assert run.backend.get("reattached") is None
    assert seen["update_status"] == 0


def test_slurm_submit_skips_when_recorded_remote_run_is_current(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    job.record_site_run(
        site="Dummy",
        work_dir=site.work_dir,
        scheduler_id="77",
        status="complete",
    )
    seen = {}

    def fake_run_login(cmd):
        if "find " in cmd:
            return "/scratch/user/project/run/jobs/simple/freq/logs/task_1.log"
        return json.dumps(
            {"task_summary": {"total": 1, "complete": 1, "succeeded": 1, "failed": 0}}
        )

    monkeypatch.setattr(site, "run_login", fake_run_login)
    monkeypatch.setattr(
        site,
        "_submit_slurm_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reran job")),
    )
    monkeypatch.setattr(
        job,
        "write_run_state",
        lambda status="completed", **extra: seen.setdefault("status", status),
    )

    run = site.submit(job)

    assert run.mode == "skipped"
    assert seen["status"] == "skipped"


def test_slurm_submit_does_not_skip_submitted_record(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    job.record_site_run(site="Dummy", work_dir=site.work_dir, scheduler_id="77")

    monkeypatch.setattr(site, "run_login", lambda cmd: "")
    monkeypatch.setattr(site, "_sync_project", lambda project: None)
    monkeypatch.setattr(site, "_submit_slurm_batch", lambda *args, **kwargs: "88")

    run = site.submit(job)

    assert run.id == "88"
    assert run.mode == "batch"


def test_slurm_submit_does_not_skip_manifest_without_task_summary(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    job.record_site_run(
        site="Dummy",
        work_dir=site.work_dir,
        scheduler_id="77",
        status="complete",
    )

    def fake_run_login(cmd):
        if "find " in cmd:
            return "/scratch/user/project/run/jobs/simple/freq/logs/task_1.log"
        return json.dumps({"exit_status": {"code": 0, "status": "success"}})

    monkeypatch.setattr(site, "run_login", fake_run_login)
    monkeypatch.setattr(site, "_sync_project", lambda project: None)
    monkeypatch.setattr(site, "_submit_slurm_batch", lambda *args, **kwargs: "88")

    run = site.submit(job)

    assert run.id == "88"
    assert run.mode == "batch"


def test_slurm_submit_does_not_skip_without_remote_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    job.record_site_run(
        site="Dummy",
        work_dir=site.work_dir,
        scheduler_id="77",
        status="complete",
    )

    def fake_run_login(cmd):
        if "find " in cmd:
            return ""
        return json.dumps(
            {"task_summary": {"total": 1, "complete": 1, "succeeded": 1, "failed": 0}}
        )

    monkeypatch.setattr(site, "run_login", fake_run_login)
    monkeypatch.setattr(site, "_sync_project", lambda project: None)
    monkeypatch.setattr(site, "_submit_slurm_batch", lambda *args, **kwargs: "88")

    run = site.submit(job)

    assert run.id == "88"
    assert run.mode == "batch"


def test_slurm_wait_all_polls_batch_runs_with_combined_status(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job_a = FrequencyDomainJob(name="freq_a", simulation=sim, f_list=[10.0])
    job_b = FrequencyDomainJob(name="freq_b", simulation=sim, f_list=[20.0])
    run_a = site.handle(job_a, job_id="101", mode="batch")
    run_b = site.handle(job_b, job_id="102", mode="batch")
    calls = {}

    def fake_update_status(job_id):
        calls[job_id] = calls.get(job_id, 0) + 1
        return "running" if calls[job_id] == 1 else "complete"

    monkeypatch.setattr(site, "update_status", fake_update_status)
    monkeypatch.setattr(
        site,
        "_read_scheduler_status",
        lambda run: {
            "successful": 1 if calls.get(str(run.id), 0) > 1 else 0,
            "failed": 0,
            "running": 0 if calls.get(str(run.id), 0) > 1 else 1,
            "pending": 0,
            "total": 1,
        },
    )
    results = site.wait([run_a, run_b], poll_interval=0.0, timeout=1.0)

    assert [result.job.name for result in results] == ["freq_a", "freq_b"]
    assert [result.status.state for result in results] == ["complete", "complete"]
    captured = capsys.readouterr()
    assert "Dummy freq_a [101]: running" in captured.out
    assert "Dummy freq_b [102]: running" in captured.out
    assert "Dummy freq_a [101]: complete" in captured.out
    assert "Dummy freq_b [102]: complete" in captured.out


def test_wait_all_status_html_uses_count_columns():
    site = DummyBaseSite()
    job = DummyJob()
    job.name = "freq_a"
    run = RunHandle(site=site, job=job, id="101")
    status = JobStatus(
        state="running",
        job_id="101",
        raw={
            "task_status": {
                "successful": 2,
                "failed": 1,
                "running": 3,
                "pending": 4,
                "total": 10,
            }
        },
    )

    panel = status_table_html([run], {0: status})

    assert "background:#ffffff" in panel
    assert "Progress" not in panel
    for heading in (
        "Site",
        "Job",
        "Job ID",
        "State",
        "Succeeded",
        "Failed",
        "Running",
        "Pending",
        "Total",
    ):
        assert heading in panel
    for value in ("2", "1", "3", "4", "10"):
        assert f">{value}</td>" in panel
    assert "border-left:1px solid #d0d7de" in panel
    assert "background:#052e16" not in panel


def test_global_wait_all_accepts_runs_from_multiple_sites(capsys):
    site_a = DummyBaseSite()
    site_b = DummyBaseSite()
    site_a.site_name = "local"
    site_b.site_name = "stampede"
    job_a = DummyJob()
    job_b = DummyJob()
    job_a.name = "debug"
    job_b.name = "production"
    run_a = RunHandle(
        site=site_a,
        job=job_a,
        id="local:debug",
        poll_interval=0.0,
        _status_fn=lambda run: JobStatus(
            state="completed", return_code=0, job_id=run.id
        ),
    )
    run_b = RunHandle(
        site=site_b,
        job=job_b,
        id="123",
        poll_interval=0.0,
        _status_fn=lambda run: JobStatus(
            state="completed", return_code=0, job_id=run.id
        ),
    )

    results = wait_all([run_a, run_b], poll_interval=0.0)

    assert [result.job.name for result in results] == ["debug", "production"]
    captured = capsys.readouterr()
    assert "local debug [local:debug]: completed" in captured.out
    assert "stampede production [123]: completed" in captured.out


def test_slurm_fetch_logs_also_fetches_run_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")

    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    calls = []
    collected = []

    def fake_get(remote, local, overwrite=False):
        calls.append((Path(remote), Path(local)))
        Path(local).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(site, "get", fake_get)
    monkeypatch.setattr(
        job, "collect_task_run_manifests", lambda: collected.append(True)
    )

    assert site.fetch_logs(job) == job._local_path / "logs"
    assert (
        site.work_dir / "jobs" / "simple" / "freq" / "results" / "_fs_run",
        job._result_path / "_fs_run",
    ) in calls
    assert (
        site.work_dir / "jobs" / "simple" / "freq" / "logs",
        job._local_path / "logs",
    ) in calls
    assert collected == [True]


def test_slurm_fetch_wavefields_downloads_wavefield_output(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")

    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job += WavefieldOutput(
        name="pressure_wavefield",
        field="pressure",
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
    )
    job.save()
    calls = []
    opened = {}

    def fake_get(remote, local, overwrite=False):
        calls.append((Path(remote), Path(local)))
        Path(local).mkdir(parents=True, exist_ok=True)

    def fake_from_manifest(cls, manifest, upscale=1):
        opened["groups"] = manifest.groups
        opened["output_path"] = manifest.output_path
        opened["upscale"] = upscale
        return "wavefield-db"

    monkeypatch.setattr(site, "get", fake_get)
    monkeypatch.setattr(
        hpc.TraceDataset,
        "from_manifest",
        classmethod(fake_from_manifest),
    )

    assert site.fetch_wavefields(job, upscale=4) == "wavefield-db"
    assert calls == [
        (
            site.work_dir / "jobs" / "simple" / "freq" / "results" / "wavefields",
            job._local_path / "results" / "wavefields",
        )
    ]
    assert opened == {
        "groups": ["pressure_wavefield"],
        "output_path": job._result_path / "wavefields",
        "upscale": 4,
    }


def test_slurm_batch_poll_reads_scheduler_status(monkeypatch, capsys):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setenv("DUMMY_HPC_WORK_DIR", "/scratch/user")
    site = DummySlurmSite("project/run")

    monkeypatch.setattr(site, "update_status", lambda job_id: "running")
    monkeypatch.setattr(
        site,
        "run_login",
        lambda cmd: (
            '{"state":"running","total":8,"successful":2,'
            '"failed":1,"running":3,"pending":2,"complete":3}'
        ),
    )

    run = RunHandle(site=site, job=DummyJob(), id="77", mode="batch")
    status = site._poll_run(run)

    captured = capsys.readouterr()
    assert status.state == "running"
    assert status.raw["task_status"]["successful"] == 2
    assert status.message == (
        "tasks: 2 successful, 1 failed, 3 running, 2 pending, 8 total"
    )
    assert captured.out == ""


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
    fresh_batch_script = site._sweep_SLURM_script(
        n_tasks=4,
        n_nodes=1,
        stdout="/scratch/user/project/run/jobs/dummy/logs",
        duration="00-00:30:00",
        procs_per_node=4,
        fresh=True,
    )
    attached_script = site._sweep_script(DummyJob())
    fresh_attached_script = site._sweep_script(DummyJob(), fresh=True)
    attached_disabled_script = site._sweep_script(DummyJob(), pack=False)
    disabled_script = site._sweep_SLURM_script(
        n_tasks=4,
        n_nodes=1,
        stdout="/scratch/user/project/run/jobs/dummy/logs",
        duration="00-00:30:00",
        procs_per_node=4,
        pack=False,
    )

    assert '$fresh_flag --pack >> "$dir_out/pack.log" 2>&1' in batch_script
    assert "scheduler_status.json" in batch_script
    assert "FS_SCHEDULER_STATUS" in batch_script
    assert '"successful"' in batch_script
    assert '"pending"' in batch_script
    assert "$fresh_flag --pack >> $dir_out/pack.log 2>&1" in attached_script
    assert 'fresh_flag="--fresh"' in fresh_batch_script
    assert 'fresh_flag="--fresh"' in fresh_attached_script
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
        site, "_submit_attached", lambda job, procs_per_task=2, **kwargs: DummyFuture()
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

    def fake_submit(job, procs_per_task=2, *, pack=True, fresh=False):
        seen["pack"] = pack
        seen["fresh"] = fresh
        return DummyFuture()

    monkeypatch.setattr(site, "_submit_attached", fake_submit)

    site.submit(DummyJob(), pack=False)

    assert seen["pack"] is False
    assert seen["fresh"] is False


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
            site, "_submit_attached", lambda job, procs_per_task=2, **kwargs: future
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
