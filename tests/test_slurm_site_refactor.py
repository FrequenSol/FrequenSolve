import asyncio
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunHandle,
    SubmitPlan,
)
from frequensolve.orchestrator.sites.config_file import _host_tmp_path_for_config
from frequensolve.orchestrator.sites.hpc import (
    SlurmLoginCredentials,
    SlurmPartitionConfig,
    SlurmRunConfig,
    SlurmSite,
    SlurmSiteConfig,
    _normalize_slurm_state,
    _parse_sbatch_job_id,
)
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.slurm_helpers import temporary_text_file
from frequensolve.orchestrator.sites.hpc.stampede3 import (
    Stampede3Config,
    Stampede3Site,
    TACCLoginCredentials,
)
from frequensolve.orchestrator.utils.pool import PoolStatus
from frequensolve.orchestrator.utils.progress import status_table_html, wait_all
from frequensolve.project.project import Project
from frequensolve.simulation.jobs import FrequencyDomainJob, SkipPolicy
from frequensolve.simulation.jobs.imaging import ImagingJob
from frequensolve.simulation.outputs import WavefieldOutput

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


class DummyStream:
    def __init__(self, text=""):
        self._text = text

    def read(self):
        return self._text.encode()


class DummyRawClient:
    def exec_command(self, command, environment=None):
        if command == "echo $WORK":
            return None, DummyStream("/scratch/user\n"), DummyStream("")
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
    default_solver_executable = "/remote/bin/FS_seismic"

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


def test_site_dry_run_reports_run_and_skip_counts_without_task_plan():
    class DryRunJob(DummyJob):
        name = "dry"
        n_tasks = 4
        f_list = [10.0, 20.0, 30.0, 40.0]

        def __init__(self):
            self.saved = False
            self.validated = False
            self.task_plan_called = False

        def save(self):
            self.saved = True

        def validate(self, raise_errors=True):
            self.validated = raise_errors

        def current_tasks(self):
            return [2, 4]

        def task_run_plan(self, reuse=True, force=False):
            self.task_plan_called = True
            raise AssertionError("dry_run must not call task_run_plan")

    site = DummyBaseSite()
    job = DryRunJob()

    plan = site.dry_run(job)

    assert isinstance(plan, SubmitPlan)
    assert plan.n_tasks_to_run == 2
    assert plan.n_tasks_to_skip == 2
    assert plan.pending_tasks == (1, 3)
    assert plan.skipped_tasks == (2, 4)
    assert plan.pending_indices == (0, 2)
    assert plan.pending_frequencies == (10.0, 30.0)
    assert "2 / 4 frequency tasks would run; 2 would skip" in str(plan)
    assert "Pending tasks: 1, 3" in str(plan)
    assert "Skipped tasks: 2, 4" in str(plan)
    assert job.saved is True
    assert job.validated is True
    assert job.task_plan_called is False

    force_plan = site.dry_run(job, skip="false")

    assert force_plan.n_tasks_to_run == 4
    assert force_plan.n_tasks_to_skip == 0
    assert force_plan.pending_tasks == (1, 2, 3, 4)
    assert force_plan.skipped_tasks == ()


def test_site_dry_run_reports_reusable_frequency_outputs(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 3.0])
    job.save()
    old_files = job.expected_trace_files()
    old_files[0].parent.mkdir(parents=True, exist_ok=True)
    old_files[0].write_text("1 Hz")
    old_files[1].write_text("3 Hz")
    job.write_run_state(status="completed")

    expanded = FrequencyDomainJob(
        name="freq",
        simulation=sim,
        f_list=[1.0, 2.0, 3.0],
    )
    expanded.save()
    new_files = expanded.expected_trace_files()

    plan = DummyBaseSite().dry_run(expanded)

    assert plan.n_tasks_to_run == 1
    assert plan.n_tasks_to_skip == 2
    assert plan.current_tasks == (1,)
    assert plan.reused_tasks == (3,)
    assert plan.skipped_tasks == (1, 3)
    assert plan.pending_tasks == (2,)
    assert plan.pending_indices == (1,)
    assert "1 current, 1 reusable" in str(plan)
    assert not new_files[2].exists()

    no_reuse = DummyBaseSite().dry_run(expanded, reuse=False)

    assert no_reuse.pending_tasks == (2, 3)
    assert no_reuse.current_tasks == (1,)
    assert no_reuse.reused_tasks == ()


def test_site_dry_run_reports_existing_failed_traces_that_would_rerun(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    trace_file = job.expected_trace_files()[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text("trace exists")
    job.write_run_state(
        status="partial",
        tasks=[
            {
                "task": 1,
                "status": "failed",
                "fingerprint": job.task_fingerprint(1),
            }
        ],
    )

    plan = DummyBaseSite().dry_run(job)

    assert plan.n_tasks_to_run == 1
    assert plan.n_tasks_to_skip == 0
    assert plan.failed_existing_tasks == (1,)
    assert "Existing traces marked failed; would rerun: 1" in str(plan)


def test_site_dry_run_tolerant_policy_accepts_failed_trace_below_residual(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    trace_file = job.expected_trace_files()[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text("trace exists")
    job.write_run_state(
        status="partial",
        tasks=[
            {
                "task": 1,
                "status": "failed",
                "fingerprint": job.task_fingerprint(1),
                "path": job._stored_trace_path(trace_file),
                "solver": {
                    "convergence": {
                        "converged": False,
                        "status": "failed",
                        "solve_count": 1,
                        "failure_count": 1,
                        "residual": 5.0e-4,
                    }
                },
            }
        ],
    )

    strict = DummyBaseSite().dry_run(job)
    tolerant = DummyBaseSite().dry_run(job, skip="tolerant", residual=1.0e-3)
    compatible_with_residual = DummyBaseSite().dry_run(
        job,
        skip="compatible",
        residual=1.0e-3,
    )

    assert strict.pending_tasks == (1,)
    assert strict.failed_existing_tasks == (1,)
    assert tolerant.pending_tasks == ()
    assert tolerant.accepted_failed_tasks == (1,)
    assert tolerant.failed_existing_tasks == ()
    assert "1 accepted failed" in str(tolerant)
    assert compatible_with_residual.pending_tasks == ()
    assert compatible_with_residual.accepted_failed_tasks == (1,)
    assert compatible_with_residual.failed_existing_tasks == ()


def test_task_run_plan_tolerant_policy_marks_accepted_failed_current(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    trace_file = job.expected_trace_files()[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text("trace exists")
    job.write_run_state(
        status="partial",
        tasks=[
            {
                "task": 1,
                "status": "failed",
                "fingerprint": job.task_fingerprint(1),
                "path": job._stored_trace_path(trace_file),
                "solver": {
                    "convergence": {
                        "converged": False,
                        "status": "failed",
                        "solve_count": 1,
                        "failure_count": 1,
                        "residual": 5.0e-4,
                    }
                },
            }
        ],
    )

    plan = job.task_run_plan(skip_policy=SkipPolicy.tolerant(residual=1.0e-3))
    state = job.run_state()

    assert plan["pending_indices"] == []
    assert plan["accepted_failed_tasks"] == [1]
    assert state["tasks"][0]["status"] == "accepted_failed"
    assert state["task_summary"] == {
        "total": 1,
        "complete": 1,
        "succeeded": 1,
        "failed": 0,
        "not_run": 0,
    }


def test_dry_run_compatible_policy_ignores_solver_option_changes(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    sim.solver.tolerance = 1.0e-6
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    trace_file = job.expected_trace_files()[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text("trace exists")
    job.write_run_state(status="completed")

    sim.solver.tolerance = 1.0e-3
    sim.save()
    changed = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    changed.save()

    strict = DummyBaseSite().dry_run(changed)
    compatible = DummyBaseSite().dry_run(changed, skip="compatible")

    assert strict.pending_tasks == (1,)
    assert compatible.pending_tasks == ()
    assert compatible.accepted_tasks == (1,)
    assert "1 compatible" in str(compatible)


def test_generic_slurm_site_can_be_instantiated_without_site_specific_class(
    monkeypatch,
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    site = DummySlurmSite("project/run", default_queue="normal")

    assert site.config.queue == "normal"
    assert site.work_dir == Path("/scratch/user/project/run")
    assert site.executable == "/remote/bin/FS_seismic"
    assert site.mpi_cmd == "srun"
    assert site.config_for_queue("debug").queue == "debug"


def test_slurm_site_defaults_to_frequensolve_under_remote_work(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    site = DummySlurmSite()

    assert site.work_dir == Path("/scratch/user/frequensolve")


def test_slurm_site_uses_configured_base_dir_and_stores_scratch_dir(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    site = DummySlurmSite(
        work_dir="/work/user/frequensolve",
        scratch_dir="/scratch/user/frequensolve",
    )

    assert site.work_dir == Path("/work/user/frequensolve")
    assert site.scratch_dir == Path("/scratch/user/frequensolve")


def test_slurm_site_uses_configured_paths_and_runtime(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    site = DummySlurmSite(
        "project/run",
        solver="/configured/bin/FS",
        work_dir="/configured/work",
        modules=["compiler/1.0", "mpi"],
        environment={"MKL_NUM_THREADS": 4, "OMP_NUM_THREADS": 2},
        username="configured-user",
        credential_store=object(),
    )

    assert site.work_dir == Path("/configured/work/project/run")
    assert site.executable == "/configured/bin/FS"
    runtime_setup = site._runtime_setup_lines()
    assert runtime_setup[:3] == [
        "module load compiler/1.0",
        "module load mpi",
        "module list",
    ]
    assert {
        "export MKL_DYNAMIC=FALSE",
        "export MKL_NUM_THREADS=4",
        "export OMP_WAIT_POLICY=PASSIVE",
        "export KMP_STACKSIZE=20M",
        "export OMP_NUM_THREADS=2",
    } <= set(runtime_setup)
    assert site._render_template("sweep/sweep_SLURM.sh", runtime_setup=[]).startswith(
        "#!/bin/bash"
    )
    assert site.credentials.username == "configured-user"


def test_slurm_site_exposes_configured_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    config = SlurmSiteConfig(
        hostname="login.example.edu",
        queue="debug",
        tmp_dir="/scratch/user/frequensolve-tmp",
    )
    site = DummySlurmSite("project/run", config=config)
    config_path = tmp_path / "site.toml"
    config_path.write_text(
        f'[host]\ntmp_dir = "{(tmp_path / "local-staging").as_posix()}"\n'
    )
    site._site_config_path = config_path

    assert site.remote_tmp_dir == Path("/scratch/user/frequensolve-tmp")
    assert _host_tmp_path_for_config(site._site_config_path) == (
        tmp_path / "local-staging"
    )
    assert not hasattr(site, "local_host_tmp_dir")
    assert not hasattr(site, "local_host_config")


def test_temporary_slurm_script_does_not_use_current_directory(monkeypatch, tmp_path):
    working_directory = tmp_path / "read-only-cwd"
    working_directory.mkdir()
    working_directory.chmod(0o555)
    monkeypatch.chdir(working_directory)

    try:
        with temporary_text_file(
            "#!/bin/sh\n",
            suffix=".sh",
            prefix="slurm_",
        ) as script_path:
            assert script_path.parent != working_directory
            assert script_path.read_text() == "#!/bin/sh\n"
            assert script_path.stat().st_mode & 0o777 == 0o700
    finally:
        working_directory.chmod(0o755)


def test_slurm_site_rejects_relative_remote_tmp_dir(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    config = SlurmSiteConfig(hostname="login.example.edu", tmp_dir="scratch/tmp")
    site = DummySlurmSite("project/run", config=config)

    with pytest.raises(ValueError, match="tmp_dir.*absolute"):
        _ = site.remote_tmp_dir


@pytest.mark.parametrize("name", ["work_dir", "scratch_dir"])
def test_slurm_site_rejects_relative_remote_work_and_scratch_dirs(monkeypatch, name):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    with pytest.raises(ValueError, match=rf"{name}.*absolute"):
        DummySlurmSite(**{name: "relative/path"})


def test_slurm_fetch_message_uses_multiline_from_to():
    message = SlurmSite._fetch_message(
        "Fetching logs",
        Path("/remote/job/logs"),
        Path("/local/job/logs"),
    )

    assert message == (
        "Fetching logs\n" "\tFrom: /remote/job/logs\n" "\tTo: /local/job/logs"
    )


def test_slurm_site_rejects_string_modules_and_secret_environment(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    with pytest.raises(ValueError, match="modules must be an array"):
        DummySlurmSite(
            "project/run",
            work_dir="/scratch/user",
            modules="compiler mpi",
        )
    with pytest.raises(ValueError, match="Credential variable"):
        DummySlurmSite(
            "project/run",
            work_dir="/scratch/user",
            environment={"SERVICE_PASSWORD": "secret"},
        )


def test_slurm_site_handles_missing_job_id_without_remote_cancel(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

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

    run_config = SlurmRunConfig(queue="normal", nodes=2, duration="00-00:30:00")
    site = DummySlurmSite("project/run", run_config=run_config)

    assert site.run_config is run_config
    assert site.run_config.nodes == 2


def test_slurm_provision_uses_profile_account(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite(
        "project/run",
        run_config=SlurmRunConfig(account="allocation-from-profile"),
    )

    script = site._generate_provision_script(
        n_nodes=1,
        ranks_per_node=2,
        duration="00-00:30:00",
        queue="debug",
    )

    assert "#SBATCH -A allocation-from-profile" in script


def test_slurm_site_verbose_initialization(monkeypatch, capsys):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)

    DummySlurmSite("project/run", verbose=True)

    captured = capsys.readouterr()
    assert "Dummy initialized with work_dir: /scratch/user/project/run" in captured.out


def test_slurm_submit_auto_uses_batch_when_not_attached(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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


@pytest.mark.parametrize("skip", [False, "false"])
def test_slurm_submit_skip_false_passes_fresh_to_batch(monkeypatch, skip):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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

    run = site.submit(CurrentJob(), skip=skip)

    assert run.id == "79"
    assert seen["fresh"] is True


def test_slurm_submit_runs_smooth_only_for_current_imaging_shards(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="smooth", physics="elastic", dimension=2)
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )
    observed = project.path / "observed" / "traces"
    observed.mkdir(parents=True)
    job = ImagingJob(
        name="rtm",
        simulation=sim,
        data_path=observed,
        f_list=[5.0],
        grid=CartesianGrid(n=[2, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )
    job.save()
    trace_file = job.expected_trace_files()[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.touch()
    job.write_run_state(status="completed")
    seen = {}

    monkeypatch.setattr(site, "_remote_image_part_outputs_exist", lambda job: True)
    monkeypatch.setattr(site, "_remote_image_output_exists", lambda job: False)
    monkeypatch.setattr(site, "_sync_project", lambda project: None)

    def fake_submit(job, config, **kwargs):
        seen.update(kwargs)
        return "80"

    monkeypatch.setattr(site, "_submit_slurm_batch", fake_submit)

    run = site.submit(job, validate=False)

    assert run.id == "80"
    assert seen["smooth_only"] is True
    assert seen["task_plan"]["pending_indices"] == []


def test_slurm_submit_overrides_site_run_config(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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

    run = site.submit(
        DummyJob(),
        nodes=3,
        queue="normal",
        ranks_per_node=7,
        duration="00-00:45:00",
        mpi_async_progress=False,
        scheduler_heartbeat_timeout=17,
    )

    assert seen["config"].nodes == 3
    assert seen["config"].queue == "normal"
    assert seen["config"].ranks_per_node == 7
    assert seen["config"].duration == "00-00:45:00"
    assert seen["config"].mpi_async_progress is False
    assert seen["config"].scheduler_heartbeat_timeout == 17.0
    assert run.backend["mpi_async_progress"] is False
    assert run.backend["scheduler_heartbeat_timeout"] == 17.0


def test_slurm_submit_exposes_direct_resource_parameters():
    parameters = inspect.signature(SlurmSite.submit).parameters

    assert {"queue", "nodes", "ranks_per_node", "duration"} <= set(parameters)


def test_slurm_run_config_normalizes_rank_aliases():
    config = SlurmRunConfig(
        procs_per_node=4,
        procs_per_task=2,
    )

    assert config.ranks_per_node == 4
    assert config.ranks_per_task == 2
    assert config.procs_per_node == 4
    assert config.procs_per_task == 2
    assert config.mpi_async_progress is False
    assert SlurmRunConfig(tolerate_failures=None).tolerate_failures is None

    merged = config.merged(ranks_per_node=8, procs_per_task=3)

    assert merged.ranks_per_node == 8
    assert merged.ranks_per_task == 3
    assert merged.mpi_async_progress is False
    with pytest.raises(ValueError, match="Pass either"):
        config.merged(ranks_per_node=8, procs_per_node=4)

    with pytest.raises(ValueError, match="mpi_async_progress"):
        SlurmRunConfig(mpi_async_progress="yes")

    with pytest.raises(NotImplementedError, match="race during MPI initialization"):
        SlurmRunConfig(mpi_async_progress=True)


def test_slurm_run_config_validates_scheduler_heartbeat_timeout():
    assert SlurmRunConfig().scheduler_heartbeat_timeout == 60.0
    assert (
        SlurmRunConfig(scheduler_heartbeat_timeout=None).scheduler_heartbeat_timeout
        is None
    )

    with pytest.raises(ValueError, match="scheduler_heartbeat_timeout"):
        SlurmRunConfig(scheduler_heartbeat_timeout=0)


def test_existing_slurm_run_config_public_call_shape_is_unchanged():
    assert tuple(inspect.signature(SlurmRunConfig).parameters) == (
        "queue",
        "nodes",
        "duration",
        "ranks_per_node",
        "ranks_per_task",
        "mpi_async_progress",
        "tolerate_failures",
        "account",
        "notify_on",
        "notify_email",
        "poll_interval",
        "scheduler_heartbeat_timeout",
        "run_path",
        "slurm_args",
        "aliases",
    )
    config = SlurmRunConfig(
        queue="existing-partition",
        nodes=2,
        duration="00-00:30:00",
        ranks_per_node=4,
        ranks_per_task=2,
        account="existing-account",
        slurm_args=["--exclusive"],
    )
    assert config.queue == "existing-partition"
    assert config.account == "existing-account"
    assert config.slurm_args == ["--exclusive"]


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
    staging_parent = tmp_path / "transfer-staging"
    config_path = tmp_path / "site.toml"
    config_path.write_text(f'[host]\ntmp_dir = "{staging_parent.as_posix()}"\n')
    captured = {}

    class CaptureSite:
        work_dir = remote
        _site_config_path = config_path

        def put(self, local, target):
            local = Path(local)
            if local.is_dir():
                captured["staging_parent"] = local.parent
                sim_json = local / "simulations" / "axisym" / "axisym.json"
                captured["simulation"] = json.loads(sim_json.read_text())
                captured["mesh_exists"] = (
                    local / "simulations" / "axisym" / "mesh.gmp"
                ).exists()

    sim_payload = json.loads(sim.save().read_text())

    assert sim_payload["Mesh"]["file"] == "simulations/axisym/mesh.gmp"

    project._transfer(CaptureSite())

    assert captured["mesh_exists"] is True
    assert captured["staging_parent"] == staging_parent
    assert project.path not in captured["staging_parent"].parents
    assert captured["simulation"]["project_path"] == str(remote)
    assert captured["simulation"]["Mesh"]["file"] == "simulations/axisym/mesh.gmp"
    assert str(tmp_path) not in json.dumps(captured["simulation"])


def test_slurm_job_transfer_overwrites_remote_simulation_and_mesh(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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
    assert site.work_dir / "adaptive_scheduler.py" in remote_paths
    runner = next(
        local for local, remote in puts if remote.name == "adaptive_scheduler.py"
    )
    assert runner.name == "adaptive_scheduler.py"
    assert captured["simulation"]["project_path"] == str(site.work_dir)
    assert captured["simulation"]["Mesh"]["file"] == "simulations/axisym/mesh.gmp"
    assert str(tmp_path) not in json.dumps(captured["simulation"])


def test_slurm_batch_maps_local_project_run_path_to_remote(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
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
    assert "executable" not in seen["script_kwargs"]
    assert seen["cmd"].startswith(
        f"mkdir -p {site.work_dir}/jobs/simple/freq/logs/batch"
    )
    assert f"{site.work_dir}/jobs/batch" not in seen["cmd"]
    assert (
        f"rm -f {site.work_dir}/jobs/simple/freq/logs/scheduler_status.json"
        in seen["cmd"]
    )
    record = job.latest_run(site="Dummy")
    assert record is not None
    assert record.scheduler_id == "88"
    assert record.job_file == site.work_dir / "jobs" / "simple" / "freq" / "freq.json"


def test_attached_hpc_script_uses_fs_seismic_router(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    site.pool.nhost = 1
    site.pool.nproc = 2
    site.pool.ncore = 8

    script = site._sweep_script(DummyJob(), executable="/remote/bin/fs3d")

    assert "executable=/remote/bin/FS_seismic\n" in script
    assert "executable=/remote/bin/fs3d\n" not in script


def test_batch_hpc_script_uses_fs_seismic_router_in_scheduler_config(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        stdout="/scratch/user/jobs/simple/freq/logs",
        executable="/remote/bin/fs3d",
    )

    assert "executable=/remote/bin/FS_seismic\n" in script
    assert '"executable": "/remote/bin/FS_seismic"' in script
    assert "/remote/bin/fs3d" not in script


def test_batch_hpc_script_keeps_scheduler_logs_with_job_logs(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    log_dir = "/scratch/user/jobs/simple/freq/logs"

    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        stdout=log_dir,
    )

    assert f"#SBATCH -o {log_dir}/batch/job_%j.o" in script
    assert f"#SBATCH -e {log_dir}/batch/job_%j.e" in script
    assert "jobs/batch" not in script
    assert 'mkdir -p "$dir_out/batch"' in script
    assert "! -name batch -exec rm -rf -- {} +" in script
    sizing_file = "/scratch/user/jobs/simple/freq/FS_sizing.json"
    assert f"sizing_json={sizing_file}" in script
    assert f'"sizing_json": "{sizing_file}"' in script


def test_batch_hpc_script_preserves_explicit_sizing_path(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    sizing_file = "/scratch/user/checkpoints/custom-sizing.json"

    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        stdout="/scratch/user/jobs/simple/freq/logs",
        sizing_json=sizing_file,
    )

    assert f"sizing_json={sizing_file}" in script
    assert f'"sizing_json": "{sizing_file}"' in script


def test_slurm_batch_submits_only_pending_frequency_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0, 20.0, 30.0])
    job.save()
    for task in (1, 3):
        file = job.expected_trace_files()[task - 1]
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"task {task}")
    job.write_run_state(
        status="partial",
        tasks=[
            {"task": 1, "status": "success", "fingerprint": job.task_fingerprint(1)},
            {"task": 2, "status": "timeout", "fingerprint": job.task_fingerprint(2)},
            {"task": 3, "status": "success", "fingerprint": job.task_fingerprint(3)},
        ],
    )
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
    monkeypatch.setattr(site, "_submit_sbatch", lambda cmd: "89")
    monkeypatch.setattr(
        site, "_store_remote_run_records", lambda job, record=None: None
    )

    site._submit_slurm_batch(job, SlurmRunConfig(queue="debug"))

    assert seen["script_kwargs"]["n_tasks"] == 1
    assert seen["script_kwargs"]["n_job_tasks"] == 3
    assert seen["script_kwargs"]["task_indices"] == [2]
    assert seen["script_kwargs"]["skip_sizing"] is True


def test_adaptive_slurm_script_renders_pending_task_indices(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    script = site._sweep_SLURM_script(
        n_tasks=2,
        n_job_tasks=5,
        task_indices=[2, 5],
        n_nodes=1,
        stdout="/scratch/user/jobs/simple/freq/logs",
        duration="00-00:10:00",
    )

    assert "n_tasks=2" in script
    assert "n_job_tasks=5" in script
    assert '"task_indices": [' in script
    assert "    2," in script
    assert "    5" in script
    assert '"skip_sizing": false' in script
    assert "--job $job_file" in script
    assert "adaptive_scheduler.py" in script
    assert '--validate-sizing "$sizing_json" "$n_job_tasks"' in script
    assert "memory_bytes" not in script


def test_slurm_scripts_use_profile_runtime_setup(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite(
        "project/run",
        work_dir="/scratch/user",
        modules=["compiler suite", "mpi/5"],
        environment={"OMP_NUM_THREADS": "3", "VALUE": "two words"},
    )

    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        stdout="/scratch/user/jobs/simple/freq/logs",
        duration="00-00:10:00",
    )

    assert "module load 'compiler suite'" in script
    assert "module load mpi/5" in script
    assert "export MKL_DYNAMIC=FALSE" in script
    assert "export MKL_NUM_THREADS=1" in script
    assert "export OMP_WAIT_POLICY=PASSIVE" in script
    assert "export KMP_STACKSIZE=20M" in script
    assert "export OMP_NUM_THREADS=3" in script
    assert "export OMP_NUM_THREADS=$n_threads" in script
    assert '"omp_threads": 1' in script
    assert "export VALUE='two words'" in script
    assert "intel/25.1" not in script


def test_slurm_runtime_setup_expands_profile_environment_after_modules(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite(
        "project/run",
        modules=["dependency/1", "parallel-hdf5/2"],
        environment={
            "LD_LIBRARY_PATH": "${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}",
        },
    )

    lines = site._runtime_setup_lines()

    assert lines.index("module load dependency/1") < lines.index(
        "module load parallel-hdf5/2"
    )
    assert lines.index("module load parallel-hdf5/2") < lines.index("module list")
    assert lines.index("module list") < lines.index(
        'export LD_LIBRARY_PATH="${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}"'
    )


def test_slurm_runtime_environment_expands_only_simple_braced_references():
    value = "${SAFE}:$(unsafe):`unsafe`:$HOME"

    quoted = hpc._quote_runtime_environment_value(value)

    assert quoted == '"${SAFE}:\\$(unsafe):\\`unsafe\\`:\\$HOME"'


def test_slurm_batch_mpi_async_progress_reserves_one_core_per_rank(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    spr = Stampede3Config("spr")
    monkeypatch.setattr(site, "config_for_partition", lambda queue: spr)

    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_nodes=1,
        stdout="/scratch/user/jobs/simple/freq/logs",
        duration="00-00:10:00",
        queue="spr",
        ranks_per_node=8,
        mpi_async_progress=True,
    )

    assert "n_threads=13" in script
    assert "export OMP_NUM_THREADS=$n_threads" in script
    assert "export OMP_PLACES=cores" in script
    assert "export OMP_PROC_BIND=close" in script
    assert "export I_MPI_ASYNC_PROGRESS=1" in script
    assert "export I_MPI_ASYNC_PROGRESS_THREADS=1" in script
    assert "export I_MPI_ASYNC_PROGRESS_PIN=13,27,41,55,69,83,97,111" in script
    assert '"omp_threads": 13' in script
    assert '-nthreads "$n_threads"' in script


def test_slurm_attached_mpi_async_progress_uses_per_node_topology(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    site.pool.nhost = 2
    site.pool.nproc = 16
    site.pool.ncore = 224

    script = site._sweep_script(DummyJob(), mpi_async_progress=True)

    assert "n_threads=13" in script
    assert "export I_MPI_ASYNC_PROGRESS_PIN=13,27,41,55,69,83,97,111" in script
    assert "$executable -nthreads $n_threads --job $input_file" in script


def test_slurm_mpi_async_progress_validates_rank_core_layout():
    with pytest.raises(ValueError, match="evenly divisible"):
        SlurmSite._mpi_async_progress_layout(
            cores_per_node=10,
            ranks_per_node=3,
        )

    with pytest.raises(ValueError, match="at least two cores"):
        SlurmSite._mpi_async_progress_layout(
            cores_per_node=8,
            ranks_per_node=8,
        )


def test_adaptive_slurm_script_skips_sizing_for_single_task(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    script = site._sweep_SLURM_script(
        n_tasks=1,
        n_job_tasks=5,
        task_indices=[4],
        n_nodes=2,
        ranks_per_node=4,
        stdout="/scratch/user/jobs/simple/freq/logs",
        duration="00-00:10:00",
    )

    assert '"skip_sizing": true' in script
    assert "--init-no-size" in script
    assert '--init-no-size > "$dir_out/init.log" 2>&1' in script
    assert "--map" not in script
    assert '"total_ranks": 8' in script
    assert '"task_indices": [' in script
    assert "    4" in script


def test_adaptive_slurm_script_can_run_imaging_smooth_only(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    script = site._sweep_SLURM_script(
        n_tasks=0,
        n_job_tasks=1,
        task_indices=[],
        n_nodes=1,
        stdout="/scratch/user/jobs/simple/rtm/logs",
        duration="00-00:10:00",
        imaging_job=True,
        smooth_only=True,
    )

    assert "Skipping frequency sweep; running imaging postprocess only." in script
    assert (
        '$mpi_exec -n "$n_procs" "$executable" -nthreads "$n_threads" '
        '--job "$job_file" $fresh_flag --smooth >> "$dir_out/smooth.log" 2>&1'
    ) in script
    assert "--init" not in script
    assert '"--task", str(task_id)' not in script


def test_slurm_submit_ignores_local_current_without_remote_record(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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


def test_slurm_run_record_recreates_factory_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    site._site_config_path = tmp_path / "site.toml"
    site._site_profile = "cluster"
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    monkeypatch.setattr(site, "_store_remote_run_records", lambda job, record: None)

    record = site._record_site_run(job, scheduler_id="77")

    assert record.metadata == {
        "site_config_path": str(tmp_path / "site.toml"),
        "site_profile": "cluster",
    }
    recreated = object()
    from frequensolve.orchestrator.sites import config_file

    monkeypatch.setattr(
        config_file,
        "Site",
        lambda **kwargs: (recreated, kwargs),
    )
    resolved, kwargs = job._site_from_run_record(record)
    assert resolved is recreated
    assert kwargs == {
        "config_path": str(tmp_path / "site.toml"),
        "profile": "cluster",
    }


def test_slurm_run_record_recreates_direct_site_with_configured_work_dir(
    monkeypatch, tmp_path
):
    import frequensolve.simulation.jobs.records as job_records

    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    record = job.record_site_run(
        site="Recorded",
        work_dir="/work/user/frequensolve",
        scheduler_id="77",
        site_module="test.recorded_site",
        site_class="RecordedSite",
    )
    seen = {}

    class RecordedSite:
        def __init__(self, *, work_dir):
            seen["work_dir"] = work_dir

    module = type("RecordedSiteModule", (), {"RecordedSite": RecordedSite})
    monkeypatch.setattr(job_records.importlib, "import_module", lambda name: module)

    resolved = job._site_from_run_record(record)

    assert isinstance(resolved, RecordedSite)
    assert seen["work_dir"] == Path("/work/user/frequensolve")


def test_slurm_submit_reattach_compares_numpy_payload_values(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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
            {
                "execution": {
                    "mpi": {"ranks": 64},
                    "openmp": {"threads": 7},
                },
                "task_summary": {
                    "total": 1,
                    "complete": 1,
                    "succeeded": 1,
                    "failed": 0,
                },
            }
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


def test_slurm_fetch_logs_uses_per_job_batch_dir_with_legacy_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job._job_id = "88"
    calls = []
    remote_logs = site.work_dir / "jobs" / "simple" / "freq" / "logs"

    def fake_get(remote, local, overwrite=False):
        remote = Path(remote)
        local = Path(local)
        calls.append((remote, local))
        if remote.parent == remote_logs / "batch":
            raise FileNotFoundError(remote)

    monkeypatch.setattr(site, "get", fake_get)
    monkeypatch.setattr(site, "fetch_run_metadata", lambda job: None)

    site.fetch_logs(job, include_batch=True)

    local_batch = job._local_path / "logs" / "batch"
    for suffix in (".o", ".e"):
        filename = f"job_88{suffix}"
        assert (remote_logs / "batch" / filename, local_batch / filename) in calls
        assert (
            site.work_dir / "jobs" / "batch" / filename,
            local_batch / filename,
        ) in calls


def test_slurm_fetch_wavefields_downloads_wavefield_output(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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


def test_slurm_fetch_vtk_downloads_configured_output_path(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.vtk("qc", path="paraview/qc", fields="pressure")
    job.save()
    calls = []

    def fake_get(remote, local, overwrite=False):
        calls.append((Path(remote), Path(local)))
        Path(local).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(site, "get", fake_get)
    monkeypatch.setattr(job, "collect_task_run_manifests", lambda: None)

    assert site.fetch_vtk(job) == {"qc": job._result_path / "paraview/qc"}
    assert (
        site.work_dir / "jobs" / "simple" / "freq" / "results" / "paraview/qc",
        job._result_path / "paraview/qc",
    ) in calls


def test_slurm_batch_poll_reads_scheduler_status(monkeypatch, capsys):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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


def test_slurm_batch_poll_fails_when_scheduler_heartbeat_stops(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    payload = {
        "state": "running",
        "updated_at": "2026-07-18T07:00:00Z",
        "total": 4,
        "successful": 1,
        "failed": 0,
        "running": 2,
        "pending": 1,
    }

    monkeypatch.setattr(site, "update_status", lambda job_id: "running")
    monkeypatch.setattr(site, "_read_scheduler_status", lambda run: payload)
    clock = iter([100.0, 111.0])
    monkeypatch.setattr(hpc.time, "monotonic", lambda: next(clock))

    run = RunHandle(site=site, job=DummyJob(), id="77", mode="batch")
    run.backend["scheduler_heartbeat_timeout"] = 10.0

    assert site._poll_run(run).state == "running"
    status = site._poll_run(run)

    assert status.state == "failed"
    assert status.return_code == 1
    assert "heartbeat stopped advancing" in status.message
    assert status.raw["scheduler_liveness"] == {
        "state": "running",
        "heartbeat": "2026-07-18T07:00:00Z",
        "age_seconds": 11.0,
        "timeout_seconds": 10.0,
        "stale": True,
    }


def test_slurm_batch_poll_accepts_advancing_scheduler_heartbeat(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    heartbeats = iter(
        [
            "2026-07-18T07:00:00Z",
            "2026-07-18T07:00:01Z",
        ]
    )

    monkeypatch.setattr(site, "update_status", lambda job_id: "running")
    monkeypatch.setattr(
        site,
        "_read_scheduler_status",
        lambda run: {
            "state": "running",
            "updated_at": next(heartbeats),
            "total": 1,
            "successful": 0,
            "failed": 0,
            "running": 1,
            "pending": 0,
        },
    )
    clock = iter([100.0, 200.0])
    monkeypatch.setattr(hpc.time, "monotonic", lambda: next(clock))

    run = RunHandle(site=site, job=DummyJob(), id="77", mode="batch")
    run.backend["scheduler_heartbeat_timeout"] = 10.0

    assert site._poll_run(run).state == "running"
    assert site._poll_run(run).state == "running"


def test_slurm_batch_poll_stops_when_adaptive_scheduler_reports_failure(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    monkeypatch.setattr(site, "update_status", lambda job_id: "complete")
    monkeypatch.setattr(
        site,
        "_read_scheduler_status",
        lambda run: {
            "state": "failed",
            "abort_reason": "scheduler process exited",
            "total": 4,
            "successful": 1,
            "failed": 1,
            "running": 0,
            "pending": 2,
        },
    )

    run = RunHandle(site=site, job=DummyJob(), id="77", mode="batch")
    status = site._poll_run(run)

    assert status.state == "failed"
    assert status.return_code == 1
    assert "scheduler process exited" in status.message
    assert status.raw["scheduler_liveness"] == {
        "state": "failed",
        "stale": False,
    }


def test_slurm_batch_poll_does_not_heartbeat_timeout_postprocessing(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    monkeypatch.setattr(site, "update_status", lambda job_id: "running")
    monkeypatch.setattr(
        site,
        "_read_scheduler_status",
        lambda run: {
            "state": "complete",
            "updated_at": "2026-07-18T07:00:00Z",
            "total": 1,
            "successful": 1,
            "failed": 0,
            "running": 0,
            "pending": 0,
        },
    )

    run = RunHandle(site=site, job=DummyJob(), id="77", mode="batch")
    run.backend.update(
        {
            "scheduler_heartbeat_timeout": 10.0,
            "_adaptive_scheduler_heartbeat": "2026-07-18T07:00:00Z",
            "_adaptive_scheduler_heartbeat_seen_at": 0.0,
        }
    )

    status = site._poll_run(run)

    assert status.state == "running"
    assert "scheduler_liveness" not in status.raw


def test_slurm_batch_poll_includes_already_current_tasks(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    job = DummyJob()
    job.n_tasks = 90

    monkeypatch.setattr(site, "update_status", lambda job_id: "running")
    monkeypatch.setattr(
        site,
        "_read_scheduler_status",
        lambda run: {
            "successful": 1,
            "failed": 0,
            "running": 50,
            "pending": 20,
            "total": 71,
        },
    )

    run = RunHandle(site=site, job=job, id="77", mode="batch")
    run.backend["task_plan"] = {
        "current_tasks": list(range(1, 20)),
        "pending_indices": list(range(19, 90)),
        "reused_tasks": [],
    }

    status = site._poll_run(run)
    panel = status_table_html([run], {0: status})

    assert status.message == (
        "tasks: 20 successful, 0 failed, 50 running, 20 pending, 90 total"
    )
    assert status.raw["task_status"] == {
        "successful": 20,
        "succeeded": 20,
        "failed": 0,
        "running": 50,
        "pending": 20,
        "total": 90,
        "current": 19,
        "submitted_total": 71,
        "includes_current_tasks": True,
    }
    assert ">20</td>" in panel
    assert ">90</td>" in panel


def test_slurm_batch_poll_ignores_scheduler_status_while_pending(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")

    monkeypatch.setattr(site, "update_status", lambda job_id: "pending")
    monkeypatch.setattr(
        site,
        "run_login",
        lambda cmd: pytest.fail("pending batch polls should not read task counts"),
    )

    run = RunHandle(site=site, job=DummyJob(), id="77", mode="batch")
    status = site._poll_run(run)

    assert status.state == "pending"
    assert status.message == ""
    assert "task_status" not in status.raw


def test_slurm_sweep_scripts_run_solver_pack_after_tasks(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(hpc, "ImagingJob", DummyJob)
    site = DummySlurmSite("project/run")
    site.pool.nproc = 4
    site.pool.ncore = 8

    batch_script = site._sweep_SLURM_script(
        n_tasks=4,
        n_nodes=1,
        stdout="/scratch/user/project/run/jobs/dummy/logs",
        duration="00-00:30:00",
        ranks_per_node=4,
        tolerate_failures=2,
        imaging_job=True,
    )
    fresh_batch_script = site._sweep_SLURM_script(
        n_tasks=4,
        n_nodes=1,
        stdout="/scratch/user/project/run/jobs/dummy/logs",
        duration="00-00:30:00",
        ranks_per_node=4,
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
        ranks_per_node=4,
        pack=False,
    )

    assert '$fresh_flag --pack >> "$dir_out/pack.log" 2>&1' in batch_script
    assert "scheduler_status.json" in batch_script
    assert "scheduler_config.json" in batch_script
    assert '"failure_tolerance": 2' in batch_script
    assert "FS_SCHEDULER_STATUS" not in batch_script
    assert '"successful"' in batch_script
    assert '"pending"' in batch_script
    assert "$fresh_flag --pack >> $dir_out/pack.log 2>&1" in attached_script
    assert 'fresh_flag="--fresh"' in fresh_batch_script
    assert 'fresh_flag="--fresh"' in fresh_attached_script
    assert "export OMP_NUM_THREADS=$n_threads" in batch_script
    assert "export OMP_NUM_THREADS=$n_threads" in attached_script
    assert "I_MPI_ASYNC_PROGRESS" not in batch_script
    assert "I_MPI_ASYNC_PROGRESS" not in attached_script
    assert "-nthreads $n_threads" in batch_script
    assert "-nthreads $n_threads" in attached_script
    assert (
        '$mpi_exec -n "$n_procs" "$executable" -nthreads "$n_threads" '
        '--job "$job_file" $fresh_flag --smooth'
    ) in batch_script
    assert (
        "$mpi_exec -n $n_procs $executable -nthreads $n_threads "
        "--job $input_file $fresh_flag --smooth"
    ) in attached_script
    assert '--init > "$dir_out/init.log" 2>&1' in batch_script
    assert "--init > $dir_out/init.log 2>&1" in attached_script
    assert "--map" not in batch_script
    assert "--map" not in attached_script
    assert "--pack" not in attached_disabled_script
    assert "--pack" not in disabled_script


def test_slurm_submit_attached_requires_active_allocation(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: False))
    site = DummySlurmSite("project/run")

    with pytest.raises(RuntimeError, match="No active compute allocation"):
        site.submit(DummyJob(), mode="attached")


def test_slurm_submit_auto_uses_attached_when_provisioned(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")
    site.pool.id = "42"

    class DummyFuture:
        def done(self):
            return False

    monkeypatch.setattr(
        site, "_submit_attached", lambda job, ranks_per_task=2, **kwargs: DummyFuture()
    )

    run = site.submit(DummyJob())

    assert run.mode == "attached"
    assert run.status().state == "running"


def test_fresh_attached_submit_without_allocation_id_fails_before_launch(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")
    launched = []
    monkeypatch.setattr(
        site,
        "_submit_attached",
        lambda job, ranks_per_task=2, **kwargs: launched.append(job),
    )

    with pytest.raises(
        RuntimeError,
        match="^Active SLURM allocation is missing a valid scheduler job id$",
    ):
        site.submit(DummyJob(), mode="attached")

    assert launched == []


def test_slurm_submit_attached_can_disable_pack(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")
    site.pool.id = "42"
    seen = {}

    class DummyFuture:
        def done(self):
            return False

    def fake_submit(
        job,
        ranks_per_task=2,
        *,
        pack=True,
        fresh=False,
        mpi_async_progress=False,
    ):
        seen["pack"] = pack
        seen["fresh"] = fresh
        seen["mpi_async_progress"] = mpi_async_progress
        return DummyFuture()

    monkeypatch.setattr(site, "_submit_attached", fake_submit)

    site.submit(DummyJob(), pack=False)

    assert seen["pack"] is False
    assert seen["fresh"] is False
    assert seen["mpi_async_progress"] is False


def test_slurm_submit_rejects_mpi_async_progress(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")
    site.pool.id = "42"

    def fail_submit(*args, **kwargs):
        raise AssertionError("unsupported configuration reached submission")

    monkeypatch.setattr(site, "_submit_attached", fail_submit)

    with pytest.raises(NotImplementedError, match="race during MPI initialization"):
        site.submit(DummyJob(), mpi_async_progress=True)


def test_slurm_allocation_attach_returns_awaitable_handle(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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
    site = DummySlurmSite("project/run")

    site.close()

    assert site._login_client is None
    assert site._compute_client is None


def test_slurm_provision_returns_allocation_handle(monkeypatch, tmp_path):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    site = DummySlurmSite("project/run")
    staging_directory = tmp_path / "local-staging"
    config_path = tmp_path / "site.toml"
    config_path.write_text(f'[host]\ntmp_dir = "{staging_directory.as_posix()}"\n')
    site._site_config_path = config_path
    uploaded_scripts = []

    monkeypatch.setattr(
        site,
        "put",
        lambda local, remote: uploaded_scripts.append((Path(local), Path(remote))),
    )
    monkeypatch.setattr(site, "_submit_sbatch", lambda cmd: "44")
    monkeypatch.setattr(
        site, "_generate_provision_script", lambda *args, **kwargs: "script"
    )

    allocation = site.provision(nodes=1, tasks=1, duration="00-00:10:00")

    assert isinstance(allocation, RunHandle)
    assert allocation.id == "44"
    assert allocation.mode == "allocation"
    assert uploaded_scripts[0][0].parent == staging_directory
    assert uploaded_scripts[0][1].parent == site.remote_tmp_dir


def test_slurm_attached_run_is_awaitable(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    monkeypatch.setattr(DummySlurmSite, "provisioned", property(lambda self: True))
    site = DummySlurmSite("project/run")
    site.pool.id = "42"

    async def submit_and_wait():
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        monkeypatch.setattr(
            site, "_submit_attached", lambda job, ranks_per_task=2, **kwargs: future
        )
        run = site.submit(DummyJob())
        return await run

    result = asyncio.run(submit_and_wait())

    assert result.successful
    assert result.status.state == "completed"


def test_slurm_attached_sync_wait_inside_event_loop_has_clear_error(monkeypatch):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
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


def test_slurm_site_config_resolves_partition_shapes_and_rejects_unknown():
    config = SlurmSiteConfig(
        hostname="login.example.edu",
        queue="debug",
        partitions={
            "debug": SlurmPartitionConfig(
                max_duration="00-00:30:00",
                max_nodes=2,
                cores_per_node=8,
                memory_per_node=32768,
            ),
            "normal": {
                "max_duration": "00-08:00:00",
                "max_nodes": 16,
                "cores_per_node": 64,
                "memory_per_node": 262144,
            },
        },
    )

    assert config.default_partition == "debug"
    assert config.cores_per_node == 8
    normal = config.for_partition("normal")
    assert normal.queue == "normal"
    assert normal.max_nodes == 16
    assert normal.cores_per_node == 64
    assert normal.memory_per_node == 262144
    with pytest.raises(ValueError, match="Unknown SLURM partition.*missing"):
        config.for_partition("missing")


def test_pool_status_recognizes_slurm_terminal_states():
    for status in ("complete", "failed", "timeout", "cancelled"):
        assert PoolStatus(status=status).is_complete


def test_stampede3_site_is_specific_slurm_subclass():
    assert issubclass(Stampede3Site, SlurmSite)
    assert Stampede3Site.config_cls is Stampede3Config
    assert Stampede3Site.credentials_cls is TACCLoginCredentials
    assert Stampede3Site.default_solver_executable is None
    config = Stampede3Config("icx")
    assert config.hostname == "stampede3.tacc.utexas.edu"
    assert config.queue == "icx"
    assert config.cores_per_node == 80
    assert config.cores_per_socket == 40
    assert config.memory_per_node == 262144
    assert config.memory_per_core == 3276.8
    assert config.for_partition("spr").cores_per_node == 112
