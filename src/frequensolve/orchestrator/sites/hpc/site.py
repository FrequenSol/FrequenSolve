"""Generic SSH/SLURM HPC site support.

This module contains the reusable mechanics for SLURM-backed remote sites.
Site-specific modules should provide credentials, queue configuration, and
default paths by subclassing :class:`SlurmSite`.
"""

import asyncio
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from asyncio import Future
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from select import select
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, Union

from frequensolve._optional import optional_dependency_error

try:
    from dotenv import load_dotenv
    from paramiko import SSHClient
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "SlurmSite",
        extra="hpc",
        dependencies=("paramiko", "python-dotenv"),
        error=exc,
    ) from exc

from jinja2 import Environment, FileSystemLoader

from frequensolve.orchestrator.credentials import Credentials
from frequensolve.orchestrator.pool import PoolInfo
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunHandle,
    RunResult,
    _check_if_notebook,
)
from frequensolve.orchestrator.sites.config import BaseSiteConfig
from frequensolve.orchestrator.sites.hpc.auth import SlurmAuthenticator
from frequensolve.orchestrator.sites.hpc.slurm_helpers import as_list as _as_list
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    hms_to_seconds as _hms_to_seconds,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    normalize_slurm_state as _normalize_slurm_state,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    parse_sbatch_job_id as _parse_sbatch_job_id,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    read_stream as _read_stream,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    seconds_to_hms as _seconds_to_hms,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    temporary_text_file as _temporary_text_file,
)
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager
from frequensolve.orchestrator.ssh import SSHClientClass
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.imaging import ImageDatabase, ImagingJob
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

__all__ = [
    "SlurmSiteConfig",
    "SlurmLoginCredentials",
    "SlurmRunConfig",
    "SlurmSite",
]

# Initialize the logger
logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")


# ----------------------------------
# Generic SLURM Config
# ----------------------------------
@dataclass
class SlurmSiteConfig(BaseSiteConfig):
    """Minimal reusable configuration for a SLURM-backed site.

    Site-specific config classes can expose the same attributes and do not need
    to inherit from this class.
    """

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

    def validate_request(self, nodes: int, tasks: int, duration: Optional[str]) -> str:
        """Validate a SLURM allocation request and return a usable duration."""

        duration = duration or self.max_duration
        if nodes < self.min_nodes:
            raise ValueError(f"Minimum number of nodes is {self.min_nodes}")
        if nodes > self.max_nodes:
            raise ValueError(f"Maximum number of nodes is {self.max_nodes}")
        if tasks < nodes:
            raise ValueError("Number of tasks must be at least the number of nodes")
        if _hms_to_seconds(duration) > _hms_to_seconds(self.max_duration):
            logger.warning(
                "Requested duration %s exceeds maximum %s; using maximum",
                duration,
                self.max_duration,
            )
            return self.max_duration
        return duration


# ----------------------------------
# Generic Login Credentials
# ----------------------------------
class SlurmLoginCredentials(Credentials):
    """Generic SSH credentials for SLURM-backed HPC sites."""

    user_env: str = "HPC_USERNAME"
    pw_env: str = "HPC_PASSWORD"
    ssh_key_env: str = "SSH_PASSPHRASE"


@dataclass
class SlurmRunConfig:
    """Default resource request for SLURM job submissions."""

    queue: Optional[str] = None
    nodes: int = 1
    duration: Optional[str] = None
    procs_per_node: Optional[int] = None
    procs_per_task: Optional[int] = None
    account: Optional[str] = None
    notify_on: Optional[Literal["begin", "end", "fail", "all", "none"]] = None
    notify_email: Optional[str] = None
    poll_interval: Optional[int] = None
    run_path: Optional[Union[str, Path]] = None
    slurm_args: List[str] = field(default_factory=list)

    @classmethod
    def field_names(cls) -> set[str]:
        return {item.name for item in dataclass_fields(cls)}

    def merged(self, **overrides) -> "SlurmRunConfig":
        values = {
            "queue": self.queue,
            "nodes": self.nodes,
            "duration": self.duration,
            "procs_per_node": self.procs_per_node,
            "procs_per_task": self.procs_per_task,
            "account": self.account,
            "notify_on": self.notify_on,
            "notify_email": self.notify_email,
            "poll_interval": self.poll_interval,
            "run_path": self.run_path,
            "slurm_args": list(self.slurm_args),
        }
        for key, value in overrides.items():
            if key in values and value is not None:
                values[key] = value
        return SlurmRunConfig(**values)

    def resolved(
        self,
        site_config: Any,
        **overrides,
    ) -> tuple["SlurmRunConfig", Dict[str, Any]]:
        run_keys = self.field_names()
        config = self.merged(**{k: v for k, v in overrides.items() if k in run_keys})
        if config.queue is None:
            config.queue = site_config.queue
        if config.poll_interval is None:
            config.poll_interval = site_config.poll_interval
        extra = {k: v for k, v in overrides.items() if k not in run_keys}
        return config, extra


# ----------------------------------
# Generic SLURM Site
# ----------------------------------
@dataclass(kw_only=True, init=False)
class SlurmSite(BaseSite):
    """
    Generic SLURM HPC site.

    Manages authentication, transfer, provisioning, and job execution for SLURM-backed HPC systems.
    """

    credentials: SlurmLoginCredentials
    config: Any
    run_config: SlurmRunConfig
    pool: PoolInfo
    transfer_method: Literal["rsync", "sftp"] = "rsync"
    _executable: str
    _login_client: SSHClientClass
    _compute_client: Optional[SSHClientClass] = None
    _work_dir: Path
    _FS_dir: Path

    site_name: str = "SLURM"
    credentials_cls: Type["SlurmLoginCredentials"] = None
    config_cls: Optional[Type[Any]] = None
    default_queue: Optional[str] = None
    default_host: Optional[str] = None
    work_dir_env: str = "FS_HPC_WORK_DIR"
    solver_executable_env: str = "FS_SOLVER_EXECUTABLE"
    default_solver_executable: Optional[str] = None
    python_path_env: str = "FS_PYTHON_PATH"

    def __init__(
        self,
        rel_path: Union[str, Path],
        transfer_method: Literal["rsync", "sftp"] = "rsync",
        default_queue: Optional[str] = None,
        config: Optional[Any] = None,
        credentials: Optional["SlurmLoginCredentials"] = None,
        run_config: Optional[SlurmRunConfig] = None,
        verbose: bool = False,
    ):
        queue = default_queue if default_queue is not None else self.default_queue
        self.verbose = verbose
        logger.debug(
            "Initializing %s with rel_path=%s, queue=%s",
            self.site_name,
            rel_path,
            queue,
        )

        if self.credentials_cls is None:
            self.credentials_cls = SlurmLoginCredentials
        self.credentials = credentials or self.credentials_cls()
        if config is not None:
            self.config = config
        elif self.config_cls is not None:
            self.config = (
                self.config_cls(queue=queue) if queue is not None else self.config_cls()
            )
        else:
            raise ValueError("SlurmSite requires either a config object or config_cls")
        self.transfer_method = transfer_method
        self.run_config = run_config or SlurmRunConfig(queue=queue)
        self._rel_proj_path = Path(rel_path)
        self._authenticator = SlurmAuthenticator(self)
        self._transfer = SlurmTransferManager(self)

        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH client authenticated successfully")

        self._work_dir = self._get_work_dir(self._rel_proj_path)
        self._executable = self._get_solver_path()
        self._FS_dir = self._get_FS_path()

        self.pool = PoolInfo()
        self._is_notebook = _check_if_notebook()
        self._compute_client = None

        self._emit(f"{self.site_name} initialized with work_dir: {self._work_dir}")

    @property
    def executable(self) -> str:
        """Get the solver executable."""

        if self._executable is None:
            raise ValueError(
                "Solver executable not specified; set "
                f"{self.solver_executable_env} or override default_solver_executable."
            )
        return self._executable

    @property
    def compute_client(self) -> SSHClient:
        """Get the compute client."""
        if self._compute_client is None:
            raise RuntimeError(
                "No compute client is attached; call provision/attach first"
            )
        return self._compute_client.client

    @property
    def compute_host(self) -> str:
        """Get the compute host."""
        if self._compute_client is None:
            raise RuntimeError(
                "No compute client is attached; call provision/attach first"
            )
        return self._compute_client.hostname

    @property
    def login_client(self) -> SSHClient:
        """Get the login client."""
        return self._login_client.client

    @property
    def login_host(self) -> str:
        """Get the login host."""
        return self._login_client.hostname

    @property
    def mpi_cmd(self) -> str:
        """Get the MPI launch command."""
        return f"{self.config.mpi_wrapper}"

    @property
    def pool_host(self) -> str:
        """Get the resource pool host node."""
        return self._authenticator.get_job_host(self.pool.id)

    @property
    def work_dir(self) -> Path:
        """Gets the remote work directory path."""
        return self._work_dir

    @property
    def provisioned(self):
        """Check if the site is provisioned."""
        self.update_status()
        return self.pool.is_running

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self) -> None:
        """Close active SSH clients held by the site."""

        if getattr(self, "_compute_client", None):
            self._compute_client.close()
            self._compute_client = None
        if getattr(self, "_login_client", None):
            self._login_client.close()
            self._login_client = None

    def authenticate(self, host: Optional[str] = None):
        """Connect to the login node."""

        return self._authenticator.authenticate(host)

    def submit(
        self,
        job: SimulationJob,
        *,
        force: bool = False,
        force_run: bool = False,
        mode: Literal["auto", "attached", "batch"] = "auto",
        fetch: bool = False,
        **overrides,
    ) -> RunHandle:
        """Submit job and return an awaitable run handle."""

        force_run = bool(force_run or force or overrides.pop("rerun", False))
        self.prepare_job(job)
        if mode not in {"auto", "attached", "batch"}:
            raise ValueError("mode must be 'auto', 'attached', or 'batch'")

        run_config, extra_kwargs = self.run_config.resolved(self.config, **overrides)

        if not force_run:
            handle = self._reattach_inflight_run(
                job,
                poll_interval=run_config.poll_interval,
                fetch=fetch,
            )
            if handle is not None:
                return handle

            if self.is_run_current(job):
                job.write_run_state(status="skipped")
                self._emit(f"Skipping {job.name}; run is current")
                handle = RunHandle.skipped(self, job)
                if fetch:
                    self.fetch_outputs(job)
                return handle

        self.prepare_job(job, sync_project=True)

        active_allocation = self.provisioned if mode in {"auto", "attached"} else False
        use_attached = mode == "attached" or (mode == "auto" and active_allocation)
        if use_attached:
            if not active_allocation:
                raise RuntimeError(
                    "No active compute allocation is attached; use mode='batch' "
                    "or allow mode='auto' to submit a batch job."
                )
            pack = bool(extra_kwargs.pop("pack", True))
            future = self._submit_attached(
                job,
                procs_per_task=run_config.procs_per_task or 2,
                fresh=force_run,
                **({"pack": pack} if not pack else {}),
            )
            self._emit(f"Submitted {job.name} to active {self.site_name} allocation")
            self._record_site_run(job, status="running")
            handle = RunHandle(
                site=self,
                job=job,
                id=getattr(job, "_job_id", None),
                mode="attached",
                poll_interval=run_config.poll_interval,
                _status_fn=self._poll_attached_run,
                _wait_fn=self._wait_attached_run,
                _wait_async_fn=self._wait_attached_run_async,
                _generic_wait=False,
                _cancel_fn=lambda run: self.cancel_job(str(run.id)),
                _fetch_fn=(lambda run: self.fetch_outputs(run.job)) if fetch else None,
            )
            handle.backend["future"] = future
            return handle

        job_id = self._submit_slurm_batch(
            job, run_config, fresh=force_run, **extra_kwargs
        )
        handle = self.handle(job, job_id=job_id, mode="batch")
        handle.poll_interval = run_config.poll_interval or self.config.poll_interval
        handle._fetch_fn = (lambda run: self.fetch_outputs(run.job)) if fetch else None
        return handle

    def handle(
        self, job, job_id: Optional[str] = None, mode: str = "attached"
    ) -> RunHandle:
        """Create a run handle and attach SLURM-specific wait behavior."""

        return super().handle(job, job_id=job_id, mode=mode)

    def provision(
        self, nodes: int, tasks: int, duration: Optional[str] = None, **kwargs
    ) -> RunHandle:
        """Submit an interactive SLURM allocation and return a run handle.

        The returned handle completes when the allocation is running and the
        compute client is attached.
        """

        nhost = nodes
        nproc = tasks
        logger.info(
            "Provisioning SLURM job with nhost=%d, nproc=%d, duration=%s",
            nhost,
            nproc,
            duration,
        )
        duration = duration or getattr(self.config, "max_duration", "00-02:00:00")
        self._emit(
            f"Submitting {self.site_name} allocation: nodes={nhost}, "
            f"tasks={nproc}, duration={duration}"
        )
        duration = self.config.validate_request(nhost, nproc, duration)
        script = self._generate_provision_script(nhost, nproc, duration, **kwargs)

        with _temporary_text_file(script, suffix=".sh", prefix="slurm_") as script_path:
            logger.debug("Temporary SLURM script created at %s", script_path)
            remote_path = f"/tmp/{os.path.basename(script_path)}"
            try:
                self.put(script_path, remote_path)

                self.pool.id = self._submit_sbatch(f"sbatch {remote_path}")
                logger.debug("Job submitted successfully with job ID: %s", self.pool.id)
                self.pool._status.status = "pending"
                self._emit(f"Allocation submitted: {self.pool.id}")

            except Exception as e:
                logger.exception("Exception occurred during provisioning: %s", str(e))
                self.pool._status.status = "failed"
                self.pool._status.stderr = str(e)
                raise

        return self._allocation_handle(self.pool.id)

    def attach_allocation(self, job_id: Optional[str] = None) -> RunHandle:
        """Create a handle for an existing SLURM allocation.

        If job_id is not provided, queued jobs will be listed and the user will
        be prompted to select a job.
        """
        if job_id is None:
            job_id = self._select_job()
        self.pool.id = job_id
        self._emit(f"Tracking existing allocation: {self.pool.id}")
        return self._allocation_handle(str(job_id))

    def run_cmd(self, client, cmd: str):
        """Run a command using exec_command, passing the captured environment if available."""
        if client is None:
            raise RuntimeError("SSH client is not connected")
        env = getattr(client, "environ", None)
        logger.debug("Executing on %s: %s", client.hostname, cmd)
        return (
            client.client.exec_command(cmd, environment=env)
            if env
            else client.client.exec_command(cmd)
        )

    def run_compute_cmd(self, cmd: str):
        """Run a command on compute node using exec_command."""
        return self.run_cmd(self._compute_client, cmd)

    def run_login_cmd(self, cmd: str):
        """Run a command on login node using exec_command."""
        return self.run_cmd(self._login_client, cmd)

    def run_compute(self, cmd: str) -> str:
        """Run a command on compute node and return its stdout as a stripped string."""
        _, stdout, _ = self.run_compute_cmd(cmd)
        return stdout.read().decode().strip()

    def run_login(self, cmd: str) -> str:
        """Run a command on login node and return its stdout as a stripped string."""
        _, stdout, _ = self.run_login_cmd(cmd)
        return _read_stream(stdout)

    def is_run_current(self, job: SimulationJob) -> bool:
        """Return True when this site has current successful results for a job."""

        if not isinstance(job, SimulationJob):
            return bool(job.is_run_current())
        record = job.latest_run(site=self.site_name)
        if record is None:
            return False
        try:
            if record.fingerprint != job.fingerprint():
                return False
        except Exception as exc:
            logger.debug("Could not fingerprint job %s: %s", job.name, exc)
            return False
        return self._remote_run_successful(record)

    def update_status(self, job_id: Optional[str] = None):
        """Check the status of the resource request."""

        if job_id is None:
            job_id = self.pool.id
            job_specified = False  # Checking status of pool
        else:
            job_specified = True  # Checking status of a specific job

        if not job_id:
            self.pool._status.status = "unknown"
            return "unknown"

        # Get job status
        queue_status = self.run_login(f"squeue -j {job_id} -h -o %t").strip()
        logger.debug("Job %s status from squeue: '%s'", job_id, queue_status)

        # If no status returned, job is not in queue - check sacct for completion status
        if not queue_status:
            sacct_cmd = f"sacct -j {job_id} -n -o State%20"
            completion_status = self.run_login(sacct_cmd).strip()
            logger.debug(
                "Job %s completion status from sacct: '%s'", job_id, completion_status
            )
            status = "unknown"
            for line in completion_status.splitlines():
                status = _normalize_slurm_state(line)
                if status != "unknown":
                    break
        else:
            status = _normalize_slurm_state(queue_status)

        if not job_specified:
            self.pool._status.status = status
        return status

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """Transfer files from a local path to a remote path."""

        return self._transfer.put(local_path, remote_path)

    def fetch_traces(
        self,
        job: Union[SimulationJob, List[SimulationJob]],
        upscale: int = 1,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Get trace results from the remote site.

        Args:
            job: A SimulationJob object.
        """

        jobs, single = _as_list(job, SimulationJob)

        db_map = {}

        for j in jobs:
            try:
                trace_dir_name = Path(j.trace_outputs.path).name
                remote_dir = self._remote_result_dir(j) / trace_dir_name
                local_dir = j._local_path / "results" / trace_dir_name
                local_dir.mkdir(parents=True, exist_ok=True)
                self.get(remote_dir, local_dir)

                db = TraceDataset.from_job(j, upscale)
                db_map[j.name] = db

            except Exception as e:
                logger.exception("Error downloading traces: %s", str(e))
                raise

        if single:
            return db_map[jobs[0].name]
        else:
            return db_map

    def fetch_wavefields(
        self,
        job: Union[SimulationJob, List[SimulationJob]],
        upscale: int = 1,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Get wavefield trace results from the remote site."""

        jobs, single = _as_list(job, SimulationJob)
        db_map = {}

        for j in jobs:
            try:
                wavefield_outputs = j.wavefield_trace_outputs
                if not wavefield_outputs.groups:
                    raise ValueError("Job has no wavefield outputs")
                wavefield_dir_name = Path(wavefield_outputs.path).name
                remote_dir = self._remote_result_dir(j) / wavefield_dir_name
                local_dir = j._local_path / "results" / wavefield_dir_name
                local_dir.mkdir(parents=True, exist_ok=True)
                self.get(remote_dir, local_dir)

                db_map[j.name] = j.wavefields.open(upscale=upscale)

            except Exception as e:
                logger.exception("Error downloading wavefields: %s", str(e))
                raise

        if single:
            return db_map[jobs[0].name]
        return db_map

    def fetch_outputs(self, job: SimulationJob):
        """Fetch common result metadata and trace outputs for a completed job."""

        local_results = job._local_path / "results"
        local_results.mkdir(parents=True, exist_ok=True)

        self.fetch_run_metadata(job)
        try:
            self.get(
                self._remote_logs_dir(job),
                job._local_path / "logs",
            )
        except Exception as exc:
            logger.debug("Could not fetch logs for job %s: %s", job.name, exc)

        try:
            self.fetch_traces(job)
        except Exception as exc:
            logger.debug("Could not fetch traces for job %s: %s", job.name, exc)

        if getattr(job.outputs, "wavefields", None):
            try:
                self.fetch_wavefields(job)
            except Exception as exc:
                logger.debug(
                    "Could not fetch wavefields for job %s: %s",
                    job.name,
                    exc,
                )

        return local_results

    def fetch_run_metadata(self, job: SimulationJob) -> Optional[Path]:
        """Fetch ``_fs_run`` metadata and aggregate task manifests locally."""

        remote_run_dir = self._remote_result_dir(job) / "_fs_run"
        local_run_dir = job._result_path / "_fs_run"
        try:
            self.get(remote_run_dir, local_run_dir)
        except Exception as exc:
            logger.debug("Could not fetch _fs_run for job %s: %s", job.name, exc)
            return None
        return job.collect_task_run_manifests()

    def fetch_paraview(self, job: SimulationJob):
        """Get Paraview files from the remote site.

        Args:
            job: A SimulationJob object.
        """

        try:
            remote_dir = self._remote_result_dir(job) / "ParaView/"
            local_dir = job._local_path / "results" / "ParaView/"
            self._emit(f"Fetching ParaView outputs from {remote_dir} to {local_dir}")
            self.get(remote_dir, local_dir)

        except Exception as e:
            logger.exception("Error downloading ParaView outputs: %s", str(e))

    def fetch_image(
        self,
        job: Union[ImagingJob, List[ImagingJob]],
    ):
        """Get image files from the remote site.

        Args:
            job: An ImagingJob object.
        """

        jobs, single = _as_list(job, ImagingJob)
        images = {}
        for job in jobs:
            try:
                remote = job._remote_image_path(self.work_dir)
                local = job._local_image_path
                self.get(remote, local)

                images[job.name] = ImageDatabase(
                    path=local,
                    shape=job.grid.shape,
                    parts=job.n_tasks,
                )

            except Exception as e:
                logger.exception("Error retrieving payload: %s", str(e))
                raise

        if single:
            return images[jobs[0].name]
        else:
            return images

    def fetch_logs(
        self,
        job: Union[SimulationJob, List[SimulationJob]],
        *,
        local_dir: Optional[Union[str, Path]] = None,
        task: Optional[int] = None,
        frequency: Optional[Union[float, complex]] = None,
        include_batch: bool = False,
        show: bool = False,
    ) -> Union[Path, dict]:
        """Fetch log files from the remote site to the local machine.

        Downloads the task log directory (e.g. task_1.txt, task_2.txt, ...) from
        the remote job run. Optionally fetches SLURM batch stdout/stderr files
        (job_<id>.o, job_<id>.e) when include_batch is True.

        Args:
            job: A SimulationJob or list of SimulationJobs whose logs to fetch.
            local_dir: Optional local directory to write logs into. If None,
                logs are written to job._local_path / "logs" for each job.
            task: Optional one-based frequency task number. When provided,
                returns that task's log file instead of the log directory.
            frequency: Optional physical frequency. When provided, selects the
                matching frequency task and returns that task's log file.
            include_batch: If True, also fetch SLURM batch logs from
                run_path/jobs/batch/ (requires job._job_id and run_path).
            show: If True, print log contents after fetching. When
                include_batch is True, prints in order: batch .o file, batch
                .e file, then the selected or latest task log file. Otherwise
                prints only the selected or latest task log file.

        Returns:
            If a single job: Path to the local log directory or selected log
            file. If a list of jobs: dict mapping job name to Path.
        """
        jobs, single = _as_list(job, SimulationJob)
        requested_local_dir = Path(local_dir) if local_dir is not None else None
        result = {}

        for j in jobs:
            if requested_local_dir is None:
                log_dir = j._local_path / "logs"
            elif single:
                log_dir = requested_local_dir
            else:
                log_dir = requested_local_dir / j.name
            remote_run_dir = self._remote_result_dir(j) / "_fs_run"
            try:
                local_run_dir = j._result_path / "_fs_run"
                self._emit(
                    f"Fetching run metadata from {remote_run_dir} to {local_run_dir}"
                )
                self.fetch_run_metadata(j)
            except Exception as e:
                logger.debug(
                    "Could not aggregate _fs_run metadata for job %s from %s: %s",
                    j.name,
                    remote_run_dir,
                    e,
                )
            remote_log_dir = self._remote_logs_dir(j)
            try:
                self._emit(f"Fetching logs from {remote_log_dir} to {log_dir}")
                self.get(remote_log_dir, log_dir)
                selected_log = self._select_log_path(
                    j,
                    log_dir,
                    task=task,
                    frequency=frequency,
                )
                result[j.name] = selected_log

                if include_batch and getattr(j, "_job_id", None):
                    batch_remote = self.work_dir / "jobs" / "batch"
                    batch_local = log_dir.parent / "batch"
                    self._emit(
                        f"Fetching batch logs from {batch_remote} to {batch_local}"
                    )
                    for suffix in (".o", ".e"):
                        remote_batch = batch_remote / f"job_{j._job_id}{suffix}"
                        local_batch = batch_local / f"job_{j._job_id}{suffix}"
                        try:
                            self.get(remote_batch, local_batch)
                        except Exception as e:
                            logger.debug(
                                "Could not fetch batch log %s: %s",
                                remote_batch,
                                e,
                            )
                elif include_batch:
                    logger.warning(
                        "Job %s has no _job_id; skipping SLURM batch logs",
                        j.name,
                    )

                if show:
                    if include_batch:
                        self._print_batch_logs(
                            job_name=j.name,
                            log_dir=log_dir,
                            job_id=getattr(j, "_job_id", None),
                        )
                    self._show_logs(selected_log, job_name=j.name)
            except Exception as e:
                logger.exception("Error fetching logs for job %s: %s", j.name, e)
                raise
        if single:
            return result[jobs[0].name]
        return result

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Transfer files from a remote path to a local path."""

        return self._transfer.get(remote_path, local_path, overwrite=overwrite)

    def cancel_job(self, job_id: Optional[str] = None) -> bool:
        """Cancel a job."""
        if job_id is None:
            job_id = self._select_job()
        if not job_id:
            logger.debug("No SLURM job id supplied; skipping cancellation")
            return False
        _, stdout, _ = self.run_login_cmd(f"scancel {job_id}")
        logger.info("Job %s cancelled: %s", job_id, stdout.read().decode().strip())
        return True

    def deprovision(self, **kwargs):
        """Release HPC resources."""
        return self.cancel_job(self.pool.id)

    def sync(self, project):
        """Sync the project to the site."""
        self._sync_project(project)

    def config_for_queue(self, queue: str):
        """Return the site configuration for a queue/partition name."""
        if self.config_cls is not None:
            return self.config_cls(queue)
        return self.config

    def _reattach_inflight_run(
        self,
        job: SimulationJob,
        *,
        poll_interval: Optional[float] = None,
        fetch: bool = False,
    ) -> Optional[RunHandle]:
        """Return a handle for a matching active scheduler job, if one exists."""

        record_status = self._matching_inflight_run_record(job)
        if record_status is None:
            return None
        record, scheduler_status = record_status
        job._job_id = record.scheduler_id
        updated = record.with_updates(
            status=scheduler_status,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        job.write_run_record(updated)
        self._emit(
            f"Reattached {job.name} to running {self.site_name} job "
            f"{record.scheduler_id}"
        )
        handle = self.handle(job, job_id=record.scheduler_id, mode="batch")
        handle.poll_interval = poll_interval or self.config.poll_interval
        handle._fetch_fn = (lambda run: self.fetch_outputs(run.job)) if fetch else None
        handle.backend["reattached"] = True
        return handle

    def _matching_inflight_run_record(self, job: SimulationJob):
        if not isinstance(job, SimulationJob):
            return None
        try:
            current_payload = job.fingerprint_payload()
            current_fingerprint = job._hash_payload(current_payload)
        except Exception as exc:
            logger.debug("Could not fingerprint job %s: %s", job.name, exc)
            return None

        records = [
            record
            for record in job.run_records()
            if record.site == self.site_name and record.scheduler_id is not None
        ]
        records = sorted(
            records,
            key=lambda record: record.updated_at or record.submitted_at or "",
            reverse=True,
        )
        for record in records:
            if not self._run_record_matches_fingerprint(
                record,
                fingerprint=current_fingerprint,
                fingerprint_payload=current_payload,
            ):
                continue
            scheduler_status = self.update_status(record.scheduler_id)
            if scheduler_status in {"pending", "running"}:
                return record, scheduler_status
        return None

    @staticmethod
    def _run_record_matches_fingerprint(
        record,
        *,
        fingerprint: str,
        fingerprint_payload: Dict[str, Any],
    ) -> bool:
        if record.fingerprint != fingerprint:
            return False
        record_payload = record.fingerprint_payload or {}
        if not record_payload:
            return True
        return SlurmSite._fingerprint_section_matches(
            record_payload, fingerprint_payload, "simulation"
        ) and SlurmSite._fingerprint_section_matches(
            record_payload, fingerprint_payload, "job"
        )

    @staticmethod
    def _fingerprint_section_matches(
        record_payload: Dict[str, Any],
        fingerprint_payload: Dict[str, Any],
        section: str,
    ) -> bool:
        if section not in record_payload or section not in fingerprint_payload:
            return False
        return SimulationJob._hash_payload(
            record_payload[section]
        ) == SimulationJob._hash_payload(fingerprint_payload[section])

    def _remote_run_successful(self, record) -> bool:
        if not self._record_status_successful(record.status):
            return False

        manifest = self._read_remote_json(
            record.result_dir / "_fs_run" / "run_manifest.json"
        )
        if not isinstance(manifest, dict):
            return False

        task_summary = manifest.get("task_summary")
        if not isinstance(task_summary, dict):
            return False
        try:
            failed = int(task_summary.get("failed") or 0)
            complete = int(task_summary.get("complete") or 0)
            total = int(task_summary.get("total") or 0)
        except (TypeError, ValueError):
            return False
        expected_total = self._record_expected_task_count(record)
        if expected_total is not None and total != expected_total:
            return False
        if failed != 0 or total <= 0 or complete != total:
            return False
        succeeded = task_summary.get("succeeded", task_summary.get("successful"))
        if succeeded is not None:
            try:
                if int(succeeded) != total:
                    return False
            except (TypeError, ValueError):
                return False
        return self._remote_path_has_files(record.logs_dir)

    @staticmethod
    def _record_status_successful(status: Optional[str]) -> bool:
        return str(status or "").lower() in {
            "complete",
            "completed",
            "success",
            "successful",
            "succeeded",
        }

    @staticmethod
    def _record_expected_task_count(record) -> Optional[int]:
        payload = record.fingerprint_payload or {}
        job_payload = payload.get("job") if isinstance(payload, Mapping) else None
        if not isinstance(job_payload, Mapping):
            return None
        frequencies = job_payload.get("f_list")
        if frequencies is None:
            return None
        try:
            return len(frequencies)
        except TypeError:
            return None

    def _remote_path_has_files(self, path: Union[str, Path]) -> bool:
        quoted = shlex.quote(str(path))
        try:
            text = self.run_login(
                f"find {quoted} -type f -print -quit 2>/dev/null || true"
            ).strip()
        except Exception as exc:
            logger.debug("Could not inspect remote path %s: %s", path, exc)
            return False
        return bool(text)

    def _read_remote_json(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        quoted = shlex.quote(str(path))
        try:
            text = self.run_login(f"test -f {quoted} && cat {quoted} || true").strip()
        except Exception as exc:
            logger.debug("Could not read remote JSON %s: %s", path, exc)
            return None
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("Could not parse remote JSON %s: %s", path, exc)
            return None
        return payload if isinstance(payload, dict) else None

    def _record_site_run(
        self,
        job: SimulationJob,
        *,
        scheduler_id: Optional[str] = None,
        status: str = "submitted",
    ):
        if not isinstance(job, SimulationJob):
            return None
        record = job.record_site_run(
            site=self.site_name,
            work_dir=self.work_dir,
            scheduler_id=scheduler_id,
            status=status,
            site_module=self.__class__.__module__,
            site_class=self.__class__.__name__,
            rel_path=self._rel_proj_path,
        )
        self._store_remote_run_records(job, record)
        return record

    def _store_remote_run_records(self, job: SimulationJob, record=None) -> None:
        record = record or job.latest_run(site=self.site_name)
        if record is None:
            return
        try:
            self.put(job.run_records_file, record.result_dir / "_fs_run" / "runs.json")
        except Exception as exc:
            logger.debug(
                "Could not write remote run record for job %s: %s",
                job.name,
                exc,
            )

    def _finalize_run_record(self, run: RunHandle, status: JobStatus) -> None:
        job = getattr(run, "job", None)
        if not isinstance(job, SimulationJob):
            return
        record = job.latest_run(site=self.site_name)
        if record is None:
            return
        updated = record.with_updates(
            status=status.state,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        job.write_run_record(updated)
        self._store_remote_run_records(job, updated)

    def _remote_job_dir(self, job: SimulationJob) -> Path:
        record = (
            job.latest_run(site=self.site_name)
            if isinstance(job, SimulationJob)
            else None
        )
        return record.job_dir if record is not None else job._remote_path(self.work_dir)

    def _remote_result_dir(self, job: SimulationJob) -> Path:
        record = (
            job.latest_run(site=self.site_name)
            if isinstance(job, SimulationJob)
            else None
        )
        if record is not None:
            return record.result_dir
        return job._remote_path(self.work_dir) / "results"

    def _remote_logs_dir(self, job: SimulationJob) -> Path:
        record = (
            job.latest_run(site=self.site_name)
            if isinstance(job, SimulationJob)
            else None
        )
        if record is not None:
            return record.logs_dir
        return job._remote_path(self.work_dir) / "logs"

    def _poll_run(self, run: RunHandle) -> JobStatus:
        status = self.update_status(str(run.id))
        scheduler_status = self._read_scheduler_status(run)
        return_code = (
            0
            if status == "complete"
            else (1 if status in {"failed", "cancelled", "timeout"} else -1)
        )
        message = ""
        raw: Dict[str, Any] = {"scheduler": "slurm"}
        if scheduler_status is not None:
            raw["task_status"] = scheduler_status
            message = self._format_scheduler_status(scheduler_status)
        job_status = JobStatus(
            state=status,
            return_code=return_code,
            job_id=str(run.id),
            message=message,
            raw=raw,
        )
        return job_status

    def _scheduler_status_path(self, job: SimulationJob) -> Path:
        """Return the remote scheduler progress file for a batch simulation job."""

        return self._remote_logs_dir(job) / "scheduler_status.json"

    def _read_scheduler_status(self, run: RunHandle) -> Optional[Dict[str, Any]]:
        """Read the remote adaptive scheduler status payload if it exists."""

        job = getattr(run, "job", None)
        if job is None:
            return None
        status_path = self._scheduler_status_path(job)
        cmd = f"test -f {shlex.quote(str(status_path))} && cat {shlex.quote(str(status_path))}"
        try:
            text = self.run_login(cmd).strip()
        except Exception as exc:
            logger.debug("Could not read scheduler status %s: %s", status_path, exc)
            return None
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("Could not parse scheduler status %s: %s", status_path, exc)
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _format_scheduler_status(payload: Dict[str, Any]) -> str:
        """Format task counts from an adaptive scheduler status payload."""

        total = int(payload.get("total") or 0)
        successful = int(payload.get("successful") or payload.get("succeeded") or 0)
        failed = int(payload.get("failed") or 0)
        running = int(payload.get("running") or 0)
        pending = int(
            payload.get("pending") or max(0, total - successful - failed - running)
        )
        return (
            "tasks: "
            f"{successful} successful, {failed} failed, {running} running, "
            f"{pending} pending, {total} total"
        )

    def _allocation_handle(self, job_id: str) -> RunHandle:
        return RunHandle(
            site=self,
            job=None,
            id=str(job_id),
            mode="allocation",
            poll_interval=self.config.poll_interval,
            _status_fn=self._poll_allocation,
            _wait_fn=self._wait_allocation,
            _wait_async_fn=self._wait_allocation_async,
            _finalize_fn=self._finalize_allocation,
            _cancel_fn=lambda run: self.cancel_job(str(run.id)),
        )

    def _poll_allocation(self, run: RunHandle) -> JobStatus:
        status = self.update_status(str(run.id))
        if status == "running":
            return JobStatus(
                state="completed",
                return_code=0,
                job_id=str(run.id),
                message="Allocation is running",
                raw={"scheduler_state": status},
            )
        return_code = (
            1 if status in {"failed", "cancelled", "timeout", "complete"} else -1
        )
        return JobStatus(
            state=status,
            return_code=return_code,
            job_id=str(run.id),
            raw={"scheduler_state": status},
        )

    def _finalize_allocation(self, run: RunHandle, status: JobStatus) -> RunResult:
        if status.is_successful:
            self._attach_compute_client()
        return run._make_result(status)

    def _wait_allocation(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        interval = self.config.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()
        last_state = object()
        while True:
            status = self._poll_allocation(run)
            if status.state != last_state:
                self._emit_status(status)
                last_state = status.state
            if status.is_successful:
                self._attach_compute_client()
                return run._make_result(status)
            if status.is_complete:
                return run._make_result(status)
            if timeout is not None and time.monotonic() - start > timeout:
                return run._make_result(
                    JobStatus(
                        state="timeout",
                        job_id=str(run.id),
                        message=f"Timed out waiting for allocation after {timeout} seconds",
                    )
                )
            time.sleep(interval)

    async def _wait_allocation_async(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        interval = self.config.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()
        last_state = object()
        while True:
            status = await asyncio.to_thread(self._poll_allocation, run)
            if status.state != last_state:
                self._emit_status(status)
                last_state = status.state
            if status.is_successful:
                await asyncio.to_thread(self._attach_compute_client)
                return run._make_result(status)
            if status.is_complete:
                return run._make_result(status)
            if timeout is not None and time.monotonic() - start > timeout:
                return run._make_result(
                    JobStatus(
                        state="timeout",
                        job_id=str(run.id),
                        message=f"Timed out waiting for allocation after {timeout} seconds",
                    )
                )
            await asyncio.sleep(interval)

    def _sync_project(self, project):
        """Sync the project to the site."""
        project._transfer(self)

    def _sync_result(self, result):
        """Sync a result with the site."""
        raise NotImplementedError(
            f"Syncing results is not implemented for {self.site_name}"
        )

    def _sync_simulation(self, simulation):
        """Sync the simulation to the site."""
        raise NotImplementedError(
            f"Syncing simulations is not implemented for {self.site_name}"
        )

    def _submit_slurm_batch(
        self,
        job: SimulationJob,
        config: SlurmRunConfig,
        **kwargs,
    ) -> str:
        duration = config.duration or getattr(
            self.config, "max_duration", "00-02:00:00"
        )
        run_path = self._remote_run_path(config.run_path, job=job)
        script = self._sweep_SLURM_script(
            n_tasks=job.n_tasks,
            n_nodes=config.nodes,
            stdout=str(job._remote_path(self.work_dir) / "logs"),
            duration=duration,
            imaging_job=isinstance(job, ImagingJob),
            **(
                {"procs_per_task": config.procs_per_task}
                if config.procs_per_task is not None
                else {}
            ),
            **(
                {"procs_per_node": config.procs_per_node}
                if config.procs_per_node is not None
                else {}
            ),
            **({"queue": config.queue} if config.queue is not None else {}),
            **({"account": config.account} if config.account is not None else {}),
            **({"notify_on": config.notify_on} if config.notify_on is not None else {}),
            **(
                {"notify_email": config.notify_email}
                if config.notify_email is not None
                else {}
            ),
            run_path=run_path,
            **kwargs,
        )

        remote_script, remote_job = self._transfer_SLURM_job(script, job)

        cmd = f"mkdir -p {run_path}/jobs/batch && "
        cmd += "sbatch "
        if config.slurm_args:
            for arg in config.slurm_args:
                cmd += f"{arg} "
        cmd += f"{remote_script} {remote_job}"

        job_id = self._submit_sbatch(cmd)

        logger.info(
            "Job %s submitted successfully to %s:%s",
            job_id,
            self.site_name,
            config.queue or self.config.queue,
        )
        self._emit(
            f"Submitted {job.name} to {self.site_name}:{config.queue or self.config.queue} "
            f"as job {job_id}"
        )
        job._job_id = job_id
        self._record_site_run(job, scheduler_id=job_id, status="submitted")
        return job_id

    def _poll_attached_run(self, run: RunHandle) -> JobStatus:
        future = run.backend.get("future")
        if future is None:
            return JobStatus(state="unknown", job_id=run.id)
        if future.done():
            if future.exception() is not None:
                return JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message=str(future.exception()),
                )
            return JobStatus(state="completed", return_code=0, job_id=run.id)
        return JobStatus(state="running", job_id=run.id)

    def _wait_attached_run(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        future = run.backend.get("future")
        if future is None:
            status = JobStatus(state="unknown", job_id=run.id)
            return RunResult(job=run.job, status=status, site=self)

        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._get_or_create_event_loop()
        else:
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call run.wait() for an attached SLURM run while an "
                    "asyncio loop is already running; use 'await run' instead."
                )
        if timeout is not None:
            loop.run_until_complete(asyncio.wait_for(future, timeout=timeout))
        else:
            loop.run_until_complete(future)
        status = self._poll_attached_run(run)
        self._emit_status(status)
        return run._make_result(status)

    async def _wait_attached_run_async(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        future = run.backend.get("future")
        if future is None:
            status = JobStatus(state="unknown", job_id=run.id)
            return RunResult(job=run.job, status=status, site=self)

        if timeout is not None:
            await asyncio.wait_for(future, timeout=timeout)
        else:
            await future
        status = self._poll_attached_run(run)
        self._emit_status(status)
        return run._make_result(status)

    def _submit_attached(
        self,
        job: SimulationJob,
        procs_per_task: int = 2,
        *,
        pack: bool = True,
        fresh: bool = False,
    ) -> Future:
        """Submit a job into an already attached compute allocation."""

        loop = self._get_or_create_event_loop()
        future = loop.create_future()
        if self._compute_client is None:
            self._attach_compute_client()

        remote_script, remote_job = self._transfer_job(job, pack=pack, fresh=fresh)
        ntasks_per_item = max(procs_per_task, self.pool.nproc // job.n_tasks)

        if self._compute_client.is_proxy():
            interactive = self.compute_client.invoke_shell()
            cmd = (
                f"cd {self.work_dir} && "
                f"{remote_script} {remote_job} {ntasks_per_item}\n"
            )
            interactive.stdin.write(cmd.encode())
            interactive.stdin.flush()
            monitor = self._monitor_command_output(future, job, interactive)
        else:
            cmd = (
                f"cd {self.work_dir} && "
                f"{remote_script} {remote_job} {ntasks_per_item}"
            )
            interactive = self.login_client.invoke_shell()
            interactive.send(f"ssh {self.compute_host}\n")
            time.sleep(1)
            interactive.send(cmd)
            monitor = self._monitor_command_output(future, job, interactive)

        loop.create_task(monitor)
        return future

    @staticmethod
    def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
        """Return the current event loop, creating one for sync contexts if needed."""

        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _template_env(self, *, keep_trailing_newline: bool = False) -> Environment:
        """Return the Jinja environment used for remote execution scripts."""

        return Environment(
            loader=FileSystemLoader(
                self._FS_dir / "src/frequensolve/orchestrator/sites/hpc/templates"
            ),
            keep_trailing_newline=keep_trailing_newline,
        )

    def _render_template(
        self,
        template_name: str,
        *,
        keep_trailing_newline: bool = False,
        **context,
    ) -> str:
        """Render a scheduler template from the FrequenSolve template directory."""

        return (
            self._template_env(keep_trailing_newline=keep_trailing_newline)
            .get_template(template_name)
            .render(**context)
        )

    def _attach_compute_client(self):
        """Connect to the current pool's compute host and populate pool metadata."""

        compute_client = self._authenticator.connect_to_job_host(self.pool.id)
        self._compute_client = SSHClientClass(compute_client)
        self._set_pool_info()

    def _submit_sbatch(self, cmd: str) -> str:
        """Run an sbatch command and return the submitted job id."""

        _, stdout, stderr = self.run_login_cmd(cmd)
        output = _read_stream(stdout)
        err = _read_stream(stderr)
        logger.debug("sbatch output: %s", output)
        try:
            job_id = _parse_sbatch_job_id(output)
        except ValueError:
            if err:
                logger.error("sbatch error: %s", err)
            raise
        if err:
            logger.debug("sbatch stderr: %s", err)
        return job_id

    def _get_solver_path(self) -> str:
        """Get the solver executable path on the remote system."""
        load_dotenv()
        executable = os.getenv(self.solver_executable_env)
        if executable is None or executable == "":
            executable = self.default_solver_executable
        if executable is None or executable == "":
            raise ValueError(
                f"Solver executable not specified; set {self.solver_executable_env} "
                "or override default_solver_executable."
            )
        return executable

    def _get_FS_path(self) -> Path:
        """Get the local FrequenSolve repository path used for script templates."""
        load_dotenv()
        env_path = os.getenv(self.python_path_env)
        path = Path(env_path) if env_path else Path(__file__).resolve().parents[4]
        if not path.exists():
            raise FileNotFoundError(
                f"env var {self.python_path_env}:{path} does not exist"
            )
        return path

    def _set_pool_info(self):
        """Get information about the pool."""
        logger.debug("Getting pool info for job %s", self.pool.id)

        # Get SLURM job details using scontrol
        stdout = self.run_login(f"scontrol show job {self.pool.id}")
        entries = stdout.split()

        deets = {}
        for entry in entries:
            try:
                key, value = entry.split("=")
                deets[key] = value
            except ValueError:
                continue

        host = deets["BatchHost"]
        nproc = int(deets["NumTasks"])
        nhost = int(deets["NumNodes"])
        ncore = int(deets["NumCPUs"])
        start_time = deets["StartTime"]
        end_time = deets["EndTime"]
        time_limit = deets["TimeLimit"]
        run_time = deets["RunTime"]
        seconds = _hms_to_seconds(time_limit) - _hms_to_seconds(run_time)
        _seconds_to_hms(seconds)

        logger.info(
            "Pool info - host: %s, nodes: %d, tasks: %d, cores: %d, start: %s, end: %s",
            host,
            nhost,
            nproc,
            ncore,
            start_time,
            end_time,
        )

        self.pool.hostnode = host
        self.pool.nhost = nhost
        self.pool.nproc = nproc
        self.pool.ncore = ncore
        self.pool.start_time = start_time
        self.pool.end_time = end_time
        logger.info("Current status of pool %s: %s", self.pool.id, self.pool.status)

    def _list_jobs(self):
        """List all queued jobs."""
        jobs = self.run_login(
            f'squeue -u {self.credentials.username} -h --format="%.10i %.10B %.5D %.4t %.10L"'
        )
        return jobs

    def _select_job(self):
        """Select a queued job when exactly one is available.

        Interactive prompts are intentionally avoided in the SDK core. Callers
        should pass a job id explicitly when more than one allocation exists.
        """
        jobs = self._list_jobs()

        job_lines = [line for line in jobs.splitlines() if line.strip()]
        if not job_lines:
            raise RuntimeError("No jobs found")
        if len(job_lines) == 1:
            return job_lines[0].split()[0]
        job_ids = ", ".join(line.split()[0] for line in job_lines)
        raise RuntimeError(
            "Multiple SLURM jobs are available; pass job_id explicitly. "
            f"Available job ids: {job_ids}"
        )

    def _transfer_SLURM_job(self, script: str, job: SimulationJob):
        """Transfer a SLURM job to the remote site."""
        remote_script = (self.work_dir / "sweep").with_suffix(".slurm")
        with _temporary_text_file(
            script, suffix=".slurm", prefix="sweep"
        ) as script_path:
            logger.debug("Temporary sweep script created at %s", script_path)
            self.put(script_path, remote_script)

        local_job, remote_job = job.save_for_remote(
            self.__class__.__name__, self.work_dir
        )

        self._transfer_remote_simulation_inputs(job)
        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(local_job), Path(remote_job))
        self.run_login(f"chmod 700 {remote_script}")

        return remote_script, remote_job

    def _remote_run_path(
        self,
        run_path: Optional[Union[str, Path]],
        *,
        job: Optional[SimulationJob] = None,
    ) -> Path:
        """Return the remote directory where SLURM scripts should run."""

        if run_path is None:
            return self.work_dir

        path = Path(run_path)
        if job is None:
            return path

        try:
            local_project = Path(job.project_path).resolve()
            relative = path.expanduser().resolve().relative_to(local_project)
        except Exception:
            return path
        return self.work_dir / relative

    def _transfer_job(
        self, job: SimulationJob, *, pack: bool = True, fresh: bool = False
    ):
        """Submit a simulation job to the remote site.

        Args:
            job (SimulationJob): The simulation job to submit
        """
        if self._compute_client is None:
            raise NotImplementedError("Batch sweep job not implemented yet.")

        # Note: job must be saved for remote **before** script is generated
        local_job, remote_job = job.save_for_remote(
            self.__class__.__name__, self.work_dir
        )
        script = self._sweep_script(job, pack=pack, fresh=fresh)

        self._transfer_remote_simulation_inputs(job)
        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(local_job), Path(remote_job))

        remote_script = (self.work_dir / "sweep").with_suffix(".sh")
        with _temporary_text_file(script, suffix=".sh", prefix="sweep") as script_path:
            logger.debug("Temporary sweep script created at %s", script_path)
            self.put(script_path, remote_script)

        self.run_login(f"chmod 700 {remote_script}")

        return remote_script, remote_job

    def _transfer_remote_simulation_inputs(self, job: SimulationJob) -> None:
        """Transfer the simulation JSON and direct file inputs for a staged job."""

        local_sim, remote_sim = job.save_simulation_for_remote(
            self.__class__.__name__, self.work_dir
        )
        logger.debug("Transferring simulation file to remote path: %s", remote_sim)
        self.put(Path(local_sim), Path(remote_sim))

        for local_file, remote_file in job.remote_input_files(self.work_dir):
            logger.debug("Transferring input file to remote path: %s", remote_file)
            self.put(Path(local_file), Path(remote_file))

    def _is_running(self, job_id: int):
        """Check if a job is running."""
        status = self.run_login(f"squeue -j {job_id} -h -o %t")
        return status == "R"

    def _sweep_script(self, job: SimulationJob, **kwargs) -> str:
        """Generate a script for sweeping through tasks on pre-provisioned resources."""

        n_tasks = job.n_tasks
        dir_out = str(job._remote_path(self.work_dir) / "logs")
        pack_job = kwargs.pop("pack", None)
        if pack_job is None:
            pack_job = kwargs.pop("pack_job", True)
        return self._render_template(
            "sweep/sweep_SLURM.sh",
            batch_job=False,
            n_tasks=n_tasks,
            n_procs=self.pool.nproc,
            n_threads=self.pool.ncore // self.pool.nproc,
            mpi=self.mpi_cmd,
            dir_out=dir_out,
            executable=self.executable,
            imaging_job=isinstance(job, ImagingJob),
            pack_job=bool(pack_job),
            fs_dir=str(Path(self.executable).parent),
            **kwargs,
        )

    def _sweep_SLURM_script(
        self,
        n_tasks: int,
        n_nodes: int,
        stdout: str,
        name: str = "FrequenSolve",
        duration: str = "00-02:00:00",
        procs_per_node: int = 8,
        notify_on: Optional[Literal["begin", "end", "fail", "all", "none"]] = None,
        notify_email: Optional[str] = None,
        imaging_job: bool = False,
        **kwargs,
    ) -> str:
        """Generate a SLURM sweep script.

        Args:
            n_tasks:        Number of tasks (frequencies) to run
            duration:       Duration of the job (DD-HH:MM:SS)
            n_nodes:        Number of nodes to run on
            procs_per_node: Number of processes per node
            procs_per_task: Number of processes per task
            queue:          Queue/partition to run on (optional, defaults to site queue)
            account:        Account/allocation to run on
            notify_on:      Notify on event (optional)
            notify_email:   Email address to notify (optional)
            **kwargs:       Additional keyword arguments
        """

        # Unpack keyword arguments
        queue = str(kwargs.pop("queue", self.config.queue))
        config = self.config_for_queue(queue)
        account = str(kwargs.pop("account", self.config.account))
        run_path = str(kwargs.pop("run_path", self.work_dir))
        mem_cushion = float(kwargs.pop("mem_cushion", 1.5))
        min_ranks = int(kwargs.pop("min_ranks", 1))
        round_to = int(kwargs.pop("round_to", 1))
        cap_fraction = float(kwargs.pop("cap_fraction", 1.0))
        tail_threshold = int(kwargs.pop("tail_threshold", 8))
        boost_max_factor = float(kwargs.pop("boost_max_factor", 8.0))
        sizing_json = kwargs.pop("sizing_json", None)
        pack_job = bool(kwargs.pop("pack", True))
        proc_memory = (config.memory_per_node / procs_per_node) / 1024.0
        duration = config.validate_request(n_nodes, n_nodes * procs_per_node, duration)

        return self._render_template(
            "sweep/adaptive_sweep.sh",
            batch_job=True,
            name=name,
            dir_out=stdout,
            proc_memory=proc_memory,
            mem_cushion=mem_cushion,
            min_ranks=min_ranks,
            round_to=round_to,
            cap_fraction=cap_fraction,
            tail_threshold=tail_threshold,
            boost_max_factor=boost_max_factor,
            n_nodes=n_nodes,
            n_procs=n_nodes * procs_per_node,
            n_threads=config.cores_per_node // procs_per_node,
            n_tasks=n_tasks,
            duration=duration,
            queue=queue,
            account=account,
            imaging_job=imaging_job,
            pack_job=pack_job,
            mpi=self.mpi_cmd,
            executable=self.executable,
            fs_dir=str(Path(self.executable).parent),
            **({"sizing_json": sizing_json} if sizing_json is not None else {}),
            **({"run_path": run_path} if run_path is not None else {}),
            **({"notify_on": notify_on.upper()} if notify_on is not None else {}),
            **({"notify_email": notify_email} if notify_email is not None else {}),
            **kwargs,
        )

    def _generate_provision_script(
        self,
        n_nodes: int,
        procs_per_node: int,
        duration: str = "00-02:00:00",
        queue: Optional[str] = None,
        account: Optional[str] = None,
        notify_email: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a script for provisioning a SLURM allocation.

        Args:
            n_nodes:        Number of nodes to provision
            procs_per_node: Number of processes per node
            duration:       Duration of the job (DD-HH:MM:SS)
            queue:          Queue/partition to run on (optional, defaults to site queue)
            account:        Account/allocation to run on
            notify_email:   Email address to notify (optional)
        """
        name = kwargs.get("name", "FS_cluster")

        return self._render_template(
            "provision/provision_SLURM.sh",
            keep_trailing_newline=True,
            name=name,
            n_nodes=n_nodes,
            procs_per_node=procs_per_node,
            queue=queue,
            account=account,
            duration=duration,
            work_dir=self.work_dir,
            mpi=self.config.mpi_wrapper,
            **({"notify_email": notify_email} if notify_email is not None else {}),
        )

    def _print_batch_logs(
        self,
        job_name: str,
        log_dir: Path,
        job_id: Optional[str],
    ) -> None:
        """Print SLURM batch stdout/stderr logs if they were fetched."""
        log_path = Path(log_dir)

        if job_id:
            batch_dir = log_path.parent / "batch"
            for suffix, label in ((".o", "batch stdout"), (".e", "batch stderr")):
                f = batch_dir / f"job_{job_id}{suffix}"
                if f.exists():
                    self._print_file_with_header(str(f), f"[{job_name}] {label}")

    @staticmethod
    def _latest_task_log_path(log_dir: Union[str, Path]) -> Optional[Path]:
        """Return path to the task log file with the highest index (task_N.txt or task_N.out)."""
        log_path = Path(log_dir)
        if not log_path.is_dir():
            return None
        best_index = -1
        best_path = None
        for pattern in ("task_*.log", "task_*.txt", "task_*.out"):
            for f in log_path.glob(pattern):
                try:
                    n = int(f.stem.rsplit("_", 1)[-1])
                    if n > best_index:
                        best_index = n
                        best_path = f
                except (ValueError, IndexError):
                    continue
        return best_path

    @staticmethod
    def _print_file_with_header(path: str, header: str) -> None:
        """Print a file's contents with a clear header."""
        try:
            text = Path(path).read_text(errors="replace")
        except OSError as e:
            print(f"--- {header} (could not read: {e}) ---")
            return
        print(f"\n{'='*60}\n{header}\n{path}\n{'='*60}\n{text}\n")

    def _get_work_dir(self, rel_proj_path: Union[str, Path]) -> Path:
        """Gets the remote work directory path."""
        work_dir = os.getenv(self.work_dir_env)

        # If the configured variable is not set, try $WORK on the login node.
        if not work_dir or work_dir == "":
            _, stdout, stderr = self._login_client.client.exec_command("echo $WORK")
            work_dir = stdout.read().decode().strip()
            if not work_dir:
                raise RuntimeError(
                    f"Failed to get remote work directory for {self.site_name}; "
                    f"set {self.work_dir_env} in your environment or .env file"
                )

        self._work_dir = Path(work_dir) / rel_proj_path
        logger.info("Work directory: %s", self._work_dir)
        return self._work_dir

    def _get_free_port(self) -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

    async def _monitor_command_output(self, future, job, interactive=None):
        """Monitor command output and update future accordingly."""
        try:
            output_buffer = ""
            error_buffer = ""

            while True:
                # Handle subprocess case
                if isinstance(interactive, subprocess.Popen):
                    # Check if process has ended
                    if interactive.poll() is not None:
                        if not future.done():
                            future.set_exception(
                                RuntimeError("Process ended unexpectedly")
                            )
                        return

                    # Read from stdout/stderr
                    reads, _, _ = select(
                        [interactive.stdout, interactive.stderr], [], [], 0.1
                    )
                    for fd in reads:

                        data = fd.read(4096).decode("utf-8", errors="replace")
                        if fd == interactive.stdout:
                            output_buffer += data
                        else:
                            error_buffer += data

                # Process output buffer
                while "\n" in output_buffer:
                    line, output_buffer = output_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._emit(line)
                    if "Sweep Complete" in line:
                        future.set_result(job.traces)
                        logger.info("Sweep job completed successfully")
                        if interactive:
                            if isinstance(interactive, subprocess.Popen):
                                try:
                                    pgid = os.getpgid(interactive.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except Exception:
                                    try:
                                        interactive.kill()
                                    except Exception:
                                        pass
                            else:
                                interactive.close()
                        return

                # Process error buffer
                while "\n" in error_buffer:
                    line, error_buffer = error_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._emit(line, level=logging.ERROR)
                        future.set_exception(RuntimeError(f"Sweep job failed: {line}"))
                        if interactive:
                            if isinstance(interactive, subprocess.Popen):
                                try:
                                    pgid = os.getpgid(interactive.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except Exception:
                                    try:
                                        interactive.kill()
                                    except Exception:
                                        pass
                            else:
                                interactive.close()
                        return

                await asyncio.sleep(0.2)

        except Exception as e:
            logger.exception("Error in monitor task")
            future.set_exception(e)
            if interactive:
                if isinstance(interactive, subprocess.Popen):
                    try:
                        pgid = os.getpgid(interactive.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        try:
                            interactive.kill()
                        except Exception:
                            pass
                else:
                    interactive.close()
