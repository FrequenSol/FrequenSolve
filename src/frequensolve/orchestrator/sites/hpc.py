"""Generic SSH/SLURM HPC site support.

This module contains the reusable mechanics for SLURM-backed remote sites.
Site-specific modules should provide credentials, queue configuration, and
default paths by subclassing :class:`SlurmSite`.
"""

import asyncio
import glob
import os
import signal
import socket
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from asyncio import Future
from dataclasses import dataclass
from pathlib import Path
from select import select
from threading import Event, Thread
from typing import Any, Dict, List, Literal, Optional, Type, Union

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from paramiko import (
    AuthenticationException,
    AutoAddPolicy,
    SSHClient,
    Transport,
)

from frequensolve.orchestrator.config.base import BaseSiteConfig
from frequensolve.orchestrator.credentials import Credentials
from frequensolve.orchestrator.pool import PoolInfo
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    _check_if_notebook,
    _wait_for_path,
)
from frequensolve.orchestrator.sites.slurm_helpers import (
    SLURM_STATUS_COLORS,
)
from frequensolve.orchestrator.sites.slurm_helpers import as_list as _as_list
from frequensolve.orchestrator.sites.slurm_helpers import (
    hms_to_seconds as _hms_to_seconds,
)
from frequensolve.orchestrator.sites.slurm_helpers import (
    normalize_slurm_state as _normalize_slurm_state,
)
from frequensolve.orchestrator.sites.slurm_helpers import (
    parse_sbatch_job_id as _parse_sbatch_job_id,
)
from frequensolve.orchestrator.sites.slurm_helpers import read_stream as _read_stream
from frequensolve.orchestrator.sites.slurm_helpers import (
    seconds_to_hms as _seconds_to_hms,
)
from frequensolve.orchestrator.sites.slurm_helpers import (
    temporary_text_file as _temporary_text_file,
)
from frequensolve.orchestrator.ssh import SSHClientClass, SSHProxy
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.imaging import ImageDatabase, ImagingJob
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

__all__ = [
    "SlurmSiteConfig",
    "SlurmLoginCredentials",
    "SlurmSite",
    "HPCSite",
    "HPCSiteConfig",
    "HPCSiteCredentials",
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


HPCSiteConfig = SlurmSiteConfig


# ----------------------------------
# Generic Login Credentials
# ----------------------------------
class SlurmLoginCredentials(Credentials):
    """Generic SSH credentials for SLURM-backed HPC sites."""

    user_env: str = "HPC_USERNAME"
    pw_env: str = "HPC_PASSWORD"
    ssh_key_env: str = "SSH_PASSPHRASE"


HPCSiteCredentials = SlurmLoginCredentials


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
    ):
        queue = default_queue if default_queue is not None else self.default_queue
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
        self._rel_proj_path = Path(rel_path)

        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH client authenticated successfully")

        self._work_dir = self._get_work_dir(self._rel_proj_path)
        self._executable = self._get_solver_path()
        self._FS_dir = self._get_FS_path()

        self.pool = PoolInfo()
        self._is_notebook = _check_if_notebook()
        self._compute_client = None

        logger.info("%s initialized with work_dir: %s", self.site_name, self._work_dir)

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
        return self._get_job_host(self.pool.id)

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
        logger.info("Entering %s context manager.", self.site_name)
        self.credentials = self.credentials_cls()
        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH Client re-established in context manager.")
        self._work_dir = self._get_work_dir(self._rel_proj_path)
        logger.info("Work directory re-set to: %s", self._work_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Exiting %s context manager.", self.site_name)
        if getattr(self, "_compute_client", None):
            self.compute_client.close()
            logger.debug("Compute client closed.")
        if getattr(self, "_login_client", None):
            self.login_client.close()
            logger.debug("SSH client closed.")

    def __del__(self):
        logger.info("Deleting %s instance and cleaning up resources.", self.site_name)
        if getattr(self, "_compute_client", None):
            self._compute_client.close()
            logger.debug("Compute client closed in __del__.")
        if getattr(self, "_login_client", None):
            self._login_client.close()
            logger.debug("SSH client closed in __del__.")

    def authenticate(self, host: Optional[str] = None):
        """Connect to the login node using Paramiko or an existing SSH control socket."""

        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Authentication must be called from the main thread")
        host = host or getattr(self.config, "hostname", None) or self.default_host
        if not host:
            raise ValueError("No login host configured for SLURM site")

        logger.info("Starting authentication with host: %s", host)

        # Check for existing control sockets
        control_dir = os.path.expanduser("~/.ssh/control")
        if os.path.exists(control_dir):
            # Look for control sockets
            for control_path in glob.glob(f"{control_dir}/*"):
                try:
                    result = subprocess.run(
                        [
                            "ssh",
                            "-q",
                            "-o",
                            "StrictHostKeyChecking=no",
                            "-o",
                            f"ControlPath={control_path}",
                            f"{self.credentials.username}@{host}",
                            "echo 'Connection test'",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    print(result.stdout.strip())

                    if result.returncode == 0:
                        logger.debug(f"Found working control socket at {control_path}")

                        # Create proxy client with the username from credentials
                        proxy_client = SSHProxy(
                            control_path=control_path,
                            username=self.credentials.username,
                            host=host,
                        )
                        logger.info("Secure connection established with host: %s", host)
                        return proxy_client

                except Exception as e:
                    logger.debug(
                        f"Failed to use control socket {control_path}: {str(e)}"
                    )
                    continue
        return self._interactive_authentication(host)

    def _interactive_authentication(self, host: str):
        """Normal authentication flow when control socket is not available."""
        login_client = SSHClient()
        login_client.set_missing_host_key_policy(AutoAddPolicy())

        # Create a direct socket connection to SSH service.
        sock = socket.create_connection((host, 22))
        transport = Transport(sock)
        transport.start_client()

        authenticated = False
        try:
            from paramiko.agent import Agent

            logger.debug("Attempting agent-based authentication.")
            agent = Agent()
            agent_keys = agent.get_keys()
            for key in agent_keys:
                try:
                    transport.auth_publickey(self.credentials.username, key)
                    if transport.is_authenticated():
                        authenticated = True
                        break
                except Exception as err:
                    logger.debug("Agent key authentication failed: %s", str(err))
                    continue
        except Exception as err:
            logger.debug("Agent-based authentication exception: %s", str(err))

        if not authenticated:
            logger.debug("Attempting keyboard-interactive authentication.")

            def handler(title, instructions, prompt_list):
                responses = []
                for prompt, echo in prompt_list:
                    if "Password" in prompt:
                        responses.append(self.credentials.password)
                    elif "Token" in prompt or "2FA" in prompt or "Code" in prompt:
                        responses.append(self.credentials.duo_code)
                    else:
                        responses.append("")
                return responses

            try:
                transport.auth_interactive(self.credentials.username, handler)
                authenticated = transport.is_authenticated()
                if authenticated:
                    logger.debug("Keyboard-interactive authentication successful.")
                else:
                    logger.debug("Keyboard-interactive authentication failed.")
            except Exception as err:
                logger.debug(
                    "Keyboard-interactive authentication exception: %s", str(err)
                )

        if not transport.is_authenticated():
            logger.error(
                "Authentication failed for user: %s", self.credentials.username
            )
            raise AuthenticationException("Authentication failed.")

        transport.set_keepalive(120)
        login_client._transport = transport
        logger.info("Secure connection established with host: %s", host)
        return login_client

    def provision(
        self, nodes: int, tasks: int, duration: Optional[str] = None, **kwargs
    ):
        nhost = nodes
        nproc = tasks
        logger.info(
            "Provisioning SLURM job with nhost=%d, nproc=%d, duration=%s",
            nhost,
            nproc,
            duration,
        )
        duration = duration or getattr(self.config, "max_duration", "00-02:00:00")
        duration = self.config.validate_request(nhost, nproc, duration)
        script = self._generate_provision_script(nhost, nproc, duration, **kwargs)

        start_event = Event()

        with _temporary_text_file(script, suffix=".sh", prefix="slurm_") as script_path:
            logger.debug("Temporary SLURM script created at %s", script_path)
            remote_path = f"/tmp/{os.path.basename(script_path)}"
            try:
                self.put(script_path, remote_path)

                self.pool.id = self._submit_sbatch(f"sbatch {remote_path}")
                logger.debug("Job submitted successfully with job ID: %s", self.pool.id)
                self.pool._status.status = "pending"

                def monitor_remote_pool():
                    while True:
                        status = self.update_status()
                        if status == "running":
                            start_event.set()
                            break
                        if status not in {"pending", "unknown"}:
                            break
                        time.sleep(self.config.poll_interval)

                Thread(target=monitor_remote_pool, daemon=True).start()

            except Exception as e:
                logger.exception("Exception occurred during provisioning: %s", str(e))
                self.pool._status.status = "failed"
                self.pool._status.stderr = str(e)

        return start_event

    def attach_to_existing_job(self, job_id: Optional[str] = None):
        """Attach to an existing job.

        If job_id is not provided, queued jobs will be listed and the user will
        be prompted to select a job.
        """
        if job_id is None:
            job_id = self._select_job()
        self.pool.id = job_id
        logger.info("Attaching to existing job with ID: %s", self.pool.id)

        while True:
            status = self.update_status()
            if status == "running":
                self._attach_compute_client()
                return
            if status != "pending":
                raise RuntimeError(f"Job {job_id} ended with status: {status}")
            print(
                f"\033[38;5;244mJob {job_id} status: "
                f"\033[38;5;27m{status.capitalize()}\033[0m",
                end="\r",
            )
            time.sleep(self.config.poll_interval)

    def sync(self, project):
        """Sync the project to the site."""
        self._sync_project(project)

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

    def submit_SLURM(
        self,
        job: SimulationJob,
        nodes: int,
        wait: bool = False,
        duration: str = "00-02:00:00",
        procs_per_node: Optional[int] = None,
        procs_per_task: Optional[int] = None,
        queue: Optional[str] = None,
        account: Optional[str] = None,
        notify_on: Optional[Literal["begin", "end", "fail", "all", "none"]] = None,
        notify_email: Optional[str] = None,
        run_path: Optional[str] = None,
        **kwargs,
    ):
        """Submit job to SLURM queue.

        Args:
            n_tasks:        Number of tasks to run
            n_nodes:        Number of nodes to run on
            procs_per_node: Number of processes per node
            procs_per_task: Number of processes per task
            wait:           Wait for job to complete
            duration:       Duration of the job
            queue:          Queue to run on
            account:        Allocation/account to run on
            notify_on:      Notify on event
            notify_email:   Email address to notify
            run_path:       Path where the slurm job will be run
        """

        script = self._sweep_SLURM_script(
            n_tasks=job.n_tasks,
            n_nodes=nodes,
            stdout=str(job._remote_path(self.work_dir) / "logs"),
            duration=duration,
            imaging_job=isinstance(job, ImagingJob),
            **(
                {"procs_per_task": procs_per_task} if procs_per_task is not None else {}
            ),
            **(
                {"procs_per_node": procs_per_node} if procs_per_node is not None else {}
            ),
            **({"queue": queue} if queue is not None else {}),
            **({"account": account} if account is not None else {}),
            **({"notify_on": notify_on} if notify_on is not None else {}),
            **({"notify_email": notify_email} if notify_email is not None else {}),
            **({"run_path": run_path} if run_path is not None else {}),
            **kwargs,
        )
        # if procs_per_node * nodes > procs_per_task * job.n_tasks:
        #     raise ValueError(
        #         f"Number of workers ({procs_per_node * nodes}) "
        #         f"is greater than number of tasks ({procs_per_task * job.n_tasks}), "
        #         "reduce the number of workers (by decreasing the "
        #         "n_nodes or processes_per_node) or increase the number of "
        #         "tasks."
        #     )

        if run_path is None:
            run_path = self.work_dir
        remote_script, remote_job = self._transfer_SLURM_job(script, job)

        cmd = f"mkdir -p {run_path}/jobs/batch && "
        cmd += "sbatch "
        if "slurm_args" in kwargs:
            for arg in kwargs["slurm_args"]:
                cmd += f"{arg} "
        cmd += f"{remote_script} {remote_job}"

        job_id = self._submit_sbatch(cmd)

        print(
            f"Job {job_id} submitted successfully to {self.site_name}:{queue or self.config.queue}"
        )
        job._job_id = job_id

        if wait:
            self.wait_completion(job)
        return job_id

    def submit_slurm(self, *args, **kwargs):
        """Snake-case alias for submit_SLURM."""

        return self.submit_SLURM(*args, **kwargs)

    def wait_completion(self, job: Union[SimulationJob, List[SimulationJob]]):
        """Wait for job to complete and download results."""

        jobs, _ = _as_list(job, SimulationJob)
        job_ids = [str(j._job_id) for j in jobs]
        if any(not j_id or j_id == "None" for j_id in job_ids):
            raise ValueError("All jobs must have a SLURM _job_id before waiting")

        statuses = {j_id: "pending" for j_id in job_ids}
        active_jobs = set(job_ids)
        name_width = max(len(job.name) for job in jobs)
        for job in jobs:
            j_id = str(job._job_id)
            print(
                f"\033[38;5;244mJob {j_id} ({job.name:<{name_width}}): "
                f"\033[38;5;27m{statuses[j_id].capitalize()}\033[0m"
            )

        while active_jobs:
            print(f"\033[{len(job_ids)}A", end="")
            active_ids = ",".join(active_jobs)
            cmd = f"sacct -j {active_ids} --format=JobID,State --noheader --parsable2"
            _, stdout, stderr = self.run_login_cmd(cmd)
            output = _read_stream(stdout)

            for line in output.split("\n"):
                if not line:
                    continue
                job_id, state = line.split("|")
                job_id = job_id.split(".")[0]  # Remove any array task IDs
                if job_id in active_jobs:
                    statuses[job_id] = _normalize_slurm_state(state)

            for job in jobs:
                j_id = str(job._job_id)
                status = statuses[j_id]

                if j_id in active_jobs:
                    if status not in ["pending", "running"]:
                        active_jobs.remove(j_id)

                print(
                    f"\033[38;5;244mJob {j_id} ({job.name:<{name_width}}): "
                    f"{SLURM_STATUS_COLORS[status]}{status.capitalize()}\033[0m\033[K"
                )

            time.sleep(self.config.poll_interval)

        print()  # Final newline

        for job in jobs:
            j_id = str(job._job_id)
            status = statuses[j_id]
            if status in ["complete", "failed", "timeout"]:
                print(
                    f"Job {j_id} ({job.name}) completed with status: {status.capitalize()}"
                )
            elif status == "cancelled":
                print(f"Job {j_id} ({job.name}) was cancelled.")
            else:
                print(
                    f"Job {j_id} ({job.name}) returned with unknown status: {status}."
                )

    def submit(self, job: SimulationJob, procs_per_task: int = 2):
        """Submit job and block until completion."""

        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        loop = asyncio.get_event_loop()
        future = self.submit_async(job, procs_per_task)
        return loop.run_until_complete(future)

    def submit_async(self, job: SimulationJob, procs_per_task: int = 2) -> Future:
        """Submit job asynchronously and return a future."""

        future = Future()
        if self.provisioned:  # Run on already provisioned compute node
            remote_script, remote_job = self._transfer_job(job)
            ntasks_per_item = max(procs_per_task, self.pool.nproc // job.n_tasks)

            if self._compute_client.is_proxy():
                interactive = self.compute_client.invoke_shell()
                cmd = f"cd {self.work_dir} && {remote_script} {remote_job} {ntasks_per_item}\n"
                interactive.stdin.write(cmd.encode())
                interactive.stdin.flush()
                monitor = self._monitor_command_output(future, job, interactive)
            else:
                cmd = f"cd {self.work_dir} && {remote_script} {remote_job} {ntasks_per_item}"
                interactive = self.login_client.invoke_shell()
                interactive.send(f"ssh {self.compute_host}\n")
                time.sleep(1)
                interactive.send(cmd)
                monitor = self._monitor_command_output(future, job, interactive)

        # Submit job to SLURM queue
        else:
            raise ValueError(
                "This submit method requires the site to "
                "be provisioned (attached to a running job). "
                "Use submit_SLURM to queue a job."
            )

            # TODO: automatically compute job sizing and submit to queue

        loop = asyncio.get_event_loop()
        loop.create_task(monitor)
        return future

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

    def _template_env(self, *, keep_trailing_newline: bool = False) -> Environment:
        """Return the Jinja environment used for remote execution scripts."""

        return Environment(
            loader=FileSystemLoader(
                self._FS_dir / "src/frequensolve/orchestrator/templates"
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

        compute_client = self._connect_to_job_host(self.pool.id)
        self._compute_client = SSHClientClass(compute_client)
        self._set_pool_info()

    def _remote_spec(self, remote_path: Union[str, Path]) -> str:
        """Return an rsync-compatible user@host:path target."""

        return f"{self.credentials.username}@{self.config.hostname}:{remote_path}"

    def _run_rsync(self, source: str, target: str) -> None:
        """Run rsync and raise a concise error on failure."""

        rsync_cmd = ["rsync", "-azP", source, target]
        logger.debug("rsync: %s", rsync_cmd)
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rsync failed: {result.stderr}")

    def _submit_sbatch(self, cmd: str) -> str:
        """Run an sbatch command and return the submitted job id."""

        _, stdout, stderr = self.run_login_cmd(cmd)
        output = _read_stream(stdout)
        err = _read_stream(stderr)
        if err:
            logger.error("sbatch error: %s", err)
            print(f"sbatch error: \033[91m{err}\033[0m")
        logger.debug("sbatch output: %s", output)
        return _parse_sbatch_job_id(output)

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
        """
        Transfer files from a local path to a remote path.

        Args:
            local_path: Local path to transfer from
            remote_path: Remote path to transfer to
        """
        logger.debug("Transferring %s to %s", local_path, remote_path)
        if not _wait_for_path(local_path):
            logger.error("Local path %s does not exist", local_path)
            raise FileNotFoundError(f"Local path {local_path} does not exist")

        local_path = Path(local_path)
        remote_path = Path(remote_path)

        try:
            # Create parent directory on remote
            parent_path = str(remote_path.parent)
            self.run_login(f"mkdir -p {parent_path}")

            if self.transfer_method == "sftp":
                sftp = self.login_client.open_sftp()
                try:
                    if local_path.is_dir():
                        self._put_dir(sftp, local_path, remote_path)
                    else:
                        sftp.put(str(local_path), str(remote_path))
                finally:
                    sftp.close()
            else:
                source = f"{local_path}/" if local_path.is_dir() else str(local_path)
                self._run_rsync(source, self._remote_spec(remote_path))

            logger.debug("Transfer completed successfully")

        except Exception as e:
            logger.exception("Error during file transfer: %s", str(e))
            raise

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
                trace_dir_name = Path(j.trace_outputs["path"]).name
                remote_dir = j._remote_path(self.work_dir) / "results" / trace_dir_name
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

    def fetch_paraview(self, job: SimulationJob):
        """Get Paraview files from the remote site.

        Args:
            job: A SimulationJob object.
        """

        try:
            remote_dir = job._remote_path(self.work_dir) / "results" / "ParaView/"
            local_dir = job._local_path / "results" / "ParaView/"
            print("Fetching ParaView outputs")
            print(f"from: {remote_dir}")
            print(f"to  : {local_dir}")
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
        local_dir: Optional[Union[str, Path]] = None,
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
            include_batch: If True, also fetch SLURM batch logs from
                run_path/jobs/batch/ (requires job._job_id and run_path).
            print: If True, print log contents after fetching. When
                include_batch is True, prints in order: batch .o file, batch
                .e file, then the last (highest index) task log file. Otherwise
                prints only the last task log file.

        Returns:
            If a single job: Path to the local log directory. If a list of jobs:
            dict mapping job name to Path of that job's local log directory.
        """
        jobs, single = _as_list(job, SimulationJob)
        requested_local_dir = Path(local_dir) if local_dir is not None else None
        result = {}

        for j in jobs:
            log_dir = (
                requested_local_dir
                if requested_local_dir is not None
                else j._local_path / "logs"
            )
            remote_log_dir = j._remote_path(self.work_dir) / "logs"
            try:
                print("Fetching Logs:\nfrom: ", remote_log_dir, "\nto: ", log_dir)
                self.get(remote_log_dir, log_dir)
                result[j.name] = log_dir

                if include_batch and getattr(j, "_job_id", None):
                    batch_remote = self.work_dir / "jobs" / "batch"
                    batch_local = log_dir.parent / "batch"
                    print(
                        "Fetching Batch Logs:\nfrom: ",
                        batch_remote,
                        "\nto: ",
                        batch_local,
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
                    self._print_fetched_logs(
                        job_name=j.name,
                        log_dir=log_dir,
                        include_batch=include_batch,
                        job_id=getattr(j, "_job_id", None),
                    )
            except Exception as e:
                logger.exception("Error fetching logs for job %s: %s", j.name, e)
                raise
        if single:
            return result[jobs[0].name]
        return result

    def download_trace_files(self, traces: dict, project_dir: Union[str, Path]):
        """Download trace files from the remote site.

        Args:
            traces: A dictionary of trace files to get.
            project_dir: The directory to download the traces to.
        """
        project_dir = Path(project_dir)
        files = traces["datasets"].keys()
        try:
            # Create temporary directory name for the payload
            payload_name = f"traces_{int(time.time())}"
            remote_payload = self.work_dir / f"{payload_name}.tar.gz"
            local_payload = project_dir / f"{payload_name}.tar.gz"

            # Create payload on remote
            tar_cmd = f"cd {self.work_dir} && tar czf {remote_payload.name} "
            tar_cmd += " ".join(files)
            _, _, stderr = self.run_login_cmd(tar_cmd)
            err = stderr.read().decode().strip()
            # if err:
            #     raise RuntimeError(f"Failed to create payload on remote: {err}")

            # Download, extract, and cleanup
            self.get(remote_payload, local_payload)

            with tarfile.open(local_payload, "r:gz") as tar:
                logger.debug("Extracting files from payload:")
                tar.extractall(path=local_payload.parent)

            local_payload.unlink()
            self.run_login(f"rm {remote_payload}")
            return traces

        except Exception as e:
            logger.exception("Error downloading traces: %s", str(e))
            raise

    def download_record_files(self, records: dict, project_dir: Union[str, Path]):
        """Compatibility alias for ``download_trace_files``."""
        return self.download_trace_files(records, project_dir)

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Transfer files from a remote path to a local path.

        Args:
            remote_path: Remote path to transfer to
            local_path: Local path to transfer from
            overwrite: Overwrite existing files
        """
        logger.debug("Attempting to transfer from %s to %s", remote_path, local_path)

        local_path = Path(local_path)
        remote_path = Path(remote_path)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if self.transfer_method == "sftp":
                logger.debug("Transferring %s to %s (SFTP)", remote_path, local_path)
                sftp = self.login_client.open_sftp()
                try:
                    if self._sftp_is_dir(sftp, remote_path):
                        self._get_dir(sftp, remote_path, local_path)
                    else:
                        sftp.get(str(remote_path), str(local_path))
                finally:
                    sftp.close()
            else:
                if remote_path.suffix == "":
                    remote_str = f"{remote_path}/"
                else:
                    remote_str = str(remote_path)
                local_str = f"{local_path}/" if local_path.is_dir() else str(local_path)
                self._run_rsync(self._remote_spec(remote_str), local_str)

            logger.debug("Transfer completed successfully")

        except Exception as e:
            logger.exception("Error during file transfer: %s", str(e))
            raise

    def wait_provisioned(self):
        """Wait for the job to be provisioned."""
        while True:
            self.update_status()
            if self.pool.is_running:
                break
            time.sleep(self.config.poll_interval)
        self._attach_compute_client()

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
            except BaseException as e:
                continue

        host = deets["BatchHost"]
        nproc = int(deets["NumTasks"])
        nhost = int(deets["NumNodes"])
        ncore = int(deets["NumCPUs"])
        start_time = deets["StartTime"]
        end_time = deets["EndTime"]
        partition = deets["Partition"]
        time_limit = deets["TimeLimit"]
        run_time = deets["RunTime"]
        seconds = _hms_to_seconds(time_limit) - _hms_to_seconds(run_time)
        duration = _seconds_to_hms(seconds)

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
        print(f" --- User Jobs ---\n {jobs}")
        return jobs

    def _select_job(self):
        """Select a job from the list of jobs.

        If there is only one job, attach automatically.
        """
        jobs = self._list_jobs()

        if len(jobs) == 0:
            raise RuntimeError("No jobs found")
        if len(jobs.split("\n")) == 1:
            print("\nAttached to job:")
            print(jobs)
            return jobs.split()[0]
        else:
            job_id = input("Enter job ID: ")
            return int(job_id)

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

        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(local_job), Path(remote_job))
        self.run_login(f"chmod 700 {remote_script}")

        return remote_script, remote_job

    def _transfer_job(self, job: SimulationJob):
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
        script = self._sweep_script(job)

        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(local_job), Path(remote_job))

        remote_script = (self.work_dir / "sweep").with_suffix(".sh")
        with _temporary_text_file(script, suffix=".sh", prefix="sweep") as script_path:
            logger.debug("Temporary sweep script created at %s", script_path)
            self.put(script_path, remote_script)

        self.run_login(f"chmod 700 {remote_script}")

        return remote_script, remote_job

    def _is_running(self, job_id: int):
        """Check if a job is running."""
        status = self.run_login(f"squeue -j {job_id} -h -o %t")
        return status == "R"

    def _sweep_script(self, job: SimulationJob, **kwargs) -> str:
        """Generate a script for sweeping through tasks on pre-provisioned resources."""

        n_tasks = job.n_tasks
        dir_out = str(job._remote_path(self.work_dir) / "logs")
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
            mpi=self.mpi_cmd,
            executable=self.executable,
            fs_dir=str(Path(self.executable).parent),
            **({"sizing_json": sizing_json} if sizing_json is not None else {}),
            **({"run_path": run_path} if run_path is not None else {}),
            **({"notify_on": notify_on.upper()} if notify_on is not None else {}),
            **({"notify_email": notify_email} if notify_email is not None else {}),
            **kwargs,
        )

    def config_for_queue(self, queue: str):
        """Return the site configuration for a queue/partition name."""
        if self.config_cls is not None:
            return self.config_cls(queue)
        return self.config

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

    def _print_fetched_logs(
        self,
        job_name: str,
        log_dir: Path,
        include_batch: bool,
        job_id: Optional[str],
    ) -> None:
        """Print batch logs (if requested) and the latest task log file.

        Order when include_batch: batch .o, batch .e, then highest-index task log.
        """
        log_path = Path(log_dir)

        if include_batch and job_id:
            batch_dir = log_path.parent / "batch"
            for suffix, label in ((".o", "batch stdout"), (".e", "batch stderr")):
                f = batch_dir / f"job_{job_id}{suffix}"
                if f.exists():
                    self._print_file_with_header(str(f), f"[{job_name}] {label}")

        last_task = self._latest_task_log_path(log_path)
        if last_task is not None:
            self._print_file_with_header(
                str(last_task),
                f"[{job_name}] task log (highest index)",
            )

    @staticmethod
    def _latest_task_log_path(log_dir: Union[str, Path]) -> Optional[Path]:
        """Return path to the task log file with the highest index (task_N.txt or task_N.out)."""
        log_path = Path(log_dir)
        if not log_path.is_dir():
            return None
        best_index = -1
        best_path = None
        for pattern in ("task_*.txt", "task_*.out"):
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

    def _get_job_host(self, job_id: int) -> str:
        """Get the job host."""
        # Check if job is running
        status = self.run_login(f"squeue -j {job_id} -h -o %t").strip()
        if status != "R":
            raise RuntimeError(f"Job {job_id} is not running")

        # Get the hostname of the compute node
        hostname = self.run_login(f"squeue -j {job_id} -h -o %B").strip()
        if not hostname:
            raise RuntimeError(f"Could not get hostname for job {job_id}")

        return hostname

    def _connect_to_job_host(self, job_id: int):
        """Connect to the job host.

        Args:
            job_id (int): The SLURM job ID.

        Returns:
            Union[SSHClient, SSHProxy]: A client connected to the job host.
        """
        job_host = self._get_job_host(job_id)
        logger.debug(f"Got compute node hostname: {job_host}")

        # If using a proxy, just create a new proxy for the compute node
        if self._login_client.is_proxy():
            logger.debug("Using proxy connection to connect to compute node")
            control_path, username = self._login_client.get_proxy_details()
            if not control_path or not username:
                raise RuntimeError("Missing proxy details")
            return SSHProxy(control_path, username, self.login_client.host, job_host)
        else:
            # Otherwise use paramiko SSH tunneling
            logger.debug("Using SSH tunneling to connect to compute node")
            transport = self._login_client.get_transport()
            if not transport:
                raise RuntimeError("No transport available for SSH tunneling")

            channel = transport.open_channel("direct-tcpip", (job_host, 22), ("", 0))
            job_client = SSHClient()
            job_client.set_missing_host_key_policy(AutoAddPolicy())

            try:
                job_client.connect(
                    job_host,
                    username=self.credentials.username,
                    sock=channel,
                    allow_agent=True,
                    look_for_keys=False,
                )
                logger.info("Connected to job host: %s", job_host)
                return job_client
            except Exception as e:
                logger.error(f"Failed to connect to job host: {str(e)}")
                channel.close()
                raise

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

    def _put_dir(self, sftp, local_dir: Path, remote_dir: Path):
        """Transfer directory via SFTP.

        Args:
            sftp: The SFTP client.
            local_dir: The local directory to transfer.
            remote_dir: The remote directory to transfer to.
        """

        # Create temporary tar file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(local_dir, arcname=local_dir.name)

            remote_tar = str(remote_dir.parent / f"{remote_dir.name}.tar.gz")
            sftp.put(tmp.name, remote_tar)

            _, _, stderr = self.run_login_cmd(
                f"cd {remote_dir.parent} && "
                f"tar xzf {remote_dir.name}.tar.gz && "
                f"rm {remote_dir.name}.tar.gz"
            )

            err = stderr.read().decode().strip()
            if err:
                logger.error("Error extracting directory on remote: %s", err)
                raise RuntimeError(f"Failed to extract directory on remote: {err}")

    def _get_dir(self, sftp, remote_dir: Path, local_dir: Path):
        """Transfer directory via SFTP.

        Args:
            sftp: The SFTP client.
            remote_dir: The remote directory to transfer.
            local_dir: The local directory to transfer to.
        """
        remote_tar = str(remote_dir.parent / f"{remote_dir.name}.tar.gz")
        _, _, stderr = self.run_login_cmd(
            f"cd {remote_dir.parent} && tar czf {remote_dir.name}.tar.gz {remote_dir.name}"
        )

        err = stderr.read().decode().strip()
        if err:
            raise RuntimeError(f"Failed to create tar payload on remote: {err}")

        local_tar = str(local_dir.parent / f"{local_dir.name}.tar.gz")
        sftp.get(remote_tar, local_tar)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=local_dir.parent)

        os.remove(local_tar)
        self.run_login(f"rm {remote_tar}")

    @staticmethod
    def _sftp_is_dir(sftp, remote_path: Union[str, Path]) -> bool:
        """Return True if a remote SFTP path is a directory."""

        try:
            return stat.S_ISDIR(sftp.stat(str(remote_path)).st_mode)
        except OSError:
            return False

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
                        print(line, flush=True)
                    if "Sweep Complete" in line:
                        future.set_result(job.traces)
                        logger.info("Sweep job completed successfully")
                        if interactive:
                            if isinstance(interactive, subprocess.Popen):
                                try:
                                    pgid = os.getpgid(interactive.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except:
                                    try:
                                        interactive.kill()
                                    except:
                                        pass
                            else:
                                interactive.close()
                        return

                # Process error buffer
                while "\n" in error_buffer:
                    line, error_buffer = error_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        print(f"\033[91m{line}\033[0m", flush=True)
                        future.set_exception(RuntimeError(f"Sweep job failed: {line}"))
                        if interactive:
                            if isinstance(interactive, subprocess.Popen):
                                try:
                                    pgid = os.getpgid(interactive.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except:
                                    try:
                                        interactive.kill()
                                    except:
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
                    except:
                        try:
                            interactive.kill()
                        except:
                            pass
                else:
                    interactive.close()


# Backward-compatible generic site name.
HPCSite = SlurmSite
