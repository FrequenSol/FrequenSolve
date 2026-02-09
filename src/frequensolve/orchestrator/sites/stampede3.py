"""
Stampede3 HPC site.

Manages authentication, transfer, and resource provisioning on Stampede3.
"""

import asyncio
import glob
import os
import re
import signal
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
from asyncio import Future
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from select import select
from threading import Event, Thread
from typing import Dict, List, Literal, Optional, TextIO, Union

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from paramiko import (
    AuthenticationException,
    AutoAddPolicy,
    SSHClient,
    Transport,
)

from frequensolve.orchestrator.config.stampede3 import (
    Stampede3Config,
    _hms_to_seconds,
    _seconds_to_hms,
)
from frequensolve.orchestrator.credentials import Credentials
from frequensolve.orchestrator.pool import PoolInfo
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    _check_if_notebook,
    _wait_for_path,
)
from frequensolve.orchestrator.ssh import SSHClientClass, SSHProxy
from frequensolve.seismic.record_database import RecordDatabase
from frequensolve.simulation.imaging import ImageDatabase, ImagingJob
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

__all__ = ["Stampede3Site"]

# Initialize the logger
logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/stampede3.log")


# ----------------------------------
# TACC Login Credentials
# ----------------------------------
class TACCLoginCredentials(Credentials):
    """Credentials for Stampede3 HPC."""

    user_env: str = "TACC_USERNAME"
    pw_env: str = "TACC_PASSWORD"
    ssh_key_env: str = "SSH_PASSPHRASE"


# ----------------------------------
# Stampede3 Site
# ----------------------------------
@dataclass(kw_only=True, init=False)
class Stampede3Site(BaseSite):
    """
    Stampede3 HPC site.

    Manages authentication, transfer to and from, and running jobs on Stampede3.
    """

    credentials: TACCLoginCredentials
    config: Stampede3Config
    pool: PoolInfo
    remote_env: dict
    transfer_method: Literal["rsync", "sftp"] = "rsync"
    _executable: str
    _login_client: SSHClientClass
    _compute_client: Optional[SSHClientClass] = None
    _work_dir: Path
    _log_file: TextIO
    _FS_dir: Path

    def __init__(
        self,
        rel_path: Union[str, Path],
        transfer_method: Literal["rsync", "sftp"] = "rsync",
        default_queue: str = "skx-dev",
    ):
        logger.debug(
            "Initializing Stampede3Site with rel_path: %s, default_queue: %s",
            rel_path,
            default_queue,
        )

        # Get TACC credentials and node configuration
        self.credentials = TACCLoginCredentials()
        self.config = Stampede3Config(queue=default_queue)
        self.transfer_method = transfer_method

        # SSH into Stampede3
        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH client authenticated successfully")

        # Get work directory and solver path
        self._work_dir = self._get_work_dir(rel_path)
        self._executable = self._get_solver_path()
        self._FS_dir = self._get_FS_path()

        self.pool = PoolInfo()
        self._is_notebook = _check_if_notebook()

        logger.info("Stampede3Site initialized with work_dir: %s", self._work_dir)

    @property
    def executable(self) -> str:
        """Get the solver executable."""

        if self._executable is None:
            raise ValueError(
                "Solver executable not specified; set STAMPEDE3_SOLVER_EXECUTABLE environment variable."
            )
        return self._executable

    @property
    def compute_client(self) -> SSHClient:
        """Get the compute client."""
        return self._compute_client.client

    @property
    def compute_host(self) -> str:
        """Get the compute host."""
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
        """Get the MPI command for Stampede3."""
        return f"{self.config.mpi_wrapper}"

    # TODO: fix this, shouldn't be cached or it won't handle state changes.
    @cached_property
    def pool_host(self) -> str:
        """Get the resource pool host node."""
        return self._get_job_host(self.pool.id)

    @property
    def work_dir(self) -> Path:
        """Gets the $WORK directory path on Stampede3."""
        return self._work_dir

    @property
    def provisioned(self):
        """Check if the site is provisioned."""
        self.update_status()
        return self.pool.is_running

    def __enter__(self):
        logger.info("Entering Stampede3Site context manager.")
        self.credentials = TACCLoginCredentials()
        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH Client re-established in context manager.")
        self._work_dir = self._get_work_dir()
        logger.info("Work directory re-set to: %s", self._work_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Exiting Stampede3Site context manager.")
        if self._compute_client:
            self.compute_client.close()
            logger.debug("Compute client closed.")
        if self._login_client:
            self.login_client.close()
            logger.debug("SSH client closed.")

    def __del__(self):
        logger.info("Deleting Stampede3Site instance and cleaning up resources.")
        if self._compute_client:
            self.compute_client.close()
            logger.debug("Compute client closed in __del__.")
        if self._login_client:
            self.login_client.close()
            logger.debug("SSH client closed in __del__.")

    def authenticate(self, host: str = "stampede3.tacc.utexas.edu"):
        """Connects to Stampede3 Login Node using Paramiko's built-in authentication mechanisms."""

        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Authentication must be called from the main thread")

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
        duration = self.config.validate_request(nhost, nproc, duration)
        script = self._generate_provision_script(nhost, nproc, duration, **kwargs)
        fd, script_path = tempfile.mkstemp(
            suffix=".sh", prefix="slurm_", dir=self.work_dir
        )
        logger.debug("Temporary SLURM script created at %s", script_path)
        with os.fdopen(fd, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o700)

        start_event = Event()

        try:
            remote_path = f"/tmp/{os.path.basename(script_path)}"
            self.put(script_path, remote_path)

            output = self.run_login(f"sbatch {remote_path}")
            logger.debug("sbatch output: %s", output)

            # Get job ID
            job_id = re.search(r"Submitted batch job (\d+)", output)
            if not job_id:
                logger.error("Failed to retrieve job ID from sbatch output: %s", output)
                raise ValueError("failed to get job ID from sbatch output")
            job_id = job_id.group(1)
            logger.debug("Job submitted successfully with job ID: %s", job_id)
            self.pool.id = job_id
            self.pool._status.status = "pending"

            def monitor_remote_pool():
                while True:
                    status = self.run_login(f"squeue -j {self.pool.id} -h -o %t")
                    if "R" in status:
                        self.pool._status.status = "running"
                        start_event.set()
                        break
                    elif "PD" in status:
                        self.pool._status.status = "pending"
                    elif not status:
                        self.pool._status.status = "failed"
                        break
                    else:
                        time.sleep(self.config.poll_interval)

            Thread(target=monitor_remote_pool, daemon=True).start()

        except Exception as e:
            logger.exception("Exception occurred during provisioning: %s", str(e))
            self.pool._status.status = "failed"
            self.pool._status.stderr = str(e)
        finally:
            os.unlink(script_path)
            logger.debug("Temporary provisioning script %s removed.", script_path)

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

        self.update_status()
        if self.pool.is_running:
            compute_client = self._connect_to_job_host(self.pool.id)
            self._compute_client = SSHClientClass(compute_client)
            self._set_pool_info()
        elif self.pool.is_queued:
            logger.debug("Job %s is queued. Waiting for job to start...", self.pool.id)
            while True:
                self.update_status()
                if self.pool.is_running:
                    break
                if self.pool.is_complete:
                    raise RuntimeError(
                        f"Job {job_id} ended with status: {self.pool.status}"
                    )
                print(
                    f"\033[38;5;244mJob {job_id} status: \033[38;5;27m{self.pool.status.capitalize()}\033[0m",
                    end="\r",
                )
                time.sleep(self.config.poll_interval)

            if self.pool.is_running:
                compute_client = self._connect_to_job_host(self.pool.id)
                self._compute_client = SSHClientClass(compute_client)
                self._set_pool_info()
            else:
                raise RuntimeError(f"Job {self.pool.id} is not running")
        else:
            raise RuntimeError(
                f"Unable to attach to {self.pool.id}, status: {self.pool.status}"
            )

    def sync(self, project):
        """Sync the project to the site."""
        self._sync_project(project)

    def _sync_project(self, project):
        """Sync the project to the site."""
        project._transfer(self)

    def _sync_result(self, result):
        """Sync a result with the site."""
        raise NotImplementedError("Syncing results is not implemented for Stampede3")

    def _sync_simulation(self, simulation):
        """Sync the simulation to the site."""
        raise NotImplementedError(
            "Syncing simulations is not implemented for Stampede3"
        )

    def submit_SLURM(
        self,
        job: SimulationJob,
        nodes: int,
        procs_per_node: int = 2,
        procs_per_task: Optional[int] = None,
        wait: bool = False,
        duration: str = "00-02:00:00",
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
            account:        TACC account to run on
            notify_on:      Notify on event
            notify_email:   Email address to notify
            run_path:       Path where the slurm job will be run
        """

        if procs_per_task is None:
            procs_per_task = max(1, (nodes * procs_per_node) // job.n_tasks)

        script = self._sweep_SLURM_script(
            n_tasks=job.n_tasks,
            n_nodes=nodes,
            stdout=str(job._remote_path(self.work_dir) / "logs"),
            procs_per_node=procs_per_node,
            procs_per_task=procs_per_task,
            duration=duration,
            imaging_job=isinstance(job, ImagingJob),
            **({"queue": queue} if queue is not None else {}),
            **({"account": account} if account is not None else {}),
            **({"notify_on": notify_on} if notify_on is not None else {}),
            **({"notify_email": notify_email} if notify_email is not None else {}),
            **({"run_path": run_path} if run_path is not None else {}),
            **kwargs,
        )

        if procs_per_node * nodes > procs_per_task * job.n_tasks:
            raise ValueError(
                f"Number of workers ({procs_per_node * nodes}) "
                f"is greater than number of tasks ({procs_per_task * job.n_tasks}), "
                "reduce the number of workers (by decreasing the "
                "n_nodes or processes_per_node) or increase the number of "
                "tasks."
            )

        if run_path is None:
            run_path = self.work_dir

        # Transfer job and script to Stampede3
        remote_script, remote_job = self._transfer_SLURM_job(script, job)

        cmd = f"mkdir -p {run_path}/jobs/batch && "
        cmd += "sbatch "
        if "slurm_args" in kwargs:
            for arg in kwargs["slurm_args"]:
                cmd += f"{arg} "
        cmd += f"{remote_script} {remote_job}"

        # Submit job to SLURM queue
        _, stdout, stderr = self.run_login_cmd(cmd)
        output = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if err:
            logger.error("sbatch error: %s", err)
            print(f"sbatch error: \033[91m{err}\033[0m")

        logger.debug("sbatch output: %s", output)
        job_id = re.search(r"Submitted batch job (\d+)", output)
        if not job_id:
            logger.error("Failed to retrieve job ID from sbatch output: %s", output)
            raise ValueError("failed to get job ID from sbatch output")
        job_id = job_id.group(1)

        print(
            f"Job {job_id} submitted successfully to Stampede3:{queue or self.config.queue}"
        )
        job._job_id = job_id

        return job_id

    def wait_completion(self, job: Union[SimulationJob, List[SimulationJob]]):
        """Wait for job to complete and download results."""

        status_colors = {
            "pending": "\033[38;5;27m",
            "running": "\033[38;5;28m",
            "complete": "\033[38;5;40m",
            "timeout": "\033[38;5;202m",
            "failed": "\033[38;5;160m",
            "cancelled": "\033[38;5;160m",
        }

        if isinstance(job, SimulationJob):
            jobs = [job]
            job_ids = [job._job_id]
        else:
            jobs = job
            job_ids = [j._job_id for j in job]

        statuses = {j_id: "pending" for j_id in job_ids}
        active_jobs = set(job_ids)
        name_width = max(len(job.name) for job in jobs)
        for job in jobs:
            j_id = job._job_id
            print(
                f"\033[38;5;244mJob {j_id} ({job.name:<{name_width}}): \033[38;5;27m{statuses[j_id].capitalize()}\033[0m"
            )

        while active_jobs:
            print(f"\033[{len(job_ids)}A", end="")
            active_ids = ",".join(active_jobs)
            cmd = f"sacct -j {active_ids} --format=JobID,State --noheader --parsable2"
            _, stdout, stderr = self.run_login_cmd(cmd)
            output = stdout.read().decode().strip()

            for line in output.split("\n"):
                if not line:
                    continue
                job_id, state = line.split("|")
                job_id = job_id.split(".")[0]  # Remove any array task IDs
                state = state.split(" ")[0]  # Remove info about cancelling user
                if job_id in active_jobs:
                    if state in ["PENDING", "CONFIGURING"]:
                        status = "pending"
                    elif state in ["RUNNING", "COMPLETING"]:
                        status = "running"
                    elif state in ["COMPLETED"]:
                        status = "complete"
                    elif state in ["FAILED", "NODE_FAIL", "PREEMPTED"]:
                        status = "failed"
                    elif state in ["TIMEOUT"]:
                        status = "timeout"
                    elif state in ["CANCELLED"]:
                        status = "cancelled"
                    else:
                        status = "unknown"
                    statuses[job_id] = status

            for job in jobs:
                j_id = job._job_id
                status = statuses[j_id]

                if j_id in active_jobs:
                    if status not in ["pending", "running"]:
                        active_jobs.remove(j_id)

                print(
                    f"\033[38;5;244mJob {j_id} ({job.name:<{name_width}}): {status_colors[status]}{status.capitalize()}\033[0m\033[K"
                )

            time.sleep(self.config.poll_interval)

        print()  # Final newline

        for job in jobs:
            j_id = job._job_id
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
                "This submit method requires the site to"
                "be provisioned (attached to a running job)."
                "Use submit_SLURM to queue a job."
            )

            # TODO: automatically compute job sizing and submit to queue

        loop = asyncio.get_event_loop()
        loop.create_task(monitor)
        return future

    def run_cmd(self, client, cmd: str):
        """Run a command using exec_command, passing the captured environment if available."""
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
        output = stdout.read()
        if isinstance(output, bytes):
            return output.decode().strip()
        return output.strip()

    def update_status(self, job_id: Optional[str] = None):
        """Check the status of the resource request."""

        # Map SLURM status codes to our status codes
        status_map = {
            "PD": "pending",
            "R": "running",
            "CG": "running",
            "CD": "complete",
            "F": "failed",
            "TO": "timeout",
            "CA": "cancelled",
        }

        if job_id is None:
            job_id = self.pool.id
            job_specified = False  # Checking status of pool
        else:
            job_specified = True  # Checking status of a specific job

        if job_id is None:
            self.pool._status.status = "unknown"

        # Get job status
        status = self.run_login(f"squeue -j {job_id} -h -o %t").strip()
        logger.debug("Job %s status from squeue: '%s'", job_id, status)

        # If no status returned, job is not in queue - check sacct for completion status
        if not status:
            sacct_cmd = f"sacct -j {job_id} -n -o State%20"
            completion_status = self.run_login(sacct_cmd).strip()
            logger.debug(
                "Job %s completion status from sacct: '%s'", job_id, completion_status
            )

            if "COMPLETED" in completion_status:
                status = "complete"
            elif "FAILED" in completion_status:
                status = "failed"
            elif "CANCELLED" in completion_status:
                status = "cancelled"
            elif "TIMEOUT" in completion_status:
                status = "timeout"
            else:
                status = "unknown"
        else:
            status = status_map.get(status, "unknown")

        if not job_specified:
            self.pool._status.status = status
        return status

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """
        Transfer files from local path to remote path on Stampede3.

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
                remote_str = f"{self.credentials.username}@stampede3.tacc.utexas.edu:{remote_path}"
                rsync_cmd = ["rsync", "-azP"]

                if local_path.is_dir():
                    local_str = f"{local_path}/"
                else:
                    local_str = str(local_path)

                cmd_str = " ".join([*rsync_cmd, local_str, remote_str])
                logger.info("Transferring via rsync: %s", cmd_str)

                result = subprocess.run(
                    [*rsync_cmd, local_str, remote_str], capture_output=True, text=True
                )

                if result.returncode != 0:
                    raise RuntimeError(f"rsync failed: {result.stderr}")

            logger.debug("Transfer completed successfully")

        except Exception as e:
            logger.exception("Error during file transfer: %s", str(e))
            raise

    def fetch_traces(
        self,
        job: Union[SimulationJob, List[SimulationJob]],
        upscale: int = 1,
    ) -> Union[RecordDatabase, Dict[str, RecordDatabase]]:
        """Get results from Stampede3.

        Args:
            job: A SimulationJob object.
        """

        if isinstance(job, SimulationJob):
            jobs = [job]
        else:
            jobs = job

        db_map = {}

        for j in jobs:
            try:
                remote_dir = j._remote_path(self.work_dir) / "results" / "receivers/"
                local_dir = j._local_path / "results" / "receivers/"
                local_dir.mkdir(parents=True, exist_ok=True)
                self.get(remote_dir, local_dir)

                db = RecordDatabase.from_job(j, upscale)
                db_map[j.name] = db

            except Exception as e:
                logger.exception("Error downloading records: %s", str(e))
                raise

        if len(db_map) == 1:
            return db_map[jobs[0].name]
        else:
            return db_map

    def fetch_paraview(self, job: SimulationJob):
        """Get paraview files from Stampede3.

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
        """Get image files from Stampede3.

        Args:
            job: An ImagingJob object.
        """

        if isinstance(job, ImagingJob):
            jobs = [job]
        else:
            jobs = job

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

        if len(images) == 1:
            return images[jobs[0].name]
        else:
            return images

    def download_record_files(self, records: dict, project_dir: Union[str, Path]):
        """Download records from Stampede3.

        Args:
            records: A dictionary of records to get.
            project_dir: The directory to download the records to.
        """
        project_dir = Path(project_dir)
        files = records["datasets"].keys()
        try:
            # Create temporary directory name for the payload
            payload_name = f"records_{int(time.time())}"
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

            cwd = os.getcwd()
            os.chdir(local_payload.parent)
            with tarfile.open(local_payload, "r:gz") as tar:
                logger.debug("Extracting files from payload:")
                tar.extractall()
            os.chdir(cwd)

            local_payload.unlink()
            self.run_login(f"rm {remote_payload}")
            return records

        except Exception as e:
            logger.exception("Error downloading records: %s", str(e))
            raise

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Transfer files from remote path to local path on Stampede3.

        Args:
            remote_path: Remote path to transfer to
            local_path: Local path to transfer from
            overwrite: Overwrite existing files
        """
        logger.debug("Attempting to transfer from %s to %s", remote_path, local_path)

        local_path = Path(local_path)
        remote_path = Path(remote_path)

        try:
            parent_path = str(local_path.parent)
            self.run_login(f"mkdir -p {parent_path}")

            if self.transfer_method == "sftp":
                logger.debug("Transferring %s to %s (SFTP)", remote_path, local_path)
                # Use SFTP through existing connection
                sftp = self.login_client.open_sftp()
                try:
                    if remote_path.is_dir():
                        self._get_dir(sftp, remote_path, local_path)
                    else:
                        sftp.get(str(remote_path), str(local_path))
                finally:
                    sftp.close()
            else:
                # Use rsync
                if remote_path.suffix == "":
                    remote_str = f"{remote_path}/"
                else:
                    remote_str = str(remote_path)
                remote_str = f"{self.credentials.username}@stampede3.tacc.utexas.edu:{remote_str}"
                local_str = f"{local_path}/" if local_path.is_dir() else str(local_path)
                rsync_cmd = ["rsync", "-azP"]
                logger.debug("rsync: %s", [*rsync_cmd, remote_str, local_str])

                result = subprocess.run(
                    [*rsync_cmd, remote_str, local_str],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.returncode != 0:
                    raise RuntimeError(f"rsync failed: {result.stderr}")

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
        self._connect_to_job_host(self.pool.id)

    def cancel_job(self, job_id: Optional[str] = None) -> bool:
        """Cancel a job."""
        if job_id is None:
            job_id = self._select_job()
        _, stdout, _ = self.run_login_cmd(f"scancel {job_id}")
        logger.info("Job %s cancelled: %s", job_id, stdout.read().decode().strip())

    def deprovision(self, **kwargs):
        """Release HPC resources."""
        self.cancel_job(self.pool.id)

    def _get_solver_path(self) -> str:
        """Get the solver path."""
        load_dotenv()
        executable = os.getenv("STAMPEDE3_SOLVER_EXECUTABLE")
        if executable is None:
            executable = "/work2/06472/jbadger/shared/stampede3/FS_stable/FS_seismic"
        return executable

    def _get_FS_path(self) -> str:
        """Get the Frequensolve path."""
        load_dotenv()
        path = Path(os.getenv("FS_PYTHON_PATH"))
        if not path.exists():
            raise FileNotFoundError(f"env var FS_PYTHON_PATH:{path} does not exist")
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
        """Transfer a SLURM job to Stampede3."""
        fd, script_path = tempfile.mkstemp(suffix=".slurm", prefix="sweep", dir="./")
        logger.debug("Temporary sweep script created at %s", script_path)
        with os.fdopen(fd, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o700)

        remote_script = (self.work_dir / "sweep").with_suffix(".slurm")
        self.put(Path(script_path), Path(remote_script))
        os.unlink(script_path)

        local_job, remote_job = job.save_for_remote(
            self.__class__.__name__, self.work_dir
        )

        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(local_job), Path(remote_job))
        self.run_login(f"chmod 700 {remote_script}")

        return remote_script, remote_job

    def _transfer_job(self, job: SimulationJob):
        """Submit a simulation job to Stampede3.

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

        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="sweep", dir="./")
        logger.debug("Temporary sweep script created at %s", script_path)
        with os.fdopen(fd, "w") as f:
            f.write(script)

        remote_script = (self.work_dir / "sweep").with_suffix(".sh")
        self.put(Path(script_path), Path(remote_script))
        os.unlink(script_path)

        self.run_login(f"chmod 700 {remote_script}")

        return remote_script, remote_job

    def _is_running(self, job_id: int):
        """Check if a job is running."""
        status = self.run_login(f"squeue -j {job_id} -h -o %t")
        return status == "R"

    def _sweep_script(self, job: SimulationJob, **kwargs) -> str:
        """Generate a scripte for sweeping through frequencies (tasks) on pre-provisioned resources."""

        n_tasks = job.n_tasks
        dir_out = str(job._remote_path(self.work_dir) / "logs")
        env_dir = self._FS_dir / "src/frequensolve/orchestrator/templates"
        env = Environment(
            loader=FileSystemLoader(env_dir),
            keep_trailing_newline=True,
        )
        template = env.get_template("sweep/sweep_SLURM.sh")
        script = template.render(
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
        return script

    def _sweep_SLURM_script(
        self,
        n_tasks: int,
        n_nodes: int,
        stdout: str,
        name: str = "FrequenSolve",
        procs_per_node: int = 2,
        procs_per_task: int = 2,
        duration: str = "00-02:00:00",
        queue: Optional[str] = None,
        account: Optional[str] = None,
        notify_on: Optional[Literal["begin", "end", "fail", "all", "none"]] = None,
        notify_email: Optional[str] = None,
        run_path: Optional[str] = None,
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
            queue:          Queue to run on (optional, defaults to site queue)
            account:        TACC account to run on (defaults to TACC_ACCOUNT env var if set)
            notify_on:      Notify on event (optional)
            notify_email:   Email address to notify (optional)
            **kwargs:       Additional keyword arguments
        """

        if account is None:
            account = self.config.account
        if queue is None:
            queue = self.config.queue
        config = Stampede3Config(queue)

        # Check that job fits in queue limits
        duration = config.validate_request(n_nodes, n_nodes * procs_per_node, duration)

        if run_path is None:
            run_path = self.work_dir

        # Load and populate Jinja2 template
        env_dir = self._FS_dir / "src/frequensolve/orchestrator/templates"
        env = Environment(
            loader=FileSystemLoader(env_dir),
            keep_trailing_newline=True,
        )
        template = env.get_template("sweep/sweep_SLURM.sh")
        script = template.render(
            batch_job=True,
            name=name,
            dir_out=stdout,
            n_nodes=n_nodes,
            n_procs=n_nodes * procs_per_node,
            n_threads=config.cores_per_node // procs_per_node,
            n_tasks=n_tasks,
            procs_per_task=procs_per_task,
            duration=duration,
            queue=queue,
            account=account,
            imaging_job=imaging_job,
            mpi=self.mpi_cmd,
            executable=self.executable,
            fs_dir=str(Path(self.executable).parent),
            **({"run_path": run_path} if run_path is not None else {}),
            **({"notify_on": notify_on.upper()} if notify_on is not None else {}),
            **({"notify_email": notify_email} if notify_email is not None else {}),
            **kwargs,
        )
        return script

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
        """Generate a script for provisioning a Stampede3 cluster.

        Args:
            n_nodes:        Number of nodes to provision
            procs_per_node: Number of processes per node
            duration:       Duration of the job (DD-HH:MM:SS)
            queue:          Queue to run on (optional, defaults to site queue)
            account:        TACC account to run on (defaults to TACC_ACCOUNT env var if set)
            notify_email:   Email address to notify (optional)
        """
        env_dir = self._FS_dir / "src/frequensolve/orchestrator/templates"
        env = Environment(
            loader=FileSystemLoader(env_dir),
            keep_trailing_newline=True,
        )
        template = env.get_template("provision/provision_SLURM.sh")
        name = kwargs.get("name", "FS_cluster")

        return template.render(
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
        """Gets the $WORK directory path on Stampede3."""
        work_dir = os.getenv("STAMPEDE3_WORK_DIR")

        # If $WORK is not set, try getting it from the login node
        if not work_dir or work_dir == "":
            _, stdout, stderr = self._login_client.client.exec_command("echo $WORK")
            work_dir = stdout.read().decode().strip()
            if not work_dir:
                raise RuntimeError(
                    "Failed to get $WORK directory path from Stampede3; you can work around "
                    "this by setting STAMPEDE3_WORK_DIR in your environment or .env file"
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
                f"cd {remote_dir.parent} && tar xzf {remote_dir.name}.tar.gz && rm {remote_dir.name}.tar.gz"
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

    def _print_interactive_output(self, interactive, timeout=5):
        """Monitor command output and update future accordingly."""
        try:
            output_buffer = ""
            error_buffer = ""

            start_time = time.time()
            while time.time() - start_time < timeout:
                if isinstance(interactive, subprocess.Popen):
                    if interactive.poll() is not None:
                        return

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

                # Process error buffer
                while "\n" in error_buffer:
                    line, error_buffer = error_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        print(f"\033[91m{line}\033[0m", flush=True)

                time.sleep(0.2)

        except Exception as e:
            logger.exception(f"Monitor exception: {e}")

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
                        future.set_result(job.records)
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
