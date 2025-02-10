"""
Frontera HPC site.

Manages authentication, transfer, and resource provisioning on Frontera.
"""

import asyncio
import os
import re
import socket
import subprocess
import tempfile
import time
from asyncio import Future, create_task
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from threading import Event, Thread
from typing import Literal, Optional, TextIO, Union

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from paramiko import (
    AuthenticationException,
    AutoAddPolicy,
    SSHClient,
    Transport,
)

from frequensolve.orchestrator.credentials import Credentials
from frequensolve.orchestrator.pool import PoolInfo, PoolStatus
from frequensolve.orchestrator.sites.base_site import BaseSite, _wait_for_path
from frequensolve.orchestrator.sites.frontera.config import (
    FronteraConfig,
    _hms_to_seconds,
    _seconds_to_hms,
)
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

__all__ = ["FronteraSite"]

# Initialize the logger
logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/frontera.log")


# ----------------------------------
# TACC Login Credentials
# ----------------------------------
class TACCLoginCredentials(Credentials):
    """Credentials for Frontera HPC."""

    user_env: str = "TACC_USERNAME"
    pw_env: str = "TACC_PASSWORD"
    ssh_key_env: str = "SSH_PASSPHRASE"


class SSHClientClass:
    """SSH client class."""

    def __init__(self, client: SSHClient):
        self.client = client
        _, stdout, stderr = self.client.exec_command("echo $HOSTNAME")
        self._hostname = stdout.read().decode().strip().split("@")[0]

    @property
    def hostname(self) -> str:
        """Get the hostname."""
        return self._hostname


# ----------------------------------
# Frontera Site
# ----------------------------------
@dataclass(kw_only=True, init=False)
class FronteraSite(BaseSite):
    """Frontera HPC site.

    Manages authentication, transfer, etc. to Frontera.
    """

    credentials: TACCLoginCredentials
    config: FronteraConfig
    pool: PoolInfo
    executable: str
    remote_env: dict
    transfer_method: Literal["rsync", "sftp"] = "rsync"
    _login_client: SSHClientClass
    _compute_client: Optional[SSHClientClass] = None
    _work_dir: Path
    _log_file: TextIO
    _FS_dir: Path

    def __init__(
        self,
        rel_path: Union[str, Path],
        transfer_method: Literal["rsync", "sftp"] = "rsync",
        queue: str = "debug",
    ):
        logger.info(
            "Initializing FronteraSite with rel_path: %s, queue: %s",
            rel_path,
            queue,
        )

        # Get Frontera credentials and configuration
        self.credentials = TACCLoginCredentials()
        self.config = FronteraConfig(queue=queue)
        self.transfer_method = transfer_method

        # SSH into Frontera
        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH client authenticated successfully")

        # Get work directory and solver path
        self._work_dir = self._get_work_dir(rel_path)
        self.executable = self._get_solver_path()
        self._FS_dir = self._get_FS_path()

        # Set remote environment
        self.remote_env = {"PWD": str(self._work_dir), "OLDPWD": str(self._work_dir)}

        self.pool = PoolInfo()
        self._is_notebook = self._check_if_notebook()

        logger.info(
            "FronteraSite initialized successfully with work_dir: %s", self._work_dir
        )

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

    def __enter__(self):
        logger.info("Entering FronteraSite context manager.")
        self.credentials = TACCLoginCredentials()
        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH Client re-established in context manager.")
        self._work_dir = self._get_work_dir()
        logger.info("Work directory re-set to: %s", self._work_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Exiting FronteraSite context manager.")
        if self._compute_client:
            self.compute_client.close()
            logger.debug("Compute client closed.")
        if self._login_client:
            self.login_client.close()
            logger.debug("SSH client closed.")

    def __del__(self):
        logger.info("Deleting FronteraSite instance and cleaning up resources.")
        if self._compute_client:
            self.compute_client.close()
            logger.debug("Compute client closed in __del__.")
        if self._login_client:
            self.login_client.close()
            logger.debug("SSH client closed in __del__.")

    def _get_solver_path(self) -> str:
        """Get the solver path."""
        load_dotenv()
        return os.getenv("FRONTERA_SOLVER_EXECUTABLE")

    def _get_FS_path(self) -> str:
        """Get the Frequensolve path."""
        load_dotenv()
        path = Path(os.getenv("FS_PYTHON_PATH"))
        if not path.exists():
            logger.error(
                "FS_PYTHON_PATH environment variable not set or path does not exist: %s",
                path,
            )
            raise FileNotFoundError(
                f"environment variable FS_PYTHON_PATH {path} does not appear to be set"
            )
        return path

    def authenticate(self, host: str = "frontera.tacc.utexas.edu"):
        """Connects to Frontera Login Node using Paramiko's built-in authentication mechanisms."""
        logger.info("Starting authentication with host: %s", host)
        login_client = SSHClient()
        login_client.set_missing_host_key_policy(AutoAddPolicy())

        # Create a direct socket connection to Frontera's SSH service.
        sock = socket.create_connection((host, 22))
        transport = Transport(sock)
        transport.start_client()

        # Attempt agent-based authentication.
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
                        logger.info("Agent-based authentication successful.")
                        break
                except Exception as err:
                    logger.debug("Agent key authentication failed: %s", str(err))
                    continue
        except Exception as err:
            logger.debug(
                "Agent-based authentication encountered an exception: %s", str(err)
            )

        # If agent authentication did not succeed, try password authentication.
        if not authenticated:
            try:
                logger.debug("Attempting password-based authentication.")
                transport.auth_password(
                    self.credentials.username, self.credentials.password
                )
                authenticated = transport.is_authenticated()
                if authenticated:
                    logger.info("Password-based authentication successful.")
                else:
                    logger.debug("Password-based authentication failed.")
            except Exception as err:
                logger.debug("Password-based authentication exception: %s", str(err))

        # If still not authenticated, attempt keyboard-interactive authentication.
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
                    logger.info("Keyboard-interactive authentication successful.")
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

        transport.set_keepalive(30)
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
        start_event = Event()

        try:
            remote_path = f"/tmp/{os.path.basename(script_path)}"
            logger.debug(
                "Transferring provision script to remote path: %s", remote_path
            )
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
                logger.debug(
                    "Starting monitor thread for job status (job ID: %s)", self.pool.id
                )
                while True:
                    status = self.run_login(f"squeue -j {self.pool.id} -h -o %t")
                    logger.debug("Current job status: %s", status)
                    if "R" in status:
                        self.pool._status.status = "running"
                        logger.info("Job %s is now running.", self.pool.id)
                        start_event.set()
                        break
                    elif "PD" in status:
                        self.pool._status.status = "pending"
                        logger.info("Job %s is still pending.", self.pool.id)
                    elif not status:
                        self.pool._status.status = "failed"
                        logger.error(
                            "Job %s not found in squeue; marking as failed.",
                            self.pool.id,
                        )
                        break
                    else:
                        logger.debug(
                            "Job %s status: %s. Waiting for next poll.",
                            self.pool.id,
                            status,
                        )
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

        If job_id is not provided, queued jobs will be listed and the user will be prompted to select a job.
        """
        if job_id is None:
            job_id = self._select_job()
        self.pool.id = job_id
        logger.info("Attaching to existing job with ID: %s", self.pool.id)

        self.update_status()

        if self.pool.is_running:
            logger.info("Job %s is running. Attaching...", self.pool.id)
            self._compute_client = SSHClientClass(
                self._connect_to_job_host(self.pool.id)
            )
            self._set_pool_info()

        elif self.pool.is_queued:
            logger.debug("Job %s is queued. Waiting for job to start...", self.pool.id)
            while True:
                self.update_status()
                if self.pool.is_running:
                    break
                if self.pool.is_complete:
                    logger.error(
                        "Job %s completed with status %s", job_id, self.pool.status
                    )
                    raise RuntimeError(
                        f"Job {job_id} completed with status {self.pool.status}"
                    )
                logger.debug(
                    "Current status of job %s: %s", self.pool.id, self.pool.status
                )
                time.sleep(self.config.poll_interval)
            if self.pool.is_running:
                self._compute_client = SSHClientClass(
                    self._connect_to_job_host(self.pool.id)
                )
                self._set_pool_info()
            else:
                logger.error("Job %s is not running after waiting", self.pool.id)
                raise RuntimeError(f"Job {self.pool.id} is not running")
        else:
            logger.error("Job %s is neither queued nor running", self.pool.id)
            raise RuntimeError(
                f"Job {self.pool.id} is not queued or running: {self.pool.status}"
            )

    def submit(self, job: SimulationJob) -> list:
        """Submit job and block until completion."""
        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        loop = asyncio.get_event_loop()
        future = self.submit_async(job)
        return loop.run_until_complete(future)

    def submit_async(self, job: SimulationJob) -> Future:
        """Submit job asynchronously and return a future."""
        remote_script, remote_job = self._transfer_job(job)

        nproc = self.pool.nproc
        nitems = job.n_tasks
        ntasks_per_item = max(2, nproc // nitems)

        _, stdout, stderr = self.run_compute_cmd(
            f"{remote_script} {remote_job} {ntasks_per_item}"
        )
        future = Future()

        async def monitor_output():
            try:
                logger.info("Monitoring output of the sweep job.")

                # Create async iterators for stdout/stderr
                stdout_iter = iter(stdout)
                stderr_iter = iter(stderr)

                while True:
                    # Check stdout
                    try:
                        line = next(stdout_iter).strip()
                        logger.info("Sweep output: %s", line)
                        if "Sweep Complete" in line:
                            future.set_result(job)
                            logger.info("Sweep job completed successfully")
                            return
                    except StopIteration:
                        break

                    # Check stderr
                    try:
                        error = next(stderr_iter).strip()
                        logger.error("Sweep error: %s", error)
                        future.set_exception(RuntimeError(f"Sweep job failed: {error}"))
                        return
                    except StopIteration:
                        pass

                    # Give other tasks a chance to run
                    await asyncio.sleep(1)

                # If we get here without setting result/exception
                future.set_exception(
                    RuntimeError("Sweep job ended without completion marker")
                )

            except Exception as e:
                logger.exception("Error in monitor task")
                future.set_exception(e)

        # Create and start the monitor task
        create_task(monitor_output())
        return future

    def provision_dask(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ) -> Client:
        logger.info(
            "Provisioning Dask cluster with nhost=%d, nproc=%d, duration=%s",
            nhost,
            nproc,
            duration,
        )
        duration = self.config.validate_request(nhost, nproc, duration)
        cores_per_node = self.config.cores_per_socket * self.config.sockets_per_node
        cores_per_proc = cores_per_node * nhost // nproc
        logger.debug(
            "Calculated cores_per_node=%d, cores_per_proc=%d",
            cores_per_node,
            cores_per_proc,
        )
        cluster = SLURMCluster(
            queue=self.config.queue,
            account=self.config.account,
            processes=nproc,
            cores=cores_per_proc,
            walltime=duration,
            job_extra=[
                f"--nodes={nhost}",
                f"--ntasks-per-node={nproc // nhost}",
            ],
            **kwargs,
        )
        logger.info("SLURMCluster created; scaling cluster with 1 job.")
        cluster.scale(jobs=1)
        return Client(cluster)

    @property
    def mpi_cmd(self) -> str:
        """Get the MPI command for Frontera."""
        return f"{self.config.mpi_wrapper}"

    @cached_property
    def pool_host(self) -> str:
        """Get the root host of the resource pool."""
        return self._get_job_host(self.pool.id)

    @property
    def work_dir(self) -> Path:
        """Gets the $WORK directory path on Frontera."""
        return self._work_dir

    def run_cmd(self, client, cmd: str):
        """Run a command using exec_command."""
        logger.info("Executing command on %s: %s", client.hostname, cmd)
        return client.client.exec_command(cmd)

    def run_compute_cmd(self, cmd: str):
        """Run a command on compute node using exec_command."""
        return self.run_cmd(self._compute_client, cmd)

    def run_login_cmd(self, cmd: str):
        """Run a command on login node using exec_command."""
        return self.run_cmd(self._login_client, cmd)

    def run_compute(self, cmd: str) -> str:
        """Run a command on compute nodeand return its stdout as a stripped string."""
        _, stdout, _ = self.run_compute_cmd(cmd)
        return stdout.read().decode().strip()

    def run_login(self, cmd: str) -> str:
        """Run a command on login node and return its stdout as a stripped string."""
        _, stdout, _ = self.run_login_cmd(cmd)
        return stdout.read().decode().strip()

    def update_status(self):
        """Check the status of the resource request."""
        job_id = self.pool.id
        if job_id is None:
            return PoolStatus("unknown", -1, "", "No job ID found")
        try:
            _, stdout, stderr = self.run_login_cmd(f"squeue -j {job_id} -h -o %t")
            status = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            exit_code = stdout.channel.recv_exit_status()

            self.pool._status.return_code = exit_code

            if exit_code != 0:
                self.pool._status.status = "unknown"
                self.pool._status.stdout = status
                self.pool._status.stderr = error
                return

            if status == "R":
                self.pool._status.status = "running"
            elif status == "PD":
                self.pool._status.status = "pending"
            else:
                self.pool._status.status = "completed"
        except Exception as e:
            return PoolStatus("failed", -1, "", str(e))

    def wait_provisioned(self):
        """Wait for the job to be provisioned."""
        while True:
            self.update_status()
            if self.pool.is_running:
                break
            time.sleep(self.config.poll_interval)
        self._jump_to_pool_host()

    def cancel_job(self, job_id: Optional[str] = None) -> bool:
        """Cancel a job."""
        if job_id is None:
            job_id = self._select_job()
        _, stdout, _ = self.run_login_cmd(f"scancel {job_id}")
        logger.info("Job %s cancelled: %s", job_id, stdout.read().decode().strip())

    def deprovision(self, **kwargs):
        """Release HPC resources."""
        self.cancel_job(self.pool.id)

    def _set_pool_info(self):
        """Get information about the pool."""
        logger.info("Getting pool information for job %s", self.pool.id)

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
            print("\nAttatched to job:")
            print(jobs)
            return jobs.split()[0]
        else:
            job_id = input("Enter job ID: ")
            return int(job_id)

    def _transfer_job(self, job: SimulationJob):
        """Submit a simulation job to Frontera.

        Args:
            job (SimulationJob): The simulation job to submit
        """
        if self._compute_client is None:
            logger.error("Batch sweep job not implemented yet.")
            raise RuntimeError("Batch sweep job not implemented yet.")

        script = self._sweep_script(job.n_tasks)

        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="sweep", dir="./")
        os.chmod(script_path, 0o700)
        logger.debug("Temporary sweep script created at %s", script_path)
        with os.fdopen(fd, "w") as f:
            f.write(script)

        remote_script = (self.work_dir / "sweep").with_suffix(".sh")
        logger.debug("Transferring sweep script to remote path: %s", remote_script)
        self.put(Path(script_path), Path(remote_script))
        os.unlink(script_path)
        logger.debug("Temporary sweep script %s removed", script_path)

        file = job.save_for_remote(self.work_dir)
        remote_job = ((self.work_dir / "jobs") / job.name).with_suffix(".json")
        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(file), Path(remote_job))

        return remote_script, remote_job

    def _is_running(self, job_id: int):
        """Check if a job is running."""
        status = self.run_login(f"squeue -j {job_id} -h -o %t")
        return status == "R"

    def _sweep_script(self, n_tasks: int, **kwargs) -> str:
        """Generate a sweep script."""
        env = Environment(
            loader=FileSystemLoader(
                self._FS_dir / "src/frequensolve/orchestrator/templates"
            ),
            keep_trailing_newline=True,
        )
        template = env.get_template("sweep/sweep_SLURM.sh")
        script = template.render(
            batch_job=False,
            nrank=self.pool.nproc,
            nthread=self.config.cores_per_socket,
            njob=n_tasks,
            mpi=self.mpi_cmd,
            executable=self.executable,
            fs_dir=str(Path(self.executable).parent),
            project_dir=str(self.work_dir),
            slurm_nodelist=self.pool.hostnode,
            **kwargs,
        )
        return script

    def _generate_provision_script(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(
                self._FS_dir / "src/frequensolve/orchestrator/templates"
            ),
            keep_trailing_newline=True,
        )
        template = env.get_template("provision/provision_SLURM.sh")

        name = kwargs.get("name", "FS_cluster")

        return template.render(
            name=name,
            nhost=nhost,
            nproc=nproc,
            queue=self.config.queue,
            account=self.config.account,
            duration=duration,
            work_dir=self.work_dir,
            mpi=self.config.mpi_wrapper,
        )

    def _connect_to_job_host(self, job_id: int):
        """Connect to the job host."""

        job_host = self._get_job_host(job_id)
        transport = self.login_client.get_transport()

        # Create tunnel to job host
        channel = transport.open_channel("direct-tcpip", (job_host, 22), ("", 0))

        # Create new SSH client for job host
        job_client = SSHClient()
        job_client.set_missing_host_key_policy(AutoAddPolicy())

        try:
            # Connect to job host
            job_client.connect(
                job_host, username=self.credentials.username, sock=channel
            )
            logger.info("Connected to job host: %s", job_host)
            return job_client

        except Exception as e:
            logger.error("Failed to connect to job host: %s", str(e))
            channel.close()
            raise

    def _get_job_host(self, job_id: int) -> int:
        status = self.run_login(f"squeue -j {job_id} -h -o %t")
        if status != "R":
            raise RuntimeError(f"Job {job_id} is not running")

        return self.run_login(f"squeue -j {job_id} -h -o %B")

    def _get_work_dir(self, rel_proj_path: Union[str, Path]) -> Path:
        """Gets the $WORK directory path on Frontera."""
        _, stdout, stderr = self._login_client.client.exec_command("echo $WORK")
        work_dir = stdout.read().decode().strip()
        if not work_dir:
            raise RuntimeError("Failed to get $WORK directory path from Frontera")
        error = stderr.read().decode().strip()
        if error or ('is mounted not "FULL" nor "IDLE"' in work_dir):
            logger.info("Error getting $WORK directory path from Frontera: %s", error)
            self._work_dir = Path("$WORK") / rel_proj_path
        else:
            self._work_dir = Path(work_dir) / rel_proj_path
        logger.info("Work directory: %s", self._work_dir)
        return self._work_dir

    def _get_free_port(self) -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """Transfer files from local path to remote path on Frontera.

        Args:
            local_path: Local path to transfer from
            remote_path: Remote path to transfer to
        """
        logger.info("Attempting to transfer from %s to %s", local_path, remote_path)

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
                # Use SFTP through existing connection
                logger.debug("Using SFTP for file transfer")
                sftp = self.login_client.open_sftp()
                try:
                    if local_path.is_dir():
                        logger.info(
                            "Transferring directory %s to %s", local_path, remote_path
                        )
                        self._put_dir(sftp, local_path, remote_path)
                    else:
                        logger.info(
                            "Transferring file %s to %s", local_path, remote_path
                        )
                        sftp.put(str(local_path), str(remote_path))
                finally:
                    sftp.close()
            else:
                # Use rsync
                logger.debug("Using rsync for file transfer")
                remote_str = f"{self.credentials.username}@frontera.tacc.utexas.edu:{remote_path}"
                rsync_cmd = ["rsync", "-azP"]

                if local_path.is_dir():
                    local_str = f"{local_path}/"
                    logger.info(
                        "Transferring directory %s to %s", local_path, remote_path
                    )
                else:
                    local_str = str(local_path)
                    logger.info("Transferring file %s to %s", local_path, remote_path)

                result = subprocess.run(
                    [*rsync_cmd, local_str, remote_str], capture_output=True, text=True
                )

                if result.returncode != 0:
                    logger.error("rsync failed with output: %s", result.stderr)
                    raise RuntimeError(f"rsync failed: {result.stderr}")

            logger.info("Transfer completed successfully")

        except Exception as e:
            logger.exception("Error during file transfer: %s", str(e))
            raise

    def _put_dir(self, sftp, local_dir: Path, remote_dir: Path):
        """Transfer directory by compressing, sending single file, then extracting."""
        import tarfile
        import tempfile

        # Create temporary tar file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            # Compress directory
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(local_dir, arcname=local_dir.name)

            # Upload compressed file
            remote_tar = str(remote_dir.parent / f"{remote_dir.name}.tar.gz")
            sftp.put(tmp.name, remote_tar)

            # Extract on remote side
            _, stdout, stderr = self.run_login_cmd(
                f"cd {remote_dir.parent} && tar xzf {remote_dir.name}.tar.gz && rm {remote_dir.name}.tar.gz"
            )

            # Check for errors
            err = stderr.read().decode().strip()
            if err:
                logger.error("Error extracting directory on remote: %s", err)
                raise RuntimeError(f"Failed to extract directory on remote: {err}")

    # def _decode_hosts(self, hosts: str) -> List[str]:
    #     """Decode the hosts string into a list of hostnames."""
    #     i = 0
    #     while i > -1:
    #         i = hosts.find("[")
    #         j = hosts.find("]", i)
    #         prefix = hosts[i - 5 : i]
    #         old_str = prefix + hosts[i : j + 1]
    #         suffixes = []

    #         group = hosts[i + 1 : j - 1]
    #         for entries in group.split(","):
    #             if "-" in entries:
    #                 start, end = entries.split("-")
    #                 for i in range(int(start), int(end) + 1):
    #                     suffixes.append(f"{str(i).zfill(len(start))}")
    #             else:
    #                 suffixes.append(entries)

    #         for suffix in suffixes:
    #             new_str += f"{prefix}{suffix},"

    #         new_hosts = hosts.replace(old_str, new_str[:-1])
    #     return new_hosts.split(",")
