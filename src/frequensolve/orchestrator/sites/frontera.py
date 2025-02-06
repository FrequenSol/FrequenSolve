"""Frontera HPC site.

Manages authentication, transfer, and resource provisioning on Frontera.
"""

import getpass
import os
import re
import socket
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from threading import Event, Thread
from typing import List, Optional, TextIO, Union

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from paramiko import (
    AuthenticationException,
    AutoAddPolicy,
    PasswordRequiredException,
    RSAKey,
    SFTPClient,
    SSHClient,
    Transport,
)
from pexpect import EOF, TIMEOUT, spawn

from frequensolve.orchestrator.sites.base_site import BaseSite, BaseSiteConfig
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.printing import print_warn

__all__ = ["FronteraSite"]


# ----------------------------------
# Frontera queue and machine info
# ----------------------------------
@dataclass(frozen=True)
class _BaseQueue:
    """Base class for Frontera queues."""

    _name: str
    _max_duration: str
    _max_nodes: int
    _min_nodes: int


@dataclass(frozen=True)
class _LargeQueue(_BaseQueue):
    """Frontera Large queue."""

    _name: str = "large"
    _max_duration: str = "2-00:00:00"
    _max_nodes: int = 2048
    _min_nodes: int = 513


@dataclass(frozen=True)
class _NormalQueue(_BaseQueue):
    """Frontera Normal queue."""

    _name: str = "normal"
    _max_duration: str = "2-00:00:00"
    _max_nodes: int = 512
    _min_nodes: int = 4


@dataclass(frozen=True)
class _DebugQueue(_BaseQueue):
    """Frontera Debug queue."""

    _name: str = "debug"
    _max_duration: str = "02:00:00"
    _max_nodes: int = 40
    _min_nodes: int = 1


@dataclass(frozen=True)
class _FronteraBaseConfig(BaseSiteConfig):
    _hostname: str = "frontera.tacc.utexas.edu"
    _scheduler: str = "SLURM"
    _mpi_wrapper: str = "ibrun"
    _poll_interval: int = 5
    _sockets_per_node: int = 2
    _gpus_per_node: int = 0
    _cores_per_socket: int = 28
    _memory_per_node: int = 198000
    _account: str = field(default_factory=lambda: os.getenv("TACC_ACCOUNT", ""))


@dataclass
class FronteraConfig:
    """Combines immutable base configuration with queue info for Frontera."""

    _queue: _BaseQueue
    _base_config: _FronteraBaseConfig = _FronteraBaseConfig()

    def __init__(self, queue: str = "debug"):
        if queue == "debug":
            self._queue = _DebugQueue()
        elif queue == "normal":
            self._queue = _NormalQueue()
        elif queue == "large":
            self._queue = _LargeQueue()
        else:
            raise ValueError(f"Invalid queue: {queue}")

    @property
    def hostname(self):
        return self._base_config._hostname

    @property
    def scheduler(self):
        return self._base_config._scheduler

    @property
    def mpi_wrapper(self):
        return self._base_config._mpi_wrapper

    @property
    def poll_interval(self):
        return self._base_config._poll_interval

    @property
    def sockets_per_node(self):
        return self._base_config._sockets_per_node

    @property
    def gpus_per_node(self):
        return self._base_config._gpus_per_node

    @property
    def cores_per_socket(self):
        return self._base_config._cores_per_socket

    @property
    def memory_per_node(self):
        return self._base_config._memory_per_node

    @property
    def account(self):
        return self._base_config._account

    @property
    def queue(self):
        return self._queue._name

    @property
    def max_duration(self):
        return self._queue._max_duration

    @property
    def max_nodes(self):
        return self._queue._max_nodes

    @property
    def min_nodes(self):
        return self._queue._min_nodes

    def validate_request(self, nhost: int, nproc: int, duration: str):
        """Checks that request is within queue parameters"""
        if nhost < self.min_nodes:
            raise ValueError(f"Minimum number of nodes is {self.min_nodes}")
        if nhost > self.max_nodes:
            raise ValueError(f"Maximum number of nodes is {self.max_nodes}")
        if nproc < nhost:
            raise ValueError(f"Number of processes per node must be at least {nhost}")
        return self._validate_duration(duration)

    def _validate_duration(self, duration: str) -> str:
        """Validate duration."""
        duration_secs = _hms_to_seconds(duration)
        max_duration_secs = _hms_to_seconds(self.max_duration)

        if duration_secs > max_duration_secs:
            print_warn(
                f"Requested duration {duration} exceeds maximum allowed duration"
                f"({self.max_duration}), using {self.max_duration} instead"
            )
            duration = self.max_duration
        return duration


# ----------------------------------
# TACC Login Credentials
# ----------------------------------
class TACCLoginCredentials:
    """Credentials for Frontera HPC."""

    def __init__(self):
        load_dotenv()

    @cached_property
    def username(self):
        user = os.getenv("TACC_USERNAME")
        if user is None:
            print(
                "Avoid providing this each time by adding the TACC_USERNAME to FrequenSolve/.env"
            )
            user = input("TACC Username:")
        return user

    @cached_property
    def password(self):
        pw = os.getenv("TACC_PASSWORD")
        if pw is None:
            print(
                "Avoid providing this each time by adding the TACC_PASSWORD to FrequenSolve/.env"
            )
            pw = input("TACC Password:")
        return pw

    @cached_property
    def ssh_key(self):
        filename = os.path.expanduser("~/.ssh/id_rsa")
        try:
            return RSAKey.from_private_key_file(filename)
        except PasswordRequiredException:
            passphrase = self._ssh_passphrase
            return RSAKey.from_private_key_file(filename, password=passphrase)

    @cached_property
    def _ssh_passphrase(self):
        passphrase = os.getenv("SSH_PASSPHRASE")
        if passphrase is None:
            print(
                "Avoid providing this each time by adding the ssh_key to FrequenSolve/.env"
            )
            passphrase = getpass.getpass(f"SSH key passphrase: ")
        return passphrase

    @property
    def duo_code(self):
        return input("TACC 2FA Code:")

    def __str__(self):
        """Don't print credentials."""
        return ""

    def __repr__(self):
        """Don't print credentials."""
        return ""


@dataclass
class PoolStatus:
    status: str = "unknown"
    return_code: int = -1
    stdout: str = ""
    stderr: str = ""

    @property
    def is_queued(self) -> bool:
        return self.status == "pending"

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def is_complete(self) -> bool:
        return self.status in ["completed", "failed"]


@dataclass
class PoolInfo:
    """Information about the resource pool."""

    id: int = 0
    hostnode: str = ""
    nhost: int = 0
    ncore: int = 0
    start_time: str = ""
    end_time: str = ""
    _status: PoolStatus = field(default_factory=PoolStatus)

    @property
    def status(self):
        return self._status.status

    @property
    def is_queued(self) -> bool:
        return self._status.is_queued

    @property
    def is_running(self) -> bool:
        return self._status.is_running

    @property
    def is_complete(self) -> bool:
        return self._status.is_complete


def _hms_to_seconds(hms: str) -> int:
    """Convert D-HH:MM:SS to seconds."""
    d, tmp = map(int, hms.split("-"))
    h, m, s = map(int, tmp.split(":"))
    return d * 86400 + h * 3600 + m * 60 + s


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
    _compute: bool = False
    _ssh_client: SSHClient
    _sftp_client: SFTPClient
    _work_dir: Path
    _log_file: TextIO

    def __init__(
        self,
        rel_path: Union[str, Path],
        queue: str = "debug",
        log_file: str = "/tmp/FS_ssh.log",
    ):

        self._log_file = open(log_file, "wb")

        self.credentials = TACCLoginCredentials()
        self.config = FronteraConfig(queue=queue)
        self._ssh_client = self.login()
        self._sftp_client = self._ssh_client.open_sftp()
        self._work_dir = self._get_work_dir(rel_path)
        self.executable = self._get_solver_path()
        self.pool = PoolInfo()

        load_dotenv()
        self._FS_dir = Path(os.getenv("FS_PYTHON_PATH"))
        if not self._FS_dir.exists():
            raise FileNotFoundError(
                f"environment variable FS_PYTHON_PATH {self._FS_dir} does not appear to be set"
            )

    def _get_solver_path(self) -> str:
        """Get the solver path."""
        load_dotenv()
        return os.getenv("FRONTERA_SOLVER_EXECUTABLE")

    def __enter__(self):
        self.credentials = TACCLoginCredentials()
        self._ssh_client = self.login()
        self._sftp_client = self._ssh_client.open_sftp()
        self._work_dir = self._get_work_dir()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._sftp_client:
            self._sftp_client.close()
        if self._ssh_client:
            self._ssh_client.close()
        if self._log_file is not None:
            self._log_file.close()

    def __del__(self):
        if self._sftp_client:
            self._sftp_client.close()
        if self._ssh_client:
            self._ssh_client.close()
        if self._log_file is not None:
            self._log_file.close()

    def login(self):
        """Connects to Frontera Login Node using Paramiko's built-in authentication mechanisms."""

        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(AutoAddPolicy())

        # Create a direct socket connection to Frontera's SSH service.
        sock = socket.create_connection(("frontera.tacc.utexas.edu", 22))
        transport = Transport(sock)
        transport.start_client()

        # Attempt agent-based authentication.
        authenticated = False
        try:
            from paramiko.agent import Agent

            agent = Agent()
            agent_keys = agent.get_keys()
            for key in agent_keys:
                try:
                    transport.auth_publickey(self.credentials.username, key)
                    if transport.is_authenticated():
                        authenticated = True
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # If agent authentication did not succeed, try password authentication.
        if not authenticated:
            try:
                transport.auth_password(
                    self.credentials.username, self.credentials.password
                )
                authenticated = transport.is_authenticated()
            except Exception:
                pass

        # If still not authenticated, attempt keyboard-interactive authentication.
        if not authenticated:

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
            except Exception:
                pass

        if not transport.is_authenticated():
            raise AuthenticationException("Authentication failed.")

        ssh_client._transport = transport  # Bind transport to SSH client
        print("Frontera connection established")
        return ssh_client

    def attach_to_existing_job(self, job_id: Optional[str] = None):
        """Attach to an existing job.

        If job_id is not provided, queued jobs will be listed and the user will be prompted to select a job.
        """
        if job_id is None:
            job_id = self._select_job()
        self.pool.id = job_id

        self.update_status()

        if self.pool.is_running:
            print(f"Attaching to running job: {self.pool.id}")
            self._jump_to_pool_host()

        elif self.pool.is_queued:
            print("Waiting on job to start...")
            while True:
                self.update_status()
                if self.pool.is_running:
                    break
                if self.pool.is_complete:
                    raise RuntimeError(
                        f"Job {job_id} completed with status {self.pool.status}"
                    )
                print(f"Status: {self.pool.status}")
                time.sleep(self.config.poll_interval)
            if self.pool.is_running:
                self._jump_to_pool_host()
            else:
                raise RuntimeError(f"Job {self.pool.id} is not running")
        else:
            raise RuntimeError(f"Job {self.pool.id} is not queued or running")

    def submit(self, job: SimulationJob):
        """Submit a job."""
        if self._compute:
            script = self._sweep_script(job.n_tasks)
        else:
            NotImplementedError("Batch sweep job not implemented")

        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="sweep", dir="./")
        with os.fdopen(fd, "w") as f:
            f.write(script)

        remote_path = (self.work_dir / "sweep").with_suffix(".sh")
        self.put(Path(script_path), Path(remote_path))
        os.unlink(script_path)

        file = job.save()
        remote_file = ((self.work_dir / "jobs") / job.name).with_suffix(".json")
        self.put(Path(file), Path(remote_file))

        stdin, stdout, stderr = self.run_cmd(f"./sweep.sh")
        complete_event = Event()

        def monitor_output():
            for line in stdout:
                print(line.strip())
            complete_event.set()

        Thread(target=monitor_output, daemon=True).start()
        return complete_event

    def run_cmd(self, cmd: str):
        """Shorthand for calling client exec_command."""
        stdin, stdout, stderr = self._ssh_client.exec_command(cmd)
        return stdin, stdout, stderr

    def provision(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ):
        """Create a SLURM resource request.

        Args:
           nhost:    Number of hosts to request
           nproc:    Number of processes per host
           duration: Job duration in HH:MM:SS format
           **kwargs: Additional arguments passed to sbatch

        Returns:
           Event: Event that is set when the job starts running
        """
        duration = self.config.validate_request(nhost, nproc, duration)
        script = self._generate_provision_script(nhost, nproc, duration, **kwargs)
        fd, script_path = tempfile.mkstemp(
            suffix=".sh", prefix="slurm_", dir=self.work_dir
        )
        with os.fdopen(fd, "w") as f:
            f.write(script)

        start_event = Event()

        try:
            remote_path = f"/tmp/{os.path.basename(script_path)}"
            self.put(script_path, remote_path)

            _, stdout, stderr = self._ssh_client.exec_command(f"sbatch {remote_path}")

            # Get job ID
            job_id = re.search(r"Submitted batch job (\d+)", stdout.read().decode())
            if not job_id:
                raise ValueError("failed to get job ID from sbatch output")
            job_id = job_id.group(1)

            self.pool.id = job_id
            self.pool._status.status = "pending"

            # Monitor job status and get compute node
            def monitor_remote_pool():
                while True:
                    _, stdout, _ = self._ssh_client.exec_command(
                        f"squeue -j {self.pool.id} -h -o %t"
                    )
                    status = stdout.read().decode().strip()
                    if "R" in status:
                        self.pool._status.status = "running"
                        start_event.set()
                        break
                    elif "PD" in status:
                        self.pool._status.status = "pending"
                        break
                    elif not status:
                        self.pool._status.status = "failed"
                        break
                    else:
                        time.sleep(self.config.poll_interval)

            Thread(target=monitor_remote_pool, daemon=True).start()

        except Exception as e:
            self.pool._status.status = "failed"
            self.pool._status.stderr = str(e)
            print(e)
        finally:
            os.unlink(script_path)

        return start_event

    def provision_dask(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ) -> Client:
        """Provision resources using Dask JobQueue."""
        duration = self.config.validate_request(nhost, nproc, duration)
        cores_per_node = self.config.cores_per_socket * self.config.sockets_per_node
        cores_per_proc = cores_per_node * nhost // nproc

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
            # job_script_prologue=[f"{self.config.mpi_wrapper} flux start --boot"],
            **kwargs,
        )
        cluster.scale(jobs=1)
        return Client(cluster)

    # TODO: move this to file manager; just pass ssh & sftp clients
    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """Sends files from local path to remote path on Frontera."""
        if not self._wait_for_path(local_path):
            raise FileNotFoundError(
                f"Local path {local_path} does not exist after waiting"
            )

        local_path = Path(local_path)
        remote_path = Path(remote_path)

        try:
            if local_path.is_file():
                parent_path = str(remote_path.parent)
                self._ssh_client.exec_command(f"mkdir -p {parent_path}")
                self._sftp_client.put(str(local_path), str(remote_path))

            elif local_path.is_dir():
                with tempfile.NamedTemporaryFile(
                    suffix=".tar.gz", delete=False
                ) as temp_tar:
                    tar_path = Path(temp_tar.name)
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(str(local_path), arcname=local_path.name)

                remote_archive = remote_path / (local_path.name + ".tar.gz")
                self._ssh_client.exec_command(f"mkdir -p {remote_path}")
                self._sftp_client.put(str(tar_path), str(remote_archive))

                untar_command = f"tar -xzvf {remote_archive} -C {remote_path}"
                stdin, stdout, stderr = self._ssh_client.exec_command(untar_command)
                exit_status = stdout.channel.recv_exit_status()
                if exit_status != 0:
                    error = stderr.read().decode()
                    raise IOError(f"Error untarring remote archive: {error}")

                self._ssh_client.exec_command(f"rm -f {remote_archive}")
                tar_path.unlink()

        except Exception as e:
            print(e)

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

    @property
    def on_login_node(self) -> bool:
        """Check if the user is on the login node."""
        _, stdout, _ = self._ssh_client.exec_command("echo $HOSTNAME")
        hostname = stdout.read().decode().strip()
        return hostname.startswith("login")

    @property
    def on_compute_node(self) -> bool:
        """Check if the user is on the login node."""
        _, stdout, _ = self._ssh_client.exec_command("echo $HOSTNAME")
        hostname = stdout.read().decode().strip()
        return re.match(r"c\d{3}-\d{3}", hostname)

    def update_status(self):
        """Check the status of the resource request."""
        job_id = self.pool.id
        if job_id is None:
            return PoolStatus("unknown", -1, "", "No job ID found")
        try:
            _, stdout, stderr = self._ssh_client.exec_command(
                f"squeue -j {job_id} -h -o %t"
            )
            status = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            exit_code = stdout.channel.recv_exit_status()

            self.pool._status.return_code = exit_code

            if exit_code != 0:
                self.pool._status.status = "unknown"
                self.pool._status.stdout = stdout
                self.pool._status.stderr = stderr
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
        _, stdout, _ = self._ssh_client.exec_command(f"scancel {job_id}")
        print(stdout.read().decode())

    def deprovision(self, **kwargs):
        """Release HPC resources."""
        if self.config.local:
            subprocess.run(["scancel", self.pool.id])
        else:
            stdin, stdout, stderr = self._ssh_client.exec_command(
                f"scancel {self.pool.id}"
            )

    def _set_pool_info(self):
        """Get information about the pool."""
        _, stdout, _ = self._ssh_client.exec_command(
            f'squeue -j {self.pool.id} -h --format="%.10B %.8D %.10C %.20S %.20e"'
        )
        host, nhost, ncore, start_time, end_time = (
            stdout.read().decode().strip().split()
        )
        nhost = int(nhost)
        ncore = int(ncore)

        self.pool.hostnode = host
        self.pool.nhost = nhost
        self.pool.ncore = ncore
        self.pool.start_time = start_time
        self.pool.end_time = end_time

    def _list_jobs(self):
        """List all queued jobs."""
        _, stdout, _ = self._ssh_client.exec_command(
            f'squeue -u {self.credentials.username} -h --format="%.10i %.10B %.5D %.4t %.10L"'
        )
        jobs = stdout.read().decode().strip()
        print(f" --- User Jobs ---\n {jobs}")

    def _select_job(self):
        """Select a job from the list of jobs."""
        self._list_jobs()
        job_id = input("Enter job ID: ")
        return int(job_id)

    def _is_running(self, job_id: int):
        """Check if a job is running."""
        _, stdout, _ = self._ssh_client.exec_command(f"squeue -j {job_id} -h -o %t")
        status = stdout.read().decode().strip()
        return status == "R"

    # TODO: This is temporary until Dask is set up
    def _sweep_script(
        self,
        n_tasks: int,
        n_nodes: Optional[int] = None,
        duration: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a sweep script."""
        env = Environment(
            loader=FileSystemLoader(
                self._FS_dir / "src/frequensolve/orchestrator/templates"
            )
        )
        template = env.get_template("sweep/sweep_SLURM.sh")

        # if self.on_login_node:
        #     return template.render(
        #         batch_job=True,
        #         nnode=n_nodes,
        #         nrank=2*n_nodes,
        #         nthread=self.config.cores_per_socket,
        #         ntask=n_tasks,
        #         queue=self.config.queue,
        #         account=self.config.account,
        #         duration=duration,
        #         mpi=self.mpi_cmd,
        #         executable=self.executable,
        #         **kwargs,
        #     )
        # else:
        script = template.render(
            batch_job=False,
            nrank=2 * self.pool.nhost,
            nthread=self.config.cores_per_socket,
            ntask=n_tasks,
            mpi=self.mpi_cmd,
            executable=self.executable,
            **kwargs,
        )
        return script

    def _generate_provision_script(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(
                self._FS_dir / "src/frequensolve/orchestrator/templates"
            )
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

    def _jump_to_pool_host(self):
        """SSH from login node to the pool host."""
        self._compute = True
        self._set_pool_info()
        self.run_cmd(f"cd {self.work_dir}")
        self._ssh_client.exec_command(f"ssh {self.pool_host}")

    def _jump_to_job_host(self, job_id: int):
        """SSH from login node to the first node in a job."""
        self._compute = True
        self._set_pool_info()
        self.run_cmd(f"cd {self.work_dir}")
        self._ssh_client.exec_command(f"ssh {self._get_job_host(job_id)}")

    def _get_job_host(self, job_id: int) -> int:
        _, stdout, _ = self._ssh_client.exec_command(f"squeue -j {job_id} -h -o %t")
        status = stdout.read().decode().strip()
        if status != "R":
            raise RuntimeError(f"Job {job_id} is not running")

        _, stdout, _ = self._ssh_client.exec_command(f"squeue -j {job_id} -h -o %B")
        host = stdout.read().decode().strip()
        return host

    def _decode_hosts(self, hosts: str) -> List[str]:
        """Decode the hosts string into a list of hostnames."""
        i = 0
        while i > -1:
            i = hosts.find("[")
            j = hosts.find("]", i)
            prefix = hosts[i - 5 : i]
            old_str = prefix + hosts[i : j + 1]
            suffixes = []

            group = hosts[i + 1 : j - 1]
            for entries in group.split(","):
                if "-" in entries:
                    start, end = entries.split("-")
                    for i in range(int(start), int(end) + 1):
                        suffixes.append(f"{str(i).zfill(len(start))}")
                else:
                    suffixes.append(entries)

            for suffix in suffixes:
                new_str += f"{prefix}{suffix},"

            new_hosts = hosts.replace(old_str, new_str[:-1])
        return new_hosts.split(",")

    def _wait_for_path(
        self, path: Union[str, Path], timeout: float = 5.0, poll_interval: float = 0.2
    ) -> bool:
        """Wait for the given path to exist."""
        waited = 0.0
        path = Path(path)
        while not path.exists() and waited < timeout:
            time.sleep(poll_interval)
            waited += poll_interval
        return path.exists()

    def _get_work_dir(self, rel_proj_path: Union[str, Path]) -> Path:
        """Gets the $WORK directory path on Frontera."""
        stdin, stdout, stderr = self._ssh_client.exec_command("echo $WORK")
        work_dir = stdout.read().decode().strip()
        if not work_dir:
            raise RuntimeError("Failed to get $WORK directory path from Frontera")
        self._work_dir = Path(work_dir) / rel_proj_path
        return self._work_dir

    def _get_free_port(self) -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

    def _login_pexpect(self):
        """Logs into Frontera HPC using interactive 2FA authentication."""

        raise NotImplementedError("Does not work yet")

        remote_host = self.credentials.username + "@frontera.tacc.utexas.edu"
        local_port = self._get_free_port()
        command = f"ssh -L {local_port}:localhost:22 {remote_host}"
        pexpect_session = spawn(command)
        pexpect_session.timeout = 30
        pexpect_session.logfile = self._log_file

        idx = pexpect_session.expect(
            ["Password:", r".*frontera\(\d*\)\$ .*", EOF, TIMEOUT]
        )

        if idx == 0:
            # Handle password prompt
            pw = self.credentials.password
            pexpect_session.sendline(pw)

            # Handle 2FA prompt
            pexpect_session.expect("TACC Token Code:")
            duo_code = self.credentials.duo_code
            pexpect_session.sendline(duo_code)

            # Wait for the command prompt after successful login
            pexpect_session.expect(r".*frontera\(\d*\)\$ .*")
        elif idx == 1:
            pass  # Already connected; nothing to do.
        elif idx == 2:
            raise RuntimeError("SSH session ended unexpectedly.")
        elif idx == 3:
            raise RuntimeError("SSH session timed out.")
