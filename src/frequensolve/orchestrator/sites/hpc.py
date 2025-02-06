import json
import os
import re
import subprocess
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Literal, Optional, Union

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from jinja2 import Environment, FileSystemLoader
from paramiko import AutoAddPolicy, SFTPClient, SSHClient, SSHException

from frequensolve.orchestrator.sites.base_site import (
    BaseSite,
    BaseSiteConfig,
    SiteStatus,
)
from frequensolve.orchestrator.tasks.base_task import BaseTask

__all__ = ["HPCSiteConfig", "HPCSite", "HPCSiteCredentials"]


@dataclass
class HPCSiteCredentials:
    username: str

    @classmethod
    def load(cls, name: str) -> "HPCSiteCredentials":
        raise NotImplementedError


@dataclass
class HPCSiteConfig(BaseSiteConfig):
    """HPC site configuration.

    Attributes:
       hostname:         Hostname for HPC facility (e.g. "frontera.tacc.utexas.edu")
       scheduler:        Job scheduler (SLURM, PBS, LSF).
       local:            Whether to run locally.
       mpi_wrapper:      MPI wrapper (srun, mpirun, etc.).
       poll_interval:    Interval to poll job status in seconds.
       partition:        Job partition.
       account:          Job account.
       min_hosts:        Minimum number of hosts to use.
       max_hosts:        Maximum number of hosts to use.
       sockets_per_host: Number of sockets per host.
       gpus_per_host:    Number of GPUs per host.
       cores_per_socket: Number of cores per socket.
       max_duration:     Maximum time resources can be requested (HH:MM:SS).
       memory_per_rank:  Memory allocation per MPI rank in megabytes.
    """

    hostname: str
    scheduler: Literal["SLURM", "PBS", "LSF"]
    local: bool = False
    mpi_wrapper: str = "srun"
    poll_interval: int = 5
    partition: Optional[str] = None
    account: Optional[str] = None
    min_hosts: Optional[int] = None
    max_hosts: Optional[int] = None
    sockets_per_host: Optional[int] = None
    gpus_per_host: Optional[int] = None
    cores_per_socket: Optional[int] = None
    max_duration: Optional[str] = None
    memory_per_rank: Optional[int] = None
    cpu_info: Optional[Dict[str, Any]] = None
    gpu_info: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "HPCSiteConfig":
        return cls(**data)

    @classmethod
    def load(cls, name: str) -> "HPCSiteConfig":
        name += ".json" if not name.endswith(".json") else ""
        try:
            with open(name, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"failed to load configuration from {name}: {e}")
        return cls.from_dict(data)

    def save(self, name: str) -> None:
        name += ".json" if not name.endswith(".json") else ""
        try:
            with open(name, "w") as f:
                json.dump(self.__dict__(), f, indent=3)
        except Exception as e:
            warnings.warn(f"failed to save configuration to {name}: {e}")

    def __dict__(self) -> dict:
        return asdict(self)


@dataclass
class HPCSite(BaseSite):
    """HPC site manager."""

    config: HPCSiteConfig
    credentials: HPCSiteCredentials
    work_dir: Optional[Union[str, Path]] = None
    tmp_dir: Optional[Union[str, Path]] = None
    status: SiteStatus = field(default_factory=lambda: SiteStatus(status="unknown"))
    _sftp: Optional[SFTPClient] = None
    _ssh_client: Optional[SSHClient] = None
    _dask_client: Optional[Client] = None

    def __init__(self, config_file: str, credential_file: str):
        self.config = HPCSiteConfig.load(config_file)
        self.credentials = HPCSiteCredentials.load(credential_file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.deprovision()
        if self._sftp:
            self._sftp.close()
        if self._ssh_client:
            self._ssh_client.close()

    def __del__(self):
        self.deprovision()
        if self._sftp:
            self._sftp.close()
        if self._ssh_client:
            self._ssh_client.close()

    # TODO: check whether resources are already provisioned
    # TODO: method to run without provisioning (if resources are already provisioned)
    # TODO: check for timeout, if so reprovision (make option) and restart interrupted jobs

    def provision(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ):

        # Validate duration
        if self.config.max_duration:
            if duration:
                h, m, s = map(int, duration.split(":"))
                duration_secs = h * 3600 + m * 60 + s

                h, m, s = map(int, self.config.max_duration.split(":"))
                max_duration_secs = h * 3600 + m * 60 + s

                if duration_secs > max_duration_secs:
                    warnings.warn(
                        f"Requested duration {duration} exceeds maximum allowed duration ({self.config.max_duration});"
                        f"using maximum allowed duration ({self.config.max_duration}) instead."
                    )
                    duration = self.config.max_duration
            else:
                duration = self.config.max_duration

        if self.config.scheduler == "SLURM":
            return self._provision_SLURM(nhost, nproc, duration, **kwargs)
        elif self.config.scheduler == "PBS":
            return self._provision_PBS(nhost, nproc, duration, **kwargs)
        elif self.config.scheduler == "LSF":
            return self._provision_LSF(nhost, nproc, duration, **kwargs)
        else:
            raise ValueError(f"Scheduler {self.config.scheduler} not supported")

    def _generate_provision_script(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ) -> str:
        name = kwargs.get("name", "FS_server")
        env = Environment(
            loader=FileSystemLoader("src/frequensolve/orchestrator/templates")
        )
        if self.config.scheduler == "SLURM":
            template = env.get_template("provision_SLURM.sh")
        elif self.config.scheduler == "PBS":
            template = env.get_template("provision_PBS.sh")
        elif self.config.scheduler == "LSF":
            template = env.get_template("provision_LSF.sh")
        else:
            raise ValueError(f"Scheduler {self.config.scheduler} not supported")

        return template.render(
            name=name,
            nhost=nhost,
            nproc=nproc,
            partition=self.config.partition,
            account=self.config.account,
            duration=duration,
            work_dir=self.config.work_dir,
            mpi=self.config.mpi_wrapper,
        )

    def _provision_SLURM(
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

        script = self._generate_provision_script(nhost, nproc, duration, **kwargs)
        fd, script_path = tempfile.mkstemp(
            suffix=".sh", prefix="slurm_", dir=self.work_dir
        )
        with os.fdopen(fd, "w") as f:
            f.write(script)

        start_event = Event()

        if self.config.local:
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["sbatch", script_path], capture_output=True, text=True, check=True
                )

                # Get job ID from sbatch output
                job_id = re.search(r"Submitted batch job (\d+)", result.stdout)
                if not job_id:
                    raise ValueError("failed to get job ID from sbatch output")
                job_id = job_id.group(1)

                self.status.job_id = job_id
                self.status.status = "pending"
                self.status.start_time = start_time

                # Start monitoring thread to set event when job starts
                def monitor_job():
                    while True:
                        result = subprocess.run(
                            ["squeue", "-j", job_id, "-h", "-o", "%T"],
                            capture_output=True,
                            text=True,
                        )
                        if "running" in result.stdout:
                            self.status.status = "running"
                            start_event.set()

                            # TODO: may need to improve parsing to handle compressed notation
                            result = subprocess.run(
                                ["squeue", "-j", job_id, "-h", "-o", "%N"],
                                capture_output=True,
                                text=True,
                            )
                            node = result.stdout.decode().strip().split()[0]
                            self.status.hostname = node
                            break
                        elif not result.stdout.strip():
                            self.status.status = "failed"
                            break
                        time.sleep(self.config.poll_interval)

                Thread(target=monitor_job, daemon=True).start()

            except Exception as e:
                self.status.status = "failed"
                print(e)
            finally:
                os.unlink(script_path)

        else:
            try:
                try:
                    # TODO: will need to handle 2FA login
                    self._ssh_client = self._ssh_login()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to connect to {self.config.hostname}: {str(e)}"
                    )

                # Open SFTP client
                self._sftp = self._ssh_client.open_sftp()

                # Copy script to remote
                remote_path = f"/tmp/{os.path.basename(script_path)}"
                self._sftp.put(script_path, remote_path)

                # Submit job
                stdin, stdout, stderr = self._ssh_client.exec_command(
                    f"sbatch {remote_path}"
                )
                job_id = re.search(r"Submitted batch job (\d+)", stdout.read().decode())
                if not job_id:
                    raise ValueError("failed to get job ID from sbatch output")
                job_id = job_id.group(1)

                # Set job status
                self.status.job_id = job_id
                self.status.status = "pending"
                self.status.start_time = time.time()

                # Monitor job status and get compute node
                def monitor_remote_job():
                    while True:
                        stdin, stdout, stderr = self._ssh_client.exec_command(
                            f"squeue -j {job_id} -h -o %T"
                        )
                        status = stdout.read().decode().strip()
                        if "running" in status:
                            self.status.status = "running"
                            start_event.set()

                            # TODO: may need to improve parsing to handle compressed notation
                            # Get the compute node
                            stdin, stdout, stderr = self._ssh_client.exec_command(
                                f"squeue -j {job_id} -h -o %N"
                            )

                            # TODO: improve logging
                            self.status.stdout += stdout.read().decode()
                            self.status.stderr += stderr.read().decode()

                            node = stdout.read().decode().strip().split()[0]
                            self.status.hostname = node
                            break
                        elif not status:
                            self.status.status = "failed"
                            break
                        time.sleep(self.config.poll_interval)

                Thread(target=monitor_remote_job, daemon=True).start()

            except Exception as e:
                self.status.status = "failed"
                self.status.error_msg = str(e)
            finally:
                os.unlink(script_path)

        return start_event

    def _ssh_login(self, **kwargs) -> None:
        """Establish an SSH connection and set up port forwarding.

        Returns:
           SSHClient: The SSH client object.

        Raises:
           ValueError: If 'hostname' or 'username' is not provided in kwargs.
           SSHException: If there is an error connecting via SSH.
        """
        if "hostname" not in kwargs or "username" not in kwargs:
            raise ValueError(
                "Both 'hostname' and 'username' must be provided in kwargs"
            )

        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())

        try:
            ssh.connect(kwargs["hostname"], username=kwargs["username"])
        except SSHException as e:
            raise SSHException(f"Failed to connect to {kwargs['hostname']}: {str(e)}")

        cmd = f"ssh -N -L 8786:localhost:8786 {self.credentials.username}@{self.config.hostname}"
        stdin, stdout, stderr = ssh.exec_command(cmd)

        error = stderr.read().decode().strip()
        if error:
            raise RuntimeError(f"Failed to execute port forwarding command: {error}")

        return ssh

    # TODO: wait for event to be set, then call
    def _ssh_compute(self, **kwargs) -> None:
        """Jump from login to compute.

        Returns:
           SSHClient: The SSH client object.

        Raises:
           ConnectionError: If the SSH connection is not active.
           RuntimeError: If the jump command fails.
        """
        if not self._ssh_client.get_transport().is_active():
            raise ConnectionError("SSH login client connection is not active.")

        jump_cmd = f"ssh -N -L 8786:localhost:8786 {self.credentials.username}@{self.status.hostname}"
        stdin, stdout, stderr = self._ssh_client.exec_command(jump_cmd)

        error = stderr.read().decode().strip()
        if error:
            raise RuntimeError(f"Failed to execute jump command: {error}")
        if not self._ssh_client.get_transport().is_active():
            raise ConnectionError("Failed to establish connection to the compute node.")

    def _start_dask_client(self, **kwargs) -> None:
        from dask.distributed import Client

        # TODO: correct this
        client = Client("tcp://1.2.3.4:8786")

    def _provision_PBS(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ):
        """Create a PBS resource request."""
        raise NotImplementedError

    def _provision_LSF(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ):
        """Create an LSF resource request."""
        raise NotImplementedError

    def wait_provisioned(self):
        """Wait for the job to be provisioned."""
        while True:
            status = self.check_status()
            if status.is_complete:
                break
            time.sleep(5)

    def cancel_jobs(self, job_id: str) -> bool:
        job_id = self.status.job_id
        try:
            os.kill(int(job_id), 15)  # SIGTERM
            time.sleep(self.config.poll_interval)

            try:
                os.kill(int(job_id), 9)  # SIGKILL
            except ProcessLookupError:
                pass  # Process already terminated
            return True
        except (ProcessLookupError, ValueError):
            return False

    def deprovision(self, **kwargs):
        """Release HPC resources."""
        if self.config.local:
            subprocess.run(["scancel", self.status.job_id])
        else:
            stdin, stdout, stderr = self._ssh_client.exec_command(
                f"scancel {self.status.job_id}"
            )

    def get_site_status(self):
        """Check the status of the resource request."""
        job_id = self.status.job_id
        if job_id is None:
            return SiteStatus(
                status="unknown", return_code=-1, stdout="", stderr="No job ID found"
            )
        try:
            if self.config.scheduler == "SLURM":
                if self.config.local:
                    result = subprocess.run(
                        ["squeue", "--job", job_id], capture_output=True, text=True
                    )
                else:
                    stdin, stdout, stderr = self._ssh_client.exec_command(
                        f"squeue -j {job_id} -h -o %T"
                    )
                    status = stdout.read().decode().strip()
            elif self.config.scheduler == "PBS":
                raise NotImplementedError
            elif self.config.scheduler == "LSF":
                raise NotImplementedError

            if result.returncode != 0:
                return SiteStatus(
                    status="unknown",
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            if re.search(r"\bR\b", result.stdout):
                return SiteStatus(
                    status="running",
                    return_code=0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            elif re.search(r"\bPD\b", result.stdout):
                return SiteStatus(
                    status="pending",
                    return_code=0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            else:
                return SiteStatus(
                    status="completed",
                    return_code=0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
        except Exception as e:
            return SiteStatus(status="failed", return_code=-1, stdout="", stderr=str(e))

    # TODO: neet to add run_mpi_flux method, etc.
    def submit_jobs(self, jobs: List[BaseTask]) -> None:
        """Submit a job to the scheduler."""

        # Could also just use Dask's map feature

        futures = []
        for job in jobs:
            fut = self._client.submit(
                job.run_mpi_flux, job.nodes, job.ranks_per_node, job.cmd
            )
            futures.append(fut)
        return futures

    def cancel_jobs(self, jobs: List[BaseTask]) -> None:
        """Cancel a submitted flux job."""
        for job in jobs:
            self._client.submit(job.cancel_mpi_flux, job.job_id)

    def provision_dask_jobqueue(
        self, nhost: int, nproc: int, duration: Optional[str] = None, **kwargs
    ) -> Client:
        """Provision resources using Dask JobQueue and initialize Flux.

        Args:
           nhost: Number of hosts to request.
           nproc: Number of processes per host.
           duration: Job duration in HH:MM:SS format.
           **kwargs: Additional arguments for job configuration.

        Returns:
           client: A Dask distributed client connected to the cluster.
        """

        if self.config.scheduler == "SLURM":
            cluster = SLURMCluster(
                queue=self.config.partition,
                account=self.config.account,
                processes=nproc,
                cores=self.config.cores_per_node,
                walltime=duration,
                job_extra=[
                    f"--nodes={nhost}",
                    f"--ntasks-per-node={nproc // nhost}",
                ],
                job_script_prologue=[f"{self.config.mpi_wrapper} flux start --boot"],
                **kwargs,
            )
        elif self.config.scheduler == "PBS":
            raise NotImplementedError
        elif self.config.scheduler == "LSF":
            raise NotImplementedError

        cluster.scale(jobs=nhost)

        # Connect a Dask client to the cluster
        client = Client(cluster)

        return client


# Use Dask to co-launch a Jupyter server

# Dask can help you by launching other services alongside it. For example, you can run a Jupyter notebook server on the machine running the dask-scheduler process with the following commands

# from dask.distributed import Client
# client = Client(scheduler_file='scheduler.json')

# import socket
# host = client.run_on_scheduler(socket.gethostname)

# def start_jlab(dask_scheduler):
#     import subprocess
#     proc = subprocess.Popen(['/path/to/jupyter', 'lab', '--ip', host, '--no-browser'])
#     dask_scheduler.jlab_proc = proc

# client.run_on_scheduler(start_jlab)
