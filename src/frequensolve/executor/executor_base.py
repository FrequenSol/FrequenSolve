"""
executor_base.py

Defines abstract base classes and core types for an extensible
Python executor framework supporting multiple backends (local,
remote SSH, SLURM, MPI, etc.).
"""

import abc
import typing as t


class ExecutionResult:
    """
    Simple container class for storing the result of a command execution.
    """
    def __init__(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        job_id: t.Optional[str] = None
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.job_id = job_id

    def __repr__(self):
        return (
            f"ExecutionResult(stdout={self.stdout!r}, stderr={self.stderr!r}, "
            f"return_code={self.return_code}, job_id={self.job_id})"
        )


class ExecutorBase(abc.ABC):
    """
    Abstract base executor. All executors must implement these methods.

    This design allows us to add functionality such as:
      - run_command: synchronous or asynchronous job submission
      - upload_file / download_file: for transferring files if needed
      - run_mpi_command: convenience method for launching MPI tasks
    """

    @abc.abstractmethod
    def run_command(self, command: str, *args, **kwargs) -> ExecutionResult:
        """
        Execute a command synchronously or as a batch job.
        Return an ExecutionResult which contains the stdout, stderr, etc.
        """
        pass

    @abc.abstractmethod
    def run_mpi_command(
        self, command: str, nproc: int, *args, **kwargs
    ) -> ExecutionResult:
        """
        Execute an MPI command, possibly using the underlying HPC or local environment.
        """
        pass

    @abc.abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        Upload a file from local_path to remote_path.
        (For local executors, this might just do a copy.)
        """
        pass

    @abc.abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Download a file from remote_path to local_path.
        (For local executors, also just a copy.)
        """
        pass

