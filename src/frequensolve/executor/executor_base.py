"""Defines abstract base classes and core types for an extensible Python executor framework.

The framework supports multiple backends (local, remote SSH, SLURM, MPI, etc.).
"""

import abc
import typing as t
import paramiko
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ExecutionStatus:
   """Status and result information for a command execution.
   
   This class tracks both immediate execution results (return code, output) 
   and ongoing job status information for batch/queued jobs.
   
   Attributes:
      status (str): Current status of the execution:
         "PENDING": Job is queued/waiting to start
         "RUNNING": Job is currently executing
         "COMPLETED": Job finished successfully
         "FAILED": Job failed or was cancelled
         "UNKNOWN": Status cannot be determined
      return_code (int): Exit code from the command (0 typically indicates success)
      stdout (str): Standard output captured from the command
      stderr (str): Standard error output from the command
      job_id (Optional[str]): Job identifier for batch/queued jobs
      start_time (Optional[float]): Unix timestamp when job started
      end_time (Optional[float]): Unix timestamp when job completed
      working_dir (Optional[str]): Directory where command was executed
      error_msg (Optional[str]): Detailed error message if status is FAILED
   """
   status: str = "UNKNOWN"
   return_code: int = -1
   stdout: str = ""
   stderr: str = ""
   job_id: Optional[str] = None
   start_time: Optional[float] = None
   end_time: Optional[float] = None
   working_dir: Optional[str] = None
   error_msg: Optional[str] = None

   def __repr__(self) -> str:
      """Create a detailed string representation.
      
      Returns:
         str: Multi-line description of execution status and results.
      """
      lines = [
         f"ExecutionStatus(",
         f"   status={self.status}",
         f"   return_code={self.return_code}",
         f"   job_id={self.job_id}",
      ]
      if self.start_time:
         lines.append(f"   start_time={self.start_time}")
      if self.end_time:
         lines.append(f"   end_time={self.end_time}")
      if self.error_msg:
         lines.append(f"   error_msg={self.error_msg}")
      if self.stdout:
         lines.append(f"   stdout={self.stdout!r}")
      if self.stderr:
         lines.append(f"   stderr={self.stderr!r}")
      lines.append(")")
      return "\n".join(lines)

   @property
   def is_complete(self) -> bool:
      """Check if execution has finished (successfully or not).
      
      Returns:
         bool: True if status is COMPLETED or FAILED.
      """
      return self.status in ["COMPLETED", "FAILED"]

   @property
   def is_successful(self) -> bool:
      """Check if execution completed successfully.
      
      Returns:
         bool: True if status is COMPLETED and return code is 0.
      """
      return self.status == "COMPLETED" and self.return_code == 0


class ExecutorBase(abc.ABC):
   """Abstract base executor that all executors must implement.
   
   This design allows us to add functionality such as:
      - Command execution (sync/async)
      - File transfer operations
      - Remote filesystem operations
      - MPI job management
   """
   home_dir: str
   work_dir: str
   os_type: str = "unix"
   refresh_rate: float = 5
   is_initialized: bool = False
   requires_ssh: bool = False
   sftp_client: t.Optional[paramiko.SFTPClient] = None
   ssh_client: t.Optional[paramiko.SSHClient] = None
   license_tokens: t.Optional[t.Dict[str, str]] = None



   @abc.abstractmethod
   def run_command(self, command: str, *args, **kwargs) -> ExecutionStatus:
      """Execute a command synchronously or as a batch job."""
      pass

   @abc.abstractmethod
   def run_mpi_command(self, command: str, nproc: int, *args, **kwargs) -> ExecutionStatus:
      """Execute an MPI command using the underlying HPC or local environment."""
      pass

   @abc.abstractmethod
   def upload_file(self, local_path: str, remote_path: str) -> None:
      """Upload a file from local path to remote path.

      For local executors, this might just do a copy.

      Args:
         local_path: Path to the source file on the local system.
         remote_path: Destination path on the remote system.
      """
      pass

   @abc.abstractmethod
   def download_file(self, remote_path: str, local_path: str) -> None:
      """Download a file from remote path to local path.

      For local executors, this might just do a copy.

      Args:
         remote_path: Path to the source file on the remote system.
         local_path: Destination path on the local system.
      """
      pass

   @abc.abstractmethod
   def execute_command(self, command: str, *args, **kwargs) -> ExecutionStatus:
      """Execute a command synchronously."""
      pass

   @abc.abstractmethod
   def launch_mpi_job(self, command: str, nproc: int, *args, **kwargs) -> ExecutionStatus:
      """Launch a single MPI job."""
      pass

   @abc.abstractmethod 
   def launch_mpi_jobs(self, commands: List[str], nproc: int, *args, **kwargs) -> List[ExecutionStatus]:
      """Launch multiple MPI jobs."""
      pass

   @abc.abstractmethod
   def cancel_job(self, job_id: str) -> bool:
      """Cancel a running job.
      
      Args:
         job_id: Identifier for the job to cancel.
      
      Returns:
         bool: True if job was cancelled successfully.
      """
      pass

   @abc.abstractmethod
   def get_job_status(self, job_id: str) -> str:
      """Get the current status of a job.
      
      Args:
         job_id: Identifier for the job to check.
      
      Returns:
         str: Job status (e.g., "RUNNING", "COMPLETED", "FAILED", etc.)
      """
      pass

   @abc.abstractmethod
   def remote_exists(self, path: str) -> bool:
      """Check if a path exists on the remote system.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if path exists.
      """
      pass

   @abc.abstractmethod
   def remote_listdir(self, path: str) -> List[str]:
      """List contents of a remote directory.
      
      Args:
         path: Directory path to list.
      
      Returns:
         List[str]: Names of files/directories in the path.
      """
      pass

   @abc.abstractmethod
   def remote_mkdir(self, path: str, parents: bool = False) -> bool:
      """Create a directory on the remote system.
      
      Args:
         path: Directory path to create.
         parents: If True, create parent directories as needed.
      
      Returns:
         bool: True if directory was created successfully.
      """
      pass

   @abc.abstractmethod
   def remote_rmdir(self, path: str) -> bool:
      """Remove a directory on the remote system.
      
      Args:
         path: Directory path to remove.
      
      Returns:
         bool: True if directory was removed successfully.
      """
      pass

   @abc.abstractmethod
   def safe_remote_rmdir(self, path: str) -> bool:
      """Safely remove a directory on the remote system.
      
      Like remote_rmdir but with additional safety checks.
      
      Args:
         path: Directory path to remove.
      
      Returns:
         bool: True if directory was removed successfully.
      """
      pass

   @abc.abstractmethod
   def remote_put(self, local_path: str, remote_path: str) -> bool:
      """Copy a file from local to remote system.
      
      Args:
         local_path: Source path on local system.
         remote_path: Destination path on remote system.
      
      Returns:
         bool: True if file was copied successfully.
      """
      pass

   @abc.abstractmethod
   def remote_get(self, remote_path: str, local_path: str) -> bool:
      """Copy a file from remote to local system.
      
      Args:
         remote_path: Source path on remote system.
         local_path: Destination path on local system.
      
      Returns:
         bool: True if file was copied successfully.
      """
      pass

   @abc.abstractmethod
   def is_folder_writeable(self, path: str) -> bool:
      """Check if a folder is writeable.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if folder exists and is writeable.
      """
      pass

   def pretty_print(self, msg: str, level: int = 0) -> None:
      """Print a formatted message with indentation.
      
      Args:
         msg: Message to print.
         level: Indentation level (0 = no indent).
      """
      indent = "   " * level
      print(f"{indent}{msg}")
