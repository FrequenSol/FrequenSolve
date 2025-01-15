"""
local_executor.py

Implements a local executor that runs commands and handles file
operations on the same machine without any remote submission.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .executor_base import ExecutorBase, ExecutionStatus


class LocalExecutor(ExecutorBase):
   """Local executor that runs commands on the same machine.
   
   This executor implements operations for local command execution and file operations
   without any remote access or job scheduling.
   """

   def __init__(self, work_dir: Optional[str] = None):
      """Initialize local executor.
      
      Args:
         work_dir: Working directory for command execution. Defaults to current directory.
      """
      self.home_dir = os.path.expanduser("~")
      self.work_dir = work_dir or os.getcwd()
      self.os_type = "unix" if os.name != "nt" else "windows"
      self.is_initialized = True

   def execute_command(self, command: str, *args, **kwargs) -> ExecutionStatus:
      """Execute a command synchronously on local machine.
      
      Args:
         command: The command to execute.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      try:
         start_time = time.time()
         process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.work_dir
         )
         end_time = time.time()

         return ExecutionStatus(
            status="COMPLETED" if process.returncode == 0 else "FAILED",
            return_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            start_time=start_time,
            end_time=end_time,
            working_dir=self.work_dir
         )
      except Exception as e:
         return ExecutionStatus(
            status="FAILED",
            error_msg=str(e),
            working_dir=self.work_dir
         )

   def launch_mpi_job(self, command: str, nproc: int, *args, **kwargs) -> ExecutionStatus:
      """Launch a single MPI job locally.
      
      Args:
         command: The command to execute.
         nproc: Number of MPI processes.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      mpi_cmd = f"mpirun -n {nproc} {command}"
      return self.execute_command(mpi_cmd, *args, **kwargs)

   def launch_mpi_jobs(self, commands: List[str], nproc: int, *args, **kwargs) -> List[ExecutionStatus]:
      """Launch multiple MPI jobs sequentially.
      
      Args:
         commands: List of commands to execute.
         nproc: Number of MPI processes per command.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         List[ExecutionStatus]: Results for each command.
      """
      return [self.launch_mpi_job(cmd, nproc, *args, **kwargs) for cmd in commands]

   def cancel_job(self, job_id: str) -> bool:
      """Cancel a running job by process ID.
      
      Args:
         job_id: Process ID to terminate.
      
      Returns:
         bool: True if process was terminated successfully.
      """
      try:
         os.kill(int(job_id), 15)  # SIGTERM
         return True
      except (ProcessLookupError, ValueError):
         return False

   def get_job_status(self, job_id: str) -> str:
      """Get status of a job by process ID.
      
      Args:
         job_id: Process ID to check.
      
      Returns:
         str: "RUNNING" if process exists, "COMPLETED" otherwise.
      """
      try:
         os.kill(int(job_id), 0)  # Check if process exists
         return "RUNNING"
      except (ProcessLookupError, ValueError):
         return "COMPLETED"

   def remote_exists(self, path: str) -> bool:
      """Check if a local path exists.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if path exists.
      """
      return os.path.exists(path)

   def remote_listdir(self, path: str) -> List[str]:
      """List contents of a local directory.
      
      Args:
         path: Directory to list.
      
      Returns:
         List[str]: Names of files/directories in the path.
      """
      return os.listdir(path)

   def remote_mkdir(self, path: str, parents: bool = False) -> bool:
      """Create a local directory.
      
      Args:
         path: Directory to create.
         parents: If True, create parent directories as needed.
      
      Returns:
         bool: True if directory was created successfully.
      """
      try:
         if parents:
            os.makedirs(path, exist_ok=True)
         else:
            os.mkdir(path)
         return True
      except OSError:
         return False

   def remote_rmdir(self, path: str) -> bool:
      """Remove a local directory.
      
      Args:
         path: Directory to remove.
      
      Returns:
         bool: True if directory was removed successfully.
      """
      try:
         shutil.rmtree(path)
         return True
      except OSError:
         return False

   def safe_remote_rmdir(self, path: str) -> bool:
      """Safely remove a local directory with additional checks.
      
      Verifies that:
         1. Path exists and is a directory
         2. Path is within work_dir
         3. Path is not a system directory
      
      Args:
         path: Directory to remove.
      
      Returns:
         bool: True if directory was removed safely.
      """
      try:
         path = os.path.abspath(path)
         if not os.path.isdir(path):
            return False
         if not path.startswith(self.work_dir):
            return False
         if path == "/" or path == self.home_dir:
            return False
         return self.remote_rmdir(path)
      except OSError:
         return False

   def remote_put(self, local_path: str, remote_path: str) -> bool:
      """Copy a file to another local path.
      
      Args:
         local_path: Source path.
         remote_path: Destination path.
      
      Returns:
         bool: True if file was copied successfully.
      """
      try:
         shutil.copy2(local_path, remote_path)
         return True
      except OSError:
         return False

   def remote_get(self, remote_path: str, local_path: str) -> bool:
      """Copy a file from another local path.
      
      Args:
         remote_path: Source path.
         local_path: Destination path.
      
      Returns:
         bool: True if file was copied successfully.
      """
      return self.remote_put(remote_path, local_path)

   def is_folder_writeable(self, path: str) -> bool:
      """Check if a local folder is writeable.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if folder exists and is writeable.
      """
      return os.access(path, os.W_OK)
