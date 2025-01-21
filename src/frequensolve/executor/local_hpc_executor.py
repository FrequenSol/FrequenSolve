"""Local HPC executor implementation for running on compute nodes directly."""

import os
import time
import subprocess

from typing import List, Optional
from pathlib import Path

from .executor_base import ExecutorBase, ExecutionStatus
from .resource_config import ResourceConfig

__all__ = ['LocalHPCExecutor']

class LocalHPCExecutor(ExecutorBase):
   """Executor for running commands directly on HPC compute nodes.
   
   This executor is designed for cases where you're already logged into 
   a compute node and want to run MPI jobs directly without going through
   a job scheduler.
   """

   def __init__(
      self,
      resource_config: ResourceConfig,
      work_dir: Optional[str] = None,
      mpi_wrapper: str = "mpirun",
      hostfile: Optional[str] = None
   ):
      """Initialize Local HPC executor.
      
      Args:
         resource_config: Resource requirements for jobs.
         work_dir: Working directory for job execution.
         mpi_wrapper: MPI launch command (e.g., 'mpirun', 'mpiexec', 'ibrun').
         hostfile: Optional path to MPI hostfile/machinefile.
      """
      self.config = resource_config
      self.mpi_wrapper = mpi_wrapper
      self.hostfile = hostfile
      
      self.home_dir = os.path.expanduser("~")
      self.work_dir = work_dir or os.getcwd()
      self.os_type = "unix"
      self.is_initialized = True

   def execute_command(self, command: str, *args, **kwargs) -> ExecutionStatus:
      """Execute a command on the compute node.
      
      Args:
         command: Command to execute.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      try:
         start_time = time.time()
         process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.work_dir,
            text=True,
            bufsize=1,
            universal_newlines=True
         )
         
         # Store process ID for potential cancellation
         job_id = str(process.pid)
         
         stdout, stderr = process.communicate()
         end_time = time.time()
         
         return ExecutionStatus(
            status="COMPLETED" if process.returncode == 0 else "FAILED",
            return_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            job_id=job_id,
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
      """Launch a single MPI job on compute node.
      
      Args:
         command: Command to execute.
         nproc: Number of MPI processes.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      mpi_cmd = [self.mpi_wrapper, "-n", str(nproc)]
      
      if self.hostfile:
         if self.mpi_wrapper == "mpirun":
            mpi_cmd.extend(["--hostfile", self.hostfile])
         elif self.mpi_wrapper == "mpiexec":
            mpi_cmd.extend(["-f", self.hostfile])
         elif self.mpi_wrapper == "ibrun":
            # ibrun typically uses SLURM/PBS node allocation
            pass
      
      mpi_cmd.extend(command.split())
      return self.execute_command(" ".join(mpi_cmd), *args, **kwargs)

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
         # Try graceful termination first
         os.kill(int(job_id), 15)  # SIGTERM
         time.sleep(2)
         
         # Force kill if still running
         try:
            os.kill(int(job_id), 9)  # SIGKILL
         except ProcessLookupError:
            pass  # Process already terminated
            
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
      except ProcessLookupError:
         return "COMPLETED"
      except ValueError:
         return "UNKNOWN"

   def remote_exists(self, path: str) -> bool:
      """Check if a path exists on the compute node.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if path exists.
      """
      return os.path.exists(path)

   def remote_listdir(self, path: str) -> List[str]:
      """List contents of directory on compute node.
      
      Args:
         path: Directory to list.
      
      Returns:
         List[str]: Names of files/directories in the path.
      """
      return os.listdir(path)

   def remote_mkdir(self, path: str, parents: bool = False) -> bool:
      """Create a directory on compute node.
      
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
      """Remove a directory on compute node.
      
      Args:
         path: Directory to remove.
      
      Returns:
         bool: True if directory was removed successfully.
      """
      try:
         subprocess.run(["rm", "-rf", path], check=True)
         return True
      except subprocess.CalledProcessError:
         return False

   def safe_remote_rmdir(self, path: str) -> bool:
      """Safely remove a directory on compute node with additional checks.
      
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
      except:
         return False

   def remote_put(self, local_path: str, remote_path: str) -> bool:
      """Copy a file on compute node.
      
      Args:
         local_path: Source path.
         remote_path: Destination path.
      
      Returns:
         bool: True if file was copied successfully.
      """
      try:
         subprocess.run(["cp", local_path, remote_path], check=True)
         return True
      except subprocess.CalledProcessError:
         return False

   def remote_get(self, remote_path: str, local_path: str) -> bool:
      """Copy a file on compute node.
      
      Args:
         remote_path: Source path.
         local_path: Destination path.
      
      Returns:
         bool: True if file was copied successfully.
      """
      return self.remote_put(remote_path, local_path)

   def is_folder_writeable(self, path: str) -> bool:
      """Check if a folder is writeable on compute node.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if folder exists and is writeable.
      """
      return os.access(path, os.W_OK) 