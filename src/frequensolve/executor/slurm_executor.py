"""SLURM executor implementation for HPC job submission and management."""

import os
import re
import time
import tempfile
import subprocess
from typing import List, Optional, Dict
from pathlib import Path

from .executor_base import ExecutorBase, ExecutionStatus
from .resource_config import ResourceConfig

__all__ = ['SlurmExecutor']

class SlurmExecutor(ExecutorBase):
   """SLURM-based executor for HPC job submission and management.
   
   Handles job submission, monitoring, and file operations on SLURM-based clusters.
   """

   def __init__(
      self,
      resource_config: ResourceConfig,
      work_dir: Optional[str] = None,
      partition: str = "debug",
      account: Optional[str] = None
   ):
      """Initialize SLURM executor.
      
      Args:
         resource_config: Resource requirements for jobs.
         work_dir: Working directory for job execution.
         partition: SLURM partition/queue to use.
         account: Optional SLURM account for billing.
      """
      self.config = resource_config
      self.partition = partition
      self.account = account
      
      self.home_dir = os.path.expanduser("~")
      self.work_dir = work_dir or os.getcwd()
      self.os_type = "unix"  # SLURM only runs on Unix-like systems
      self.is_initialized = True

   def _create_job_script(self, command: str, **kwargs) -> str:
      """Create a SLURM job submission script.
      
      Args:
         command: Command to execute.
         **kwargs: Additional SLURM directives.
      
      Returns:
         str: Path to generated job script.
      """
      job_name = kwargs.get("job_name", "slurm_job")
      
      script = [
         "#!/bin/bash",
         f"#SBATCH --job-name={job_name}",
         f"#SBATCH --partition={self.partition}",
      ]
      
      if self.account:
         script.append(f"#SBATCH --account={self.account}")
      
      if self.config.max_duration_in_seconds:
         minutes = self.config.max_duration_in_seconds // 60
         script.append(f"#SBATCH --time={minutes}")
      
      if self.config.memory_per_rank_in_MB:
         script.append(f"#SBATCH --mem-per-cpu={self.config.memory_per_rank_in_MB}")
      
      script.extend([
         f"cd {self.work_dir}",
         command
      ])
      
      # Write script to temp file
      fd, path = tempfile.mkstemp(suffix='.sh', prefix='slurm_', dir=self.work_dir)
      with os.fdopen(fd, 'w') as f:
         f.write('\n'.join(script))
      
      return path

   def execute_command(self, command: str, *args, **kwargs) -> ExecutionStatus:
      """Execute a command through SLURM batch submission.
      
      Args:
         command: Command to execute.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      script_path = self._create_job_script(command, **kwargs)
      
      try:
         start_time = time.time()
         result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
         )
         
         # Extract job ID from sbatch output
         job_id = re.search(r"Submitted batch job (\d+)", result.stdout)
         if not job_id:
            raise ValueError("Failed to get job ID from sbatch output")
         
         job_id = job_id.group(1)
         
         # Wait for job completion if requested
         if kwargs.get("wait", True):
            while True:
               status = self.get_job_status(job_id)
               if status in ["COMPLETED", "FAILED"]:
                  break
               time.sleep(self.refresh_rate)
         
         end_time = time.time()
         
         # Get job output if available
         stdout = stderr = ""
         out_file = f"slurm-{job_id}.out"
         if os.path.exists(out_file):
            with open(out_file) as f:
               stdout = f.read()
         
         return ExecutionStatus(
            status=status,
            job_id=job_id,
            stdout=stdout,
            stderr=stderr,
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
      finally:
         os.unlink(script_path)

   def launch_mpi_job(self, command: str, nproc: int, *args, **kwargs) -> ExecutionStatus:
      """Launch a single MPI job through SLURM.
      
      Args:
         command: Command to execute.
         nproc: Number of MPI processes.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      mpi_cmd = f"srun -n {nproc} {command}"
      return self.execute_command(mpi_cmd, *args, **kwargs)

   def launch_mpi_jobs(self, commands: List[str], nproc: int, *args, **kwargs) -> List[ExecutionStatus]:
      """Launch multiple MPI jobs through SLURM.
      
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
      """Cancel a SLURM job.
      
      Args:
         job_id: SLURM job ID to cancel.
      
      Returns:
         bool: True if job was cancelled successfully.
      """
      try:
         subprocess.run(["scancel", job_id], check=True)
         return True
      except subprocess.CalledProcessError:
         return False

   def get_job_status(self, job_id: str) -> str:
      """Get status of a SLURM job.
      
      Args:
         job_id: SLURM job ID to check.
      
      Returns:
         str: Job status from SLURM (e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED").
      """
      try:
         result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader"],
            capture_output=True,
            text=True,
            check=True
         )
         status = result.stdout.strip().split()[0]
         
         # Map SLURM states to our status values
         status_map = {
            "PENDING": "PENDING",
            "RUNNING": "RUNNING",
            "COMPLETED": "COMPLETED",
            "FAILED": "FAILED",
            "CANCELLED": "FAILED",
            "TIMEOUT": "FAILED"
         }
         return status_map.get(status, "UNKNOWN")
         
      except subprocess.CalledProcessError:
         return "UNKNOWN"

   def remote_exists(self, path: str) -> bool:
      """Check if a path exists on the SLURM cluster.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if path exists.
      """
      return os.path.exists(path)

   def remote_listdir(self, path: str) -> List[str]:
      """List contents of directory on SLURM cluster.
      
      Args:
         path: Directory to list.
      
      Returns:
         List[str]: Names of files/directories in the path.
      """
      return os.listdir(path)

   def remote_mkdir(self, path: str, parents: bool = False) -> bool:
      """Create a directory on SLURM cluster.
      
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
      """Remove a directory on SLURM cluster.
      
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
      """Safely remove a directory on SLURM cluster with additional checks.
      
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
      """Copy a file to SLURM cluster.
      
      For local SLURM, this is just a file copy.
      
      Args:
         local_path: Source path on local machine.
         remote_path: Destination path on cluster.
      
      Returns:
         bool: True if file was copied successfully.
      """
      try:
         subprocess.run(["cp", local_path, remote_path], check=True)
         return True
      except subprocess.CalledProcessError:
         return False

   def remote_get(self, remote_path: str, local_path: str) -> bool:
      """Copy a file from SLURM cluster.
      
      For local SLURM, this is just a file copy.
      
      Args:
         remote_path: Source path on cluster.
         local_path: Destination path on local machine.
      
      Returns:
         bool: True if file was copied successfully.
      """
      return self.remote_put(remote_path, local_path)

   def is_folder_writeable(self, path: str) -> bool:
      """Check if a folder is writeable on SLURM cluster.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if folder exists and is writeable.
      """
      return os.access(path, os.W_OK)
