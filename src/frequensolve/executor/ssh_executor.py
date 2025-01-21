"""SSH executor implementation for running commands on remote machines."""

import os
import time
from typing import List, Optional
import paramiko
from pathlib import Path

from .executor_base import ExecutorBase, ExecutionStatus

__all__ = ['SSHExecutor']

class SSHExecutor(ExecutorBase):
   """SSH-based executor for remote command execution and file operations.
   
   Uses paramiko to establish SSH connections and SFTP file transfers.
   """

   def __init__(
      self,
      hostname: str,
      username: str,
      password: Optional[str] = None,
      key_file: Optional[str] = None,
      port: int = 22,
      work_dir: Optional[str] = None
   ):
      """Initialize SSH executor.
      
      Args:
         hostname: Remote host to connect to.
         username: SSH username.
         password: Optional password for authentication.
         key_file: Optional path to SSH private key file.
         port: SSH port number (default 22).
         work_dir: Working directory on remote host.
      
      Raises:
         paramiko.SSHException: If connection fails.
      """
      self.hostname = hostname
      self.username = username
      self.password = password
      self.key_file = key_file
      self.port = port
      
      self._client = None
      self._sftp = None
      self.requires_ssh = True
      
      # Connect and initialize
      self._connect()
      self.home_dir = self.execute_command("echo $HOME").stdout.strip()
      self.work_dir = work_dir or self.home_dir
      self.os_type = self.execute_command("uname").stdout.strip().lower()
      self.is_initialized = True

   def _connect(self) -> None:
      """Establish SSH connection and SFTP client.
      
      Raises:
         paramiko.SSHException: If connection fails.
      """
      self._client = paramiko.SSHClient()
      self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
      
      if self.key_file:
         self._client.connect(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            key_filename=self.key_file
         )
      else:
         self._client.connect(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password
         )
      
      self._sftp = self._client.open_sftp()

   def execute_command(self, command: str, *args, **kwargs) -> ExecutionStatus:
      """Execute a command on the remote host.
      
      Args:
         command: Command to execute.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      try:
         start_time = time.time()
         stdin, stdout, stderr = self._client.exec_command(
            command, 
            get_pty=kwargs.get("get_pty", False)
         )
         
         exit_status = stdout.channel.recv_exit_status()
         end_time = time.time()
         
         return ExecutionStatus(
            status="COMPLETED" if exit_status == 0 else "FAILED",
            return_code=exit_status,
            stdout=stdout.read().decode(),
            stderr=stderr.read().decode(),
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
      """Launch a single MPI job on remote host.
      
      Args:
         command: Command to execute.
         nproc: Number of MPI processes.
         *args: Additional positional arguments.
         **kwargs: Additional keyword arguments.
      
      Returns:
         ExecutionStatus: Execution results and status.
      """
      mpi_cmd = f"mpirun -n {nproc} {command}"
      return self.execute_command(mpi_cmd, *args, **kwargs)

   def launch_mpi_jobs(self, commands: List[str], nproc: int, *args, **kwargs) -> List[ExecutionStatus]:
      """Launch multiple MPI jobs sequentially on remote host.
      
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
      """Cancel a running job on remote host.
      
      Args:
         job_id: Process ID to terminate.
      
      Returns:
         bool: True if job was cancelled successfully.
      """
      result = self.execute_command(f"kill -15 {job_id}")
      return result.is_successful

   def get_job_status(self, job_id: str) -> str:
      """Get status of a job on remote host.
      
      Args:
         job_id: Process ID to check.
      
      Returns:
         str: "RUNNING" if process exists, "COMPLETED" otherwise.
      """
      result = self.execute_command(f"ps -p {job_id}")
      return "RUNNING" if result.is_successful else "COMPLETED"

   def remote_exists(self, path: str) -> bool:
      """Check if a path exists on remote host.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if path exists.
      """
      try:
         self._sftp.stat(path)
         return True
      except FileNotFoundError:
         return False

   def remote_listdir(self, path: str) -> List[str]:
      """List contents of remote directory.
      
      Args:
         path: Directory to list.
      
      Returns:
         List[str]: Names of files/directories in the path.
      """
      return self._sftp.listdir(path)

   def remote_mkdir(self, path: str, parents: bool = False) -> bool:
      """Create a directory on remote host.
      
      Args:
         path: Directory to create.
         parents: If True, create parent directories as needed.
      
      Returns:
         bool: True if directory was created successfully.
      """
      try:
         if parents:
            self.execute_command(f"mkdir -p {path}")
         else:
            self._sftp.mkdir(path)
         return True
      except:
         return False

   def remote_rmdir(self, path: str) -> bool:
      """Remove a directory on remote host.
      
      Args:
         path: Directory to remove.
      
      Returns:
         bool: True if directory was removed successfully.
      """
      try:
         self.execute_command(f"rm -rf {path}")
         return True
      except:
         return False

   def safe_remote_rmdir(self, path: str) -> bool:
      """Safely remove a directory on remote host with additional checks.
      
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
         # Check if path exists and is directory
         if not self.remote_exists(path):
            return False
         if not self.execute_command(f"test -d {path}").is_successful:
            return False
            
         # Get absolute path
         abs_path = self.execute_command(f"readlink -f {path}").stdout.strip()
         
         # Safety checks
         if not abs_path.startswith(self.work_dir):
            return False
         if abs_path in ["/", self.home_dir]:
            return False
            
         return self.remote_rmdir(abs_path)
      except:
         return False

   def remote_put(self, local_path: str, remote_path: str) -> bool:
      """Copy a file to remote host.
      
      Args:
         local_path: Source path on local machine.
         remote_path: Destination path on remote host.
      
      Returns:
         bool: True if file was copied successfully.
      """
      try:
         self._sftp.put(local_path, remote_path)
         return True
      except:
         return False

   def remote_get(self, remote_path: str, local_path: str) -> bool:
      """Copy a file from remote host.
      
      Args:
         remote_path: Source path on remote host.
         local_path: Destination path on local machine.
      
      Returns:
         bool: True if file was copied successfully.
      """
      try:
         self._sftp.get(remote_path, local_path)
         return True
      except:
         return False

   def is_folder_writeable(self, path: str) -> bool:
      """Check if a folder is writeable on remote host.
      
      Args:
         path: Path to check.
      
      Returns:
         bool: True if folder exists and is writeable.
      """
      result = self.execute_command(f"test -w {path}")
      return result.is_successful

   def __del__(self):
      """Clean up SSH and SFTP connections."""
      if self._sftp:
         self._sftp.close()
      if self._client:
         self._client.close()
