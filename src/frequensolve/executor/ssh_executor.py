"""
ssh_executor.py

Implements an executor that uses SSH (via paramiko, fabric, or similar)
to remotely run commands and transfer files.
"""

import os
import paramiko
from typing import Optional
from executor_base import ExecutorBase, ExecutionResult


class SSHExecutor(ExecutorBase):
    """
    SSH-based executor. Uses paramiko to:
      - open an SSH connection
      - run commands on the remote host
      - transfer files via SFTP
    """

    def __init__(
        self,
        hostname: str,
        username: str,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        port: int = 22
    ):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_file = key_file
        self.port = port
        self._client = None

    def _ensure_connection(self):
        """
        Ensure that self._client is initialized and connected.
        """
        if self._client is not None:
            return  # Already connected

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

    def run_command(self, command: str, *args, **kwargs) -> ExecutionResult:
        self._ensure_connection()
        try:
            stdin, stdout, stderr = self._client.exec_command(command)
            rc = stdout.channel.recv_exit_status()
            return ExecutionResult(
                stdout=stdout.read().decode("utf-8"),
                stderr=stderr.read().decode("utf-8"),
                return_code=rc
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                return_code=-1
            )

    def run_mpi_command(self, command: str, nproc: int, *args, **kwargs) -> ExecutionResult:
        """
        For remote SSH, we would still rely on `mpirun` or `srun` 
        being available on the remote host. 
        """
        mpi_cmd = f"mpirun -n {nproc} {command}"
        return self.run_command(mpi_cmd)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self._ensure_connection()
        with self._client.open_sftp() as sftp:
            sftp.put(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        self._ensure_connection()
        with self._client.open_sftp() as sftp:
            sftp.get(remote_path, local_path)

