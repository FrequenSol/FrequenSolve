"""
local_executor.py

Implements a local executor that runs commands and handles file
operations on the same machine without any remote submission.
"""

import subprocess
import shutil
from typing import Optional
from executor_base import ExecutorBase, ExecutionResult


class LocalExecutor(ExecutorBase):
    """
    Local executor simply calls subprocess.run for running commands,
    and shutil.copy for file transfers.
    """

    def run_command(self, command: str, *args, **kwargs) -> ExecutionResult:
        try:
            process = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                check=False  # We won't raise here; we handle return code ourselves
            )
            return ExecutionResult(
                stdout=process.stdout,
                stderr=process.stderr,
                return_code=process.returncode
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                return_code=-1
            )

    def run_mpi_command(
        self, command: str, nproc: int, *args, **kwargs
    ) -> ExecutionResult:
        mpi_cmd = f"mpirun -n {nproc} {command}"
        return self.run_command(mpi_cmd, *args, **kwargs)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        shutil.copyfile(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        shutil.copyfile(remote_path, local_path)

