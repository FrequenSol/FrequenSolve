"""
executor_factory.py

Example of a factory to select the executor based on some configuration.
"""

import os
from typing import Optional
from .local_executor import LocalExecutor
from .ssh_executor import SSHExecutor
from .slurm_executor import SlurmExecutor
from .executor_base import ExecutorBase

__all__ = []

def get_executor(mode: str, **kwargs) -> ExecutorBase:
   """Return an executor instance based on 'mode'.
   """
   if mode == "local":
      return LocalExecutor()
   elif mode == "ssh":
      # Expected kwargs: hostname, username, etc.
      return SSHExecutor(**kwargs)
   elif mode == "slurm":
      # Expected kwargs: partition, time, account, wait
      return SlurmExecutor(**kwargs)
   else:
      raise ValueError(f"Unknown executor mode: {mode}")

