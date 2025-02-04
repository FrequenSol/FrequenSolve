"""Local job class."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

__all__ = ["LocalJob"]


class LocalJob:
    """Defines a job to be run locally."""

    def run(self) -> str:
        """Command to run the job."""
        return f"mpirun -np {self.ranks} {self.cmd}"

    @property
    def status(self) -> str:
        """Status of the job."""
        try:
            os.kill(int(job_id), 0)
            return "running"
        except ProcessLookupError:
            return "completed"
        except ValueError:
            return "unknown"
