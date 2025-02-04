"""Base class for jobs."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

__all__ = ["BaseJob"]


class BaseJob(ABC):
    """Base class for jobs."""

    memory: int  # Required memory (MB)
    ranks: int  # Number of ranks to run on
    cmd: str  # Command to run

    @abstractmethod
    def run(self) -> str:
        """Command to run the job."""
        pass

    @abstractmethod
    def status(self) -> str:
        """Status of the job."""
        pass
