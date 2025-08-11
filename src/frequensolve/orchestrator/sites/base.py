import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

__all__ = ["BaseSite", "SiteStatus", "_check_if_notebook"]


def _wait_for_path(
    path: Union[str, Path], timeout: float = 5.0, poll_interval: float = 0.2
) -> bool:
    """Wait for the given path to exist."""
    waited = 0.0
    path = Path(path)
    while not path.exists() and waited < timeout:
        time.sleep(poll_interval)
        waited += poll_interval
    return path.exists()


def _check_if_notebook() -> bool:
    """Check if we're running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False  # Probably standard Python interpreter


@dataclass
class SiteStatus:
    """Status and result information for a command execution.

    This class tracks both immediate execution results (return code, output)
    and ongoing job status information for batch/queued jobs.

    Attributes:
       status (str):
          Current status of the execution:
             "pending":   Job is queued/waiting to start
             "running":   Job is currently executing
             "completed": Job finished successfully
             "failed":    Job failed or was cancelled
             "unknown":   Status cannot be determined
       return_code (int):
          Exit code from the command (0 typically indicates success)
       stdout (str):
          Standard output captured from the command
       stderr (str):
          Standard error output from the command
       job_id (Optional[str]):
          Job identifier for batch/queued jobs
       start_time (Optional[float]):
          Unix timestamp when job started
    """

    status: str = "unknown"
    return_code: int = -1
    stdout: str = ""
    stderr: str = ""
    job_id: Optional[str] = None
    hostname: Optional[str] = None
    start_time: Optional[float] = None

    @property
    def is_queued(self) -> bool:
        return self.status == "pending"

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def is_complete(self) -> bool:
        return self.status in ["completed", "failed"]

    @property
    def is_successful(self) -> bool:
        return self.status == "completed" and self.return_code == 0


@dataclass(kw_only=True)
class BaseSite(ABC):
    """Base class for site configuration."""

    _is_notebook: bool = field(default_factory=_check_if_notebook)

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running job.

        Args:
            job_id: The ID of the job to cancel
        """
        pass

    @abstractmethod
    def submit(self, **kwargs) -> str:
        """Submit a job and return its ID.

        Returns:
            str: The job ID
        """
        pass
