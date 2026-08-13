"""Configuration for executing FrequenSolve jobs on the local machine."""

from dataclasses import dataclass

from frequensolve.orchestrator.sites.config import BaseSiteConfig
from frequensolve.util.system_info import SystemInfo

__all__ = ["LocalSiteConfig"]


@dataclass
class LocalSiteConfig(BaseSiteConfig):
    """Local machine resource summary used by ``LocalSite``.

    Attributes:
        cores: Detected physical CPU cores.
        memory: Detected system memory in bytes.
        mpi_wrapper: MPI launcher executable.
    """

    cores: int
    memory: int
    mpi_wrapper: str = "mpirun"

    def __init__(self) -> None:
        """Detect local CPU and memory defaults for local execution."""

        system_info = SystemInfo()
        info = system_info.gather_all_info()
        self.cores = info["cpu"]["physical_cores"]
        self.memory = info["cpu"]["memory"]
        self.mpi_wrapper = "mpirun"
