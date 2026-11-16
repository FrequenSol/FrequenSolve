"""Configuration for executing FrequenSolve jobs on the local machine."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from frequensolve.orchestrator.sites.config import BaseSiteConfig
from frequensolve.util.system_info import SystemInfo

__all__ = ["LocalSiteConfig"]


def _parallel_resource_limits() -> Tuple[Optional[int], Optional[int]]:
    """Use the optional execution backend's process-aware resource detection."""
    try:
        from dask.system import cpu_count
    except ImportError:
        available_cores = None
    else:
        available_cores = cpu_count()
    try:
        from distributed.system import memory_limit
    except ImportError:
        available_bytes = None
    else:
        available_bytes = memory_limit()
    return available_cores, available_bytes


def _positive_number(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(value)


@dataclass
class LocalSiteConfig(BaseSiteConfig):
    """Local resources available to ``LocalSite``.

    Attributes:
        cores: Physical CPU count capped by process affinity/container quota
            when the optional parallel backend is installed.
        memory: Available memory budget in MiB, capped by the parallel backend's
            container/process limit when installed. Zero means unknown.
        mpi_wrapper: MPI launcher executable.
    """

    cores: int
    memory: float
    mpi_wrapper: str = "mpirun"

    def __init__(self) -> None:
        """Detect defaults without exceeding the execution backend's budget."""
        info = SystemInfo().get_cpu_info()
        host_cores = (
            _positive_number(info.get("physical_cores"))
            or _positive_number(info.get("logical_cores"))
            or _positive_number(os.cpu_count())
            or 1.0
        )
        host_memory = _positive_number(info.get("memory"))
        available_cores, available_bytes = _parallel_resource_limits()
        core_limit = _positive_number(available_cores)
        byte_limit = _positive_number(available_bytes)
        if core_limit is not None:
            host_cores = min(host_cores, core_limit)
        self.cores = max(1, int(host_cores))
        if byte_limit is not None:
            memory_limit_mib = byte_limit / 1024**2
            host_memory = (
                memory_limit_mib
                if host_memory is None
                else min(host_memory, memory_limit_mib)
            )
        self.memory = host_memory or 0.0
        self.mpi_wrapper = "mpirun"
