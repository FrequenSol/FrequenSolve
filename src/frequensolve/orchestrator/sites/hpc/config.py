"""Configuration objects for SLURM-backed HPC sites."""

import os
from dataclasses import dataclass, field

from frequensolve.orchestrator.sites.config import BaseSiteConfig
from frequensolve.util.printing import print_warn

__all__ = [
    "Stampede3Config",
    "_BaseQueue",
    "_DebugQueue",
    "_ICXQueue",
    "_SKXQueue",
    "_SPRQueue",
]


def _hms_to_seconds(hms: str) -> int:
    """Convert D-HH:MM:SS to seconds."""
    if "-" in hms:
        d, tmp = hms.split("-")
        d = int(d)
    else:
        d = 0
        tmp = hms
    h, m, s = map(int, tmp.split(":"))
    return d * 86400 + h * 3600 + m * 60 + s


def _seconds_to_hms(seconds: int) -> str:
    """Convert seconds to D-HH:MM:SS."""
    d = seconds // 86400
    seconds %= 86400
    h = seconds // 3600
    seconds %= 3600
    m = seconds // 60
    s = seconds % 60
    return f"{d}-{h:02d}:{m:02d}:{s:02d}"


# ----------------------------------
# Stampede3 queue and machine info
# ----------------------------------
@dataclass(frozen=True)
class _BaseQueue:
    """Base class for Stampede3 queues."""

    _name: str
    _max_duration: str
    _max_nodes: int
    _min_nodes: int


@dataclass(frozen=True)
class _SPRQueue(_BaseQueue):
    """Stampede3 SPR queue."""

    _name: str = "spr"
    _max_duration: str = "2-00:00:00"
    _max_nodes: int = 32
    _min_nodes: int = 1


@dataclass(frozen=True)
class _SKXQueue(_BaseQueue):
    """Stampede3 SKX queue."""

    _name: str = "skx"
    _max_duration: str = "2-00:00:00"
    _max_nodes: int = 256
    _min_nodes: int = 1


@dataclass(frozen=True)
class _ICXQueue(_BaseQueue):
    """Stampede3 ICX queue."""

    _name: str = "icx"
    _max_duration: str = "2-00:00:00"
    _max_nodes: int = 32
    _min_nodes: int = 1


@dataclass(frozen=True)
class _DebugQueue(_BaseQueue):
    """Stampede3 development queue."""

    _name: str = "skx-dev"
    _max_duration: str = "02:00:00"
    _max_nodes: int = 16
    _min_nodes: int = 1


@dataclass(frozen=True)
class _Stampede3SPRConfig(BaseSiteConfig):
    _hostname: str = "stampede3.tacc.utexas.edu"
    _scheduler: str = "SLURM"
    _mpi_wrapper: str = "ibrun"
    _poll_interval: int = 5
    _sockets_per_node: int = 2
    _gpus_per_node: int = 0
    _cores_per_socket: int = 56
    _memory_per_node: int = 128000
    _account: str = field(default_factory=lambda: os.getenv("TACC_ACCOUNT", ""))


@dataclass(frozen=True)
class _Stampede3SKXConfig(BaseSiteConfig):
    _hostname: str = "stampede3.tacc.utexas.edu"
    _scheduler: str = "SLURM"
    _mpi_wrapper: str = "ibrun"
    _poll_interval: int = 5
    _sockets_per_node: int = 2
    _gpus_per_node: int = 0
    _cores_per_socket: int = 24
    _memory_per_node: int = 198000
    _account: str = field(default_factory=lambda: os.getenv("TACC_ACCOUNT", ""))


@dataclass(frozen=True)
class _Stampede3ICXConfig(BaseSiteConfig):
    _hostname: str = "stampede3.tacc.utexas.edu"
    _scheduler: str = "SLURM"
    _mpi_wrapper: str = "ibrun"
    _poll_interval: int = 5
    _sockets_per_node: int = 2
    _gpus_per_node: int = 0
    _cores_per_socket: int = 40
    _memory_per_node: int = 198000
    _account: str = field(default_factory=lambda: os.getenv("TACC_ACCOUNT", ""))


@dataclass
class Stampede3Config:
    """Combines immutable base configuration with queue info for Stampede3.

    Args:
        queue: Stampede3 queue/partition name. Supported values are ``"spr"``,
            ``"icx"``, ``"skx"``, and ``"skx-dev"``.

    Raises:
        ValueError: If ``queue`` is not supported.
    """

    _queue: _BaseQueue
    _base_config: BaseSiteConfig

    def __init__(self, queue: str = "skx-dev"):
        if queue == "spr":
            self._queue = _SPRQueue()
            self._base_config = _Stampede3SPRConfig()
        elif queue == "icx":
            self._queue = _ICXQueue()
            self._base_config = _Stampede3ICXConfig()
        elif queue == "skx":
            self._queue = _SKXQueue()
            self._base_config = _Stampede3SKXConfig()
        elif queue == "skx-dev":
            self._queue = _DebugQueue()
            self._base_config = _Stampede3SKXConfig()
        else:
            raise ValueError(f"Invalid queue: {queue}")

    @property
    def hostname(self):
        """Return the Stampede3 login host name."""

        return self._base_config._hostname

    @property
    def scheduler(self):
        """Return the scheduler name used by this site."""

        return self._base_config._scheduler

    @property
    def mpi_wrapper(self):
        """Return the MPI launcher command."""

        return self._base_config._mpi_wrapper

    @property
    def poll_interval(self):
        """Return the default queue polling interval in seconds."""

        return self._base_config._poll_interval

    @property
    def sockets_per_node(self):
        """Return the number of CPU sockets per node."""

        return self._base_config._sockets_per_node

    @property
    def gpus_per_node(self):
        """Return the number of GPUs per node."""

        return self._base_config._gpus_per_node

    @property
    def cores_per_socket(self):
        """Return the number of CPU cores per socket."""

        return self._base_config._cores_per_socket

    @property
    def cores_per_node(self):
        """Return the total CPU cores per node."""

        return self.cores_per_socket * self.sockets_per_node

    @property
    def memory_per_node(self):
        """Return memory per node in megabytes."""

        return self._base_config._memory_per_node

    @property
    def memory_per_core(self):
        """Return memory per CPU core in megabytes."""

        return self._base_config._memory_per_node / self.cores_per_node

    @property
    def account(self):
        """Return the default allocation account."""

        return self._base_config._account

    @property
    def queue(self):
        """Return the active queue/partition name."""

        return self._queue._name

    @property
    def max_duration(self):
        """Return the maximum wall-clock duration for the queue."""

        return self._queue._max_duration

    @property
    def max_nodes(self):
        """Return the maximum number of nodes allowed by the queue."""

        return self._queue._max_nodes

    @property
    def min_nodes(self):
        """Return the minimum number of nodes required by the queue."""

        return self._queue._min_nodes

    def validate_request(self, nhost: int, nproc: int, duration: str):
        """Validate a requested allocation against queue limits.

        Args:
            nhost: Requested node count.
            nproc: Requested process count.
            duration: Requested wall time as ``HH:MM:SS`` or ``D-HH:MM:SS``.

        Returns:
            Validated duration, possibly clamped to the queue maximum.

        Raises:
            ValueError: If node or process counts violate queue limits.
        """
        if nhost < self.min_nodes:
            raise ValueError(f"Minimum number of nodes is {self.min_nodes}")
        if nhost > self.max_nodes:
            raise ValueError(f"Maximum number of nodes is {self.max_nodes}")
        if nproc < nhost:
            raise ValueError(f"Number of processes per node must be at least {nhost}")
        return self._validate_duration(duration)

    def _validate_duration(self, duration: str) -> str:
        """Validate duration."""
        duration_secs = _hms_to_seconds(duration)
        max_duration_secs = _hms_to_seconds(self.max_duration)

        if duration_secs > max_duration_secs:
            print_warn(
                f"Requested duration {duration} exceeds maximum allowed duration"
                f"({self.max_duration}), using {self.max_duration} instead"
            )
            duration = self.max_duration
        return duration
