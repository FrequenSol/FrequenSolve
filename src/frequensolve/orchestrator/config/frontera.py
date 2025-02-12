import os
from dataclasses import dataclass, field

from frequensolve.orchestrator.config.base import BaseSiteConfig
from frequensolve.util.printing import print_warn

__all__ = ["FronteraConfig", "_BaseQueue", "_DebugQueue", "_NormalQueue", "_LargeQueue"]


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
# Frontera queue and machine info
# ----------------------------------
@dataclass(frozen=True)
class _BaseQueue:
    """Base class for Frontera queues."""

    _name: str
    _max_duration: str
    _max_nodes: int
    _min_nodes: int


@dataclass(frozen=True)
class _LargeQueue(_BaseQueue):
    """Frontera Large queue."""

    _name: str = "large"
    _max_duration: str = "2-00:00:00"
    _max_nodes: int = 2048
    _min_nodes: int = 513


@dataclass(frozen=True)
class _NormalQueue(_BaseQueue):
    """Frontera Normal queue."""

    _name: str = "normal"
    _max_duration: str = "2-00:00:00"
    _max_nodes: int = 512
    _min_nodes: int = 4


@dataclass(frozen=True)
class _DebugQueue(_BaseQueue):
    """Frontera Debug queue."""

    _name: str = "debug"
    _max_duration: str = "02:00:00"
    _max_nodes: int = 40
    _min_nodes: int = 1


@dataclass(frozen=True)
class _FronteraBaseConfig(BaseSiteConfig):
    _hostname: str = "frontera.tacc.utexas.edu"
    _scheduler: str = "SLURM"
    _mpi_wrapper: str = "ibrun"
    _poll_interval: int = 5
    _sockets_per_node: int = 2
    _gpus_per_node: int = 0
    _cores_per_socket: int = 28
    _memory_per_node: int = 198000
    _account: str = field(default_factory=lambda: os.getenv("TACC_ACCOUNT", ""))


@dataclass
class FronteraConfig:
    """Combines immutable base configuration with queue info for Frontera."""

    _queue: _BaseQueue
    _base_config: _FronteraBaseConfig = _FronteraBaseConfig()

    def __init__(self, queue: str = "debug"):
        if queue == "debug":
            self._queue = _DebugQueue()
        elif queue == "normal":
            self._queue = _NormalQueue()
        elif queue == "large":
            self._queue = _LargeQueue()
        else:
            raise ValueError(f"Invalid queue: {queue}")

    @property
    def hostname(self):
        return self._base_config._hostname

    @property
    def scheduler(self):
        return self._base_config._scheduler

    @property
    def mpi_wrapper(self):
        return self._base_config._mpi_wrapper

    @property
    def poll_interval(self):
        return self._base_config._poll_interval

    @property
    def sockets_per_node(self):
        return self._base_config._sockets_per_node

    @property
    def gpus_per_node(self):
        return self._base_config._gpus_per_node

    @property
    def cores_per_socket(self):
        return self._base_config._cores_per_socket

    @property
    def memory_per_node(self):
        return self._base_config._memory_per_node

    @property
    def account(self):
        return self._base_config._account

    @property
    def queue(self):
        return self._queue._name

    @property
    def max_duration(self):
        return self._queue._max_duration

    @property
    def max_nodes(self):
        return self._queue._max_nodes

    @property
    def min_nodes(self):
        return self._queue._min_nodes

    def validate_request(self, nhost: int, nproc: int, duration: str):
        """Checks that request is within queue parameters"""
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
