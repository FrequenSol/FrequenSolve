"""Resource-pool status objects shared by site implementations."""

from dataclasses import dataclass, field

__all__ = ["PoolStatus", "PoolInfo"]


@dataclass
class PoolStatus:
    """Status returned for a provisioned execution resource pool.

    Args:
        status: Public pool state such as ``"pending"``, ``"running"``, or
            ``"complete"``.
        return_code: Exit code from the provisioning command when available.
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    status: str = "unknown"
    return_code: int = -1
    stdout: str = ""
    stderr: str = ""

    @property
    def is_queued(self) -> bool:
        """Return whether the pool request is queued."""

        return self.status == "pending"

    @property
    def is_running(self) -> bool:
        """Return whether the pool is active."""

        return self.status == "running"

    @property
    def is_complete(self) -> bool:
        """Return whether the pool request has reached a terminal state."""

        return self.status in [
            "complete",
            "completed",
            "failed",
            "timeout",
            "cancelled",
        ]


@dataclass
class PoolInfo:
    """Information about a provisioned resource pool.

    Args:
        id: Scheduler or provider pool id.
        hostnode: Primary host name.
        nhost: Number of hosts.
        nproc: Number of processes.
        ncore: Number of CPU cores.
        start_time: Provider-reported start time.
        end_time: Provider-reported end time.
    """

    id: int = 0
    hostnode: str = ""
    nhost: int = 0
    nproc: int = 0
    ncore: int = 0
    start_time: str = ""
    end_time: str = ""
    _status: PoolStatus = field(default_factory=PoolStatus)

    @property
    def status(self) -> str:
        """Return the public pool state string."""

        return self._status.status

    @property
    def is_queued(self) -> bool:
        """Return whether the pool request is queued."""

        return self._status.is_queued

    @property
    def is_running(self) -> bool:
        """Return whether the pool is active."""

        return self._status.is_running

    @property
    def is_complete(self) -> bool:
        """Return whether the pool request has reached a terminal state."""

        return self._status.is_complete
