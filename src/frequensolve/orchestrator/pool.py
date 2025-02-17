from dataclasses import dataclass, field

__all__ = ["PoolStatus", "PoolInfo"]


@dataclass
class PoolStatus:
    status: str = "unknown"
    return_code: int = -1
    stdout: str = ""
    stderr: str = ""

    @property
    def is_queued(self) -> bool:
        return self.status == "pending"

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def is_complete(self) -> bool:
        return self.status in ["completed", "failed"]


@dataclass
class PoolInfo:
    """Information about the resource pool."""

    id: int = 0
    hostnode: str = ""
    nhost: int = 0
    nproc: int = 0
    ncore: int = 0
    start_time: str = ""
    end_time: str = ""
    _status: PoolStatus = field(default_factory=PoolStatus)

    @property
    def status(self):
        return self._status.status

    @property
    def is_queued(self) -> bool:
        return self._status.is_queued

    @property
    def is_running(self) -> bool:
        return self._status.is_running

    @property
    def is_complete(self) -> bool:
        return self._status.is_complete
