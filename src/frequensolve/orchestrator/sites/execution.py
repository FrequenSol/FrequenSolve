"""Workflow-scoped execution routing without changing caller-owned sites."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

_EXECUTORS: ContextVar[dict[int, Any]] = ContextVar(
    "frequensolve_executors", default={}
)


def resolve_execution(site: Any) -> Any:
    """Resolve an executor bound to this site in the current workflow context."""
    return _EXECUTORS.get().get(id(site), site)


@contextmanager
def execution_scope(site: Any, executor: Any) -> Iterator[Any]:
    """Route existing and newly constructed problem views through an executor."""
    token = _EXECUTORS.set({**_EXECUTORS.get(), id(site): executor})
    try:
        yield executor
    finally:
        _EXECUTORS.reset(token)


@dataclass(frozen=True)
class PersistentAllocation:
    """An allocation policy owned by ``FWI.run(execution=...)``.

    Resource defaults come from the problem's SLURM site. ``workers`` may be
    an ``AdaptiveWorkers`` configuration. Construction requires no HPC extras;
    opening it requires a site with persistent-session support.

    ``heartbeat_timeout`` is how long the allocation controller's heartbeat
    may stop advancing (client monotonic clock, while Slurm reports the job
    running) before its runs fail. ``"auto"`` uses the site's
    ``scheduler_heartbeat_timeout`` (default 300 s) as configured; seconds
    override it and ``None`` disables it.
    ``cleanup_timeout`` caps draining on normal exit; ``None`` waits while the
    controller is alive. A drain that reaches the cap is left to the lease and
    walltime with a warning rather than cancelled mid-write.
    ``session_command_timeout`` bounds each session filesystem command on the
    login node; ``"auto"`` uses the site's ``session_command_timeout``
    (default 120 s).

    ``reuse`` (a project or a string) adopts a compatible, unattached session
    started earlier with the same ``reuse`` instead of queueing a new
    allocation; ``share=True`` also joins one in use. ``linger`` (seconds or a
    duration such as ``"30m"``; default 0) keeps the allocation that long
    once nothing is attached or running, so the workflow detaches instead of
    releasing it; ``min_time_left`` is the least walltime a reused session
    must have left.
    """

    nodes: Optional[int] = None
    ranks_per_node: Optional[int] = None
    duration: Optional[str] = None
    queue: Optional[str] = None
    workers: Any = None
    lease_timeout: float = 300.0
    startup_timeout: float = 1800.0
    cleanup_timeout: Optional[float] = None
    heartbeat_timeout: Union[float, str, None] = "auto"
    session_command_timeout: Union[float, str] = "auto"
    reuse: Any = None
    linger: Union[float, str] = 0
    share: bool = False
    min_time_left: Union[float, str] = 900.0

    def open(self, site: Any) -> Any:
        factory = getattr(site, "session", None)
        if not callable(factory):
            raise NotImplementedError(
                f"{type(site).__name__} does not support persistent allocation sessions"
            )
        resources = {
            key: value
            for key, value in {
                "nodes": self.nodes,
                "ranks_per_node": self.ranks_per_node,
                "duration": self.duration,
                "queue": self.queue,
            }.items()
            if value is not None
        }
        return factory(
            workers=self.workers,
            lease_timeout=self.lease_timeout,
            startup_timeout=self.startup_timeout,
            cleanup_timeout=self.cleanup_timeout,
            heartbeat_timeout=self.heartbeat_timeout,
            session_command_timeout=self.session_command_timeout,
            reuse=self.reuse,
            linger=self.linger,
            share=self.share,
            min_time_left=self.min_time_left,
            **resources,
        )
