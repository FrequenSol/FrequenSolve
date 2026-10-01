"""Workflow-scoped execution routing without changing caller-owned sites."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

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
    """

    nodes: Optional[int] = None
    ranks_per_node: Optional[int] = None
    duration: Optional[str] = None
    queue: Optional[str] = None
    workers: Any = None
    lease_timeout: float = 300.0
    startup_timeout: float = 1800.0
    cleanup_timeout: float = 30.0

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
            **resources,
        )
