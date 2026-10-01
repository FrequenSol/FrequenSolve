"""Run consumers of partitioned solver artifacts on their producer's MPI ranks.

Sauce partitions several ``fwi_operator`` artifacts by MPI rank and records
the rank count of the task that wrote them:

- the objective state a ``linearize`` task saves
  (``fs-objective-linearization-3``) and objective vectors
  (``fs-objective-vector-3``), in ``partition.n_ranks``;
- receiver states and vectors (``fs-receiver-state-1``,
  ``fs-receiver-vector-1``, or a ``fs-receiver-*-bundle-1`` of per-group
  files), in ``partition.n_ranks``;
- background checkpoints (``fs-background-state-1``), in ``ranks``.

Every later task reading one of them must run on the recorded count. Sauce
rejects another count for states, vectors and receiver states ("Objective
state requires the saved mesh partition and MPI rank count") and recomputes a
background instead of restoring it. Sites therefore pin such tasks to the
recorded count and size all other tasks as usual. Partition-independent
(canonical ``-4``) objective artifacts impose no count.

Sauce resolves an input of a multi-task job as its ``<stem>_<task><ext>``
sibling when that file exists and as the named file otherwise; a single-task
job reads the named file.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "TaskPartition",
    "partitioned_inputs",
    "recorded_ranks",
    "task_partitions",
]

logger = logging.getLogger(__name__)

# (job attribute, actions that write it rather than read it, whether Sauce
# fails on another rank count rather than only recomputing).
_INPUTS: Tuple[Tuple[str, frozenset[str], bool], ...] = (
    ("state", frozenset({"linearize"}), True),
    ("objective_vector", frozenset({"jvp"}), True),
    ("receiver_state", frozenset({"receiver_linearize"}), True),
    ("receiver_vector", frozenset({"receiver_jvp"}), True),
    ("background", frozenset({"linearize"}), False),
)

RemoteReader = Callable[[Mapping[Hashable, Sequence[Path]]], Mapping[Hashable, Any]]
"""Read manifests a site keeps elsewhere.

Called with local candidate paths per key, in lookup order; returns the
parsed JSON of the first candidate found, keyed alike.
"""


@dataclass(frozen=True)
class TaskPartition:
    """The MPI rank count one task must run on.

    Attributes:
        ranks: Rank count recorded by the producing task.
        required: Whether Sauce fails on another count; otherwise only a
            background checkpoint would be recomputed.
        inputs: Job inputs that recorded the count.
    """

    ranks: int
    required: bool
    inputs: Tuple[str, ...]


@dataclass(frozen=True)
class _Input:
    task: int
    name: str
    required: bool
    candidates: Tuple[Path, ...]


def partitioned_inputs(job: Any) -> List[_Input]:
    """Return every partitioned input ``job`` reads, per one-based task."""

    if getattr(job, "workflow", None) != "fwi_operator":
        return []
    action = getattr(job, "action", None)
    n_tasks = int(getattr(job, "n_tasks", 1) or 1)
    inputs = []
    for name, writers, required in _INPUTS:
        value = getattr(job, name, None)
        if action in writers or not isinstance(value, (str, os.PathLike)):
            continue
        path = Path(value)
        for task in range(1, n_tasks + 1):
            sibling = path.with_name(f"{path.stem}_{task}{path.suffix}")
            candidates = (path,) if n_tasks == 1 else (sibling, path)
            inputs.append(_Input(task, name, required, candidates))
    return inputs


def recorded_ranks(payload: Any) -> Optional[int]:
    """Return the MPI rank count a partitioned manifest records, if any."""

    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema") == "fs-background-state-1":
        value = payload.get("ranks")
    else:
        partition = payload.get("partition")
        if (
            not isinstance(partition, Mapping)
            or partition.get("compatibility") != "same_mesh_partition"
        ):
            return None
        value = partition.get("n_ranks")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _bundle_member(payload: Any, manifest: Path) -> Optional[Tuple[Path, ...]]:
    """Return candidate paths of a receiver bundle's first group file."""

    if not isinstance(payload, Mapping) or not str(payload.get("schema", "")).endswith(
        "-bundle-1"
    ):
        return None
    groups = payload.get("groups")
    if isinstance(groups, Mapping):
        groups = [groups[key] for key in sorted(groups, key=str)]
    first = groups[0] if isinstance(groups, list) and groups else None
    name = first.get("file") if isinstance(first, Mapping) else None
    if not isinstance(name, str) or not name:
        return None
    member = Path(name)
    if member.is_absolute():
        # Members are named as Sauce wrote them; a fetched copy sits beside.
        return (member, manifest.parent / member.name)
    return (manifest.parent / member,)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read(
    requests: Mapping[Hashable, Sequence[Path]],
    read_remote: Optional[RemoteReader],
) -> Dict[Hashable, Tuple[Any, Path]]:
    """Read each key's first existing candidate, locally then remotely."""

    found: Dict[Hashable, Tuple[Any, Path]] = {}
    missing: Dict[Hashable, Sequence[Path]] = {}
    for key, candidates in requests.items():
        for candidate in candidates:
            payload = _read_json(candidate)
            if payload is not None:
                found[key] = (payload, candidate)
                break
        else:
            missing[key] = candidates
    if missing and read_remote is not None:
        for key, payload in read_remote(missing).items():
            if key in missing and payload is not None:
                # Relative members resolve against the named manifest.
                found[key] = (payload, Path(missing[key][-1]))
    return found


def task_partitions(
    job: Any,
    tasks: Optional[Iterable[int]] = None,
    *,
    read_remote: Optional[RemoteReader] = None,
) -> Dict[int, TaskPartition]:
    """Return the rank count each task of ``job`` must run on.

    Args:
        job: Job whose partitioned inputs are read.
        tasks: One-based tasks to consider (default: every task).
        read_remote: Optional reader for manifests not available locally,
            called at most twice (receiver bundles name their group files).

    Tasks without a readable partitioned input are omitted; the site sizes
    them as usual and Sauce reports any missing input.
    """

    inputs = partitioned_inputs(job)
    if tasks is not None:
        wanted = {int(task) for task in tasks}
        inputs = [item for item in inputs if item.task in wanted]
    found = _read(
        {(item.task, item.name): item.candidates for item in inputs}, read_remote
    )
    members = {}
    for key, (payload, manifest) in found.items():
        candidates = _bundle_member(payload, manifest)
        if candidates is not None:
            members[key] = candidates
    for key, value in _read(members, read_remote).items():
        found[key] = value
    recorded: Dict[int, List[Tuple[str, bool, int]]] = {}
    for item in inputs:
        entry = found.get((item.task, item.name))
        ranks = None if entry is None else recorded_ranks(entry[0])
        if ranks is not None:
            recorded.setdefault(item.task, []).append((item.name, item.required, ranks))
    partitions: Dict[int, TaskPartition] = {}
    for task, entries in sorted(recorded.items()):
        required = [entry for entry in entries if entry[1]]
        name, _, ranks = (required or entries)[0]
        if any(entry[2] != ranks for entry in entries):
            logger.warning(
                "Task %d inputs record different MPI rank counts (%s); using %d from %s",
                task,
                ", ".join(f"{entry[0]}={entry[2]}" for entry in entries),
                ranks,
                name,
            )
        partitions[task] = TaskPartition(
            ranks=ranks,
            required=bool(required),
            inputs=tuple(entry[0] for entry in entries),
        )
    return partitions
