"""Reduction of the per-task covector sets of a direction-set ``normal`` job.

A direction set (``fs-control-vector-set-1``) carries every direction of a job
once for all frequency tasks; each task writes one covector set whose rows are
that task's covectors. :func:`reduce_covector_sets` sums the tasks row by row,
as :func:`frequensolve.imaging._backend.reduce_covectors` does for one covector.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from frequensolve.imaging._artifacts import ControlVectorFile, ControlVectorSetFile
from frequensolve.imaging._backend import _require_file, _tasks, frequency_weights
from frequensolve.imaging.jobs import FWIOperatorJob

__all__ = ["reduce_covector_sets"]


def reduce_covector_sets(
    job: FWIOperatorJob,
    weights: Optional[Sequence[float]] = None,
    factors: Optional[Sequence[Mapping[str, float]]] = None,
) -> List[ControlVectorFile]:
    """Reduce each task's covector set of ``job`` to one covector per direction.

    Rows are summed over tasks with ``weights`` (default: the job weights,
    then ones). ``factors`` (one ``block -> factor`` mapping per task) also
    multiplies a task's block: Sauce writes mechanism covectors in the
    executing task's coordinates, converted to the reference ones with
    ``s_ref / s_task``. Every task must hold the same rows, block layout,
    material bases and control identity; support masks combine by AND, and
    each covector carries task 1's fingerprints.
    """

    weight = frequency_weights(job, weights)
    if factors is not None and len(factors) != job.n_tasks:
        raise ValueError(
            f"expected {job.n_tasks} per-task block factor tables, "
            f"received {len(factors)}"
        )

    def scale(task: int, name: str) -> float:
        factor = 1.0 if factors is None else float(factors[task - 1].get(name, 1.0))
        return float(weight[task - 1]) * factor

    first: Optional[ControlVectorSetFile] = None
    rows: Dict[str, np.ndarray] = {}
    support: Dict[str, np.ndarray] = {}
    for task in _tasks(job):
        path = _require_file(job.covector_set_file(task), f"task {task} covector set")
        part = ControlVectorSetFile.read(path)
        if first is None:
            first = part
            rows = {name: scale(task, name) * values for name, values in part.rows.items()}
            support = {
                name: part.support.get(name, np.ones(values.shape[1], dtype=bool))
                for name, values in part.rows.items()
            }
            continue
        if part.n_vectors != first.n_vectors or part.sizes != first.sizes:
            raise ValueError(f"{path} has a different layout than task 1")
        if part.control_spaces != first.control_spaces:
            raise ValueError(f"{path} has different material basis identities than task 1")
        if part.control_identity != first.control_identity:
            raise ValueError(f"{path} has a different control identity than task 1")
        for name, values in part.rows.items():
            rows[name] = rows[name] + scale(task, name) * values
            if name in part.support:
                support[name] = support[name] & part.support[name]
    assert first is not None
    return [
        ControlVectorFile(
            {name: values[i] for name, values in rows.items()},
            state_fingerprint=first.state_fingerprint,
            control_registry_fingerprint=first.control_registry_fingerprint,
            support={name: mask for name, mask in support.items() if not mask.all()},
            support_min_support=first.support_min_support,
            control_spaces=dict(first.control_spaces),
            control_identity=first.control_identity,
        )
        for i in range(first.n_vectors)
    ]
