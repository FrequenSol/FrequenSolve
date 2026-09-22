"""Execution backend for the imaging API.

The backend sits between :class:`~frequensolve.imaging.jobs.FWIOperatorJob`
and an execution site.  It owns three concerns that the operator layer should
not repeat:

- **Submission.** :class:`Backend` submits jobs with one pinned set of site
  options so every action of one linearization runs with the same rank count
  and mesh partition, names jobs consistently, and waits for job families.
- **Artifact reduction.** Sauce writes one task-suffixed output per frequency
  task (``<stem>_<task><ext>``).  The module-level helpers reduce per-task
  covectors with the misfit frequency weights, assemble per-task objective
  vectors into one :class:`~frequensolve.imaging.data.DataVector`, and read
  the reports, baselines and registry manifests a job produced.
- **Caching.** :class:`LinearizationCache` keeps the most recent
  linearizations keyed by a content fingerprint and removes evicted job
  directories that live inside the backend work directory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from frequensolve.imaging._artifacts import (
    ControlRegistryManifest,
    ControlStateFile,
    ControlVectorFile,
    ObjectiveReport,
)
from frequensolve.imaging.data import DataSpace, DataVector
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.orchestrator.sites.base import BaseSite, RunHandle, RunResult
from frequensolve.simulation.jobs.base import BaseJob

__all__ = [
    "Backend",
    "LinearizationCache",
    "LinearizationEntry",
    "content_fingerprint",
    "fingerprint",
    "frequency_weights",
    "read_manifest",
    "read_report",
    "read_smoothed_covector",
    "read_state_output",
    "read_task_objective_vectors",
    "reduce_covectors",
    "total_value",
    "write_task_objective_vectors",
]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert fingerprint parts to canonical JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return float(repr(value)) if np.isfinite(value) else repr(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    to_fs = getattr(value, "to_fs", None)
    if callable(to_fs):
        return _jsonable(to_fs())
    return repr(value)


def fingerprint(**parts: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON of ``parts``.

    Parts typically name the problem identity, the control state digest, the
    active blocks, the frequencies, the misfit payload, and the content
    fingerprints of the observed data (see :func:`content_fingerprint`).
    NumPy arrays, paths, complex numbers, and objects exposing ``to_fs`` are
    normalized before hashing.
    """

    if not parts:
        raise ValueError("fingerprint requires at least one part")
    text = json.dumps(
        _jsonable(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_fingerprint(path: Union[str, Path]) -> Dict[str, Any]:
    """Hash one required file or directory tree exactly as job inputs are hashed."""

    return BaseJob._path_content_fingerprint(path)


# ---------------------------------------------------------------------------
# Frequency weights and per-task helpers
# ---------------------------------------------------------------------------


def frequency_weights(
    job: FWIOperatorJob, weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Return one finite nonnegative weight per frequency task.

    ``weights`` defaults to the job's own postprocess weights and then to
    ones, matching Sauce's unweighted sum.
    """

    values = job.weights if weights is None else weights
    if values is None:
        return np.ones(job.n_tasks, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != job.n_tasks:
        raise ValueError(
            f"expected {job.n_tasks} frequency weights, received {array.size}"
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("frequency weights must be finite and nonnegative")
    return array


def _tasks(job: FWIOperatorJob) -> range:
    return range(1, job.n_tasks + 1)


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{what} is missing: {path}")
    return path


# ---------------------------------------------------------------------------
# Covectors
# ---------------------------------------------------------------------------


def reduce_covectors(
    job: FWIOperatorJob, weights: Optional[Sequence[float]] = None
) -> ControlVectorFile:
    """Reduce the per-task covector parts of ``job`` with frequency weights.

    Blocks are summed with ``weights`` (default: the job weights, then ones)
    and support masks are combined by AND across tasks.  Sauce fingerprints
    every task's saved state on its own (per-frequency mesh adaptation makes
    them differ), so only the block layout must agree; the reduced file
    carries task 1's fingerprints.
    """

    scale = frequency_weights(job, weights)
    first: Optional[ControlVectorFile] = None
    blocks: Dict[str, np.ndarray] = {}
    support: Dict[str, np.ndarray] = {}
    for task in _tasks(job):
        path = _require_file(job.covector_file(task), f"task {task} covector")
        part = ControlVectorFile.read(path, native=False)
        if first is None:
            first = part
            blocks = {
                name: scale[task - 1] * values for name, values in part.blocks.items()
            }
            support = {name: part.support_mask(name) for name in part.blocks}
            continue
        if part.names != first.names or part.sizes != first.sizes:
            raise ValueError(f"{path} has a different block layout than task 1")
        for name, values in part.blocks.items():
            blocks[name] = blocks[name] + scale[task - 1] * values
            support[name] = support[name] & part.support_mask(name)
    assert first is not None
    return ControlVectorFile(
        blocks,
        state_fingerprint=first.state_fingerprint,
        control_registry_fingerprint=first.control_registry_fingerprint,
        support={name: mask for name, mask in support.items() if not mask.all()},
        support_min_support=first.support_min_support,
    )


def read_smoothed_covector(
    job: FWIOperatorJob, *, raw: bool = False
) -> ControlVectorFile:
    """Read the smoothed aggregate covector (or the ``_raw`` weighted sum).

    Sauce's ``--smooth`` postprocess writes the weighted, smoothed aggregate at
    the unsuffixed covector path and the unsmoothed aggregate beside it.
    """

    if not job.requires_postprocess():
        raise ValueError("job has no smoothing postprocess; use reduce_covectors")
    path = _require_file(
        job.covector_file(raw=True) if raw else job.covector_file(),
        "raw aggregate covector" if raw else "smoothed covector",
    )
    return ControlVectorFile.read(path, native=False)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def read_report(job: FWIOperatorJob) -> List[ObjectiveReport]:
    """Read the per-task ``fs-objective-report-1`` reports in task order."""

    return [
        ObjectiveReport.load(
            _require_file(job.report_file(task), f"task {task} report")
        )
        for task in _tasks(job)
    ]


def total_value(
    reports: Sequence[ObjectiveReport], weights: Optional[Sequence[float]] = None
) -> float:
    """Return the frequency-weighted sum of per-task report totals."""

    if not reports:
        raise ValueError("total_value requires at least one report")
    scale = (
        np.ones(len(reports), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if scale.size != len(reports):
        raise ValueError(
            f"expected {len(reports)} frequency weights, received {scale.size}"
        )
    return float(np.dot(scale, [report.total for report in reports]))


# ---------------------------------------------------------------------------
# Objective vectors
# ---------------------------------------------------------------------------


def write_task_objective_vectors(
    job: FWIOperatorJob,
    data_vector: Union[DataVector, np.ndarray],
    space: DataSpace,
    state_fingerprint: str,
    *,
    n_ranks: int = 1,
) -> List[Path]:
    """Write one ``fs-objective-vector-3`` per frequency task of ``job``.

    Task ``t`` receives the rows of ``space`` at ``job.f_list[t - 1]``; the
    files are written at ``job.objective_vector_file(t)``.
    """

    vector = (
        data_vector
        if isinstance(data_vector, DataVector)
        else DataVector(data_vector, space)
    )
    if vector.space != space:
        raise ValueError("data_vector belongs to a different DataSpace")
    paths = []
    for task in _tasks(job):
        layouts = space.term_layouts(frequency=job.f_list[task - 1])
        paths.append(
            vector.write_objective_vector(
                job.objective_vector_file(task),
                state_fingerprint=state_fingerprint,
                term_layout=layouts,
                n_ranks=n_ranks,
            )
        )
    return paths


def read_task_objective_vectors(
    job: FWIOperatorJob,
    space: DataSpace,
    *,
    state_fingerprint: Optional[str] = None,
    verify: bool = True,
) -> DataVector:
    """Assemble the per-task objective vectors of ``job`` into one vector.

    Each task file fills the frequency block of ``space`` at
    ``job.f_list[t - 1]``; entries of frequencies that are not in ``job`` stay
    zero.
    """

    values = np.zeros(space.size, dtype=space.dtype)
    for task in _tasks(job):
        path = _require_file(
            job.objective_vector_file(task), f"task {task} objective vector"
        )
        part = DataVector.read_objective_vector(
            path,
            space,
            frequency=job.f_list[task - 1],
            verify=verify,
            state_fingerprint=state_fingerprint,
        )
        values += part.values
    return DataVector(values, space)


# ---------------------------------------------------------------------------
# Baseline and registry
# ---------------------------------------------------------------------------


def read_state_output(job: FWIOperatorJob) -> ControlStateFile:
    """Read the ``fs-control-state-1`` baseline exported by ``job``."""

    return ControlStateFile.read(
        _require_file(job.state_output_file(), "control state output")
    )


def read_manifest(job: FWIOperatorJob) -> ControlRegistryManifest:
    """Read the ``fs-control-registry-1`` manifest exported by ``job``."""

    return ControlRegistryManifest.load(
        _require_file(job.manifest_file(), "control registry manifest")
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class Backend:
    """Submit imaging jobs to one site with pinned submission options.

    Args:
        site: Execution site.
        workdir: Directory owned by the backend for staged inputs and cache
            bookkeeping.  Cache eviction deletes job directories only when
            they live inside it.
        submit_options: Site submission keyword arguments applied to every
            submission (rank count, partition, validation flags, ...).  Every
            action of one linearization must use the same profile so Sauce
            states and objective vectors stay partition-compatible.
        prefix: Job name prefix; names are ``f"{prefix}_{counter:04d}"``.
    """

    def __init__(
        self,
        site: BaseSite,
        workdir: Union[str, Path],
        *,
        submit_options: Optional[Mapping[str, Any]] = None,
        prefix: str = "imaging",
    ) -> None:
        self.site = site
        self.workdir = Path(workdir).expanduser().resolve()
        self.submit_options: Dict[str, Any] = dict(submit_options or {})
        for key in ("check", "postprocess_only"):
            if key in self.submit_options:
                raise ValueError(f"submit_options cannot pin {key!r}; run() owns it")
        self.prefix = str(prefix).strip()
        if not self.prefix:
            raise ValueError("Backend prefix must be non-empty")
        self._counter = itertools.count(1)

    # -- naming and staging ---------------------------------------------------

    def job_name(self, kind: Optional[str] = None) -> str:
        """Return the next job name, optionally tagged with an action kind."""

        counter = next(self._counter)
        if kind:
            return f"{self.prefix}_{kind}_{counter:04d}"
        return f"{self.prefix}_{counter:04d}"

    def staging_dir(self, *parts: str) -> Path:
        """Return (and create) a directory for staged inputs under ``workdir``."""

        path = self.workdir.joinpath("inputs", *parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def owns(self, path: Union[str, Path]) -> bool:
        """Return whether ``path`` lies inside the backend work directory."""

        try:
            Path(path).expanduser().resolve().relative_to(self.workdir)
        except ValueError:
            return False
        return True

    # -- execution ------------------------------------------------------------

    def _options(self, *, postprocess_only: bool) -> Dict[str, Any]:
        options = dict(self.submit_options)
        if postprocess_only:
            options["postprocess_only"] = True
        return options

    def submit(self, job: BaseJob, *, postprocess_only: bool = False) -> RunHandle:
        """Submit ``job`` with the pinned options and return its handle."""

        return self.site.submit(job, **self._options(postprocess_only=postprocess_only))

    def run(
        self,
        job: BaseJob,
        *,
        postprocess_only: bool = False,
        check: bool = True,
    ) -> RunResult:
        """Submit ``job`` and wait for it.

        Args:
            job: Job to run.
            postprocess_only: Re-run only the site postprocess (Sauce's
                ``--smooth``) over existing task parts.
            check: Raise :class:`~frequensolve.orchestrator.sites.base.RunFailedError`
                when the run does not succeed.
        """

        return self.submit(job, postprocess_only=postprocess_only).wait(check=check)

    def run_many(
        self, jobs: Iterable[BaseJob], *, check: bool = True
    ) -> List[RunResult]:
        """Submit every job before waiting, returning results in input order.

        Siblings of one family run concurrently.  A site that shuts its
        cluster down when a run completes (``LocalSite`` by default) is asked
        to keep it for the whole family, and closed once afterwards, so the
        first sibling to finish cannot cancel the others.
        """

        jobs = list(jobs)
        if not jobs:
            return []
        site = self.site
        keep_cluster = bool(getattr(site, "shutdown_on_completion", False))
        extra = {"shutdown_on_completion": False} if keep_cluster else {}
        try:
            handles = [
                site.submit(job, **self._options(postprocess_only=False), **extra)
                for job in jobs
            ]
            return site.wait_all(handles, check=check)
        finally:
            if keep_cluster:
                site.close(wait=True, retire=True)

    def dry_run(self, job: BaseJob) -> Dict[str, Any]:
        """Describe what :meth:`run` would submit without touching the site.

        Returns a JSON-compatible mapping with the job name, workflow, action,
        frequencies, the resolved output paths and the solver payload.
        """

        payload: Dict[str, Any] = {
            "name": job.name,
            "site": type(self.site).__name__,
            "workflow": job.workflow,
            "simulation": getattr(job.simulation, "name", None),
            "frequencies": _jsonable(list(job.f_list)),
            "n_tasks": job.n_tasks,
            "submit_options": _jsonable(self.submit_options),
            "requires_postprocess": bool(job.requires_postprocess()),
            "job": _jsonable(job.to_fs()),
        }
        if isinstance(job, FWIOperatorJob):
            payload["action"] = job.action
            payload["active"] = list(job.active or [])
            outputs: Dict[str, Any] = {}
            for label in ("state", "covector", "objective_vector"):
                value = getattr(job, label)
                if value is None:
                    continue
                if label == "state" and job.action != "linearize":
                    continue
                if label == "objective_vector" and job.action != "jvp":
                    continue
                stem = Path(value)
                outputs[label] = [
                    str(stem.with_name(f"{stem.stem}_{task}{stem.suffix}"))
                    for task in _tasks(job)
                ]
            if job.objective is not None or (
                job.action == "linearize" and job.state is not None
            ):
                outputs["objective"] = [
                    str(job.report_file(task)) for task in _tasks(job)
                ]
            if job.requires_postprocess() and job.covector is not None:
                outputs["smoothed_covector"] = str(job.covector_file())
                outputs["raw_covector"] = str(job.covector_file(raw=True))
            for label in ("state_output", "manifest"):
                value = getattr(job, label)
                if value is not None:
                    outputs[label] = str(value)
            payload["outputs"] = outputs
        return payload


# ---------------------------------------------------------------------------
# Linearization cache
# ---------------------------------------------------------------------------


@dataclass
class LinearizationEntry:
    """Paths and identity of one saved ``linearize`` run.

    Args:
        fingerprint: Cache key (see :func:`fingerprint`).
        job: The ``linearize`` job that produced the state.
        directory: Directory removed when the entry is evicted, provided it
            lies inside the cache work directory.  Defaults to the job's
            result directory.
        state: State stem (``job.state_file()``).
        report: Optional report stem.
        covector: Optional covector stem.
        manifest: Optional ``fs-control-registry-1`` path.
        state_output: Optional ``fs-control-state-1`` path.
        state_fingerprint: Sauce state fingerprint every jvp/vjp/normal input
            must carry.
        control_registry_fingerprint: Sauce control registry fingerprint.
        extra: Free-form bookkeeping for the operator layer.
    """

    fingerprint: str
    job: FWIOperatorJob
    directory: Optional[Path] = None
    state: Optional[Path] = None
    report: Optional[Path] = None
    covector: Optional[Path] = None
    manifest: Optional[Path] = None
    state_output: Optional[Path] = None
    state_fingerprint: Optional[str] = None
    control_registry_fingerprint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.fingerprint).strip():
            raise ValueError("LinearizationEntry requires a fingerprint")
        if self.directory is None:
            self.directory = Path(self.job._result_path)
        if self.state is None and self.job.state is not None:
            self.state = self.job.state_file()
        if self.report is None and self.job.objective is not None:
            self.report = self.job.report_file()
        if self.covector is None and self.job.covector is not None:
            self.covector = self.job.covector_file()
        if self.manifest is None and self.job.manifest is not None:
            self.manifest = self.job.manifest_file()
        if self.state_output is None and self.job.state_output is not None:
            self.state_output = self.job.state_output_file()

    @classmethod
    def from_job(
        cls, fingerprint: str, job: FWIOperatorJob, **extra: Any
    ) -> "LinearizationEntry":
        """Build an entry from a finished ``linearize`` job.

        The Sauce fingerprints are read from the task 1 covector when the job
        wrote one.
        """

        if job.action != "linearize":
            raise ValueError("LinearizationEntry requires a linearize job")
        state_fp: Optional[str] = None
        registry_fp: Optional[str] = None
        if job.covector is not None and job.covector_file(1).is_file():
            part = ControlVectorFile.read(job.covector_file(1), native=False)
            state_fp = part.state_fingerprint
            registry_fp = part.control_registry_fingerprint
        return cls(
            fingerprint=fingerprint,
            job=job,
            state_fingerprint=state_fp,
            control_registry_fingerprint=registry_fp,
            extra=dict(extra),
        )


class LinearizationCache:
    """Least-recently-used cache of :class:`LinearizationEntry` objects.

    Args:
        workdir: Directory the cache is allowed to delete inside.  Evicted
            entries whose ``directory`` lies elsewhere are forgotten but kept
            on disk.
        capacity: Maximum number of live entries.
    """

    def __init__(self, workdir: Union[str, Path], capacity: int = 2) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("LinearizationCache capacity must be at least 1")
        self._entries: "OrderedDict[str, LinearizationEntry]" = OrderedDict()
        self.evicted: List[str] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._entries))

    def keys(self) -> List[str]:
        """Return fingerprints from least to most recently used."""

        return list(self._entries)

    def get(self, key: str) -> Optional[LinearizationEntry]:
        """Return the entry for ``key`` and mark it most recently used."""

        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, entry: LinearizationEntry) -> List[LinearizationEntry]:
        """Insert ``entry`` as most recently used and return evicted entries."""

        self._entries.pop(entry.fingerprint, None)
        self._entries[entry.fingerprint] = entry
        evicted = []
        while len(self._entries) > self.capacity:
            _, old = self._entries.popitem(last=False)
            self._remove(old)
            evicted.append(old)
        return evicted

    def evict(self, key: str) -> Optional[LinearizationEntry]:
        """Remove one entry and its owned directory."""

        entry = self._entries.pop(key, None)
        if entry is not None:
            self._remove(entry)
        return entry

    def clear(self) -> List[LinearizationEntry]:
        """Remove every entry and every owned directory."""

        entries = list(self._entries.values())
        self._entries.clear()
        for entry in entries:
            self._remove(entry)
        return entries

    def owns(self, path: Union[str, Path]) -> bool:
        """Return whether ``path`` lies inside the cache work directory."""

        try:
            Path(path).expanduser().resolve().relative_to(self.workdir)
        except ValueError:
            return False
        return True

    def _remove(self, entry: LinearizationEntry) -> None:
        self.evicted.append(entry.fingerprint)
        directory = entry.directory
        if directory is None:
            return
        directory = Path(directory)
        if not directory.exists():
            return
        resolved = directory.resolve()
        if resolved == self.workdir or not self.owns(resolved):
            return
        shutil.rmtree(resolved, ignore_errors=True)
