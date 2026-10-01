"""Execution backend for the imaging API.

The backend sits between :class:`~frequensolve.imaging.jobs.FWIOperatorJob`
and an execution site.  It owns three concerns that the operator layer should
not repeat:

- **Submission.** :class:`Backend` submits jobs with one pinned set of site
  options, stages client-written operator inputs on remote sites, names jobs
  consistently, and waits for job families.  Saved states and objective
  vectors do not depend on the MPI rank count.
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

import functools
import hashlib
import inspect
import itertools
import json
import shutil
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from frequensolve.imaging._artifacts import (
    ControlRegistryManifest,
    ControlStateFile,
    ControlVectorFile,
    ObjectiveReport,
)
from frequensolve.imaging._block_digest import block_digest
from frequensolve.imaging.data import DataSpace, DataVector
from frequensolve.imaging.jobs import FWIOperatorJob, _task_path
from frequensolve.orchestrator.sites.base import BaseSite, RunHandle, RunResult
from frequensolve.orchestrator.sites.execution import resolve_execution
from frequensolve.project import Project
from frequensolve.simulation.jobs.base import BaseJob

__all__ = [
    "Backend",
    "LinearizationCache",
    "LinearizationEntry",
    "background_checkpoint_files",
    "content_fingerprint",
    "fingerprint",
    "frequency_weights",
    "read_manifest",
    "read_report",
    "read_smoothed_covector",
    "read_state_output",
    "read_task_objective_vectors",
    "reduce_covectors",
    "task_inputs",
    "total_value",
    "write_task_objective_vectors",
]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


# Arrays with at least this many elements are identified by their canonical
# bytes instead of JSON lists (see :func:`fingerprint`); smaller ones keep the
# JSON form, so their fingerprints are unchanged.
_ARRAY_DIGEST_SIZE = 4096
# Canonical element type of each array kind on the byte path.
_ARRAY_KINDS = {"b": "bool", "f": "float", "i": "int", "u": "int", "c": "complex"}


def _array_identity(value: np.ndarray) -> Optional[Dict[str, Any]]:
    """Identify a large array by kind, shape and an ``fs-block-sha256-1`` digest.

    Contiguous float64 arrays (and boolean masks, viewed as uint8 0/1 bytes)
    are hashed in place on the shared digest thread pool; other real floats are
    widened to float64, integers to int64, and complex values become float64
    (real, imaginary) pairs. Returns ``None`` for small or other arrays.
    """

    kind = value.dtype.kind
    if (
        value.size < _ARRAY_DIGEST_SIZE
        or isinstance(value, np.ma.MaskedArray)
        or kind not in _ARRAY_KINDS
        or value.dtype.itemsize > (16 if kind == "c" else 8)
        or value.dtype == np.uint64
    ):
        return None
    if kind == "b":
        digest = block_digest(np.ascontiguousarray(value).view(np.uint8), np.uint8)
    elif kind == "c":
        pairs = np.ascontiguousarray(value, dtype=np.complex128).view(np.float64)
        digest = block_digest(pairs, np.float64)
    else:
        digest = block_digest(value, np.float64 if kind == "f" else np.int64)
    return {"fs_array": digest, "kind": _ARRAY_KINDS[kind], "shape": list(value.shape)}


def _jsonable(value: Any) -> Any:
    """Convert fingerprint parts to canonical JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, np.ndarray):
        identity = _array_identity(value)
        if identity is not None:
            return identity
        if value.dtype.kind == "b":
            value = value.astype(np.uint8)  # masks fingerprint as 0/1 on both paths
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
    normalized before hashing. Arrays with at least 4096 elements enter as
    their kind (bool, float, int or complex), shape and ``fs-block-sha256-1``
    digest of canonical bytes: booleans as uint8 0/1, real floats as float64,
    integers as int64 and complex values as float64 pairs. Contiguous float64
    data and masks are hashed in place, in parallel, without JSON lists.
    Smaller arrays enter as JSON values (booleans as 0/1).
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


def task_covectors(
    job: FWIOperatorJob,
    factors: Optional[Sequence[Mapping[str, float]]] = None,
) -> Iterable[ControlVectorFile]:
    """Read each unsummed task covector once, in reference coordinates."""

    if factors is not None and len(factors) != job.n_tasks:
        raise ValueError(f"expected {job.n_tasks} per-task block factor tables")
    first = None
    for task in _tasks(job):
        path = _require_file(job.covector_file(task), f"task {task} covector")
        part = ControlVectorFile.read(path, native=False)
        if first is None:
            first = (part.names, part.sizes)
        elif (part.names, part.sizes) != first:
            raise ValueError(f"{path} has a different block layout than task 1")
        if factors is not None:
            for name in part.blocks:
                part.blocks[name] = part.blocks[name] * float(
                    factors[task - 1].get(name, 1.0)
                )
        yield part


def reduce_covectors(
    job: FWIOperatorJob,
    weights: Optional[Sequence[float]] = None,
    factors: Optional[Sequence[Mapping[str, float]]] = None,
    *,
    output: Optional[int] = None,
) -> ControlVectorFile:
    """Reduce the per-task covector parts of ``job`` with frequency weights.

    Blocks are summed with ``weights`` (default: the job weights, then ones)
    and support masks are combined by AND across tasks.  ``factors`` (one
    ``block -> factor`` mapping per task) additionally multiplies a task's
    block before summation: Sauce writes mechanism covectors in the executing
    task's coordinates, and FrequenSolve converts each task's part to its
    reference coordinates with ``s_ref / s_task``.  Sauce fingerprints every
    task's saved state on its own (per-frequency mesh adaptation makes them
    differ), so the block layout and material-basis identities must agree;
    the reduced file preserves those identities and carries task 1's fingerprints.
    ``output`` selects the covector of one entry of a multi-direction normal.
    """

    weight = frequency_weights(job, weights)
    if job.action == "wri":
        weight = job.wri_reduction_weights(weight)
    if factors is not None and len(factors) != job.n_tasks:
        raise ValueError(
            f"expected {job.n_tasks} per-task block factor tables, "
            f"received {len(factors)}"
        )

    def scale(task: int, name: str) -> float:
        factor = 1.0 if factors is None else float(factors[task - 1].get(name, 1.0))
        return float(weight[task - 1]) * factor

    first: Optional[ControlVectorFile] = None
    blocks: Dict[str, np.ndarray] = {}
    support: Dict[str, np.ndarray] = {}
    for task in _tasks(job):
        path = _require_file(
            job.covector_file(task, output=output), f"task {task} covector"
        )
        part = ControlVectorFile.read(path, native=False)
        if first is None:
            first = part
            blocks = {
                name: scale(task, name) * values for name, values in part.blocks.items()
            }
            support = {name: part.support_mask(name) for name in part.blocks}
            continue
        if part.names != first.names or part.sizes != first.sizes:
            raise ValueError(f"{path} has a different block layout than task 1")
        if part.control_spaces != first.control_spaces:
            raise ValueError(
                f"{path} has different material basis identities than task 1"
            )
        for name, values in part.blocks.items():
            blocks[name] = blocks[name] + scale(task, name) * values
            support[name] = support[name] & part.support_mask(name)
    assert first is not None
    return ControlVectorFile(
        blocks,
        state_fingerprint=first.state_fingerprint,
        control_registry_fingerprint=first.control_registry_fingerprint,
        support={name: mask for name, mask in support.items() if not mask.all()},
        support_min_support=first.support_min_support,
        control_spaces=dict(first.control_spaces),
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


def task_inputs(stem: Path, n_tasks: int) -> List[Tuple[int, Path]]:
    """Return ``(task, path)`` of each task's copy of a per-task operator input.

    A job with several frequency tasks names ``stem`` and Sauce reads
    ``<stem>_<task><ext>`` in task ``task``; a single-task job reads ``stem``
    exactly.
    """

    if n_tasks == 1:
        return [(1, stem)]
    return [
        (task, stem.with_name(f"{stem.stem}_{task}{stem.suffix}"))
        for task in range(1, n_tasks + 1)
    ]


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
    state_fingerprint: Optional[Union[str, Sequence[Optional[str]]]] = None,
    verify: bool = True,
) -> DataVector:
    """Assemble the per-task objective vectors of ``job`` into one vector.

    Each task file fills the frequency block of ``space`` at
    ``job.f_list[t - 1]``; entries of frequencies that are not in ``job`` stay
    zero.  ``state_fingerprint`` is one fingerprint for every task or one per
    task (Sauce fingerprints each frequency task's saved state on its own).
    """

    if state_fingerprint is None or isinstance(state_fingerprint, str):
        expected: List[Optional[str]] = [state_fingerprint] * job.n_tasks
    else:
        expected = list(state_fingerprint)
        if len(expected) != job.n_tasks:
            raise ValueError(
                f"expected {job.n_tasks} state fingerprints, received {len(expected)}"
            )
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
            state_fingerprint=expected[task - 1],
        )
        values += part.values
    return DataVector(values, space)


# ---------------------------------------------------------------------------
# Baseline and registry
# ---------------------------------------------------------------------------


def read_state_output(job: FWIOperatorJob, task: int = 1) -> ControlStateFile:
    """Read the ``fs-control-state-1`` baseline exported by ``task`` of ``job``.

    Every task exports the complete baseline in its own mechanism
    coordinates; ``/scaling/<block>`` is that task's scale ``s_t`` (the
    baseline of any task replays in every other one).
    """

    return ControlStateFile.read(
        _require_file(job.state_output_file(task), f"task {task} control state output")
    )


def read_manifest(job: FWIOperatorJob, task: int = 1) -> ControlRegistryManifest:
    """Read the ``fs-control-registry-1`` manifest exported by ``task`` of ``job``."""

    return ControlRegistryManifest.load(
        _require_file(job.manifest_file(task), f"task {task} control registry manifest")
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def _operator_input_files(job: BaseJob) -> List[Path]:
    """Return the existing input files a job reads, including per-task siblings."""

    candidates: List[Any] = [
        getattr(job, name, None)
        for name in ("direction", "objective_vector", "control_state", "input_vector")
    ]
    extension = getattr(job, "extension", None)
    if isinstance(extension, Mapping):
        candidates.append(extension.get("direction"))
    n_tasks = int(getattr(job, "n_tasks", 1) or 1)
    files: List[Path] = []
    for value in candidates:
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value)
        variants = [path, *(_task_path(path, task) for task in range(1, n_tasks + 1))]
        for variant in variants:
            if variant.is_file():
                files.append(variant)
                files.extend(_manifest_payload(variant))
    return list(dict.fromkeys(files))


def _manifest_payload(path: Path) -> List[Path]:
    """Return the data file an objective-vector manifest names beside itself."""

    if path.suffix != ".json":
        return []
    try:
        name = json.loads(path.read_text()).get("file")
    except (OSError, ValueError, AttributeError):
        return []
    if not isinstance(name, str) or not name:
        return []
    payload = Path(name)
    if not payload.is_absolute():
        payload = path.parent / payload
    return [payload] if payload.is_file() else []


class Backend:
    """Submit imaging jobs to one site with pinned submission options.

    Args:
        site: Execution site.
        workdir: Directory owned by the backend for staged inputs and cache
            bookkeeping.  Cache eviction deletes job directories only when
            they live inside it.
        submit_options: Site submission keyword arguments applied to every
            submission (rank count, partition, validation flags, ...).
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
        self._site = site
        self.workdir = Path(workdir).expanduser().resolve()
        self.submit_options: Dict[str, Any] = dict(submit_options or {})
        for key in ("check", "postprocess_only"):
            if key in self.submit_options:
                raise ValueError(f"submit_options cannot pin {key!r}; run() owns it")
        self.prefix = str(prefix).strip()
        if not self.prefix:
            raise ValueError("Backend prefix must be non-empty")
        self._counter = itertools.count(1)
        self._timing_lock = RLock()
        self._worker_seconds = 0.0
        self._unmeasured_runs = 0
        self._recorders: Dict[int, List[BaseJob]] = {}

    @property
    def site(self) -> BaseSite:
        """The executor currently bound to this backend's caller-owned site."""
        return resolve_execution(self._site)

    @site.setter
    def site(self, value: BaseSite) -> None:
        self._site = value

    def timing_snapshot(self) -> Tuple[float, int]:
        """Return counters for measuring a complete stage through this backend."""
        with self._timing_lock:
            return self._worker_seconds, self._unmeasured_runs

    @contextmanager
    def recording(self) -> Iterator[List[BaseJob]]:
        """Collect every job this backend submits inside the ``with`` block.

        Workflows use the list for per-run job counts and native diagnostics;
        nested and concurrent recordings each receive every job.
        """
        jobs: List[BaseJob] = []
        token = object()
        with self._timing_lock:
            self._recorders[id(token)] = jobs
        try:
            yield jobs
        finally:
            with self._timing_lock:
                self._recorders.pop(id(token), None)

    def _record(self, job: BaseJob) -> None:
        with self._timing_lock:
            for jobs in self._recorders.values():
                jobs.append(job)

    def curvature(self, **options: Any) -> Any:
        """Return Sauce curvature postprocessing on this site and workdir.

        Operations run through ``site.run_curvature``. ``options`` override
        that site's resources for every operation of the returned object, for
        example ``ranks``/``threads_per_rank`` on ``LocalSite`` or ``nodes``,
        ``ranks_per_node``, ``queue`` and ``duration`` on SLURM sites. Without
        options the site's defaults apply (``LocalSite.curvature_ranks``,
        ``SlurmSite.curvature_run_config``). Mesh transfer and sampling always
        run on one rank.
        """
        from .curvature import NativeCurvature

        if not getattr(self.site, "supports_curvature", False):
            raise NotImplementedError(
                f"{type(self.site).__name__} does not run Sauce curvature "
                "operations; use LocalSite or a SLURM site."
            )
        runner: Callable[[Path], None] = self.site.run_curvature
        if options:
            try:
                inspect.signature(runner).bind(Path("request.json"), **options)
            except TypeError as error:
                raise TypeError(
                    f"Unsupported curvature options for {type(self.site).__name__}: "
                    f"{error}"
                ) from None
            runner = functools.partial(runner, **options)
        return NativeCurvature(workdir=self.workdir / "curvature", runner=runner)

    def cost_since(self, snapshot: Tuple[float, int]) -> Dict[str, Any]:
        seconds, missing = self.timing_snapshot()
        missing -= snapshot[1]
        return {
            "summed_worker_seconds": None if missing else seconds - snapshot[0],
            "worker_time_scope": "site-reported native initialization, tasks and postprocessing",
            "unmeasured_native_runs": missing,
        }

    def _record_timing(self, result: RunResult, handle: RunHandle) -> None:
        if result.status.state == "skipped":
            return
        raw = result.status.raw or {}
        rows = list(raw.get("tasks", []))
        backend = getattr(handle, "backend", {})
        for name in ("pack", "smooth"):
            if isinstance(raw.get(name), Mapping):
                rows.append(raw[name])
        if isinstance(backend.get("mesh_result"), Mapping):
            rows.append(backend["mesh_result"])
        # A shared-frequency worker reports one row per frequency. Count its
        # common process log only once rather than multiplying its duration.
        seen = set()
        seconds = 0.0
        measured = bool(rows)
        for row in rows:
            duration = row.get("duration_seconds")
            if duration is None:
                measured = False
                continue
            log = row.get("stdout")
            if log is not None and log in seen:
                continue
            if log is not None:
                seen.add(log)
            seconds += float(duration)
        # Serial benchmark sites can report a complete job duration including
        # input staging, initialization and packing through this explicit field.
        if "worker_seconds" in raw:
            seconds, measured = float(raw["worker_seconds"]), True
        with self._timing_lock:
            self._worker_seconds += seconds
            self._unmeasured_runs += int(not measured)

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
        options.setdefault("fetch", True)
        if postprocess_only:
            options["postprocess_only"] = True
        return options

    def submit(self, job: BaseJob, *, postprocess_only: bool = False) -> RunHandle:
        """Submit ``job`` with the pinned options and return its handle."""

        if getattr(job, "frequency_groups", 1) > 1 and not getattr(
            self.site, "supports_frequency_groups", False
        ):
            raise NotImplementedError(
                "This site does not yet launch shared-frequency workers; use LocalSite or the native --frequency-groups launcher"
            )
        self._stage_remote_inputs(job)
        handle = self.site.submit(
            job, **self._options(postprocess_only=postprocess_only)
        )
        self._record(job)
        return handle

    def _stage_remote_inputs(self, job: BaseJob) -> None:
        """Upload operator inputs written on this machine to a remote site.

        Project sync carries only simulations, and remote jobs see their own
        results; client-written directions and objective vectors live under
        the project's ``imaging`` tree and must be uploaded explicitly.  Paths
        are mapped under the remote project root exactly as the job payload
        rewrites them.
        """

        # Local sites read inputs in place; only remote sites expose a work_dir.
        if getattr(self.site, "work_dir", None) is None or not callable(
            getattr(self.site, "put", None)
        ):
            return
        try:
            root = job._project_path()
        except (AttributeError, ValueError):
            return
        remote_root = Path(Project.remote_root_for(self.site, root))
        upload = getattr(self.site, "put")
        input_files = getattr(job, "remote_input_files", None)
        if callable(input_files):
            for local, remote in input_files(remote_root):
                upload(local, remote)
            return
        for path in _operator_input_files(job):
            try:
                relative = path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            upload(path, remote_root / relative)

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

        handle = self.submit(job, postprocess_only=postprocess_only)
        result = handle.wait(check=check)
        self._record_timing(result, handle)
        return result

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
            for job in jobs:
                self._stage_remote_inputs(job)
            handles = [
                site.submit(job, **self._options(postprocess_only=False), **extra)
                for job in jobs
            ]
            for job in jobs:
                self._record(job)
            results = site.wait_all(handles, check=check)
            for result, handle in zip(results, handles):
                self._record_timing(result, handle)
            return results
        finally:
            if keep_cluster:
                site.close(wait=True, retire=True)

    def run_preparation(self, job: BaseJob, *, check: bool = True) -> RunResult:
        """Account geometry-only preparation with the site's default rank profile."""
        handle = self.site.submit(job)
        self._record(job)
        result = handle.wait(check=check)
        self._record_timing(result, handle)
        return result

    def release_background(self, job: FWIOperatorJob) -> int:
        """Delete a linearize job's native background checkpoint everywhere.

        Removes the files :func:`background_checkpoint_files` lists from this
        machine and from the site's copy of the job result directory
        (``site.remove_result_files``), and clears ``job.background``: later
        actions on that linearization run uncached and a new
        ``background=True`` request linearizes again. Returns the released
        size in bytes (estimated for payloads kept only on the site).
        """
        if getattr(job, "background", None) is None:
            return 0
        files, size = background_checkpoint_files(job)
        root = Path(job._result_path)
        # Never reference a checkpoint that may be partially deleted.
        job.background = None
        for name in files:
            path = root / name
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
        if files:
            self.site.remove_result_files(job, files)
        return size

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


_BACKGROUND_ROLES = ("checkpoint", "background_shard", "background_state")


def background_checkpoint_files(job: FWIOperatorJob) -> Tuple[List[str], int]:
    """Return a linearize job's background checkpoint files and their size.

    Paths are relative to the job result directory: each task's
    ``fs-background-state-1`` manifest, the per-rank shards it lists (HDF5
    shards holding every source batch, or, from earlier solver builds, JSON
    shard manifests naming ``<shard>_batch_<b>.h5`` payloads) and every
    task-catalog record with a checkpoint role. The manifests are
    authoritative because earlier builds recorded only rank 0's files in the
    task catalog. Files present on this machine count their size, payloads
    kept only on the site their catalog size, and unrecorded payloads are
    estimated from the recorded ones.
    """
    if getattr(job, "background", None) is None:
        return [], 0
    root = Path(job._result_path).expanduser().resolve(strict=False)
    sizes: Dict[str, Optional[int]] = {}

    def add(path: Path, size: Optional[int] = None) -> Optional[Path]:
        path = Path(path).expanduser().resolve(strict=False)
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            return None
        if name in ("", "."):
            return None
        if path.is_file():
            size = path.stat().st_size
        sizes[name] = size if size is not None else sizes.get(name)
        return path

    for task in range(1, int(getattr(job, "n_tasks", 1) or 1) + 1):
        manifest = add(job.background_file(task))
        if manifest is None:
            continue
        try:
            state = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        shards = state.get("shards") if isinstance(state, Mapping) else None
        if isinstance(shards, Mapping):
            shards = list(shards.values())
        batches = state.get("batches", 0) if isinstance(state, Mapping) else 0
        for shard in shards if isinstance(shards, list) else []:
            name = shard.get("file") if isinstance(shard, Mapping) else None
            if not isinstance(name, str) or not name:
                continue
            path = add(manifest.parent / name)
            if path is None or path.suffix != ".json":
                continue
            # Earlier builds: one payload per batch beside a JSON shard manifest.
            for batch in range(batches if isinstance(batches, int) else 0):
                add(path.parent / f"{path.stem}_batch_{batch}.h5")
            try:
                records = json.loads(path.read_text()).get("shards") or []
            except (OSError, ValueError, AttributeError):
                records = []
            for record in records if isinstance(records, list) else []:
                payload = record.get("file") if isinstance(record, Mapping) else None
                if isinstance(payload, str) and payload:
                    add(path.parent / payload)
    try:
        from frequensolve.simulation.task_index import load_task_catalog

        catalog = load_task_catalog(
            root, tasks=range(1, int(getattr(job, "n_tasks", 1) or 1) + 1)
        )
        for role in _BACKGROUND_ROLES:
            for record in catalog.query(role=role):
                add(record.path, record.bytes)
    except (OSError, ValueError, AttributeError, TypeError):
        pass  # no committed task results on this machine (e.g. test sites)
    payloads = [size for name, size in sizes.items() if name.endswith(".h5")]
    known = [size for size in payloads if size is not None]
    estimate = int(sum(known) / len(known)) if known else 0
    total = sum(size for size in sizes.values() if size is not None)
    total += estimate * sum(size is None for size in payloads)
    return sorted(sizes), int(total)


class LinearizationCache:
    """Least-recently-used cache of :class:`LinearizationEntry` objects.

    Evicted entries release their native background checkpoints through
    ``release`` (wired by the problem to ``Backend.release_background``), as
    do the least recently used entries once the retained checkpoints exceed
    ``background_budget`` bytes; the newest checkpoint is never released by
    the budget, because the workflow that requested it is still using it.

    Args:
        workdir: Directory the cache is allowed to delete inside.  Evicted
            entries whose ``directory`` lies elsewhere are forgotten but kept
            on disk.
        capacity: Maximum number of live entries.
        release: Callback deleting one entry's background checkpoint.
        background_budget: Optional limit on retained checkpoint bytes.
    """

    def __init__(
        self,
        workdir: Union[str, Path],
        capacity: int = 2,
        *,
        release: Optional[Callable[[LinearizationEntry], Any]] = None,
        background_budget: Optional[float] = None,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("LinearizationCache capacity must be at least 1")
        self._entries: "OrderedDict[str, LinearizationEntry]" = OrderedDict()
        self.evicted: List[str] = []
        self.release = release
        self.background_budget = background_budget

    @property
    def background_budget(self) -> Optional[float]:
        """Retained background checkpoint bytes before LRU release (``None``: no limit)."""
        return self._background_budget

    @background_budget.setter
    def background_budget(self, value: Optional[float]) -> None:
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError("background_budget must be finite and nonnegative")
        self._background_budget = value

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

        replaced = self._entries.pop(entry.fingerprint, None)
        if replaced is not None and replaced.job is not entry.job:
            self._release(replaced)
        self._entries[entry.fingerprint] = entry
        evicted = []
        while len(self._entries) > self.capacity:
            _, old = self._entries.popitem(last=False)
            self._remove(old)
            evicted.append(old)
        self._enforce_background_budget(entry)
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

    def _release(self, entry: LinearizationEntry) -> None:
        if self.release is None or getattr(entry.job, "background", None) is None:
            return
        try:
            self.release(entry)
        except Exception as exc:  # cleanup must never fail an inversion
            warnings.warn(
                f"Could not delete the background checkpoint of "
                f"{getattr(entry.job, 'name', entry.fingerprint)!r}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    def background_bytes(self) -> Dict[str, int]:
        """Return retained background checkpoint bytes per cache key (LRU first)."""

        sizes = {}
        for key, entry in self._entries.items():
            if getattr(entry.job, "background", None) is None:
                continue
            size = entry.extra.get("background_bytes")
            if size is None:
                size = background_checkpoint_files(entry.job)[1]
                entry.extra["background_bytes"] = size
            sizes[key] = int(size)
        return sizes

    def _enforce_background_budget(self, newest: LinearizationEntry) -> None:
        if self.background_budget is None or self.release is None:
            return
        sizes = self.background_bytes()
        total = sum(sizes.values())
        for key, size in sizes.items():
            if total <= self.background_budget:
                break
            if key == newest.fingerprint:
                continue
            self._release(self._entries[key])
            total -= size

    def _remove(self, entry: LinearizationEntry) -> None:
        self.evicted.append(entry.fingerprint)
        self._release(entry)
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
