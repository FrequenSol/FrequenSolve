# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Orchestration of Sauce's inverse-curvature and reference-image operations.

Sauce evaluates inverse BFGS actions, exact compact covariance factors (a QR of
the secant basis and a small signed eigensolve), posterior variances and grid
projections, mesh prior transfers, analytic envelopes, smoothing and image
weighting. Each result is published only after the solver-reported dataset
digests, array counts and identities match what this process staged.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import operator
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import h5py
import numpy as np

from ._block_digest import (
    SCHEME,
    block_digest,
    dataset_block_digest,
    stacked_block_digest,
)

__all__ = ["BFGSHistory", "CurvatureResult", "NativeCurvature"]

_HISTORY_ARRAYS = ("base_inverse_diagonal", "steps", "gradient_differences")
# Element type Sauce reads each staged dataset into (float64 otherwise); digests use it.
_ELEMENT_TYPES = dict(
    offsets=np.int64,
    indices=np.int32,
    source_roots=np.int32,
    target_roots=np.int32,
    grid_shape=np.int32,
    smoothing_radii=np.int32,
)
# Methods whose outputs are reusable covariance factors, and the datasets every one writes.
_FACTOR_METHODS = frozenset({"bfgs_rsvd", "curvature_refresh", "curvature_warm_start"})
_FACTOR_OUTPUTS = (
    "base_inverse_diagonal",
    "prior_std",
    "variance",
    "standard_deviation",
)


# Staged history files of this process: path -> [users, write lock]. Users are
# operations in flight and ``NativeCurvature.retain`` blocks; the last one deletes it.
_STAGED_HISTORIES: dict = {}
_STAGED_LOCK = threading.Lock()


@contextlib.contextmanager
def _history_lease(path: Path) -> Iterator[threading.Lock]:
    """Count one user of a staged history file; the last user to leave deletes it.

    The yielded lock serializes writing the file, so concurrent operations on
    one history stage it once and none deletes it while another reads it.
    """
    with _STAGED_LOCK:
        lease = _STAGED_HISTORIES.setdefault(path, [0, threading.Lock()])
        lease[0] += 1
    try:
        yield lease[1]
    finally:
        with _STAGED_LOCK:
            lease[0] -= 1
            if not lease[0]:
                del _STAGED_HISTORIES[path]
                path.unlink(missing_ok=True)


def _links_into(path: Path, target: Path) -> bool:
    """Return whether an HDF5 external link of ``path`` resolves to ``target``.

    Relative link names resolve against the linking file's directory, as in
    HDF5. Sauce links a warm start's unchanged ``/modes`` into its input file.
    """
    from frequensolve.orchestrator.sites.curvature import external_link_targets

    for name in external_link_targets(path):
        linked = Path(os.path.normpath(path.parent / name))
        if linked == target:
            return True
        with contextlib.suppress(OSError):
            if os.path.samefile(linked, target):
                return True
    return False


def _staged(name: str, value: Any) -> np.ndarray:
    """Return ``value`` in Sauce's element type for ``name``; integer narrowing must be exact."""
    element = np.dtype(_ELEMENT_TYPES.get(name, np.float64))
    array = np.asarray(value)
    staged = np.ascontiguousarray(array, dtype=element)
    if (
        element.kind == "i"
        and array.dtype != element
        and not np.array_equal(staged, array)
    ):
        raise ValueError(f"{name} does not fit Sauce's {element} elements")
    return staged


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(f"{SCHEME}:")
        and len(value) == len(SCHEME) + 65
    )


def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _real(value: Any, name: str) -> np.ndarray:
    """Validate an input array without altering its physical coordinates."""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=np.float64)
    if not _finite(array):
        raise ValueError(f"{name} must be finite")
    return np.array(array, copy=True)


def _adopted(value: Any, name: str) -> np.ndarray:
    """Keep a read-only float64 array its producer handed over; copy anything else.

    A read-only C-contiguous float64 array (seed modes released by
    ``_CurvatureSeed.take``, arrays just read from a file) is validated in place
    and shared; a writable input is copied once, so its owner may keep
    modifying it.
    """
    if (
        isinstance(value, np.ndarray)
        and value.dtype == np.float64
        and value.flags.c_contiguous
        and not value.flags.writeable
    ):
        if not _finite(value):
            raise ValueError(f"{name} must be finite")
        return value
    return _real(value, name)


def _finite(array: np.ndarray) -> bool:
    """Exact finiteness of a float array from two reductions, without a mask copy."""
    return not array.size or bool(np.isfinite(array.min()) and np.isfinite(array.max()))


def _real_input(value: Any, name: str) -> np.ndarray:
    """Validate a staged input in place; only a dtype or layout conversion copies.

    h5py writes C-contiguous float64 datasets straight from the caller's
    buffer and digests are hashed from that memory, so a controls-by-rank
    block is staged without a second copy. Do not mutate it during the call.
    """
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=np.float64, order="C")
    if not _finite(array):
        raise ValueError(f"{name} must be finite")
    return array


def _relative_tolerance(value: Any, name: str) -> float:
    """Validate a relative tolerance Sauce accepts: ``sqrt(float64 eps) <= value < 1``."""
    minimum = np.sqrt(np.finfo(np.float64).eps)
    if not np.isfinite(value) or not minimum <= value < 1:
        raise ValueError(f"{name} must be at least sqrt(float64 eps) and below one")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    """Validate JSON/native integer options without silently truncating floats."""
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, (bool, np.bool_)) or not minimum <= result <= 2147483647:
        raise ValueError(f"{name} is outside the native integer range")
    return result


def _property_dimension(*meshes: dict) -> int:
    """Spatial dimension shared by property-space meshes, for solver routing."""
    found = set()
    for mesh in meshes:
        value = mesh.get("dimension")
        if value is None:
            with h5py.File(mesh["path"], "r") as h5:
                value = np.asarray(h5["property_space/header"][()]).reshape(-1)[0]
        found.add(_integer(value, "property-space dimension"))
    if len(found) != 1 or not found <= {2, 3}:
        raise ValueError("Property-space meshes must share dimension 2 or 3")
    return found.pop()


class BFGSHistory:
    """Archive a fixed inverse diagonal and optimizer-produced curvature pairs.

    Use as an L-BFGS callback (directly or with ``FWI(callback=history)``).
    Set optimizer memory large enough for the entire stage and freeze its
    preconditioner. Pairs use the optimizer's coordinates; with FWI, use
    ``scaling=None`` or explicitly transform the base and prior into the
    scaled coordinates. A new objective/stage requires a new archive.
    Losing pairs through truncation, reset, or a changed stage raises instead
    of silently reporting a full-history covariance. The archive shares the
    optimizer's read-only pair vectors by reference, so following an
    optimization copies nothing; each new pair is validated once.
    ``steps``/``gradient_differences`` stack the pairs on first access, while
    staging, saving and ``dataset_digests`` read the pair vectors in place.
    ``dataset_digests`` identify the datasets once per content and ``digest``
    names the one staged copy that backend calls reuse.
    """

    def __init__(
        self,
        base_inverse_diagonal: Any,
        *,
        state: str,
        coordinates: str,
        seed_modes: Any = None,
        seed_eigenvalues: Any = None,
        provenance: Optional[dict] = None,
    ) -> None:
        base = _adopted(base_inverse_diagonal, "base inverse diagonal")
        if base.ndim != 1 or not base.size or base.min() <= 0:
            raise ValueError("base inverse diagonal must be a positive vector")
        if not state or not coordinates:
            raise ValueError("state and coordinate identities are required")
        if (seed_modes is None) != (seed_eigenvalues is None):
            raise ValueError("Supply both seed modes and seed eigenvalues")
        # Read-only seed blocks are adopted, not copied (a controls-by-rank
        # block is the largest array a stage holds).
        modes = (
            np.empty((0, base.size))
            if seed_modes is None
            else _adopted(seed_modes, "seed modes")
        )
        eigenvalues = (
            np.empty(0)
            if seed_eigenvalues is None
            else _adopted(seed_eigenvalues, "seed eigenvalues")
        )
        if (
            modes.ndim != 2
            or modes.shape[1] != base.size
            or eigenvalues.shape != (len(modes),)
        ):
            raise ValueError("Seed modes and eigenvalues have inconsistent coordinates")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("Seed provenance must be a JSON mapping")
        self._provenance_json = json.dumps(
            provenance or {}, sort_keys=True, allow_nan=False
        )
        self._seed_modes = _frozen(modes)
        self._seed_eigenvalues = _frozen(eigenvalues)
        self._base = _frozen(base)
        self.state = str(state)
        self.coordinates = str(coordinates)
        self._digests: dict = {}
        self._stacked: Optional[tuple] = None
        self._replace_pairs((), (), ())
        self._stage: Optional[int] = None
        self._accepted_iterations = 0

    @property
    def base_inverse_diagonal(self) -> np.ndarray:
        return self._base

    @property
    def seed_modes(self) -> np.ndarray:
        return self._seed_modes

    @property
    def seed_eigenvalues(self) -> np.ndarray:
        return self._seed_eigenvalues

    @property
    def seed_rank(self) -> int:
        return len(self._seed_eigenvalues)

    @property
    def provenance(self) -> dict:
        """Return a copy of the immutable seed's numerical provenance."""
        return json.loads(self._provenance_json)

    @property
    def pair_count(self) -> int:
        """Return the number of archived curvature pairs."""
        return len(self._steps)

    @property
    def accepted_iterations(self) -> int:
        """Return the optimizer's accepted iterations when the pairs were last taken."""
        return self._accepted_iterations

    @property
    def pairs(self) -> tuple:
        """Return ``(steps, gradient_differences)``: shared read-only vectors, oldest first."""
        return self._steps, self._differences

    @property
    def pair_ids(self) -> tuple:
        """Return the optimizer's identifiers of the archived pairs (empty when unknown)."""
        return self._pair_ids

    @property
    def steps(self) -> np.ndarray:
        """Return the steps stacked as ``(pairs, controls)`` (one cached copy)."""
        return self._stack()[0]

    @property
    def gradient_differences(self) -> np.ndarray:
        """Return the gradient differences stacked as ``(pairs, controls)``."""
        return self._stack()[1]

    def _stack(self) -> tuple:
        if self._stacked is None:
            empty = np.empty((0, self._base.size))
            self._stacked = tuple(
                _frozen(np.stack(vectors) if vectors else empty.copy())
                for vectors in (self._steps, self._differences)
            )
        return self._stacked

    def _vectors(self, values: Any, current: tuple, name: str) -> tuple:
        """Adopt pair vectors, validating only those not already archived.

        Read-only float64 vectors are shared by reference; anything else is
        copied once into a read-only vector.
        """
        if isinstance(values, np.ndarray) and values.ndim == 2:
            values = list(values)
        known = {id(vector) for vector in current}
        size = self._base.size
        out = []
        for value in values:
            if id(value) in known:
                out.append(value)
                continue
            if (
                isinstance(value, np.ndarray)
                and value.dtype == np.float64
                and value.shape == (size,)
                and not value.flags.writeable
            ):
                if not (np.isfinite(value.min()) and np.isfinite(value.max())):
                    raise ValueError(f"{name} must be finite")
                out.append(value)
                continue
            vector = _real(value, name).reshape(-1)
            if vector.shape != (size,):
                raise ValueError("BFGS history coordinates disagree")
            out.append(_frozen(vector))
        return tuple(out)

    def _replace_pairs(self, steps: tuple, differences: tuple, pair_ids: tuple) -> None:
        if len(steps) != len(differences):
            raise ValueError("BFGS history shapes disagree")
        self._steps, self._differences = steps, differences
        self._pair_ids = tuple(pair_ids) if len(pair_ids) == len(steps) else ()
        self._stacked = None
        self._digest: Optional[str] = None
        # The base and seed never change; only the secant digests are renewed.
        self._digests = {
            key: value
            for key, value in self._digests.items()
            if key not in ("/steps", "/gradient_differences")
        }

    def _set_pairs(self, steps: Any, differences: Any, pair_ids: Any = ()) -> None:
        """Replace the archived pairs (rows of arrays or sequences of vectors)."""
        self._replace_pairs(
            self._vectors(steps, self._steps, "BFGS steps"),
            self._vectors(differences, self._differences, "BFGS differences"),
            tuple(pair_ids),
        )

    def _follow(self, state: Any) -> None:
        """Mirror a limited-memory optimizer window; older pairs may slide out."""
        self._set_pairs(state.steps, state.gradient_differences, state.pair_ids)
        self._accepted_iterations = int(state.accepted_iterations)

    @property
    def dataset_digests(self) -> dict:
        """``fs-block-sha256-1`` digests of the datasets Sauce reads, once per content."""
        names = _HISTORY_ARRAYS + (
            ("seed_modes", "seed_eigenvalues") if self.seed_rank else ()
        )
        pairs = dict(steps=self._steps, gradient_differences=self._differences)
        for name in names:
            if f"/{name}" not in self._digests:
                self._digests[f"/{name}"] = (
                    stacked_block_digest(pairs[name], self._base.size)
                    if name in pairs
                    else block_digest(getattr(self, name))
                )
        return {f"/{name}": self._digests[f"/{name}"] for name in names}

    @property
    def digest(self) -> str:
        """Identity of the archived datasets and seed provenance, computed once per history."""
        if self._digest is None:
            text = json.dumps(
                dict(datasets=self.dataset_digests, provenance=self._provenance_json),
                sort_keys=True,
            )
            self._digest = hashlib.sha256(text.encode()).hexdigest()
        return self._digest

    def __call__(self, event: Any) -> None:
        """Retain the optimizer's curvature pairs by reference, rejecting history loss."""
        from frequensolve.inversion.optimization import LBFGSRestart

        stage = getattr(event, "stage_index", None)
        if stage is not None:
            if self._stage is not None and stage != self._stage:
                raise ValueError(
                    "BFGS uncertainty requires a separate history per stage"
                )
            self._stage = stage
        diagnostics = getattr(event, "diagnostics", event)
        state = diagnostics.optimizer_state
        if not isinstance(state, LBFGSRestart):
            raise ValueError("BFGS history requires an L-BFGS optimizer checkpoint")
        accepted = state.accepted_iterations
        if accepted < self._accepted_iterations:
            raise ValueError("BFGS history iteration count moved backwards")
        previous = self.pair_count
        steps, differences = state.steps, state.gradient_differences
        if len(steps) < previous or any(
            new is not old and not np.array_equal(new, old)
            for new, old in zip(steps, self._steps)
        ):
            raise ValueError(
                "BFGS history was truncated or reset; full-history UQ is unavailable"
            )
        if any(
            new is not old and not np.array_equal(new, old)
            for new, old in zip(differences, self._differences)
        ):
            raise ValueError("BFGS gradient-difference history changed")
        if len(steps) != previous:
            self._set_pairs(
                self._steps + tuple(steps[previous:]),
                self._differences + tuple(differences[previous:]),
                state.pair_ids,
            )
        self._accepted_iterations = accepted

    def _metadata(self) -> dict:
        return dict(
            schema="fs-bfgs-history-1",
            state=self.state,
            coordinates=self.coordinates,
            accepted_iterations=self._accepted_iterations,
            stage=self._stage,
            seed_rank=self.seed_rank,
            provenance=self.provenance,
        )

    def _write_datasets(self, h5: Any, names: Optional[tuple] = None) -> None:
        """Write archive datasets, streaming pair vectors row by row (no stack)."""
        if names is None:
            names = _HISTORY_ARRAYS + (
                ("seed_modes", "seed_eigenvalues") if self.seed_rank else ()
            )
        pairs = dict(steps=self._steps, gradient_differences=self._differences)
        for name in names:
            if name not in pairs:
                h5.create_dataset(name, data=getattr(self, name))
                continue
            dataset = h5.create_dataset(
                name, shape=(len(pairs[name]), self._base.size), dtype=np.float64
            )
            for row, vector in enumerate(pairs[name]):
                dataset[row] = vector

    def save(self, path: Any) -> Path:
        """Atomically persist the exact coordinates and secants for backend replay."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with h5py.File(pending, "w") as h5:
                self._write_datasets(h5)
                h5.create_dataset("metadata", data=json.dumps(self._metadata()))
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: Any) -> "BFGSHistory":
        """Restore a full archive without reconstructing numerical curvature."""
        with h5py.File(path, "r") as h5:
            metadata = json.loads(h5["metadata"][()])
            if metadata.get("schema") != "fs-bfgs-history-1":
                raise ValueError("Unsupported BFGS history schema")

            def owned(name: str) -> Optional[np.ndarray]:
                # Freshly read arrays are handed over to the history uncopied.
                if name not in h5:
                    return None
                return _frozen(np.asarray(h5[name][()], dtype=np.float64))

            history = cls(
                owned("base_inverse_diagonal"),
                state=metadata["state"],
                coordinates=metadata["coordinates"],
                seed_modes=owned("seed_modes"),
                seed_eigenvalues=owned("seed_eigenvalues"),
                provenance=metadata.get("provenance"),
            )
            if history.seed_rank != metadata.get("seed_rank", history.seed_rank):
                raise ValueError("Saved BFGS seed rank disagrees with its arrays")
            steps = np.asarray(h5["steps"][()], dtype=np.float64)
            differences = np.asarray(h5["gradient_differences"][()], dtype=np.float64)
        if steps.shape != differences.shape or steps.ndim != 2:
            raise ValueError("BFGS history shapes disagree")
        if steps.shape[1] != history.base_inverse_diagonal.size:
            raise ValueError("BFGS history coordinates disagree")
        if not (np.isfinite(steps).all() and np.isfinite(differences).all()):
            raise ValueError("BFGS history pairs must be finite")
        # The freshly read stacks are owned here: freeze them and share their rows.
        _frozen(steps)
        _frozen(differences)
        history._replace_pairs(tuple(steps), tuple(differences), ())
        history._stacked = (steps, differences)
        history._accepted_iterations = int(metadata["accepted_iterations"])
        if history._accepted_iterations < history.pair_count:
            raise ValueError("BFGS history iteration count is inconsistent")
        history._stage = metadata.get("stage")
        return history


@dataclass(frozen=True)
class CurvatureResult:
    """Backend-owned HDF5 result with explicit state and coordinate identities.

    Reusable covariance factors carry Sauce's ``output_digests``: the
    ``fs-block-sha256-1`` digest of each dataset it wrote. ``read_verified``
    checks arrays against them in memory as they are loaded; ``verify`` is an
    explicit check of every recorded dataset, never needed on the hot path.
    """

    path: Path
    metadata: dict

    def read(self, name: str) -> np.ndarray:
        """Read a backend output dataset."""
        with h5py.File(self.path, "r") as h5:
            return h5[name][()]

    @property
    def output_digests(self) -> dict:
        """Digests Sauce recorded for the datasets of reusable factors."""
        digests = self.metadata.get("output_digests")
        if not isinstance(digests, dict) or not digests:
            raise ValueError(
                "Curvature factors record no output_digests; regenerate them with the current solver build"
            )
        return digests

    @property
    def identity(self) -> str:
        """Content identity of reusable factors: a fingerprint of their dataset digests."""
        text = json.dumps(self.output_digests, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest()

    def read_verified(self, *names: str) -> dict:
        """Read factor datasets, checking each against its recorded digest before use."""
        recorded = self.output_digests
        with h5py.File(self.path, "r") as h5:
            if any(name not in h5 for name in names):
                raise ValueError("Covariance factors changed after they were recorded")
            arrays = {name: h5[name][()] for name in names}
        for name, value in arrays.items():
            if block_digest(value) != recorded.get(f"/{name}"):
                raise ValueError("Covariance factors changed after they were recorded")
        return arrays

    def verify(self) -> dict:
        """Recheck every recorded dataset of the file and return the digests.

        Datasets are hashed one leaf tile at a time, so verifying a
        controls-by-rank block never holds it in memory.
        """
        recorded = self.output_digests
        with h5py.File(self.path, "r") as h5:
            for key, digest in recorded.items():
                name = key.lstrip("/")
                if name not in h5 or dataset_block_digest(h5[name]) != digest:
                    raise ValueError(
                        "Covariance factors changed after they were recorded"
                    )
        return dict(recorded)


class NativeCurvature:
    """Invoke Sauce postprocessing with a local executable or a site runner.

    ``runner(request_path)`` (for example ``Backend.curvature()``, bound to a
    site's ``run_curvature``) chooses ranks and threads, may stage files and
    run on remote compute nodes, and returns only after the request's output
    is available locally; an ``executable`` runs one local rank.
    Each operation has an independent directory and publishes only verified
    completed output. Existing results are never reused implicitly. Once a
    result is verified its staged ``input.h5`` is deleted, unless the result
    has an HDF5 external link into it (warm-start factors link their
    unchanged ``/modes``); a failed operation keeps its directory. A BFGS
    history operation stages the history as ``workdir/histories/<digest>.h5``;
    request inputs reference it through relative HDF5 external links, so a
    staging runner must carry the ``histories`` directory beside the request.
    No result links it, so the file (one controls-by-pairs block) is deleted
    as soon as the operation finishes, successfully or not; :meth:`retain`
    keeps it staged for a sequence of operations on one history. Remote
    mirrors persist until ``SlurmSite.remove_curvature_files`` deletes them,
    so restaging a history there uploads nothing. ``symmetry_tolerance`` is
    the default of :meth:`refresh_curvature` (``None``: Sauce's ``1e-2``).
    """

    def __init__(
        self,
        executable: Any = None,
        *,
        workdir: Any,
        runner: Optional[Callable[[Path], None]] = None,
        symmetry_tolerance: Optional[float] = None,
    ) -> None:
        if executable is None and runner is None:
            raise ValueError("A Sauce executable or runner is required")
        self.executable = (
            None if executable is None else str(Path(executable).resolve())
        )
        self.workdir = Path(workdir).resolve()
        self.runner = runner
        self.symmetry_tolerance = (
            None
            if symmetry_tolerance is None
            else _relative_tolerance(symmetry_tolerance, "symmetry_tolerance")
        )

    def _execute(
        self,
        method: str,
        arrays: dict,
        *,
        state: str,
        coordinates: str,
        expect: Optional[dict] = None,
        links: Optional[dict] = None,
        factors: Optional[CurvatureResult] = None,
        **options: Any,
    ) -> CurvatureResult:
        """Stage, run and verify one request.

        ``expect`` holds counts and options derived from what this process
        staged. Sauce must report them, its dtype and the ``fs-block-sha256-1``
        digest of every dataset it read: ``input_digests`` must equal the
        digests of the staged arrays, computed from memory (``links`` map a
        linked dataset to its file and known digest), and every
        ``factors_digests`` entry must equal the digest the factors' producer
        recorded. No staged or produced file is rehashed.
        """
        if not state or not coordinates:
            raise ValueError("state and coordinate identities are required")
        directory = self.workdir / f"{method}-{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        input_path, pending = directory / "input.h5", directory / "pending.h5"
        metadata = dict(
            schema="fs-curvature-input-1",
            method=method,
            state=state,
            coordinates=coordinates,
        )
        staged = {}
        with h5py.File(input_path, "w") as h5:
            for name, value in arrays.items():
                array = _staged(name, value)
                h5.create_dataset(name, data=array)
                staged[f"/{name}"] = block_digest(array, array.dtype)
            for name, (target, digest) in (links or {}).items():
                # HDF5 resolves relative targets against the linking file.
                relative = os.path.relpath(target, directory)
                h5[name] = h5py.ExternalLink(relative, f"/{name}")
                staged[f"/{name}"] = digest
            h5.create_dataset("metadata", data=np.bytes_(json.dumps(metadata)))
        expected = dict(expect or {}, dtype="float64")
        if factors is not None:
            recorded = factors.output_digests
            options["factors"] = str(factors.path.resolve())
        request = directory / "request.json"
        request.write_text(
            json.dumps(
                dict(
                    metadata,
                    schema="fs-curvature-request-1",
                    input=str(input_path),
                    output=str(pending),
                    **options,
                ),
                indent=2,
            )
            + "\n"
        )
        if self.runner is not None:
            self.runner(request)
        else:
            if self.executable is None:
                raise ValueError("A Sauce executable or runner is required")
            with (directory / "solver.log").open("w") as log:
                subprocess.run(
                    [self.executable, "--curvature", str(request)],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        with h5py.File(pending, "r") as h5:
            result = json.loads(h5["metadata"][()])
        if result.get("schema") != "fs-curvature-output-1":
            raise ValueError("Unsupported Sauce curvature result")
        for key in ("method", "state", "coordinates"):
            if result.get(key) != metadata[key]:
                raise ValueError(f"Sauce curvature result has mismatched {key}")
        reported = dict(input_digests=staged)
        if factors is not None:
            reported["factors_digests"] = None
        if method in _FACTOR_METHODS:
            reported["output_digests"] = None
        for key, value in dict(expected, **reported).items():
            if key not in result:
                raise ValueError(
                    f"Sauce curvature result lacks {key}; update the solver build"
                )
            if value is not None and result[key] != value:
                raise ValueError(f"Sauce curvature result has mismatched {key}")
        if factors is not None:
            # Sauce reads a subset of the stored factors; each must be what was produced.
            read = result["factors_digests"]
            if (
                not isinstance(read, dict)
                or not read
                or any(recorded.get(key) != value for key, value in read.items())
            ):
                raise ValueError(
                    "Sauce curvature result has mismatched factors_digests"
                )
        if method in _FACTOR_METHODS:
            written = result["output_digests"]
            required = _FACTOR_OUTPUTS + (
                ("modes", "eigenvalues") if result.get("rank") else ()
            )
            if (
                not isinstance(written, dict)
                or any(f"/{name}" not in written for name in required)
                or not all(_is_digest(value) for value in written.values())
            ):
                raise ValueError("Sauce curvature result has invalid output_digests")
        path = directory / "result.h5"
        pending.replace(path)
        # The staged input (directions or images: up to rank x controls) is
        # dead once the result is verified, unless the result links into it.
        if not _links_into(path, input_path):
            input_path.unlink(missing_ok=True)
        return CurvatureResult(path, result)

    def _execute_history(
        self, method: str, arrays: dict, history: BFGSHistory, **options: Any
    ) -> CurvatureResult:
        """Write each distinct history once; requests link that immutable file."""
        names = _HISTORY_ARRAYS + (
            ("seed_modes", "seed_eigenvalues") if history.seed_rank else ()
        )
        expected = dict(
            controls=history.base_inverse_diagonal.size,
            history_pairs=history.pair_count,
        )
        if history.seed_rank:
            expected["seed_rank"] = history.seed_rank
            options["seed_rank"] = history.seed_rank
        path = self._history_path(history)
        with _history_lease(path) as writing:
            with writing:
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pending = path.with_name(f".{uuid.uuid4().hex}.h5")
                    try:
                        with h5py.File(pending, "w") as h5:
                            history._write_datasets(h5, names)
                        pending.replace(path)
                    finally:
                        pending.unlink(missing_ok=True)
            # Sauce must report the cached in-memory digests for the linked file's contents.
            digests = history.dataset_digests
            return self._execute(
                method,
                arrays,
                state=history.state,
                coordinates=history.coordinates,
                links={name: (path, digests[f"/{name}"]) for name in names},
                expect=expected,
                **options,
            )

    def _history_path(self, history: BFGSHistory) -> Path:
        return self.workdir / "histories" / f"{history.digest}.h5"

    @contextlib.contextmanager
    def retain(self, history: BFGSHistory) -> Iterator[None]:
        """Keep ``history`` staged for every operation inside the block.

        Without it each operation writes the history and deletes it when it
        finishes. Inside the block the file is written once, shared by every
        operation on that content (from any ``NativeCurvature`` with this
        workdir) and deleted at exit.
        """
        with _history_lease(self._history_path(history)):
            yield

    def bfgs_uncertainty(
        self,
        history: BFGSHistory,
        *,
        prior_std: Any = 1.0,
        rank: Optional[int] = None,
        oversampling: int = 8,
        seed: int = 0,
        curvature_tolerance: float = 1e-6,
    ) -> CurvatureResult:
        """Ask Sauce for exact compact signed factors and prior-scaled posterior marginals.

        The supplied history must optimize the prior-whitened, noise-weighted
        objective of Liu et al. Eq. 13 for a Bayesian interpretation. An
        arbitrary optimizer metric is not automatically calibrated uncertainty.
        Default rank covers the correction's possible ``seed_rank + 2 * pairs`` range.
        The result records Sauce's digest of each factor dataset for later reuse.
        """
        prior = _real_input(prior_std, "prior standard deviation")
        shape = history.base_inverse_diagonal.shape
        if prior.shape != shape:
            prior = np.broadcast_to(prior, shape).copy()
        if (
            not np.isfinite(curvature_tolerance)
            or not np.finfo(float).eps <= curvature_tolerance < 1
        ):
            raise ValueError("Invalid relative curvature_tolerance")
        options: dict[str, Any] = dict(
            curvature_tolerance=float(curvature_tolerance),
            oversampling=_integer(oversampling, "oversampling"),
            seed=_integer(seed, "seed", minimum=-2147483648),
        )
        if rank is not None:
            options["rank"] = _integer(rank, "rank")
        return self._execute_history(
            "bfgs_rsvd", dict(prior_std=prior), history, **options
        )

    def inverse_action(self, history: BFGSHistory, vectors: Any) -> CurvatureResult:
        """Schedule a backend compact inverse action on vectors or probe batches."""
        value = _real_input(vectors, "inverse directions")
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != history.base_inverse_diagonal.size:
            raise ValueError("inverse directions must have shape (probes, controls)")
        return self._execute_history("bfgs_action", dict(vectors=value), history)

    def gaussian_prior(
        self, point: Any, reference: Any, std: Any, *, state: str, coordinates: str
    ) -> CurvatureResult:
        """Evaluate a Gaussian prior and its derivatives in Sauce.

        ``GaussianPrior`` mirrors this formula in NumPy; this call is the
        backend reference for that equivalence.
        """
        point = _real_input(point, "point")
        return self._execute(
            "gaussian_prior",
            dict(
                point=point,
                reference=_real_input(reference, "reference"),
                prior_std=_real_input(std, "prior std"),
            ),
            state=state,
            coordinates=coordinates,
            expect=dict(controls=point.size),
        )

    def mesh_prior(
        self, source: dict, target: dict, reference: Any, std: Any
    ) -> CurvatureResult:
        """Lift fields through Sauce and rebuild the normalized volume prior."""
        if source["material"] != target["material"]:
            raise ValueError("Mesh prior cannot change materials")
        arrays = dict(
            source_roots=source["roots"],
            target_roots=target["roots"],
            reference=_real_input(reference, "prior reference"),
            prior_std=_real_input(std, "prior scale"),
        )
        return self._execute(
            "mesh_prior",
            arrays,
            state=source["identity"],
            coordinates=target["identity"],
            expect=dict(controls=target["size"]),
            source_mesh=source["path"],
            target_mesh=target["path"],
            source_identity=source["identity"],
            target_identity=target["identity"],
            material=target["material"],
            dimension=_property_dimension(source, target),
        )

    @staticmethod
    def _transfer_inputs(base_inverse_diagonal: Any, directions: Any) -> tuple:
        base = _real_input(base_inverse_diagonal, "transfer inverse diagonal")
        if base.ndim != 1 or not base.size or np.any(base <= 0):
            raise ValueError("Transfer inverse diagonal must be a positive vector")
        directions = _real_input(directions, "transfer directions")
        if directions.ndim == 1:
            directions = directions[None, :]
        if directions.ndim != 2 or directions.shape[1] != base.size:
            raise ValueError(
                "Transfer directions must have shape (directions, controls)"
            )
        return base, directions

    @staticmethod
    def _basis_tolerance(value: Any) -> float:
        return _relative_tolerance(value, "basis_tolerance")

    def transfer_basis(
        self,
        base_inverse_diagonal: Any,
        directions: Any,
        *,
        state: str,
        coordinates: str,
        rank: Optional[int] = None,
        basis_tolerance: float = 1e-4,
    ) -> CurvatureResult:
        """Reduce physical directions in Sauce's inverse-diagonal metric.

        Output ``directions`` has shape (rank, controls) and is orthonormal in
        the metric ``diag(base_inverse_diagonal)**-1``. Rank reduction and all
        QR/eigen work belong to Sauce.
        """
        base, directions = self._transfer_inputs(base_inverse_diagonal, directions)
        tolerance = self._basis_tolerance(basis_tolerance)
        options: dict[str, Any] = dict(basis_tolerance=tolerance)
        if rank is not None:
            options["rank"] = _integer(rank, "transfer rank")
        return self._execute(
            "curvature_basis",
            dict(base_inverse_diagonal=base, directions=directions),
            state=state,
            coordinates=coordinates,
            expect=dict(controls=base.size, basis_tolerance=tolerance),
            **options,
        )

    def refresh_curvature(
        self,
        base_inverse_diagonal: Any,
        directions: Any,
        images: Any,
        *,
        state: str,
        coordinates: str,
        basis_tolerance: float = 1e-4,
        symmetry_tolerance: Optional[float] = None,
    ) -> CurvatureResult:
        """Rebuild reduced inverse factors from the current stage's Hessian actions.

        Images of an iteratively solved normal are symmetric only to the
        solver tolerance, so Sauce symmetrizes the reduced Hessian
        ``K = R^T images`` and rejects a relative asymmetry
        ``||K - K^T||_F / ||K||_F`` above ``symmetry_tolerance`` (default: this
        instance's, else Sauce's ``1e-2``), as from an inconsistent JVP/VJP
        pair. The result metadata records the applied ``symmetry_tolerance``
        and the measured ``hessian_asymmetry``.
        """
        base, directions = self._transfer_inputs(base_inverse_diagonal, directions)
        images = _real_input(images, "refreshed Hessian images")
        if images.ndim == 1:
            images = images[None, :]
        if images.shape != directions.shape:
            raise ValueError("Refreshed Hessian images must match the direction batch")
        options: dict[str, Any] = dict(
            basis_tolerance=self._basis_tolerance(basis_tolerance)
        )
        # Sauce must report both; a requested tolerance must be the one applied.
        expected: dict[str, Any] = dict(
            controls=base.size, symmetry_tolerance=None, hessian_asymmetry=None
        )
        if symmetry_tolerance is None:
            symmetry_tolerance = self.symmetry_tolerance
        if symmetry_tolerance is not None:
            tolerance = _relative_tolerance(symmetry_tolerance, "symmetry_tolerance")
            options["symmetry_tolerance"] = expected["symmetry_tolerance"] = tolerance
        result = self._execute(
            "curvature_refresh",
            dict(base_inverse_diagonal=base, directions=directions, images=images),
            state=state,
            coordinates=coordinates,
            expect=expected,
            **options,
        )
        asymmetry = result.metadata["hessian_asymmetry"]
        if not np.isfinite(asymmetry) or not (
            0 <= asymmetry <= result.metadata["symmetry_tolerance"]
        ):
            raise ValueError("Sauce returned an invalid hessian_asymmetry")
        return result

    def warm_start_curvature(
        self,
        base_inverse_diagonal: Any,
        modes: Any,
        eigenvalues: Any,
        *,
        state: str,
        coordinates: str,
    ) -> CurvatureResult:
        """Damp a transferred physical inverse correction to positivity in Sauce."""
        base, modes = self._transfer_inputs(base_inverse_diagonal, modes)
        eigenvalues = _real_input(eigenvalues, "warm-start eigenvalues")
        if eigenvalues.shape != (len(modes),):
            raise ValueError("Warm-start modes and eigenvalues have inconsistent rank")
        result = self._execute(
            "curvature_warm_start",
            dict(base_inverse_diagonal=base, modes=modes, eigenvalues=eigenvalues),
            state=state,
            coordinates=coordinates,
            expect=dict(controls=base.size),
        )
        damping = result.metadata.get("transfer_damping")
        if damping is None or not np.isfinite(damping) or not 0 <= damping <= 1:
            raise ValueError("Sauce returned invalid or missing transfer_damping")
        return result

    def mesh_directions(
        self, source: dict, target: dict, directions: Any
    ) -> CurvatureResult:
        """Lift physical curvature directions through constrained native mesh bases."""
        if source["material"] != target["material"]:
            raise ValueError("Mesh curvature transfer cannot change materials")
        directions = _real_input(directions, "mesh transfer directions")
        if directions.ndim == 1:
            directions = directions[None, :]
        if directions.ndim != 2 or directions.shape[1] != source["size"]:
            raise ValueError(
                "Mesh directions must match the source control coordinates"
            )
        return self._execute(
            "mesh_directions",
            dict(
                source_roots=source["roots"],
                target_roots=target["roots"],
                directions=directions,
            ),
            state=source["identity"],
            coordinates=target["identity"],
            expect=dict(controls=target["size"]),
            source_mesh=source["path"],
            target_mesh=target["path"],
            source_identity=source["identity"],
            target_identity=target["identity"],
            material=target["material"],
            dimension=_property_dimension(source, target),
        )

    def mesh_sampling(self, mesh: dict, points: Any) -> tuple:
        """Get the exact constrained native basis on Cartesian points in metres."""
        from scipy.sparse import csr_matrix

        metres = mesh.get("metres_per_native_unit")
        if metres is None:
            raise ValueError(
                "Mesh artifact omits geometry units; regenerate it with the current Sauce build"
            )
        output = self._execute(
            "mesh_sample",
            dict(
                target_roots=mesh["roots"],
                points=_real_input(points, "sample points") / metres,
            ),
            state=mesh["identity"],
            coordinates=mesh["identity"],
            expect=dict(controls=mesh["size"]),
            target_mesh=mesh["path"],
            target_identity=mesh["identity"],
            material=mesh["material"],
            dimension=_property_dimension(mesh),
        )
        operator = csr_matrix(
            (output.read("weights"), output.read("indices"), output.read("offsets")),
            shape=(len(points), mesh["size"]),
        )
        operator.sum_duplicates()
        operator.sort_indices()
        return operator, output.read("valid").astype(bool)

    def covariance(
        self, factors: CurvatureResult, *, vectors: Any = None, projection: Any = None
    ) -> CurvatureResult:
        """Apply covariance factors or project their diagonal through a CSR map.

        Sauce reads the factor file directly and must report, for each factor
        dataset it read, the digest recorded when the factors were produced.
        ``UncertaintyResult`` applies the covariance in memory; this call is
        its backend reference.
        """
        if (vectors is None) == (projection is None):
            raise ValueError("Supply either vectors or a CSR projection")
        rank = int(factors.metadata["rank"])
        # Sauce reads the immutable factors directly; only directions/CSR are staged.
        with h5py.File(factors.path, "r") as h5:
            size = h5["prior_std"].size
        arrays: dict[str, Any] = {}
        if projection is not None:
            from scipy.sparse import csr_matrix

            matrix = csr_matrix(projection, dtype=float, copy=True)
            if matrix.shape[1] != size:
                raise ValueError("Projection and covariance coordinates disagree")
            matrix.sum_duplicates()
            matrix.sort_indices()
            # Sauce reads 64-bit row offsets; column indices stay 32-bit.
            arrays.update(
                offsets=matrix.indptr.astype(np.int64),
                indices=matrix.indices.astype(np.int32),
                weights=matrix.data,
            )
            method = "covariance_project"
        else:
            arrays["vectors"] = np.atleast_2d(
                _real_input(vectors, "covariance directions")
            )
            if arrays["vectors"].shape[1] != size:
                raise ValueError("Covariance directions have wrong coordinates")
            method = "covariance_action"
        return self._execute(
            method,
            arrays,
            state=factors.metadata["state"],
            coordinates=factors.metadata["coordinates"],
            expect=dict(controls=size),
            factors=factors,
            rank=rank,
        )

    def rickett(
        self,
        reference: Any,
        image: Any,
        *,
        normal: Any = None,
        normal_reference: Any = None,
        state: str,
        coordinates: str,
        depth_axis: int = 0,
        smoothing_radii: Any = 0,
        damping: Optional[float] = None,
        relative_damping: Optional[float] = None,
        padding: Optional[int] = None,
    ) -> CurvatureResult:
        """Schedule frozen modeling/remigration and native Rickett Eq. 7.

        Supply exactly one of ``normal`` (e.g. ``linearization.normal``) and
        ``normal_reference``. Inputs use the same regular Cartesian image/control
        grid. ``normal`` excludes regularization and data-dependent normalization.
        Damping is the additive ``epsilon**2`` in remigrated-envelope units;
        ``relative_damping`` is a fraction of the maximum smoothed remigrated
        envelope. Supply at most one; without either, relative damping is
        ``1e-2``, and an explicit zero disables damping. ``padding`` reflects
        that many depth samples at both ends before the FFT; the default is the
        full depth extent and zero is periodic. Padding and the chosen damping
        are always sent explicitly, so solver defaults never apply.
        Output vectors use depth-fast order; reshape using ``grid_shape`` in
        reverse order and move the last axis back to ``depth_axis``.
        """
        reference = _real_input(reference, "reference")
        image = _real_input(image, "migration")
        if (
            reference.ndim not in (1, 2, 3)
            or not reference.size
            or reference.shape != image.shape
        ):
            raise ValueError("reference and image need the same 1D/2D/3D grid")
        if (normal is None) == (normal_reference is None):
            raise ValueError(
                "Supply exactly one normal operator or remigrated reference"
            )
        axis = _integer(depth_axis, "depth_axis", minimum=-reference.ndim)
        if axis < -reference.ndim or axis >= reference.ndim:
            raise ValueError("depth_axis is outside the grid")
        axis %= reference.ndim
        radii = np.broadcast_to(np.asarray(smoothing_radii), (reference.ndim,))
        if not np.issubdtype(radii.dtype, np.integer) or np.any(radii < 0):
            raise ValueError("smoothing radii must be nonnegative integers")
        if np.any(radii > reference.shape):
            raise ValueError("smoothing radius exceeds its grid axis")
        padding = (
            reference.shape[axis] if padding is None else _integer(padding, "padding")
        )
        if damping is not None and relative_damping is not None:
            raise ValueError("Absolute and relative damping are mutually exclusive")
        name, value = (
            ("damping", damping)
            if damping is not None
            else (
                "relative_damping",
                1e-2 if relative_damping is None else relative_damping,
            )
        )
        chosen: dict[str, Any] = {name: float(value)}
        if not np.isfinite(chosen[name]) or chosen[name] < 0:
            raise ValueError(f"{name} must be nonnegative finite")
        if normal is not None:
            direction = reference.reshape(-1)
            action = (
                normal @ direction if hasattr(normal, "matvec") else normal(direction)
            )
            normal_reference = np.asarray(action).reshape(reference.shape)
        remigrated = _real_input(normal_reference, "remigrated reference")
        if remigrated.shape != reference.shape:
            raise ValueError("remigrated reference grid differs")
        packed = np.moveaxis(reference, axis, -1)
        order = [i for i in range(reference.ndim) if i != axis] + [axis]
        arrays = dict(
            reference=packed.reshape(-1),
            image=np.moveaxis(image, axis, -1).reshape(-1),
            normal_reference=np.moveaxis(remigrated, axis, -1).reshape(-1),
            grid_shape=np.asarray(packed.shape[::-1], dtype=np.int32),
            smoothing_radii=np.asarray(radii[order][::-1], dtype=np.int32),
        )
        return self._execute(
            "rickett",
            arrays,
            state=state,
            coordinates=coordinates,
            expect=dict(controls=reference.size, padding=padding),
            padding=padding,
            **chosen,
        )
