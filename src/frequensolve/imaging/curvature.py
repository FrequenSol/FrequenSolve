# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Orchestration of Sauce's inverse-curvature and reference-image operations.

Secant recursion, random probing, eigendecomposition, posterior variances,
analytic envelopes, smoothing and image weighting are evaluated by Sauce.
"""

from __future__ import annotations

import json
import operator
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np

__all__ = ["BFGSHistory", "CurvatureResult", "NativeCurvature"]


def _real(value: Any, name: str) -> np.ndarray:
    """Validate an input array without altering its physical coordinates."""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.array(array, copy=True)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    """Validate JSON/native integer options without silently truncating floats."""
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, (bool, np.bool_)) or not minimum <= result <= 2147483647:
        raise ValueError(f"{name} is outside the native integer range")
    return result


class BFGSHistory:
    """Archive a fixed inverse diagonal and optimizer-produced curvature pairs.

    Use as an L-BFGS callback (directly or with ``FWI(callback=history)``).
    Set optimizer memory large enough for the entire stage and freeze its
    preconditioner. Pairs use the optimizer's coordinates; with FWI, use
    ``scaling=None`` or explicitly transform the base and prior into the
    scaled coordinates. A new objective/stage requires a new archive.
    Losing pairs through truncation, reset, or a changed stage raises instead
    of silently reporting a full-history covariance.
    """

    def __init__(
        self, base_inverse_diagonal: Any, *, state: str, coordinates: str
    ) -> None:
        base = _real(base_inverse_diagonal, "base inverse diagonal")
        if base.ndim != 1 or not base.size or np.any(base <= 0):
            raise ValueError("base inverse diagonal must be a positive vector")
        if not state or not coordinates:
            raise ValueError("state and coordinate identities are required")
        self.base_inverse_diagonal = base
        self.state = str(state)
        self.coordinates = str(coordinates)
        self.steps = np.empty((0, base.size))
        self.gradient_differences = np.empty((0, base.size))
        self._stage: Optional[int] = None
        self._accepted_iterations = 0

    def __call__(self, event: Any) -> None:
        """Retain native optimizer checkpoint pairs, rejecting history loss."""
        stage = getattr(event, "stage_index", None)
        if stage is not None:
            if self._stage is not None and stage != self._stage:
                raise ValueError(
                    "BFGS uncertainty requires a separate history per stage"
                )
            self._stage = stage
        diagnostics = getattr(event, "diagnostics", event)
        checkpoint = diagnostics.optimizer_state
        if checkpoint is None or checkpoint.get("schema") != "fs-lbfgs-restart-1":
            raise ValueError("BFGS history requires an L-BFGS optimizer checkpoint")
        accepted = int(checkpoint["accepted_iterations"])
        if accepted < self._accepted_iterations:
            raise ValueError("BFGS history iteration count moved backwards")
        pairs = checkpoint["pairs"]
        size = self.base_inverse_diagonal.size
        steps = np.asarray([p["step"] for p in pairs], dtype=float).reshape(-1, size)
        differences = np.asarray([p["difference"] for p in pairs], dtype=float).reshape(
            -1, size
        )
        previous = len(self.steps)
        if len(steps) < previous or not np.array_equal(steps[:previous], self.steps):
            raise ValueError(
                "BFGS history was truncated or reset; full-history UQ is unavailable"
            )
        if not np.array_equal(differences[:previous], self.gradient_differences):
            raise ValueError("BFGS gradient-difference history changed")
        self.steps = _real(steps, "BFGS steps")
        self.gradient_differences = _real(differences, "BFGS differences")
        self._accepted_iterations = accepted

    def save(self, path: Any) -> Path:
        """Persist the exact coordinates and secants for backend replay."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            h5.create_dataset("base_inverse_diagonal", data=self.base_inverse_diagonal)
            h5.create_dataset("steps", data=self.steps)
            h5.create_dataset("gradient_differences", data=self.gradient_differences)
            h5.create_dataset(
                "metadata",
                data=json.dumps(
                    dict(
                        schema="fs-bfgs-history-1",
                        state=self.state,
                        coordinates=self.coordinates,
                        accepted_iterations=self._accepted_iterations,
                        stage=self._stage,
                    )
                ),
            )
        return path

    @classmethod
    def load(cls, path: Any) -> "BFGSHistory":
        """Restore a full archive without reconstructing numerical curvature."""
        with h5py.File(path, "r") as h5:
            metadata = json.loads(h5["metadata"][()])
            if metadata.get("schema") != "fs-bfgs-history-1":
                raise ValueError("Unsupported BFGS history schema")
            history = cls(
                h5["base_inverse_diagonal"][()],
                state=metadata["state"],
                coordinates=metadata["coordinates"],
            )
            history.steps = _real(h5["steps"][()], "BFGS steps")
            history.gradient_differences = _real(
                h5["gradient_differences"][()], "BFGS differences"
            )
            if (
                history.steps.shape != history.gradient_differences.shape
                or history.steps.ndim != 2
            ):
                raise ValueError("BFGS history shapes disagree")
            if history.steps.shape[1] != history.base_inverse_diagonal.size:
                raise ValueError("BFGS history coordinates disagree")
            history._accepted_iterations = int(metadata["accepted_iterations"])
            if history._accepted_iterations < len(history.steps):
                raise ValueError("BFGS history iteration count is inconsistent")
            history._stage = metadata.get("stage")
        return history


@dataclass(frozen=True)
class CurvatureResult:
    """Backend-owned HDF5 result with explicit state and coordinate identities."""

    path: Path
    metadata: dict

    def read(self, name: str) -> np.ndarray:
        """Read a backend output dataset; no covariance algebra runs here."""
        with h5py.File(self.path, "r") as h5:
            return h5[name][()]


class NativeCurvature:
    """Invoke one-rank Sauce postprocessing locally or through a supplied runner.

    ``runner(request_path)`` may stage files and execute a pinned remote solver;
    it must return only after the request's output is available locally.
    Each operation has an independent directory and publishes only verified
    completed output. Existing results are never reused implicitly.
    """

    def __init__(
        self,
        executable: Any = None,
        *,
        workdir: Any,
        runner: Optional[Callable[[Path], None]] = None,
    ) -> None:
        if executable is None and runner is None:
            raise ValueError("A Sauce executable or runner is required")
        self.executable = (
            None if executable is None else str(Path(executable).resolve())
        )
        self.workdir = Path(workdir).resolve()
        self.runner = runner

    def _execute(
        self, method: str, arrays: dict, *, state: str, coordinates: str, **options: Any
    ) -> CurvatureResult:
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
        with h5py.File(input_path, "w") as h5:
            for name, value in arrays.items():
                h5.create_dataset(name, data=value)
            h5.create_dataset("metadata", data=np.bytes_(json.dumps(metadata)))
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
        path = directory / "result.h5"
        pending.replace(path)
        return CurvatureResult(path, result)

    @staticmethod
    def _history_arrays(history: BFGSHistory) -> dict:
        return dict(
            base_inverse_diagonal=history.base_inverse_diagonal,
            steps=history.steps,
            gradient_differences=history.gradient_differences,
        )

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
        Default rank covers the correction's possible ``2 * pairs`` range.
        """
        arrays = self._history_arrays(history)
        arrays["prior_std"] = np.broadcast_to(
            _real(prior_std, "prior standard deviation"),
            history.base_inverse_diagonal.shape,
        ).copy()
        if (
            not np.isfinite(curvature_tolerance)
            or not np.finfo(float).eps <= curvature_tolerance < 1
        ):
            raise ValueError("Invalid relative curvature_tolerance")
        options = dict(
            curvature_tolerance=float(curvature_tolerance),
            oversampling=_integer(oversampling, "oversampling"),
            seed=_integer(seed, "seed", minimum=-2147483648),
        )
        if rank is not None:
            options["rank"] = _integer(rank, "rank")
        return self._execute(
            "bfgs_rsvd",
            arrays,
            state=history.state,
            coordinates=history.coordinates,
            **options,
        )

    def inverse_action(self, history: BFGSHistory, vectors: Any) -> CurvatureResult:
        """Schedule a backend compact inverse action on vectors or probe batches."""
        arrays = self._history_arrays(history)
        value = _real(vectors, "inverse directions")
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != history.base_inverse_diagonal.size:
            raise ValueError("inverse directions must have shape (probes, controls)")
        arrays["vectors"] = value
        return self._execute(
            "bfgs_action",
            arrays,
            state=history.state,
            coordinates=history.coordinates,
        )

    def gaussian_prior(
        self, point: Any, reference: Any, std: Any, *, state: str, coordinates: str
    ) -> CurvatureResult:
        """Evaluate a Gaussian prior and its derivatives in Sauce."""
        return self._execute(
            "gaussian_prior",
            dict(
                point=_real(point, "point"),
                reference=_real(reference, "reference"),
                prior_std=_real(std, "prior std"),
            ),
            state=state,
            coordinates=coordinates,
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
            reference=_real(reference, "prior reference"),
            prior_std=_real(std, "prior scale"),
        )
        return self._execute(
            "mesh_prior",
            arrays,
            state=source["identity"],
            coordinates=target["identity"],
            source_mesh=source["path"],
            target_mesh=target["path"],
            source_identity=source["identity"],
            target_identity=target["identity"],
            material=target["material"],
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
                points=_real(points, "sample points") / metres,
            ),
            state=mesh["identity"],
            coordinates=mesh["identity"],
            target_mesh=mesh["path"],
            target_identity=mesh["identity"],
            material=mesh["material"],
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
        """Apply covariance factors or project their diagonal through a CSR map."""
        if (vectors is None) == (projection is None):
            raise ValueError("Supply either vectors or a CSR projection")
        rank = int(factors.metadata["rank"])
        # Sauce reads the immutable factors directly; only directions/CSR are staged.
        with h5py.File(factors.path) as h5:
            size = h5["prior_std"].size
        arrays: dict[str, Any] = {}
        if projection is not None:
            from scipy.sparse import csr_matrix

            matrix = csr_matrix(projection, dtype=float, copy=True)
            if matrix.shape[1] != size:
                raise ValueError("Projection and covariance coordinates disagree")
            matrix.sum_duplicates()
            matrix.sort_indices()
            if matrix.nnz > np.iinfo(np.int32).max:
                raise ValueError("Projection exceeds native CSR capacity")
            arrays.update(
                offsets=matrix.indptr.astype(np.int32),
                indices=matrix.indices.astype(np.int32),
                weights=matrix.data,
            )
            method = "covariance_project"
        else:
            arrays["vectors"] = np.atleast_2d(_real(vectors, "covariance directions"))
            if arrays["vectors"].shape[1] != size:
                raise ValueError("Covariance directions have wrong coordinates")
            method = "covariance_action"
        return self._execute(
            method,
            arrays,
            state=factors.metadata["state"],
            coordinates=factors.metadata["coordinates"],
            rank=rank,
            factors=str(factors.path.resolve()),
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
        damping: float = 0.0,
        relative_damping: Optional[float] = None,
        padding: int = 0,
    ) -> CurvatureResult:
        """Schedule frozen modeling/remigration and native Rickett Eq. 7.

        Supply exactly one of ``normal`` (e.g. ``linearization.normal``) and
        ``normal_reference``. Inputs use the same regular Cartesian image/control
        grid. ``normal`` excludes regularization and data-dependent normalization.
        Damping is the additive ``epsilon**2`` in remigrated-envelope units.
        Output vectors use depth-fast order; reshape using ``grid_shape`` in
        reverse order and move the last axis back to ``depth_axis``.
        """
        padding = _integer(padding, "padding")
        reference = _real(reference, "reference")
        image = _real(image, "migration")
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
        if not np.isfinite(damping) or damping < 0:
            raise ValueError("damping must be nonnegative finite")
        if relative_damping is not None and (
            not np.isfinite(relative_damping) or relative_damping < 0 or damping != 0
        ):
            raise ValueError(
                "relative_damping must be nonnegative finite and excludes absolute damping"
            )
        if normal is not None:
            direction = reference.reshape(-1)
            action = (
                normal @ direction if hasattr(normal, "matvec") else normal(direction)
            )
            normal_reference = np.asarray(action).reshape(reference.shape)
        remigrated = _real(normal_reference, "remigrated reference")
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
            padding=padding,
            state=state,
            coordinates=coordinates,
            **(
                {"damping": float(damping)}
                if relative_damping is None
                else {"relative_damping": float(relative_damping)}
            ),
        )
