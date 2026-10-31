# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Reduced receiver curvature for native HV's coupled linearized auxiliaries."""

import json
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


class HVReceiverCurvature:
    """Eliminate free signal/deformation DOFs and return endpoint reactions.

    The full matrix is the native mixed (f, v, q) Gauss–Newton system. Its
    auxiliary block is indefinite because q represents velocity curvature;
    its reduced receiver operator is positive semidefinite. Factorizations
    and solves use FP32, with residual refinement to relative tolerance 1e-4.
    """

    def __init__(self, matrix, trace, free, *, scale=1.0, weight=1.0):
        self.matrix = matrix.tocsr().astype(np.float64)
        self.trace = trace.tocsr().astype(np.float64)
        self.free = np.asarray(free, dtype=np.int64).reshape(-1)
        size = self.matrix.shape[0]
        if (
            self.matrix.shape != (size, size)
            or self.trace.shape[0] != size
            or np.any(self.free < 0)
            or np.any(self.free >= size)
            or len(np.unique(self.free)) != len(self.free)
            or not np.isfinite(self.matrix.data).all()
            or not np.isfinite(self.trace.data).all()
        ):
            raise ValueError("Invalid HV curvature matrix, trace map, or free DOFs")
        if self.trace[self.free].nnz:
            raise ValueError("HV trace map must prescribe boundary DOFs only")
        if (
            not np.isfinite(scale)
            or scale <= 0
            or not np.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("HV curvature scale and weight must be positive")
        asymmetry = self.matrix - self.matrix.T
        if np.linalg.norm(asymmetry.data) > 1e-8 * np.linalg.norm(self.matrix.data):
            raise ValueError("HV Gauss–Newton auxiliary matrix is not symmetric")
        self.multiplier = weight / scale**2
        self.interior = self.matrix[self.free][:, self.free].tocsc()
        self.factor = splu(self.interior.astype(np.float32)) if len(self.free) else None

    @classmethod
    def read(cls, path: Path):
        """Read one shot's real or imaginary native auxiliary linearization."""
        with h5py.File(path) as handle:
            schema = handle["schema"][()]
            if isinstance(schema, np.ndarray):
                schema = schema.item()
            if isinstance(schema, bytes):
                schema = schema.decode()
            schema = schema.rstrip(" \x00")
            if schema != "fs-hv-receiver-gn-1":
                raise ValueError(f"Unsupported HV curvature schema in {path}")
            size = int(np.asarray(handle["size"]).item())
            receivers = np.asarray(handle["receiver_id"], dtype=np.int64).reshape(-1)

            def sparse(name, shape):
                group = handle[name]
                return coo_matrix(
                    (
                        np.asarray(group["value"]).reshape(-1),
                        (
                            np.asarray(group["row"], dtype=np.int64).reshape(-1),
                            np.asarray(group["column"], dtype=np.int64).reshape(-1),
                        ),
                    ),
                    shape=shape,
                )

            result = cls(
                sparse("matrix", (size, size)),
                sparse("trace", (size, len(receivers))),
                np.asarray(handle["free_dofs"]),
                scale=float(np.asarray(handle["amplitude_scale"]).item()),
                weight=float(np.asarray(handle["term_weight"]).item()),
            )
            result.receiver_ids = receivers
            result.source_id = int(np.asarray(handle["source_id"]).item())
            result.predicted = np.asarray(handle["predicted"]).reshape(-1)
            result.observed = np.asarray(handle["observed"]).reshape(-1)
            result.scale = float(np.asarray(handle["amplitude_scale"]).item())
        return result

    def apply(self, values):
        """Apply Bᵀ(K − K[:, I] K[I, I]⁻¹ K[I, :])B to real samples."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (self.trace.shape[1],) or not np.isfinite(values).all():
            raise ValueError("HV receiver direction has an invalid shape or values")
        state = np.asarray(self.trace @ values).reshape(-1)
        if self.factor is not None:
            rhs = -(self.matrix @ state)[self.free]
            solution = self.factor.solve(rhs.astype(np.float32)).astype(np.float64)
            norm = np.linalg.norm(rhs)
            for _ in range(8):
                residual = rhs - self.interior @ solution
                if np.linalg.norm(residual) <= 1e-4 * norm:
                    break
                solution += self.factor.solve(residual.astype(np.float32))
            else:
                raise RuntimeError(
                    "HV auxiliary GN solve did not reach relative tolerance 1e-4"
                )
            state[self.free] = solution
        return self.multiplier * np.asarray(
            self.trace.T @ (self.matrix @ state)
        ).reshape(-1)


def _tangent_values(path, collection):
    """Read verified physical tangent rows in each receiver state's order."""
    from ._receiver import _member

    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["schema"] == "fs-receiver-vector-bundle-1":
        members = [_member(path, entry) for entry in manifest["groups"]]
    elif manifest["schema"] == "fs-receiver-vector-1":
        members = [path]
    else:
        raise ValueError("Expected a native receiver tangent")
    result = {}
    for member in members:
        meta = json.loads(member.read_text())
        name = meta["receiver_group"]
        state = collection.groups[name]
        if (
            name in result
            or meta.get("role") != "tangent"
            or meta["state_fingerprint"] != state.manifest["state_fingerprint"]
            or meta["source_state_sha256"] != state.checksum
        ):
            raise ValueError("HV tangent does not belong to its receiver state")
        keys, values = [], []
        for entry in meta["shards"]:
            with h5py.File(_member(member, entry)) as handle:
                keys.append(np.asarray(handle["keys"]))
                raw = np.asarray(handle["base"])
                values.append(raw[:, 0] + 1j * raw[:, 1])
        keys = np.concatenate(keys)
        order = np.lexsort(keys.T[::-1])
        if not np.array_equal(keys[order], state.keys):
            raise ValueError("HV tangent physical receiver keys differ")
        result[name] = np.concatenate(values)[order]
    if set(result) != set(collection.groups):
        raise ValueError("HV tangent does not cover every receiver group")
    return result


class HVReceiverNormal:
    """Native wave JVP → reduced HV receiver metric → native wave VJP."""

    def __init__(self, linearization):
        from ._backend import fingerprint
        from ._receiver import ReceiverCollection
        from .jobs import FWIOperatorJob
        from .misfit import Misfit

        self.linearization = linearization
        problem = linearization.problem._problem
        self.problem = problem.restrict(
            misfit=Misfit.l2(include_default_preprocess=False), weights=None
        )
        self.directory = problem.workdir / f"hv_gn_{uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.misfit = self.problem._misfit_payload.to_fs()
        for group in self.misfit["receiver_groups"]:
            group["observed_source_basis"] = "source_encoding"
            if isinstance(group.get("observed"), dict):
                group["observed"]["source_basis"] = "source_encoding"
        self.problem._sync_simulation(linearization.state)
        _, control_state = self.problem._stage_state(
            fingerprint(hv_gn_state=linearization.state.values), linearization.state
        )
        self.job = FWIOperatorJob(
            problem.backend.job_name("hv_receiver"),
            self.problem.simulation,
            self.problem.frequencies,
            action="receiver_linearize",
            control_state=control_state,
            active=list(linearization.space.blocks),
            receiver_state=self.directory / "state.json",
            channel="base",
            source_derivative="none",
            field_retention="checkpoint",
            misfit=self.misfit,
        )
        self.problem._run_job(self.job)
        self.states = [
            ReceiverCollection.read(self.job.receiver_state_file(task))
            for task in range(1, self.job.n_tasks + 1)
        ]
        self.metrics = []
        terms = linearization.problem.hv.payload(problem.observed_groups)[
            "objective_terms"
        ]
        root = linearization.curvature_job._result_path
        for task, collection in enumerate(self.states, 1):
            current = []
            for term_index, term in enumerate(terms, 1):
                name = term["receiver_group"]
                state = collection.groups[name]
                lookup = {tuple(key): i for i, key in enumerate(state.keys)}
                for source in np.unique(state.keys[:, 0]):
                    for component in ("real", "imag"):
                        path = root / (
                            f"hv_curvature_task_{task:06d}_term_{term_index:04d}_"
                            f"source_{source:06d}_{component}.h5"
                        )
                        metric = HVReceiverCurvature.read(path)
                        if metric.source_id != source:
                            raise ValueError("HV curvature source identity differs")
                        rows = np.array(
                            [
                                lookup[(source, receiver, 1)]
                                for receiver in metric.receiver_ids
                            ]
                        )
                        values = getattr(state.values, component)
                        for column, recorded in (
                            (0, metric.predicted),
                            (2, metric.observed),
                        ):
                            if not np.allclose(
                                values[rows, column],
                                recorded,
                                rtol=1e-3,
                                atol=1e-6 * metric.scale,
                            ):
                                raise ValueError(
                                    "HV curvature does not match the physical receiver state"
                                )
                        current.append((name, rows, component, metric))
            self.metrics.append(current)
        self.applications = 0

    def apply(self, direction):
        from ._artifacts import ControlVectorFile
        from .controls import ControlVector, _check_artifact_basis
        from .jobs import FWIOperatorJob
        from .problem import _vector_from_file

        lin = self.linearization
        if not isinstance(direction, ControlVector):
            direction = ControlVector(direction, lin.space)
        elif not direction.space.equivalent(lin.space):
            raise ValueError("HV normal direction belongs to another control space")
        if not np.any(direction.values):
            return ControlVector(np.zeros(lin.space.size), lin.space)
        self.problem._sync_simulation(lin.state)
        self.applications += 1
        prefix = self.directory / f"normal_{self.applications:04d}"

        def input_path(path):
            # Single-task packaging requires a concrete input, whereas a band
            # resolves its per-task members from the common stem.
            return (
                path.with_name(f"{path.stem}_1{path.suffix}")
                if len(self.states) == 1
                else path
            )

        for task, collection in enumerate(self.states, 1):
            registry = next(iter(collection.groups.values())).manifest[
                "control_registry_fingerprint"
            ]
            direction.to_file(
                state_fingerprint=collection.fingerprint, registry_fingerprint=registry
            ).write(prefix.with_name(f"{prefix.name}_direction_{task}.h5"))
        tangent = FWIOperatorJob(
            self.problem.backend.job_name("hv_jvp"),
            self.problem.simulation,
            self.problem.frequencies,
            action="receiver_jvp",
            control_state=self.job.control_state,
            active=list(lin.space.blocks),
            receiver_state=input_path(self.directory / "state.json"),
            receiver_vector=prefix.with_name(f"{prefix.name}_tangent.json"),
            direction=input_path(prefix.with_name(f"{prefix.name}_direction.h5")),
            misfit=self.misfit,
        )
        self.problem._run_job(tangent)
        for task, (collection, metrics) in enumerate(zip(self.states, self.metrics), 1):
            values = _tangent_values(tangent.receiver_vector_file(task), collection)
            dual = {name: np.zeros_like(value) for name, value in values.items()}
            for name, rows, component, metric in metrics:
                contribution = metric.apply(getattr(values[name], component)[rows])
                if component == "imag":
                    contribution = 1j * contribution
                dual[name][rows] += contribution
            if self.problem.weights is not None:
                raise RuntimeError("Internal receiver problem must be unweighted")
            weight = 1 if lin.problem.weights is None else lin.problem.weights[task - 1]
            collection.write_dual(
                prefix.with_name(f"{prefix.name}_dual_{task}.json"),
                {
                    name: (weight * value, np.zeros_like(value))
                    for name, value in dual.items()
                },
            )
        pullback = FWIOperatorJob(
            self.problem.backend.job_name("hv_vjp"),
            self.problem.simulation,
            self.problem.frequencies,
            action="receiver_vjp",
            control_state=self.job.control_state,
            active=list(lin.space.blocks),
            receiver_state=input_path(self.directory / "state.json"),
            receiver_vector=input_path(prefix.with_name(f"{prefix.name}_dual.json")),
            covector=prefix.with_name(f"{prefix.name}_covector.h5"),
            field_reuse="checkpoint",
            misfit=self.misfit,
        )
        self.problem._run_job(pullback)
        total = np.zeros(lin.space.size)
        for task, collection in enumerate(self.states, 1):
            part = ControlVectorFile.read(pullback.covector_file(task), native=False)
            registry = next(iter(collection.groups.values())).manifest[
                "control_registry_fingerprint"
            ]
            if (
                part.state_fingerprint != collection.fingerprint
                or part.control_registry_fingerprint != registry
            ):
                raise ValueError(
                    "HV normal covector has a different receiver state or registry"
                )
            _check_artifact_basis(part, lin.space)
            total += _vector_from_file(part, lin.space).values
        return ControlVector(total, lin.space)
