"""Solver-free execution site for imaging tests.

:class:`FakeImagingSite` executes :class:`~frequensolve.imaging.jobs.FWIOperatorJob`
requests by writing the artifacts Sauce would write, computed from a
deterministic *linear* surrogate of the forward problem:

``F(m) = J m``, ``residual = J m - d``, ``objective = 0.5 * ||J m - d||^2``

``J`` is a seeded complex matrix of shape ``(data_space.size, n_active)`` and
``d`` a seeded observed vector.  Both depend only on the seed, the simulation
name, the active blocks, their sizes and the frequencies of the job, so every
job of one problem sees the same operator; the rows of one frequency do not
depend on which other frequencies the job carries.

Per action (``task`` is one-based; ``f`` its frequency; ``R`` the rows of
``f``; ``m`` the active slice of ``controls.state`` or zeros):

``linearize``
    writes ``state_<task>.json`` (a stub holding the invented fingerprints,
    active blocks, sizes, ``m`` and ``f``), ``<report stem>_<task>.json``
    (``fs-objective-report-1`` with ``total = 0.5*||J[R] m - d[R]||^2`` and one
    term per receiver group), the covector ``Re(J[R]^H (J[R] m - d[R]))`` as
    ``<covector stem>_<task>.h5`` (``fs-control-vector-1`` with all-ones
    ``/support/<block>`` masks and the fingerprints), plus ``state_output``
    (``fs-control-state-1``) and ``manifest`` (``fs-control-registry-1``) when
    requested.
``jvp``
    reads ``direction`` and writes ``<objective_vector stem>_<task>.json`` with
    ``J[R] dv`` (``fs-objective-vector-3``).
``vjp``
    reads ``<objective_vector stem>_<task>.json`` (or the unsuffixed file) and
    writes ``Re(J[R]^H r[R])`` as the task covector.
``normal``
    writes ``Re(J[R]^H J[R] dv)`` as the task covector.

When ``job.smoothing`` is set the site emulates the ``--smooth`` postprocess:
the unsuffixed covector receives the frequency-weighted sum of the parts and
``<stem>_raw.h5`` receives the same vector (the fake's smoothing is the
identity).  Every direction and objective vector must carry the state and
registry fingerprints of the state it is applied to, as Sauce requires.

Forward modelling (``FrequencyDomainJob``) is not executed; use
:meth:`FakeImagingSite.forward` and :meth:`FakeImagingSite.observed` instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frequensolve.imaging._artifacts import (
    ControlStateFile,
    ControlVectorFile,
    qualified_block_name,
)
from frequensolve.imaging.data import DataSpace, DataVector
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunHandle,
    RunResult,
)

FAKE_STATE_SCHEMA = "fake-imaging-state-1"


def _sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(list(parts), sort_keys=True, default=str).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class FakeLinearization:
    """One linearization the fake produced (for reference computations)."""

    state_fingerprint: str
    control_registry_fingerprint: str
    active: Tuple[str, ...]
    sizes: Dict[str, int]
    frequencies: Tuple[complex, ...]
    space: DataSpace
    J: np.ndarray
    d: np.ndarray
    m: np.ndarray

    def rows(self, frequency: Any) -> np.ndarray:
        """Return the data-space rows of ``frequency``."""

        return _rows(self.space, frequency)

    @property
    def residual(self) -> np.ndarray:
        return self.J @ self.m - self.d

    @property
    def gradient(self) -> np.ndarray:
        """Return ``Re(J^H (J m - d))`` summed over every frequency."""

        return np.real(self.J.conj().T @ self.residual)


def _rows(space: DataSpace, frequency: Any) -> np.ndarray:
    return np.concatenate(
        [layout.indices for layout in space.term_layouts(frequency=frequency)]
    )


class FakeImagingSite(BaseSite):
    """Execute ``fwi_operator`` jobs against a seeded linear surrogate.

    Args:
        block_sizes: Real DOF count per qualified block (``model.vp`` ...).
            Required when a job has no ``controls.state``; when it has one the
            file's sizes are used and checked against this table.
        seed: Seed of the surrogate.
        n_ranks: Rank count recorded in objective-vector manifests.
        verbose: Print status messages like other sites.
    """

    def __init__(
        self,
        block_sizes: Optional[Mapping[str, int]] = None,
        *,
        seed: int = 0,
        n_ranks: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose=verbose)
        self.block_sizes: Dict[str, int] = {
            qualified_block_name(name): int(size)
            for name, size in dict(block_sizes or {}).items()
        }
        self.seed = int(seed)
        self.n_ranks = int(n_ranks)
        self.submissions: List[Dict[str, Any]] = []
        self.linearizations: Dict[str, FakeLinearization] = {}

    # -- surrogate ------------------------------------------------------------

    def registry_fingerprint(
        self, simulation: Any, active: Sequence[str], sizes: Mapping[str, int]
    ) -> str:
        """Return the control registry fingerprint of one active subspace."""

        return _sha256(
            {
                "kind": "registry",
                "seed": self.seed,
                "simulation": simulation.name,
                "active": [qualified_block_name(name) for name in active],
                "sizes": {qualified_block_name(k): int(v) for k, v in sizes.items()},
            }
        )

    def state_fingerprint(self, registry_fingerprint: str, m: np.ndarray) -> str:
        """Return the state fingerprint of baseline ``m`` (task independent)."""

        return _sha256(
            {
                "kind": "state",
                "registry": registry_fingerprint,
                "m": np.asarray(m, dtype=np.float64).round(12).tolist(),
            }
        )

    def surrogate(
        self,
        simulation: Any,
        active: Sequence[str],
        sizes: Mapping[str, int],
        frequencies: Sequence[Any],
    ) -> Tuple[np.ndarray, np.ndarray, DataSpace]:
        """Return ``(J, d, space)`` for one simulation, subspace and frequencies."""

        names = [qualified_block_name(name) for name in active]
        n = int(sum(int(sizes[name]) for name in names))
        space = DataSpace.from_simulation(simulation, frequencies=frequencies)
        J = np.zeros((space.size, n), dtype=np.complex128)
        d = np.zeros(space.size, dtype=np.complex128)
        key = [self.seed, simulation.name, names, [int(sizes[k]) for k in names]]
        for frequency in space.frequencies:
            f = complex(frequency)
            for layout in space.term_layouts(frequency=frequency):
                rng = np.random.default_rng(_seed(*key, [f.real, f.imag], layout.id))
                rows = layout.indices
                J[rows, :] = rng.standard_normal((rows.size, n)) + 1j * (
                    rng.standard_normal((rows.size, n))
                )
                d[rows] = rng.standard_normal(rows.size) + 1j * rng.standard_normal(
                    rows.size
                )
        return J, d, space

    def forward(
        self,
        simulation: Any,
        vector: Union[ControlVectorFile, ControlStateFile, np.ndarray],
        frequencies: Sequence[Any],
        *,
        active: Optional[Sequence[str]] = None,
        sizes: Optional[Mapping[str, int]] = None,
    ) -> DataVector:
        """Return ``J m`` for ``vector`` over ``frequencies``.

        ``vector`` may be a control vector (its blocks are the active
        subspace), a control state (``active`` selects the blocks), or a
        packed array (``active`` and ``sizes`` required).
        """

        names, table, m = self._coefficients(vector, active, sizes)
        J, _, space = self.surrogate(simulation, names, table, frequencies)
        return DataVector(J @ m, space)

    def observed(
        self,
        simulation: Any,
        active: Sequence[str],
        frequencies: Sequence[Any],
        *,
        sizes: Optional[Mapping[str, int]] = None,
    ) -> DataVector:
        """Return the surrogate observed data ``d``."""

        _, d, space = self.surrogate(
            simulation, active, self._sizes(active, sizes), frequencies
        )
        return DataVector(d, space)

    # -- site protocol --------------------------------------------------------

    def submit(self, job: Any, *, check: bool = False, **kwargs: Any) -> RunHandle:
        """Execute ``job`` synchronously and return a completed handle."""

        options = dict(kwargs)
        postprocess_only = bool(options.pop("postprocess_only", False))
        validate = bool(options.pop("validate", True))
        self.submissions.append(
            {
                "job": job.name,
                "action": getattr(job, "action", None),
                "options": options,
            }
        )
        if not isinstance(job, FWIOperatorJob):
            raise TypeError(
                f"{type(self).__name__} executes FWIOperatorJob only; "
                f"received {type(job).__name__}"
            )
        self.prepare_job(job, validate=validate)
        job._job_id = f"fake:{job.name}"
        started = datetime.now()
        try:
            if postprocess_only:
                self._postprocess(job)
            else:
                self._execute(job)
        except Exception as exc:  # the fake reports failures like a site would
            status = JobStatus(
                state="failed",
                return_code=1,
                job_id=job._job_id,
                message=f"{type(exc).__name__}: {exc}",
                start_time=started,
                end_time=datetime.now(),
            )
            job.write_run_state(status="failed", error=str(exc))
        else:
            status = JobStatus(
                state="completed",
                return_code=0,
                job_id=job._job_id,
                message=f"fake {job.action} over {job.n_tasks} task(s)",
                start_time=started,
                end_time=datetime.now(),
            )
            job.write_run_state(status="completed")
        handle = RunHandle(
            site=self,
            job=job,
            id=job._job_id,
            mode="fake",
            poll_interval=0.0,
            check=check,
        )
        handle._last_status = status
        handle._result = RunResult(job=job, status=status, site=self)
        self._emit_status(status, force=False)
        return handle

    def cancel_job(self, job_id: str) -> bool:
        return False

    # -- execution ------------------------------------------------------------

    def _execute(self, job: FWIOperatorJob) -> None:
        if job.action not in {"linearize", "jvp", "vjp", "normal"}:
            raise NotImplementedError(
                f"{type(self).__name__} does not execute action {job.action!r}"
            )
        if job.extension is not None or job.reflectivity is not None:
            raise NotImplementedError("the fake site has no extension or reflectivity")
        if job.action == "linearize":
            self._linearize(job)
        else:
            self._apply(job)
        if job.requires_postprocess():
            self._postprocess(job)

    def _linearize(self, job: FWIOperatorJob) -> None:
        active = list(job.active or [])
        baseline = self._baseline(job)
        sizes = {name: int(baseline[name].size) for name in active}
        m = (
            np.concatenate([baseline[name] for name in active])
            if active
            else np.zeros(0)
        )
        registry = self.registry_fingerprint(job.simulation, active, sizes)
        state_fp = self.state_fingerprint(registry, m)
        J, d, space = self.surrogate(job.simulation, active, sizes, job.f_list)
        lin = FakeLinearization(
            state_fingerprint=state_fp,
            control_registry_fingerprint=registry,
            active=tuple(active),
            sizes=sizes,
            frequencies=tuple(complex(f) for f in job.f_list),
            space=space,
            J=J,
            d=d,
            m=m,
        )
        self.linearizations[state_fp] = lin
        for task in range(1, job.n_tasks + 1):
            frequency = job.f_list[task - 1]
            rows = lin.rows(frequency)
            residual = J[rows] @ m - d[rows]
            self._write_state(job, task, lin, frequency)
            if job.objective is not None:
                self._write_report(job, task, lin, frequency, residual)
            if job.covector is not None:
                gradient = np.real(J[rows].conj().T @ residual)
                self._write_covector(job.covector_file(task), lin, gradient)
        if job.state_output is not None:
            ControlStateFile(
                dict(baseline.blocks),
                support={name: np.ones(sizes[name], dtype=bool) for name in active},
                support_min_support=job.min_support,
            ).write(job.state_output_file())
        if job.manifest is not None:
            self._write_manifest(job, lin, baseline)

    def _apply(self, job: FWIOperatorJob) -> None:
        for task in range(1, job.n_tasks + 1):
            lin, frequency = self._load_state(job, task)
            rows = lin.rows(frequency)
            J = lin.J[rows]
            if job.action == "jvp":
                dv = self._direction(job, lin)
                values = np.zeros(lin.space.size, dtype=np.complex128)
                values[rows] = J @ dv
                DataVector(values, lin.space).write_objective_vector(
                    job.objective_vector_file(task),
                    state_fingerprint=lin.state_fingerprint,
                    term_layout=lin.space.term_layouts(frequency=frequency),
                    n_ranks=self.n_ranks,
                )
            elif job.action == "vjp":
                r = self._objective_vector(job, task, lin, frequency)
                self._write_covector(
                    job.covector_file(task), lin, np.real(J.conj().T @ r[rows])
                )
            else:  # normal
                dv = self._direction(job, lin)
                self._write_covector(
                    job.covector_file(task), lin, np.real(J.conj().T @ (J @ dv))
                )
            if job.objective is not None:
                residual = J @ lin.m - lin.d[rows]
                self._write_report(job, task, lin, frequency, residual)

    def _postprocess(self, job: FWIOperatorJob) -> None:
        """Emulate ``--smooth``: weighted sum at the stem, ``_raw`` beside it."""

        if not job.requires_postprocess():
            raise ValueError("postprocess requires a smoothing configuration")
        weights = (
            np.ones(job.n_tasks) if job.weights is None else np.asarray(job.weights)
        )
        total: Optional[ControlVectorFile] = None
        blocks: Dict[str, np.ndarray] = {}
        for task in range(1, job.n_tasks + 1):
            path = job.covector_file(task)
            if not path.is_file():
                raise FileNotFoundError(f"missing covector part {path}")
            part = ControlVectorFile.read(path, native=False)
            if total is None:
                total = part
                blocks = {k: weights[task - 1] * v for k, v in part.blocks.items()}
            else:
                for name, values in part.blocks.items():
                    blocks[name] = blocks[name] + weights[task - 1] * values
        assert total is not None
        aggregate = ControlVectorFile(
            blocks,
            state_fingerprint=total.state_fingerprint,
            control_registry_fingerprint=total.control_registry_fingerprint,
            support={name: np.ones(v.size, dtype=bool) for name, v in blocks.items()},
        )
        aggregate.write(job.covector_file(raw=True))
        aggregate.write(job.covector_file())

    # -- inputs ---------------------------------------------------------------

    def _sizes(
        self, active: Sequence[str], sizes: Optional[Mapping[str, int]]
    ) -> Dict[str, int]:
        table = dict(self.block_sizes)
        if sizes is not None:
            table.update(
                {qualified_block_name(k): int(v) for k, v in dict(sizes).items()}
            )
        missing = [
            qualified_block_name(name)
            for name in active
            if qualified_block_name(name) not in table
        ]
        if missing:
            raise ValueError(f"unknown block sizes for {', '.join(missing)}")
        return {
            qualified_block_name(name): table[qualified_block_name(name)]
            for name in active
        }

    def _coefficients(
        self,
        vector: Union[ControlVectorFile, ControlStateFile, np.ndarray],
        active: Optional[Sequence[str]],
        sizes: Optional[Mapping[str, int]],
    ) -> Tuple[List[str], Dict[str, int], np.ndarray]:
        if isinstance(vector, ControlVectorFile):
            names = list(vector.names) if active is None else list(active)
            table = {qualified_block_name(n): int(vector[n].size) for n in names}
            return names, table, vector.pack(names)
        if isinstance(vector, ControlStateFile):
            if active is None:
                raise ValueError("forward with a control state requires active")
            names = list(active)
            table = {qualified_block_name(n): int(vector[n].size) for n in names}
            return names, table, np.concatenate([vector[n] for n in names])
        if active is None:
            raise ValueError("forward with a packed array requires active")
        names = list(active)
        table = self._sizes(names, sizes)
        m = np.asarray(vector, dtype=np.float64).reshape(-1)
        if m.size != sum(table.values()):
            raise ValueError("packed vector size does not match the active blocks")
        return names, table, m

    def _baseline(self, job: FWIOperatorJob) -> ControlStateFile:
        active = [qualified_block_name(name) for name in job.active or ()]
        if job.control_state is not None:
            state = ControlStateFile.read(job.control_state)
            missing = [name for name in active if name not in state.names]
            if missing:
                raise ValueError(
                    f"controls.state lacks active block(s) {', '.join(missing)}"
                )
            for name in active:
                expected = self.block_sizes.get(name)
                if expected is not None and expected != state[name].size:
                    raise ValueError(
                        f"controls.state block {name!r} has {state[name].size} "
                        f"DOFs; the fake site expects {expected}"
                    )
            return state
        names = list(self.block_sizes)
        for name in active:
            if name not in self.block_sizes:
                raise ValueError(
                    f"no controls.state and no block size for {name!r}; pass "
                    "block_sizes to FakeImagingSite"
                )
        return ControlStateFile(
            {name: np.zeros(self.block_sizes[name]) for name in names}
        )

    def _direction(self, job: FWIOperatorJob, lin: FakeLinearization) -> np.ndarray:
        if job.direction is None:
            raise ValueError(f"action {job.action!r} requires a direction")
        vector = ControlVectorFile.read(job.direction, native=False)
        self._check_binding(vector, lin, job.direction)
        dv = vector.pack(lin.active)
        if dv.size != lin.J.shape[1]:
            raise ValueError("direction size does not match the active subspace")
        return dv

    def _objective_vector(
        self,
        job: FWIOperatorJob,
        task: int,
        lin: FakeLinearization,
        frequency: Any,
    ) -> np.ndarray:
        if job.objective_vector is None:
            raise ValueError("vjp requires an objective_vector input")
        path = job.objective_vector_file(task)
        if not path.is_file():
            path = job.objective_vector_file()
        return DataVector.read_objective_vector(
            path,
            lin.space,
            frequency=frequency,
            state_fingerprint=lin.state_fingerprint,
        ).values

    @staticmethod
    def _check_binding(
        vector: ControlVectorFile, lin: FakeLinearization, path: Path
    ) -> None:
        if vector.state_fingerprint != lin.state_fingerprint:
            raise ValueError(f"{path} belongs to another objective state")
        if vector.control_registry_fingerprint != lin.control_registry_fingerprint:
            raise ValueError(f"{path} belongs to another control registry")

    # -- outputs --------------------------------------------------------------

    def _write_state(
        self,
        job: FWIOperatorJob,
        task: int,
        lin: FakeLinearization,
        frequency: Any,
    ) -> None:
        path = job.state_file(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = complex(frequency)
        payload = {
            "schema": FAKE_STATE_SCHEMA,
            "state_fingerprint": lin.state_fingerprint,
            "control_registry_fingerprint": lin.control_registry_fingerprint,
            "task": task,
            "frequency": [f.real, f.imag],
            "frequencies": [
                [complex(v).real, complex(v).imag] for v in lin.frequencies
            ],
            "active": list(lin.active),
            "sizes": dict(lin.sizes),
            "m": lin.m.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _load_state(
        self, job: FWIOperatorJob, task: int
    ) -> Tuple[FakeLinearization, complex]:
        path = job.state_file(task)
        if not path.is_file():
            raise FileNotFoundError(f"missing objective state {path}")
        data = json.loads(path.read_text())
        if data.get("schema") != FAKE_STATE_SCHEMA:
            raise ValueError(f"{path} is not a fake imaging state")
        frequency = complex(*data["frequency"])
        expected = complex(job.f_list[task - 1])
        if not np.isclose(frequency, expected):
            raise ValueError(
                f"task {task} frequency {expected} differs from the state's {frequency}"
            )
        if list(job.active or []) != list(data["active"]):
            raise ValueError("job active blocks differ from the state's")
        state_fp = str(data["state_fingerprint"])
        lin = self.linearizations.get(state_fp)
        if lin is None:
            frequencies = [complex(*pair) for pair in data["frequencies"]]
            J, d, space = self.surrogate(
                job.simulation, data["active"], data["sizes"], frequencies
            )
            lin = FakeLinearization(
                state_fingerprint=state_fp,
                control_registry_fingerprint=str(data["control_registry_fingerprint"]),
                active=tuple(data["active"]),
                sizes={str(k): int(v) for k, v in data["sizes"].items()},
                frequencies=tuple(frequencies),
                space=space,
                J=J,
                d=d,
                m=np.asarray(data["m"], dtype=np.float64),
            )
            self.linearizations[state_fp] = lin
        return lin, frequency

    def _write_report(
        self,
        job: FWIOperatorJob,
        task: int,
        lin: FakeLinearization,
        frequency: Any,
        residual: np.ndarray,
    ) -> None:
        rows = lin.rows(frequency)
        position = {int(row): index for index, row in enumerate(rows)}
        terms = []
        for layout in lin.space.term_layouts(frequency=frequency):
            local = residual[[position[int(i)] for i in layout.indices]]
            raw = float(np.vdot(local, local).real)
            terms.append(
                {
                    "id": layout.id,
                    "raw_sum": raw,
                    "effective_weight_mass": float(local.size),
                    "normalized_value": 0.5 * raw,
                    "weight": 1.0,
                    "weighted_value": 0.5 * raw,
                    "active_samples": int(local.size),
                    "scale": [1.0],
                }
            )
        payload = {
            "schema": "fs-objective-report-1",
            "state_fingerprint": lin.state_fingerprint,
            "total": float(sum(term["weighted_value"] for term in terms)),
            "terms": terms,
            "runtime": {"site": type(self).__name__, "task": task},
        }
        path = job.report_file(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _write_covector(path: Path, lin: FakeLinearization, packed: np.ndarray) -> None:
        vector = ControlVectorFile.from_packed(
            packed,
            lin.sizes,
            state_fingerprint=lin.state_fingerprint,
            control_registry_fingerprint=lin.control_registry_fingerprint,
        )
        vector.support = {
            name: np.ones(size, dtype=bool) for name, size in lin.sizes.items()
        }
        vector.write(path)

    def _write_manifest(
        self, job: FWIOperatorJob, lin: FakeLinearization, baseline: ControlStateFile
    ) -> None:
        blocks = []
        offset = 1
        active_offsets = [0] * len(baseline.names)
        active_ids = []
        active_offset = 1
        for block_id, name in enumerate(baseline.names, start=1):
            size = int(baseline[name].size)
            blocks.append(
                {
                    "id": block_id,
                    "name": name,
                    "binding": [1, block_id, 1],
                    "layout": [offset, size, 1, 1],
                    "units": "1",
                    "actions": 3,
                    "transform": 1,
                    "scaling": [1.0, 0.0, 1.0, 0.0],
                    "basis_identity": f"fake:{name}",
                    "distributed": False,
                }
            )
            offset += size
        for name in lin.active:
            block_id = list(baseline.names).index(name) + 1
            active_ids.append(block_id)
            active_offsets[block_id - 1] = active_offset
            active_offset += lin.sizes[name]
        values = np.concatenate([baseline[name] for name in baseline.names])
        payload = {
            "schema": "fs-control-registry-1",
            "fingerprint": lin.control_registry_fingerprint,
            "packing": "real_interleaved",
            "pairing": "real_euclidean",
            "coordinates": [0.0] * int(values.size),
            "values": values.tolist(),
            "active_blocks": active_ids,
            "active_offsets": active_offsets,
            "blocks": blocks,
            "descriptor_rank": 0,
            "n_ranks": self.n_ranks,
            "rank_descriptors": [],
        }
        path = job.manifest_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
