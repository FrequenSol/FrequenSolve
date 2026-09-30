# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Frozen patch execution and assembly on parent control/observation identities."""

from __future__ import annotations

import json
import shutil
import weakref
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast
from uuid import uuid4

import numpy as np

from frequensolve.mesh._stage_mesh import PatchStageMesh
from frequensolve.mesh._stage_snapshot import PatchStageSnapshot, _contained, _digest
from frequensolve.mesh.patches import PreparedPatchSet
from frequensolve.simulation.simulation import BaseSimulation

from ._artifacts import ControlRegistryManifest, ControlStateFile
from ._backend import LinearizationEntry, fingerprint, read_report, reduce_covectors
from ._objective import ObjectiveState, _ObjectiveSpace
from ._patch_objective import patch_objective_keys, restrict_patch_misfit
from .controls import ControlSpace, ControlState, ControlVector
from .data import DataSpace, DataVector, _DataSegment
from .jobs import FWIOperatorJob
from .operators import Jacobian, Normal
from .problem import ImagingProblem, Linearization


def _remove_candidate_artifacts(root: Path, paths: Iterable[Path]) -> None:
    """Remove only candidate-owned directories after their last operator is released."""
    for path in paths:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved != root:
            shutil.rmtree(resolved, ignore_errors=True)


class _CandidateArtifacts:
    def __init__(self, root: Path, directory: Path) -> None:
        self.paths = [directory]
        self.root = root.resolve()

    def activate(self) -> None:
        # Failed native evaluations retain their inputs and logs for diagnosis.
        weakref.finalize(self, _remove_candidate_artifacts, self.root, self.paths)

    def track(self, job: FWIOperatorJob) -> None:
        self.paths.append(Path(job._result_path).parent)


class _ChildProblem(ImagingProblem):
    _patch_misfit: dict[str, Any]
    _patch_stage: PatchStageSnapshot
    _patch_mesh: PatchStageMesh
    _patch_artifacts: _CandidateArtifacts
    """Action context whose stage definitions stay fixed across candidate states."""

    def _sync_simulation(self, state: ControlState | None) -> None:
        pass  # Candidates are passed exclusively through controls.state.

    def _operator_job(
        self,
        space: ControlSpace,
        action: str,
        *,
        frequencies: Sequence[Any],
        **options: Any,
    ) -> FWIOperatorJob:
        job = FWIOperatorJob(
            self.backend.job_name(action),
            self.simulation,
            frequencies,
            action=action,
            active=list(space.blocks),
            misfit=self._patch_misfit,
            pml_stage=self._patch_stage,
            stage_mesh=self._patch_mesh,
            gram_derivative="total",
            source_controls=self._source_controls(space),
            reflectivity=self._reflectivity(space),
            **options,
        )
        self._patch_artifacts.track(job)
        return job


class _PatchRuntime:
    def __init__(self, problem: ImagingProblem) -> None:
        self.cache: dict[str, CompositeLinearization] = {}
        self.problem = problem
        assert problem.patches is not None
        self.policy = deepcopy(problem.patches.to_dict())
        parent = problem.restrict(patches=None, support="refresh")
        # The full parent establishes the authoritative registry and objective
        # normalization. Geometry-only prepare_patches never calls this path.
        baseline = parent.linearize(gradient=False)
        self.parent = copy(parent)
        self.parent._shared = copy(parent._shared)
        self.parent._shared.simulation = deepcopy(parent.simulation)
        self.state = baseline.state
        assert problem.patches is not None
        assert problem._shared.manifest is not None
        prepared = problem.prepare_patches()
        self.prepared = prepared
        directory = (
            problem.workdir
            / "patch_stages"
            / (problem.backend.job_name("stage") + "_" + uuid4().hex[:12])
        )
        self.stage = prepared.freeze_stage(
            problem._complete_state_file(self.state),
            name=problem.name,
            directory=directory,
        )
        pinned = BaseSimulation.load(self.stage.simulation_file, project_path=directory)
        self.parent._shared.simulation = deepcopy(pinned)
        self.children: list[tuple[int, Any, _ChildProblem, FWIOperatorJob | None]] = []
        jobs = []
        for index, simulation in enumerate(
            prepared.simulations(name=problem.name + "_patch")
        ):
            simulation.model = deepcopy(pinned.model)
            simulation.project_path = directory
            simulation._file = None
            simulation.mesh.file = str(directory / "parent.gmp")
            for task, frequency in enumerate(problem.frequencies, 1):
                context = cast(
                    _ChildProblem,
                    problem.restrict(frequencies=[frequency], patches=None),
                )
                context.__class__ = _ChildProblem
                context._shared = copy(problem._shared)
                context._shared.simulation = deepcopy(simulation)
                # Native acquisition caches live beside Mesh/file. Separate
                # immutable copies prevent sibling jobs from replacing a cache
                # while another patch or frequency is reading it.
                mesh_directory = directory / "contexts" / f"{index:04d}" / f"{task:04d}"
                mesh_directory.mkdir(parents=True)
                mesh_file = mesh_directory / "parent.gmp"
                shutil.copyfile(directory / "parent.gmp", mesh_file)
                context.simulation.mesh.file = str(mesh_file)
                context.simulation.name = f"{simulation.name}_f{task:04d}"
                context._masks = {}
                context._masks_adopted = True
                context._patch_stage = self.stage
                context._patch_misfit = restrict_patch_misfit(
                    parent._misfit_payload.to_fs(),
                    baseline.job.state_file(task),
                    receiver_groups=[
                        g.name for g in simulation.acquisition.receiver_groups
                    ],
                )
                capture = FWIOperatorJob(
                    problem.backend.job_name("patch_capture"),
                    context.simulation,
                    [frequency],
                    action="linearize",
                    active=[],
                    state="state.json",
                    misfit=context._patch_misfit,
                    control_state=self.stage.control_state,
                    pml_stage=self.stage,
                    stage_mesh="capture",
                )
                self.children.append((index, frequency, context, capture))
                jobs.append(capture)
        problem.backend.run_many(jobs)
        for ordinal, (index, frequency, context, captured_job) in enumerate(
            self.children
        ):
            assert captured_job is not None
            problem._check_tasks(captured_job)
            manifests = list(
                captured_job._result_path.glob(
                    "_fs_run/tasks/*/stage_mesh/manifest.json"
                )
            )
            if len(manifests) != 1:
                raise ValueError(
                    "Patch capture must publish exactly one frequency mesh"
                )
            context._patch_mesh = self.stage.publish_mesh(
                manifests[0],
                directory / "meshes" / f"{ordinal:04d}",
            )

    def checkpoint(self) -> dict[str, Any]:
        """Describe verified immutable stage inputs, meshes and child definitions."""
        self.stage.verify()
        root = self.stage.manifest.parent
        records = []
        for index, frequency, context, _ in self.children:
            context._patch_mesh.verify(stage=self.stage)
            simulation = Path(context.simulation._file).resolve(strict=True)
            records.append(
                {
                    "patch": index,
                    "frequency": [complex(frequency).real, complex(frequency).imag],
                    "simulation": simulation.relative_to(root).as_posix(),
                    "simulation_sha256": _digest(simulation),
                    "misfit": context._patch_misfit,
                    "mesh": context._patch_mesh.manifest.relative_to(root).as_posix(),
                    "mesh_identity": context._patch_mesh.identity,
                }
            )
        return {
            "schema": "fs-patch-runtime-1",
            "stage": self.stage.to_fs(),
            "policy": self.policy,
            "registry": cast(
                ControlRegistryManifest, self.problem._shared.manifest
            ).raw,
            "children": records,
        }

    @classmethod
    def restore(
        cls, problem: ImagingProblem, record: Mapping[str, Any]
    ) -> _PatchRuntime:
        """Reopen the exact interrupted stage; never recapture at the accepted model."""
        if record.get("schema") != "fs-patch-runtime-1":
            raise ValueError("Unsupported patch runtime checkpoint")
        if problem.patches is None or record["policy"] != problem.patches.to_dict():
            raise ValueError("Checkpoint patch selection policy changed")
        runtime = object.__new__(cls)
        runtime.cache = {}
        runtime.problem = problem
        runtime.policy = deepcopy(record["policy"])
        runtime.stage = PatchStageSnapshot.read(**record["stage"])
        root = runtime.stage.manifest.parent
        registry = ControlRegistryManifest.from_dict(record["registry"])
        baseline = ControlStateFile.read(runtime.stage.control_state)
        problem._shared.adopt_baseline(baseline, registry)
        runtime.state = ControlState.from_file(
            baseline, problem.full_space.without_support()
        )
        pinned = BaseSimulation.load(runtime.stage.simulation_file, project_path=root)
        runtime.parent = problem.restrict(patches=None)
        runtime.parent._shared = copy(problem._shared)
        runtime.parent._shared.simulation = deepcopy(pinned)
        runtime.prepared = PreparedPatchSet(
            json.loads((root / "geometry.json").read_text()),
            json.loads((root / "acquisition.json").read_text()),
            (),
            problem.patches.pml,
        )
        runtime.children = []
        seen = set()
        for item in record["children"]:
            frequency = complex(*item["frequency"])
            frequency = frequency.real if not frequency.imag else frequency
            identity = (item["patch"], frequency)
            if identity in seen or frequency not in problem.frequencies:
                raise ValueError(
                    "Checkpoint has duplicate or unknown patch/frequency jobs"
                )
            seen.add(identity)
            simulation = _contained(root, item["simulation"])
            if _digest(simulation) != item["simulation_sha256"]:
                raise ValueError("Checkpoint patch simulation changed")
            context = cast(
                _ChildProblem, problem.restrict(frequencies=[frequency], patches=None)
            )
            context.__class__ = _ChildProblem
            context._shared = copy(problem._shared)
            context._shared.simulation = BaseSimulation.load(
                simulation, project_path=root
            )
            context._masks = {}
            context._masks_adopted = True
            context._patch_stage = runtime.stage
            context._patch_misfit = item["misfit"]
            context._patch_mesh = PatchStageMesh.read(
                _contained(root, item["mesh"]),
                identity=item["mesh_identity"],
                stage=runtime.stage,
            )
            runtime.children.append((item["patch"], frequency, context, None))
        frequencies = {f for _, f, _, _ in runtime.children}
        expected = {
            (index, f)
            for index in range(len(runtime.prepared.geometry["patches"]))
            for f in frequencies
        }
        if seen != expected or not seen or frequencies != set(problem.frequencies):
            raise ValueError("Checkpoint omits a patch/frequency mesh")
        return runtime

    def linearize(
        self, problem: ImagingProblem, state: ControlState | None, *, gradient: bool
    ) -> CompositeLinearization:
        self.stage.verify()
        assert problem._shared.manifest is not None
        if state is None:
            state = problem._require_state()
        key = fingerprint(
            problem=problem.identity(), state=state.values, stage=self.stage.identity
        )
        cached = self.cache.get(key)
        if cached is not None and (not gradient or cached.gradient is not None):
            # Refresh LRU order without discarding artifacts held by a caller.
            self.cache.pop(key)
            self.cache[key] = cached
            problem._adopt_masks(cached.support_masks)
            return cached
        directory = self.stage.manifest.parent / "candidates" / uuid4().hex
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = _CandidateArtifacts(self.stage.manifest.parent, directory)
        control_state = problem._complete_state_file(state).write(
            directory / "candidate.h5"
        )
        pending = []
        for index, frequency, original, _ in self.children:
            if frequency not in problem.frequencies or (
                problem._patch_selection is not None
                and index not in problem._patch_selection
            ):
                continue
            context = copy(original)
            context._patch_artifacts = artifacts
            context._active = problem._active
            context._overrides = dict(problem._overrides)
            context._frequencies = (frequency,)
            context._overrides["weights"] = (
                (
                    1.0
                    if problem.weights is None
                    else problem.weights[problem.frequencies.index(frequency)]
                ),
            )
            space = context.space
            job = FWIOperatorJob(
                problem.backend.job_name("patch_linearize"),
                context.simulation,
                [frequency],
                action="linearize",
                active=list(space.blocks),
                state="state.json",
                objective="report.json",
                covector="gradient.h5" if gradient else None,
                state_output="support.h5",
                manifest="registry.json" if not gradient else None,
                control_state=control_state,
                misfit=context._patch_misfit,
                pml_stage=self.stage,
                stage_mesh=context._patch_mesh,
                gram_derivative="total",
                min_support=problem.min_support,
                source_controls=context._source_controls(space),
                reflectivity=context._reflectivity(space),
            )
            artifacts.track(job)
            pending.append((index, context, job))
        if not pending:
            raise ValueError("Patch stage has no meshes for the requested frequencies")
        problem.backend.run_many([job for _, _, job in pending])
        children: list[Linearization] = []
        for index, context, job in pending:
            problem._check_tasks(job)
            reports = read_report(job)
            masks = context._read_masks(job, context.space) or {}
            factors = context._task_factors(job, context.space)
            part = reduce_covectors(job, context.weights, factors) if gradient else None
            report_fp = reports[0].state_fingerprint
            entry = LinearizationEntry(
                fingerprint=fingerprint(parent=key, patch=index, frequency=job.f_list),
                job=job,
                directory=directory / f"child_{len(children):04d}",
                state_fingerprint=report_fp,
                control_registry_fingerprint=(
                    part.control_registry_fingerprint
                    if part
                    else problem._shared.manifest.fingerprint
                ),
            )
            children.append(
                Linearization(
                    context,
                    space=context.space,
                    state=state,
                    entry=entry,
                    manifest=problem._shared.manifest,
                    reports=reports,
                    gradient_file=part,
                    support_masks=masks,
                    task_factors=factors,
                )
            )
        composite = CompositeLinearization(problem, state, key, children)
        composite._patch_artifacts = artifacts
        artifacts.activate()
        self.cache[key] = composite
        # Retain the same bounded number of candidate points as ordinary imaging.
        while len(self.cache) > problem.cache.capacity:
            self.cache.pop(next(iter(self.cache)))
        return composite


def linearize_patches(
    problem: ImagingProblem,
    state: ControlState | None,
    *,
    gradient: bool,
    receiver_diagonal: Mapping[str, Any] | None,
) -> CompositeLinearization:
    if receiver_diagonal is not None:
        raise ValueError("Patch receiver diagonal probes are not implemented")
    if problem.kernel_derivative is not None:
        raise ValueError("Patch objectives require ordinary waveform data")
    from ._patch_objective import validate_patch_misfit

    validate_patch_misfit(problem._misfit_payload.to_fs())
    if any(
        block.kind == "interface"
        or (block.kind == "source" and block.quantity == "position")
        for block in problem.space.resolved_blocks
    ):
        raise ValueError("Patch stages require fixed geometry and source positions")
    assert problem.patches is not None
    if (
        problem._patch_runtime is not None
        and problem._patch_runtime.policy != problem.patches.to_dict()
    ):
        problem._patch_runtime = None
        problem._prepared_patches = None
        problem._masks = {}
        problem._masks_adopted = False
        problem._data_space = None
    if problem._patch_runtime is None:
        problem._patch_runtime = _PatchRuntime(problem)
    runtime = problem._patch_runtime
    assert runtime is not None
    return runtime.linearize(problem, state, gradient=gradient)


class CompositeLinearization(Linearization):
    _patch_artifacts: _CandidateArtifacts
    """Sum patch derivatives; expose each original observation exactly once."""

    def __init__(
        self,
        problem: ImagingProblem,
        state: ControlState,
        key: str,
        children: Sequence[Linearization],
    ) -> None:
        self.problem = problem
        self.state = state
        self.fingerprint = key
        self.children = tuple(children)
        self.frequencies = problem.frequencies
        manifest = problem._shared.manifest
        assert manifest is not None
        self.manifest = manifest
        self.registry_fingerprint = self.manifest.fingerprint
        self.state_fingerprint = fingerprint(state=state.values)
        self.reports = [r for child in children for r in child.reports]
        self.report = {}
        self.value = sum(child.value for child in children)
        masks: dict[str, np.ndarray] = {}
        for child in children:
            for name, sl in child.space.full_slices.items():
                mask = child.support_masks.get(name, np.ones(sl.stop - sl.start, bool))
                masks[name] = masks.get(name, np.zeros_like(mask)) | mask
        self.support_masks = masks
        problem._adopt_masks(masks)
        self.space = problem.space
        self.point = state.vector(self.space)
        self.gradient = (
            None
            if any(c.gradient is None for c in children)
            else self._sum([cast(ControlVector, c.gradient) for c in children])
        )
        for child in children:
            for name, value in child.report.items():
                self.report[name] = self.report.get(name, 0.0) + value
        self._maps: list[tuple[np.ndarray, np.ndarray]] = []
        keys_by_frequency: dict[Any, dict[str, set[tuple[int, ...]]]] = {
            f: {} for f in self.frequencies
        }
        child_keys = []
        for child in children:
            keys = patch_objective_keys(
                child.job.state_file(1), child.problem.simulation.acquisition.to_fs()
            )
            child_keys.append(keys)
            tables = keys_by_frequency[child.frequencies[0]]
            for name, rows in keys.items():
                table = tables.setdefault(name, set())
                tuples = [tuple(row) for row in rows]
                if table.intersection(tuples):
                    raise ValueError(
                        "Patch objectives duplicate an original observation"
                    )
                table.update(tuples)
        names = sorted(
            {name for tables in keys_by_frequency.values() for name in tables}
        )
        canonical = [
            {
                name: np.asarray(sorted(tables.get(name, set())), dtype=int).reshape(
                    -1, 3
                )
                for name in names
            }
            for tables in keys_by_frequency.values()
        ]
        if any(
            any(not np.array_equal(table[name], canonical[0][name]) for name in names)
            for table in canonical[1:]
        ):
            raise ValueError(
                "Patch observation selection must agree across frequencies"
            )
        segments = [
            _DataSegment(
                name, ("objective",), (1,), tuple(range(1, len(canonical[0][name]) + 1))
            )
            for name in names
        ]
        self._data_space = _ObjectiveSpace(self.frequencies, segments)
        self._data_space._dense = set()
        self._data_space._keys = canonical
        for child, keys in zip(children, child_keys):
            local: list[int] = []
            global_: list[int] = []
            for layout in child.data_space.term_layouts():
                target = self.data_space.term_layout(
                    layout.id, frequency=child.frequencies[0]
                )
                assert target.indices is not None
                lookup = {
                    tuple(row): index
                    for row, index in zip(target.coordinate_keys, target.indices)
                }
                assert layout.indices is not None
                assert target.indices is not None
                local.extend(layout.indices)
                global_.extend(
                    lookup[tuple(row)] for row in keys[layout.id][layout.row_ids - 1]
                )
            self._maps.append((np.asarray(local, int), np.asarray(global_, int)))
        self._jacobian: Jacobian | None = None
        self._normal: Normal | None = None
        self.frequency_weights = (
            np.ones(len(self.frequencies))
            if problem.weights is None
            else np.asarray(problem.weights, dtype=float)
        )

    def __repr__(self) -> str:
        return f"CompositeLinearization(patches={len(self.children)}, value={self.value:g})"

    @property
    def regularization_job(self) -> FWIOperatorJob:
        runtime = self.problem._patch_runtime
        assert runtime is not None
        parent = runtime.parent
        return parent._linearize_job(self.space, None, gradient=True)

    @property
    def jobs(self) -> tuple[FWIOperatorJob, ...]:
        return tuple(child.job for child in self.children)

    @property
    def job(self) -> FWIOperatorJob:
        if len(self.jobs) != 1:
            raise ValueError("Composite linearization has multiple jobs; use .jobs")
        return self.jobs[0]

    @property
    def data_space(self) -> DataSpace:
        assert self._data_space is not None
        return self._data_space

    @property
    def objective_states(self) -> list[ObjectiveState]:
        """Native states in child-job order; canonical vectors use the row maps."""
        return [state for child in self.children for state in child.objective_states]

    def _sum(self, vectors: Iterable[ControlVector]) -> ControlVector:
        values = sum(
            (v.space.to_sauce_vector(v) for v in vectors),
            np.zeros(self.space.full_size),
        )
        return self.space.from_sauce_vector(values)

    def _assemble(self, vectors: Iterable[DataVector]) -> DataVector:
        values = np.zeros(self.data_space.size, complex)
        for vector, (local, global_) in zip(vectors, self._maps):
            values[global_] = vector.values[local]
        return DataVector(values, self.data_space)

    def _split(self, vector: Any) -> list[DataVector]:
        dual = self._data_vector(vector)
        result = []
        for child, (local, global_) in zip(self.children, self._maps):
            values = np.zeros(child.data_space.size, complex)
            values[local] = dual.values[global_]
            result.append(DataVector(values, child.data_space))
        return result

    def _directions(self, direction: Any) -> list[ControlVector]:
        vector = self._control_vector(direction)
        full = self.space.to_sauce_vector(vector)
        return [child.space.from_sauce_vector(full) for child in self.children]

    def jvp(self, direction: Any) -> DataVector:
        return self._assemble(
            [c.jvp(v) for c, v in zip(self.children, self._directions(direction))]
        )

    def vjp(self, dual: Any) -> ControlVector:
        return self._sum([c.vjp(v) for c, v in zip(self.children, self._split(dual))])

    def vjp_tasks(self, dual: Any) -> list[ControlVector]:
        parts = [c.vjp(v) for c, v in zip(self.children, self._split(dual))]
        return [
            self._sum(
                [
                    v
                    for c, v in zip(self.children, parts)
                    if c.frequencies[0] == frequency
                ]
            )
            for frequency in self.frequencies
        ]

    def apply_normal(self, direction: Any) -> ControlVector:
        return self._sum(
            [
                c.apply_normal(v)
                for c, v in zip(self.children, self._directions(direction))
            ]
        )

    def weight_data(self, dual: Any) -> DataVector:
        return self._assemble(
            [c.weight_data(v) for c, v in zip(self.children, self._split(dual))]
        )

    def objective_residual(self) -> DataVector:
        return self._assemble([c.objective_residual() for c in self.children])

    def simulated(self) -> DataVector:
        return self._assemble([c.simulated() for c in self.children])

    def observed(self) -> DataVector:
        return self._assemble([c.observed() for c in self.children])

    @property
    def residual_sign(self) -> float:
        signs = {c.residual_sign for c in self.children}
        if len(signs) != 1:
            raise ValueError("Patch residual conventions disagree")
        return signs.pop()

    def _with_covector(self) -> Linearization:
        return self if self.gradient is not None else self.problem.linearize(self.state)

    @property
    def jacobian(self) -> Jacobian:
        if self._jacobian is None:
            self._jacobian = Jacobian(self._with_covector())
        return self._jacobian

    @property
    def normal(self) -> Normal:
        if self._normal is None:
            self._normal = Normal(self._with_covector())
        return self._normal


def patch_observed_vector(problem: ImagingProblem) -> DataVector:
    """Select raw observations on the same canonical rows as patch predictions."""
    lin = cast(CompositeLinearization, problem.linearize(gradient=False))
    parent = problem.restrict(patches=None)
    observed = parent.observed_vector()
    groups = {
        name: term["receiver_group"]
        for child in lin.children
        for name, term in child.objective_states[0].terms.items()
    }
    values = np.zeros(lin.data_space.size, complex)
    for frequency in lin.frequencies:
        for target in lin.data_space.term_layouts(frequency=frequency):
            source = observed.space.term_layout(groups[target.id], frequency=frequency)
            assert target.indices is not None
            assert source.indices is not None
            lookup = {
                tuple(row): index
                for row, index in zip(source.coordinate_keys, source.indices)
            }
            values[target.indices] = observed.values[
                [lookup[tuple(row)] for row in target.coordinate_keys]
            ]
    return DataVector(values, lin.data_space)
