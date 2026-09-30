# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Public patch imaging dispatch and canonical composite operator acceptance."""

import numpy as np
import pytest

from frequensolve.imaging import DepthProfile, ImagingProblem, Misfit
from frequensolve.mesh.patches import PatchSet
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.seismic.sources import SourceGeometry
from frequensolve.units import ureg
from tests.test_root_patch_integration import _parent, _solver

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
def test_public_patch_linearization_and_actions(tmp_path, dimension, physics):
    parent = _parent(tmp_path, dimension, physics)
    parent.acquisition.source_geometry = SourceGeometry.points(
        kind="scalar",
        coords=[
            [x, 0.08] if dimension == 2 else [x, 0.2, 0.08] for x in (0.2, 0.8, 0.25)
        ],
    )
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        problem = ImagingProblem(
            parent,
            controls=DepthProfile("vp", "lower", count=2),
            observed=None,
            frequencies=[3],
            site=site,
            misfit=Misfit(normalization=1.0),
            patches=PatchSet.around_sources(
                shots_per_patch=2, max_offset=2000 * ureg.m, padding=0 * ureg.m
            ),
            submit_options={"procs_per_job": 2},
        )
        prepared = problem.prepare_patches()
        assert len(prepared.acquisition) == 2
        assert not any(job.workflow == "fwi_operator" for job in prepared.jobs)
        baseline = problem.linearize()
        assert len(baseline.jobs) == 2
        with pytest.raises(ValueError, match="multiple jobs"):
            _ = baseline.job
        point = baseline.point.values + 0.03
        lin = problem.linearize(point)
        assert problem.linearize(point) is lin
        assert lin.value == pytest.approx(sum(c.value for c in lin.children))
        np.testing.assert_allclose(
            lin.gradient.values, sum(c.gradient.values for c in lin.children)
        )
        keys = lin.data_space.term_layout("surface").coordinate_keys
        assert set(keys[:, 0]) == {1, 2, 3}
        assert len({tuple(row) for row in keys}) == len(keys)
        direction = np.array([0.2, -0.1])
        tangent = lin.jvp(direction)
        rng = np.random.default_rng(29)
        dual = rng.normal(size=tangent.values.size) + 1j * rng.normal(
            size=tangent.values.size
        )
        lhs = np.vdot(tangent.values, dual).real
        rhs = np.dot(direction, lin.vjp(dual).values)
        assert lhs == pytest.approx(rhs, rel=1e-3, abs=1e-9)
        np.testing.assert_allclose(
            lin.apply_normal(direction).values,
            lin.vjp(lin.weight_data(tangent)).values,
            rtol=1e-3,
            atol=1e-9,
        )
        epsilon = 0.05
        fd = (
            problem.value(point + epsilon * direction)
            - problem.value(point - epsilon * direction)
        ) / (2 * epsilon)
        assert fd == pytest.approx(
            np.dot(lin.gradient.values, direction), rel=0.01, abs=1e-9
        )
        problem._patch_runtime.stage.verify()
        assert (
            problem.restrict(frequencies=[3])._patch_runtime is problem._patch_runtime
        )
        assert problem.restrict(support="refresh")._patch_runtime is None


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
def test_cut_mesh_composite_with_multiple_frequencies(tmp_path, dimension, physics):
    from frequensolve.imaging import MeshParameters
    from frequensolve.simulation import FrequencyDomainJob

    parent = _parent(tmp_path, dimension, physics, root_columns=8)
    parent.mesh.set_adapt(elems_per_wave=0.5, order=2, f_low=3, f_high=7.5)
    parent.acquisition.source_geometry = SourceGeometry.points(
        kind="scalar",
        coords=[[x, 0.08] if dimension == 2 else [x, 0.2, 0.08] for x in (0.2, 0.25)],
    )
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob("observed", parent, [3, 7.5])
        assert site.run(observed, check=True, procs_per_job=2).successful
        problem = ImagingProblem(
            parent,
            controls=MeshParameters(
                "vp",
                "lower",
                frequency=1,
                epw=1,
                artifact=str(tmp_path / "material.h5"),
            ),
            observed=observed,
            frequencies=[3, 7.5],
            site=site,
            misfit=Misfit(normalization="observed_rms"),
            patches=PatchSet.around_sources(
                shots_per_patch=1, max_offset=120 * ureg.m, padding=0 * ureg.m
            ),
            submit_options={"procs_per_job": 2},
        ).restrict(weights=[0.5, 2.0])
        baseline = problem.linearize()
        prepared = problem._patch_runtime.prepared
        assert all(
            p["cut_boundary"] > 0 and p["root_fraction"] < 1
            for p in prepared.geometry["patches"]
        )
        assert len(baseline.jobs) == 4
        point = baseline.point.values + 0.15
        lin = problem.linearize(point)
        direction = lin.space.random(29)
        tangent = lin.jvp(direction)
        dual = lin.data_space.random(30)
        lhs = np.vdot(tangent.values, dual.values).real
        rhs = direction.values @ lin.vjp(dual).values
        assert lhs == pytest.approx(rhs, rel=1e-3, abs=1e-8)
        normal = lin.apply_normal(direction)
        np.testing.assert_allclose(
            normal.values,
            lin.vjp(lin.weight_data(tangent)).values,
            rtol=1e-3,
            atol=1e-8,
        )
        parts = lin.vjp_tasks(dual)
        assert len(parts) == 2
        np.testing.assert_allclose(sum(v.values for v in parts), lin.vjp(dual).values)
        raw = problem.forward(point)
        assert raw.space == lin.data_space
        observed_vector = problem.observed_vector()
        assert observed_vector.space == raw.space
        assert np.linalg.norm(observed_vector.values) > 0
        subset = problem.restrict(frequencies=[7.5])
        selected = subset.linearize(point)
        assert len(selected.jobs) == 2
        assert selected.value == pytest.approx(
            sum(c.value for c in lin.children if c.frequencies == [7.5])
        )
        for child in lin.children:
            assert child.space.full_size == problem.full_space.full_size
        if dimension == 2 and physics == "acoustic":
            from frequensolve.imaging import FWI, LBFGS, Stage, Tikhonov

            problem.state = lin.state
            result = FWI(
                problem,
                [Stage([3, 7.5], 1)],
                optimizer=LBFGS(
                    gradient_tolerance=0, objective_tolerance=0, step_tolerance=0
                ),
                regularization=Tikhonov(0.0001),
                step_limit=0.05,
            ).run(resume=False)
            assert result.stages[0].success
            assert (
                result.stages[0].final_loss.total <= result.stages[0].initial_loss.total
            )


def test_global_patch_checkpoint_reopens_the_frozen_stage(tmp_path):
    import json

    from frequensolve.imaging import FWI, LBFGS, Stage
    from frequensolve.inversion import OptimizationCheckpoint

    parent = _parent(tmp_path, 2, "acoustic")
    parent.acquisition.source_geometry = SourceGeometry.points(
        kind="scalar", coords=[[0.2, 0.08], [0.25, 0.08]]
    )
    policy = PatchSet.around_sources(
        shots_per_patch=1, max_offset=2000 * ureg.m, padding=0 * ureg.m
    )
    with LocalSite(
        solver=_solver(2),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:

        def make():
            return ImagingProblem(
                parent,
                controls=DepthProfile("vp", "lower", count=2),
                observed=None,
                frequencies=[3],
                site=site,
                patches=policy,
                misfit=Misfit(normalization=1.0),
            )

        checkpoint = tmp_path / "fwi.h5"
        options = dict(
            optimizer=LBFGS(
                gradient_tolerance=0, objective_tolerance=0, step_tolerance=0
            ),
            checkpoint=checkpoint,
            history=tmp_path / "history.json",
            step_limit=0.05,
        )

        class Interrupted(RuntimeError):
            pass

        def interrupt(event):
            if event.stage_iteration == 1:
                raise Interrupted()

        with pytest.raises(Interrupted):
            FWI(make(), [Stage([3], 2)], callback=interrupt, **options).run(
                resume=False
            )
        saved = OptimizationCheckpoint.load(checkpoint)
        record = json.loads(saved.metadata["patch_runtime"])
        identity = record["stage"]["identity"]
        assert saved.metadata["model_epoch"] in saved.metadata["state_path"]
        with pytest.raises(ValueError, match="stage objective or settings changed"):
            FWI(make(), [Stage([3], 2, weights=[0.5])], **options).run(resume=True)
        resumed_models = []

        def record_resumed(event):
            if event.diagnostics.iteration > 0:
                resumed_models.append(event.model.values.copy())

        workflow = FWI(make(), [Stage([3], 2)], callback=record_resumed, **options)
        result = workflow.run(resume=True)
        assert result.stages[0].resumed
        runtime = workflow._views["stage_01"]._patch_runtime
        assert runtime.stage.identity == identity
        assert all(capture is None for _, _, _, capture in runtime.children)
        again = FWI(make(), [Stage([3], 2)], **options).run(resume=True)
        np.testing.assert_array_equal(again.state.values, result.state.values)
        full_models = []

        def record_full(event):
            if event.diagnostics.iteration > 0:
                full_models.append(event.model.values.copy())

        full_options = dict(
            options,
            checkpoint=tmp_path / "full.h5",
            history=tmp_path / "full_history.json",
        )
        full = FWI(make(), [Stage([3], 2)], callback=record_full, **full_options).run(
            resume=False
        )
        assert len(full_models) == 2 and len(resumed_models) == 1
        # Fresh FP32 native assemblies can differ in their final reduction bits.
        # Optimizer-only restart tests separately require exact iterates; this
        # comparison uses the same native vector tolerance as local resume.
        for resumed, uninterrupted in zip(
            [saved.model, resumed_models[0]], full_models
        ):
            assert np.linalg.norm(resumed - uninterrupted) <= 1e-3 * max(
                np.linalg.norm(uninterrupted), 1e-12
            )
        assert result.stages[0].metrics["summed_worker_seconds"] > 0
        assert result.stages[0].metrics["unmeasured_native_runs"] == 0
        np.testing.assert_allclose(
            full.state.values, result.state.values, rtol=1e-3, atol=1e-9
        )


@pytest.mark.parametrize("mode", ["local_serial", "local_parallel"])
def test_local_patch_checkpoint_resumes_accepted_sequence(tmp_path, mode):
    import json

    from frequensolve.imaging import FWI, LBFGS, PatchUpdates, Stage, Tikhonov
    from frequensolve.inversion import OptimizationCheckpoint

    parent = _parent(tmp_path, 2, "acoustic")
    parent.acquisition.source_geometry = SourceGeometry.points(
        kind="scalar", coords=[[0.2, 0.08], [0.25, 0.08]]
    )
    with LocalSite(
        solver=_solver(2),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:

        def make():
            return ImagingProblem(
                parent,
                controls=DepthProfile("vp", "lower", count=2),
                observed=None,
                frequencies=[3],
                site=site,
                patches=PatchSet.around_sources(
                    shots_per_patch=1, max_offset=2000 * ureg.m, padding=0 * ureg.m
                ),
                misfit=Misfit(normalization=1.0),
            )

        options = dict(
            optimizer=LBFGS(
                gradient_tolerance=0, objective_tolerance=0, step_tolerance=0
            ),
            regularization=Tikhonov(0.0001),
            step_limit=0.05,
            patch_updates=PatchUpdates(mode, local_steps=2),
            checkpoint=tmp_path / "resume.h5",
            history=tmp_path / "resume.json",
        )

        class Interrupted(RuntimeError):
            pass

        interrupted_models = []

        def interrupt(event):
            interrupted_models.append(event.model.values.copy())
            raise Interrupted()

        with pytest.raises(Interrupted):
            FWI(make(), [Stage([3], 1)], callback=interrupt, **options).run(
                resume=False
            )
        checkpoint = OptimizationCheckpoint.load(options["checkpoint"])
        pending = json.loads(checkpoint.metadata["local_updates"])
        assert pending["next_patch"] == 0
        assert pending["proposals"]["0"]["iteration"] == 1
        stage_identity = json.loads(checkpoint.metadata["patch_runtime"])["stage"][
            "identity"
        ]
        resumed_models = []
        workflow = FWI(
            make(),
            [Stage([3], 1)],
            callback=lambda event: resumed_models.append(event.model.values.copy()),
            **options,
        )
        resumed = workflow.run(resume=True)
        assert (
            workflow._views["stage_01"]._patch_runtime.stage.identity == stage_identity
        )
        full_models = []
        full = FWI(
            make(),
            [Stage([3], 1)],
            callback=lambda event: full_models.append(event.model.values.copy()),
            **dict(
                options, checkpoint=tmp_path / "full.h5", history=tmp_path / "full.json"
            ),
        ).run(resume=False)
        assert len(full_models) == 4 and len(resumed_models) == 3

        # Independently prepared FP32 solves can perturb the first gradient by
        # roundoff; L-BFGS curvature amplifies that in later coefficients. The
        # optimizer-only restart test separately requires exact iterates.
        def assert_native_sequence(actual, expected):
            assert np.linalg.norm(actual - expected) <= 1e-3 * np.linalg.norm(expected)

        assert_native_sequence(interrupted_models[0], full_models[0])
        for actual, expected in zip(resumed_models, full_models[1:]):
            assert_native_sequence(actual, expected)
        assert_native_sequence(resumed.state.values, full.state.values)
        assert resumed.stages[0].metrics["combined_checks"] == 1


@pytest.mark.parametrize("mode", ["local_serial", "local_parallel"])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
def test_cut_mesh_local_updates_only_touch_parent_core_coefficients(
    tmp_path, mode, dimension, physics
):
    from frequensolve.imaging import (
        FWI,
        LBFGS,
        MeshParameters,
        PatchUpdates,
        Stage,
        Tikhonov,
    )
    from frequensolve.imaging._patch_masks import core_masks

    parent = _parent(
        tmp_path, dimension, physics, root_columns=8, curved=dimension == 3
    )
    parent.acquisition.source_geometry = SourceGeometry.points(
        kind="scalar",
        coords=[[x, 0.08] if dimension == 2 else [x, 0.2, 0.08] for x in (0.2, 0.25)],
    )
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        problem = ImagingProblem(
            parent,
            controls=MeshParameters(
                "vp",
                "lower",
                frequency=1,
                epw=1,
                artifact=str(tmp_path / "material.h5"),
            ),
            observed=None,
            frequencies=[3],
            site=site,
            misfit=Misfit(normalization=1.0),
            patches=PatchSet.around_sources(
                shots_per_patch=1, max_offset=120 * ureg.m, padding=0 * ureg.m
            ),
            submit_options={"procs_per_job": 2},
        )
        baseline = problem.linearize()
        original = baseline.state
        accepted = []
        workflow = FWI(
            problem,
            [Stage([3], 1)],
            optimizer=LBFGS(
                gradient_tolerance=0, objective_tolerance=0, step_tolerance=0
            ),
            patch_updates=PatchUpdates(mode),
            regularization=Tikhonov(0.0001),
            step_limit=0.05,
            callback=lambda event: accepted.append(event),
        )
        result = workflow.run(resume=False)
        assert len(accepted) == 2
        view = accepted[0].model.space
        # Events carry the global layout; the immutable stage's coverage is the
        # authoritative bridge from material-space IDs to parent block slots.
        # The workflow's stage view owns the refreshed snapshot.
        stage_problem = workflow._views["stage_01"]
        runtime = stage_problem._patch_runtime
        masks = [
            core_masks(stage_problem, runtime.prepared, i)["model.vp"] for i in range(2)
        ]
        previous = original.values
        for event, mask in zip(accepted, masks):
            full = original.with_update(event.model).values
            assert mask.any() and not mask.all()
            exterior = ~mask
            reference = previous if mode == "local_serial" else original.values
            np.testing.assert_array_equal(full[exterior], reference[exterior])
            previous = full
        assert result.stages[0].success
        assert view.full_size == original.size


@pytest.mark.parametrize("dimension", [2, 3])
def test_central_sparse_vertex_cache_and_concurrent_evaluation(tmp_path, dimension):
    import h5py

    from frequensolve.seismic.receivers import CoordsArray

    parent = _parent(tmp_path, dimension, "acoustic", root_columns=40)
    parent.mesh.set_adapt(elems_per_wave=0.5, order=3, f_low=3, f_high=7.5)
    parent.solver.grids = 2
    parent.acquisition.source_geometry = SourceGeometry.points(
        kind="scalar",
        coords=[[x, 0.08] if dimension == 2 else [x, 0.2, 0.08] for x in (0.50, 0.51)],
    )
    parent.acquisition.receiver_groups[0].coordinates = CoordsArray(
        coordinates=np.array(
            [
                [x, 0.05] if dimension == 2 else [x, 0.2, 0.05]
                for x in (0.49, 0.50, 0.51)
            ]
        )
    )
    options = dict(
        solver=_solver(dimension),
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    )
    with LocalSite(n_workers=1, **options) as serial:
        problem = ImagingProblem(
            parent,
            controls=DepthProfile("vp", "lower", count=2),
            observed=None,
            frequencies=[3, 7.5],
            site=serial,
            misfit=Misfit(normalization=1),
            patches=PatchSet.around_sources(
                shots_per_patch=1, max_offset=30 * ureg.m, padding=0 * ureg.m
            ),
        )
        baseline = problem.linearize()
        prediction = baseline.simulated().values.copy()
        saw_sparse_ids = False
        for _, _, context, _ in problem._patch_runtime.children:
            from pathlib import Path

            mesh_file = Path(context.simulation.mesh.file)
            if not mesh_file.is_absolute():
                mesh_file = Path(context.simulation.project_path) / mesh_file
            cache = mesh_file.parent / "root_point_mapping.h5"
            with h5py.File(cache) as container:
                ids = container["mesh/connectivity"][:]
                ids = ids[ids > 0]
                extent = container["mesh/points"].shape[0]
                assert ids.max() <= extent
                saw_sparse_ids |= ids.max() > len(np.unique(ids))
        assert saw_sparse_ids
        # Reevaluate the same point and immutable frequency meshes through two
        # workers, retaining the first operator across cache eviction.
        problem._patch_runtime.cache.clear()
        with LocalSite(n_workers=2, **options) as concurrent:
            problem.backend.site = concurrent
            repeated = problem.linearize()
            assert repeated.value == pytest.approx(baseline.value, rel=1e-3)
            np.testing.assert_allclose(
                repeated.gradient.values, baseline.gradient.values, rtol=1e-3, atol=1e-9
            )
            np.testing.assert_allclose(
                repeated.simulated().values, prediction, rtol=1e-3, atol=1e-8
            )
            # A retained operator must still work after its cache slot is evicted.
            assert np.isfinite(baseline.jvp(baseline.space.random(42)).values).all()
