# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Parent-normalized patch contributions using canonical controls and stage PML."""

import numpy as np
import pytest

from frequensolve.imaging import DepthProfile, ImagingProblem, Misfit
from frequensolve.imaging._artifacts import ControlStateFile, ControlVectorFile
from frequensolve.imaging._backend import (
    read_report,
    read_task_objective_vectors,
    reduce_covectors,
    total_value,
)
from frequensolve.imaging._objective import ObjectiveState, objective_space
from frequensolve.imaging._patch_objective import (
    patch_objective_keys,
    restrict_patch_misfit,
)
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.mesh.patches import Patch, PatchSet
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.seismic.sources import SourceGeometry
from frequensolve.simulation import FrequencyDomainJob
from frequensolve.units import ureg
from tests.test_root_patch_integration import _child, _parent, _solver

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
@pytest.mark.parametrize("normalization", [1.0, "observed_rms"])
def test_parent_normalized_patch_values_and_covectors(
    tmp_path, dimension, physics, normalization
):
    parent = _parent(tmp_path, dimension, physics)
    if normalization == "observed_rms":
        # Near-truth residuals amplify forward-solve error in gradient comparisons.
        parent.solver.tolerance = 1e-6
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
        observed = None
        if normalization == "observed_rms":
            observed = FrequencyDomainJob("observed", parent, [3])
            assert site.run(observed, check=True, procs_per_job=2).successful
        problem = ImagingProblem(
            parent,
            controls=DepthProfile("vp", "lower", count=2),
            observed=observed,
            frequencies=[3],
            site=site,
            misfit=Misfit(normalization=normalization, weights=2.0),
            submit_options={"procs_per_job": 2},
        )
        baseline = problem.linearize(gradient=False).job
        inventory = PatchSet.around_sources(
            shots_per_patch=3,
            max_offset=2000 * ureg.m,
            padding=0 * ureg.m,
        ).prepare(baseline.simulation, [3], site=site)
        roots = tuple(row["root"] for row in inventory.geometry["roots"])
        prepared = PatchSet(
            [
                Patch(name="odd", roots=roots, sources=(1, 3)),
                Patch(name="even", roots=roots, sources=(2,)),
            ],
            max_offset=2000 * ureg.m,
            padding=0 * ureg.m,
        ).prepare(baseline.simulation, [3], site=site)
        stage = prepared.freeze_stage(
            baseline.state_output_file(),
            name="stage",
            directory=tmp_path / "stage",
        )
        candidate = ControlStateFile.read(stage.control_state)
        # Profile log-control baselines can be zero; perturb native coordinates.
        # Keep RMS candidate costs above the FP32 near-zero comparison floor.
        candidate.blocks["model.vp"] += (
            0.15 if normalization == "observed_rms" else 0.03
        )
        candidate_path = candidate.write(tmp_path / "candidate.h5")
        children = prepared.simulations(name="restricted")
        reference = _child(baseline.simulation, prepared.jobs[-1], 0, "reference")
        jobs = []
        for index, simulation in enumerate((reference, *children)):
            simulation.mesh.file = str(stage.manifest.parent / "parent.gmp")
            misfit = restrict_patch_misfit(
                baseline.misfit.to_fs(),
                baseline.state_file(1),
                receiver_groups=[
                    group.name for group in simulation.acquisition.receiver_groups
                ],
            )
            capture = FWIOperatorJob(
                f"capture_{index}",
                simulation,
                [3],
                action="linearize",
                active=[],
                state="unused.json",
                misfit=misfit,
                control_state=stage.control_state,
                pml_stage=stage,
                stage_mesh="capture",
            )
            assert site.run(capture, check=True, procs_per_job=2).successful
            ImagingProblem._check_tasks(capture)
            manifests = list(
                capture._result_path.glob("_fs_run/tasks/*/stage_mesh/manifest.json")
            )
            assert len(manifests) == 1
            mesh = stage.publish_mesh(manifests[0], tmp_path / f"mesh_{index}")
            job = FWIOperatorJob(
                f"linearize_{index}",
                simulation,
                [3],
                action="linearize",
                active=["model.vp"],
                state="state.json",
                objective="report.json",
                covector="gradient.h5",
                misfit=misfit,
                gram_derivative="total",
                control_state=candidate_path,
                pml_stage=stage,
                stage_mesh=mesh,
            )
            assert site.run(job, check=True, procs_per_job=2).successful
            ImagingProblem._check_tasks(job)
            jobs.append(job)
            if normalization == "observed_rms":
                matched = FWIOperatorJob(
                    f"matched_{index}",
                    simulation,
                    [3],
                    action="linearize",
                    active=["model.vp"],
                    state="matched.json",
                    objective="report.json",
                    covector="gradient.h5",
                    misfit=misfit,
                    gram_derivative="total",
                    control_state=stage.control_state,
                    pml_stage=stage,
                    stage_mesh=mesh,
                )
                assert site.run(matched, check=True, procs_per_job=2).successful
                ImagingProblem._check_tasks(matched)
                assert total_value(read_report(matched)) < 1e-9
                assert (
                    np.linalg.norm(reduce_covectors(matched).blocks["model.vp"]) < 1e-6
                )
        values = [total_value(read_report(job)) for job in jobs]
        assert values[0] > 0
        np.testing.assert_allclose(sum(values[1:]), values[0], rtol=3e-4, atol=1e-10)
        gradients = [reduce_covectors(job).blocks["model.vp"] for job in jobs]
        assert np.linalg.norm(gradients[0]) > 0
        # Measure the full covector error; small cancelling components should
        # not set a tighter effective tolerance than the full covector norm.
        gradient_error = np.linalg.norm(
            sum(gradients[1:]) - gradients[0]
        ) / np.linalg.norm(gradients[0])
        assert gradient_error < 3e-4, gradient_error
        if normalization == 1.0:
            np.testing.assert_allclose(
                sum(gradients[1:]), gradients[0], rtol=3e-4, atol=1e-10
            )
        keys = [
            patch_objective_keys(job.state_file(1), job.simulation.acquisition.to_fs())[
                "surface"
            ]
            for job in jobs
        ]
        assert set(keys[1][:, 0]) == {1, 3}
        assert set(keys[2][:, 0]) == {2}
        assert {tuple(key) for key in np.concatenate(keys[1:])} == {
            tuple(key) for key in keys[0]
        }

        tangents, normals = [], []
        direction = np.array([0.2, -0.1])
        for index, job in enumerate(jobs):
            gradient = reduce_covectors(job)
            direction_path = ControlVectorFile(
                {"model.vp": direction},
                state_fingerprint=gradient.state_fingerprint,
                control_registry_fingerprint=gradient.control_registry_fingerprint,
                control_spaces=gradient.control_spaces,
            ).write(tmp_path / f"direction_{index}.h5")
            common = dict(
                simulation=job.simulation,
                f_list=[3],
                active=["model.vp"],
                state=job.state_file(1),
                control_state=candidate_path,
                misfit=job.misfit,
                pml_stage=stage,
                stage_mesh=job.stage_mesh,
                gram_derivative="total",
            )
            jvp = FWIOperatorJob(
                f"jvp_{index}",
                action="jvp",
                direction=direction_path,
                objective_vector="jvp.json",
                **common,
            )
            assert site.run(jvp, check=True, procs_per_job=2).successful
            ImagingProblem._check_tasks(jvp)
            space = objective_space(
                job.simulation, [3], [ObjectiveState(job.state_file(1))]
            )
            tangent = read_task_objective_vectors(
                jvp, space, state_fingerprint=gradient.state_fingerprint
            )
            layout = space.term_layout("surface", frequency=3)
            # Typed dense vectors use receiver-major packing, while saved keys
            # are indexed by native row ID. Compare rows through that index.
            tangents.append(tangent.values[layout.indices[np.argsort(layout.row_ids)]])
            normal = FWIOperatorJob(
                f"normal_{index}",
                action="normal",
                direction=direction_path,
                covector="normal.h5",
                **common,
            )
            assert site.run(normal, check=True, procs_per_job=2).successful
            ImagingProblem._check_tasks(normal)
            normal_vector = reduce_covectors(normal).blocks["model.vp"]
            normals.append(normal_vector)
            np.testing.assert_allclose(
                np.dot(direction, normal_vector), tangent.dot(tangent), rtol=1e-3
            )
            vjp = FWIOperatorJob(
                f"vjp_{index}",
                action="vjp",
                objective_vector=jvp.objective_vector_file(1),
                covector="vjp.h5",
                **common,
            )
            assert site.run(vjp, check=True, procs_per_job=2).successful
            ImagingProblem._check_tasks(vjp)
            np.testing.assert_allclose(
                reduce_covectors(vjp).blocks["model.vp"],
                normal_vector,
                rtol=1e-3,
                atol=1e-10,
            )
        np.testing.assert_allclose(sum(normals[1:]), normals[0], rtol=3e-4)
        reference_rows = {tuple(key): value for key, value in zip(keys[0], tangents[0])}
        for child_keys, tangent in zip(keys[1:], tangents[1:]):
            np.testing.assert_allclose(
                tangent,
                [reference_rows[tuple(key)] for key in child_keys],
                rtol=3e-4,
                atol=1e-10,
            )
