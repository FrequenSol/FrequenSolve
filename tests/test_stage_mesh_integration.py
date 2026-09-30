# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
from pathlib import Path

import numpy as np
import pytest

from frequensolve.imaging import (
    ControlSpace,
    DepthProfile,
    ImagingProblem,
    MeshParameters,
    ObservedData,
    SourceParameters,
)
from frequensolve.imaging._artifacts import ControlStateFile, ControlVectorFile
from frequensolve.imaging._backend import (
    read_report,
    read_task_objective_vectors,
    reduce_covectors,
    total_value,
    write_task_objective_vectors,
)
from frequensolve.imaging.data import DataSpace
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.mesh.patches import PatchSet
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.simulation import FrequencyDomainJob
from frequensolve.simulation.simulation import BaseSimulation
from frequensolve.units import ureg
from tests.test_root_patch_integration import _child, _parent, _solver

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("physics", ["acoustic", "coupled"])
@pytest.mark.parametrize("control_kind", ["profile", "mesh"])
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "curved"])
def test_frequency_stage_mesh_capture_and_candidate_replay(
    tmp_path, dimension, physics, control_kind, curved
):
    parent = _parent(tmp_path, dimension, physics, curved=curved, root_columns=8)
    # Objective differences require accurate solves in cancelling directions.
    # Set FS_PATCH_SOLVER_*D to double-precision executables for this suite.
    parent.solver.precision = "double"
    parent.solver.extra["mumps_precision"] = "double"
    parent.solver.tolerance = 1e-10
    parent.mesh.set_adapt(elems_per_wave=0.5, order=2, f_low=3, f_high=7.5)
    with LocalSite(
        solver=_solver(dimension),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob("observed", parent, [3, 7.5])
        assert site.run(observed, check=True).successful
        problem = ImagingProblem(
            parent,
            controls=ControlSpace(
                vp=(
                    DepthProfile("vp", "lower", count=2)
                    if control_kind == "profile"
                    else MeshParameters(
                        "vp",
                        "lower",
                        frequency=1,
                        epw=1,
                        artifact=str(tmp_path / "parent_material.h5"),
                    )
                ),
                src=SourceParameters(signature=True),
            ),
            observed=ObservedData(observed),
            frequencies=[3],
            site=site,
            name="baseline",
        )
        baseline = problem.linearize(gradient=False).job
        prepared = PatchSet.around_sources(
            shots_per_patch=1,
            max_offset=120 * ureg.m,
            padding=0 * ureg.m,
        ).prepare(baseline.simulation, [3, 7.5], site=site)
        assert prepared.geometry["patches"][0]["root_fraction"] < 1
        assert prepared.geometry["patches"][0]["cut_boundary"] > 0
        if curved:
            boundaries = {
                item["name"]: item
                for item in prepared.geometry["patches"][0]["boundaries"]
            }
            for side, horizon in (("bottom", "surface_1"), ("top", "surface_3")):
                for kind in ("edges", "quads", "triangles"):
                    assert set(boundaries[side][kind]) == set(boundaries[horizon][kind])
            edges = np.concatenate(
                [
                    np.asarray(root["edge_points"]).reshape(dimension, -1, 9)
                    for root in prepared.geometry["roots"]
                ],
                axis=1,
            )
            assert (
                np.max(
                    np.linalg.norm(
                        edges[:, :, 4] - 0.5 * (edges[:, :, 0] + edges[:, :, -1]),
                        axis=0,
                    )
                )
                > 1
            )
        stage = prepared.freeze_stage(
            baseline.state_output_file(), name="stage", directory=tmp_path / "stage"
        )
        simulation = BaseSimulation.load(
            stage.simulation_file, project_path=stage.manifest.parent
        )
        simulation = _child(simulation, prepared.jobs[-1], 0, "stage_patch")
        simulation.mesh.file = str(stage.manifest.parent / "parent.gmp")
        capture = FWIOperatorJob(
            "capture",
            simulation,
            [3],
            action="linearize",
            active=[],
            misfit=baseline.misfit,
            state="unused.json",
            control_state=stage.control_state,
            pml_stage=stage,
            stage_mesh="capture",
        )
        assert site.run(capture, check=True).successful
        manifests = list(
            Path(capture._result_path).glob("_fs_run/tasks/*/stage_mesh/manifest.json")
        )
        assert len(manifests) == 1
        mesh = stage.publish_mesh(manifests[0], tmp_path / "mesh")
        low_context = json.loads(mesh.manifest.read_text())["context"]
        assert low_context["pml_frequency_hz"] == 0
        candidate = ControlStateFile.read(stage.control_state)
        candidate.blocks["source.1.signature"] *= 2
        candidate_path = tmp_path / "candidate.h5"
        candidate.write(candidate_path)
        reference = FWIOperatorJob(
            "stage_reference",
            simulation,
            [3],
            action="linearize",
            active=[],
            misfit=baseline.misfit,
            state="reference.json",
            control_state=stage.control_state,
            pml_stage=stage,
            stage_mesh=mesh,
        )
        assert site.run(reference, check=True).successful
        expected = reference.traces.open().fd("surface", "p").values
        assert np.linalg.norm(expected) > 0
        replay = FWIOperatorJob(
            "candidate",
            simulation,
            [3],
            action="linearize",
            active=[],
            misfit=baseline.misfit,
            state="candidate.json",
            control_state=candidate_path,
            pml_stage=stage,
            stage_mesh=mesh,
        )
        assert site.run(replay, check=True, procs_per_job=2).successful
        actual = replay.traces.open().fd("surface", "p").values
        np.testing.assert_allclose(actual, 2 * expected, rtol=3e-4, atol=1e-10)
        assert not Path(
            capture.state
        ).exists(), "Capture must stop before retained wave solves"

        material = ControlStateFile.read(stage.control_state)
        material.blocks["model.vp"] += 0.05
        material_path = tmp_path / "material_candidate.h5"
        material.write(material_path)
        changed_material = FWIOperatorJob(
            "material_candidate",
            simulation,
            [3],
            action="linearize",
            active=["model.vp"] if control_kind == "mesh" else [],
            covector="gradient.h5" if control_kind == "mesh" else None,
            objective="objective.json" if control_kind == "mesh" else None,
            gram_derivative="total",
            misfit=baseline.misfit,
            state="material.json",
            control_state=material_path,
            pml_stage=stage,
            stage_mesh=mesh,
        )
        assert site.run(changed_material, check=True).successful
        changed_values = changed_material.traces.open().fd("surface", "p").values
        assert np.all(np.isfinite(changed_values))
        assert (
            np.linalg.norm(changed_values - expected) / np.linalg.norm(expected) > 1e-5
        )
        if control_kind == "mesh":
            if dimension == 2 and physics == "acoustic":
                for label, provided_stage, expected_error in (
                    ("no_stage", None, "require a verified frozen stage"),
                    ("no_mesh", stage, "require a frozen frequency mesh"),
                ):
                    missing = FWIOperatorJob(
                        label,
                        simulation,
                        [3],
                        action="linearize",
                        active=[],
                        misfit=baseline.misfit,
                        state=f"{label}.json",
                        control_state=stage.control_state,
                        pml_stage=provided_stage,
                    )
                    with pytest.raises(RuntimeError, match="Mesh task failed"):
                        site.run(missing, check=False)
                    errors = list(
                        Path(missing._result_path).glob("_fs_run/**/error.json")
                    )
                    assert errors and any(
                        expected_error in error.read_text() for error in errors
                    )
            gradient = reduce_covectors(changed_material)
            direction = np.random.default_rng(7).uniform(
                -1, 1, material.blocks["model.vp"].shape
            )
            pairing = np.dot(gradient.blocks["model.vp"], direction)
            assert np.isfinite(pairing) and abs(pairing) > 0
            errors = []
            # Check two centered steps without relaxing the derivative tolerance.
            for sample, step in enumerate((0.032, 0.016)):
                costs = []
                for sign in (-1, 1):
                    perturbed = ControlStateFile.read(material_path)
                    perturbed.blocks["model.vp"] += sign * step * direction
                    label = f"material_fd_{sample}_{sign}"
                    perturbed_path = tmp_path / f"{label}.h5"
                    perturbed.write(perturbed_path)
                    fd = FWIOperatorJob(
                        label,
                        simulation,
                        [3],
                        action="linearize",
                        active=[],
                        misfit=baseline.misfit,
                        state="fd.json",
                        objective="fd_objective.json",
                        control_state=perturbed_path,
                        pml_stage=stage,
                        stage_mesh=mesh,
                    )
                    assert site.run(fd, check=True).successful
                    costs.append(total_value(read_report(fd)))
                difference = (costs[1] - costs[0]) / (2 * step)
                errors.append(
                    abs(difference - pairing) / max(abs(difference), abs(pairing))
                )
            assert min(errors) < 0.01, errors
            direction_path = ControlVectorFile(
                {"model.vp": direction},
                state_fingerprint=gradient.state_fingerprint,
                control_registry_fingerprint=gradient.control_registry_fingerprint,
                control_spaces=gradient.control_spaces,
            ).write(tmp_path / "direction.h5")
            jvp = FWIOperatorJob(
                "material_jvp",
                simulation,
                [3],
                action="jvp",
                gram_derivative="total",
                active=["model.vp"],
                misfit=baseline.misfit,
                state=changed_material.state_file(1),
                direction=direction_path,
                objective_vector="jvp.json",
                control_state=material_path,
                pml_stage=stage,
                stage_mesh=mesh,
            )
            assert site.run(jvp, check=True).successful
            space = DataSpace.from_simulation(simulation, [3])
            j_direction = read_task_objective_vectors(
                jvp, space, state_fingerprint=gradient.state_fingerprint
            )
            dual = space.random(11)
            vjp = FWIOperatorJob(
                "material_vjp",
                simulation,
                [3],
                action="vjp",
                gram_derivative="total",
                active=["model.vp"],
                misfit=baseline.misfit,
                state=changed_material.state_file(1),
                objective_vector=tmp_path / "dual.json",
                covector="vjp.h5",
                control_state=material_path,
                pml_stage=stage,
                stage_mesh=mesh,
            )
            vjp.objective_vector = write_task_objective_vectors(
                vjp, dual, space, gradient.state_fingerprint
            )[0]
            assert site.run(vjp, check=True).successful
            pullback = reduce_covectors(vjp)
            left = j_direction.dot(dual)
            right = np.dot(direction, pullback.blocks["model.vp"])
            assert abs(left - right) / max(abs(left), abs(right)) < 1e-3

        if dimension == 2 and physics == "acoustic":
            high_capture = FWIOperatorJob(
                "high_capture",
                simulation,
                [7.5],
                action="linearize",
                active=[],
                misfit=baseline.misfit,
                state="high_unused.json",
                control_state=stage.control_state,
                pml_stage=stage,
                stage_mesh="capture",
            )
            assert site.run(high_capture, check=True).successful
            high_manifests = list(
                Path(high_capture._result_path).glob(
                    "_fs_run/tasks/*/stage_mesh/manifest.json"
                )
            )
            assert len(high_manifests) == 1
            high_mesh = stage.publish_mesh(high_manifests[0], tmp_path / "high_mesh")
            high_context = json.loads(high_mesh.manifest.read_text())["context"]
            assert high_context["frequency_hz"] == [7.5, 0]
            assert high_context["pml_frequency_hz"] == 0
            # Nondimensional adaptation frequencies may match after unit scaling;
            # the realized PML geometry must still reflect the physical frequency.
            assert (
                high_context["geometry_parameters"]
                != low_context["geometry_parameters"]
            )
            high_replay = FWIOperatorJob(
                "high_replay",
                simulation,
                [7.5],
                action="linearize",
                active=[],
                misfit=baseline.misfit,
                state="high.json",
                control_state=material_path,
                pml_stage=stage,
                stage_mesh=high_mesh,
            )
            assert site.run(high_replay, check=True).successful
            high_values = high_replay.traces.open().fd("surface", "p").values
            assert np.all(np.isfinite(high_values)) and np.linalg.norm(high_values) > 0
            moved = ControlStateFile.read(stage.control_state)
            moved.blocks["source.1.position"][0] += 1
            moved_path = tmp_path / "moved_source.h5"
            moved.write(moved_path)
            bad = FWIOperatorJob(
                "moved_source",
                simulation,
                [3],
                action="linearize",
                active=[],
                misfit=baseline.misfit,
                state="bad.json",
                control_state=moved_path,
                pml_stage=stage,
                stage_mesh=mesh,
            )
            site.run(bad, check=False)
            errors = list(Path(bad._result_path).glob("_fs_run/tasks/*/error.json"))
            assert errors and "execution context differs" in errors[0].read_text()
