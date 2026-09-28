"""Stage-only mesh adaptation against the real local solver."""

import dataclasses

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.simulation import FrequencyDomainJob
from tests.test_imaging_integration import FREQUENCY, START_VP, _executable, _simulation

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]


@pytest.fixture
def site():
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=4,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as local:
        yield local


def mesh_problem(tmp_path, site):
    project = Project(
        name="mesh_stages", path=tmp_path / "project", load_if_exists=False
    )
    simulation = _simulation(project, "initial", START_VP)
    observed = FrequencyDomainJob("observed", simulation, [FREQUENCY])
    site.run(observed, check=True)
    spec = im.MeshParameters(
        "vp",
        "layer_2",
        frequency=1.0,
        epw=1.0,
        artifact=str(tmp_path / "initial.h5"),
        transform="log",
    )
    problem = im.ImagingProblem(
        simulation,
        controls=im.ControlSpace(vp=spec),
        observed=im.ObservedData(observed),
        site=site,
        name="mesh_stages",
    )
    return problem, spec


def test_mesh_adaptation_uses_recovered_material_and_transfers_constants(
    tmp_path, site
):
    problem, spec = mesh_problem(tmp_path, site)
    initial = problem.state
    target = dataclasses.replace(spec, frequency=FREQUENCY, epw=2.0)
    recovered = im.ControlState(
        initial.space, np.full(initial.space.full_size, np.log(0.6))
    )
    problem.state = recovered
    adapted = problem.with_controls({"vp": target})
    np.testing.assert_allclose(adapted.state.values, recovered.values[0], atol=1e-12)
    assert adapted.full_space.full_size > initial.space.full_size
    basis = adapted.full_space.resolved_blocks[0].basis_identity
    # Trial evaluation must never change the basis or regenerate its artifact.
    artifact = adapted.full_space.specs["vp"].artifact
    from pathlib import Path

    stamp = Path(artifact).stat().st_mtime_ns
    adapted.value()
    adapted.value(adapted.vector().values + 0.001)
    assert Path(artifact).stat().st_mtime_ns == stamp
    assert adapted.full_space.resolved_blocks[0].basis_identity == basis
    coarse = adapted.with_controls({"vp": spec})
    np.testing.assert_allclose(coarse.state.values, recovered.values[0], atol=1e-12)
    assert coarse.full_space.full_size < adapted.full_space.full_size
    problem.state = initial
    fast = problem.with_controls({"vp": target})
    assert fast.full_space.full_size < adapted.full_space.full_size
    np.testing.assert_array_equal(fast.state.values, 0)


def test_mesh_stage_checkpoint_replays_accepted_sizing_state(tmp_path, site):
    problem, spec = mesh_problem(tmp_path, site)
    initial = problem.state
    problem.state = im.ControlState(
        initial.space, np.full(initial.space.full_size, np.log(0.9))
    )
    stage = im.Stage(
        [FREQUENCY],
        iterations=1,
        controls={"vp": dataclasses.replace(spec, frequency=FREQUENCY, epw=2.0)},
        regularization=im.TV(1e-8),
    )
    checkpoint = tmp_path / "checkpoint.h5"
    result = im.FWI(problem, stage, checkpoint=checkpoint, step_limit=0.01).run()
    # Restart with an independently authored initial state, not the sizing model.
    problem.state = initial
    restored = im.FWI(problem, stage, checkpoint=checkpoint, step_limit=0.01).run()
    assert restored.problem.full_space.equivalent(result.problem.full_space)
    np.testing.assert_array_equal(restored.state.values, result.state.values)


def test_material_averaging_reduces_thin_slow_feature_refinement(tmp_path, site):
    import h5py

    problem, spec = mesh_problem(tmp_path, site)
    initial = problem.state
    coefficients = initial.values.copy()
    # The initial fixture has unrefined straight quads; use independent native
    # root vertices, without depending on optional visualization datasets.
    selected = []
    with h5py.File(next(tmp_path.glob("initial*.h5")), "r") as h5:
        group = h5["property_space"]
        offset = group["material_ranges"][1, 0]
        for root in group["roots"].values():
            if root["meta"][0] != 2:
                continue
            points = root["physical_vertices"][:]
            ids = root["global_ids"][:]
            indices = root["index"][:]
            ptr = root["offset"][:] - 1
            for vertex in np.flatnonzero(np.isclose(points[:, 0], 0.5)):
                selected.extend(
                    ids[indices[ptr[vertex] : ptr[vertex + 1]] - 1] - offset - 1
                )
    assert selected
    coefficients[np.unique(selected).astype(int)] = np.log(0.3)
    problem.state = im.ControlState(initial.space, coefficients)
    target = dataclasses.replace(spec, frequency=FREQUENCY, epw=2.0)
    localized = problem.with_controls({"vp": target}, mesh_averaging_wavelengths=0.01)
    averaged = problem.with_controls({"vp": target})
    from frequensolve.imaging.property_mesh import read_reference_vertices

    def minimum_cell_span(adapted):
        spans = []
        with h5py.File(adapted.full_space.specs["vp"].artifact, "r") as h5:
            for root in h5["property_space/roots"].values():
                if root["meta"][0] == 2:
                    spans.append(np.ptp(read_reference_vertices(root), axis=1).min())
        return min(spans)

    # Averages suppress peak refinement; a broader sizing field need not use
    # fewer coefficients overall, especially near dyadic refinement thresholds.
    assert minimum_cell_span(averaged) > minimum_cell_span(localized)
    assert averaged.full_space.full_size < localized.full_space.full_size
    # Sizing smooths only the reference used to choose the basis, not the model.
    assert averaged.state.values.min() < -0.5
    np.testing.assert_array_equal(problem.state.values, coefficients)


def test_adapted_mesh_objective_and_gradient_are_rank_invariant(tmp_path, site):
    import hashlib
    import shutil
    from pathlib import Path

    if shutil.which("mpiexec") is None:
        pytest.skip("MPI launcher is not available")
    problem, spec = mesh_problem(tmp_path, site)
    initial = problem.state
    problem.state = im.ControlState(
        initial.space, np.full(initial.space.full_size, np.log(0.9))
    )
    adapted = problem.with_controls(
        {"vp": dataclasses.replace(spec, frequency=FREQUENCY, epw=2.0)}
    )
    artifact = Path(adapted.full_space.specs["vp"].artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    serial = adapted.linearize()
    value, gradient = serial.value, serial.gradient.values.copy()
    assert value > 0
    assert np.linalg.norm(gradient) > 0
    adapted.clear_cache()
    adapted.backend.submit_options["procs_per_job"] = 2
    distributed = adapted.linearize()
    import json

    report = distributed.job._result_path / "_fs_run/tasks/task_000001/result.json"
    assert json.loads(report.read_text())["execution"]["mpi_ranks"] == 2
    np.testing.assert_allclose(distributed.value, value, rtol=2e-5, atol=1e-12)
    np.testing.assert_allclose(
        distributed.gradient.values, gradient, rtol=2e-4, atol=1e-10
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
