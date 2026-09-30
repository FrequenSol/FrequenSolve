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


def mesh_problem(tmp_path, site, *, fixed_geometry=False):
    project = Project(
        name="mesh_stages", path=tmp_path / "project", load_if_exists=False
    )
    simulation = _simulation(project, "initial", START_VP)
    if fixed_geometry:
        # Explicit affine roots isolate transfer from the layered mesh generator.
        mesh = tmp_path / "roots.gmp"
        mesh.write_text(
            """@Version = 2
@SubVersion = 0
@Patch = 0
@Dimension = 2
@Manifold = 2
@Domains = 2
@Units = "km"
$Points
@Count = 9
@TypePos = 1
@IndexPos = 2
Point 1 0 0
Point 2 .5 0
Point 3 1 0
Point 4 0 .25
Point 5 .5 .25
Point 6 1 .25
Point 7 0 .6
Point 8 .5 .6
Point 9 1 .6
$EndPoints
$Curves
@Count = 12
@TypePos = 1
Seg 1 2
Seg 2 3
Seg 4 5
Seg 5 6
Seg 7 8
Seg 8 9
Seg 1 4
Seg 2 5
Seg 3 6
Seg 4 7
Seg 5 8
Seg 6 9
$EndCurves
$Quads
@Count = 4
@TypePos = 1
@DomainPos = 2
Linear 1 1 2 5 4
Linear 1 2 3 6 5
Linear 2 4 5 8 7
Linear 2 5 6 9 8
$EndQuads
"""
        )
        simulation.mesh.mesh = None
        simulation.mesh.file = str(mesh)
        simulation.mesh.format = "gmp"
        from frequensolve.mesh import BoundaryCondition

        simulation.BCs.clear()
        simulation += BoundaryCondition(
            conditions=["free"], boundaries=["x_min", "x_max", "z_min", "z_max"]
        )
        simulation.save()
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


@pytest.mark.parametrize("transfer,length", [("nodal", 0.0), ("l2", 0.0), ("l2", 50.0)])
def test_mesh_adaptation_uses_recovered_material_and_transfers_constants(
    tmp_path, site, transfer, length
):
    problem, spec = mesh_problem(tmp_path, site, fixed_geometry=transfer == "l2")
    initial = problem.state
    target = dataclasses.replace(spec, frequency=FREQUENCY, epw=2.0)
    recovered = im.ControlState(
        initial.space, np.full(initial.space.full_size, np.log(0.6))
    )
    problem.state = recovered
    adapted = problem.with_controls(
        {"vp": target}, mesh_transfer=transfer, mesh_smoothing_length=length
    )
    np.testing.assert_allclose(adapted.state.values, recovered.values[0], atol=1e-9)
    assert adapted.full_space.full_size > initial.space.full_size
    basis = adapted.full_space.resolved_blocks[0].basis_identity
    # Trial evaluation must never change the basis or regenerate its artifact.
    artifact = adapted.full_space.specs["vp"].artifact
    from pathlib import Path

    stamp = Path(artifact).stat().st_mtime_ns
    if transfer == "nodal":
        adapted.value()
        adapted.value(adapted.vector().values + 0.001)
    assert Path(artifact).stat().st_mtime_ns == stamp
    assert adapted.full_space.resolved_blocks[0].basis_identity == basis
    coarse = adapted.with_controls(
        {"vp": spec}, mesh_transfer=transfer, mesh_smoothing_length=length
    )
    np.testing.assert_allclose(coarse.state.values, recovered.values[0], atol=1e-9)
    assert coarse.full_space.full_size < adapted.full_space.full_size
    problem.state = initial
    fast = problem.with_controls(
        {"vp": target}, mesh_transfer=transfer, mesh_smoothing_length=length
    )
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


def _projection_roots(problem):
    """Decode constrained quad leaves for independent physical-space integration."""
    import h5py

    from frequensolve.imaging.property_mesh import read_reference_vertices

    roots = {}
    with h5py.File(problem.full_space.specs["vp"].artifact) as h5:
        offset, count = h5["property_space/material_ranges"][1]
        for name, root in h5["property_space/roots"].items():
            if root["meta"][0] != 2:
                continue
            vertices = read_reference_vertices(root)[:, :4]
            bounds = np.stack([vertices.min(axis=1), vertices.max(axis=1)], axis=1)
            leaves = []
            for i, box in enumerate(bounds):
                children = np.all(bounds[:, 0] >= box[0] - 1e-12, axis=1) & np.all(
                    bounds[:, 1] <= box[1] + 1e-12, axis=1
                )
                children[i] = False
                if not children.any():
                    leaves.append(i)
            ptr = root["offset"][:] - 1
            ids = root["global_ids"][:][root["index"][:] - 1] - offset - 1
            weights = root["weight"][:]
            cells = []
            slots = (len(ptr) - 1) // len(leaves)
            for leaf, node in enumerate(leaves):
                constraint = np.zeros((4, count))
                for corner in range(4):
                    lo, hi = ptr[slots * leaf + corner : slots * leaf + corner + 2]
                    np.add.at(constraint[corner], ids[lo:hi], weights[lo:hi])
                cells.append((bounds[node], constraint))
            roots[name] = (root["physical_vertices"][:4], cells)
    # Native artifacts may be saved under different frequency normalizations.
    # Anchor both to this fixture's authored 1-km width before integrating.
    width = np.ptp(np.concatenate([p for p, _ in roots.values()])[:, 0])
    roots = {
        name: (physical / width, cells) for name, (physical, cells) in roots.items()
    }
    return int(count), roots


def _quad_basis(x):
    """Bilinear values and derivatives on [0,1]^2, independent of native routines."""
    a, b = x
    return np.array([(1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b]), np.array(
        [[b - 1, a - 1], [1 - b, -a], [b, a], [-b, 1 - a]]
    )


def _projection_operators(source, target, *, wavelength=None):
    """Dense reference M, K, B from exact intersections of the fixture's quad leaves."""
    ns, old = _projection_roots(source)
    nt, new = _projection_roots(target)
    mass, stiffness, cross = np.zeros((nt, nt)), np.zeros((nt, nt)), np.zeros((nt, ns))
    points, weights = np.polynomial.legendre.leggauss(
        8 if wavelength is not None else 3
    )
    points, weights = (points + 1) / 2, weights / 2
    for root, (physical, target_cells) in new.items():
        np.testing.assert_allclose(physical, old[root][0], atol=1e-12)
        for target_box, target_constraint in target_cells:
            for source_box, source_constraint in old[root][1]:
                lo = np.maximum(target_box[0], source_box[0])
                hi = np.minimum(target_box[1], source_box[1])
                if np.any(hi <= lo):
                    continue
                for i, x in enumerate(points):
                    for j, y in enumerate(points):
                        eta = lo + (hi - lo) * [x, y]
                        _, derivative = _quad_basis(eta)
                        jacobian = physical.T @ derivative
                        weight = (
                            abs(np.linalg.det(jacobian))
                            * np.prod(hi - lo)
                            * weights[i]
                            * weights[j]
                        )
                        nt_shape, nt_grad = _quad_basis(
                            (eta - target_box[0]) / np.diff(target_box, axis=0)[0]
                        )
                        ns_shape, _ = _quad_basis(
                            (eta - source_box[0]) / np.diff(source_box, axis=0)[0]
                        )
                        t = nt_shape @ target_constraint
                        s = ns_shape @ source_constraint
                        grad = (
                            target_constraint.T
                            @ (nt_grad / (target_box[1] - target_box[0]))
                            @ np.linalg.inv(jacobian)
                        )
                        mass += weight * np.outer(t, t)
                        length = 1.0
                        if wavelength is not None:
                            values, fraction, frequency = wavelength
                            length = (
                                fraction * START_VP * np.exp(s @ values) / frequency
                            )
                        stiffness += weight * length**2 * (grad @ grad.T)
                        cross += weight * np.outer(t, s)
    return mass, stiffness, cross


@pytest.mark.parametrize("direction", ["finer", "coarser", "crossed"])
def test_mesh_smooth_project_matches_integral_equations(tmp_path, site, direction):
    problem, coarse = mesh_problem(tmp_path, site, fixed_geometry=True)
    # A physical-unit fixture makes the independent stiffness check unambiguous.
    # The authored model spans 1 km horizontally; recovered artifacts must agree.
    fine = dataclasses.replace(coarse, frequency=12.0, epw=2.0)
    if direction == "crossed":
        problem = problem.with_controls(
            {"vp": dataclasses.replace(fine, epw=(3.0, 1.0))}
        )
        target = dataclasses.replace(fine, epw=(1.0, 3.0))
    elif direction == "coarser":
        problem = problem.with_controls({"vp": fine})
        target = coarse
    else:
        target = fine
    state = problem.state
    values = 0.06 * np.random.default_rng(182).standard_normal(len(state.values))
    problem.state = im.ControlState(state.space, values)
    projected = problem.with_controls({"vp": target}, mesh_transfer="l2")
    smoothed = problem.with_controls(
        {"vp": target}, mesh_transfer="l2", mesh_smoothing_length=50.0
    )
    if direction != "crossed":
        assert (projected.full_space.full_size > len(values)) == (direction == "finer")
    mass, stiffness, cross = _projection_operators(problem, projected)
    rhs = cross @ values
    np.testing.assert_allclose(
        projected.state.values, np.linalg.solve(mass, rhs), rtol=2e-7, atol=2e-9
    )
    # Artifacts store root coordinates in runtime units; derive the fixture's km scale.
    _, roots = _projection_roots(projected)
    width = np.ptp(np.concatenate([p for p, _ in roots.values()])[:, 0])
    alpha = (0.050 * width) ** 2
    expected = np.linalg.solve(mass + alpha * stiffness, rhs)
    np.testing.assert_allclose(smoothed.state.values, expected, rtol=2e-7, atol=2e-9)
    for result in (projected, smoothed):
        np.testing.assert_allclose(
            np.ones(len(rhs)) @ mass @ result.state.values,
            rhs.sum(),
            rtol=1e-8,
            atol=1e-10,
        )
    assert (
        smoothed.state.values @ stiffness @ smoothed.state.values
        < projected.state.values @ stiffness @ projected.state.values
    )
    np.testing.assert_array_equal(problem.state.values, values)
    if direction == "finer":
        interpolated = problem.with_controls({"vp": target})
        np.testing.assert_allclose(
            projected.state.values, interpolated.state.values, rtol=2e-7, atol=2e-9
        )
    else:
        interpolated = problem.with_controls({"vp": target})
        assert np.linalg.norm(projected.state.values - interpolated.state.values) > 1e-3


@pytest.mark.parametrize("direction", ["finer", "coarser"])
def test_wavelength_smoothing_uses_accepted_local_wavespeed(tmp_path, site, direction):
    from frequensolve.units import ureg as u

    problem, coarse = mesh_problem(tmp_path, site, fixed_geometry=True)
    fine = dataclasses.replace(coarse, frequency=12.0, epw=2.0)
    if direction == "coarser":
        problem = problem.with_controls({"vp": fine})
        target = coarse
    else:
        target = fine
    state = problem.state
    values = 0.15 * np.sin(np.arange(len(state.values)) * 1.7)
    problem.state = im.ControlState(state.space, values)
    fraction, frequency = 0.2, 3.0
    result = problem.with_controls(
        {"vp": target},
        transfer=im.Transfer.l2(
            smooth_wavelengths=fraction, frequency=frequency * u.Hz
        ),
    )
    mass, stiffness, cross = _projection_operators(
        problem, result, wavelength=(values, fraction, frequency)
    )
    rhs = cross @ values
    np.testing.assert_allclose(
        result.state.values,
        np.linalg.solve(mass + stiffness, rhs),
        rtol=2e-6,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        np.ones(len(rhs)) @ mass @ result.state.values, rhs.sum(), rtol=1e-8, atol=1e-10
    )
    _, fixed_stiffness, _ = _projection_operators(problem, result)
    fixed = np.linalg.solve(
        mass + (fraction * START_VP / frequency) ** 2 * fixed_stiffness, rhs
    )
    assert np.linalg.norm(result.state.values - fixed) > 1e-5
    # Omitting frequency uses the target sizing frequency, including on coarsening.
    default = problem.with_controls(
        {"vp": target}, transfer=im.Transfer.l2(smooth_wavelengths=fraction)
    )
    explicit = problem.with_controls(
        {"vp": target},
        transfer=im.Transfer.l2(
            smooth_wavelengths=fraction, frequency=target.frequency
        ),
    )
    np.testing.assert_allclose(
        default.state.values, explicit.state.values, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_array_equal(problem.state.values, values)
