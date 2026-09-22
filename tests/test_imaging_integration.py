"""Solver-backed acceptance test for :class:`frequensolve.imaging.ImagingProblem`.

A tiny 2D acoustic layered model (mirroring the Sauce ``acoustic1`` e2e
fixture: 1 km x 0.6 km, two layers, one scalar source, a hydrophone line)
produces observed data on a "truth" simulation; an ``ImagingProblem`` on a
perturbed starting model then linearizes one ``DepthProfile`` control and its
Jacobian is checked against the adjoint identity.

The test needs a Sauce executable: ``FS_SAUCE_EXECUTABLE`` or
``LOCAL_SOLVER_EXECUTABLE`` in the environment, or the staged build at
``/tmp/FS_stage-imaging-merge/agent/install/fs2d_s``.  It skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from frequensolve.imaging import (
    ControlSpace,
    DepthProfile,
    ImagingProblem,
    ObservedData,
    SourceParameters,
)
from frequensolve.imaging._artifacts import ControlStateFile, ControlVectorFile
from frequensolve.imaging.extension import Extension, ExtensionVector, Lags
from frequensolve.mesh import BoundaryCondition
from frequensolve.model.layered import LayeredModel
from frequensolve.project import Project
from frequensolve.seismic import Acquisition, ReceiverNode
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from frequensolve.units import ureg as u

pytestmark = pytest.mark.integration

DEFAULT_EXECUTABLE = Path("/tmp/FS_stage-imaging-merge/agent/install/fs2d_s")
FREQUENCY = 6.0
TRUTH_VP = 2.5
START_VP = 2.2


def _executable() -> Path:
    for key in ("FS_SAUCE_EXECUTABLE", "LOCAL_SOLVER_EXECUTABLE"):
        raw = os.environ.get(key)
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        pytest.skip(f"{key}={raw!r} is not an executable file")
    if DEFAULT_EXECUTABLE.is_file() and os.access(DEFAULT_EXECUTABLE, os.X_OK):
        return DEFAULT_EXECUTABLE
    pytest.skip(
        "no Sauce executable: set FS_SAUCE_EXECUTABLE (or LOCAL_SOLVER_EXECUTABLE)"
    )


def _simulation(project: Project, name: str, vp_lower: float, *, robust=False):
    simulation = project.new_simulation(name=name, physics="acoustic", dimension=2)
    if robust:
        # Robust runtime scaling derives its unit scales from the task
        # frequency (at least Mesh/adapt/f_low), so nondimensional mechanism
        # coordinates differ between the tasks of a multi-frequency job.
        simulation.units.extra["scaling"] = "robust"
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer_1", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="interface", depth=0.25)
    model.add_layer(name="layer_2", properties={"vp": vp_lower, "rho": 2.2})
    model.add_surface(name="bottom", depth=0.6)
    simulation += model
    simulation += model.hex_mesh_generator(n=[8, 6])
    simulation.mesh.set_adapt(
        elems_per_wave=2.0,
        order=4,
        f_low=FREQUENCY - 1.0 if robust else FREQUENCY,
        f_high=FREQUENCY,
        adapt_order=True,
    )
    simulation += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    simulation += BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=1.2,
        pml_exponent=3.0,
        pml_constant=20.0,
    )
    acquisition = Acquisition()
    acquisition.add_sources(kind="scalar", coords=[[0.5, 0.08]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[x, 0.05] for x in np.linspace(0.1, 0.9, 17)],
    )
    simulation += acquisition
    simulation += Discretization()
    # Single-precision FS_MG stalls near 1e-5; the e2e acoustic fixture uses 1e-4.
    simulation += SolverConfig(solve_on="final", max_iter=300, tolerance=1.0e-4)
    simulation.save()
    return simulation


def test_imaging_problem_linearizes_against_sauce(tmp_path):
    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="imaging", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)

    observed_job = FrequencyDomainJob("observed", truth, [FREQUENCY])
    result = site.run(observed_job, check=True)
    assert result.successful

    problem = ImagingProblem(
        initial,
        controls=DepthProfile("vp", "layer_2", count=4),
        observed=ObservedData(observed_job),
        site=site,
        name="fwi",
    )
    assert problem.frequencies == [FREQUENCY]
    assert problem.space.blocks == ("model.vp",)

    lin = problem.linearize()

    assert np.isfinite(lin.value) and lin.value > 0.0
    assert lin.gradient is not None and lin.gradient.size == lin.space.size
    assert np.all(np.isfinite(lin.gradient.values))
    assert np.linalg.norm(lin.gradient.values) > 0.0
    assert lin.report.keys() == {"surface"}
    assert lin.state_fingerprint.startswith("sha256:")
    assert lin.registry_fingerprint == lin.manifest.fingerprint

    # Sauce exported a decodable support mask for the profile
    baseline = ControlStateFile.read(lin.job.state_output_file())
    assert "model.vp" in baseline.support
    assert baseline.support["model.vp"].shape == (4,)
    assert set(lin.support_masks) == {"model.vp"}
    assert lin.support["vp"].dtype == bool and lin.support["vp"].any()
    assert lin.space.size == int(lin.support["vp"].sum())

    # <J dv, r>_Re == <dv, J^H r> through Sauce's jvp / vjp
    objective_gradient = lin.vjp(lin.objective_residual()).values
    np.testing.assert_allclose(
        objective_gradient, lin.gradient.values, rtol=2e-3, atol=1e-6
    )
    report = lin.jacobian.dot_test(seed=1, tolerance=1.0e-3)
    assert report["passed"], report
    assert report["relative_error"] < 1.0e-3

    # the value at the current state is cached; a moved point is not
    assert problem.value() == lin.value
    moved = problem.value(problem.vector() + 0.01)
    assert np.isfinite(moved) and moved != lin.value


def test_multi_frequency_operators_and_source_baseline_against_sauce(tmp_path):
    """One job per action over two frequencies; mechanisms start at Sauce's baseline."""

    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="multi", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP, robust=True)
    initial = _simulation(project, "initial", START_VP, robust=True)
    frequencies = [FREQUENCY - 1.0, FREQUENCY]

    observed_job = FrequencyDomainJob("observed", truth, frequencies)
    assert site.run(observed_job, check=True).successful

    problem = ImagingProblem(
        initial,
        controls=ControlSpace(
            vp=DepthProfile("vp", "layer_2", count=4),
            src=SourceParameters(mechanism=True, signature=False),
        ),
        observed=ObservedData(observed_job),
        frequencies=frequencies,
        site=site,
        name="multi",
    )
    assert problem.space.blocks == ("model.vp", "source.1.mechanism")

    # the state is Sauce's baseline: a nonzero mechanism with its scaling
    state = problem.state
    mechanism = state["source.1.mechanism"]
    assert np.all(np.isfinite(mechanism)) and np.abs(mechanism).max() > 0.0
    assert state.scaling["source.1.mechanism"] > 0.0
    discovery = problem.linearize(gradient=False)
    assert discovery.job.n_tasks == 2 and discovery.job.control_state is None

    lin = problem.linearize()
    assert lin.frequencies == frequencies and lin.job.n_tasks == 2
    assert np.all(np.isfinite(lin.gradient.values))

    # jvp / vjp resolve each task's saved state and direction from one job.
    # The mechanism sensitivities exceed the vp ones by ~1e10, so each block
    # is tested on its own.  The two-frequency mechanism pairing of this Sauce
    # build agrees to ~2e-3 only (one frequency: ~1e-5); the one-job outputs
    # match per-frequency single-task jobs to ~4e-6, so the gap is Sauce's.
    J = lin.jacobian
    r = lin.data_space.random(2)
    jh_r = np.asarray(J.H @ r)
    tolerances = {"model.vp": 1.0e-3, "source.1.mechanism": 1.0e-2}
    for block, sl in lin.space.slices.items():
        dv = np.zeros(lin.space.size)
        dv[sl] = lin.space.random(1).values[sl]
        left, right = (J @ dv).dot(r), float(np.dot(dv, jh_r))
        scale = max(abs(left), abs(right))
        assert abs(left - right) <= tolerances[block] * scale, (block, left, right)

    # a point off the authored one stages controls.state (mechanism scaling
    # included) and replays in both tasks
    step = -1.0e-3 * lin.gradient / max(float(np.abs(lin.gradient.values).max()), 1.0)
    moved = problem.linearize(problem.vector() + step, gradient=False)
    staged = ControlStateFile.read(moved.job.control_state)
    assert staged.scaling == {"source.1.mechanism": state.scaling["source.1.mechanism"]}
    assert np.isfinite(moved.value) and moved.value != lin.value

    # The mechanism gradient against a central difference of the objective.
    # Each task stores the mechanism in its own nondimensional units
    # (robust scaling depends on the task frequency), so the covector parts
    # must be converted to the reference coordinate (s_ref = task 1's scale)
    # before they are summed; the dot tests above cannot see a missing
    # conversion.  The objective is quadratic in the (linear) source, so the
    # central difference is exact up to single-precision noise.
    name = "source.1.mechanism"
    s_ref = problem.mechanism_scaling[name]
    assert s_ref == state.scaling[name]
    scales = [
        ControlStateFile.read(lin.job.state_output_file(task)).scaling[name]
        for task in (1, 2)
    ]
    assert scales[0] == s_ref and abs(scales[1] - scales[0]) > 1e-3 * s_ref, scales
    index = lin.space.slices[name].start  # Re of the first component
    x = problem.vector().values
    h = 0.05 * max(abs(float(x[index])), 1.0e-12)
    e = np.zeros_like(x)
    e[index] = h
    fd = (problem.value(x + e) - problem.value(x - e)) / (2.0 * h)
    converted = float(lin.gradient.values[index])
    # the pre-fix reduction: task parts summed in their own coordinates
    raw = sum(
        float(
            ControlVectorFile.read(lin.job.covector_file(task), native=False)[name][0]
        )
        for task in (1, 2)
    )
    print(
        f"mechanism FD check: fd={fd:.6e} converted={converted:.6e} "
        f"(rel {abs(converted - fd) / abs(fd):.2e}) unconverted={raw:.6e} "
        f"(rel {abs(raw - fd) / abs(fd):.2e}); s_task={scales}"
    )
    assert abs(converted - fd) <= 3.0e-2 * abs(fd), (fd, converted, raw)
    assert abs(raw - fd) > 3.0e-2 * abs(fd), (fd, converted, raw)


def test_source_positions_are_metres_in_sauce_and_km_in_the_simulation(tmp_path):
    """The fixture authors the source at (0.5, 0.08) km (Sauce's default unit)."""

    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="units", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)
    observed_job = FrequencyDomainJob("observed", truth, [FREQUENCY])
    assert site.run(observed_job, check=True).successful

    problem = ImagingProblem(
        initial,
        controls=ControlSpace(
            vp=DepthProfile("vp", "layer_2", count=4),
            src=SourceParameters(position=True, signature=False),
        ),
        observed=ObservedData(observed_job),
        site=site,
        name="units",
    )
    # Sauce's registry baseline (the state) is in metres
    np.testing.assert_allclose(problem.state["source.1.position"], [500.0, 80.0])
    assert problem.simulation_at().acquisition.to_fs() == (
        problem.simulation.acquisition.to_fs()
    )

    values = problem.vector().values.copy()
    sl = problem.space.slices["source.1.position"]
    values[sl] += [10.0, 5.0]  # metres
    moved = problem.simulation_at(values)
    np.testing.assert_allclose(
        moved.acquisition.source_point_coords(), [[0.51, 0.085]], rtol=1e-12
    )
    # the staged candidate is metres; Sauce accepts it (a km reading would
    # put the source 500 km outside the model)
    lin = problem.linearize(values)
    staged = ControlStateFile.read(lin.job.control_state)
    np.testing.assert_allclose(staged["source.1.position"], [510.0, 85.0])
    assert lin.jacobian.dot_test(seed=1, tolerance=3e-3)["passed"]
    direction = lin.space.random(4)
    np.testing.assert_allclose(
        lin.apply_normal(direction).values,
        lin.vjp(lin.jvp(direction)).values,
        rtol=3e-3,
        atol=1e-5,
    )
    assert np.isfinite(lin.value) and lin.value > 0.0


# ---------------------------------------------------------------------------
# auxiliary model extension (FWIME)
# ---------------------------------------------------------------------------


def _extension_simulation(project: Project, name: str, vp_lower: float):
    """Return the layered case configured for Sauce's extension actions.

    Mirrors ``test/e2e/model_extension_case.py`` (``control_sensitivity_case
    .prepare(..., accurate_condensation=True)``): compiled Forms, unrelaxed
    assembly with fp64 Schur condensation; Galerkin is the e2e default.
    """

    simulation = _simulation(project, name, vp_lower)
    simulation += Discretization(form_execution="compiled")
    simulation += SolverConfig(
        solve_on="final",
        max_iter=600,
        tolerance=1.0e-4,
        relaxed_assembly=False,
        schur_precision="fp64",
    )
    simulation.save()
    return simulation


@pytest.mark.timeout(1800)  # the extension actions take ~450 s
def test_extended_problem_solves_against_sauce(tmp_path):
    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="fwime", path=tmp_path / "project", load_if_exists=False)
    truth = _extension_simulation(project, "truth", TRUTH_VP)
    initial = _extension_simulation(project, "initial", START_VP)

    observed_job = FrequencyDomainJob("observed", truth, [FREQUENCY])
    result = site.run(observed_job, check=True)
    assert result.successful

    problem = ImagingProblem(
        initial,
        controls=DepthProfile("vp", "layer_2", count=4),
        observed=ObservedData(observed_job),
        site=site,
        name="fwime",
    )
    xp = problem.extend(
        Extension(
            [Lags("vp", count=5, origin=-20 * u.ms, spacing=10 * u.ms)],
            damping=0.1,
            lag_penalty=1.0,
            lag_scale=20 * u.ms,
            tolerance=1.0e-5,
            max_iterations=200,
        )
    )
    report = xp.capabilities()
    assert report["ok"], report
    assert xp.extension_space.size == 4 * 5

    lin = xp.linearize()
    assert lin.frequencies == [FREQUENCY]
    assert len(lin.manifests) == 1
    assert lin.manifests[0].baseline == lin.state_fingerprint
    assert np.isfinite(lin.baseline_value) and lin.baseline_value > 0.0
    assert lin.covector.size == 20 and np.all(np.isfinite(lin.covector.values))

    taps, solve_report = xp.solve()
    assert isinstance(taps, ExtensionVector) and taps.size == 20
    assert solve_report.converged, solve_report.raw
    assert solve_report.baseline == lin.state_fingerprint
    assert np.all(np.isfinite(taps.values)) and taps.norm() > 0.0
    assert solve_report.lag_scale_seconds == pytest.approx(0.02)

    value = xp.value()
    assert np.isfinite(value) and value > 0.0

    gradient = xp.gradient()
    assert gradient.size == lin.space.size
    assert np.all(np.isfinite(gradient.values))
    assert np.linalg.norm(gradient.values) > 0.0

    H = xp.normal()
    a = lin.space.random(11)
    b = lin.space.random(12)
    ha = H @ a
    hb = H @ b
    assert np.all(np.isfinite(ha.values)) and np.all(np.isfinite(hb.values))
    left = float(np.dot(ha.values, b.values))
    right = float(np.dot(a.values, hb.values))
    assert abs(left - right) <= 1.0e-2 * max(abs(left), abs(right)), (left, right)

    # An intentionally under-iterated solve must expose the failed task, even
    # though LocalSite's default policy tolerates a small number of failures.
    unfinished = problem.extend(
        Extension(
            [Lags("vp", count=5, origin=-20 * u.ms, spacing=10 * u.ms)],
            damping=0.1,
            lag_penalty=1.0,
            lag_scale=20 * u.ms,
            tolerance=1.0e-5,
            max_iterations=1,
        )
    )
    with pytest.raises(RuntimeError, match="has failed task"):
        unfinished.solve()


def test_named_terms_and_normalized_residual_on_two_mpi_ranks(tmp_path):
    """Saved term rows feed VJP under the baseline MPI partition."""
    import shutil

    from frequensolve.imaging import Misfit, Normalization, ObjectiveTerm
    from frequensolve.orchestrator.sites.local import LocalSite

    if shutil.which("mpirun") is None:
        pytest.skip("MPI launcher unavailable")
    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="terms", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)
    observed = FrequencyDomainJob("observed", truth, [FREQUENCY])
    assert site.run(observed, check=True).successful
    problem = ImagingProblem(
        initial,
        controls=DepthProfile("vp", "layer_2", count=4),
        observed=ObservedData(observed),
        site=site,
        name="terms",
        submit_options={"procs_per_job": 2},
        misfit=Misfit.terms(
            ObjectiveTerm("surface", id="amplitude", weight=0.3),
            ObjectiveTerm(
                "surface",
                id="energy",
                weight=2.0,
                normalization=Normalization(reduction="sum"),
            ),
        ),
    )
    lin = problem.linearize()
    assert lin.data_space.groups == ("amplitude", "energy")
    assert lin.objective_states[0].n_ranks == 2
    residual = lin.objective_residual()
    np.testing.assert_allclose(0.5 * residual.dot(residual), lin.value, rtol=1e-5)
    np.testing.assert_allclose(
        lin.vjp(residual).values, lin.gradient.values, rtol=3e-3, atol=1e-6
    )
    assert lin.jacobian.dot_test(seed=3, tolerance=3e-3)["passed"]
