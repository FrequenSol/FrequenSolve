"""Solver-backed acceptance tests for joint background + coordinate reflectivity.

The 2D acoustic layered case of :mod:`tests.test_imaging_integration` is
solved with the settings of Sauce's ``joint-reflectivity-acoustic`` e2e case
(first-order acoustic DPG, compiled Forms, unrelaxed assembly, fp64 Schur
complements) and an ``ImagingProblem`` over ``DepthProfile("vp", ...)`` plus a
``ReflectivityParameters`` field borrowing that basis: the joint linearization
must carry both blocks, its Jacobian must satisfy the adjoint identity and its
Gauss-Newton normal must be symmetric.

Nonzero two-field states also check restricted/full prediction and derivative
equivalence. Fixed-background LSRTM recovers a weak impedance contrast from
nonlinear density data generated on a separate finer mesh, with the correct
velocity, known source, and noise-free pressure data.

The test needs a Sauce executable: ``FS_SAUCE_EXECUTABLE`` or
``LOCAL_SOLVER_EXECUTABLE`` in the environment, or the staged build at
``/tmp/FS_stage-imaging-merge/agent/install/fs2d_s``.  It skips otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from frequensolve.imaging import (
    LSRTM,
    ControlSpace,
    DepthProfile,
    ImagingProblem,
    ObservedData,
    ReflectivityField,
    ReflectivityParameters,
)
from frequensolve.imaging._artifacts import ControlStateFile
from frequensolve.mesh import BoundaryCondition
from frequensolve.model.layered import LayeredModel
from frequensolve.project import Project
from frequensolve.seismic import Acquisition, ReceiverNode
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig

pytestmark = pytest.mark.integration

DEFAULT_EXECUTABLE = Path("/tmp/FS_stage-imaging-merge/agent/install/fs2d_s")
FREQUENCY = 6.0
TRUTH_VP = 2.5
START_VP = 2.2
VP_COUNT = 4


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


def _simulation(project: Project, name: str, vp_lower: float):
    simulation = project.new_simulation(name=name, physics="acoustic", dimension=2)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer_1", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="interface", depth=0.25)
    model.add_layer(name="layer_2", properties={"vp": vp_lower, "rho": 2.2})
    model.add_surface(name="bottom", depth=0.6)
    simulation += model
    simulation += model.hex_mesh_generator(n=[8, 6])
    simulation.mesh.set_adapt(
        elems_per_wave=2.0, order=4, f_low=FREQUENCY, f_high=FREQUENCY, adapt_order=True
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
    # Sauce's joint reflectivity e2e fixture: first-order acoustic DPG with
    # compiled Forms, unrelaxed assembly and fp64 Schur complements so the
    # adjoint comparison is not limited by the local factors.
    simulation += Discretization(method="DPG", form_execution="compiled")
    simulation += SolverConfig(
        solve_on="final",
        max_iter=600,
        tolerance=1.0e-4,
        relaxed_assembly=False,
        schur_precision="fp64",
    )
    simulation.save()
    return simulation


@pytest.mark.parametrize("threads", [1, 16])
def test_joint_reflectivity_linearizes_against_sauce(tmp_path, threads):
    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(solver=_executable(), n_workers=1, threads_per_worker=threads)
    project = Project(name="imaging", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)

    observed_job = FrequencyDomainJob("observed", truth, [FREQUENCY])
    result = site.run(observed_job, check=True)
    assert result.successful

    controls = ControlSpace(
        vp=DepthProfile("vp", "layer_2", count=VP_COUNT),
        refl=ReflectivityParameters(
            "vp_ip", fields=[ReflectivityField("ip", layer=2, axis=2, basis="vp")]
        ),
    )
    problem = ImagingProblem(
        initial,
        controls=controls,
        observed=ObservedData(observed_job),
        site=site,
        name="joint",
    )
    assert problem.capabilities()["ok"], problem.capabilities()
    assert problem.space.blocks == ("model.vp", "reflectivity.ip")
    assert problem.space.sizes == {"model.vp": VP_COUNT, "reflectivity.ip": VP_COUNT}

    lin = problem.linearize()

    payload = lin.job.to_fs()["fwi_operator"]
    assert payload["controls"]["active"] == ["model.vp", "reflectivity.ip"]
    assert payload["reflectivity"] == {
        "parameterization": "vp_ip",
        "fields": [{"name": "ip", "layer": 2, "axis": 2, "basis": "vp"}],
    }
    assert np.isfinite(lin.value) and lin.value > 0.0
    assert lin.gradient is not None and lin.gradient.size == lin.space.size
    assert np.all(np.isfinite(lin.gradient.values))
    assert np.linalg.norm(lin.gradient["vp"]) > 0.0
    assert np.linalg.norm(lin.gradient["refl"]) > 0.0
    assert "reflectivity.ip" in lin.manifest.names
    assert lin.registry_fingerprint == lin.manifest.fingerprint

    # Sauce's complete baseline (exported by this single-task discovery job)
    # covers background and reflectivity; a borrowed basis starts at zero
    baseline = ControlStateFile.read(lin.job.state_output_file(1))
    assert {"model.vp", "reflectivity.ip"} <= set(baseline.names)
    np.testing.assert_array_equal(baseline["reflectivity.ip"], 0.0)
    assert baseline.support["reflectivity.ip"].shape == (VP_COUNT,)

    # <J dv, r>_Re == <dv, J^H r> through Sauce's jvp / vjp on the joint space
    report = lin.jacobian.dot_test(seed=1, tolerance=1.0e-3)
    assert report["passed"], report
    assert report["relative_error"] < 1.0e-3

    # the joint Gauss-Newton normal (with background-reflectivity cross
    # blocks) is symmetric
    N = lin.normal
    a, b = lin.space.random(2), lin.space.random(3)
    left, right = (N @ a).dot(b), (N @ b).dot(a)
    assert np.isfinite(left) and np.isfinite(right)
    assert abs(left - right) <= 1.0e-2 * max(abs(left), abs(right)), (left, right)
    assert np.linalg.norm((N @ a)["refl"]) > 0.0


@pytest.fixture(scope="module")
def recovery_case(tmp_path_factory):
    """Independent nonlinear density truth with a fixed, correct velocity model."""
    from frequensolve.orchestrator.sites.local import LocalSite

    root = tmp_path_factory.mktemp("reflectivity_recovery")
    site = LocalSite(solver=_executable(), n_workers=1, threads_per_worker=4)
    project = Project(name="recovery", path=root / "project", load_if_exists=False)
    frequencies = [4.0, 6.0, 8.0]
    nodes = np.linspace(0.25, 0.6, VP_COUNT)
    delta_rho = np.array([0.0, 0.06, -0.04, 0.0])

    def build(name, truth):
        sim = project.new_simulation(name=name, physics="acoustic", dimension=2)
        model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
        model.add_surface(name="top", depth=0.0)
        model.add_layer(name="water", properties={"vp": 1.5, "rho": 1.0})
        model.add_surface(name="seabed", depth=nodes[0])
        # Imaging sensitivities exclude PML cells. Keep the independent
        # nonlinear anomaly inside that same physical domain: a depth-only
        # xarray would extend the density perturbation into the lateral PML.
        # The very narrow outside transition lies below the mesh resolution.
        rho = (
            xr.DataArray(
                2.2 + np.outer([0.0, 1.0, 1.0, 0.0], delta_rho),
                dims=["x", "z"],
                coords={"x": [-1e-6, 0.0, 1.0, 1.0 + 1e-6], "z": nodes},
            )
            if truth
            else 2.2
        )
        model.add_layer(name="sediment", properties={"vp": START_VP, "rho": rho})
        model.add_surface(name="bottom", depth=nodes[-1])
        sim += model
        sim += model.hex_mesh_generator(n=[20, 15] if truth else [16, 12])
        sim.mesh.set_adapt(
            elems_per_wave=3.0,
            order=7 if truth else 6,
            f_low=min(frequencies),
            f_high=max(frequencies),
            adapt_order=True,
        )
        sim += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
        sim += BoundaryCondition(
            conditions=["pml"],
            boundaries=["x_min", "x_max", "z_max"],
            pml_wavelengths=1.2,
            pml_exponent=3.0,
            pml_constant=20.0,
        )
        acquisition = Acquisition()
        acquisition.add_sources(kind="scalar", coords=[[0.3, 0.08], [0.7, 0.08]])
        hydrophone = ReceiverNode(name="hydrophone")
        hydrophone.add_component(name="p", field="pressure")
        acquisition.add_receiver_group(
            name="surface",
            device=hydrophone,
            coords=[[x, 0.05] for x in np.linspace(0.1, 0.9, 17)],
        )
        sim += acquisition
        sim += Discretization(method="DPG", form_execution="compiled")
        sim += SolverConfig(
            solve_on="final",
            max_iter=600,
            tolerance=1.0e-4,
            relaxed_assembly=False,
            schur_precision="fp64",
        )
        sim.save()
        return sim

    truth = build("truth", True)
    initial = build("background", False)
    observed = FrequencyDomainJob("observed", truth, frequencies)
    assert site.run(observed, check=True).successful
    problem = ImagingProblem(
        initial,
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=VP_COUNT),
            refl=ReflectivityParameters(
                "vp_ip",
                fields=[
                    ReflectivityField("ip", layer=2, axis=2, basis="vp"),
                    ReflectivityField("dv", layer=2, axis=1, basis="vp"),
                ],
            ),
        ),
        observed=ObservedData(observed),
        site=site,
        name="reflectivity",
        # Compare several live views without evicting their staged state files.
        cache_capacity=8,
    )
    problem.linearize()
    yield root, problem, START_VP * delta_rho
    site.close()


@pytest.mark.timeout(1800)
def test_native_restrictions_preserve_nonzero_reflectivity(recovery_case):
    root, problem, expected = recovery_case
    baseline = problem.state
    point = baseline.with_update(
        problem.space.pack(
            {
                "vp": problem.vector()["vp"],
                "refl.ip": expected * 0.7,
                "refl.dv": np.array([0.0, 0.01, -0.01, 0.0]),
            }
        )
    )
    full = problem.linearize(point)
    prediction = full.simulated().values
    # Use raw arrays to express the same objective rows under distinct state fingerprints.
    dual = full.data_space.random(12).values
    full_pullback = full.jacobian.H @ dual
    errors = []
    for active in (["vp"], ["refl.ip"], ["refl.dv", "vp"]):
        view = problem.restrict(active=active)
        restricted = view.linearize(point)
        np.testing.assert_allclose(
            restricted.simulated().values, prediction, rtol=2e-5, atol=1e-10
        )
        assert restricted.value == pytest.approx(full.value, rel=2e-5)
        direction = restricted.space.random(17)
        embedded = full.space.pack(
            {
                name: (
                    direction[name]
                    if name in restricted.space.blocks
                    else np.zeros(full.space.sizes[name])
                )
                for name in full.space.blocks
            }
        )
        actual = (restricted.jacobian @ direction).values
        reference = (full.jacobian @ embedded).values
        error = np.linalg.norm(actual - reference) / max(
            np.linalg.norm(reference), 1e-20
        )
        assert error < 2e-3, (active, error)
        errors.append(float(error))
        pullback = restricted.jacobian.H @ dual
        for name in restricted.space.blocks:
            np.testing.assert_allclose(
                pullback[name], full_pullback[name], rtol=2e-3, atol=1e-10
            )
            np.testing.assert_allclose(
                restricted.gradient[name], full.gradient[name], rtol=2e-3, atol=1e-10
            )
        actual_normal = restricted.normal @ direction
        full_normal = full.normal @ embedded
        for name in restricted.space.blocks:
            np.testing.assert_allclose(
                actual_normal[name], full_normal[name], rtol=2e-3, atol=1e-10
            )
    (root / "restriction_metrics.json").write_text(
        json.dumps({"relative_jvp_errors": errors}, indent=2)
    )
    problem.state = baseline


@pytest.mark.timeout(1800)
def test_fixed_background_reflectivity_recovers_independent_truth(recovery_case):
    root, problem, expected = recovery_case
    baseline = problem.state
    view = problem.restrict(active=["refl.ip"])
    solver = LSRTM(view, method="lsqr", iterations=20, tolerance=1.0e-4)
    recovered = solver.run(baseline)
    final = view.linearize(baseline.with_update(recovered))
    error = np.linalg.norm(recovered.values - expected) / np.linalg.norm(expected)
    ratio = final.value / solver.linearization.value
    metrics = {
        "expected_ip_contrast": expected.tolist(),
        "recovered_ip_contrast": recovered.values.tolist(),
        "relative_model_error": float(error),
        "objective_ratio": float(ratio),
        "initial_objective": float(solver.linearization.value),
        "final_objective": float(final.value),
        "solver_converged": bool(solver.info["converged"]),
        "truth": "nonlinear density perturbation inside physical domain, finer independent mesh/order",
    }
    (root / "recovery_metrics.json").write_text(json.dumps(metrics, indent=2))
    assert solver.info["converged"], solver.info
    assert error < 0.10, metrics
    assert ratio < 0.02, metrics
    assert recovered.values[1] > 0 and recovered.values[2] < 0, metrics
    np.testing.assert_array_equal(final.state["vp"], baseline["vp"])
    np.testing.assert_array_equal(final.state["refl.dv"], baseline["refl.dv"])


@pytest.mark.timeout(1800)
def test_native_rwi_shared_fit_and_reduced_gradient(recovery_case):
    root, problem, expected = recovery_case
    # A smooth two-coefficient background and a separate four-coefficient image.
    problem = ImagingProblem(
        problem._shared.source_simulation,
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=2),
            refl=ReflectivityParameters(
                "vp_ip",
                fields=[
                    ReflectivityField(
                        "ip",
                        layer=2,
                        axis=2,
                        control=DepthProfile("ip", "sediment", count=VP_COUNT),
                    ),
                    ReflectivityField(
                        "dv",
                        layer=2,
                        axis=1,
                        control=DepthProfile("vp", "sediment", count=VP_COUNT),
                    ),
                ],
            ),
        ),
        observed=problem.observed_data,
        frequencies=problem.frequencies,
        site=problem.site,
        name="rwi-coarse",
        cache_capacity=8,
    )
    problem.state = problem.state.with_update(
        problem.full_space.restrict("refl").zeros()
    )
    rwi = problem.rwi(
        execution="native", reflectivity=["refl.ip"], damping=1e-8, cache_mb=0
    )
    lin = rwi.linearize()
    assert all(r["converged"] for r in lin.reports)
    np.testing.assert_allclose(lin.state["refl.ip"], expected, rtol=0.1, atol=0.007)
    assert all(r["incident_solves"] == 1 for r in lin.reports)
    assert all(r["cache_bytes"] == 0 for r in lin.reports)
    assert all(r["batch_replays"] > 1 for r in lin.reports)
    direction = rwi.space.pack({"vp": np.ones(2) * 0.1})
    x = lin.point - direction
    trial = rwi.linearize(x)
    h = 0.01
    fd = (rwi.value(x + h * direction) - rwi.value(x - h * direction)) / (2 * h)
    analytic = trial.gradient.dot(direction)
    assert abs(fd - analytic) <= 0.01 * max(abs(fd), abs(analytic)), (fd, analytic)
    from frequensolve.imaging import FWI, LBFGS, Stage

    rwi.state = problem.state.with_update(x)
    initial_error = np.linalg.norm(x["vp"] - lin.point["vp"])
    recovered = FWI(
        rwi,
        stages=[Stage(rwi.frequencies, iterations=8)],
        optimizer=LBFGS(
            gradient_tolerance=1e-4, objective_tolerance=0, step_tolerance=0
        ),
        scaling=None,
        step_limit=0.05,
    ).run(resume=False)
    final_error = np.linalg.norm(recovered.state["vp"] - lin.point["vp"])
    assert final_error < 0.5 * initial_error, (initial_error, final_error)
    assert rwi.value(recovered.state) < trial.value * 0.1
    (root / "rwi-validation.json").write_text(
        json.dumps(
            {
                "fd": fd,
                "gradient": analytic,
                "reports": lin.reports,
                "initial_error": initial_error,
                "final_error": final_error,
            },
            indent=2,
        )
    )


@pytest.mark.timeout(600)
def test_native_rwi_batched_storage_and_nonstationarity(recovery_case):
    _, problem, _ = recovery_case
    simulation = problem.simulation
    previous = simulation.acquisition.max_batch
    simulation.acquisition.max_batch = 1
    simulation.save()
    try:
        point = problem.state.with_update(
            problem.full_space.pack(
                {
                    "vp": np.zeros(VP_COUNT),
                    "refl.ip": np.zeros(VP_COUNT),
                    "refl.dv": np.zeros(VP_COUNT),
                }
            )
        )
        view = problem.restrict(frequencies=[FREQUENCY])
        memory = view.rwi(
            execution="native", reflectivity=["refl.ip"], damping=1e-8
        ).linearize(point)
        disk = view.rwi(
            execution="native", reflectivity=["refl.ip"], damping=1e-8, cache_mb=0
        ).linearize(point)
        np.testing.assert_allclose(
            memory.state["refl.ip"], disk.state["refl.ip"], rtol=1e-4, atol=1e-6
        )
        assert memory.value == pytest.approx(disk.value, rel=1e-4)
        # Iterative stopping bounds the vector residual, not the relative
        # error of individual near-zero gradient components.
        assert np.linalg.norm(
            memory.gradient.values - disk.gradient.values
        ) <= 1e-4 * max(
            np.linalg.norm(memory.gradient.values), np.linalg.norm(disk.gradient.values)
        )
        assert (
            memory.reports[0]["incident_solves"]
            == disk.reports[0]["incident_solves"]
            == 2
        )
        assert memory.reports[0]["cache_bytes"] > 0
        assert disk.reports[0]["cache_bytes"] == 0
        with pytest.raises(RuntimeError, match="not stationary"):
            view.rwi(
                execution="native",
                reflectivity=["refl.ip"],
                max_iterations=0,
                require_convergence=False,
            ).linearize(point)
    finally:
        simulation.acquisition.max_batch = previous
        simulation.save()


@pytest.mark.timeout(1800)
def test_sdk_rwi_matches_native_shared_fit_and_reduced_normal(recovery_case):
    root, problem, expected = recovery_case
    view = problem.restrict(weights=[0.5, 1.0, 2.0])
    portable = view.rwi(reflectivity=["refl.ip"], damping=1e-8)
    fit = portable.linearize()
    native = view.rwi(
        reflectivity=["refl.ip"], damping=1e-8, execution="native"
    ).linearize()
    np.testing.assert_allclose(fit.state["refl.ip"], expected, rtol=0.1, atol=0.007)
    np.testing.assert_allclose(
        fit.state["refl.ip"], native.state["refl.ip"], rtol=2e-3, atol=2e-5
    )
    np.testing.assert_allclose(fit.value, native.value, rtol=2e-3)
    np.testing.assert_allclose(
        fit.gradient.values, native.gradient.values, rtol=3e-3, atol=1e-8
    )
    np.testing.assert_array_equal(fit.state["refl.dv"], problem.state["refl.dv"])
    p = fit.space.random(123) * 0.01
    q = fit.space.random(456) * 0.01
    hp, hq = fit.apply_normal_batch([p, q])
    assert p.dot(hp) > 0
    np.testing.assert_allclose(p.dot(hq), q.dot(hp), rtol=2e-3, atol=1e-10)
    # At step=.01 the objective difference is comparable to the error from
    # production-tolerance wave solves (1e-4). Use two resolvable steps without
    # tightening production solver tolerances or relaxing the derivative check.
    differences = []
    for step in (0.1, 0.05):
        fd = (
            portable.value(fit.point + step * p) - portable.value(fit.point - step * p)
        ) / (2 * step)
        np.testing.assert_allclose(fd, fit.gradient.dot(p), rtol=1e-2, atol=1e-8)
        differences.append(fd)
    (root / "sdk-rwi-validation.json").write_text(
        json.dumps(
            dict(
                fit=fit.reports,
                responses=fit.response_reports,
                objective=fit.value,
                native_objective=native.value,
                finite_differences=differences,
                gradient=fit.gradient.dot(p),
            ),
            indent=2,
        )
    )
