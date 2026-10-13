"""Elastic shared-frequency RWI recovery and reduced operator acceptance."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from frequensolve import imaging as im
from frequensolve.mesh import BoundaryCondition
from frequensolve.model.layered import LayeredModel
from frequensolve.project import Project
from frequensolve.seismic import Acquisition, ReceiverNode
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from tests.test_imaging_reflectivity_integration import _executable

pytestmark = pytest.mark.integration


def _case(
    root,
    site,
    anisotropy,
    chart,
    *,
    zero_residual=False,
    wave_tolerance=1e-4,
    independent_mesh=True,
):
    project = Project(name="elastic_rwi", path=root / "project", load_if_exists=False)
    frequencies = [2.0, 3.0]
    nodes = np.linspace(0.2, 0.6, 3)
    # Independently perturb the final coordinate in nonlinear truth, rather
    # than synthesizing observations with the same Born operator being tested.
    contrast = np.array([0.0, 0.03, 0.0])
    vp, vs, rho = 2.5, 1.3, 2.2

    def build(name, truth):
        sim = project.new_simulation(name=name, physics="elastic", dimension=2)
        model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
        model.add_surface(name="top", depth=0.0)
        extra = (
            dict(epsilon=0.18, gamma=0.09, delta=0.05, theta=0.37, phi=-0.29)
            if anisotropy == "tti"
            else {}
        )
        model.add_layer(
            name="upper",
            physics=f"elastic:{anisotropy}",
            properties=dict(vp=2.0, vs=1.0, rho=2.0, **extra),
        )
        model.add_surface(name="interface", depth=nodes[0])
        props = dict(vp=vp, vs=vs, rho=rho, **extra)
        if truth:
            perturbation = xr.DataArray(
                np.outer([0.0, 1.0, 1.0, 0.0], contrast),
                dims=["x", "z"],
                coords={"x": [-1e-6, 0.0, 1.0, 1.0 + 1e-6], "z": nodes},
            )
            if chart == "vp_vs_ip":
                props["rho"] = rho + perturbation / vp
            else:
                props["rho"] = rho + perturbation
                props["vp"] = rho * vp / props["rho"]
                props["vs"] = rho * vs / props["rho"]
        model.add_layer(
            name="target", physics=f"elastic:{anisotropy}", properties=props
        )
        model.add_surface(name="bottom", depth=nodes[-1])
        sim += model
        finer = truth and independent_mesh
        sim += model.hex_mesh_generator(n=[20, 15] if finer else [16, 12])
        sim.mesh.set_adapt(
            elems_per_wave=3.0,
            order=8 if finer else 7,
            f_low=min(frequencies),
            f_high=max(frequencies),
            adapt_order=True,
        )
        sim += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
        sim += BoundaryCondition(
            conditions=["impedance"], boundaries=["x_min", "x_max", "z_max"]
        )
        acquisition = Acquisition()
        # Keep the zero-residual derivative check's near-source samples. For
        # independent-mesh recovery, separate sources and receivers so local
        # point-source discretization error does not dominate weak reflections.
        sources = [0.3, 0.7] if zero_residual else [0.4, 0.6]
        receiver_x = (
            np.linspace(0.1, 0.9, 17)
            if zero_residual
            else [0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]
        )
        acquisition.add_sources(
            kind="vector", coords=[[x, 0.08] for x in sources], direction=[0.6, 1.0]
        )
        receivers = ReceiverNode(name="geophone")
        receivers.add_component(name="vx", field="velocity", direction=[1.0, 0.0])
        receivers.add_component(name="vz", field="velocity", direction=[0.0, 1.0])
        acquisition.add_receiver_group(
            name="surface",
            device=receivers,
            coords=[[x, 0.05] for x in receiver_x],
        )
        sim += acquisition
        sim += Discretization(method="DPG", form_execution="compiled")
        sim += SolverConfig(
            solve_on="final",
            max_iter=600,
            tolerance=wave_tolerance,
            relaxed_assembly=False,
            schur_precision="fp64",
        )
        sim.save()
        return sim

    truth, background = build("truth", True), build("background", False)
    observed = FrequencyDomainJob(
        "observed", background if zero_residual else truth, frequencies
    )
    assert site.run(observed, check=True).successful
    props = (
        ["vp", "vs", "rho", "epsilon", "theta"]
        if anisotropy == "tti"
        else ["vp", "vs", "rho"]
    )
    names = ["dv", "ds", "ip"] if chart == "vp_vs_ip" else ["ip", "is", "drho"]
    problem = im.ImagingProblem(
        background,
        controls=im.ControlSpace(
            **{prop: im.DepthProfile(prop, "target", count=3) for prop in props},
            refl=im.ReflectivityParameters(
                chart,
                fields=[
                    im.ReflectivityField(name, layer=2, axis=i + 1, basis="vp")
                    for i, name in enumerate(names)
                ],
            ),
        ),
        observed=im.ObservedData(observed),
        misfit=im.Misfit(normalization="observed_rms"),
        site=site,
        name="rwi",
        cache_capacity=16,
    ).restrict(weights=[0.5, 1.5])
    return problem, names[-1], contrast


@pytest.mark.parametrize("anisotropy", ["iso", "tti"])
@pytest.mark.parametrize("chart", ["vp_vs_ip", "ip_is_rho"])
@pytest.mark.timeout(1800)
def test_elastic_rwi_recovery_and_reduced_actions(tmp_path, anisotropy, chart):
    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=4,
        shutdown_on_completion=False,
    )
    try:
        problem, inner, expected = _case(tmp_path, site, anisotropy, chart)
        rwi = problem.rwi(reflectivity=f"refl.{inner}", damping=1e-8)
        baseline = problem.linearize()
        fit = rwi.linearize()
        metrics = dict(
            initial_value=baseline.value,
            fitted_value=fit.value,
            expected=expected.tolist(),
            recovered=fit.state[f"refl.{inner}"].tolist(),
            fit=fit.reports,
        )
        path = tmp_path / "elastic-rwi-metrics.json"
        path.write_text(json.dumps(metrics, indent=2))
        np.testing.assert_allclose(
            fit.state[f"refl.{inner}"], expected, rtol=0.15, atol=0.003
        )
        assert np.linalg.norm(
            fit.state[f"refl.{inner}"] - expected
        ) < 0.15 * np.linalg.norm(expected)
        assert fit.value < 0.1 * baseline.value
        # Independent meshes leave a direct-field discretization residual that
        # reflectivity cannot remove. Compare against the known-contrast Born
        # prediction as well as checking the physical coefficients themselves.
        truth_vector = problem.full_space.pack(
            {
                name: (
                    expected
                    if name == f"reflectivity.{inner}"
                    else baseline.state[name]
                )
                for name in problem.full_space.blocks
            }
        )
        oracle = problem.linearize(baseline.state.with_update(truth_vector))
        metrics["known_contrast_value"] = oracle.value
        path.write_text(json.dumps(metrics, indent=2))
        assert oracle.value < baseline.value
        assert baseline.value - fit.value >= 0.95 * (baseline.value - oracle.value)
        p = fit.space.random(123) * 0.01
        q = fit.space.random(456) * 0.01
        hp, hq = fit.apply_normal_batch([p, q])
        assert p.dot(hp) > 0
        np.testing.assert_allclose(p.dot(hq), q.dot(hp), rtol=3e-3, atol=1e-10)
        metrics.update(symmetry=[p.dot(hq), q.dot(hp)], energy=p.dot(hp))
        # Separate checks prevent a larger velocity sensitivity from hiding a
        # missing anisotropy/orientation mixed contribution.
        directions = {"joint": p}
        if chart == "ip_is_rho":
            # This fit removes >99.9% of the data objective. Random projections
            # can nearly cancel, below the resolution of 1e-4 wave solves. Use
            # a resolvable descent direction; the zero-residual test separately
            # compares every gradient component with a random Schur action.
            directions["joint"] = fit.gradient * (
                0.01 / np.linalg.norm(fit.gradient.values)
            )
        if anisotropy == "tti":
            for prop in ("epsilon", "theta"):
                block = f"model.{prop}"
                values = np.full(fit.space.sizes[block], 0.01)
                if chart == "ip_is_rho":
                    # Avoid cancellation within this nearly stationary block
                    # as well as between distinct material properties.
                    values = fit.gradient[block]
                    values = values * (0.01 / np.linalg.norm(values))
                directions[prop] = fit.space.pack(
                    {
                        name: (
                            values if name == block else np.zeros(fit.space.sizes[name])
                        )
                        for name in fit.space.blocks
                    }
                )
        metrics["derivatives"] = {}
        for name, direction in directions.items():
            analytic = fit.gradient.dot(direction)
            differences = []
            for step in (0.1, 0.05):
                fd = (
                    rwi.value(fit.point + step * direction)
                    - rwi.value(fit.point - step * direction)
                ) / (2 * step)
                differences.append(fd)
            metrics["derivatives"][name] = dict(analytic=analytic, fd=differences)
            path.write_text(json.dumps(metrics, indent=2))
            np.testing.assert_allclose(differences, analytic, rtol=0.02, atol=1e-8)
        metrics["responses"] = fit.response_reports
        path.write_text(json.dumps(metrics, indent=2))
    finally:
        site.close()


@pytest.mark.timeout(1800)
def test_tti_all_fields_reduced_curvature_at_zero_residual(tmp_path):
    """At zero residual the reduced gradient derivative equals the GN Schur action."""
    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=4,
        shutdown_on_completion=False,
    )
    try:
        problem, _, _ = _case(tmp_path, site, "tti", "ip_is_rho", zero_residual=True)
        # An absolute floor is needed for fitting numerical noise at the exact
        # zero-residual solution; keep the normal production relative tolerance.
        rwi = problem.rwi(damping=1e-4, absolute_tolerance=1e-7)
        fit = rwi.linearize()
        assert fit.value < 1e-12
        p = fit.space.random(817) * 0.01
        hp = fit.normal @ p
        assert p.dot(hp) > 0
        errors = []
        for step in (0.1, 0.05):
            plus = rwi.linearize(fit.point + step * p)
            minus = rwi.linearize(fit.point - step * p)
            fd = (plus.gradient.values - minus.gradient.values) / (2 * step)
            errors.append(
                float(np.linalg.norm(fd - hp.values) / np.linalg.norm(hp.values))
            )
        (tmp_path / "tti-schur-metrics.json").write_text(
            json.dumps(
                dict(value=fit.value, errors=errors, response=fit.response_reports),
                indent=2,
            )
        )
        assert max(errors) < 0.02, errors
    finally:
        site.close()
