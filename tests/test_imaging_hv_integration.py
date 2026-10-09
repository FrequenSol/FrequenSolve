"""HV first-order SDK path against a native acoustic material derivative."""

import numpy as np
import pytest
import xarray as xr
from frequensolve.model.layered import LayeredModel

from frequensolve import imaging as im
from frequensolve.mesh import BoundaryCondition, BoundaryConditions
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from frequensolve.units import ureg
from tests.test_imaging_integration import _executable, _simulation

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@pytest.mark.parametrize("curved,meshed", [(False, False), (True, False), (True, True)])
def test_hv_native_gradient_and_frequency_weight(tmp_path, curved, meshed):
    project = Project(name="hv", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", 2.5)
    initial = _simulation(project, "initial", 2.2)
    for sim in (truth, initial):
        if curved:
            # HV must retain authored line coordinates through the metadata/root-map
            # shortcut. Reconstructing them from curved-element eta perturbs the line.
            model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
            model.add_surface(name="top", depth=0.0)
            model.add_layer(name="layer_1", properties={"vp": 1.5, "rho": 1.0})
            x = np.linspace(0, 1, 101)
            depth = xr.DataArray(
                0.25 + 0.05 * np.sin(2 * np.pi * x), dims="x", coords={"x": x}
            )
            depth.attrs["units"] = "km"
            depth.x.attrs["units"] = "km"
            model.add_surface(name="interface", depth=depth)
            model.add_layer(
                name="layer_2",
                properties={"vp": 2.5 if sim is truth else 2.45, "rho": 2.2},
            )
            model.add_surface(name="bottom", depth=0.6)
            sim += model
            sim += model.hex_mesh_generator(n=[8, 6])
        sim.BCs = BoundaryConditions(
            [
                BoundaryCondition(
                    conditions=["free"], boundaries=["x_min", "x_max", "z_min", "z_max"]
                )
            ]
        )
        sim.mesh.set_adapt(elems_per_wave=0.05, order=3, adapt_order=False)
        sim += Discretization(method="Galerkin", form_execution="compiled")
        sim += SolverConfig(
            grids=1,
            tolerance=1e-10,
            relaxed_assembly=False,
            schur_precision="fp64",
            solve_precision="fp64",
            workspace_precision="fp64",
            operator_precision="fp64",
            mumps_precision="double",
        )
        sim.save()
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob("observed", truth, [5.0])
        assert site.run(observed, check=True).successful
        base = im.ImagingProblem(
            initial,
            controls=(
                im.MeshParameters(
                    "vp",
                    "layer_2",
                    frequency=5,
                    epw=1,
                    artifact=str(project.path / "vp.h5"),
                )
                if meshed
                else (
                    im.DepthProfile(
                        "vp",
                        "layer_2",
                        datum="global",
                        nodes=np.array([0.2, 0.4, 0.6]) * ureg.km,
                    )
                    if curved
                    else im.DepthProfile("vp", "layer_2", count=3)
                )
            ),
            observed=im.ObservedData(observed, source_basis="source_geometry"),
            site=site,
            workdir=tmp_path / "jobs",
            name="hv",
        )
        hv = base.hv(
            im.HV(
                0.1 * ureg.Pa,
                kappa=1.0,
                relative_mesh_tolerance=0.1,
                max_refinements=8,
                max_iterations=600,
                relative_energy_tolerance=1e-8,
                relative_stationarity_tolerance=1e-6,
            )
        )
        x = hv.vector().values.copy()
        lin = hv.linearize(x)
        assert lin.value > 0
        assert np.isfinite(lin.gradient.values).all()
        direction = (
            0.7 * np.cos(np.linspace(0, 2, len(x)))
            if meshed
            else np.array([0.7, -0.4, 0.2])
        )
        step = 1e-3
        fd = (hv.value(x + step * direction) - hv.value(x - step * direction)) / (
            2 * step
        )
        print(
            "HV value, pairing, finite difference:",
            lin.value,
            lin.gradient.values @ direction,
            fd,
        )
        np.testing.assert_allclose(
            lin.gradient.values @ direction, fd, rtol=3e-3, atol=1e-7
        )
        weighted = hv.restrict(weights=[0.3]).linearize(x)
        np.testing.assert_allclose(weighted.value, 0.3 * lin.value, rtol=1e-8)
        np.testing.assert_allclose(
            weighted.gradient.values, 0.3 * lin.gradient.values, rtol=1e-7, atol=1e-10
        )
        with pytest.raises(NotImplementedError, match="first-order"):
            _ = lin.normal
