# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Small 3D acoustic initialization, observations, mesh FWI and VTU acceptance."""

import dataclasses
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.mesh import BoundaryCondition
from frequensolve.model.layered import LayeredModel
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic import Acquisition, ReceiverNode
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


def _simulation(project, name, vp):
    simulation = project.new_simulation(name=name, physics="acoustic", dimension=3)
    model = LayeredModel(dimension=3, x_limits=[0, 1], y_limits=[0, 0.6])
    model.add_surface(name="top", depth=0)
    model.add_layer(name="sediment", properties={"vp": vp, "rho": 2.0})
    model.add_surface(name="bottom", depth=0.6)
    simulation += model
    simulation += model.hex_mesh_generator(n=[3, 2, 2])
    simulation.mesh.set_adapt(
        elems_per_wave=1.0, order=2, f_low=2, f_high=2, adapt_order=False
    )
    simulation += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    simulation += BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "y_min", "y_max", "z_max"],
        pml_wavelengths=0.5,
        pml_exponent=3.0,
        pml_constant=20.0,
    )
    acquisition = Acquisition()
    acquisition.add_sources(kind="scalar", coords=[[0.45, 0.3, 0.12]])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[x, 0.3, 0.08] for x in [0.2, 0.5, 0.8]],
    )
    simulation += acquisition
    simulation += Discretization()
    simulation += SolverConfig(solve_on="final", max_iter=300, tolerance=1e-4)
    simulation.save()
    return simulation


def test_3d_mesh_observations_gradient_transfer_and_vtu(tmp_path):
    pytest.importorskip("pyvista")
    solver = os.environ.get("FS_SAUCE_EXECUTABLE_3D")
    if not solver or not Path(solver).is_file():
        pytest.skip("set FS_SAUCE_EXECUTABLE_3D to a local 3D solver")
    project = Project(
        name="property_3d", path=tmp_path / "project", load_if_exists=False
    )
    truth = _simulation(project, "truth", 2.5)
    initial = _simulation(project, "initial", 2.2)
    with LocalSite(
        solver=solver,
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob("observed", truth, [2.0])
        assert site.run(observed, check=True).successful
        spec = im.MeshParameters(
            "vp",
            "sediment",
            frequency=2,
            epw=1,
            artifact=str(tmp_path / "initial.h5"),
            transform="log",
        )
        problem = im.ImagingProblem(
            initial,
            controls=im.ControlSpace(vp=spec),
            observed=im.ObservedData(observed),
            site=site,
            name="mesh_3d",
        )
        lin = problem.linearize()
        assert np.isfinite(lin.value) and lin.value > 0
        assert (
            np.isfinite(lin.gradient.values).all()
            and np.linalg.norm(lin.gradient.values) > 0
        )
        mesh = lin.gradient.to_mesh(spec.artifact, "vp")
        output = tmp_path / "gradient.vtu"
        mesh.save(output)
        import pyvista as pv

        restored = pv.read(output)
        assert restored.n_cells == mesh.n_cells > 0
        np.testing.assert_array_equal(restored.cells, mesh.cells)
        # Reference transfer remains valid when the source's optional plotting layer is absent.
        with h5py.File(spec.artifact, "a") as h5:
            for name in (
                "visualization_points_m",
                "visualization_kind",
                "visualization_offsets",
            ):
                del h5[f"property_space/{name}"]
        with pytest.raises(ValueError, match="visualization"):
            im.PropertyMesh.read(spec.artifact, material=1)
        old = problem.state
        adapted = problem.with_controls(
            {"vp": dataclasses.replace(spec, frequency=4, epw=2)}
        )
        assert adapted.full_space.full_size > old.space.full_size
        np.testing.assert_allclose(adapted.state.values, old.values[0], atol=1e-9)
