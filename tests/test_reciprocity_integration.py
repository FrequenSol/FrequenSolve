"""Pressure-source/hydrophone reciprocity through the public FrequenSolve API.

Run with::

    FS_SAUCE_EXECUTABLE=/path/to/fs2d python -m pytest -m integration \
        tests/test_reciprocity_integration.py

Supports single- or double-precision local solvers. Both density cases use
Galerkin and DPG, including amplitude-preserving time reconstruction.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from frequensolve.mesh import BoundaryCondition
from frequensolve.model.layered import LayeredModel
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic import Acquisition, ReceiverNode
from frequensolve.seismic.wavelet import RickerWavelet
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from frequensolve.units import ureg as u

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("method", ["Galerkin", "DPG"])
@pytest.mark.parametrize("lower_density", [1.0, 2.2])
def test_pressure_hydrophone_reciprocity(tmp_path, lower_density, method):
    executable = os.environ.get("FS_SAUCE_EXECUTABLE") or os.environ.get(
        "LOCAL_SOLVER_EXECUTABLE"
    )
    if not executable:
        pytest.skip("set FS_SAUCE_EXECUTABLE to a double-precision local solver")
    solver = Path(executable).expanduser()
    assert solver.is_file() and os.access(solver, os.X_OK), solver

    project = Project(name="reciprocity", path=tmp_path / "project")
    sim = project.new_simulation(name="acoustic", physics="acoustic", dimension=2)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="interface", depth=0.25)
    model.add_layer(name="lower", properties={"vp": 2.5, "rho": lower_density})
    model.add_surface(name="bottom", depth=0.6)
    sim += model
    sim += model.hex_mesh_generator(n=[6, 4])
    sim.mesh.set_adapt(
        elems_per_wave=2.0, order=6, f_low=6.0, f_high=6.0, adapt_order=True
    )
    sim += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=1.2,
        pml_exponent=3.0,
        pml_constant=20.0,
    )
    # Both shots use the same operator and physical strength. The off-diagonal
    # samples exchange source/receiver locations across the material interface.
    points = [[0.23, 0.10], [0.73, 0.41]]
    acq = Acquisition()
    acq.add_sources(
        kind="volume_injection", coords=points, amplitude=1.0 * u.m**3 / u.s
    )
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.components.append(
        acq.source_geometry.reciprocal_receiver(physics="acoustic")
    )
    acq.add_receiver_group(name="pair", device=hydrophone, coords=points)
    sim += acq
    sim += Discretization(method=method)
    sim += SolverConfig(
        precision="single" if str(solver).endswith("_s") else "double",
        tolerance=1e-6 if str(solver).endswith("_s") else 1e-8,
        max_iter=300,
    )
    job = FrequencyDomainJob("reciprocity", sim, [2.0, 4.0, 6.0])
    site = LocalSite(solver=solver, n_workers=1, threads_per_worker=2)
    result = site.run(job, check=True)
    assert result.successful
    with result.traces() as traces:
        # Equal volume-rate sources pair directly with pressure, including
        # across density contrasts. No density or fitted amplitude correction.
        forward = traces.fd("pair", "p", source=1).isel(receiver=1).values
        reverse = traces.fd("pair", "p", source=2).isel(receiver=0).values
        _assert_reciprocal(forward, reverse)

        forward_td = traces.td(
            "pair", "p", source=1, wavelet=RickerWavelet(f=2.0), upscale=1
        ).isel(receiver=1)
        for upscale in (1, 4):
            reverse_td = traces.td(
                "pair", "p", source=2, wavelet=RickerWavelet(f=2.0), upscale=upscale
            ).isel(receiver=0, time=slice(None, None, upscale))
            np.testing.assert_allclose(forward_td.time, reverse_td.time, atol=1e-14)
            _assert_reciprocal(forward_td.values, reverse_td.values)


def _assert_reciprocal(forward, reverse):
    assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse))
    norm = np.linalg.norm(forward)
    assert norm > 0.0
    relative_error = np.linalg.norm(forward - reverse) / norm
    assert relative_error < 2e-4, f"reciprocity relative error: {relative_error:.6g}"
