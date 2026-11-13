"""Recorded source strength explains trace amplitudes, through the public API.

Run with::

    FS_SAUCE_EXECUTABLE=/path/to/fs2d python -m pytest -m integration \
        tests/test_source_strength_integration.py
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
from frequensolve.simulation import FrequencyDomainJob
from frequensolve.units import ureg as u

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "kind,units,default", [("scalar", "N*m", 1e9), ("volume_injection", "m^3/s", 1.0)]
)
def test_default_and_specified_strengths_are_recorded(tmp_path, kind, units, default):
    executable = os.environ.get("FS_SAUCE_EXECUTABLE") or os.environ.get(
        "LOCAL_SOLVER_EXECUTABLE"
    )
    if not executable:
        pytest.skip("set FS_SAUCE_EXECUTABLE to a local 2D solver")
    solver = Path(executable).expanduser()
    assert solver.is_file() and os.access(solver, os.X_OK), solver

    project = Project(name="strength", path=tmp_path / "project")
    sim = project.new_simulation(name="acoustic", physics="acoustic", dimension=2)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="water", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="bottom", depth=0.6)
    sim += model
    sim += model.hex_mesh_generator(n=[6, 4])
    sim.mesh.set_adapt(elems_per_wave=2.0, order=5, f_low=4.0, f_high=4.0)
    sim += BoundaryCondition(
        conditions=["pml"], boundaries=["x_min", "x_max", "z_min", "z_max"]
    )
    # Three co-located shots differ only in how their strength is authored.
    shot = [[0.3, 0.3]]
    acq = Acquisition()
    acq.add_sources(kind=kind, coords=shot)
    acq.add_sources(kind=kind, coords=shot, amplitude=2.0 * u(units))
    acq.add_sources(kind=kind, coords=shot, amplitude=3.0)
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="line", device=hydrophone, coords=[[0.7, 0.2]])
    sim += acq
    job = FrequencyDomainJob("strength", sim, [3.0, 4.0])
    site = LocalSite(solver=solver, n_workers=1, threads_per_worker=2)
    result = site.run(job, check=True)
    assert result.successful

    expected = [
        (default, "default"),
        (2.0, "specified"),
        (3 * default, "scaled_default"),
    ]
    with result.traces() as traces:
        gathers = [traces.fd("line", "p", source=i).load() for i in (1, 2, 3)]
    for gather, (strength, origin) in zip(gathers, expected):
        assert gather.attrs["units"] == "Pa"
        assert gather.attrs["source_kind"] == kind
        assert gather.attrs["source_strength_units"] == units
        assert gather.attrs["source_strength_origin"] == origin
        np.testing.assert_allclose(
            gather.attrs["source_strength"], strength, rtol=1e-12
        )
    # Dividing by the recorded strength yields one per-unit-source response.
    unit = [g.values / g.attrs["source_strength"] for g in gathers]
    assert np.linalg.norm(unit[0]) > 0.0
    for response in unit[1:]:
        np.testing.assert_allclose(response, unit[0], rtol=1e-4)
