"""Absolute acoustic volume-rate calibration against the 2D outgoing Green function."""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.special import hankel2

from frequensolve.mesh import BoundaryCondition
from frequensolve.model.layered import LayeredModel
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic import (
    Acquisition,
    GainDelay,
    PointSource,
    ReceiverNode,
    SourceGeometry,
    SourceSignature,
)
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from frequensolve.units import ureg as u

pytestmark = pytest.mark.integration


def _run(tmp_path, method, density, source_units):
    executable = os.environ.get("FS_SAUCE_EXECUTABLE")
    if not executable:
        pytest.skip("set FS_SAUCE_EXECUTABLE to a local solver")
    solver = Path(executable)
    project = Project(name="volume-rate", path=tmp_path / method)
    sim = project.new_simulation(name="acoustic", physics="acoustic", dimension=2)
    model = LayeredModel(dimension=2, x_limits=[0, 1])
    model.add_surface(name="top", depth=0)
    model.add_layer(name="fluid", properties={"vp": 1.5, "rho": density})
    model.add_surface(name="bottom", depth=1)
    sim += model
    sim += model.hex_mesh_generator(n=[8, 8])
    sim.mesh.set_adapt(elems_per_wave=2, order=7, f_low=4, f_high=4)
    sim += BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_min", "z_max"],
        pml_wavelengths=1.2,
        pml_exponent=3,
        pml_constant=20,
    )
    # Explicit inline point overrides support a mixed physical-source catalog.
    acq = Acquisition(
        source_geometry=SourceGeometry(
            kind="volume_injection",
            sources=[
                PointSource(
                    coords=[0.25, 0.5], kind="volume_injection", amplitude=source_units
                ),
                PointSource(
                    coords=[0.25, 0.5], kind="scalar", amplitude=2.0 * u.N * u.m
                ),
            ],
        )
    )
    frequencies = np.array([2.0, 4.0])
    signature = GainDelay(gain=0.7 + 0.2j, delay=0.037)
    acq.source_signature = SourceSignature(signature, frequencies=frequencies)
    node = ReceiverNode(
        name="hydrophone",
        components=[
            PointSource(coords=[0, 0], kind="volume_injection").reciprocal_receiver(
                physics="acoustic"
            )
        ],
    )
    acq.add_receiver_group(name="line", device=node, coords=[[0.45, 0.5], [0.65, 0.5]])
    sim += acq
    sim += Discretization(method=method)
    sim += SolverConfig(
        precision="single" if str(solver).endswith("_s") else "double",
        tolerance=1e-6 if str(solver).endswith("_s") else 1e-8,
        max_iter=300,
    )
    job = FrequencyDomainJob("volume-rate", sim, frequencies.tolist())
    result = LocalSite(solver=solver, n_workers=1, threads_per_worker=2).run(
        job, check=True
    )
    assert result.successful
    with result.traces() as traces:
        gather = traces.fd("line", "p", source=1).load()
    assert gather.attrs["source_strength_units"] == "m^3/s"
    np.testing.assert_allclose(gather.attrs["source_strength"], 0.0025, rtol=1e-6)
    omega = 2 * np.pi * frequencies
    # Planar 2D spreads Q over a fixed 1 m thickness, giving line rate Q / 1 m.
    # exp(+i omega t): outgoing H_0^(2), pressure = rho*omega*(Q / 1 m)/4 * H_0^(2).
    expected = (
        density
        * 1000
        * omega[:, None]
        * 0.0025
        / 4
        * hankel2(0, omega[:, None] / 1500 * np.array([200, 400]))
    )
    expected *= signature.at_frequencies(frequencies)[:, None]
    actual = gather.transpose("frequency", "receiver").values
    np.testing.assert_allclose(actual, expected, rtol=0.015, atol=1e-5)
    return actual


@pytest.mark.parametrize("density", [1.0, 2.2])
def test_absolute_amplitude_phase_and_formulation_agreement(tmp_path, density):
    first = _run(tmp_path, "Galerkin", density, 0.0025 * u.m**3 / u.s)
    second = _run(tmp_path, "DPG", density, 2500 * u.cm**3 / u.s)
    np.testing.assert_allclose(first, second, rtol=0.003, atol=1e-5)
