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

from frequensolve.imaging import DepthProfile, ImagingProblem, ObservedData
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
    report = lin.jacobian.dot_test(seed=1, tolerance=1.0e-3)
    assert report["passed"], report
    assert report["relative_error"] < 1.0e-3

    # the value at the current state is cached; a moved point is not
    assert problem.value() == lin.value
    moved = problem.value(problem.vector() + 0.01)
    assert np.isfinite(moved) and moved != lin.value
