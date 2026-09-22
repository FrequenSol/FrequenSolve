"""Solver-backed acceptance test for joint background + coordinate reflectivity.

The 2D acoustic layered case of :mod:`tests.test_imaging_integration` is
solved with the settings of Sauce's ``joint-reflectivity-acoustic`` e2e case
(first-order acoustic DPG, compiled Forms, unrelaxed assembly, fp64 Schur
complements) and an ``ImagingProblem`` over ``DepthProfile("vp", ...)`` plus a
``ReflectivityParameters`` field borrowing that basis: the joint linearization
must carry both blocks, its Jacobian must satisfy the adjoint identity and its
Gauss-Newton normal must be symmetric.

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


def test_joint_reflectivity_linearizes_against_sauce(tmp_path):
    from frequensolve.orchestrator.sites.local import LocalSite

    site = LocalSite(solver=_executable(), n_workers=1)
    project = Project(name="imaging", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", TRUTH_VP)
    initial = _simulation(project, "initial", START_VP)

    observed_job = FrequencyDomainJob("observed", truth, [FREQUENCY])
    result = site.run(observed_job, check=True)
    assert result.successful

    # The basis is authored on the global z axis like Sauce's joint e2e
    # fixture: Sauce 5e07624 evaluates a borrowed (or own) reflectivity map
    # without its surface-coordinate context, so a ``datum="top"`` profile stops
    # with "Surface-coordinate control map has no evaluation context".
    controls = ControlSpace(
        vp=DepthProfile("vp", "layer_2", datum="global", count=VP_COUNT),
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
    baseline = ControlStateFile.read(lin.job.state_output_file())
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
