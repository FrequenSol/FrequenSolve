"""Small public-API 3D acoustic acceptance case; native execution is opt-in."""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import frequensolve as fs
from frequensolve.mesh import BoundaryCondition
from frequensolve.mesh.mesh_generators import LayeredMeshGenerator
from frequensolve.model.layered import LayeredModel
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverNode
from frequensolve.simulation.discretization import Discretization
from frequensolve.simulation.jobs import BaseJob, FrequencyDomainJob
from frequensolve.simulation.solver import SolverConfig
from frequensolve.solver import (
    SolverCompatibilityWarning,
    query_local_solver_identity,
)


def _author_3d_job(root):
    project = fs.Project(name="native_3d", path=root, load_if_exists=False)
    sim = project.new_simulation(name="acoustic_3d", physics="acoustic", dimension=3)
    sim.units.defaults.update(length="km", velocity="km/s", density="g/cm^3")
    model = LayeredModel(dimension=3, x_limits=[0.0, 1.0], y_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="homogeneous", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="bottom", depth=0.5)
    sim += model
    sim += LayeredMeshGenerator(
        l_bound=[0.0, 0.0, 0.0], u_bound=[1.0, 1.0, 0.5], n=[4, 4, 1]
    )
    sim.mesh.set_adapt(
        elems_per_wave=2.0, order=2, f_low=2.0, f_high=2.0, adapt_order=False
    )
    sim += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "y_min", "y_max", "z_max"],
        pml_wavelengths=0.5,
        pml_exponent=3.0,
        pml_constant=20.0,
        pml_reflection=0.001,
        stretch_limit=0.25,
    )
    coordinates = [[0.25, 0.5, 0.2], [0.75, 0.5, 0.2]]
    acquisition = Acquisition()
    acquisition.add_sources(coords=coordinates, kind="scalar")
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="reciprocal", device=hydrophone, coords=coordinates
    )
    sim += acquisition
    sim += SolverConfig(
        solve_on="final", coarse_solver="MUMPS", max_iter=300, tolerance=1e-6, grids=1
    )
    sim += Discretization(method="DPG")
    return FrequencyDomainJob(name="frequency", simulation=sim, f_list=[2.0])


def test_public_3d_authoring_roundtrips(tmp_path):
    job = _author_3d_job(tmp_path / "project")
    assert job.validate().issues == []
    loaded = BaseJob.load(job.save())
    assert loaded.simulation.dimension == 3
    assert loaded.simulation.physics == "acoustic"
    assert loaded.simulation.units.defaults["length"] == "km"
    assert loaded.f_list == [2.0]
    assert loaded.validate().issues == []
    contract_root = (
        Path(__file__).parent / "contracts" / "sauce-a54bdda" / "trunk" / "contracts"
    )
    registry = Registry()
    for schema_file in contract_root.rglob("*.json"):
        schema = json.loads(schema_file.read_text())
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    simulation = json.loads(loaded.simulation.save().read_text())
    for contract, payload in [
        ("fs-simulation-1", simulation),
        ("fs-acquisition-2", simulation["Acquisition"]),
    ]:
        schema = json.loads(
            (contract_root / "inputs" / contract / "schema.json").read_text()
        )
        Draft202012Validator(schema, registry=registry).validate(payload)


@pytest.mark.integration
@pytest.mark.timeout(420)
def test_native_3d_pressure_reciprocity(tmp_path):
    """Run init/task/pack and read packed results through the public SDK."""
    raw_solver = os.environ.get("LOCAL_SOLVER_EXECUTABLE")
    if not raw_solver:
        pytest.fail("LOCAL_SOLVER_EXECUTABLE is required for native 3D acceptance")
    solver = Path(raw_solver).expanduser().resolve()
    assert solver.is_file() and os.access(solver, os.X_OK)
    identity = query_local_solver_identity(solver)
    assert identity.identity is not None, identity.error
    job = _author_3d_job((tmp_path / "project").resolve())
    assert job.validate().issues == []
    site = LocalSite(
        solver=solver,
        n_workers=1,
        threads_per_worker=1,
        memory_per_worker=512,
        shutdown_on_completion=True,
        solver_policy="warn",
    )
    # Keep the production compatibility policy enabled. An unreleased checkout
    # may declare no immutable pair; retain that warning in the receipt, while
    # unknown identities and mismatched declared releases still fail this test.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SolverCompatibilityWarning)
        result = site.run(job, force=True, timeout=300, check=True)
    compatibility_warnings = [str(w.message) for w in caught]
    assert all(
        "does not contain a released preferred FrequenSolver declaration." in w
        for w in compatibility_warnings
    )
    assert result.status.state == "completed"
    loaded = BaseJob.load(job.job_file)
    manifest = loaded.run_metadata.manifest
    assert manifest["build"]["git_commit"] == identity.identity.git_commit
    assert len(manifest["build"]["git_commit"]) == 40
    assert manifest["build"]["git_dirty"] is False
    assert manifest["workflow"]["dimension"] == 3
    assert manifest["workflow"]["physics"] == "acoustic"
    assert manifest["workflow"]["finalized_by"] == "pack"
    assert manifest["task_summary"]["succeeded"] == 1
    assert manifest["task_summary"]["failed"] == 0
    assert manifest["solver"]["convergence"]["converged"] is True
    assert Path(manifest["solver"]["executable"]).name == "fs3d_s"
    with loaded.traces.open() as traces:
        assert traces.groups == ["reciprocal"]
        data = traces.open_frequency_domain("reciprocal").compute()
        assert dict(data.sizes) == {
            "frequency": 1,
            "source": 2,
            "component": 1,
            "receiver": 2,
            "complex": 2,
        }
        assert data.coords["component"].values.tolist() == ["p"]
        np.testing.assert_allclose(data.coords["frequency"], [2.0])
        survey = traces.survey_tables()
        # The authored km coordinates are normalized to metres in packed survey metadata.
        expected_coordinates = [[250.0, 750.0], [500.0, 500.0], [200.0, 200.0]]
        np.testing.assert_allclose(
            survey["sources"]["coordinates"], expected_coordinates
        )
        np.testing.assert_allclose(
            survey["receivers"]["coordinates"], expected_coordinates
        )
        pressure = data.sel(frequency=2.0, component="p")
        values = (
            pressure.sel(complex="real").values
            + 1j * pressure.sel(complex="imag").values
        )
        assert np.isfinite(values).all()
        assert np.abs(values).min() > 0
        # Reciprocal source/receiver pairs in a homogeneous acoustic medium.
        # 1e-3 is a relative single-precision invariant, not an accuracy benchmark.
        relative_error = float(
            abs(values[0, 1] - values[1, 0]) / max(abs(values[0, 1]), abs(values[1, 0]))
        )
        assert relative_error < 1e-3
    receipt = {
        "schema": "frequensolve-native-3d-acceptance-1",
        "build": manifest["build"],
        "license": manifest.get("license"),
        "workflow": manifest["workflow"],
        "task_summary": manifest["task_summary"],
        "input_hashes": {
            key: value["hash"] for key, value in manifest["inputs"].items()
        },
        "pressure_shape": list(values.shape),
        "relative_reciprocity_error": relative_error,
        "compatibility_warnings": compatibility_warnings,
        "scope": "local native 3D smoke; not full release or licensed customer acceptance",
    }
    (tmp_path / "acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
