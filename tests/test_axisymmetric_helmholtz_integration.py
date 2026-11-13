"""Native cylindrical projection regression; opt in with FS_TEST_TRANSIENT_EXECUTABLE."""

import os

import numpy as np
import pytest

import frequensolve as fs
from frequensolve.orchestrator.sites.local import LocalSite
from tests.test_transient_integration import simulation

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("FS_TEST_TRANSIENT_EXECUTABLE"),
        reason="requires a local Sauce fs2d executable",
    ),
]


@pytest.mark.parametrize(
    "physics,method",
    [
        ("elastic", "Galerkin"),
        ("elastic", "DPG"),
        ("acoustic", "Galerkin"),
        ("acoustic", "DPG"),
        ("coupled", "Galerkin"),
        ("coupled", "DPG"),
    ],
)
def test_axisymmetric_physical_velocity_projection(tmp_path, physics, method):
    vtk = pytest.importorskip("pyvista")
    sim = simulation(tmp_path / "project", physics)
    sim.physics = physics + "_axisym"
    sim.axisymmetric = True
    sim.discretization = fs.Discretization(method=method)
    sim.mesh.set_adapt(order=3, elems_per_wave=1, f_adapt=8, adapt_sources=0)
    sim.BCs = fs.BoundaryConditions(
        [
            fs.BoundaryCondition(
                name="outer",
                boundaries=["x_max", "z_min", "z_max"],
                conditions=["fixed"],
            ),
            fs.BoundaryCondition(
                name="axis", boundaries=["x_min"], conditions=["axis"]
            ),
        ]
    )
    sim.acquisition = fs.Acquisition()
    sim.acquisition.add_sources(
        kind="vector" if physics == "elastic" else "scalar",
        coords=[[0.487, 0.213]],
        **({"direction": [1, 0.7]} if physics == "elastic" else {}),
    )
    sim.acquisition.add_receiver_group(
        "receivers",
        fs.ReceiverNode(
            components=[fs.ReceiverComponent(name="radial", field="velocity_r")]
        ),
        coords=[[0.2, 0.2], [0.8, 0.3]],
    )
    # Independent native output catches errors that P+S reconstruction alone cannot.
    outputs = {}
    for projected in (False, True):
        fields = ["velocity"]
        if projected:
            fields += [
                "p_velocity",
                "s_velocity",
                "p_velocity_r",
                "s_velocity_r",
                "p_velocity_x",
                "s_velocity_x",
            ]
        job = fs.FrequencyDomainJob(
            "projected" if projected else "native",
            sim,
            f_list=[3],
            outputs=[
                fs.VtkOutput(name="velocity", fields=fields, format="vtu", order=2)
            ],
        )
        site = LocalSite(
            solver=os.environ["FS_TEST_TRANSIENT_EXECUTABLE"],
            solver_policy="off",
            n_workers=1,
            threads_per_worker=2,
            shutdown_on_completion=True,
            environment={"OMPI_MCA_btl": "self", "VECLIB_MAXIMUM_THREADS": "1"},
        )
        assert site.run(job, check=True).successful
        files = list((job._result_path / "ParaView").rglob("*.vtu"))
        assert len(files) == 1
        outputs[projected] = vtk.read(files[0])
    native, split = outputs[False], outputs[True]
    np.testing.assert_allclose(split.points, native.points)
    # Compare the L2 copy with H1-derived velocity away from point-source and
    # material-interface singularities. VTK geometry defaults to metres.
    x = native.points[:, :2] / 1000
    smooth = (
        (x[:, 0] > 0.1)
        & (np.linalg.norm(x - [0.487, 0.213], axis=1) > 0.15)
        & (np.abs(x[:, 1] - 0.25) > 0.06)
    )
    for phase in ("re", "im"):
        v = split[f"velocity_1_{phase}"]
        p = split[f"p_velocity_1_{phase}"]
        s = split[f"s_velocity_1_{phase}"]
        assert np.isfinite(v).all() and np.isfinite(p).all() and np.isfinite(s).all()
        scale = np.linalg.norm(v)
        if scale < 1e-15:
            continue
        assert np.linalg.norm(p + s - v) / scale < 2e-5
        # Derived Galerkin velocity is copied into lower-degree element-local L2.
        reference = native[f"velocity_1_{phase}"]
        assert (
            np.linalg.norm((v - reference)[smooth]) / np.linalg.norm(reference[smooth])
            < 0.05
        )
        np.testing.assert_allclose(
            split[f"p_velocity_r_1_{phase}"], p[:, 0], atol=scale * 1e-6
        )
        np.testing.assert_allclose(
            split[f"s_velocity_r_1_{phase}"], s[:, 0], atol=scale * 1e-6
        )
        np.testing.assert_array_equal(
            split[f"p_velocity_x_1_{phase}"], split[f"p_velocity_r_1_{phase}"]
        )
        np.testing.assert_array_equal(
            split[f"s_velocity_x_1_{phase}"], split[f"s_velocity_r_1_{phase}"]
        )
    assert (
        sum(np.linalg.norm(split[f"p_velocity_1_{phase}"]) for phase in ("re", "im"))
        > 0
    )
    assert (
        sum(np.linalg.norm(split[f"s_velocity_1_{phase}"]) for phase in ("re", "im"))
        > 0
    )


def test_transient_coupled_velocity_cadence_and_restart(tmp_path):
    """Fluid velocity includes every accepted pressure step, including before restart."""
    import shutil
    import xml.etree.ElementTree as ET

    vtk = pytest.importorskip("pyvista")
    sim = simulation(tmp_path / "project", "coupled")
    sim.physics = "coupled_axisym"
    sim.axisymmetric = True
    sim.BCs = fs.BoundaryConditions(
        [
            fs.BoundaryCondition(
                name="axis", boundaries=["x_min"], conditions=["axis"]
            ),
            fs.BoundaryCondition(
                name="outer",
                boundaries=["x_max", "z_min", "z_max"],
                conditions=["impedance"],
            ),
        ]
    )
    sim.acquisition = fs.Acquisition()
    sim.acquisition.add_sources(kind="scalar", coords=[[0.25, 0.125]])
    sim.extra["helmholtz_projection"] = {
        "field": "velocity",
        "parts": ["p", "s"],
        "s_projection": "complement",
    }
    fields = ["velocity", "p_velocity", "s_velocity"]
    job = fs.TransientJob(
        "dense",
        sim,
        time=fs.TimeIntegration(start=-0.4, end=0.25, dt=0.01),
        source=fs.TimeFunction.ricker(frequency=8, delay=0.05),
        output_schedule=fs.TimeOutput(stride=5, write_initial=False),
        checkpoint=fs.CheckpointSchedule(stride=50),
        outputs=[fs.VtkOutput(name="split", fields=fields, order=2)],
    )

    def run(current):
        site = LocalSite(
            solver=os.environ["FS_TEST_TRANSIENT_EXECUTABLE"],
            solver_policy="off",
            n_workers=1,
            threads_per_worker=2,
            shutdown_on_completion=True,
            environment={"OMPI_MCA_btl": "self", "VECLIB_MAXIMUM_THREADS": "1"},
        )
        assert site.run(current, check=True).successful
        root = current._result_path / "ParaView"
        return {
            round(float(e.attrib["timestep"]), 8): vtk.read(root / e.attrib["file"])
            for e in ET.parse(root / "split.pvd").findall("./Collection/DataSet")
        }

    dense = run(job)
    checkpoint = tmp_path / "restart"
    shutil.copytree(job._result_path / "checkpoints", checkpoint)
    resumed = job.with_restart(checkpoint, name="resumed")
    resumed.checkpoint = fs.CheckpointSchedule()
    resumed.output_schedule = fs.TimeOutput(times=[0.2, 0.25], write_initial=False)
    resumed_fields = run(resumed)
    assert set(resumed_fields) == {0.25}  # final checkpoint is always retained
    sparse = fs.TransientJob(
        "sparse",
        sim,
        time=job.time,
        source=job.source,
        outputs=[fs.VtkOutput(name="split", fields=fields, order=2)],
        output_schedule=fs.TimeOutput(times=[0.2, 0.25], write_initial=False),
    )
    sparse_fields = run(sparse)
    for time in (0.2, 0.25):
        reference = dense[time]
        v, p, s = [reference[f"{field}_1_re"] for field in fields]
        assert set(reference.point_data) == {f"{field}_1_re" for field in fields}
        assert not reference.cell_data
        assert float(reference.field_data["TimeValue"][0]) == pytest.approx(time)
        assert all(np.isfinite(a).all() for a in (v, p, s))
        assert np.linalg.norm(v - p - s) < 1e-5 * np.linalg.norm(v)
        fluid = reference.points[:, 1] < 240  # geometry is in metres
        solid = reference.points[:, 1] > 260
        assert np.linalg.norm(v[fluid]) > 0
        assert np.linalg.norm(v[solid]) > 0
        axis = solid & (reference.points[:, 0] == 0)
        assert axis.any()
        np.testing.assert_array_equal(v[axis, 0], 0)
        assert np.linalg.norm(p) > 0 and np.linalg.norm(s) > 0
        comparisons = [sparse_fields[time]]
        if time in resumed_fields:
            comparisons.append(resumed_fields[time])
        for result in comparisons:
            np.testing.assert_array_equal(result.points, reference.points)
            for field in fields:
                a, b = result[f"{field}_1_re"], reference[f"{field}_1_re"]
                assert np.linalg.norm(a - b) <= 5e-4 * np.linalg.norm(b)
