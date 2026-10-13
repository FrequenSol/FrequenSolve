"""Local SDK-to-Sauce propagation; opt in with FS_TEST_TRANSIENT_EXECUTABLE."""

import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

import frequensolve as fs
from frequensolve.orchestrator.sites.local import LocalSite

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("FS_TEST_TRANSIENT_EXECUTABLE"),
        reason="requires a local Sauce fs2d executable",
    ),
]


def simulation(root, physics):
    acoustic = {"Vp": {"value": 1.5}, "Rho": {"value": 1}}
    elastic = {"Vp": {"value": 2.5}, "Vs": {"value": 1.2}, "Rho": {"value": 2}}
    properties = (
        [acoustic, elastic]
        if physics == "coupled"
        else [elastic if physics == "elastic" else acoustic]
    )
    depths = [0, 0.25, 0.5] if physics == "coupled" else [0, 0.5]
    return fs.SeismicSimulation.from_fs(
        {
            "name": physics,
            "physics": physics,
            "dimension": 2,
            "f0": 8,
            "project_path": str(root),
            "Model": {
                "_type": "LayeredModel",
                "name": "layers",
                "dimension": 2,
                "attenuation": {"model": "none"},
                "x_limits": [0, 1],
                "ordering": "top_down",
                "subdomains": [
                    {"mesh_block_id": i + 1, "name": f"layer{i}", "properties": p}
                    for i, p in enumerate(properties)
                ],
                "surfaces": [
                    {"name": f"surface{i}", "interface": True, "depth": {"value": z}}
                    for i, z in enumerate(depths)
                ],
            },
            "Mesh": {
                "generator": {
                    "_type": "LayeredMeshGenerator",
                    "n": [6, 6],
                    "l_bound": [0, 0],
                    "u_bound": [1, 0.5],
                },
                "adapt": {
                    "order": 1,
                    "elems_per_wave": 0.5,
                    "f_adapt": 8,
                    "adapt_sources": 0,
                },
            },
            "BCs": [
                {
                    "name": "fixed",
                    "boundaries": ["x_min", "x_max", "z_min", "z_max"],
                    "conditions": ["fixed"],
                }
            ],
            "Solver": {
                "grids": 1,
                "tolerance": 1e-4,
                "max_iter": 200,
                "galerkin_multigrid": True,
                "block_schur": {
                    "enabled": physics == "coupled",
                    "maximum_iterations": 32,
                    "relative_tolerance": 1e-4,
                },
            },
            "Discretization": {"method": "Galerkin"},
            "Acquisition": {
                "source_geometry": {
                    "_type": "Inline",
                    "kind": "scalar",
                    "sources": [
                        {"name": "source", "coordinates": [0.487, 0.213]},
                        *(
                            [{"name": "source2", "coordinates": [0.617, 0.223]}]
                            if physics == "acoustic"
                            else []
                        ),
                    ],
                }
            },
        }
    )


@pytest.mark.parametrize(
    "physics,schedule",
    [
        ("acoustic", fs.TimeOutput(stride=10)),
        ("elastic", fs.TimeOutput(interval=0.025)),
        ("coupled", fs.TimeOutput(times=[-0.2, -0.1, 0, 0.035], write_initial=False)),
    ],
)
def test_local_native_wave_output_and_restart(tmp_path, physics, schedule):
    sim = simulation(tmp_path / "project", physics)
    field = "displacement" if physics == "elastic" else "pressure"
    sim.acquisition.add_receiver_group(
        "receivers",
        fs.ReceiverNode(
            name="sensor",
            components=[
                fs.ReceiverComponent(
                    name="value",
                    field=field,
                    direction=[1, 0] if physics == "elastic" else None,
                )
            ],
        ),
        coords=[[0.4, 0.2], [0.6, 0.2]],
    )
    job = fs.TransientJob(
        "wave",
        sim,
        time=fs.TimeIntegration(-0.2, 0.05, 0.005),
        source=fs.TimeFunction.ricker(8, delay=0.125),
        output_schedule=schedule,
        checkpoint=fs.CheckpointSchedule(stride=20),
    )
    job += fs.WavefieldOutput(
        name="field",
        field=field,
        grid=fs.CartesianGrid(
            x0=[0, 0], x1=[1, 0.5], n=[11, 7], dims=["x", "z"], units="km"
        ),
    )
    for name, format, encoding in [
        ("binary", "vtu", "appended"),
        ("ascii", "vtu", "ascii"),
        ("shared", "xdmf", "hdf5"),
    ]:
        job += fs.VtkOutput(
            name=name,
            fields=[field],
            format=format,
            encoding=encoding,
            sources=[2] if physics == "acoustic" and name == "ascii" else None,
        )
    site = LocalSite(
        solver=os.environ["FS_TEST_TRANSIENT_EXECUTABLE"],
        solver_policy="off",
        n_workers=1,
        threads_per_worker=1,
        shutdown_on_completion=True,
        environment={"OMPI_MCA_btl": "self", "VECLIB_MAXIMUM_THREADS": "1"},
    )
    result = site.run(job, check=True)
    assert result.successful
    traces = job.traces.open().read("receivers")
    assert traces.sizes["receiver"] == 2
    np.testing.assert_array_equal(traces.step, schedule.steps(job.time))
    data = job.wavefields.open()
    np.testing.assert_array_equal(data.steps, schedule.steps(job.time))
    values = data.read("field")
    assert np.isfinite(values).all() and np.max(np.abs(values)) > 0
    assert_paraview(job, schedule.steps(job.time), field)
    assert job.is_run_current()
    # Authoritative registration must not also advertise a harmonic shard.
    assert all(
        a.schema != "fs_trace_payload_shard_v1"
        for a in job.operation_result().artifacts
    )
    assert job.plan_tasks()["pending_indices"] == []
    cp = tmp_path / "checkpoint-copy"
    shutil.copytree(job._result_path / "checkpoints", cp)
    restarted = job.with_restart(cp, name="restart")
    result = site.run(restarted, check=True)
    assert result.successful
    resumed = restarted.wavefields.open().read("field")
    assert list(resumed.step.values) == [job.time.steps]
    assert_paraview(restarted, [job.time.steps], field)
    np.testing.assert_allclose(resumed.values[0], values.values[-1], rtol=1e-4, atol=0)
    if physics == "acoustic":
        assert_backend_rejects_invalid_paraview(job, tmp_path)


def assert_backend_rejects_invalid_paraview(job, root):
    """Exercise native validation independently of SDK preflight."""
    invalid = [
        (
            {"fields": ["pressure", "nonexistent_field"]},
            "Unsupported transient ParaView field",
        ),
        ({"fields": ["pressure"], "properties": ["Vp"]}, "accept fields only"),
        ({"fields": ["pressure"], "post_process": True}, "do not support post_process"),
        (
            {"items": [{"kind": "field", "field": "pressure", "parts": ["im"]}]},
            "fields are real",
        ),
    ]
    for index, (request, message) in enumerate(invalid):
        payload = json.loads(job.job_file.read_text())
        payload["Outputs"]["ParaView"] = [{"name": "invalid", **request}]
        config = root / f"invalid_{index}.json"
        config.write_text(json.dumps(payload))
        run = subprocess.run(
            [
                os.environ["FS_TEST_TRANSIENT_EXECUTABLE"],
                "-j",
                str(config),
                "-i",
                "1",
                "-nthreads",
                "1",
                "--result-dir",
                str(root / f"invalid_results_{index}"),
                "--tmp-directory",
                str(root / "native-tmp"),
            ],
            env={**os.environ, "OMPI_MCA_btl": "self"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert run.returncode != 0
        assert message in run.stdout + run.stderr


def assert_paraview(job, steps, field):
    """Read native XML/binary snapshots and verify shared HDF5 geometry."""
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    times = job.time.start + np.asarray(steps) * job.time.dt
    sources = 2 if job.simulation.physics == "acoustic" else 1
    root = job._result_path / "ParaView"
    for name in ("binary", "ascii"):
        entries = ET.parse(root / f"{name}.pvd").findall("./Collection/DataSet")
        np.testing.assert_allclose(
            [float(e.attrib["timestep"]) for e in entries], times, atol=1e-15
        )
        assert len(list(root.glob(f"{name}_*.vtu"))) == len(times)
        nonzero = False
        for entry, time in zip(entries, times):
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(root / entry.attrib["file"]))
            reader.Update()
            grid = reader.GetOutput()
            assert grid.GetNumberOfCells() > 0
            assert grid.GetCellData().GetNumberOfArrays() == 0
            selected = (
                [2] if sources == 2 and name == "ascii" else range(1, sources + 1)
            )
            assert {
                grid.GetPointData().GetArrayName(i)
                for i in range(grid.GetPointData().GetNumberOfArrays())
            } == {f"{field}_{source}_re" for source in selected}
            np.testing.assert_allclose(
                grid.GetFieldData().GetArray("TimeValue").GetValue(0), time, atol=1e-15
            )
            values = vtk_to_numpy(grid.GetPointData().GetArray(0))
            assert np.isfinite(values).all()
            nonzero |= bool(np.max(np.abs(values)) > 0)
            assert reader.GetOutputInformation(0).Get(
                vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
            )[0] == pytest.approx(time)
        assert nonzero
    collection = ET.parse(root / "shared.xmf")
    includes = collection.findall(".//{http://www.w3.org/2003/XInclude}include")
    assert len(includes) == len(times)
    with h5py.File(root / "shared.h5") as h5:
        assert len(h5) == len(times)
        geometry_groups = [key for key in h5 if "geometry" in h5[key]]
        assert geometry_groups == ["000000000000"]
        for index, (entry, time) in enumerate(zip(includes, times)):
            step = h5[f"{index:012d}"]
            assert set(step) == ({"geometry", "fields"} if index == 0 else {"fields"})
            assert set(step["fields"]) == {
                f"{field}_{source}_re" for source in range(1, sources + 1)
            }
            xml = ET.parse(root / entry.attrib["href"])
            assert float(xml.find(".//Time").attrib["Value"]) == pytest.approx(time)
            geometry_refs = xml.findall(".//Geometry/DataItem") + xml.findall(
                ".//Topology/DataItem"
            )
            assert all("/000000000000/geometry/" in item.text for item in geometry_refs)
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(str(root / "shared.xmf"))
    reader.UpdateInformation()
    np.testing.assert_allclose(
        reader.GetOutputInformation(0).Get(
            vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
        ),
        times,
        atol=1e-15,
    )
    reader.UpdateTimeStep(times[-1])
    assert reader.GetOutputDataObject(0).GetNumberOfCells() > 0
    artifacts = {a.id: a for a in job.operation_result().artifacts}
    for name in ("binary", "ascii", "shared"):
        assert len(artifacts[f"visualization-series:{name}"].dependencies) == len(times)


def test_native_helmholtz_vtu_uses_current_time(tmp_path):
    """Native velocity/P/S snapshots must be evaluated after the scheduled projection."""
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    sim = simulation(tmp_path / "project", "elastic")
    sim.extra["helmholtz_projection"] = {
        "enabled": True,
        "field": "velocity",
        "parts": ["p", "s"],
    }
    job = fs.TransientJob(
        "helmholtz",
        sim,
        time=fs.TimeIntegration(-0.2, 0.05, 0.005),
        source=fs.TimeFunction.ricker(8, delay=0.125),
        output_schedule=fs.TimeOutput(stride=10),
    )
    fields = ("velocity", "p_velocity", "s_velocity")
    job += fs.VtkOutput(name="decomposition", fields=fields)
    site = LocalSite(
        solver=os.environ["FS_TEST_TRANSIENT_EXECUTABLE"],
        solver_policy="off",
        n_workers=1,
        threads_per_worker=1,
        shutdown_on_completion=True,
        environment={"OMPI_MCA_btl": "self", "VECLIB_MAXIMUM_THREADS": "1"},
    )
    assert site.run(job, check=True).successful
    root = job._result_path / "ParaView"
    entries = ET.parse(root / "decomposition.pvd").findall("./Collection/DataSet")
    assert len(entries) == len(job.output_schedule.steps(job.time))
    energy = np.zeros(3)
    for entry in entries:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(root / entry.attrib["file"]))
        reader.Update()
        grid = reader.GetOutput()
        assert grid.GetPointData().GetNumberOfArrays() == 3
        values = [
            vtk_to_numpy(grid.GetPointData().GetArray(f"{field}_1_re"))
            for field in fields
        ]
        assert all(np.isfinite(value).all() for value in values)
        error = np.linalg.norm(values[0] - values[1] - values[2])
        assert error <= 1e-5 * np.linalg.norm(values[0]) + 1e-30
        energy += np.array([np.linalg.norm(value) for value in values]) ** 2
    assert np.all(energy > 0)
    assert job.is_run_current()
