import json

import h5py
import numpy as np
import pytest

import frequensolve as fs
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import JobStatus, RunResult
from frequensolve.project.project import Project
from frequensolve.seismic.eikonal import EikonalResults
from frequensolve.simulation.jobs import (
    BaseJob,
    EikonalConfig,
    EikonalJob,
    EikonalReceivers,
    EikonalSources,
)


def _eikonal_job(tmp_path, *, dimension=2):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(
        name="acoustic",
        physics="acoustic",
        dimension=dimension,
    )
    sim.mesh = MeshManager(
        HexMeshGenerator(
            l_bound=[0.0] * dimension,
            u_bound=[1000.0] * dimension,
            n=[2] * dimension,
        )
    )
    job = EikonalJob(
        "first_arrivals",
        sim,
        sources=EikonalSources.acquisition(["shot_1"], incidence_slot=1),
        receivers=EikonalReceivers.acquisition(["surface"]),
        solver={
            "abs_tolerance": {"value": 1.0e-10, "units": "s"},
            "rel_tolerance": 1.0e-9,
            "tie_tolerance": 1.0e-11,
            "max_waves": 100000,
            "max_updates": 100000000,
            "minimum_full_stencil_quality": 0.0,
        },
        products={
            "field": True,
            "characteristics": True,
            "max_points_per_characteristic": 10000,
            "max_characteristic_points": 1000000,
        },
    )
    return project, sim, job


def _string(group, name, value):
    group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))


def _write_eikonal_product(job):
    job.output_directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(job.eikonal_hdf5_file, "w") as h5:
        _string(h5, "schema", "fs-eikonal-output-1")
        metadata = h5.create_group("metadata")
        _string(metadata, "schema", "fs-eikonal-output-1")
        _string(metadata, "workflow", "eikonal")
        _string(metadata, "physics", "acoustic")
        _string(metadata, "geometry_policy", "exact_gmp_native_terminal_trace")
        _string(metadata, "coordinate_units", "m")
        _string(metadata, "travel_time_units", "s")
        _string(metadata, "submitted_config_json", "{}")
        metadata.create_dataset("dimension", data=2)
        metadata.create_dataset("source_count", data=1)
        metadata.create_dataset("receiver_count", data=2)
        metadata.create_dataset("vertex_count", data=4)
        metadata.create_dataset("characteristic_count", data=2)
        metadata.create_dataset("characteristic_point_count", data=6)

        preparation = h5.create_group("preparation")
        preparation.create_dataset("trace_vertex_count", data=4)
        preparation.create_dataset("trace_patch_count", data=1)
        preparation.create_dataset("physical_cell_count", data=1)
        preparation.create_dataset("directional_sample_count", data=8)
        preparation.create_dataset("full_stencil_count", data=4)
        preparation.create_dataset("rejected_geometry_count", data=0)
        preparation.create_dataset("minimum_full_stencil_quality", data=1.0)
        preparation.create_dataset("minimum_full_stencil_quality_threshold", data=0.0)
        preparation.create_dataset("prepared_bytes", data=1024)
        preparation.create_dataset("total_seconds", data=0.01)

        sources = h5.create_group("sources")
        sources.create_dataset("id", data=[1])
        sources.create_dataset("owner_cell", data=[1])
        sources.create_dataset("position", data=[[0.0], [0.0]])
        _string(sources, "name", ["shot_1"])

        diagnostics = h5.create_group("diagnostics")
        diagnostics.create_dataset("status", data=[0])
        diagnostics.create_dataset("wave_count", data=[2])
        diagnostics.create_dataset("update_count", data=[8])
        diagnostics.create_dataset("active_peak", data=[2])
        diagnostics.create_dataset("reachable_vertices", data=[4])
        diagnostics.create_dataset("ambiguous_vertices", data=[0])
        diagnostics.create_dataset("invalid_candidate_count", data=[0])
        diagnostics.create_dataset("wave_vertex_scan_count", data=[8])
        diagnostics.create_dataset("queue_visit_count", data=[8])
        diagnostics.create_dataset("queue_admission_count", data=[4])
        diagnostics.create_dataset("residual_max", data=[1.0e-12])

        receivers = h5.create_group("receivers")
        receivers.create_dataset("id", data=[1, 2])
        receivers.create_dataset("group_id", data=[1, 1])
        receivers.create_dataset("point_id", data=[1, 2])
        receivers.create_dataset("position", data=[[10.0, 0.0], [10.0, 10.0]])
        _string(receivers, "name", ["surface:1", "surface:2"])
        _string(receivers, "group_name", ["surface", "surface"])

        receiver_times = h5.create_group("receiver_times")
        receiver_times.create_dataset("travel_time", data=[[1.0], [np.sqrt(2.0)]])
        receiver_times.create_dataset("status", data=[[0], [0]])
        receiver_times.create_dataset("ambiguous", data=[[0], [1]])

        field = h5.create_group("field")
        field.create_dataset(
            "position",
            data=[[0.0, 10.0, 0.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
        )
        field.create_dataset("travel_time", data=[[0.0], [1.0], [1.0], [np.sqrt(2.0)]])

        paths = h5.create_group("characteristics")
        paths.create_dataset("offset", data=[0, 3, 6])
        paths.create_dataset("source_id", data=[1, 1])
        paths.create_dataset("receiver_id", data=[1, 2])
        paths.create_dataset("status", data=[0, 0])
        paths.create_dataset("ambiguous", data=[0, 1])
        paths.create_dataset(
            "position",
            data=[
                [0.0, 5.0, 10.0, 0.0, 5.0, 10.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            ],
        )
        paths.create_dataset(
            "travel_time", data=[0.0, 0.5, 1.0, 0.0, 0.7, np.sqrt(2.0)]
        )
        paths.create_dataset("vertex_id", data=[1, 0, 2, 1, 0, 4])
        paths.create_dataset("stencil_id", data=[0, 1, 0, 0, 2, 0])
        paths.create_dataset("owner_cell", data=[1, 1, 1, 1, 1, 1])

    job.eikonal_manifest_file.write_text(
        json.dumps(
            {
                "schema": "fs-eikonal-output-1",
                "status": "complete",
                "hdf5_file": "eikonal/first_arrivals.h5",
                "source_count": 1,
                "receiver_count": 2,
                "vertex_count": 4,
                "field_retained": True,
                "characteristics_retained": True,
                "characteristic_point_count": 6,
            }
        )
    )


def test_eikonal_job_serializes_current_sauce_contract_and_roundtrips(tmp_path):
    _project, _sim, job = _eikonal_job(tmp_path)

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert payload["workflow"] == "eikonal"
    assert payload["Eikonal"]["schema"] == "fs-eikonal-1"
    assert payload["Eikonal"]["sources"]["incidence_slot"] == 1
    assert "f_list" not in payload
    assert "Outputs" not in payload
    assert isinstance(loaded, EikonalJob)
    assert loaded.eikonal.to_fs() == job.eikonal.to_fs()
    assert job.task_fingerprint_payload(1)["job"]["Eikonal"] == payload["Eikonal"]
    assert "frequency" not in job.task_fingerprint_payload(1)
    assert not job.validate().errors


def test_eikonal_config_validates_contract_combinations(tmp_path):
    config = EikonalConfig(
        sources=EikonalSources.explicit([{"name": "source", "coordinates": [0.0, 0.0]}])
    )
    assert config.to_fs()["receivers"] == {"enabled": False}

    with pytest.raises(ValueError, match="disabled receivers require"):
        EikonalConfig(products={"field": False, "characteristics": False})
    with pytest.raises(ValueError, match="incidence_slot"):
        EikonalConfig(sources=EikonalSources.acquisition(incidence_slot=0))

    _project, sim, _job = _eikonal_job(tmp_path)
    sim.physics = "elastic"
    with pytest.raises(ValueError, match="acoustic"):
        EikonalJob("bad", sim)


def test_eikonal_results_read_fields_receiver_times_and_characteristics(tmp_path):
    _project, _sim, job = _eikonal_job(tmp_path)
    job.save()
    _write_eikonal_product(job)

    results = EikonalResults.from_job(job)
    first = results.field("shot_1")
    characteristic = results.characteristic(2)

    assert results.dimension == 2
    assert results.status == "complete"
    assert results.counts == {
        "sources": 1,
        "receivers": 2,
        "vertices": 4,
        "characteristics": 2,
        "characteristic_points": 6,
    }
    assert first.position.shape == (4, 2)
    np.testing.assert_allclose(first.travel_time, [0.0, 1.0, 1.0, np.sqrt(2.0)])
    np.testing.assert_allclose(
        results.receiver_times_for_source(1), [1.0, np.sqrt(2.0)]
    )
    assert results.receiver_ambiguous.tolist() == [[False, True]]
    assert characteristic.receiver_id == 2
    assert characteristic.ambiguous
    np.testing.assert_allclose(characteristic.position[-1], [10.0, 10.0])
    assert [path.characteristic_id for path in results.iter_characteristics()] == [
        1,
        2,
    ]
    assert isinstance(fs.load(job.output_directory), EikonalResults)
    assert isinstance(fs.load(job.eikonal_hdf5_file), EikonalResults)


def test_eikonal_run_state_uses_authoritative_product(tmp_path):
    _project, _sim, job = _eikonal_job(tmp_path)
    job.save()
    _write_eikonal_product(job)

    job.write_run_state(status="completed", tasks=[{"task": 1, "status": "success"}])
    plan = job.task_run_plan()

    assert job.is_run_current()
    assert plan["pending_indices"] == []
    state = json.loads(job.run_state_file.read_text())
    assert "eikonal" in state["outputs"]
    assert "traces" not in state["outputs"]


def test_completed_run_opens_eikonal_results(tmp_path):
    _project, _sim, job = _eikonal_job(tmp_path)
    job.save()
    _write_eikonal_product(job)
    completed = RunResult(
        job=job,
        status=JobStatus(state="completed", return_code=0),
    )

    assert isinstance(completed.eikonal(), EikonalResults)


@pytest.mark.visual
def test_plot_eikonal_draws_field_and_characteristics(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _project, _sim, job = _eikonal_job(tmp_path)
    job.save()
    _write_eikonal_product(job)

    ax = job.plot(show=False)

    assert ax.get_xlabel() == "x (m)"
    assert len(ax.collections) >= 3
    plt.close(ax.figure)


@pytest.mark.parametrize(
    "damage", ["bytes", "schema", "missing_group", "manifest", "shape"]
)
def test_eikonal_damaged_products_are_scheduled_again(tmp_path, damage):
    _project, _sim, job = _eikonal_job(tmp_path)
    job.save()
    _write_eikonal_product(job)
    job.write_run_state(status="completed", tasks=[{"task": 1, "status": "success"}])
    assert job.is_run_current()
    if damage == "bytes":
        job.eikonal_hdf5_file.write_bytes(b"interrupted HDF5 output")
    elif damage == "manifest":
        job.eikonal_manifest_file.write_text("[]")
    else:
        with h5py.File(job.eikonal_hdf5_file, "r+") as h5:
            if damage == "schema":
                h5["schema"][()] = "wrong-schema"
            elif damage == "missing_group":
                del h5["sources"]
            else:
                del h5["field/travel_time"]
                h5["field/travel_time"] = np.zeros((3, 3))
    assert not job.results_exist()
    assert not job.is_run_current()
    assert job.task_run_plan()["pending_indices"] == [0]
