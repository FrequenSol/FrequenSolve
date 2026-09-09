import json

import h5py
import numpy as np
import pytest

import frequensolve as fs
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.project.project import Project
from frequensolve.seismic.rays import RayResults
from frequensolve.simulation.jobs import (
    BaseJob,
    RayLaunch,
    RaySources,
    RayTracingConfig,
    RayTracingJob,
)


def _ray_job(tmp_path, *, dimension=2):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(
        name="acoustic",
        physics="acoustic",
        dimension=dimension,
    )
    bounds = [0.0] * dimension
    upper = [1000.0] * dimension
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=bounds, u_bound=upper, n=[2] * dimension)
    )
    launch = RayLaunch.fan2d(5, -45.0, 45.0) if dimension == 2 else RayLaunch.sphere(8)
    job = RayTracingJob(
        "rays",
        sim,
        launch=launch,
        sources=RaySources.acquisition(["shot_1"]),
        integrator={"tau_max": {"value": 2.0, "units": "s"}},
        receivers={
            "enabled": True,
            "kind": "acquisition",
            "groups": ["surface"],
            "capture_radius": {"value": 10.0, "units": "m"},
        },
    )
    return project, sim, job


def _string(group, name, value):
    group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))


def _write_ray_product(job):
    output = job.output_directory
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(job.ray_hdf5_file, "w") as h5:
        metadata = h5.create_group("metadata")
        _string(metadata, "schema_version", "fs-rays-1")
        _string(metadata, "layout_kind", "indexed_rays_v1")
        _string(metadata, "physics", "acoustic")
        _string(metadata, "workflow", "raytrace")
        _string(metadata, "geometry_units", "m")
        _string(metadata, "time_units", "s")
        metadata.create_dataset("dimension", data=2)

        sources = h5.create_group("sources")
        sources.create_dataset("source_id", data=[1])
        source_position = sources.create_dataset("position", data=[[0.0, 0.0]])
        source_position.attrs["logical_axes"] = "record,dimension"
        _string(sources, "source_name", ["shot_1"])
        _string(sources, "provenance", ["acquisition"])

        receivers = h5.create_group("receivers")
        receivers.create_dataset("receiver_id", data=[1])
        receiver_position = receivers.create_dataset("position", data=[[8.0, 2.0]])
        receiver_position.attrs["logical_axes"] = "record,dimension"
        receivers.create_dataset("capture_radius", data=[0.5])
        _string(receivers, "receiver_name", ["receiver_1"])
        _string(receivers, "provenance", ["surface"])

        rays = h5.create_group("rays")
        rays.create_dataset("ray_id", data=[1, 2])
        rays.create_dataset("parent_id", data=[0, 0])
        rays.create_dataset("source_id", data=[1, 1])
        rays.create_dataset("status", data=[2, 2])
        rays.create_dataset("reason", data=[1, 1])
        _string(rays, "reason_name", ["domain_exit", "domain_exit"])
        rays.create_dataset("branch_depth", data=[0, 0])
        rays.create_dataset("admitted", data=[1, 1])
        rays.create_dataset("energy_weight", data=[1.0, 0.75])
        rays.create_dataset("point_offset", data=[0, 3])
        rays.create_dataset("point_count", data=[3, 2])
        rays.create_dataset("point_offsets", data=[0, 3, 5])
        rays.create_dataset("event_offset", data=[0, 1])
        rays.create_dataset("event_count", data=[1, 0])
        rays.create_dataset("event_offsets", data=[0, 1, 1])
        rays.create_dataset("receiver_hit_offset", data=[0, 1])
        rays.create_dataset("receiver_hit_count", data=[1, 0])
        rays.create_dataset("receiver_hit_offsets", data=[0, 1, 1])

        points = h5.create_group("points")
        points.create_dataset("ray_id", data=[1, 1, 1, 2, 2])
        position = points.create_dataset(
            "position",
            data=[
                [0.0, 0.0],
                [4.0, 1.0],
                [8.0, 2.0],
                [0.0, 0.0],
                [6.0, -3.0],
            ],
        )
        position.attrs["logical_axes"] = "record,dimension"
        points.create_dataset("tau", data=[0.0, 0.4, 0.8, 0.0, 0.7])
        points.create_dataset("arc_length", data=[0.0, 4.1, 8.2, 0.0, 6.7])

        events = h5.create_group("events")
        events.create_dataset("event_id", data=[1])
        events.create_dataset("ray_id", data=[1])
        events.create_dataset("classification", data=[2])

        hits = h5.create_group("receiver_hits")
        hits.create_dataset("hit_id", data=[1])
        hits.create_dataset("ray_id", data=[1])
        hits.create_dataset("receiver_id", data=[1])
        hit_position = hits.create_dataset("position", data=[[8.0, 2.0]])
        hit_position.attrs["logical_axes"] = "record,dimension"
        hits.create_dataset("tau", data=[0.8])

        codes = h5.create_group("status_codes")
        reason = codes.create_group("reason")
        reason.create_dataset("code", data=[1])
        _string(reason, "name", ["domain_exit"])
        _string(reason, "class", ["terminal"])
        reason.create_dataset("terminal", data=[1])
        _string(reason, "description", ["Ray left the modeled domain."])

    manifest = {
        "schema": "fs-rays-1",
        "status": "complete",
        "physics": "acoustic",
        "dimension": 2,
        "geometry_policy": "exact_gmp",
        "counts": {
            "sources": 1,
            "receivers": 1,
            "rays": 2,
            "points": 5,
            "events": 1,
            "receiver_hits": 1,
            "failed_rays": 0,
            "truncated_rays": 0,
        },
        "hdf5": {
            "format": "hdf5",
            "schema": "fs-rays-1",
            "layout": "indexed_rays_v1",
            "authoritative": True,
            "relative_path": "rays/rays.h5",
            "groups": {
                name: f"/{name}"
                for name in (
                    "metadata",
                    "sources",
                    "receivers",
                    "rays",
                    "points",
                    "events",
                    "receiver_hits",
                    "status_codes",
                )
            },
        },
    }
    job.ray_manifest_file.write_text(json.dumps(manifest))


def test_ray_job_serializes_current_sauce_contract_and_roundtrips(tmp_path):
    _project, _sim, job = _ray_job(tmp_path)

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert payload["workflow"] == "raytrace"
    assert payload["RayTracing"]["schema"] == "fs-ray-tracing-1"
    assert payload["RayTracing"]["launch"]["kind"] == "fan2d"
    assert "f_list" not in payload
    assert "Outputs" not in payload
    assert isinstance(loaded, RayTracingJob)
    assert loaded.ray_tracing.to_fs() == job.ray_tracing.to_fs()
    assert loaded.n_tasks == 1
    assert loaded.frequency_independent
    assert not loaded.supports_trace_packing
    assert loaded.max_ranks_per_task == 1
    assert "RayTracing" in job.fingerprint_payload()["job"]
    assert "frequency" not in job.task_fingerprint_payload(1)
    assert not job.validate().errors


def test_ray_config_rejects_contract_and_dimension_mismatches(tmp_path):
    with pytest.raises(ValueError, match="Unknown launch field"):
        RayTracingConfig({"kind": "sphere", "count": 4, "typo": True})

    _project, sim, _job = _ray_job(tmp_path)
    with pytest.raises(ValueError, match="require a 3D simulation"):
        RayTracingJob("bad", sim, launch=RayLaunch.sphere(8))


def test_ray_results_read_indexed_paths_events_hits_and_codes(tmp_path):
    _project, _sim, job = _ray_job(tmp_path)
    job.save()
    _write_ray_product(job)

    results = RayResults.from_job(job)
    first = results.path(1)

    assert results.dimension == 2
    assert results.status == "complete"
    assert results.counts["rays"] == 2
    np.testing.assert_allclose(first.position, [[0, 0], [4, 1], [8, 2]])
    np.testing.assert_allclose(first.travel_time, [0.0, 0.4, 0.8])
    assert first.ray["reason_name"] == "domain_exit"
    assert results.events_for_ray(1)["event_id"].tolist() == [1]
    assert results.receiver_hits_for_ray(1)["receiver_id"].tolist() == [1]
    assert results.code_name("reason", 1) == "domain_exit"
    assert [path.ray_id for path in results.iter_paths(source_id=1)] == [1, 2]
    assert isinstance(fs.load(job.output_directory), RayResults)
    assert isinstance(fs.load(job.ray_hdf5_file), RayResults)


def test_ray_run_state_uses_authoritative_product_not_trace_shards(tmp_path):
    _project, _sim, job = _ray_job(tmp_path)
    job.save()
    _write_ray_product(job)

    job.write_run_state(status="completed", tasks=[{"task": 1, "status": "success"}])
    plan = job.task_run_plan()

    assert job.is_run_current()
    assert plan["pending_indices"] == []
    state = json.loads(job.run_state_file.read_text())
    assert "rays" in state["outputs"]
    assert "traces" not in state["outputs"]


@pytest.mark.visual
def test_plot_rays_draws_retained_segments(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _project, _sim, job = _ray_job(tmp_path)
    job.save()
    _write_ray_product(job)

    ax = job.plot(show_hits=True, show=False)

    assert ax.get_xlabel() == "x (m)"
    assert len(ax.collections) >= 4
    plt.close(ax.figure)
