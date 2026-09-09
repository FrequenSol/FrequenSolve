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
    from frequensolve.seismic import Acquisition
    from frequensolve.seismic.receivers import ReceiverComponent, ReceiverNode
    from frequensolve.seismic.sources import SourceGeometry

    sim.acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar", coords=[[0.0] * dimension], names=["shot_1"]
        )
    )
    sim.acquisition.add_receiver_group(
        "surface",
        ReceiverNode(
            name="hydrophone",
            components=[ReceiverComponent(name="p", field="pressure")],
        ),
        coords=[[0.0] * dimension],
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


@pytest.mark.parametrize(
    "coordinates",
    [
        None,
        ["x", "z"],
        [0.0, float("nan")],
        [0.0, float("inf")],
        [1.0],
        [1, 2, 3, 4],
        [[0.0, 1.0]],
    ],
)
@pytest.mark.parametrize("section", ["sources", "receivers"])
def test_eikonal_rejects_invalid_explicit_coordinates(coordinates, section):
    factory = EikonalSources if section == "sources" else EikonalReceivers
    with pytest.raises(ValueError, match="finite numeric coordinates"):
        EikonalConfig(
            **{
                section: factory.explicit(
                    [{"name": "point", "coordinates": coordinates}]
                )
            }
        )


def test_eikonal_coordinates_follow_simulation_dimension_and_normalize_units(tmp_path):
    _, sim, _ = _eikonal_job(tmp_path)
    point = {
        "name": "source",
        "coordinates": fs.CoordinateValue([0.0, 1.0], units="km"),
    }
    job = EikonalJob("explicit", sim, sources=EikonalSources.explicit([point]))
    job.save()
    assert job.to_fs()["Eikonal"]["sources"]["points"][0]["coordinates"] == {
        "value": [0.0, 1.0],
        "units": "km",
    }
    sim.dimension = 3
    assert "job.config.invalid" in {issue.code for issue in job.validate().issues}
    with pytest.raises(ValueError, match="3 coordinates"):
        job.to_fs()
    with pytest.raises(ValueError, match="3 coordinates"):
        EikonalJob("bad", sim, sources=EikonalSources.explicit([point]))


def test_square_eikonal_tables_keep_h5py_source_major_orientation(tmp_path):
    _, _, job = _eikonal_job(tmp_path)
    _write_eikonal_product(job)
    with h5py.File(job.eikonal_hdf5_file, "a") as h5:
        h5["metadata/source_count"][()] = 2
        h5["metadata/characteristic_count"][()] = 0
        h5["metadata/characteristic_point_count"][()] = 0
        del h5["characteristics"]
        for name in list(h5["sources"]):
            del h5["sources"][name]
        h5["sources"].create_dataset("id", data=[1, 2])
        h5["sources"].create_dataset("owner_cell", data=[1, 1])
        h5["sources"].create_dataset("position", data=[[0.0, 0.0], [1.0, 2.0]])
        _string(h5["sources"], "name", ["shot_1", "shot_2"])
        for name in list(h5["diagnostics"]):
            value = h5["diagnostics"][name][0]
            del h5["diagnostics"][name]
            h5["diagnostics"].create_dataset(name, data=[value, value])
        for name, values in {
            "travel_time": [[2.0, 3.0], [11.0, 13.0]],
            "status": [[0, 0], [0, 0]],
            "ambiguous": [[0, 1], [1, 0]],
        }.items():
            del h5["receiver_times"][name]
            h5["receiver_times"].create_dataset(name, data=values)
        del h5["field/travel_time"]
        h5["field"].create_dataset("travel_time", data=[[0, 1, 2, 3], [4, 5, 6, 7]])
    manifest = json.loads(job.eikonal_manifest_file.read_text())
    manifest.update(
        source_count=2, characteristics_retained=False, characteristic_point_count=0
    )
    job.eikonal_manifest_file.write_text(json.dumps(manifest))
    result = EikonalResults.from_job(job)
    np.testing.assert_allclose(result.receiver_times_for_source("shot_2"), [11.0, 13.0])
    np.testing.assert_allclose(result.receiver_times_for_source("shot_1"), [2.0, 3.0])


@pytest.mark.parametrize("dimension", [2.5, "2.5D"])
def test_eikonal_rejects_half_dimension_without_truncation(tmp_path, dimension):
    _, sim, job = _eikonal_job(tmp_path)
    sim.dimension = dimension
    with pytest.raises(ValueError, match="full-dimensional"):
        EikonalJob("invalid", sim)
    assert "job.config.invalid" in {issue.code for issue in job.validate().issues}
    with pytest.raises(ValueError, match="full-dimensional"):
        job.to_fs()


@pytest.mark.parametrize("section", ["sources", "receivers"])
def test_eikonal_validates_locally_known_acquisition_selectors(tmp_path, section):
    from frequensolve.seismic.acquisition import Acquisition
    from frequensolve.seismic.receivers import ReceiverComponent, ReceiverNode
    from frequensolve.seismic.sources import SourceGeometry

    _, sim, job = _eikonal_job(tmp_path)
    sim.acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar", coords=[[0.0, 0.0]], names=["shot_1"]
        )
    )
    sim.acquisition.add_receiver_group(
        "surface",
        ReceiverNode(
            name="hydrophone",
            components=[ReceiverComponent(name="p", field="pressure")],
        ),
        coords=[[0.0, 0.0]],
    )
    job.validate_outputs()
    replacement = (
        EikonalSources.acquisition(["missing-shot"])
        if section == "sources"
        else EikonalReceivers.acquisition(["missing-group"])
    )
    job.eikonal = job.eikonal.with_updates(**{section: replacement})
    assert "job.config.invalid" in {issue.code for issue in job.validate().issues}
    with pytest.raises(ValueError, match="Unknown Eikonal acquisition"):
        job.to_fs()


def test_eikonal_defers_external_source_catalog_validation(tmp_path):
    from frequensolve.seismic.acquisition import Acquisition
    from frequensolve.seismic.sources import SourceGeometry

    _, sim, job = _eikonal_job(tmp_path)
    sim.acquisition = Acquisition(
        source_geometry=SourceGeometry.hdf5(
            file="remote:source.h5", dataset="coordinates", kind="scalar"
        )
    )
    job.eikonal = job.eikonal.with_updates(
        receivers=EikonalReceivers.disabled(),
        products={"field": True, "characteristics": False},
    )
    job.validate_outputs()


@pytest.mark.parametrize("field", ["directory", "hdf5_file"])
@pytest.mark.parametrize("value", [None, 4, True, [], {}])
def test_eikonal_rejects_non_path_output_values(field, value):
    with pytest.raises(ValueError, match="relative path"):
        EikonalConfig(output={field: value})


def test_eikonal_normalizes_path_output_values():
    from pathlib import Path

    config = EikonalConfig(
        output={"directory": Path("results"), "hdf5_file": Path("arrival.h5")}
    )
    payload = config.to_fs()
    assert payload["output"]["directory"] == "results"
    assert payload["output"]["hdf5_file"] == "arrival.h5"
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "solver",
    [
        {"rel_tolerance": "1e-9"},
        {"tie_tolerance": "1e-9"},
        {"abs_tolerance": {"value": "1e-9", "units": "s"}},
        {"minimum_full_stencil_quality": "0.5"},
    ],
)
def test_eikonal_rejects_string_floating_solver_values(solver):
    with pytest.raises(ValueError):
        EikonalConfig(solver=solver)


@pytest.mark.parametrize("value", [True, False, 1.9, "1", 0])
@pytest.mark.parametrize("explicit", [False, True])
def test_eikonal_source_factories_reject_invalid_incidence(value, explicit):
    with pytest.raises(ValueError, match="incidence_slot"):
        if explicit:
            EikonalSources.explicit(
                [{"name": "s", "coordinates": [0, 0]}], incidence_slot=value
            )
        else:
            EikonalSources.acquisition(incidence_slot=value)


@pytest.mark.parametrize("value", [None, 2, True, "", "  "])
@pytest.mark.parametrize("section", ["sources", "receivers"])
def test_eikonal_rejects_invalid_explicit_point_names(value, section):
    factory = EikonalSources if section == "sources" else EikonalReceivers
    with pytest.raises(ValueError, match="name"):
        EikonalConfig(
            **{section: factory.explicit([{"name": value, "coordinates": [0, 0]}])}
        )


@pytest.mark.parametrize(
    "filename", ["manifest.json", "./manifest.json", ".//manifest.json"]
)
def test_eikonal_output_cannot_overwrite_manifest(filename):
    with pytest.raises(ValueError, match="collide"):
        EikonalConfig(output={"hdf5_file": filename})


def test_eikonal_rejects_receiver_selector_in_empty_catalog(tmp_path):
    from frequensolve.seismic.acquisition import Acquisition

    _, simulation, job = _eikonal_job(tmp_path)
    simulation.acquisition = Acquisition()
    with pytest.raises(ValueError, match="Unknown Eikonal acquisition receiver"):
        job.to_fs()
