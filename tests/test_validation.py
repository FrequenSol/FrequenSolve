import logging

import h5py
import numpy as np
import pytest

from frequensolve.geometry.frame import Axis, CoordinateSystem
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.layered import LayeredModel
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import CoordsFromFile, ReceiverNode
from frequensolve.seismic.sources import SourceEncoding, SourceGeometry
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.outputs import VtkOutput, WavefieldOutput
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import ureg
from frequensolve.validation import ValidationError


def _simple_job(
    tmp_path,
    *,
    physics="acoustic",
    source_kind="scalar",
    source_coords=None,
    receiver_coords=None,
    receiver_field="pressure",
):
    source_coords = source_coords or [[0.5, 0.25]]
    receiver_coords = receiver_coords or [[0.25, 0.75], [0.75, 0.75]]
    acquisition = Acquisition()
    acquisition.add_sources(kind=source_kind, coords=source_coords)
    device = ReceiverNode()
    device.add_component(name="component", field=receiver_field)
    acquisition.add_receiver_group("receivers", device, receiver_coords)

    simulation = SeismicSimulation(
        name="simple",
        physics=physics,
        dimension=2,
        project_path=tmp_path,
    )
    simulation.acquisition = acquisition
    simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )
    return FrequencyDomainJob(name="freq", simulation=simulation, f_list=[10.0])


def _codes(report):
    return {issue.code for issue in report.issues}


def test_valid_job_has_clean_validation_report(tmp_path):
    job = _simple_job(tmp_path)

    report = job.validate()

    assert report.ok
    assert report.issues == []


def test_surface_carpet_on_surface_coordinates_validate_in_3d(tmp_path):
    simulation = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=3,
        project_path=tmp_path,
    )
    simulation.mesh = MeshManager(
        HexMeshGenerator(
            l_bound=[0.0, 0.0, -1.0],
            u_bound=[1.0, 1.0, 1.0],
            n=[1, 1, 1],
        )
    )
    surface = simulation.model_surface("top")
    device = ReceiverNode()
    device.add_component(name="component", field="pressure")
    acquisition = Acquisition()
    acquisition.add_sources(
        kind="scalar",
        coords=surface.points_grid(x=[0.25, 0.75], y=[0.5]),
    )
    acquisition.add_receiver_carpet(
        "surface",
        device,
        surface=surface,
        x=[0.25, 0.75],
        y=[0.5],
    )
    simulation.acquisition = acquisition
    job = FrequencyDomainJob(name="freq", simulation=simulation, f_list=[10.0])

    report = job.validate()

    assert report.ok
    assert report.issues == []


def test_validation_catches_source_outside_mesh_domain(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])

    report = job.validate()

    assert not report.ok
    assert "coordinates.domain.outside" in _codes(report)
    with pytest.raises(ValidationError, match="outside model domain"):
        job.validate(raise_errors=True)


def test_validation_uses_default_length_units_for_coordinate_domain_checks(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])
    job.simulation.units.defaults["length"] = "km"
    job.simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1000.0, 1000.0], units="m")
    )

    report = job.validate()

    assert not report.ok
    assert "coordinates.domain.outside" in _codes(report)


def test_validation_catches_receiver_field_typos(tmp_path):
    job = _simple_job(tmp_path, receiver_field="presure")

    report = job.validate()

    assert not report.ok
    assert "field.unsupported" in _codes(report)
    with pytest.raises(ValidationError, match="presure"):
        job.validate(raise_errors=True)


def test_validation_accepts_component_and_derived_field_selectors(tmp_path):
    job = _simple_job(tmp_path, physics="elastic", receiver_field="velocity_z")
    job += WavefieldOutput(
        fields=["velocity_div", "velocity_curl", "strain_zz"],
        dims=("z", "x"),
        coords={"z": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    report = job.validate()

    assert report.ok


def test_validation_catches_bad_source_kind(tmp_path):
    with pytest.raises(ValueError, match="Unsupported source kind"):
        _simple_job(tmp_path, source_kind="moment")


def test_validation_catches_point_kind_that_differs_from_geometry(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.acquisition.source_geometry.sources[0].kind = "vector"

    report = job.validate()

    assert not report.ok
    assert "acquisition.source.kind.mismatch" in _codes(report)


def test_validation_catches_wavefield_grid_outside_domain(tmp_path):
    job = _simple_job(tmp_path)
    job += WavefieldOutput(
        field="pressure",
        dims=("z", "r"),
        coords={"z": [0.0, 2.0], "r": [0.0, 1.0]},
    )

    report = job.validate()

    assert not report.ok
    assert "coordinates.domain.outside" in _codes(report)


def test_validation_catches_unknown_grid_dimension(tmp_path):
    job = _simple_job(tmp_path)
    job += WavefieldOutput(
        field="pressure",
        dims=("distance", "z"),
        coords={"distance": [0.0, 1.0], "z": [0.0, 1.0]},
    )

    report = job.validate()

    assert not report.ok
    assert "grid.axis.unsupported" in _codes(report)


def test_validation_accepts_grid_dimensions_from_registered_frame(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.add_coordinate_system(
        CoordinateSystem(
            type="cartesian",
            name="section",
            axes=[Axis("station", "x"), Axis("depth", "z")],
        )
    )
    job += WavefieldOutput(
        field="pressure",
        dims=("depth", "station"),
        coords={"depth": [0.0, 1.0], "station": [0.0, 1.0]},
        system="section",
    )

    report = job.validate()

    assert report.ok


def test_validation_accepts_fixed_axis_wavefield_plane_in_3d(tmp_path):
    acquisition = Acquisition()
    acquisition.add_sources(kind="scalar", coords=[[0.5, 0.5, 0.5]])
    device = ReceiverNode()
    device.add_component(name="component", field="pressure")
    acquisition.add_receiver_group(
        "receivers",
        device,
        [[0.25, 0.5, 0.75], [0.75, 0.5, 0.75]],
    )

    simulation = SeismicSimulation(
        name="simple_3d",
        physics="acoustic",
        dimension=3,
        project_path=tmp_path,
    )
    simulation.acquisition = acquisition
    simulation.mesh = MeshManager(
        HexMeshGenerator(
            l_bound=[0.0, 0.0, 0.0],
            u_bound=[1.0, 1.0, 1.0],
            n=[1, 1, 1],
            units="m",
        )
    )
    simulation.add_coordinate_system(
        CoordinateSystem.cartesian(
            name="x0_plane",
            axes=[Axis("z", "z"), Axis("y", "y")],
            ndim=2,
            fixed_axis="x",
            fixed_value=0.5 * ureg.m,
        )
    )
    job = FrequencyDomainJob(name="freq", simulation=simulation, f_list=[10.0])
    job += WavefieldOutput(
        field="pressure",
        dims=("z", "y"),
        coords={"z": [0.0, 1.0], "y": [0.0, 1.0]},
        units="m",
        system="x0_plane",
    )

    report = job.validate()

    assert report.ok


def test_validation_catches_invalid_coordinate_units(tmp_path):
    job = _simple_job(tmp_path)
    group = job.simulation.acquisition.receiver_groups[0]
    group.coordinates.units = "definitely_not_a_unit"

    report = job.validate()

    assert not report.ok
    assert "acquisition.receiver_coordinates.units.invalid" in _codes(report)


def test_validation_accepts_pint_unit_objects_in_coordinate_metadata(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1], units="m")
    )
    group = job.simulation.acquisition.receiver_groups[0]
    group.coordinates.units = ureg.meter

    report = job.validate()

    assert report.ok


def test_validation_catches_wavefield_source_id_range(tmp_path):
    job = _simple_job(tmp_path)
    job += WavefieldOutput(
        field="pressure",
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
        sources=[2],
    )

    report = job.validate()

    assert not report.ok
    assert "outputs.source_id.out_of_range" in _codes(report)


def test_validation_requires_k_list_for_half_dimension_jobs(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.dimension = 2.5

    report = job.validate()

    assert not report.ok
    assert "job.k_list.required" in _codes(report)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("k_list", [[-0.01, 0.0, 0.01]], "job.k_list.invalid"),
        ("k_list", [float("nan")], "job.k_list.invalid"),
        ("k_list", ["not-a-number"], "job.k_list.invalid"),
        ("k_weights", [[0.5, 1.0, 0.5]], "job.k_weights.invalid"),
        ("k_weights", [float("inf")], "job.k_weights.invalid"),
        ("k_units", "m", "job.k_units.incompatible"),
        ("k_units", "not-a-unit", "job.k_units.incompatible"),
    ],
)
def test_validation_rechecks_mutated_wavenumber_contract(tmp_path, field, value, code):
    job = _simple_job(tmp_path)
    job.simulation.dimension = 2.5
    job.k_list = [-0.01, 0.0, 0.01]
    setattr(job, field, value)

    report = job.validate()

    assert not report.ok
    assert code in _codes(report)


def test_validation_skips_source_range_when_external_count_unknown(tmp_path):
    job = _simple_job(tmp_path)
    source_file = tmp_path / "sources.h5"
    with h5py.File(source_file, "w") as h5:
        h5.create_dataset("source_points", data=[[0.5, 0.25]])
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        source_file,
        dataset="source_points",
        kind="scalar",
    )
    job += WavefieldOutput(
        field="pressure",
        dims=("z", "r"),
        coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
        sources=[2],
    )

    report = job.validate()

    assert report.ok
    assert "outputs.source_id.out_of_range" not in _codes(report)


def test_validation_rejects_missing_external_source_geometry_file(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        "missing-sources.h5",
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate()

    assert not report.ok
    assert "acquisition.source_geometry.file_missing" in _codes(report)
    assert "missing-sources.h5" not in report.format()


def test_validation_warns_for_unverified_remote_source_file(tmp_path):
    job = _simple_job(tmp_path)
    remote_file = tmp_path.parent / "cluster-only" / "sources.h5"
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        remote_file,
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate(allow_unverified_remote_files=True)

    assert report.ok
    assert "files.remote_unverified" in _codes(report)
    assert "acquisition.source_geometry.file_missing" not in _codes(report)


def test_validation_checks_existing_external_source_file_for_remote_site(tmp_path):
    job = _simple_job(tmp_path)
    external_file = tmp_path.parent / "external-sources.h5"
    with h5py.File(external_file, "w"):
        pass
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        external_file,
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate(allow_unverified_remote_files=True)

    assert not report.ok
    assert "acquisition.source_geometry.dataset_missing" in _codes(report)
    assert "files.remote_unverified" not in _codes(report)


def test_validation_still_rejects_missing_project_file_for_remote_site(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        tmp_path / "missing-sources.h5",
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate(allow_unverified_remote_files=True)

    assert not report.ok
    assert "acquisition.source_geometry.file_missing" in _codes(report)


def test_validation_rejects_relative_source_path_escaping_remote_project(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        "../cluster-only/sources.h5",
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate(allow_unverified_remote_files=True)

    assert not report.ok
    assert "acquisition.source_geometry.file_missing" in _codes(report)
    assert "files.remote_unverified" not in _codes(report)


def test_validation_warns_for_unverified_remote_receiver_file(tmp_path):
    job = _simple_job(tmp_path)
    remote_file = tmp_path.parent / "cluster-only" / "receivers.h5"
    job.simulation.acquisition.receiver_groups[0].coordinates = CoordsFromFile(
        file=remote_file,
        format="HDF5",
        dset="coords",
    )

    report = job.validate(allow_unverified_remote_files=True)

    assert report.ok
    assert "files.remote_unverified" in _codes(report)
    assert "acquisition.receiver_coordinates.file_missing" not in _codes(report)


def test_validation_reads_project_relative_receiver_file(tmp_path):
    job = _simple_job(tmp_path)
    receiver_file = tmp_path / "data" / "receivers.h5"
    receiver_file.parent.mkdir()
    with h5py.File(receiver_file, "w") as h5:
        h5.create_dataset("coords", data=[[0.25, 0.75], [0.75, 0.75]])
    job.simulation.acquisition.receiver_groups[0].coordinates = CoordsFromFile(
        file="data/receivers.h5",
        format="HDF5",
        dset="coords",
    )

    report = job.validate()

    assert report.ok
    assert report.issues == []


def test_validation_rejects_missing_external_source_geometry_dataset(tmp_path):
    job = _simple_job(tmp_path)
    source_file = tmp_path / "sources.h5"
    with h5py.File(source_file, "w") as h5:
        h5.create_dataset("different", data=[[0.5, 0.25]])
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        source_file,
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate()

    assert not report.ok
    assert "acquisition.source_geometry.dataset_missing" in _codes(report)


def test_validation_rejects_external_source_geometry_outside_domain(tmp_path):
    job = _simple_job(tmp_path)
    source_file = tmp_path / "sources.h5"
    with h5py.File(source_file, "w") as h5:
        h5.create_dataset("source_points", data=[[1.5, 0.25]])
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        source_file,
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate()

    assert not report.ok
    assert "coordinates.domain.outside" in _codes(report)


def test_validation_rejects_nonfinite_external_source_geometry(tmp_path):
    job = _simple_job(tmp_path)
    source_file = tmp_path / "sources.h5"
    with h5py.File(source_file, "w") as h5:
        h5.create_dataset("source_points", data=[[np.nan, 0.25]])
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        source_file,
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate()

    assert not report.ok
    assert "acquisition.source_geometry.dataset_nonfinite" in _codes(report)


def test_validation_rejects_nonnumeric_external_source_geometry(tmp_path):
    job = _simple_job(tmp_path)
    source_file = tmp_path / "sources.h5"
    with h5py.File(source_file, "w") as h5:
        h5.create_dataset("source_points", data=[["0.5", "0.25"]])
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        source_file,
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate()

    assert not report.ok
    assert "acquisition.source_geometry.dataset_numeric" in _codes(report)


def test_validation_rejects_complex_external_source_geometry(tmp_path):
    job = _simple_job(tmp_path)
    source_file = tmp_path / "sources.h5"
    with h5py.File(source_file, "w") as h5:
        h5.create_dataset(
            "source_points",
            data=np.asarray([[0.5 + 0.1j, 0.25 + 0.0j]]),
        )
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        source_file,
        dataset="source_points",
        kind="scalar",
    )

    report = job.validate()

    assert not report.ok
    assert "acquisition.source_geometry.dataset_numeric" in _codes(report)


def test_validation_rejects_inconsistent_external_source_encoding_metadata(tmp_path):
    job = _simple_job(tmp_path)
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=np.ones((2, 1, 2)))
        h5.create_dataset("field_names", data=np.asarray([b"only-one"]))
        h5.create_dataset("reference_coordinates", data=np.ones((2, 3)))
    job.simulation.acquisition.source_encoding = SourceEncoding.hdf5(
        encoding_file,
        dataset="coefficients",
        field_names_dataset="field_names",
        reference_coordinates_dataset="reference_coordinates",
        count=2,
    )

    report = job.validate()

    assert not report.ok
    assert "acquisition.source_encoding.field_names.length_mismatch" in _codes(report)
    assert "acquisition.source_encoding.reference_coordinates.shape" in _codes(report)


def test_validation_rejects_complex_external_source_reference_coordinates(tmp_path):
    job = _simple_job(tmp_path)
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=np.ones((1, 1, 2)))
        h5.create_dataset(
            "reference_coordinates",
            data=np.asarray([[0.5 + 0.1j, 0.25 + 0.0j]]),
        )
    job.simulation.acquisition.source_encoding = SourceEncoding.hdf5(
        encoding_file,
        dataset="coefficients",
        reference_coordinates_dataset="reference_coordinates",
        count=1,
    )

    report = job.validate()

    assert not report.ok
    assert (
        "acquisition.source_encoding.reference_coordinates.dataset_numeric"
        in _codes(report)
    )


def test_validation_accepts_paired_complex_source_encoding_coefficients(tmp_path):
    job = _simple_job(tmp_path)
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=np.asarray([[[1.0, 0.5]]]))
    job.simulation.acquisition.source_encoding = SourceEncoding.hdf5(
        encoding_file,
        dataset="coefficients",
        count=1,
    )

    report = job.validate()

    assert report.ok
    assert "acquisition.source_encoding.dataset_numeric" not in _codes(report)


def test_validation_rejects_unpaired_source_encoding_coefficients(tmp_path):
    job = _simple_job(tmp_path)
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=np.ones((1, 1)))
    job.simulation.acquisition.source_encoding = SourceEncoding.hdf5(
        encoding_file,
        dataset="coefficients",
        count=1,
    )

    report = job.validate()

    assert not report.ok
    assert "acquisition.source_encoding.dataset_shape" in _codes(report)


@pytest.mark.parametrize(
    ("coefficients", "code"),
    [
        (
            [[["not-a-coefficient", "0"]]],
            "acquisition.source_encoding.dataset_numeric",
        ),
        ([[[np.inf, 0.0]]], "acquisition.source_encoding.dataset_nonfinite"),
        (
            np.asarray([[[1.0 + 0.5j, 0.0 + 0.0j]]]),
            "acquisition.source_encoding.dataset_numeric",
        ),
    ],
)
def test_validation_rejects_invalid_external_source_encoding_coefficients(
    tmp_path,
    coefficients,
    code,
):
    job = _simple_job(tmp_path)
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=coefficients)
    job.simulation.acquisition.source_encoding = SourceEncoding.hdf5(
        encoding_file,
        dataset="coefficients",
        count=1,
    )

    report = job.validate()

    assert not report.ok
    assert code in _codes(report)


@pytest.mark.parametrize(
    ("field_names", "code"),
    [
        ([b"field", b""], "acquisition.source_encoding.field_names.value_invalid"),
        ([b"field", b"field"], "acquisition.source_encoding.field_names.duplicate"),
        ([1, 2], "acquisition.source_encoding.field_names.value_invalid"),
    ],
)
def test_validation_rejects_invalid_external_source_encoding_field_names(
    tmp_path,
    field_names,
    code,
):
    job = _simple_job(tmp_path)
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=np.ones((2, 1, 2)))
        h5.create_dataset("field_names", data=field_names)
    job.simulation.acquisition.source_encoding = SourceEncoding.hdf5(
        encoding_file,
        dataset="coefficients",
        field_names_dataset="field_names",
        count=2,
    )

    report = job.validate()

    assert not report.ok
    assert code in _codes(report)


def test_validation_rejects_external_encoding_reference_outside_domain(tmp_path):
    job = _simple_job(tmp_path)
    encoding_file = tmp_path / "encoding.h5"
    with h5py.File(encoding_file, "w") as h5:
        h5.create_dataset("coefficients", data=np.ones((1, 1, 2)))
        h5.create_dataset("reference_coordinates", data=[[1.5, 0.25]])
    job.simulation.acquisition.source_encoding = SourceEncoding.hdf5(
        encoding_file,
        dataset="coefficients",
        reference_coordinates_dataset="reference_coordinates",
        count=1,
    )

    report = job.validate()

    assert not report.ok
    assert "coordinates.domain.outside" in _codes(report)


def test_validation_rejects_encoded_source_reference_outside_domain(tmp_path):
    job = _simple_job(tmp_path)
    job.simulation.acquisition.source_encoding = SourceEncoding.dense(
        [[1.0]],
        names=["outside"],
        reference_coordinates=[[1.5, 0.25]],
    )

    report = job.validate()

    assert not report.ok
    assert "coordinates.domain.outside" in _codes(report)


def _job_with_layered_model(tmp_path):
    job = _simple_job(tmp_path)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="bottom", depth=1.0)
    job.simulation += model
    return job


def test_validation_accepts_known_paraview_model_surface(tmp_path):
    job = _job_with_layered_model(tmp_path)
    job += VtkOutput.surface(surfaces=["top", 2], fields=["pressure"])

    report = job.validate()

    assert report.ok
    assert "outputs.vtk.model_surface.unknown" not in _codes(report)


def test_validation_accepts_solver_model_surface_aliases(tmp_path):
    job = _job_with_layered_model(tmp_path)
    job += VtkOutput.surface(
        surfaces=["TOP", "bottom", "surface_1", "surface_2"],
        fields=["pressure"],
    )

    report = job.validate()

    assert report.ok
    assert "outputs.vtk.model_surface.unknown" not in _codes(report)


@pytest.mark.parametrize(
    ("fracture_names", "expected_names"),
    [
        ({}, ["fault_top", "FAULT_BOTTOM"]),
        (
            {"top_name": "fault_upper", "bottom_name": "fault_lower"},
            ["fault_upper", "FAULT_LOWER"],
        ),
    ],
)
def test_validation_accepts_generated_fracture_surface_references(
    tmp_path,
    fracture_names,
    expected_names,
):
    job = _simple_job(tmp_path)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="topography", depth=0.0)
    model.add_layer(name="upper", properties={"vp": 1.5, "rho": 1.0})
    model.add_fracture(
        name="fault",
        depth=0.5,
        gap=[0.01],
        **fracture_names,
    )
    model.add_layer(name="lower", properties={"vp": 1.8, "rho": 1.1})
    model.add_surface(name="basement", depth=1.0)
    job.simulation += model
    job += VtkOutput.surface(
        surfaces=[
            "top",
            "basement",
            *expected_names,
            "surface_2",
            "surface_3",
            4,
        ],
        fields=["pressure"],
    )

    report = job.validate()

    assert report.ok
    assert "outputs.vtk.model_surface.unknown" not in _codes(report)
    assert "outputs.vtk.model_surface.index_out_of_range" not in _codes(report)


def test_validation_rejects_borehole_surface_as_paraview_model_surface(tmp_path):
    job = _job_with_layered_model(tmp_path)
    borehole = job.simulation.model.add_borehole(name="bh1", x=0.45)
    borehole.add_layer(
        "fluid",
        physics="acoustic",
        properties={"vp": 1.48, "rho": 1.03},
    )
    borehole.add_surface("fluid_wall", r=0.035)
    job += VtkOutput.surface(surfaces="BH1_FLUID_WALL", fields=["pressure"])

    report = job.validate()

    assert not report.ok
    assert "outputs.vtk.model_surface.unknown" in _codes(report)


def test_validation_rejects_bottom_alias_without_named_bottom_horizon(tmp_path):
    job = _simple_job(tmp_path)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="topography", depth=0.0)
    model.add_layer(name="layer", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="basement", depth=1.0)
    job.simulation += model
    job += VtkOutput.surface(surfaces="bottom", fields=["pressure"])

    report = job.validate()

    assert not report.ok
    assert "outputs.vtk.model_surface.unknown" in _codes(report)


def test_validation_rejects_unknown_paraview_model_surface(tmp_path):
    job = _job_with_layered_model(tmp_path)
    job += VtkOutput.surface(surfaces="not-a-surface", fields=["pressure"])

    report = job.validate()

    assert not report.ok
    assert "outputs.vtk.model_surface.unknown" in _codes(report)
    assert "Available solver surface references are:" in report.format()
    assert "bottom" in report.format()
    assert "top" in report.format()


def test_validation_rejects_out_of_range_paraview_model_surface_index(tmp_path):
    job = _job_with_layered_model(tmp_path)
    job += VtkOutput.surface(surfaces=3, fields=["pressure"])

    report = job.validate()

    assert not report.ok
    assert "outputs.vtk.model_surface.index_out_of_range" in _codes(report)


def test_validation_uses_expanded_fracture_surface_index_range(tmp_path):
    job = _simple_job(tmp_path)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", properties={"vp": 1.5, "rho": 1.0})
    model.add_fracture(name="fault", depth=0.5, gap=[0.01])
    model.add_layer(name="lower", properties={"vp": 1.8, "rho": 1.1})
    model.add_surface(name="bottom", depth=1.0)
    job.simulation += model
    job += VtkOutput.surface(surfaces=4, fields=["pressure"])

    report = job.validate()

    assert "outputs.vtk.model_surface.index_out_of_range" not in _codes(report)


@pytest.mark.parametrize("selection", [True, 1.5])
def test_validation_rejects_noninteger_paraview_model_surface_index(
    tmp_path,
    selection,
):
    job = _job_with_layered_model(tmp_path)
    job += VtkOutput.surface(
        surfaces={"kind": "model_surface", "index": selection},
        fields=["pressure"],
    )

    report = job.validate()

    assert not report.ok
    assert "outputs.vtk.model_surface.index_invalid" in _codes(report)


def test_site_prepare_job_blocks_invalid_jobs_before_submit(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])

    with pytest.raises(ValidationError, match="outside model domain"):
        BaseSite().prepare_job(job)
    assert job._file is None
    assert job.simulation._file is None


def test_site_prepare_job_can_skip_validation(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])

    assert BaseSite().prepare_job(job, validate=False) is job


def test_remote_site_submit_logs_unverified_source_once_per_validation(
    tmp_path, caplog
):
    class RemoteFileSite(BaseSite):
        def _job_validation_options(self, job):
            return {"allow_unverified_remote_files": True}

        def submit(self, job):
            self.prepare_job(job)
            return self.prepare_job(job, sync_project=True, validate=False)

    job = _simple_job(tmp_path)
    job.simulation.acquisition.sources = SourceGeometry.hdf5(
        tmp_path.parent / "cluster-only" / "sources.h5",
        dataset="source_points",
        kind="scalar",
    )
    site = RemoteFileSite()

    with caplog.at_level(logging.WARNING, logger=site.__class__.__module__):
        assert site.submit(job) is job

    remote_warnings = [
        record
        for record in caplog.records
        if "files.remote_unverified" in record.message
    ]
    assert len(remote_warnings) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=site.__class__.__module__):
        assert site.prepare_job(job) is job

    remote_warnings = [
        record
        for record in caplog.records
        if "files.remote_unverified" in record.message
    ]
    assert len(remote_warnings) == 1
