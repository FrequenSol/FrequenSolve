import pytest

from frequensolve.geometry.frame import Axis, CoordinateSystem
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverNode
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.outputs import WavefieldOutput
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


def test_site_prepare_job_blocks_invalid_jobs_before_submit(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])

    with pytest.raises(ValidationError, match="outside model domain"):
        BaseSite().prepare_job(job)


def test_site_prepare_job_can_skip_validation(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])

    assert BaseSite().prepare_job(job, validate=False) is job
