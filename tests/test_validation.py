import pytest

from frequensolve.geometry.frame import Axis, CoordinateSystem
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.layered import LayeredModel
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverNode
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.outputs import ParaviewOutput, WavefieldOutput
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import Q_
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
    if source_coords is None:
        source_coords = [[0.5, 0.25]]
    if receiver_coords is None:
        receiver_coords = [[0.25, 0.75], [0.75, 0.75]]
    acquisition = Acquisition()
    acquisition.add_source_group(kind=source_kind, coords=source_coords)
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


def test_validation_converts_pint_receiver_units_for_domain_checks(tmp_path):
    job = _simple_job(
        tmp_path,
        receiver_coords=Q_([[250.0, 750.0], [750.0, 750.0]], "m"),
    )
    job.simulation.units.defaults["length"] = "km"

    report = job.validate()

    assert report.ok


def test_validation_accepts_axis_suffixed_paraview_fields(tmp_path):
    job = _simple_job(tmp_path)
    job += ParaviewOutput(name="pv", fields=["pressure", "velocity_z"])

    report = job.validate()

    assert report.ok


def test_validation_accepts_solver_builtin_paraview_subdomain_property(tmp_path):
    job = _simple_job(tmp_path)
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="bottom", depth=1.0)
    job.simulation += model
    job += ParaviewOutput(
        name="pv",
        fields=["pressure"],
        properties=["vp", "Subdomain"],
    )

    report = job.validate()

    assert report.ok
    assert "outputs.property.unknown" not in _codes(report)


@pytest.mark.parametrize("field", ["pressure_z", "velocity_zz"])
def test_validation_rejects_invalid_axis_suffixed_paraview_fields(tmp_path, field):
    job = _simple_job(tmp_path)
    job += ParaviewOutput(name="pv", fields=[field])

    report = job.validate()

    assert not report.ok
    assert "field.unsupported" in _codes(report)


def test_validation_catches_receiver_field_typos(tmp_path):
    job = _simple_job(tmp_path, receiver_field="presure")

    report = job.validate()

    assert not report.ok
    assert "field.unsupported" in _codes(report)
    with pytest.raises(ValidationError, match="presure"):
        job.validate(raise_errors=True)


def test_validation_catches_bad_source_kind(tmp_path):
    job = _simple_job(tmp_path, source_kind="moment")

    report = job.validate()

    assert not report.ok
    assert "acquisition.source.kind.unsupported" in _codes(report)
    with pytest.raises(ValidationError, match="Use 'tensor'"):
        job.validate(raise_errors=True)


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


def test_validation_catches_invalid_coordinate_units(tmp_path):
    job = _simple_job(tmp_path)
    group = job.simulation.acquisition.receiver_groups[0]
    group.coordinates.units = "definitely_not_a_unit"

    report = job.validate()

    assert not report.ok
    assert "acquisition.receiver_coordinates.units.invalid" in _codes(report)


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


def test_site_prepare_job_blocks_invalid_jobs_before_submit(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])

    with pytest.raises(ValidationError, match="outside model domain"):
        BaseSite().prepare_job(job)


def test_site_prepare_job_can_skip_validation(tmp_path):
    job = _simple_job(tmp_path, source_coords=[[1.5, 0.25]])

    assert BaseSite().prepare_job(job, validate=False) is job
