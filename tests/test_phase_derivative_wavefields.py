import json

import numpy as np
import pytest
import xarray as xr

from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.project.project import Project
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverNode
from frequensolve.simulation.jobs import BaseJob, FrequencyDomainJob, TimeDomainJob
from frequensolve.simulation.outputs import WavefieldOutput


def _elastic_simulation(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    simulation = project.new_simulation(
        name="elastic",
        physics="elastic",
        dimension=2,
    )
    simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )
    acquisition = Acquisition()
    acquisition.add_sources(kind="vector", coords=[[0.5, 0.5]])
    receiver = ReceiverNode(name="geophone")
    receiver.add_component(name="vz", field="velocity_z")
    acquisition.add_receiver_group("surface", receiver, [[0.5, 0.75]])
    simulation.acquisition = acquisition
    simulation.save()
    return simulation


def _grid():
    return xr.DataArray(
        np.empty((2, 2)),
        dims=("z", "x"),
        coords={"z": [0.0, 1.0], "x": [0.0, 1.0]},
    )


@pytest.mark.parametrize("order", [-1, 5, 1.5, True, "3"])
def test_frequency_domain_job_rejects_invalid_phase_derivative_orders(tmp_path, order):
    simulation = _elastic_simulation(tmp_path)

    with pytest.raises(ValueError, match="integer from 0 to 4"):
        FrequencyDomainJob(
            name="derivatives",
            simulation=simulation,
            f_list=[10.0],
            phase_derivatives=order,
        )


def test_frequency_domain_job_serializes_and_loads_phase_derivatives(tmp_path):
    simulation = _elastic_simulation(tmp_path)
    job = FrequencyDomainJob(
        name="derivatives",
        simulation=simulation,
        f_list=[10.0],
        phase_derivatives=4,
    )

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert payload["workflow"] == "forward_df"
    assert payload["derivative_order"] == 4
    assert loaded.phase_derivatives == 4
    assert loaded.workflow == "forward_df"


def test_time_domain_job_combines_laplace_damping_and_phase_derivatives(tmp_path):
    simulation = _elastic_simulation(tmp_path)
    job = TimeDomainJob(
        name="damped_derivatives",
        simulation=simulation,
        f_min=0.0,
        f_max=4.0,
        df=1.0,
        laplace=-0.25,
        phase_derivatives=3,
        outputs=WavefieldOutput(
            name="s_velocity",
            field="s_velocity",
            grid=_grid(),
        ),
    )

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert job.workflow == "forward_df"
    assert job.phase_derivatives == 3
    assert [value.imag for value in job.f_list] == pytest.approx([-0.25] * 4)
    assert payload["derivative_order"] == 3
    assert payload["f_list"] == [
        [1.0, -0.25],
        [2.0, -0.25],
        [3.0, -0.25],
        [4.0, -0.25],
    ]
    assert job.trace_outputs.groups == [
        "surface",
        "surface_df",
        "surface_d2f",
        "surface_d3f",
    ]
    assert job.wavefield_trace_outputs.groups == [
        "s_velocity",
        "s_velocity_df",
        "s_velocity_d2f",
        "s_velocity_d3f",
    ]
    assert loaded.phase_derivatives == 3
    assert loaded.f_list == job.f_list


def test_standard_time_domain_taper_enables_first_phase_derivative(tmp_path):
    simulation = _elastic_simulation(tmp_path)
    job = TimeDomainJob(
        name="tapered",
        simulation=simulation,
        f_min=0.0,
        f_max=4.0,
        df=1.0,
        high_frequency_taper=True,
    )

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert job.reconstruction == "standard"
    assert job.workflow == "forward_df"
    assert job.phase_derivatives == 1
    assert payload["derivative_order"] == 1
    assert payload["time_reconstruction"] == job.time_reconstruction
    assert job.time_reconstruction["high_frequency_taper"] is True
    assert loaded.phase_derivatives == 1
    assert loaded.time_reconstruction == job.time_reconstruction


def test_time_domain_sample_every_is_specific_to_hermite_reconstruction(tmp_path):
    simulation = _elastic_simulation(tmp_path)

    for invalid in (0, 1.5, True):
        with pytest.raises(ValueError, match="sample_every must be an integer"):
            TimeDomainJob(
                name="invalid_sampling",
                simulation=simulation,
                f_max=4.0,
                df=1.0,
                reconstruction="hermite",
                sample_every=invalid,
            )
    with pytest.raises(ValueError, match="only used by Hermite"):
        TimeDomainJob(
            name="invalid_sampling",
            simulation=simulation,
            f_max=4.0,
            df=1.0,
            sample_every=2,
        )


def test_phase_derivative_order_changes_job_and_task_fingerprints(tmp_path):
    simulation = _elastic_simulation(tmp_path)
    first = FrequencyDomainJob(
        name="derivatives",
        simulation=simulation,
        f_list=[10.0],
        phase_derivatives=1,
    )
    fourth = FrequencyDomainJob(
        name="derivatives",
        simulation=simulation,
        f_list=[10.0],
        phase_derivatives=4,
    )

    assert first.fingerprint() != fourth.fingerprint()
    assert first.task_fingerprint(1) != fourth.task_fingerprint(1)


def test_phase_derivative_helmholtz_wavefields_are_exposed_as_groups(tmp_path):
    simulation = _elastic_simulation(tmp_path)
    job = FrequencyDomainJob(
        name="derivatives",
        simulation=simulation,
        f_list=[10.0],
        phase_derivatives=4,
        outputs=[
            WavefieldOutput(name="p_velocity", field="p_velocity", grid=_grid()),
            WavefieldOutput(name="s_velocity", field="s_velocity", grid=_grid()),
        ],
    )

    report = job.validate()
    payload = job.to_fs()

    assert report.ok
    assert [output["field"] for output in payload["Outputs"]["wavefields"]] == [
        "p_velocity",
        "s_velocity",
    ]
    assert job.wavefield_trace_outputs.groups == [
        "p_velocity",
        "p_velocity_df",
        "p_velocity_d2f",
        "p_velocity_d3f",
        "p_velocity_d4f",
        "s_velocity",
        "s_velocity_df",
        "s_velocity_d2f",
        "s_velocity_d3f",
        "s_velocity_d4f",
    ]
    assert job.wavefield_outputs["s_velocity_d3f"]["fields"] == ["s_velocity"]
    assert job.wavefield_outputs["s_velocity_d3f"]["phase_derivative_order"] == 3
    assert job.wavefield_outputs["s_velocity_d3f"]["base_wavefield"] == "s_velocity"


def test_base_frequency_job_does_not_advertise_derivative_wavefields(tmp_path):
    simulation = _elastic_simulation(tmp_path)
    job = FrequencyDomainJob(
        name="base",
        simulation=simulation,
        f_list=[10.0],
        outputs=WavefieldOutput(
            name="s_velocity",
            field="s_velocity",
            grid=_grid(),
        ),
    )

    payload = job.to_fs()

    assert payload["workflow"] == "forward"
    assert "derivative_order" not in payload
    assert job.wavefield_trace_outputs.groups == ["s_velocity"]
    assert list(job.wavefield_outputs) == ["s_velocity"]
