import json

import h5py
import numpy as np
import pytest
import xarray as xr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.representation import VariationalSmoothing
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import (
    EncodedReceiver,
    ReceiverComponent,
    ReceiverNode,
)
from frequensolve.seismic.sources import SourceGeometry
from frequensolve.simulation.jobs import BaseJob, FrequencyDomainJob
from frequensolve.simulation.jobs.fwi import DataSpace, ModelSpace
from frequensolve.simulation.jobs.imaging import (
    HDF5TraceStore,
    ImageDatabase,
    ImagingJob,
    LSRTMGradientJob,
    LSRTMNormalJob,
    MisfitComparison,
    MisfitGroup,
    ObservedTraceDerivatives,
)
from frequensolve.simulation.outputs import VtkOutput
from frequensolve.simulation.simulation import SeismicSimulation


def _elastic_simulation(tmp_path):
    sim = SeismicSimulation(
        name="smooth",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.model.x_limits = [0.0, 1.0]
    sim.model.z_limits = [0.0, 1.0]
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )

    acq = Acquisition()
    acq.add_sources(kind="vector", coords=np.array([[0.5, 0.1]]), direction=[0.0, 1.0])
    device = ReceiverNode(
        name="geophone",
        components=[ReceiverComponent(name="vz", field="velocity")],
    )
    acq.add_receiver_group(
        name="surface",
        device=device,
        coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    sim.acquisition = acq
    return sim


def test_model_space_packs_parameters_in_stable_order_and_round_trips():
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[2.0, 1.0])
    space = ModelSpace(grid, parameters=["Vp", "Vs", "Rho"])
    coords = space.coords
    model = xr.Dataset(
        {
            "vp": xr.DataArray(
                np.arange(6).reshape(2, 3),
                dims=space.dims,
                coords=coords,
            ),
            "vs": xr.DataArray(
                np.arange(6, 12).reshape(2, 3),
                dims=space.dims,
                coords=coords,
            ),
            "rho": xr.DataArray(
                np.arange(12, 18).reshape(2, 3),
                dims=space.dims,
                coords=coords,
            ),
        }
    )

    vector = space.pack(model)
    roundtrip = space.unpack(vector)

    assert space.parameters == ("vp", "vs", "rho")
    np.testing.assert_array_equal(vector[:6], np.arange(6))
    np.testing.assert_array_equal(vector[6:12], np.arange(6, 12))
    np.testing.assert_array_equal(roundtrip["rho"].values, model["rho"].values)
    assert roundtrip["vp"].dims == ("z", "x")


def test_data_space_packs_trace_groups_in_frequency_source_component_receiver_order(
    tmp_path,
):
    sim = _elastic_simulation(tmp_path)
    space = DataSpace.from_simulation(sim, frequencies=[5.0, 10.0])
    values = np.arange(space.size, dtype=np.float64).reshape(2, 1, 1, 2)

    vector = space.pack({"surface": values})
    roundtrip = space.unpack(vector)

    assert space.size == 4
    np.testing.assert_array_equal(vector, np.arange(4))
    np.testing.assert_array_equal(roundtrip["surface"].values, values)
    assert roundtrip["surface"].dims == (
        "frequency",
        "source",
        "component",
        "receiver",
    )
    assert roundtrip["surface"].coords["component"].values.tolist() == ["vz"]


def test_data_space_uses_encoded_receiver_outputs_and_reduced_geometry(tmp_path):
    sim = _elastic_simulation(tmp_path)
    sim.acquisition.receiver_groups[0].device = EncodedReceiver(
        components=[ReceiverComponent(name="vz", field="velocity")],
        weights=np.ones((2, 1, 2), dtype=np.complex64),
        encoding_names=["focus_a", "focus_b"],
    )

    space = DataSpace.from_simulation(sim, frequencies=[5.0])

    assert space.segments[0].components == ("focus_a", "focus_b")
    assert space.segments[0].receivers == (1,)
    assert space.size == 2

    job = FrequencyDomainJob(name="encoded", simulation=sim, f_list=[5.0])
    assert job.trace_outputs.components == [
        "surface:focus_a",
        "surface:focus_b",
    ]

    sim.acquisition.receiver_groups[0].device.reduction = "none"
    unreduced = DataSpace.from_simulation(sim, frequencies=[5.0])
    assert unreduced.segments[0].receivers == (1, 2)
    assert unreduced.size == 4


def test_data_space_requires_known_external_source_count(tmp_path):
    sim = _elastic_simulation(tmp_path)
    sim.acquisition.sources = SourceGeometry.hdf5(
        "sources.h5",
        dataset="source_points",
        kind="vector",
    )

    with pytest.raises(ValueError, match="known source field count"):
        DataSpace.from_simulation(sim, frequencies=[5.0])

    sim.acquisition.sources = SourceGeometry.hdf5(
        "sources.h5",
        dataset="source_points",
        kind="vector",
        count=2,
    )

    space = DataSpace.from_simulation(sim, frequencies=[5.0])

    assert space.segments[0].sources == (1, 2)


def test_imaging_job_syntax_serializes_trace_store_roots(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)

    job = sim.imaging_job(
        name="rtm",
        observed=observed,
        frequencies=[5.0],
        parameters=["vp", "vs", "rho"],
        fields=["pressure"],
        condition="up_down",
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        weights=[1.0],
        misfit_type="L2",
    )
    payload = job.to_fs()

    assert isinstance(job, ImagingJob)
    assert "Image" in payload
    assert "Imaging" not in payload
    assert payload["Image"]["weights"] == [1.0]
    assert payload["Image"]["misfit"]["norm"] == "L2"
    assert payload["Image"]["images"] == [
        {"name": "FWI_Vp", "IC": "FWI", "property": "Vp"},
        {"name": "FWI_Vs", "IC": "FWI", "property": "Vs"},
        {"name": "FWI_Rho", "IC": "FWI", "property": "Rho"},
        {"name": "pressure", "IC": "pressure"},
        {"name": "up_down", "IC": "up_down"},
    ]


def test_imaging_job_serializes_instantaneous_travel_time_comparison(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed"
    observed.mkdir()

    job = sim.imaging_job(
        name="instantaneous_travel_time",
        observed=observed,
        frequencies=[5.0],
        parameters=["vp"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        comparison=MisfitComparison.phase_derivative(
            source_derivative="total",
            relative_amplitude_floor=0.02,
        ),
        observed_derivatives=ObservedTraceDerivatives.packed(
            observed,
            receiver_group="surface",
        ),
    )

    payload = job.to_fs()
    comparison = payload["Image"]["misfit"]["comparison"]

    assert comparison == {
        "kind": "phase_derivative",
        "derivative_axis": "frequency",
        "source_derivative": "total",
        "relative_amplitude_floor": 0.02,
    }
    assert payload["Image"]["misfit"]["receiver_groups"][0]["observed_derivatives"] == {
        "df": {
            "_type": "HDF5TraceStore",
            "file": observed / "traces.h5",
            "dataset": "surface_df",
            "source_basis": "source_encoding",
        }
    }
    loaded = BaseJob.load(job.save())
    assert loaded.misfit.comparison == MisfitComparison.phase_derivative(
        source_derivative="total",
        relative_amplitude_floor=0.02,
    )
    assert loaded.misfit.receiver_groups[0].observed_derivatives == (
        ObservedTraceDerivatives.packed(observed, receiver_group="surface")
    )


@pytest.mark.parametrize(
    "comparison",
    [
        {"kind": "phase_derivative", "derivative_axis": "laplace"},
        {"kind": "phase_derivative", "relative_amplitude_floor": 0.0},
        {"kind": "unsupported"},
    ],
)
def test_imaging_job_rejects_invalid_comparison_configuration(tmp_path, comparison):
    sim = _elastic_simulation(tmp_path)

    with pytest.raises(ValueError):
        ImagingJob(
            name="invalid_comparison",
            simulation=sim,
            f_list=[5.0],
            grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
            comparison=comparison,
        )


def test_phase_derivative_imaging_requires_observed_df_reference(tmp_path):
    sim = _elastic_simulation(tmp_path)

    with pytest.raises(ValueError, match="observed_derivatives"):
        ImagingJob(
            name="missing_observed_df",
            simulation=sim,
            f_list=[5.0],
            grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
            comparison=MisfitComparison.phase_derivative(),
        )


def test_imaging_job_outputs_use_canonical_top_level_contract(tmp_path):
    sim = _elastic_simulation(tmp_path)

    job = sim.imaging_job(
        name="rtm",
        frequencies=[5.0],
        parameters=["vp"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        outputs=VtkOutput.domain(name="rtm_qc", fields=["velocity"]),
    )
    job_file = job.save()
    saved = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert saved["Outputs"]["ParaView"][0]["name"] == "rtm_qc"
    assert saved["Outputs"]["ParaView"][0]["fields"] == ["velocity"]
    assert saved["Outputs"]["ParaView"][0]["target"] == {"kind": "volume"}
    assert "outputs" not in saved["Image"]
    assert loaded.to_fs()["Outputs"]["ParaView"][0]["name"] == "rtm_qc"


def test_imaging_job_allows_missing_observed_for_sensitivity_kernels(tmp_path):
    sim = _elastic_simulation(tmp_path)

    job = sim.imaging_job(
        name="kernel",
        frequencies=[5.0],
        parameters=["vp"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )
    payload = job.to_fs()

    assert payload["Image"]["data_path"] is None
    assert payload["Image"]["misfit"]["receiver_groups"][0]["observed"] is None
    assert job.misfit.receiver_groups[0].observed is None


def test_lsrtm_normal_job_serializes_one_fused_born_workflow(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    job = LSRTMNormalJob(
        name="normal",
        simulation=sim,
        data_path=None,
        f_list=[5.0, 7.0],
        grid=grid,
        images={"dVp": "FWI:Vp"},
        save_path=tmp_path / "images",
    )

    payload = job.to_fs()

    assert payload["_type"] == "LSRTMNormalJob"
    assert payload["workflow"] == "born"
    assert payload["Image"]["gauss_newton"] is True
    assert payload["Image"]["born_traces_only"] is False
    assert payload["Image"]["misfit"]["receiver_groups"][0]["observed"] is None
    assert job.n_tasks == 2
    assert job.trace_outputs.groups == ["surface_inc"]
    assert job.trace_outputs.components == ["surface_inc:vz"]

    loaded = BaseJob.load(job.save())

    assert isinstance(loaded, LSRTMNormalJob)
    assert loaded.workflow == "born"
    assert loaded.to_fs()["Image"]["gauss_newton"] is True


def test_lsrtm_gradient_job_serializes_one_call_residual_workflow(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    observed = tmp_path / "observed"
    observed.mkdir()
    job = LSRTMGradientJob(
        name="gradient",
        simulation=sim,
        data_path=observed,
        f_list=[5.0, 7.0],
        grid=grid,
        images={"dVp": "FWI:Vp"},
        save_path=tmp_path / "images",
    )

    payload = job.to_fs()

    assert payload["_type"] == "LSRTMGradientJob"
    assert payload["workflow"] == "lsrtm_gradient"
    assert payload["Image"]["gauss_newton"] is True
    assert payload["Image"]["born_traces_only"] is False
    assert payload["Image"]["zero_direction"] is True
    assert "background_forward_policy" not in payload["Image"]
    assert "background_cache_key" not in payload["Image"]
    assert payload["Image"]["misfit"]["receiver_groups"][0]["observed"] is not None
    assert job.n_tasks == 2
    assert job.trace_outputs.groups == ["surface", "surface_inc"]
    assert job.trace_outputs.components == ["surface:vz", "surface_inc:vz"]

    loaded = BaseJob.load(job.save())

    assert isinstance(loaded, LSRTMGradientJob)
    assert loaded.workflow == "lsrtm_gradient"


def test_lsrtm_gradient_job_requires_observed_data(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])

    with pytest.raises(ValueError, match="requires observed data"):
        LSRTMGradientJob(
            name="gradient",
            simulation=sim,
            data_path=None,
            f_list=[5.0],
            grid=grid,
        )

    with pytest.raises(ValueError, match="cannot be trace-only"):
        LSRTMGradientJob(
            name="gradient",
            simulation=sim,
            data_path=tmp_path / "observed",
            f_list=[5.0],
            grid=grid,
            born_traces_only=True,
        )


def test_lsrtm_normal_job_rejects_conflicting_workflow_flags(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])

    with pytest.raises(ValueError, match="cannot be trace-only"):
        LSRTMNormalJob(
            name="trace_only",
            simulation=sim,
            f_list=[5.0],
            grid=grid,
            born_traces_only=True,
        )
    with pytest.raises(ValueError, match="requires gauss_newton=True"):
        LSRTMNormalJob(
            name="legacy",
            simulation=sim,
            f_list=[5.0],
            grid=grid,
            gauss_newton=False,
        )


def test_legacy_simulation_imaging_method_remains_supported(tmp_path):
    sim = _elastic_simulation(tmp_path)

    job = sim.imaging(
        name="kernel",
        frequencies=[5.0],
        parameters=["vp"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )

    assert isinstance(job, ImagingJob)
    assert job.data_path is None


def test_legacy_imaging_images_syntax_still_serializes(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)

    job = ImagingJob(
        name="rtm",
        simulation=sim,
        data_path=observed,
        f_list=[5.0],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        images={"dVp": "FWI:Vp", "p": "pressure"},
        weights=[1.0],
    )
    payload = job.to_fs()

    group = payload["Image"]["misfit"]["receiver_groups"][0]
    assert group["observed"] == observed
    assert group["simulated"] == (
        tmp_path / "jobs" / "smooth" / "rtm" / "results" / "traces"
    )
    assert payload["Image"]["images"] == [
        {"name": "dVp", "IC": "FWI", "property": "Vp"},
        {"name": "p", "IC": "pressure"},
    ]


def test_imaging_job_serializes_representation_independent_tgv_smoothing(tmp_path):
    sim = _elastic_simulation(tmp_path)

    job = ImagingJob(
        name="rtm",
        simulation=sim,
        f_list=[5.0],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        regularization=VariationalSmoothing(
            kind="tgv",
            wavelength_fraction=0.5,
            tgv_ratio=2.0,
            epsilon=0.01,
            iterations=7,
            input_role="primal",
        ),
    )

    assert job.to_fs()["Image"]["Smoothing"] == {
        "type": "tgv",
        "lambda": 0.5,
        "derivative_order": 1,
        "epsilon": 0.01,
        "iterations": 7,
        "tgv_ratio": 2.0,
        "illumination_normalization": "none",
    }


def test_imaging_job_native_smoothing_mapping_defaults_to_no_illumination(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])

    job = ImagingJob(
        name="rtm",
        simulation=sim,
        f_list=[5.0],
        grid=grid,
        regularization={"type": "tikhonov", "lambda": 0.5},
    )
    legacy = ImagingJob(
        name="legacy_rtm",
        simulation=sim,
        f_list=[5.0],
        grid=grid,
        regularization={
            "type": "tikhonov",
            "lambda": 0.5,
            "normalize_illumination": True,
        },
    )

    assert job.to_fs()["Image"]["Smoothing"]["illumination_normalization"] == "none"
    assert "illumination_normalization" not in legacy.to_fs()["Image"]["Smoothing"]


def test_imaging_job_rejects_dual_or_second_derivative_smoothing(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])

    with pytest.raises(ValueError, match="input_role='primal'"):
        ImagingJob(
            name="dual",
            simulation=sim,
            f_list=[5.0],
            grid=grid,
            regularization=VariationalSmoothing(),
        )
    with pytest.raises(ValueError, match="first-order"):
        ImagingJob(
            name="second",
            simulation=sim,
            f_list=[5.0],
            grid=grid,
            regularization=VariationalSmoothing(
                derivative_order=2,
                input_role="primal",
            ),
        )


def test_imaging_job_resolves_reference_wavelength_before_export(tmp_path):
    sim = _elastic_simulation(tmp_path)

    job = ImagingJob(
        name="reference",
        simulation=sim,
        f_list=[5.0],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        regularization=VariationalSmoothing(
            kind="tgv",
            wavelength_fraction=0.5,
            reference_wavelength=2.0,
            tgv_ratio=2.0,
            input_role="primal",
        ),
    )

    smoothing = job.to_fs()["Image"]["Smoothing"]
    alpha1 = 1.0 / (2.0 * np.pi)
    assert smoothing["alpha1"] == alpha1
    assert smoothing["alpha2"] == 2.0 * alpha1**2
    assert "reference_wavelength" not in smoothing
    assert "input_role" not in smoothing


def test_imaging_job_save_and_load_round_trips_project_relative_simulation(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)

    job = sim.imaging_job(
        name="rtm",
        observed=observed,
        frequencies=[5.0],
        parameters=["vp", "rho"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        k_list=[-0.01, 0.0, 0.01],
        k_weights=[0.25, 0.5, 0.25],
        k_units="1/m",
    )

    job_file = job.save()
    saved_text = job_file.read_text()
    saved = json.loads(saved_text)
    loaded = BaseJob.load(job_file)

    assert "simulations/smooth/smooth.json" in saved_text
    assert str(tmp_path / "jobs" / "smooth") not in saved_text
    assert saved["k_list"] == [-0.01, 0.0, 0.01]
    assert saved["k_weights"] == [0.25, 0.5, 0.25]
    assert saved["k_units"] == "1/m"
    assert "k_list" not in saved["Image"]
    assert "k_weights" not in saved["Image"]
    assert "k_units" not in saved["Image"]
    assert isinstance(loaded, ImagingJob)
    assert loaded.name == "rtm"
    assert loaded.simulation.name == sim.name
    assert loaded.k_list == [-0.01, 0.0, 0.01]
    assert loaded.k_weights == [0.25, 0.5, 0.25]
    assert loaded.k_units == "1/m"
    assert loaded.images == {"FWI_Vp": "FWI:Vp", "FWI_Rho": "FWI:Rho"}
    assert loaded.grid.shape == (2, 3)
    loaded_group = loaded.misfit.receiver_groups[0]
    assert loaded_group.observed == observed
    assert loaded_group.simulated == (
        tmp_path / "jobs" / "smooth" / "rtm" / "results" / "traces"
    )
    assert loaded.regularization == {
        "type": "TV",
        "lambda": 1.0,
        "epsilon": 1.0,
        "iterations": 5,
        "illumination_normalization": "none",
    }


def test_imaging_job_saves_trace_weights_in_job_owned_hdf5(tmp_path):
    sim = _elastic_simulation(tmp_path)
    job = sim.imaging_job(
        name="weighted_rtm",
        frequencies=[5.0],
        parameters=["vp"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )
    weights = np.asarray([[[0.25, 2.0]]])
    job.misfit.receiver_groups[0].add_trace_weights(weights)

    job_file = job.save()
    saved = json.loads(job_file.read_text())
    reference = saved["Image"]["misfit"]["receiver_groups"][0]["preprocess"][0][
        "params"
    ]["weights"]

    assert reference["_type"] == "HDF5Dense"
    assert reference["file"] == "jobs/smooth/weighted_rtm/inputs.h5"
    with h5py.File(tmp_path / reference["file"], "r") as h5:
        stored = h5[reference["dataset"]]
        np.testing.assert_allclose(stored[:], weights)
        assert "source" not in stored.attrs
        assert "receiver" not in stored.attrs
    loaded = BaseJob.load(job_file)
    assert (
        loaded.to_fs()["Image"]["misfit"]["receiver_groups"][0]["preprocess"][0][
            "params"
        ]["weights"]
        == reference
    )
    fingerprint = job.fingerprint()
    job.misfit.receiver_groups[0].preprocess.clear()
    job.misfit.receiver_groups[0].add_trace_weights(np.asarray([[[0.5, 2.0]]]))
    assert fingerprint.startswith("blake3:")
    assert job.fingerprint() != fingerprint


def test_imaging_job_without_observed_saves_and_loads_null_data_path(tmp_path):
    sim = _elastic_simulation(tmp_path)

    job = sim.imaging_job(
        name="kernel",
        frequencies=[5.0],
        parameters=["vp"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )

    job_file = job.save()
    saved = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert saved["Image"]["data_path"] is None
    assert saved["Image"]["misfit"]["receiver_groups"][0]["observed"] is None
    assert isinstance(loaded, ImagingJob)
    assert loaded.data_path is None
    assert loaded.misfit.receiver_groups[0].observed is None


def test_imaging_job_rejects_weight_frequency_length_mismatch(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)

    with pytest.raises(ValueError, match="one value per frequency"):
        sim.imaging_job(
            name="rtm",
            observed=observed,
            frequencies=[5.0, 10.0],
            parameters=["vp"],
            grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
            weights=[1.0],
        )


def test_image_database_reads_string_and_byte_labels(tmp_path):
    image_path = tmp_path / "image"
    image_path.mkdir()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    values = np.arange(6.0)

    with h5py.File(image_path / "image.h5", "w") as h5:
        group = h5.create_group("image/raw")
        group.create_dataset("properties", data=np.array(["vp"], dtype=string_dtype))
        dataset = group.create_dataset("vp", data=values)
        dataset.attrs["x0"] = np.array([0.0, 0.0])
        dataset.attrs["x1"] = np.array([2.0, 1.0])
        dataset.attrs["n_grid"] = np.array([3, 2])
        dataset.attrs["dims"] = np.array(["x", "z"], dtype=string_dtype)
        dataset.attrs["axis_units"] = np.array(["m", "m"], dtype=string_dtype)
        dataset.attrs["units"] = np.array(["m/s"], dtype=string_dtype)
        group = h5.create_group("image/smoothed")
        group.create_dataset("properties", data=np.array(["vp"], dtype=string_dtype))
        dataset = group.create_dataset("vp", data=2.0 * values)
        dataset.attrs["x0"] = np.array([0.0, 0.0])
        dataset.attrs["x1"] = np.array([2.0, 1.0])
        dataset.attrs["n_grid"] = np.array([3, 2])
        dataset.attrs["dims"] = np.array(["x", "z"], dtype=string_dtype)
        dataset.attrs["axis_units"] = np.array(["km", "km"], dtype=string_dtype)
        dataset.attrs["units"] = np.array(["km/s"], dtype=string_dtype)

    db = ImageDatabase(path=image_path, parts=1, shape=(2, 3))
    images = db.raw_images
    smoothed = db.smoothed_images

    assert images["vp"].dims == ("z", "x")
    np.testing.assert_array_equal(images["vp"].values, values.reshape(2, 3))
    assert images["vp"].attrs["units"] == "m/s"
    assert images["vp"].coords["z"].attrs["units"] == "m"
    assert images["vp"].coords["x"].attrs["units"] == "m"
    assert smoothed["vp"].dims == ("z", "x")
    np.testing.assert_array_equal(smoothed["vp"].values, (2.0 * values).reshape(2, 3))
    assert smoothed["vp"].attrs["units"] == "km/s"
    assert smoothed["vp"].coords["z"].attrs["units"] == "km"
    assert smoothed["vp"].coords["x"].attrs["units"] == "km"


def test_image_database_requires_aggregate_image(tmp_path):
    image_path = tmp_path / "image"
    image_path.mkdir()
    (image_path / "image_1.h5").touch()

    db = ImageDatabase(path=image_path, parts=1, shape=(2, 3))

    with pytest.raises(FileNotFoundError, match="imaging --smooth postprocess"):
        db.require_aggregate()


def test_imaging_job_current_requires_aggregate_image(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)
    job = sim.imaging_job(
        name="rtm",
        observed=observed,
        frequencies=[5.0],
        parameters=["vp"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )
    job_file = job.save()
    trace_file = job.expected_trace_files()[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.touch()
    job.write_run_state(status="completed")

    assert not job.is_run_current()

    job.image_file(1).touch()
    assert job.needs_image_smoothing()

    with pytest.raises(FileNotFoundError, match="imaging --smooth postprocess"):
        job.load_images()

    job.image_file().touch()
    assert job.is_run_current()
    loaded = BaseJob.load(job_file)
    images = loaded.load_images()
    assert images.path == job.save_path
    assert images.parts == loaded.n_tasks
    assert images.shape == loaded.grid.shape


def test_fwi_jacobian_dot_test_and_taylor_test_use_hermitian_products(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)
    grid = CartesianGrid(n=[2, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])

    nmodel = 3 * 4
    ndata = 2
    matrix = (np.arange(ndata * nmodel).reshape(ndata, nmodel) + 1j) / 10.0
    base = np.array([1.0 + 0.5j, -2.0 + 0.25j])

    def matvec(problem, vector):
        return matrix @ vector

    def rmatvec(problem, vector):
        return matrix.conj().T @ vector

    def nonlinear_forward(problem, vector):
        return base + matrix @ vector + 0.25 * matrix @ (vector * vector)

    problem = sim.fwi(
        observed=observed,
        frequencies=[5.0],
        parameters=["vp", "vs", "rho"],
        grid=grid,
        matvec=matvec,
        rmatvec=rmatvec,
        nonlinear_forward=nonlinear_forward,
    )
    direction = np.linspace(0.1, 1.2, problem.model_space.size).astype(np.complex128)

    dot = problem.dot_test(model_perturbation=direction, seed=3)
    taylor = problem.taylor_test(direction)

    assert dot["passed"]
    assert dot["relative_error"] < 1.0e-12
    assert taylor["passed"]
    assert taylor["rates"][-1] > 1.8


@pytest.mark.parametrize("job_type", [ImagingJob, LSRTMGradientJob])
@pytest.mark.parametrize("descriptor", [False, True])
def test_imaging_derivative_inputs_are_hashed_and_staged(
    tmp_path, job_type, descriptor
):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed"
    observed.mkdir()
    files = [observed / "surface.h5", observed / "other.h5"]
    for path in files:
        with h5py.File(path, "w") as h5:
            h5["df"] = [1.0]

    def reference(path):
        return ObservedTraceDerivatives(
            df=HDF5TraceStore(file=path, dataset="df") if descriptor else path
        )

    job = job_type(
        name="derivative_inputs",
        simulation=sim,
        f_list=[5.0],
        data_path=observed,
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
        comparison=MisfitComparison.phase_derivative(),
        observed_derivatives=reference(files[0]),
    )
    job.misfit.receiver_groups.append(
        MisfitGroup(
            name="other", observed=observed, observed_derivatives=reference(files[1])
        )
    )
    loaded = BaseJob.load(job.save())
    assert loaded._input_fingerprint_payload() == job._input_fingerprint_payload()
    if isinstance(job, LSRTMGradientJob):
        assert job._input_fingerprint_payload()["direction"] == {"kind": "zero"}

    remote_root = tmp_path / "remote"
    staged = loaded.remote_input_files(remote_root)
    for path in files:
        assert (path, remote_root / path.relative_to(tmp_path)) in staged
        before = loaded.fingerprint(), loaded.task_fingerprint(1)
        with h5py.File(path, "r+") as h5:
            h5["df"][0] += 1.0
        after = loaded.fingerprint(), loaded.task_fingerprint(1)
        assert all(old != new for old, new in zip(before, after))
    assert loaded._input_fingerprint_payload() == job._input_fingerprint_payload()


@pytest.mark.parametrize("job_type", [LSRTMGradientJob, LSRTMNormalJob])
def test_loaded_lsrtm_job_stages_and_fingerprints_direction_file(tmp_path, job_type):
    sim = _elastic_simulation(tmp_path)
    direction = tmp_path / "direction.h5"
    direction.write_bytes(b"direction")
    observed = tmp_path / "observed.h5"
    observed.write_bytes(b"observed")
    job = job_type(
        "gradient",
        sim,
        f_list=[5.0],
        data_path=observed,
        direction=direction,
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )
    loaded = BaseJob.load(job.save())
    assert (
        direction,
        type(direction)("/remote/project/direction.h5"),
    ) in loaded.remote_input_files("/remote/project")

    before = (
        loaded.fingerprint(),
        loaded.task_fingerprint(1),
        loaded.task_policy_fingerprint(1, "compatible"),
    )
    direction.write_bytes(b"updated direction")
    after = (
        loaded.fingerprint(),
        loaded.task_fingerprint(1),
        loaded.task_policy_fingerprint(1, "compatible"),
    )
    assert all(old != new for old, new in zip(before, after))
    assert loaded._input_fingerprint_payload() == job._input_fingerprint_payload()
