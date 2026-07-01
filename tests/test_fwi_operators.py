import h5py
import numpy as np
import pytest
import xarray as xr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverComponent, ReceiverNode
from frequensolve.seismic.sources import SourceGeometry
from frequensolve.simulation.jobs import BaseJob
from frequensolve.simulation.jobs.fwi import DataSpace, ModelSpace
from frequensolve.simulation.jobs.imaging import ImageDatabase, ImagingJob
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


def test_natural_imaging_syntax_serializes_to_legacy_solver_contract(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)

    job = sim.imaging(
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

    assert payload["Image"]["images"] == [
        {"name": "dVp", "IC": "FWI", "property": "Vp"},
        {"name": "p", "IC": "pressure"},
    ]


def test_imaging_job_save_and_load_round_trips_project_relative_simulation(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)

    job = sim.imaging(
        name="rtm",
        observed=observed,
        frequencies=[5.0],
        parameters=["vp", "rho"],
        grid=CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0]),
    )

    job_file = job.save()
    saved_text = job_file.read_text()
    loaded = BaseJob.load(job_file)

    assert "simulations/smooth/smooth.json" in saved_text
    assert str(tmp_path / "jobs" / "smooth") not in saved_text
    assert isinstance(loaded, ImagingJob)
    assert loaded.name == "rtm"
    assert loaded.simulation.name == sim.name
    assert loaded.images == {"FWI_Vp": "FWI:Vp", "FWI_Rho": "FWI:Rho"}
    assert loaded.grid.shape == (2, 3)
    loaded_group = loaded.misfit.receiver_groups[0]
    assert loaded_group.observed == observed / "surface"
    assert loaded_group.simulated == (
        tmp_path / "jobs" / "smooth" / "rtm" / "results" / "traces" / "surface"
    )
    assert loaded.regularization == {
        "type": "TV",
        "lambda": 1.0,
        "epsilon": 1.0,
        "iterations": 5,
    }


def test_imaging_job_rejects_weight_frequency_length_mismatch(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)

    with pytest.raises(ValueError, match="one value per frequency"):
        sim.imaging(
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

    db = ImageDatabase(path=image_path, parts=1, shape=(2, 3))
    images = db.raw_images

    assert images["vp"].dims == ("z", "x")
    np.testing.assert_array_equal(images["vp"].values, values.reshape(2, 3))


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
