"""Generated Cloud benchmark workload.

Source tutorial: 06_outputs/03_imaging.ipynb
Source SHA-256: b1dcf384350c83c4d01a689829fec57b55845d2ecc36b73e660b8515d5d496eb
"""

# %% source cell 5
from pathlib import Path
from pprint import pprint

import h5py
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    pretty_name="Imaging Tutorial",
    path="./scratch/tutorials/imaging",
    log_level="INFO",
    log_to_console=True,
)
project.path


# %% source cell 8
def build_elastic_simulation(project, *, name, interface_depth_km):
    sim = project.new_simulation(
        name=name,
        physics="elastic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )
    model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.2])
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="upper_layer",
        properties={
            "Vp": 2.0 * u.km / u.s,
            "Vs": 1.0 * u.km / u.s,
            "Rho": 2.2 * u.g / u.cm**3,
        },
    )
    model.add_surface(name="interface", depth=interface_depth_km * u.km)
    model.add_layer(
        name="lower_layer",
        properties={
            "Vp": 2.8 * u.km / u.s,
            "Vs": 1.5 * u.km / u.s,
            "Rho": 2.4 * u.g / u.cm**3,
        },
    )
    model.add_surface(name="bottom", depth=0.5 * u.km)
    sim += model
    sim += model.hex_mesh_generator([12, 5])
    sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=25.0)
    sim.mesh.set_source_grading(d1=0.05, factor=2.0)
    sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += fs.BoundaryCondition(
        conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
    )
    acq = fs.Acquisition()
    acq.add_sources(
        kind="vector",
        coords=fs.Q_([[0.25, 0.02], [0.6, 0.02], [0.95, 0.02]], "km"),
        direction=[0.0, 1.0],
    )
    geophone = fs.ReceiverNode(name="surface_geophone")
    geophone.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
    receiver_coords = [fs.Q_([x, 0.0], "km") for x in np.linspace(0.05, 1.15, 121)]
    acq.add_receiver_group(name="surface", device=geophone, coords=receiver_coords)
    sim += acq
    sim += fs.SolverConfig(ptol=1e-10)
    return sim


smooth_sim = build_elastic_simulation(
    project, name="imaging_smooth", interface_depth_km=0.25
)
true_sim = build_elastic_simulation(
    project, name="imaging_true", interface_depth_km=0.3
)

# %% source cell 12
frequencies = [8.0, 12.0, 18.0]
frequency_weights = [1.0, 0.8, 0.45]
image_grid = fs.CartesianGrid(n=[161, 81], x0=[0.0, 0.0], x1=[1.2, 0.5])
image_grid.as_xarray()

# %% source cell 14
observed_root = Path(project.path) / "observed_frequency_data"
(observed_root / "surface").mkdir(parents=True, exist_ok=True)
imaging_job = smooth_sim.imaging_job(
    name="rtm_elastic",
    observed=observed_root,
    frequencies=frequencies,
    grid=image_grid,
    parameters=["vp", "vs", "rho"],
    fields=["velocity"],
    condition="up_down",
    weights=frequency_weights,
    misfit_norm="L2",
    keep_forward=False,
    keep_adjoint=False,
    keep_unstacked=False,
)
imaging_job_file = imaging_job.save()
loaded_imaging_job = fs.BaseJob.load(imaging_job_file)
(imaging_job_file, type(loaded_imaging_job).__name__, loaded_imaging_job.images)

# %% source cell 16
image_contract = imaging_job.to_fs()["Image"]
pprint(image_contract)

# %% source cell 18
focused_job = smooth_sim.imaging_job(
    name="rtm_focused",
    observed=observed_root,
    frequencies=[12.0],
    grid=image_grid,
    images={"dVp": "FWI:Vp", "dRho": "FWI:Rho", "vz_image": "velocity"},
    weights=[1.0],
)
focused_job.to_fs()["Image"]["images"]


# %% source cell 20
def write_synthetic_image_database(path, *, grid, frequency):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    nx = int(grid.n[0])
    nz = int(grid.n[1])
    x = np.linspace(grid.x0[0], grid.x1[0], nx)[None, :]
    z = np.linspace(grid.x0[1], grid.x1[1], nz)[:, None]
    reflector = np.exp(-((z - 0.3) ** 2) / (2.0 * 0.015**2))
    aperture = np.cos(np.pi * (x - 0.6) / 1.2) ** 2
    dip = np.sin(2.0 * np.pi * (x / 1.2 + 0.8 * z))
    raw_vp = reflector * aperture * (1.0 + 0.25 * dip)
    raw_vs = 0.55 * reflector * aperture * (1.0 - 0.15 * dip)
    raw_rho = -0.35 * reflector * aperture
    smoothed = {
        "FWI_Vp": 0.72 * raw_vp,
        "FWI_Vs": 0.72 * raw_vs,
        "FWI_Rho": 0.72 * raw_rho,
    }
    raw = {"FWI_Vp": raw_vp, "FWI_Vs": raw_vs, "FWI_Rho": raw_rho}
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path / "image.h5", "w") as h5:
        for group_name, images in {"raw": raw, "smoothed": smoothed}.items():
            group = h5.create_group(f"image/{group_name}")
            group.create_dataset(
                "properties", data=np.array(list(images), dtype=string_dtype)
            )
            for name, values in images.items():
                dataset = group.create_dataset(name, data=values.reshape(-1))
                dataset.attrs["x0"] = np.array([grid.x0[0], grid.x0[1]])
                dataset.attrs["x1"] = np.array([grid.x1[0], grid.x1[1]])
                dataset.attrs["n_grid"] = np.array([nx, nz])
                dataset.attrs["dims"] = np.array(["x", "z"], dtype=string_dtype)
    with h5py.File(path / "image_1.h5", "w") as h5:
        h5.create_dataset("frequency", data=float(frequency))


preview_path = Path(project.path) / "synthetic_image_preview"
write_synthetic_image_database(preview_path, grid=image_grid, frequency=12.0)
image_db = fs.ImageDatabase(path=preview_path, parts=1, shape=image_grid.shape)
raw_images = image_db.raw_images
smoothed_images = image_db.smoothed_images

# %% source cell 23
fwi_problem = smooth_sim.fwi(
    observed=observed_root,
    frequencies=frequencies,
    parameters=["vp", "vs", "rho"],
    grid=image_grid,
)
summary = {
    "model_parameters": fwi_problem.model_space.parameters,
    "model_vector_size": fwi_problem.model_space.size,
    "data_vector_size": fwi_problem.data_space.size,
    "image_grid_dims": fwi_problem.model_space.dims,
    "image_grid_shape": fwi_problem.grid.shape,
}

# %% source cell 25
site = fs.Site()
observed_job = fs.FrequencyDomainJob(
    name="observed_true_data", simulation=true_sim, f_list=frequencies
)
observed_result = site.submit(observed_job).wait()
observed_traces = observed_result.traces()
observed_traces.summary

# %% source cell 27
rtm_job = smooth_sim.imaging_job(
    name="rtm_from_true_data",
    observed=observed_job,
    grid=image_grid,
    parameters=["vp", "vs", "rho"],
    fields=["velocity"],
    condition="up_down",
    weights=frequency_weights,
    misfit_norm="L2",
)
rtm_result = site.submit(rtm_job).wait()
solver_images = site.fetch_image(rtm_job)
solver_images.raw_images

# %% source cell 28
solver_raw = solver_images.raw_images
image_names = list(solver_raw.data_vars)
