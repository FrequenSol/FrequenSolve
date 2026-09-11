"""Generated Cloud benchmark workload.

Source tutorial: 03_velocity_model_building/03_layered_models.ipynb
Source SHA-256: 152494c4c03fd652b0e3a7c6abba9539884a9185c2b02171a0e21f85d7929917
"""

# %% source cell 4
from pathlib import Path

import numpy as np
import xarray as xr

import frequensolve as fs

u = fs.ureg

# %% source cell 6
x = np.linspace(0.0, 1.0, 101)
interface = xr.DataArray(
    0.25 + 0.04 * np.sin(4.0 * np.pi * x),
    dims=["x"],
    coords={"x": x},
    attrs={"units": "km"},
)
interface.coords["x"].attrs["units"] = "km"
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="overburden", properties={"Vp": 1.7 * u.km / u.s, "Rho": 1.8 * u.g / u.cm**3}
)
model.add_surface(name="marker", depth=0.15 * u.km)
model.add_surface(name="reservoir_top", depth=interface)
model.add_layer(
    name="reservoir", properties={"Vp": 2.4 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3}
)
model.add_surface(name="bottom", depth=0.5 * u.km)
model.add_borehole(
    name="well_1",
    x=0.55 * u.km,
    top="top",
    bottom="bottom",
    parts=[
        {
            "name": "fluid",
            "mesh_block_id": 20,
            "r": 0.03 * u.km,
            "physics": "acoustic",
            "properties": {"Vp": 1.48 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3},
        }
    ],
)
uniform = model.sample_uniform([151, 151])

# %% source cell 10
project = fs.Project(
    name="project",
    pretty_name="advanced_layered_model",
    path="./scratch/tutorials/layered_models",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="advanced_layered_model",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
sim += model
mesh = model.hex_mesh_generator([8, 4])
mesh.refine_around_borehole("well_1", padding=0.04 * u.km, max_size=0.01 * u.km)
sim += mesh
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=[[0.3, 0.04]])
hydrophone = fs.ReceiverNode(name="hydrophone")
hydrophone.add_component(name="p", field="pressure")
acq.add_receiver_group(
    name="surface",
    device=hydrophone,
    coords=[[x, 0.03] for x in np.linspace(0.1, 0.9, 41)],
)
sim += acq
sim += fs.Discretization()
site = fs.Site()
job = fs.FrequencyDomainJob(
    name="freq_layered_model",
    simulation=sim,
    f_list=[20.0],
    outputs=[
        fs.VtkOutput.domain(
            name="layered_model",
            fields=["pressure"],
            properties=["vp", "rho", "Subdomain"],
            show_pml=True,
            upscale=1,
            order=2,
        )
    ],
)
result = site.submit(job).wait()

# %% source cell 12
vtu_files = result.output_files(base="layered_model", suffix=".vtu", existing=True)
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
screenshot = image_dir / "layered_model_vp.png"
