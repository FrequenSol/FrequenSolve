"""Generated Cloud benchmark workload.

Source tutorial: 03_velocity_model_building/02_coordinate_systems.ipynb
Source SHA-256: 45213a67713db784e3124aec182cf8a6d5e8c9c30ce0b437d471433de0597b1d
"""

# %% source cell 4
import numpy as np
import xarray as xr

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project", path="./scratch/tutorials/coordinate_systems", log_level="INFO"
)
sim = project.new_simulation(
    name="surface_relative",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
x = np.linspace(0.0, 1.0, 201)
topography = xr.DataArray(
    0.02 * np.sin(2.0 * np.pi * x), dims=["x"], coords={"x": x}, attrs={"units": "km"}
)
topography.coords["x"].attrs["units"] = "km"
depth = np.linspace(0.0, 0.5, 101)
vp_depth = xr.DataArray(
    1.5 + 1.0 * depth / depth.max(),
    dims=["below"],
    coords={"below": depth},
    attrs={"units": "km/s"},
)
vp_depth.coords["below"].attrs["units"] = "km"
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=topography)
model.add_layer(
    name="near_surface",
    properties={
        "Vp": {"value": vp_depth, "coordinate_system": "top_relative"},
        "Rho": 1.8 * u.g / u.cm**3,
    },
)
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model
top_system = sim.add_surface_coordinate_system(
    name="top_relative",
    surface="top",
    axes=[fs.Axis("below", direction="z", positive="down")],
)
surface = sim.model_surface("top", name="top_points", normal="down")
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=surface.below([[0.5, 0.02]], units="km"))
node = fs.ReceiverNode(name="hydrophone")
node.add_component(name="p", field="pressure")
acq.add_receiver_group(
    name="surface",
    device=node,
    coords=surface.on(np.linspace(0.1, 0.9, 21), units="km"),
)
sim += acq

# %% source cell 8
acq_payload = sim.acquisition.to_fs(sim.export_context())
{
    "source_coordinates": acq_payload["source_geometry"]["sources"][0]["coordinates"],
    "receiver_coordinate_type": acq_payload["receiver_groups"][0]["coordinates"].get(
        "_type"
    ),
    "coordinate_systems": [system.to_fs() for system in sim.coordinate_systems],
}

# %% source cell 10
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_surface_relative", simulation=sim, f_min=0.0, f_max=30.0, T_max=0.9
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 12
wavelet = fs.RickerWavelet(f=12.0)
group = traces.groups[0]
component = traces.components(group)[0]
source = traces.sources(group)[0]
gather = traces.td(group, component, source, wavelet, upscale=4, T_max=0.9)
fs.plot_gather(
    gather,
    A=2.0 * np.nanstd(np.real(gather.values)),
    cmap="gray",
    figsize=(9, 4),
    title="Surface-relative acquisition response",
)
