"""Generated Cloud benchmark workload.

Source tutorial: 03_velocity_model_building/01_variable_properties_units.ipynb
Source SHA-256: fd6793103f1963b05d5a5f71bf1d220ccaacc0d2441e2b9c0a3175d40bb469b6
"""

# %% source cell 4
import numpy as np
import xarray as xr

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project", path="./scratch/tutorials/variable_properties", log_level="INFO"
)
sim = project.new_simulation(
    name="variable_properties",
    physics="elastic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
x = np.linspace(0.0, 1.0, 81)
z = np.linspace(0.0, 0.5, 51)
zz = xr.DataArray(z, dims=["z"], coords={"z": z})
zz.coords["z"].attrs["units"] = "km"
vp = 1.8 + 1.2 * (zz / zz.max())
vp.attrs["units"] = "km/s"
vs = 0.55 * vp
vs.attrs["units"] = "km/s"
rho = 0.31 * (1000.0 * vp) ** 0.25
rho.attrs["units"] = "g/cm^3"
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(name="gradient", properties={"Vp": vp, "Vs": vs, "Rho": rho})
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model
sampled = model.sample_uniform([101, 101])

# %% source cell 10
center_x = sampled.x.size // 2

# %% source cell 12
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)
acq = fs.Acquisition()
acq.add_sources(kind="vector", coords=[[0.5, 0.05]], direction=[0.0, 1.0])
receiver = fs.ReceiverNode(name="geophone")
receiver.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
receiver_coords = [[x, 0.04] for x in np.linspace(0.1, 0.9, 61)]
acq.add_receiver_group(name="surface", device=receiver, coords=receiver_coords)
sim += acq
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_variable_properties", simulation=sim, f_min=0.0, f_max=30.0, T_max=0.9
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 14
wavelet = fs.RickerWavelet(f=12.0)
group = traces.groups[0]
component = traces.components(group)[0]
source = traces.sources(group)[0]
gather = traces.td(group, component, source, wavelet, upscale=4, T_max=0.9)
