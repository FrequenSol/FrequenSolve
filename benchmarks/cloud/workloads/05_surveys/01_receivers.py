"""Generated Cloud benchmark workload.

Source tutorial: 05_surveys/01_receivers.ipynb
Source SHA-256: 5b1f5a8d048594dc469a499dc0d9e5e06b4ba6afbb85b542162f1ba6460936f0
"""

# %% source cell 5
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 7
project = fs.Project(
    name="project",
    pretty_name="receiver_components",
    path="./scratch/tutorials/receivers",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="receiver_components",
    physics="elastic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="elastic_halfspace",
    properties={
        "Vp": 2.5 * u.km / u.s,
        "Vs": 1.2 * u.km / u.s,
        "Rho": 2.2 * u.g / u.cm**3,
    },
)
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim.mesh.set_source_grading(d0=0.02, d1=0.08, factor=2.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)

# %% source cell 9
acq = fs.Acquisition()
acq.add_sources(kind="vector", coords=[[0.5, 0.05]], direction=[0.0, 1.0])
node = fs.ReceiverNode(name="three_component_node")
node.add_component(name="v_x", field="velocity", direction=[1.0, 0.0])
node.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
node.add_component(name="eps_xx", field="strain", direction=[1.0, 0.0])
receiver_coords = [[x, 0.04] for x in np.linspace(0.1, 0.9, 81)]
acq.add_receiver_group(name="surface_dense", device=node, coords=receiver_coords)
sim += acq
{
    "source_fields": acq.source_field_count(),
    "receiver_groups": [group.name for group in acq.receiver_groups],
    "receiver_count": len(receiver_coords),
    "components": [component.name for component in node.components],
}

# %% source cell 11
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_receivers", simulation=sim, f_min=0.0, f_max=30.0, T_max=0.9
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 13
wavelet = fs.RickerWavelet(f=12.0)
group = "surface_dense"
source = traces.sources(group)[0]
components = traces.components(group)
component_gathers = {
    component: traces.td(group, component, source, wavelet, upscale=4, T_max=0.9)
    for component in components
}
A = max(
    (float(np.nanstd(np.real(gather.values))) for gather in component_gathers.values())
)
A = 2.0 * A if A > 0 else None

# %% source cell 15
center = len(receiver_coords) // 2
