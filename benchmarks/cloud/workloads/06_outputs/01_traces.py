"""Generated Cloud benchmark workload.

Source tutorial: 06_outputs/01_traces.ipynb
Source SHA-256: 5ae164f19a946bf85cf11ab669a48ff881555e598101a9335b8daebef3297829
"""

# %% source cell 5
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 7
project = fs.Project(
    name="project",
    pretty_name="trace_outputs",
    path="./scratch/tutorials/traces",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="trace_outputs",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
)
model.add_surface(name="interface", depth=0.22 * u.km)
model.add_layer(
    name="basement", properties={"Vp": 2.4 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3}
)
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=[[0.35, 0.05], [0.65, 0.05]])
hydrophone = fs.ReceiverNode(name="hydrophone")
hydrophone.add_component(name="p", field="pressure")
receiver_coords = [[x, 0.04] for x in np.linspace(0.1, 0.9, 81)]
acq.add_receiver_group(name="surface", device=hydrophone, coords=receiver_coords)
sim += acq
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)

# %% source cell 9
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_traces", simulation=sim, f_min=0.0, f_max=30.0, T_max=0.9
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 11
{
    "trace_files": traces.files,
    "groups": traces.groups,
    "components": {group: traces.components(group) for group in traces.groups},
    "sources": {group: traces.sources(group) for group in traces.groups},
    "frequencies_hz": traces.frequencies().tolist(),
}

# %% source cell 13
group = "surface"
component = "p"
source_ids = traces.sources(group)
fd_by_source = {
    source_id: traces.fd(group, component, source=source_id) for source_id in source_ids
}

# %% source cell 15
wavelet = fs.RickerWavelet(f=12.0)
td_by_source = {
    source_id: traces.td(group, component, source_id, wavelet, upscale=4, T_max=0.9)
    for source_id in source_ids
}
A = 2.0 * max(
    (float(np.nanstd(np.real(gather.values))) for gather in td_by_source.values())
)

# %% source cell 17
first_trace_file = traces.paths[0]
reopened = fs.TraceDataset.open(first_trace_file, upscale=4)
cache_file = traces.consolidate()
{
    "opened_file": str(first_trace_file),
    "reopened_groups": reopened.groups,
    "cache_file": str(cache_file),
}
