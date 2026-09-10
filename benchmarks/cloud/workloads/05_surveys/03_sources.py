"""Generated Cloud benchmark workload.

Source tutorial: 05_surveys/03_sources.ipynb
Source SHA-256: 769daa6b1a86f4d17c39d98d4a27f559baf6b2e218a800a0d97eebd048ef21e3
"""

# %% source cell 4
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    pretty_name="source_mechanisms",
    path="./scratch/tutorials/sources",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="source_mechanisms",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
)
model.add_surface(name="interface", depth=0.25 * u.km)
model.add_layer(
    name="basement", properties={"Vp": 2.4 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3}
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

# %% source cell 8
source_geometry = fs.SourceGeometry.points(
    kind="scalar",
    coords=[[0.25, 0.05], [0.5, 0.05], [0.75, 0.05], [0.45, 0.08], [0.55, 0.08]],
    names=["shot_left", "shot_center", "shot_right", "pair_pos", "pair_neg"],
)
source_encoding = fs.SourceEncoding.named(
    {
        "shot_left": {"shot_left": 1.0},
        "shot_center": {"shot_center": 1.0},
        "shot_right": {"shot_right": 1.0},
        "difference": {"pair_pos": 1.0, "pair_neg": -1.0},
    }
)
acq = fs.Acquisition(source_geometry=source_geometry, source_encoding=source_encoding)
hydrophone = fs.ReceiverNode(name="hydrophone")
hydrophone.add_component(name="p", field="pressure")
receiver_coords = [[x, 0.04] for x in np.linspace(0.1, 0.9, 81)]
acq.add_receiver_group(name="line", device=hydrophone, coords=receiver_coords)
sim += acq
{
    "physical_source_points": acq.source_point_count(),
    "rhs_fields": acq.source_field_count(),
    "dense_trace_count": acq.source_field_count() * len(receiver_coords),
}

# %% source cell 10
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_sources", simulation=sim, f_min=0.0, f_max=30.0, T_max=0.9
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 12
wavelet = fs.RickerWavelet(f=12.0)
group = "line"
component = "p"
source_ids = traces.sources(group)
source_gathers = {
    source_id: traces.td(group, component, source_id, wavelet, upscale=4, T_max=0.9)
    for source_id in source_ids
}
A = 2.0 * max(
    (float(np.nanstd(np.real(gather.values))) for gather in source_gathers.values())
)

# %% source cell 14
center = len(receiver_coords) // 2
