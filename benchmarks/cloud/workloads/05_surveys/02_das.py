"""Generated Cloud benchmark workload.

Source tutorial: 05_surveys/02_das.ipynb
Source SHA-256: c416c8e537f5ce331438c7905273004e08f3b7803da635979981e383dc9ba391
"""

# %% source cell 5
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 7
project = fs.Project(
    name="project",
    pretty_name="das_comparison",
    path="./scratch/tutorials/das",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="das_comparison",
    physics="elastic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="elastic",
    properties={
        "Vp": 2.4 * u.km / u.s,
        "Vs": 1.1 * u.km / u.s,
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
straight_das = fs.ReceiverFiber(
    gauge_length=10 * u.m, channel_spacing=10 * u.m, sample_spacing=2 * u.m
)
straight_das.add_component(name="eps_fiber", field="strain")
helical_das = fs.ReceiverFiber(
    gauge_length=10 * u.m,
    channel_spacing=10 * u.m,
    sample_spacing=1.25 * u.m,
    radius=2 * u.m,
    angle=60 * u.deg,
)
helical_das.add_component(name="eps_fiber", field="strain")
point_strain = fs.ReceiverNode()
point_strain.add_component(name="eps_xx", field="strain_xx")
acq = fs.Acquisition()
acq.add_sources(kind="vector", coords=[[0.5, 0.05]], direction=[0.0, 1.0])
coords = [[x, 0.05] for x in np.linspace(0.1, 0.9, 61)]
acq.add_receiver_group(name="straight_das", device=straight_das, coords=coords)
acq.add_receiver_group(name="helical_das", device=helical_das, coords=coords)
acq.add_receiver_group(name="point_strain", device=point_strain, coords=coords)
sim += acq
{
    group.name: [component.name for component in group.device.components]
    for group in acq.receiver_groups
}

# %% source cell 11
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_das", simulation=sim, f_min=0.0, f_max=30.0, T_max=0.9
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 13
wavelet = fs.RickerWavelet(f=12.0)
source = traces.sources("straight_das")[0]
straight = traces.td("straight_das", "eps_fiber", source, wavelet, upscale=4, T_max=0.9)
helical = traces.td("helical_das", "eps_fiber", source, wavelet, upscale=4, T_max=0.9)
point = traces.td("point_strain", "eps_xx", source, wavelet, upscale=4, T_max=0.9)
line_x_km = np.asarray(coords, dtype=float)[:, 0]
x0_km = line_x_km[0]
straight_x_km = (
    x0_km
    + np.arange(straight.sizes["receiver"])
    * straight_das.channel_spacing.to("km").magnitude
)
helical_spacing_km = helical_das.channel_spacing.to("km").magnitude * np.cos(
    helical_das.angle.to("rad").magnitude
)
helical_x_km = x0_km + np.arange(helical.sizes["receiver"]) * helical_spacing_km


def with_receiver_x(gather, x_km):
    gather = gather.assign_coords(receiver=("receiver", x_km))
    gather.coords["receiver"].attrs.update({"long_name": "X", "units": "km"})
    return gather


straight_x = with_receiver_x(straight, straight_x_km)
helical_x = with_receiver_x(helical, helical_x_km)
point_x = with_receiver_x(point, line_x_km)
shared_x_km = line_x_km[
    line_x_km
    <= min(
        float(straight_x.receiver.max()),
        float(helical_x.receiver.max()),
        float(point_x.receiver.max()),
    )
]
straight_common = straight_x.interp(receiver=shared_x_km)
helical_common = helical_x.interp(receiver=shared_x_km)
point_common = point_x.interp(receiver=shared_x_km)
common_gathers = [straight_common, helical_common, point_common]
A = 2.0 * max((float(np.nanstd(np.real(arr.values))) for arr in common_gathers))
fs.diff_gathers(
    straight_common,
    helical_common,
    A=A,
    units="km",
    cmap="gray",
    figsize=(12, 4),
    titles=("Straight DAS", "Helical DAS", "Helical - straight"),
)
fs.diff_gathers(
    point_common,
    straight_common,
    A=A,
    units="km",
    cmap="gray",
    figsize=(12, 4),
    titles=("Point strain", "Straight DAS", "Straight - point"),
)

# %% source cell 15
center_x_km = 0.5
