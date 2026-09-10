"""Generated Cloud benchmark workload.

Source tutorial: 01_modeling_basics/06_laplace_time_domain.ipynb
Source SHA-256: 1bc0cbe7736bb4fea15ddeccb6002c7f53ff6c447d094529b419b5232ac4eb5e
"""

# %% source cell 3
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 5
project = fs.Project(
    name="project",
    pretty_name="laplace_time_domain",
    path="./scratch/tutorials/laplace_time_domain",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="laplace_time_domain",
    physics="coupled_aep",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.2])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="water", physics="acoustic", properties={"Vp": 1.48, "Rho": 1.0, "Qp": 200.0}
)
model.add_surface(name="seafloor", depth=0.06 * u.km)
model.add_layer(
    name="soft_sediment",
    physics="elastic",
    properties={"Vp": 1.55, "Vs": 0.18, "Rho": 1.65, "Qp": 80.0, "Qs": 40.0},
)
model.add_surface(name="stiffer_sediment", depth=0.28 * u.km)
model.add_layer(
    name="halfspace",
    physics="elastic",
    properties={"Vp": 2.25, "Vs": 0.75, "Rho": 2.05, "Qp": 100.0, "Qs": 60.0},
)
model.add_surface(name="bottom", depth=0.55 * u.km)
sim += model

# %% source cell 7
sim += model.hex_mesh_generator([12, 6])
sim.mesh.set_adapt(elems_per_wave=1.5, order=4, f_low=1.0, f_high=12.0)
sim.mesh.set_source_grading(d1=0.04, factor=2.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"],
    boundaries=["x_min", "x_max", "z_max"],
    pml_wavelengths=0.75,
    pml_reflection=0.001,
)
acq = fs.Acquisition()
acq.add_sources(kind="vector", coords=[[0.18, 0.08]], direction=[0.0, 1.0])
geophone = fs.ReceiverNode(name="seafloor_geophone")
geophone.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
receiver_coords = [[x, 0.07] for x in np.linspace(0.05, 1.15, 181)]
acq.add_receiver_group(name="seafloor", device=geophone, coords=receiver_coords)
sim += acq
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
project.save().relative_to(project.path)

# %% source cell 9
T_MAX = 1.2
F_MAX = 12.0
WAVELET_F = 4.0


def laplace_for_factor(factor, period=T_MAX):
    return -np.log(float(factor)) / (2.0 * np.pi * period)


{
    "df_hz": 1.0 / T_MAX,
    "laplace_for_10": laplace_for_factor(10.0),
    "laplace_for_100": laplace_for_factor(100.0),
}

# %% source cell 11
cases = [
    ("standard", None, "Standard time domain"),
    ("damping_10", 10.0, "Damping factor 10"),
    ("damping_100", 100.0, "Damping factor 100"),
]


def run_case(name, damping_factor):
    kwargs = {}
    if damping_factor is not None:
        kwargs["damping_factor"] = damping_factor
    site = fs.Site()
    job = fs.TimeDomainJob(
        name=name, simulation=sim, f_min=0.0, f_max=F_MAX, T_max=T_MAX, **kwargs
    )
    result = site.submit(job).wait()
    return result.traces(upscale=4)


trace_sets = {
    name: run_case(name, damping_factor) for name, damping_factor, _title in cases
}
trace_sets["standard"].summary

# %% source cell 13
GROUP = "seafloor"
COMPONENT = "v_z"


def read_td(traces):
    source = traces.sources(GROUP)[0]
    return traces.td(
        GROUP, COMPONENT, source, fs.RickerWavelet(f=WAVELET_F), upscale=4, T_max=T_MAX
    )


def read_ld(traces):
    source = traces.sources(GROUP)[0]
    return traces.ld(
        GROUP, COMPONENT, source, fs.RickerWavelet(f=WAVELET_F), upscale=4, T_max=T_MAX
    )


gathers = {name: read_td(traces) for name, traces in trace_sets.items()}
laplace_gathers = {
    name: read_ld(trace_sets[name]) for name in ("damping_10", "damping_100")
}
summary_keys = (
    "source_id",
    "receiver_group",
    "long_name",
    "domain",
    "laplace",
    "laplace_compensated",
    "damping_factor",
)
{
    name: {key: gather.attrs[key] for key in summary_keys}
    for name, gather in gathers.items()
}

# %% source cell 15
scale = max(
    (
        np.nanpercentile(np.abs(np.real(gather.values)), 90)
        for gather in gathers.values()
    )
)
scale = max(scale, 1e-12)

# %% source cell 17
ld_scale = max(
    (
        np.nanpercentile(np.abs(np.real(gather.values)), 90)
        for gather in laplace_gathers.values()
    )
)
ld_scale = max(ld_scale, 1e-12)

# %% source cell 19
receiver = gathers["standard"].coords["receiver"].values[-1]
