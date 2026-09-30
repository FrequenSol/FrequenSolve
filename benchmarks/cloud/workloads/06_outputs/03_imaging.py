"""Generated Cloud benchmark workload.

Source tutorial: 06_outputs/03_imaging.ipynb
Source SHA-256: 5d0babc480edb8d6b2b860c3fcec5b8f0a808868c426d1b968fe5690930b69b1
"""

# %% source cell 5
from pathlib import Path
from pprint import pprint

import numpy as np
import xarray as xr

import frequensolve as fs
from frequensolve import imaging as im

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
WATER_DEPTH = 0.1
MODEL_DEPTH = 0.6


def sediment_truth(below):
    below = np.asarray(below, dtype=float)
    vp = np.full_like(below, 2.4)
    vp[below < 0.35] = 2.2
    vp[below < 0.25] = 1.8
    vp[below < 0.15] = 2.0
    vp[below < 0.05] = 1.7
    return vp


def sediment_start(below):
    below = np.asarray(below, dtype=float)
    return 1.7 + (2.4 - 1.7) * below / (MODEL_DEPTH - WATER_DEPTH)


def build_simulation(project, *, name, sediment_vp):
    sim = project.new_simulation(
        name=name,
        physics="acoustic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )
    below = np.linspace(0.0, MODEL_DEPTH - WATER_DEPTH, 201)
    vp_column = xr.DataArray(
        sediment_vp(below), dims=["z"], coords={"z": WATER_DEPTH + below}
    )
    model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.2])
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
    )
    model.add_surface(name="seabed", depth=WATER_DEPTH * u.km)
    model.add_layer(
        name="sediment", properties={"Vp": vp_column, "Rho": 2.0 * u.g / u.cm**3}
    )
    model.add_surface(name="bottom", depth=MODEL_DEPTH * u.km)
    sim += model
    sim += model.hex_mesh_generator([12, 6])
    sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=3.0, f_high=10.0)
    sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += fs.BoundaryCondition(
        conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
    )
    acq = fs.Acquisition()
    acq.add_sources(
        kind="scalar", coords=fs.Q_([[0.2, 0.02], [0.6, 0.02], [1.0, 0.02]], "km")
    )
    hydrophone = fs.ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    receiver_coords = [fs.Q_([x, 0.05], "km") for x in np.linspace(0.1, 1.1, 51)]
    acq.add_receiver_group(name="surface", device=hydrophone, coords=receiver_coords)
    sim += acq
    sim += fs.SolverConfig(tolerance=0.0001)
    return sim


true_sim = build_simulation(project, name="imaging_true", sediment_vp=sediment_truth)
start_sim = build_simulation(project, name="imaging_start", sediment_vp=sediment_start)

# %% source cell 10
below = np.linspace(0.0, MODEL_DEPTH - WATER_DEPTH, 201)

# %% source cell 12
frequencies = [4.0, 6.0, 8.0]

# %% source cell 14
controls = im.ControlSpace(
    vp=im.DepthProfile("vp", "sediment", spacing=0.05 * u.km, transform="log")
)
bound = controls.bind(start_sim)
{
    "keys": controls.keys,
    "blocks": bound.qualified_names,
    "payload": bound.controls_payload(),
}

# %% source cell 16
observed_job = fs.FrequencyDomainJob(
    name="observed_true", simulation=true_sim, f_list=frequencies
)
observed = im.ObservedData(observed_job)
misfit = im.Misfit.huber(
    delta=1.345, preprocess=[im.Preprocess.offset_taper(d0=0.1 * u.km, d1=0.25 * u.km)]
)
print("observed frequencies:", observed.frequencies)
pprint(misfit.to_fs(observed.resolve(start_sim)))

# %% source cell 18
site = fs.Site()
observed_result = site.submit(observed_job).wait()
observed_traces = observed_result.traces()
observed_traces.summary

# %% source cell 20
problem = im.ImagingProblem(
    start_sim,
    controls=controls,
    observed=observed,
    misfit=misfit,
    frequencies=frequencies,
    site=site,
    workdir=Path(project.path) / "fwi",
    name="fwi_tutorial",
)
print("blocks:", problem.space.blocks, "size:", problem.space.size)
print("capabilities:", problem.capabilities())

# %% source cell 22
lin = problem.linearize()
gradient = lin.gradient
print("misfit value:", lin.value)
print("per term:", lin.report)

# %% source cell 23
image = im.rtm(problem)
np.allclose(np.asarray(image), np.asarray(gradient))

# %% source cell 25
J = lin.jacobian
H = lin.normal
dv = problem.space.random(seed=1)
d_lin = J @ dv
g_gn = J.H @ d_lin
h_dv = H @ dv
print("J @ dv:", d_lin.shape, "as dataset:")
print(
    "max |J.H (J dv) - H dv|:",
    float(np.max(np.abs(np.asarray(g_gn) - np.asarray(h_dv)))),
)

# %% source cell 27
grid = fs.CartesianGrid(n=[121, 61], x0=[0.0, 0.0], x1=[1.2, 0.6])
kernels = im.sensitivity_kernel(
    problem, grid, properties=["vp"], condition="fwi", frequencies=[6.0]
)
raw = kernels.raw

# %% source cell 29
stages = im.Stage.bands([[4.0], [4.0, 6.0]], iterations=[3, 3], active=["vp"])
fwi = im.FWI(
    problem,
    stages=stages,
    optimizer=im.LBFGS(memory=5, step_limit=0.05),
    regularization=im.Tikhonov(alpha=0.01, order=1),
    checkpoint="checkpoint.h5",
    history="history.json",
)
result = fwi.run(resume=True)
for stage in result.stages:
    print(
        f"{stage.name}: frequencies={stage.frequencies} iterations={stage.iterations} linearizations={stage.linearizations} data loss {stage.initial_loss.data:.4g} -> {stage.final_loss.data:.4g} ({stage.message})"
    )
print("success:", result.success, "checkpoint:", result.checkpoint)

# %% source cell 31
final = result.vector()
nodes = final.to_xarray("vp")
node_depths = np.asarray(nodes.coords[nodes.dims[0]])
recovered = sediment_start(node_depths) * np.exp(np.asarray(nodes.values))
below = np.linspace(0.0, MODEL_DEPTH - WATER_DEPTH, 201)
