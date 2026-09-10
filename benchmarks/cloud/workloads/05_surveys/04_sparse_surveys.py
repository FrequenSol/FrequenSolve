"""Generated Cloud benchmark workload.

Source tutorial: 05_surveys/04_sparse_surveys.ipynb
Source SHA-256: c7fdd46c643a2432300216f6965cb76a810796bf1e723364aa424b44f14134af
"""

# %% source cell 4
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    pretty_name="sparse_surveys",
    path="./scratch/tutorials/sparse_surveys",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="sparse_surveys",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
)
model.add_surface(name="bottom", depth=0.45 * u.km)
sim += model
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)
sources = [[0.2, 0.05], [0.5, 0.05], [0.8, 0.05]]
receivers = [[x, 0.04] for x in np.linspace(0.05, 0.95, 91)]
node = fs.ReceiverNode(name="hydrophone")
node.add_component(name="p", field="pressure")

# %% source cell 8
offset_survey = fs.SparseSurvey.offset_domain(
    "middle_offsets", min=0.1 * u.km, max=0.35 * u.km, metric="horizontal"
)
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=sources)
acq.add_sparse_receiver_group(
    "middle_offsets", node, coords=receivers, survey=offset_survey
)
sim += acq
{
    "dense_trace_count": len(sources) * len(receivers),
    "survey_kind": offset_survey.kind,
    "offset_domain": offset_survey.offset_domain,
}

# %% source cell 10
explicit = fs.SparseSurvey.from_product(
    "selected_pairs", sources=[1, 2], receivers=[10, 20, 30, 40], components="p"
)
explicit.to_fs(component_map={"p": 1})

# %% source cell 12
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_sparse_offsets", simulation=sim, f_min=0.0, f_max=30.0, T_max=0.9
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 14
tables = traces.survey_tables()
tables.keys()

# %% source cell 15
wavelet = fs.RickerWavelet(f=12.0)
group = "middle_offsets"
component = "p"
source = traces.sources(group)[0]
gather = traces.td(group, component, source, wavelet, upscale=4, T_max=0.9)
