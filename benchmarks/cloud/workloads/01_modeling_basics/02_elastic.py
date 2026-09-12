"""Generated Cloud benchmark workload.

Source tutorial: 01_modeling_basics/02_elastic.ipynb
Source SHA-256: f1a04347a94621c215a9a792637f8fb987c2a43f117c37a4fc208de2be6082f1
"""

# %% source cell 2
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 5
project = fs.Project(
    name="project", path="./scratch/elastic", log_level="INFO", log_to_console=True
)
sim = project.new_simulation(
    name="elastic_basic",
    physics="elastic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="upper_layer",
    properties={
        "Vp": 2.0 * u.km / u.s,
        "Vs": 1.0 * u.km / u.s,
        "Rho": 2.2 * u.g / u.cm**3,
    },
)
model.add_surface(name="interface", depth=0.25 * u.km)
model.add_layer(
    name="lower_layer",
    properties={
        "Vp": 2.8 * u.km / u.s,
        "Vs": 1.5 * u.km / u.s,
        "Rho": 2.4 * u.g / u.cm**3,
    },
)
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model

# %% source cell 8
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0)
sim.mesh.set_source_grading(d1=0.05, factor=4.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"],
    boundaries=["x_min", "x_max", "z_max"],
    pml_wavelengths=0.75,
    pml_reflection=0.001,
)

# %% source cell 10
acq = fs.Acquisition()
acq.add_sources(kind="vector", coords=fs.Q_([[0.5, 0.0]], "km"), direction=[0.0, 1.0])
receiver = fs.ReceiverNode(name="geophone")
receiver.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
receiver_coords = [fs.Q_([x, 0.0], "km") for x in np.linspace(0.0, 1.0, 201)]
acq.add_receiver_group(name="surface", device=receiver, coords=receiver_coords)
sim += acq

# %% source cell 12
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_no_q", simulation=sim, f_min=0.0, f_max=45.0, T_max=1.0
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 14
wavelet = fs.RickerWavelet(f=15.0)
group = traces.groups[0]
component = traces.components(group)[0]
source = traces.sources(group)[0]
no_q = traces.td(group, component, source, wavelet, upscale=4)

# %% source cell 16
attenuated = sim.copy("elastic_attenuated")
for layer in attenuated.model.layers:
    layer.properties["Qp"] = 10.0
    layer.properties["Qs"] = 5.0
project += attenuated
q_job = fs.TimeDomainJob(
    name="time_q", simulation=attenuated, f_min=0.0, f_max=45.0, T_max=1.0
)
q_result = site.submit(q_job).wait()
q_traces = q_result.traces(upscale=4)
q_group = q_traces.groups[0]
q_component = q_traces.components(q_group)[0]
q_source = q_traces.sources(q_group)[0]
with_q = q_traces.td(q_group, q_component, q_source, wavelet, upscale=4)

# %% source cell 20
site_fd = fs.Site()
elastic_pv_output = fs.VtkOutput.domain(
    name="pv_elastic",
    path="paraview",
    fields=["velocity_z", "stress_zz"],
    properties=["vp", "vs"],
    show_pml=True,
    upscale=1,
    order=2,
)
fd_job = fs.FrequencyDomainJob(
    name="freq_elastic_qc", simulation=sim, f_list=[50.0], outputs=[elastic_pv_output]
)
fd_result = site_fd.submit(fd_job).wait()
elastic_vtu_files = fd_result.output_files(
    base="pv_elastic", suffix=".vtu", existing=True
)

# %% source cell 22
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
screenshot = image_dir / "elastic_paraview_vs.png"
