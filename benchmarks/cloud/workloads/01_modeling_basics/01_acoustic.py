"""Generated Cloud benchmark workload.

Source tutorial: 01_modeling_basics/01_acoustic.ipynb
Source SHA-256: de9e0bc6b8db64d5cfbfc809b9f1a6f6459e96d1c132db1f2e6175d838727792
"""

# %% source cell 3
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 5
project = fs.Project(
    name="project", path="./scratch/acoustic", log_level="INFO", log_to_console=True
)
sim = project.new_simulation(
    name="acoustic_basic",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)

# %% source cell 8
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.ft)
model.add_layer(
    name="upper_layer", properties={"Vp": 2000 * u.m / u.s, "Rho": 2.2 * u.g / u.cm**3}
)
model.add_surface(name="interface", depth=250 * u.m)
model.add_layer(
    name="lower_layer", properties={"Vp": 2800 * u.m / u.s, "Rho": 2.4 * u.g / u.cm**3}
)
model.add_surface(name="bottom", depth=500 * u.m)
sim += model

# %% source cell 11
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0)
sim.mesh.set_source_grading(d1=0.05, factor=2.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"],
    boundaries=["x_min", "x_max", "z_max"],
    pml_wavelengths=0.5,
    pml_reflection=0.001,
)

# %% source cell 13
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=fs.Q_([[500, 25]], "m"))
hydrophone = fs.ReceiverNode(name="hydrophone")
hydrophone.add_component(name="p", field="pressure")
receiver_coords = fs.Q_([[x, 50] for x in np.linspace(0, 1000, 201)], "m")
acq.add_receiver_group(name="surface", device=hydrophone, coords=receiver_coords)
sim += acq

# %% source cell 15
project_file = project.save()
project_file.relative_to(project.path)

# %% source cell 17
site = fs.Site()
pv_coarse = fs.VtkOutput.domain(
    name="pv_coarse", properties=["Vp"], show_pml=True, upscale=0, order=1
)
pv_fine = fs.VtkOutput.domain(
    name="pv_fine",
    properties=["Vp"],
    fields=["pressure", "velocity_z"],
    show_pml=True,
    upscale=1,
    order=2,
)
fd_job = fs.FrequencyDomainJob(
    name="freq_50hz", simulation=sim, f_list=[50.0], outputs=[pv_coarse, pv_fine]
)
fd_handle = site.submit(fd_job)
fd_result = fd_handle.wait()

# %% source cell 19
vtu_files = fd_result.output_files(suffix=".vtu", existing=True)
for file in vtu_files:
    mesh = fs.read_vtu(file)
    print(f"File: {file.relative_to(project.path)}\nFields:")
    for field in fs.vtu_fields(mesh):
        print(f"\t{field}")

# %% source cell 21
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
mesh_screenshot = image_dir / "acoustic_mesh.png"
pressure_screenshot = image_dir / "acoustic_pressure.png"
if vtu_files:
    fs.plot_vtu(
        vtu_files[0],
        field="Vp",
        show_edges=True,
        scalar_bar=True,
        show=False,
        cmap=fs.BuGrOr,
        zoom=1.5,
        screenshot=mesh_screenshot,
        window_size=(900, 500),
    )
    fs.plot_vtu(
        vtu_files[1],
        field="pressure",
        part="im",
        scalar_bar=True,
        show=False,
        vmin=-30.0,
        vmax=30.0,
        cmap="RdGy",
        zoom=1.5,
        screenshot=pressure_screenshot,
        window_size=(900, 500),
    )

# %% source cell 23
td_job = fs.TimeDomainJob(name="time", simulation=sim, f_min=0.0, f_max=45.0, T_max=2.0)
print(f"\nFrequency tasks for job: {td_job.name}")
for i, f in enumerate(td_job.f_list[:10]):
    print(f"   Task {i}: {f.real} + {f.imag}i Hz")
print("   ...\n")
handle = site.submit(td_job)
result = handle.wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 25
wavelet = fs.RickerWavelet(f=15.0)
group = traces.groups[0]
component = traces.components(group)[0]
source = traces.sources(group)[0]
gather = traces.td(group, component, source, wavelet, upscale=4, T_max=0.75)
