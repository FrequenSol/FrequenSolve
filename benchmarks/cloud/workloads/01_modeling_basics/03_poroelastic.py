"""Generated Cloud benchmark workload.

Source tutorial: 01_modeling_basics/03_poroelastic.ipynb
Source SHA-256: d0c9488e77ab7ddc0a011134fe4c7de4e0687dbf69fa1f4edb7a64003790df68
"""

# %% source cell 2
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project", path="./scratch/poroelastic", log_level="INFO", log_to_console=True
)
sim = project.new_simulation(
    name="poroelastic_basic",
    physics="poroelastic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="upper_layer",
    physics="poroelastic:iso",
    properties={
        "Vp": 2.0 * u.km / u.s,
        "Vs": 1.0 * u.km / u.s,
        "Rho": 2.2 * u.g / u.cm**3,
        "K_solid": 20.0 * u.GPa,
        "K_fluid": 2.2 * u.GPa,
        "Rho_solid": 2.65 * u.g / u.cm**3,
        "Rho_fluid": 1.0 * u.g / u.cm**3,
        "Porosity": 0.3,
        "Tortuosity": 2.0,
        "Kappa": 5e-09 * u.m**2,
        "Viscosity": 0.001 * u.Pa * u.s,
    },
)
model.add_surface(name="interface", depth=0.25 * u.km)
model.add_layer(
    name="lower_layer",
    physics="poroelastic:iso",
    properties={
        "Vp": 2.8 * u.km / u.s,
        "Qp": 100.0,
        "Vs": 1.5 * u.km / u.s,
        "Qs": 50.0,
        "Rho": 2.4 * u.g / u.cm**3,
        "K_solid": 40.0 * u.GPa,
        "K_fluid": 2.2 * u.GPa,
        "Rho_solid": 2.7 * u.g / u.cm**3,
        "Rho_fluid": 1.0 * u.g / u.cm**3,
        "Porosity": 0.18,
        "Tortuosity": 2.2,
        "Kappa": 1e-09 * u.m**2,
        "Viscosity": 0.001 * u.Pa * u.s,
    },
)
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model

# %% source cell 9
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0)
sim.mesh.set_source_grading(d1=0.05, factor=4.0)
sim += fs.BoundaryCondition(conditions=["free", "sealed"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)

# %% source cell 11
acq = fs.Acquisition()
acq.add_sources(kind="vector", coords=[[0.5, 0.0]], direction=[0.0, 1.0])
receiver = fs.ReceiverNode(name="geophone")
receiver.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
receiver.add_component(name="pore_pressure", field="pressure")
receiver_coords = [[x, 0.0] for x in np.linspace(0.0, 1.0, 201)]
acq.add_receiver_group(name="surface", device=receiver, coords=receiver_coords)
sim += acq

# %% source cell 13
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_poroelastic", simulation=sim, f_min=0.0, f_max=45.0, T_max=1.0
)
result = site.submit(job).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 15
wavelet = fs.RickerWavelet(f=15.0)
group = "surface"
source = traces.sources(group)[0]
components = [("v_z", "Solid vertical velocity"), ("pore_pressure", "Pore pressure")]
gathers = {}
center = len(receiver_coords) // 2 + 10

# %% source cell 17
site_fd = fs.Site()
poro_pv_output = fs.VtkOutput.domain(
    name="pv_poroelastic",
    path="paraview",
    fields=["pressure", "velocity"],
    properties=["vp", "vs", "rho", "porosity", "kappa"],
    sources=[1],
    show_pml=True,
    upscale=1,
    order=2,
)
fd_job = fs.FrequencyDomainJob(
    name="freq_poroelastic_qc", simulation=sim, f_list=[45.0], outputs=[poro_pv_output]
)
fd_result = site_fd.submit(fd_job).wait()
poro_vtu_files = fd_result.output_files(
    base="pv_poroelastic", suffix=".vtu", existing=True
)

# %% source cell 19
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
screenshot = image_dir / "poroelastic_pressure.png"
