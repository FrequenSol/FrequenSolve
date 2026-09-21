"""Generated Cloud benchmark workload.

Source tutorial: 06_outputs/02_paraview_vtk.ipynb
Source SHA-256: 018ae1445a1e56a81d0a58c27344af5bc7bf0c370fe467dec300eac7ada2610e
"""

# %% source cell 4
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    pretty_name="paraview_outputs",
    path="./scratch/tutorials/paraview_vtk",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="paraview_outputs",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
)
model.add_surface(name="interface", depth=0.24 * u.km)
model.add_layer(
    name="basement", properties={"Vp": 2.6 * u.km / u.s, "Rho": 2.3 * u.g / u.cm**3}
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
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=[[0.5, 0.05]])
hydrophone = fs.ReceiverNode(name="hydrophone")
hydrophone.add_component(name="p", field="pressure")
acq.add_receiver_group(
    name="surface",
    device=hydrophone,
    coords=[[x, 0.04] for x in np.linspace(0.1, 0.9, 41)],
)
sim += acq
sim += fs.Discretization()

# %% source cell 8
domain_output = fs.VtkOutput.domain(
    name="pv_volume",
    path="pv_volume",
    fields=["pressure"],
    properties=["vp", "rho", "Subdomain"],
    parts=["real", "imag", "abs"],
    sources=[1],
    show_pml=True,
    upscale=1,
    order=2,
)
surface_output = fs.VtkOutput.surface(
    name="pv_interface",
    path="pv_interface",
    surfaces=["interface"],
    fields=["pressure"],
    properties=["vp", "Subdomain"],
    parts=["abs"],
    show_pml=False,
    upscale=1,
    order=2,
)
grid = fs.CartesianGrid(n=[121, 61], x0=[0.0, 0.0], x1=[1.0, 0.5], units="km")
grid_output = fs.VtkOutput.grid(
    grid,
    name="pv_grid",
    path="pv_grid",
    fields=["pressure"],
    properties=["vp"],
    parts=["abs"],
    sources=[1],
)
[
    output.to_fs(sim.export_context())
    for output in [domain_output, surface_output, grid_output]
]

# %% source cell 10
site = fs.Site()
job = fs.FrequencyDomainJob(
    name="freq_paraview",
    simulation=sim,
    f_list=[20.0],
    outputs=[domain_output, surface_output, grid_output],
)
result = site.submit(job).wait()

# %% source cell 12
volume_files = result.output_files(base="pv_volume", suffix=".vtu", existing=True)
interface_files = result.output_files(base="pv_interface", suffix=".vtu", existing=True)
grid_files = result.output_files(base="pv_grid", suffix=".vtu", existing=True)
{
    "volume": [str(path) for path in volume_files],
    "interface": [str(path) for path in interface_files],
    "grid": [str(path) for path in grid_files],
}

# %% source cell 14
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
renders = [
    (volume_files[0], "vp", image_dir / "paraview_volume_vp.png"),
    (volume_files[0], "pressure", image_dir / "paraview_volume_pressure_abs.png"),
    (interface_files[0], "vp", image_dir / "paraview_interface_vp.png"),
]

# %% source cell 16
mesh = fs.read_vtu(volume_files[0])
{
    "bounds": mesh.bounds,
    "n_points": mesh.n_points,
    "n_cells": mesh.n_cells,
    "point_arrays": list(mesh.point_data.keys()),
    "cell_arrays": list(mesh.cell_data.keys()),
}
