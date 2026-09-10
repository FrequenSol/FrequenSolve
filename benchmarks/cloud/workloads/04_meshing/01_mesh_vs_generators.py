"""Generated Cloud benchmark workload.

Source tutorial: 04_meshing/01_mesh_vs_generators.ipynb
Source SHA-256: f0f303dfc6b89c277cca81523720bb99ba8a601cd4403a4780eccb11ada3cf83
"""

# %% source cell 5
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 7
project = fs.Project(
    name="project",
    pretty_name="mesh_vs_generators",
    path="./scratch/tutorials/mesh_vs_generators",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="mesh_vs_generators",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
)
model.add_surface(name="bottom", depth=0.4 * u.km)
sim += model

# %% source cell 9
sim += fs.LayeredMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 0.4], n=[4, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=3, f_low=5.0, f_high=20.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.5
)
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=[[0.5, 0.05]])
node = fs.ReceiverNode(name="hydrophone")
node.add_component(name="p", field="pressure")
acq.add_receiver_group(
    name="surface", device=node, coords=[[x, 0.025] for x in np.linspace(0.1, 0.9, 21)]
)
sim += acq
sim += fs.Discretization()
site = fs.Site()
job = fs.FrequencyDomainJob(
    name="freq_mesh",
    simulation=sim,
    f_list=[20.0],
    outputs=[
        fs.VtkOutput.domain(
            name="mesh",
            fields=["pressure"],
            properties=["vp", "rho", "Subdomain"],
            show_pml=True,
            upscale=0,
            order=1,
        )
    ],
)
result = site.submit(job).wait()

# %% source cell 11
mesh_payload = sim.mesh.to_fs(sim.export_context())
{
    "generator_type": mesh_payload.get("generator", {}).get("_type"),
    "initial_n": mesh_payload.get("generator", {}).get("n"),
    "adaptivity": mesh_payload.get("adapt"),
}

# %% source cell 13
vtu_files = result.output_files(base="mesh", suffix=".vtu", existing=True)
all_outputs = result.output_files(existing=True)
{"vtu_files": [str(path) for path in vtu_files], "all_output_count": len(all_outputs)}

# %% source cell 15
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
screenshot = image_dir / "mesh_vs_generators_vp.png"
plotter = fs.plot_vtu(
    vtu_files[0],
    field="vp",
    show_edges=True,
    scalar_bar=True,
    show=False,
    screenshot=screenshot,
    window_size=(1100, 500),
)
