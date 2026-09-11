"""Generated Cloud benchmark workload.

Source tutorial: 04_meshing/03_gradings.ipynb
Source SHA-256: d1cf5a2bb9fd4721b4ee7a8993de6b7886e126083adeab5c90efad17a3536b9b
"""

# %% source cell 4
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    pretty_name="mesh_gradings",
    path="./scratch/tutorials/mesh_gradings",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="mesh_gradings",
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
    name="basement", properties={"Vp": 2.5 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3}
)
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model

# %% source cell 8
sim += model.hex_mesh_generator([4, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim.mesh.set_source_grading(d0=0.01, d1=0.08, factor=2.0, power=2.0)
sim.mesh.set_receiver_grading(d0=0.01, d1=0.05, factor=1.5)
sim.mesh.add_surface_grading("interface", d0=0.0, d1=0.04, factor=2.0, mode="abs_band")
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
    name="freq_gradings",
    simulation=sim,
    f_list=[25.0],
    outputs=[
        fs.VtkOutput.domain(
            name="gradings",
            fields=["pressure"],
            properties=["vp", "Subdomain"],
            show_pml=True,
            upscale=1,
            order=2,
        )
    ],
)
result = site.submit(job).wait()

# %% source cell 10
vtu_files = result.output_files(base="gradings", suffix=".vtu", existing=True)
[str(path) for path in vtu_files]

# %% source cell 12
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
rendered = []
