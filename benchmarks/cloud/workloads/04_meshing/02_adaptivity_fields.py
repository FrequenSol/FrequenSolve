"""Generated Cloud benchmark workload.

Source tutorial: 04_meshing/02_adaptivity_fields.ipynb
Source SHA-256: f1fb8a8f03f146e79ad78e87f798e6475bede0ad59dd93abead258a88e012ec5
"""

# %% source cell 4
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    pretty_name="adaptivity_fields",
    path="./scratch/adaptivity_fields",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="adaptivity_fields",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="cap",
    properties={
        "Vp": 1.5 * u.km / u.s,
        "Rho": 1.0 * u.g / u.cm**3,
        "epw_mult": 1.5,
        "hmax": 0.04 * u.km,
    },
)
model.add_surface(name="interface", depth=0.2 * u.km)
model.add_layer(
    name="basement",
    properties={
        "Vp": 2.6 * u.km / u.s,
        "Rho": 2.2 * u.g / u.cm**3,
        "vadapt": 1.4 * u.km / u.s,
        "hmin": 0.01 * u.km,
    },
)
model.add_surface(name="bottom", depth=0.5 * u.km)
sim += model

# %% source cell 8
sim += model.hex_mesh_generator([4, 4])
sim.mesh.set_adapt(
    elems_per_wave=2.0, order=4, f_low=5.0, f_high=35.0, hmin=0.005, hmax=0.08
)
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
    name="freq_adaptivity",
    simulation=sim,
    f_list=[25.0],
    outputs=[
        fs.VtkOutput.domain(
            name="adaptivity",
            fields=["pressure"],
            properties=["vp", "epw_mult", "hmin", "hmax", "Subdomain"],
            show_pml=True,
            upscale=1,
            order=2,
        )
    ],
)
result = site.submit(job).wait()

# %% source cell 10
vtu_files = result.output_files(base="adaptivity", suffix=".vtu", existing=True)
[str(path) for path in vtu_files]

# %% source cell 12
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
rendered = []
for field, filename in [("vp", "adaptivity_vp.png"), ("hmin", "adaptivity_hmin.png")]:
    screenshot = image_dir / filename
    fs.plot_vtu(
        vtu_files[0],
        field=field,
        show_edges=True,
        scalar_bar=True,
        show=False,
        screenshot=screenshot,
        window_size=(1100, 500),
    )
    rendered.append(screenshot)
