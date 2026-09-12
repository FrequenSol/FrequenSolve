"""Generated Cloud benchmark workload.

Source tutorial: 01_modeling_basics/05_acoustic_25d_3d_axisymmetric.ipynb
Source SHA-256: fa74337c5a47ba17d8d7abd7be5edcc227fefd289d3e9fc7c890f708a8bea863
"""

# %% source cell 3
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg


# %% source cell 5
def add_acoustic_layers(model):
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
    )
    model.add_surface(name="interface", depth=0.25 * u.km)
    model.add_layer(
        name="basement", properties={"Vp": 2.4 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3}
    )
    model.add_surface(name="bottom", depth=0.5 * u.km)
    return model


def add_2d_boundaries(sim, *, axisymmetric=False):
    if axisymmetric:
        sim += fs.BoundaryCondition(conditions=["symmetric_r"], boundaries=["x_min"])
        pml_boundaries = ["x_max", "z_max"]
    else:
        pml_boundaries = ["x_min", "x_max", "z_max"]
    sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += fs.BoundaryCondition(
        conditions=["pml"], boundaries=pml_boundaries, pml_wavelengths=0.75
    )


def add_2d_pressure_line(
    sim, *, source_x=0.5, source_z=0.05, receiver_z=0.04, n_receivers=61
):
    acq = fs.Acquisition()
    acq.add_sources(kind="scalar", coords=[[source_x, source_z]])
    hydrophone = fs.ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    coords = [[x, receiver_z] for x in np.linspace(0.1, 0.9, n_receivers)]
    acq.add_receiver_group(name="line", device=hydrophone, coords=coords)
    sim += acq
    return coords


# %% source cell 7
project_25d = fs.Project(
    name="project",
    pretty_name="acoustic_25d",
    path="./scratch/tutorials/acoustic_25d",
    log_level="INFO",
    log_to_console=True,
)
sim_25d = project_25d.new_simulation(
    name="acoustic_25d",
    physics="acoustic",
    dimension="2.5D",
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model_25d = add_acoustic_layers(
    fs.LayeredModel(name="model", dimension="2.5D", x_limits=[0.0, 1.0])
)
sim_25d += model_25d
sim_25d += model_25d.hex_mesh_generator([8, 4])
sim_25d.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
add_2d_boundaries(sim_25d)
receiver_coords_25d = add_2d_pressure_line(sim_25d)
sim_25d += fs.Discretization()
sim_25d += fs.SolverConfig(tolerance=0.0001, grids=3)

# %% source cell 9
site_25d = fs.Site()
job_25d = fs.TimeDomainJob(
    name="time_25d",
    simulation=sim_25d,
    f_min=0.0,
    f_max=30.0,
    T_max=0.9,
    k_list=[-0.01, 0.0, 0.01],
    k_units="1/km",
)
result_25d = site_25d.submit(job_25d).wait()
traces_25d = result_25d.traces(upscale=4)
traces_25d.summary

# %% source cell 10
wavelet = fs.RickerWavelet(f=12.0)
group = traces_25d.groups[0]
component = traces_25d.components(group)[0]
source = traces_25d.sources(group)[0]
gather_25d = traces_25d.td(group, component, source, wavelet, upscale=4, T_max=0.9)

# %% source cell 12
slice_project = fs.Project(
    name="project", path="./scratch/tutorials/acoustic_fixed_y_slice", log_level="INFO"
)
sim_slice = slice_project.new_simulation(
    name="acoustic_fixed_y_slice",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
sim_slice.global_coordinate_system = fs.CoordinateSystem.cartesian(
    name="global", ndim=2, fixed_axis="y", fixed_value=0.3 * u.km
)
model_slice = add_acoustic_layers(
    fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
)
sim_slice += model_slice
sim_slice += model_slice.hex_mesh_generator([4, 4])
sim_slice += fs.Discretization()
sim_slice.to_fs()["global_coordinate_system"]

# %% source cell 14
project_3d = fs.Project(
    name="project",
    pretty_name="acoustic_3d",
    path="./scratch/acoustic_3d",
    log_level="INFO",
    log_to_console=True,
)
sim_3d = project_3d.new_simulation(
    name="acoustic_3d",
    physics="acoustic",
    dimension=3,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model_3d = add_acoustic_layers(
    fs.LayeredModel(name="model", dimension=3, x_limits=[0.0, 1.0], y_limits=[0.0, 0.6])
)
sim_3d += model_3d
sim_3d += model_3d.hex_mesh_generator([6, 4, 4])
sim_3d.mesh.set_adapt(elems_per_wave=2.0, order=3, f_low=5.0, f_high=20.0)
sim_3d += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim_3d += fs.BoundaryCondition(
    conditions=["pml"],
    boundaries=["x_min", "x_max", "y_min", "y_max", "z_max"],
    pml_wavelengths=0.75,
)
acq_3d = fs.Acquisition()
acq_3d.add_sources(kind="scalar", coords=[[0.5, 0.3, 0.05]])
hydrophone_3d = fs.ReceiverNode(name="hydrophone")
hydrophone_3d.add_component(name="p", field="pressure")
line_3d = [[x, 0.3, 0.04] for x in np.linspace(0.1, 0.9, 31)]
acq_3d.add_receiver_group(name="centerline", device=hydrophone_3d, coords=line_3d)
sim_3d += acq_3d
sim_3d += fs.Discretization()
site_3d = fs.Site()
job_3d = fs.FrequencyDomainJob(
    name="freq_3d",
    simulation=sim_3d,
    f_list=[15.0],
    outputs=[
        fs.VtkOutput.domain(
            name="pv_3d",
            fields=["pressure"],
            properties=["vp", "rho", "Subdomain"],
            show_pml=True,
            upscale=0,
            order=1,
        )
    ],
)
result_3d = site_3d.submit(job_3d).wait()

# %% source cell 16
vtu_3d = result_3d.output_files(base="pv_3d", suffix=".vtu", existing=True)
image_dir = Path("./assets")
image_dir.mkdir(exist_ok=True)
screenshot = image_dir / "acoustic_3d_vp.png"

# %% source cell 18
project_axisym = fs.Project(
    name="project",
    pretty_name="acoustic_axisym",
    path="./scratch/tutorials/acoustic_axisym",
    log_level="INFO",
    log_to_console=True,
)
sim_axisym = project_axisym.new_simulation(
    name="acoustic_axisym",
    physics="acoustic",
    dimension=2,
    axisymmetric=True,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
sim_axisym.global_coordinate_system = fs.CoordinateSystem.cylindrical(
    name="global", ndim=2, fixed_axis="theta", fixed_value=0.0
)
model_axisym = add_acoustic_layers(
    fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
)
sim_axisym += model_axisym
sim_axisym += model_axisym.hex_mesh_generator([8, 4])
sim_axisym.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
add_2d_boundaries(sim_axisym, axisymmetric=True)
receiver_coords_axisym = add_2d_pressure_line(
    sim_axisym, source_x=0.25, source_z=0.05, receiver_z=0.04
)
sim_axisym += fs.Discretization()
sim_axisym += fs.SolverConfig(tolerance=0.0001, grids=3)
sim_axisym.to_fs()["global_coordinate_system"]

# %% source cell 20
site_axisym = fs.Site()
job_axisym = fs.TimeDomainJob(
    name="time_axisym", simulation=sim_axisym, f_min=0.0, f_max=30.0, T_max=0.9
)
result_axisym = site_axisym.submit(job_axisym).wait()
traces_axisym = result_axisym.traces(upscale=4)
traces_axisym.summary

# %% source cell 21
group = traces_axisym.groups[0]
component = traces_axisym.components(group)[0]
source = traces_axisym.sources(group)[0]
gather_axisym = traces_axisym.td(
    group, component, source, wavelet, upscale=4, T_max=0.9
)
trace_25d = gather_25d.isel(receiver=len(gather_25d.receiver) // 2)
trace_axisym = gather_axisym.isel(receiver=len(gather_axisym.receiver) // 2)
