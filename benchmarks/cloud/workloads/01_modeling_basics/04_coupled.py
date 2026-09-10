"""Generated Cloud benchmark workload.

Source tutorial: 01_modeling_basics/04_coupled.ipynb
Source SHA-256: cc737a2c5f879d8ed8dfcaacb37c2571d48db11840720c7bbaae2ab0aba8bbe9
"""

# %% source cell 2
import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    path="./scratch/coupled_aep_basic",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="coupled_aep_basic",
    physics="coupled_aep",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)
model = fs.LayeredModel(dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="water", physics="acoustic", properties={"Vp": 1.5, "Rho": 1.0, "Qp": 100.0}
)
model.add_surface(name="seafloor", depth=0.2 * u.km)
model.add_layer(
    name="elastic_sediment",
    physics="elastic",
    properties={"Vp": 2.3, "Vs": 1.1, "Rho": 2.1, "Qp": 40.0, "Qs": 20.0},
)
model.add_surface(name="poro_interface", depth=0.4 * u.km)
model.add_layer(
    name="poroelastic_basement",
    physics="poroelastic:iso",
    properties={
        "Vp": 2.8 * u.km / u.s,
        "Qp": 20.0,
        "Vs": 1.5 * u.km / u.s,
        "Qs": 10.0,
        "Rho": 2.4 * u.g / u.cm**3,
        "k_solid": 30.0 * u.GPa,
        "k_fluid": 2.2 * u.GPa,
        "rho_solid": 2.7 * u.g / u.cm**3,
        "rho_fluid": 1.0 * u.g / u.cm**3,
        "porosity": 0.18,
        "tortuosity": 2.2,
        "kappa": 2e-09 * u.m**2,
        "viscosity": 0.001 * u.Pa * u.s,
    },
)
model.add_surface(name="bottom", depth=0.6 * u.km)
sim += model

# %% source cell 9
sim += model.hex_mesh_generator([8, 4])
sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
sim.mesh.set_source_grading(d1=0.08, factor=2.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)

# %% source cell 11
acq = fs.Acquisition()
acq.add_sources(kind="vector", coords=[[0.5, 0.25]], direction=[0.0, 1.0])
hydrophone = fs.ReceiverNode(name="hydrophone")
hydrophone.add_component(name="p", field="pressure")
water_line = [[x, 0.05] for x in np.linspace(0.0, 1.0, 201)]
acq.add_receiver_group(name="water_line", device=hydrophone, coords=water_line)
geophone = fs.ReceiverNode(name="geophone")
geophone.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
sediment_line = [[x, 0.25] for x in np.linspace(0.0, 1.0, 201)]
acq.add_receiver_group(name="sediment_line", device=geophone, coords=sediment_line)
poro_node = fs.ReceiverNode(name="poro_pressure_node")
poro_node.add_component(name="pore_pressure", field="pressure")
poro_line = [[x, 0.45] for x in np.linspace(0.0, 1.0, 201)]
acq.add_receiver_group(name="poro_line", device=poro_node, coords=poro_line)
sim += acq

# %% source cell 13
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)
site = fs.Site()
job = fs.TimeDomainJob(
    name="time_coupled_aep", simulation=sim, f_min=0.0, f_max=45.0, T_max=2.0
)
result = site.submit(job, skip=False).wait()
traces = result.traces(upscale=4)
traces.summary

# %% source cell 15
wavelet = fs.RickerWavelet(f=15.0)
selections = [
    ("water_line", "p", "Acoustic layer: pressure"),
    ("sediment_line", "v_z", "Elastic layer: vertical velocity"),
    ("poro_line", "pore_pressure", "Poroelastic layer: pore pressure"),
]
gathers = {}

# %% source cell 17
site_fd = fs.Site()
fd_job = fs.FrequencyDomainJob(
    name="freq_coupled_aep_qc",
    simulation=sim,
    f_list=[30.0],
    outputs=[
        fs.VtkOutput.domain(
            name="pv_coupled_aep",
            path="paraview",
            fields=["pressure", "velocity"],
            properties=["vp"],
            sources=[1],
            show_pml=True,
            upscale=0,
            order=1,
        )
    ],
)
fd_result = site_fd.submit(fd_job).wait()
fd_result.output_files(base="pv_coupled_aep", suffix=".vtu", existing=True)
