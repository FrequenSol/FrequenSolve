"""Generated Cloud benchmark workload.

Source tutorial: 02_sites/04_save_load_projects_jobs.ipynb
Source SHA-256: 8464814c780038496c15036777804a9becb9c866ec5d695e4fd2a0bca8249538
"""

import json

# %% source cell 4
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg

# %% source cell 6
project = fs.Project(
    name="project",
    pretty_name="Save Load Tutorial",
    path="./scratch/tutorials/save_load_projects_jobs",
    log_level="INFO",
    log_to_console=True,
)
sim = project.new_simulation(
    name="save_load_acoustic",
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
sim += model.hex_mesh_generator([4, 2])
sim.mesh.set_adapt(elems_per_wave=2.0, order=3, f_low=5.0, f_high=20.0)
sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
sim += fs.BoundaryCondition(
    conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
)
acq = fs.Acquisition()
acq.add_sources(kind="scalar", coords=[[0.5, 0.05]])
receiver = fs.ReceiverNode(name="hydrophone")
receiver.add_component(name="p", field="pressure")
acq.add_receiver_group(
    name="surface",
    device=receiver,
    coords=[[x, 0.04] for x in np.linspace(0.1, 0.9, 21)],
)
sim += acq
sim += fs.Discretization()
sim += fs.SolverConfig(tolerance=0.0001, grids=3)

# %% source cell 8
time_job = fs.TimeDomainJob(
    name="time", simulation=sim, f_min=0.0, f_max=20.0, T_max=0.5
)
freq_job = fs.FrequencyDomainJob(
    name="freq_qc",
    simulation=sim,
    f_list=[15.0],
    outputs=[
        fs.VtkOutput.domain(
            name="pv",
            fields=["pressure"],
            properties=["vp", "rho", "Subdomain"],
            show_pml=True,
            upscale=0,
            order=1,
        )
    ],
)

# %% source cell 10
project_file = project.save()
sim_file = sim.save()
time_job_file = time_job.save()
freq_job_file = freq_job.save()
project_root = project.path


def rel(path):
    return str(Path(path).resolve().relative_to(project_root))


{
    "project": rel(project_file),
    "simulation": rel(sim_file),
    "time_job": rel(time_job_file),
    "frequency_job": rel(freq_job_file),
}

# %% source cell 12
saved_files = sorted(
    (
        path.relative_to(project_root)
        for path in project_root.rglob("*.json")
        if ".tmp" not in path.name
    )
)
[str(path) for path in saved_files]

# %% source cell 14
loaded_project = fs.Project.load(project_file)
loaded_project_from_dir = fs.Project.load(project_root)
loaded_sim_from_project = loaded_project.simulations["save_load_acoustic"]
loaded_sim_direct = fs.SeismicSimulation.load(sim_file)
{
    "project_name": loaded_project.name,
    "project_from_dir": loaded_project_from_dir.name,
    "simulation_from_project": loaded_sim_from_project.name,
    "simulation_direct": loaded_sim_direct.name,
    "loaded_physics": loaded_sim_direct.physics,
    "loaded_receiver_groups": [
        group.name for group in loaded_sim_direct.acquisition.receiver_groups
    ],
}

# %% source cell 16
loaded_time_job = fs.BaseJob.load(time_job_file)
loaded_freq_job = fs.BaseJob.load(freq_job_file)
{
    "time_job_class": type(loaded_time_job).__name__,
    "time_job_name": loaded_time_job.name,
    "time_job_simulation": loaded_time_job.simulation.name,
    "time_job_frequencies": len(loaded_time_job.f_list),
    "freq_job_class": type(loaded_freq_job).__name__,
    "freq_job_name": loaded_freq_job.name,
    "freq_job_outputs": [output.name for output in loaded_freq_job.outputs.vtk],
}

# %% source cell 18
job_payload = json.loads(Path(freq_job_file).read_text())
{
    "schema": job_payload["schema"],
    "type": job_payload["_type"],
    "simulation": job_payload["simulation"],
    "workflow": job_payload["workflow"],
    "n_frequencies": len(job_payload["f_list"]),
    "outputs": list(job_payload["Outputs"].keys()),
    "result_path": job_payload["result_path"],
}

# %% source cell 20
site = fs.Site()
result = site.submit(loaded_time_job).wait()
traces = result.traces(upscale=4)
traces.summary
