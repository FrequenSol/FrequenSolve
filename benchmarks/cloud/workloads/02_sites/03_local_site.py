"""Generated Cloud benchmark workload.

Source tutorial: 02_sites/03_local_site.ipynb
Source SHA-256: 20b7fd7a607050df5814c7d21ca350663efba91804d2fac6df3dca4e09577ee1
"""

# %% source cell 6
import numpy as np

import frequensolve as fs

u = fs.ureg


# %% source cell 7
def build_acoustic_tutorial_jobs(
    project_path, *, simulation_name, trace_job_name, qc_job_name, f_max=25.0
):
    project = fs.Project(
        name="project",
        pretty_name=simulation_name,
        path=project_path,
        log_level="INFO",
        log_to_console=True,
    )
    sim = project.new_simulation(
        name=simulation_name,
        physics="acoustic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )
    model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
    )
    model.add_surface(name="interface", depth=0.22 * u.km)
    model.add_layer(
        name="basement", properties={"Vp": 2.4 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3}
    )
    model.add_surface(name="bottom", depth=0.5 * u.km)
    sim += model
    sim += model.hex_mesh_generator([8, 4])
    sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=f_max)
    sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += fs.BoundaryCondition(
        conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
    )
    acq = fs.Acquisition()
    acq.add_sources(kind="scalar", coords=[[0.35, 0.05], [0.65, 0.05]])
    hydrophone = fs.ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acq.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[x, 0.04] for x in np.linspace(0.1, 0.9, 61)],
    )
    sim += acq
    sim += fs.Discretization()
    sim += fs.SolverConfig(tolerance=0.0001, grids=3)
    trace_job = fs.TimeDomainJob(
        name=trace_job_name, simulation=sim, f_min=0.0, f_max=f_max, T_max=0.9
    )
    qc_job = fs.FrequencyDomainJob(
        name=qc_job_name,
        simulation=sim,
        f_list=[12.0],
        outputs=[
            fs.VtkOutput.domain(
                name="pv_qc",
                fields=["pressure"],
                properties=["vp", "rho", "Subdomain"],
                show_pml=True,
                upscale=0,
                order=1,
            )
        ],
    )
    return (project, sim, trace_job, qc_job)


# %% source cell 9
local_site = fs.Site(profile="local")

# %% source cell 11
(project, sim, trace_job, qc_job) = build_acoustic_tutorial_jobs(
    "./scratch/tutorials/local_site",
    simulation_name="local_site_acoustic",
    trace_job_name="time_local_site",
    qc_job_name="freq_local_site_qc",
)
trace_site = fs.Site(profile="local")
trace_result = trace_site.submit(trace_job).wait()
qc_site = fs.Site(profile="local")
qc_result = qc_site.submit(qc_job).wait()
{"trace_status": trace_result.status, "qc_status": qc_result.status}

# %% source cell 13
logs = trace_result.logs()
qc_outputs = qc_result.output_files(existing=True)
traces = trace_result.traces(upscale=4)
{
    "trace_successful": trace_result.successful,
    "qc_successful": qc_result.successful,
    "logs": str(logs),
    "qc_output_count": len(qc_outputs),
    "trace_files": traces.files,
    "frequency_summary": trace_job.frequency_summary(),
}

# %% source cell 15
traces = trace_result.traces(upscale=4)
wavelet = fs.RickerWavelet(f=12.0)
group = traces.groups[0]
component = traces.components(group)[0]
source = traces.sources(group)[0]
gather = traces.td(group, component, source, wavelet, upscale=4, T_max=0.9)
fs.plot_gather(
    gather,
    A=2.0 * np.nanstd(np.real(gather.values)),
    cmap="gray",
    figsize=(9, 4),
    title=f"{trace_job.name}: {group}/{component}/source {source}",
)
