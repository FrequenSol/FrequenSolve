"""Generated Cloud benchmark workload.

Source tutorial: 07_performance/01_performance.ipynb
Source SHA-256: eb56da13d5c8726d4b50691c14144eac2cda32ac600c46a48dedf187d3aaab70
"""

# %% source cell 3
import math
from pathlib import Path

import numpy as np

import frequensolve as fs

u = fs.ureg


def _fetch_timing_metadata(site, job):
    fetch = getattr(site, "fetch_run_metadata", None)
    if callable(fetch):
        fetch(job)


# %% source cell 5
def build_performance_simulation(*, name, path, n_sources=64, receiver_count=201):
    project = fs.Project(
        name="project",
        pretty_name="performance_tutorial",
        path=path,
        log_level="INFO",
        log_to_console=True,
    )
    sim = project.new_simulation(
        name=name,
        physics="acoustic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )
    model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.2])
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="water", properties={"Vp": 1.5 * u.km / u.s, "Rho": 1.0 * u.g / u.cm**3}
    )
    model.add_surface(name="interface", depth=0.28 * u.km)
    model.add_layer(
        name="sediment", properties={"Vp": 2.4 * u.km / u.s, "Rho": 2.1 * u.g / u.cm**3}
    )
    model.add_surface(name="bottom", depth=0.7 * u.km)
    sim += model
    sim += model.hex_mesh_generator([10, 5])
    sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=4.0, f_high=28.0)
    sim.mesh.set_source_grading(d0=0.02, d1=0.1, factor=2.0)
    sim.mesh.set_receiver_grading(d0=0.02, d1=0.08, factor=2.0)
    sim += fs.BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    sim += fs.BoundaryCondition(
        conditions=["pml"], boundaries=["x_min", "x_max", "z_max"], pml_wavelengths=0.75
    )
    acq = fs.Acquisition()
    source_x = np.linspace(0.12, 1.08, n_sources)
    source_coords = fs.Q_([[x, 0.04] for x in source_x], "km")
    acq.add_sources(kind="scalar", coords=source_coords)
    hydrophone = fs.ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    dense_coords = fs.Q_(
        [[x, 0.03] for x in np.linspace(0.05, 1.15, receiver_count)], "km"
    )
    qc_coords = fs.Q_([[x, 0.12] for x in np.linspace(0.1, 1.1, 31)], "km")
    monitor_coords = fs.Q_([[0.6, z] for z in np.linspace(0.04, 0.5, 41)], "km")
    acq.add_receiver_group(name="surface_dense", device=hydrophone, coords=dense_coords)
    acq.add_receiver_group(name="surface_qc", device=hydrophone, coords=qc_coords)
    acq.add_receiver_group(
        name="vertical_monitor", device=hydrophone, coords=monitor_coords
    )
    sim += acq
    sim += fs.Discretization()
    sim += fs.SolverConfig(tolerance=0.0001, grids=3)
    return (project, sim)


# %% source cell 7
project_path = Path("./scratch/tutorials/performance")
project, sim = build_performance_simulation(
    name="performance_baseline", path=project_path, n_sources=64
)
model = sim.model
{
    "source_fields": sim.acquisition.source_field_count(),
    "receiver_groups": [group.name for group in sim.acquisition.receiver_groups],
    "receiver_counts": {
        group.name: group.size for group in sim.acquisition.receiver_groups
    },
}

# %% source cell 9
site = fs.Site()
fd_job = fs.FrequencyDomainJob(name="freq_qc", simulation=sim, f_list=[12.0])
fd_job += fs.VtkOutput.domain(
    name="qc",
    path="paraview/qc",
    properties=["vp", "rho"],
    fields=["pressure"],
    sources=[1],
    show_pml=True,
)
fd_result = site.submit(fd_job).wait()
_fetch_timing_metadata(site, fd_job)
fd_job.print_frequency_summary()
fd_job.task_timings()

# %% source cell 11
vtu_files = fd_result.output_files(base="qc", suffix=".vtu", existing=True)
print("VTU outputs:")
for file in vtu_files:
    print(f"  {file}")
mesh_screenshot = project_path / "assets" / "performance_qc_vp.png"
mesh_screenshot.parent.mkdir(parents=True, exist_ok=True)

# %% source cell 13
time_job = fs.TimeDomainJob(
    name="time_sweep", simulation=sim, f_min=0.0, f_max=28.0, T_max=0.9
)
time_result = site.submit(time_job).wait()
_fetch_timing_metadata(site, time_job)
time_job.print_frequency_summary()

# %% source cell 15
timing_rows = time_job.task_timings()

# %% source cell 17
phase_order = ["setup", "mesh", "assembly", "solve_forward", "solve_adjoint", "imaging"]
phase_rows = time_job.phase_timings(phases=phase_order)


# %% source cell 19
def expected_2d_source_batches(n_sources, internal_batch_cap=64):
    return {
        "sources": int(n_sources),
        "internal_2d_batch_cap": int(internal_batch_cap),
        "automatic_sub_batches": math.ceil(int(n_sources) / int(internal_batch_cap)),
    }


source_counts = [1, 8, 16, 32, 64, 96, 128]

# %% source cell 21
batch_site = fs.Site()
batch_rows = []
for n_source_case in source_counts:
    _, batch_sim = build_performance_simulation(
        name=f"sources_{n_source_case:03d}",
        path=project_path,
        n_sources=n_source_case,
        receiver_count=101,
    )
    batch_job = fs.FrequencyDomainJob(
        name=f"sources_{n_source_case:03d}", simulation=batch_sim, f_list=[12.0]
    )
    batch_result = batch_site.submit(batch_job).wait()
    _fetch_timing_metadata(batch_site, batch_job)
    summary = batch_job.print_frequency_summary()
    timings = batch_job.task_timings()
    elapsed = timings[0]["duration_seconds"] if timings else np.nan
    batch_rows.append(
        {
            **expected_2d_source_batches(n_source_case),
            "succeeded": summary["succeeded"],
            "failed": summary["failed"],
            "duration_seconds": elapsed,
            "seconds_per_source": elapsed / n_source_case if elapsed else np.nan,
        }
    )

# %% source cell 25
traces = time_result.traces(upscale=4)
traces.summary

# %% source cell 26
wavelet = fs.RickerWavelet(f=12.0)
