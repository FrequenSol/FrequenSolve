"""Generate an elastic transient VTU time series with native Helmholtz velocities.

Run with an installed FrequenSolve SDK and a current Sauce executable::

    python elastic_transient_helmholtz.py --solver /path/to/fs2d_s --project ./elastic_demo

Open the printed elastic_velocity.pvd file in ParaView and select Magnitude
for velocity_1_re, p_velocity_1_re, or s_velocity_1_re. Geometry is in metres,
velocity in m/s, and time in seconds. The 2D x-z plane is stored in VTK's XY plane.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import frequensolve as fs
from frequensolve.orchestrator.sites.local import LocalSite

FIELDS = ("velocity", "p_velocity", "s_velocity")


def create_job(project_path: Path) -> fs.TransientJob:
    """Excite separate P and S fronts with a vertical point force in a uniform solid."""
    project = fs.Project(name="elastic_demo", path=project_path)
    sim = project.new_simulation(name="elastic", physics="elastic", dimension=2)
    sim.f0 = 8.0
    sim += fs.Discretization(method="Galerkin")
    sim += fs.SolverConfig(
        grids=2,
        tolerance=1e-4,
        max_iter=300,
        galerkin_multigrid=True,
        gpu_backend="none",
        forms_backend="cpu",
    )
    model = fs.LayeredModel(name="uniform_solid", dimension=2, x_limits=[0, 1])
    model.attenuation_model = "none"
    model.add_surface(name="top", depth=fs.Q_(0, "km"))
    model.add_layer(
        name="solid",
        properties={
            "Vp": fs.Q_(2.5, "km/s"),
            "Vs": fs.Q_(1.2, "km/s"),
            "Rho": fs.Q_(2.0, "g/cm^3"),
        },
    )
    model.add_surface(name="bottom", depth=fs.Q_(1, "km"))
    sim += model
    sim += model.hex_mesh_generator([2, 2])
    sim.mesh.set_adapt(elems_per_wave=2.0, order=3, f_adapt=16, adapt_sources=0)
    sim += fs.BoundaryCondition(
        name="outer",
        boundaries=["x_min", "x_max", "z_min", "z_max"],
        conditions=["fixed"],
    )
    acquisition = fs.Acquisition()
    acquisition.add_sources(
        kind="vector", coords=fs.Q_([[0.5, 0.5]], "km"), direction=[0, 1]
    )
    sim += acquisition
    sim.extra["helmholtz_projection"] = {
        "enabled": True,
        "field": "velocity",
        "parts": ["p", "s"],
        "s_projection": "complement",
    }
    job = fs.TransientJob(
        "helmholtz",
        sim,
        time=fs.TimeIntegration(start=-0.25, end=0.30, dt=0.0025),
        source=fs.TimeFunction.ricker(frequency=8, delay=0.15),
        output_schedule=fs.TimeOutput(
            times=np.linspace(0, 0.30, 31), write_initial=False
        ),
    )
    job += fs.OutputUnits(geometry="m", fields={field: "m/s" for field in FIELDS})
    job += fs.VtkOutput(
        name="elastic_velocity",
        fields=FIELDS,
        format="vtu",
        encoding="appended",
        target="volume",
        order=2,
        upscale=1,
    )
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("elastic_demo"))
    args = parser.parse_args()
    job = create_job(args.project.resolve())
    site = LocalSite(
        solver=str(args.solver.resolve()),
        solver_policy="off",
        n_workers=1,
        threads_per_worker=8,
        shutdown_on_completion=True,
        environment={"OMPI_MCA_btl": "self", "VECLIB_MAXIMUM_THREADS": "1"},
    )
    site.run(job, check=True)
    print(job._result_path / "ParaView" / "elastic_velocity.pvd")


if __name__ == "__main__":
    main()
