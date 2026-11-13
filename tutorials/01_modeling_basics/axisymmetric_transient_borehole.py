"""Fluid-filled, steel-cased borehole with transient P/S velocity output.

Run with --solver /path/to/fs2d_s --project ./borehole_demo. Open the printed
borehole_velocity.pvd in ParaView. Coordinates are (r,z), stored in VTK's XY
plane, in metres; vector components are (v_r,v_z,0), in m/s; time is seconds.
The off-axis monopole represents an axisymmetric ring source in the fluid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import frequensolve as fs
from frequensolve.orchestrator.sites.local import LocalSite

FIELDS = ("velocity", "p_velocity", "s_velocity")


def create_job(project_path: Path) -> fs.TransientJob:
    """Author a meridional borehole section using ordinary model layers."""
    project = fs.Project(name="borehole_demo", path=project_path)
    sim = project.new_simulation(
        name="cased_borehole", physics="coupled_axisym", dimension=2
    )
    sim.axisymmetric = True
    sim.f0 = 5000
    sim += fs.Discretization(method="Galerkin")
    sim += fs.SolverConfig(
        grids=2,
        tolerance=1e-4,
        max_iter=300,
        galerkin_multigrid=True,
        gpu_backend="none",
        forms_backend="cpu",
    )
    model = fs.LayeredModel(name="borehole", dimension=2, x_limits=fs.Q_([0, 1], "m"))
    model.attenuation_model = "none"
    model.add_surface(name="top", depth=fs.Q_(0, "m"))
    model.add_layer(
        name="formation",
        properties={
            "Vp": fs.Q_(3500, "m/s"),
            "Vs": fs.Q_(1800, "m/s"),
            "Rho": fs.Q_(2300, "kg/m^3"),
        },
    )
    model.add_surface(name="bottom", depth=fs.Q_(2, "m"))
    model.add_borehole(
        name="well",
        x=fs.Q_(0, "m"),
        top="top",
        bottom="bottom",
        layers=[
            dict(
                name="fluid",
                r=fs.Q_(0.10, "m"),
                physics="acoustic",
                properties={"Vp": fs.Q_(1500, "m/s"), "Rho": fs.Q_(1000, "kg/m^3")},
                n=2,
            ),
            dict(
                name="casing",
                r=fs.Q_(0.11, "m"),
                physics="elastic",
                properties={
                    "Vp": fs.Q_(5900, "m/s"),
                    "Vs": fs.Q_(3200, "m/s"),
                    "Rho": fs.Q_(7850, "kg/m^3"),
                },
                n=2,
            ),
        ],
    )
    sim += model
    sim += model.hex_mesh_generator([4, 8])
    sim.mesh.set_adapt(elems_per_wave=2, order=3, f_adapt=10000, adapt_sources=0)
    sim += fs.BoundaryCondition(name="axis", boundaries=["x_min"], conditions=["axis"])
    sim += fs.BoundaryCondition(
        name="outer", boundaries=["x_max", "z_min", "z_max"], conditions=["impedance"]
    )
    acquisition = fs.Acquisition()
    acquisition.add_sources(
        kind="scalar", coords=fs.Q_([[0.035, 1.0]], "m"), amplitude=fs.Q_(1, "N*m")
    )
    sim += acquisition
    sim.extra["helmholtz_projection"] = {
        "enabled": True,
        "field": "velocity",
        "parts": ["p", "s"],
        "s_projection": "complement",
    }
    job = fs.TransientJob(
        "transient",
        sim,
        time=fs.TimeIntegration(start=-0.0004, end=0.0006, dt=0.000002),
        source=fs.TimeFunction.ricker(frequency=5000, delay=0.00024),
        output_schedule=fs.TimeOutput(
            times=[i * 0.00002 for i in range(31)], write_initial=False
        ),
    )
    job += fs.OutputUnits(geometry="m", fields={field: "m/s" for field in FIELDS})
    job += fs.VtkOutput(
        name="borehole_velocity",
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
    parser.add_argument("--project", type=Path, default=Path("borehole_demo"))
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
    print(job._result_path / "ParaView" / "borehole_velocity.pvd")


if __name__ == "__main__":
    main()
