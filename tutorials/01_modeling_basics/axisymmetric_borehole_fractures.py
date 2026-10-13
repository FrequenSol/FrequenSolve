# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Compare formation fractures around an intact, fluid-filled steel casing.

Author JSON with --directory PATH; add --solver /path/to/fs2d to run.
The (r,z) section revolves about r=0. The source at r=0.035 m is a ring,
5 m below the top by default. Compare linear slip and reduced acoustic;
the resolved fluid model remains available through --model fluid_filled.
Use --model lower_dimensional_viscous_fluid for water viscosity with 12 aperture modes.
An annular fracture has 1 cm maximum aperture and closes at r=20 m.
Unshifted cylindrical PML surrounds the physical region, with impedance
termination. The fracture is confined to the formation outside the casing.
Defaults use order 4, 2 elements per wavelength, fast solver mode, and a
25 m PML thickness cap, running to 0.12 s. Mesh density and polynomial order are configurable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import xarray as xr

import frequensolve as fs

MODELS = (
    "fluid_filled",
    "linear_slip",
    "lower_dimensional_fluid",
    "lower_dimensional_viscous_fluid",
)
DEFAULT_MODELS = ("linear_slip", "lower_dimensional_fluid")


def create_job(
    directory: Path,
    model_kind: str,
    dt: float,
    end: float,
    source_depth: float = 5.0,
    elems_per_wave: float = 2.0,
    order: int = 4,
    pml_size: float = 25.0,
    solver_mode: str = "fast",
) -> fs.TransientJob:
    """Build the same cased borehole and tapered annular fracture for each law."""
    project = fs.Project(name="borehole_fractures", path=directory)
    sim = project.new_simulation(name=model_kind, physics="coupled_axisym", dimension=2)
    sim.axisymmetric = True
    sim.extra["f0"] = 100
    sim += fs.Discretization(method="Galerkin")
    sim += fs.SolverConfig(
        grids=1,
        tolerance=1e-4,
        galerkin_multigrid=False,
        gpu_backend="none",
        forms_backend="cpu",
        mode=solver_mode,
    )
    model = fs.LayeredModel(
        name="cased_formation", dimension=2, x_limits=fs.Q_([0, 40], "m")
    )
    model.attenuation_model = "none"
    model.add_surface(name="top", depth=fs.Q_(0, "m"))
    model.add_layer(
        name="formation",
        properties={
            "Vp": fs.Q_(2500, "m/s"),
            "Vs": fs.Q_(1200, "m/s"),
            "Rho": fs.Q_(2000, "kg/m^3"),
        },
    )
    gap = xr.DataArray(
        [-0.01, 0, 0.01, 0.008, 0.004, 0, -0.01],
        dims=["r"],
        coords={
            "r": xr.DataArray(
                [0, 0.2, 0.5, 5, 12, 20, 40], dims=["r"], attrs={"units": "m"}
            )
        },
        attrs={"units": "m"},
    )
    fracture_options = {}
    if model_kind == "fluid_filled":
        fracture_options["properties"] = {
            "Vp": fs.Q_(1500, "m/s"),
            "Rho": fs.Q_(1000, "kg/m^3"),
        }
    else:
        fracture_options["constitutive"] = {
            "fluid_density": 1000,
            "fluid_bulk_modulus": 2.25e9,
            "tangential_stiffness": 0,
        }
        if model_kind == "lower_dimensional_viscous_fluid":
            fracture_options["constitutive"] = {
                "fluid_density": 1000,
                "fluid_bulk_modulus": 2.25e9,
                "dynamic_viscosity": 0.001,
                "aperture_modes": 12,
            }
    model.add_fracture(
        name="formation_fracture",
        depth=fs.Q_(40, "m"),
        gap=gap,
        fracture_model=model_kind,
        interface=False,
        adapt={
            "krauklis": {
                "enabled": True,
                "normal": {"h0": 0.5, "thickness": 3.0, "grading": 2.0},
                "tangential": {"factor": 1.0, "thickness": 3.0, "grading": 2.0},
            }
        },
        **fracture_options,
    )
    model.add_surface(name="bottom", depth=fs.Q_(80, "m"))
    model.add_borehole(
        name="well",
        x=fs.Q_(0, "m"),
        top="top",
        bottom="bottom",
        layers=[
            dict(
                name="borehole_fluid",
                r=fs.Q_(0.10, "m"),
                physics="acoustic",
                n=2,
                properties={"Vp": fs.Q_(1500, "m/s"), "Rho": fs.Q_(1000, "kg/m^3")},
            ),
            dict(
                name="steel_casing",
                r=fs.Q_(0.11, "m"),
                physics="elastic",
                n=2,
                properties={
                    "Vp": fs.Q_(5900, "m/s"),
                    "Vs": fs.Q_(3200, "m/s"),
                    "Rho": fs.Q_(7850, "kg/m^3"),
                },
            ),
        ],
    )
    sim += model
    sim += model.hex_mesh_generator([5, 8])
    sim.mesh.set_adapt(
        elems_per_wave=elems_per_wave, order=order, f_adapt=250, adapt_sources=1
    )
    sim += fs.BoundaryCondition(name="axis", boundaries=["x_min"], conditions=["axis"])
    sim += fs.BoundaryCondition(
        name="remote_absorbing",
        boundaries=["x_max", "z_min", "z_max"],
        conditions=["pml:impedance"],
        pml_wavelengths=3,
        pml_reflection=1e-4,
        pml_exponent=3,
        stretch_limit=0.75,
        size_limit={"value": pml_size, "units": "m"},
    )
    acquisition = fs.Acquisition()
    acquisition.add_sources(
        kind="scalar",
        coords=fs.Q_([[0.035, source_depth]], "m"),
        amplitude=fs.Q_(1, "N*m"),
    )
    sim += acquisition
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(start=-0.015, end=end, dt=dt),
        source=fs.TimeFunction.ricker(frequency=100, delay=0.005),
        output_schedule=fs.TimeOutput(
            times=[i * 0.001 for i in range(round(end / 0.001) + 1)],
            write_initial=False,
        ),
    )
    fields = ["velocity", "pressure"]
    if model_kind in ("lower_dimensional_fluid", "lower_dimensional_viscous_fluid"):
        fields.append("elastic:fracture_pressure")
    job += fs.OutputUnits(geometry="m", fields={"velocity": "m/s", "pressure": "Pa"})
    job += fs.VtkOutput(
        name=model_kind,
        fields=fields,
        format="vtu",
        encoding="appended",
        target="volume",
        order=2,
        upscale=1,
    )
    return job


def main():
    """Write reviewable API inputs, optionally generate meshes and run locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--dt", type=float, default=0.0000625)
    parser.add_argument("--end", type=float, default=0.12)
    parser.add_argument(
        "--source-depth",
        type=float,
        default=5.0,
        help="Source depth in metres (default: 5).",
    )
    parser.add_argument(
        "--elems-per-wave",
        type=float,
        default=2.0,
        help="Mesh adaptation density at 250 Hz (default: 2).",
    )
    parser.add_argument(
        "--order", type=int, default=4, help="Solution polynomial order (default: 4)."
    )
    parser.add_argument(
        "--pml-size",
        type=float,
        default=25.0,
        help="PML thickness cap in metres (default: 25).",
    )
    parser.add_argument(
        "--solver-mode", default="fast", help="Native solver preset (default: fast)."
    )
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.source_depth < 80:
        parser.error("--source-depth must lie inside the domain (0, 80) m")
    if not args.elems_per_wave > 0:
        parser.error("--elems-per-wave must be positive")
    if args.order < 1 or not args.pml_size > 0:
        parser.error("--order and --pml-size must be positive")
    for kind in args.model or DEFAULT_MODELS:
        work = args.directory.resolve() / kind
        work.mkdir(parents=True, exist_ok=True)
        job = create_job(
            work,
            kind,
            args.dt,
            args.end,
            args.source_depth,
            args.elems_per_wave,
            args.order,
            args.pml_size,
            args.solver_mode,
        )
        job.simulation.save()
        sim_payload = job.simulation.to_fs()
        sim_payload["Mesh"]["generator"]["path"] = str(work / "mesh")
        job_payload = job.to_fs()
        job_payload["simulation"] = str(work / "simulation.json")
        job_payload["result_path"] = str(work / "results")
        (work / "simulation.json").write_text(json.dumps(sim_payload, indent=2))
        (work / "job.json").write_text(json.dumps(job_payload, indent=2))
        if args.solver:
            command = [
                str(args.solver.resolve()),
                "-nthreads",
                str(args.threads),
                "-j",
                str(work / "job.json"),
                "--result-dir",
                str(work / "results"),
                "--tmp-directory",
                str(work / "tmp"),
            ]
            timing = {}
            for phase, flags in [("mesh", ["--mesh"]), ("solve", ["-i", "1"])]:
                start = time.perf_counter()
                with (work / (phase + ".log")).open("w") as log:
                    subprocess.run(
                        command + flags,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        env=dict(
                            os.environ, OMPI_MCA_btl="self", VECLIB_MAXIMUM_THREADS="1"
                        ),
                    )
                timing[phase] = time.perf_counter() - start
                print(kind, phase, timing[phase], flush=True)
            (work / "timing.json").write_text(json.dumps(timing, indent=2))
        print(work / "job.json", flush=True)


if __name__ == "__main__":
    main()
