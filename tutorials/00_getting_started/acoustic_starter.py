"""acoustic-starter-v1: prepare locally; submit only with --submit.

The model matches the MCP known-small-2d-acoustic starter: 10 Hz, two
acoustic layers, 101 pressure receivers and one domain VTK output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import frequensolve as fs

TUTORIAL_ID = "acoustic-starter-v1"


def build_starter(directory: Path):
    """Build and validate the fixed model without network or solver access."""
    project = fs.Project(
        name='acoustic-starter-v1',
        pretty_name='Known-small 2D acoustic',
        path=directory,
        load_if_exists=False,
    )
    simulation = project.new_simulation(
        name='acoustic_starter',
        physics='acoustic',
        dimension=2,
        units={'length': 'km', 'velocity': 'km/s', 'density': 'g/cm^3'},
    )

    model = fs.LayeredModel(
        name='model',
        dimension=2,
        x_limits=[0.0, 1.0],
    )
    model.add_surface(**{'name': 'top', 'depth': 0.0})
    model.add_layer(**{'name': 'upper_layer', 'properties': {'Vp': 2.0, 'Rho': 2.2}})
    model.add_surface(**{'name': 'interface', 'depth': 0.25})
    model.add_layer(**{'name': 'lower_layer', 'properties': {'Vp': 2.8, 'Rho': 2.4}})
    model.add_surface(**{'name': 'bottom', 'depth': 0.5})
    simulation += model

    simulation += model.hex_mesh_generator(n=[8, 4])
    simulation.mesh.set_adapt(**{'elems_per_wave': 2.0, 'order': 4, 'f_low': 5.0, 'f_high': 30.0})
    simulation.mesh.set_source_grading(**{'d1': 0.08, 'd0': 0.02, 'factor': 2.0})

    simulation += fs.BoundaryCondition(**{'conditions': ['free'], 'boundaries': ['z_min']})
    simulation += fs.BoundaryCondition(**{'conditions': ['pml'], 'boundaries': ['x_min', 'x_max', 'z_max'], 'pml_wavelengths': 0.75})

    acquisition = fs.Acquisition()
    acquisition.add_sources(**{'kind': 'scalar', 'coords': [[0.5, 0.025]]})
    receiver = fs.ReceiverNode(name='hydrophone')
    receiver.add_component(**{'name': 'p', 'field': 'pressure'})
    receiver_coordinates = [
        [x, 0.05]
        for x in np.linspace(0.0, 1.0, 101)
    ]
    acquisition.add_receiver_group(
        name='surface',
        device=receiver,
        coords=receiver_coordinates,
    )
    simulation += acquisition

    simulation += fs.Discretization(**{})
    simulation += fs.SolverConfig(**{'tolerance': 0.0001, 'grids': 3})

    vtk_output = fs.VtkOutput.domain(**{'name': 'pressure', 'fields': ['pressure'], 'properties': ['vp', 'rho'], 'upscale': 1})
    job = fs.FrequencyDomainJob(
        name='frequency_10hz',
        simulation=simulation,
        f_list=[10.0],
        outputs=[vtk_output],
    )
    validation = fs.validate_job(job)
    validation.raise_for_errors()
    return project, job


def prepare(directory: Path):
    """Save inspectable inputs, preserving an existing prepared project."""
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Choose an empty directory; existing inputs and results are preserved.")
    project, job = build_starter(directory)
    project.save()
    job.save()
    return project, job


def submit(directory: Path, profile: str):
    """Submit the saved job only after the user chooses --submit explicitly."""
    project = fs.Project.load(directory)
    job = project.load_job("frequency_10hz", simulation="acoustic_starter")
    fs.validate_job(job).raise_for_errors()
    site = fs.Site(profile=profile, interactive=True)
    run = site.submit(job, name="Acoustic starter · 10 Hz")
    print(f"Run ID: {run.id}")
    print("Keep this ID with the saved inputs. Follow this run in Cloud Projects.")
    result = run.wait(check=False)
    print(f"Run status: {result.status}")
    if not result.successful:
        print("Open this run's diagnostics before retrying. Saved inputs are unchanged.")
        return result
    print(result.traces().summary)
    print("Domain VTK files:", result.output_files(base="pressure", suffix=".vtu", existing=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd() / TUTORIAL_ID)
    parser.add_argument("--submit", action="store_true", help="Submit prepared inputs to Cloud; consumes Credits for performed work.")
    parser.add_argument("--profile", default="cloud", help="Existing site.toml Cloud profile.")
    args = parser.parse_args()
    if args.submit:
        submit(args.directory, args.profile)
    else:
        prepare(args.directory)
        print(f"Prepared {args.directory}: one 10 Hz task, 101 pressure receivers.")
        print("No login, upload or solver run occurred. Inspect the saved inputs first.")
        print("When ready: python acoustic_starter.py --submit --profile cloud")


if __name__ == "__main__":
    main()
