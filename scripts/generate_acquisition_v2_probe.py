#!/usr/bin/env python3
"""Generate the five-point/four-field acquisition-v2 solver probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import frequensolve as fs


def build_probe(project_root: Path) -> Path:
    """Write a deterministic project and return its saved job JSON path."""

    project_root = Path(project_root).expanduser().resolve()
    project = fs.Project(name="project", path=project_root)
    simulation = project.new_simulation(
        name="acquisition_v2_probe",
        physics="acoustic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )

    model = fs.LayeredModel(
        name="model",
        dimension=2,
        x_limits=[0.0, 1.0],
    )
    model.add_surface(name="top", depth=0.0 * fs.ureg.km)
    model.add_layer(
        name="water",
        properties={
            "Vp": 1.5 * fs.ureg.km / fs.ureg.s,
            "Rho": 1.0 * fs.ureg.g / fs.ureg.cm**3,
        },
    )
    model.add_surface(name="bottom", depth=0.5 * fs.ureg.km)
    simulation += model

    simulation += model.hex_mesh_generator([8, 4])
    simulation.mesh.set_adapt(
        elems_per_wave=2.0,
        order=4,
        f_low=5.0,
        f_high=30.0,
    )
    simulation.mesh.set_source_grading(d0=0.02, d1=0.08, factor=2.0)
    simulation += fs.BoundaryCondition(
        conditions=["free"],
        boundaries=["z_min"],
    )
    simulation += fs.BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=0.75,
    )

    source_geometry = fs.SourceGeometry.points(
        kind="scalar",
        coords=[
            [0.25, 0.05],
            [0.50, 0.05],
            [0.75, 0.05],
            [0.45, 0.08],
            [0.55, 0.08],
        ],
        names=["shot_left", "shot_center", "shot_right", "pair_pos", "pair_neg"],
    )
    source_encoding = fs.SourceEncoding.named(
        {
            "shot_left": {"shot_left": 1.0},
            "shot_center": {"shot_center": 1.0},
            "shot_right": {"shot_right": 1.0},
            "difference": {"pair_pos": 1.0, "pair_neg": -1.0},
        }
    )
    acquisition = fs.Acquisition(
        source_geometry=source_geometry,
        source_encoding=source_encoding,
    )
    hydrophone = fs.ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="line",
        device=hydrophone,
        coords=[[x, 0.04] for x in np.linspace(0.1, 0.9, 17)],
    )
    simulation += acquisition
    simulation += fs.Discretization()
    simulation += fs.SolverConfig(tolerance=1.0e-4, grids=3)

    project.save()
    job = fs.FrequencyDomainJob(
        name="acquisition_v2_probe",
        simulation=simulation,
        f_list=[20.0],
    )
    return job.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "project_root",
        type=Path,
        help="Destination project directory",
    )
    args = parser.parse_args()
    print(build_probe(args.project_root))


if __name__ == "__main__":
    main()
