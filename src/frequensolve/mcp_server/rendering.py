"""Deterministic Python rendering for the constrained starter draft."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["render_starter_python_source"]


def render_starter_python_source(setup: Mapping[str, Any]) -> str:
    """Render validated starter setup without save, submit, or run calls."""

    project = setup["project"]
    simulation = setup["simulation"]
    model = setup["model"]
    mesh = setup["mesh"]
    boundaries = setup["boundary_conditions"]
    acquisition = setup["acquisition"]
    receiver = acquisition["receiver_group"]
    line = receiver["coordinate_line"]
    job = setup["job"]
    vtk = job["outputs"]["vtk"][0]

    source = f"""from pathlib import Path

import numpy as np

import frequensolve as fs


project = fs.Project(
    name={project["name"]!r},
    pretty_name={project["pretty_name"]!r},
    path=Path.cwd() / {project["name"]!r},
    load_if_exists=False,
)
simulation = project.new_simulation(
    name={simulation["name"]!r},
    physics={simulation["physics"]!r},
    dimension={simulation["dimension"]!r},
    units={simulation["units"]!r},
)

model = fs.LayeredModel(
    name={model["name"]!r},
    dimension={model["dimension"]!r},
    x_limits={model["x_limits"]!r},
)
"""
    for index, surface in enumerate(model["surfaces"]):
        source += f"model.add_surface(**{surface!r})\n"
        if index < len(model["layers"]):
            source += f"model.add_layer(**{model['layers'][index]!r})\n"

    source += f"""simulation += model

simulation += model.hex_mesh_generator(n={mesh["n"]!r})
simulation.mesh.set_adapt(**{mesh["adapt"]!r})
simulation.mesh.set_source_grading(**{mesh["source_grading"]!r})

"""
    for boundary in boundaries:
        source += f"simulation += fs.BoundaryCondition(**{boundary!r})\n"

    source += f"""
acquisition = fs.Acquisition()
acquisition.add_sources(**{acquisition["source"]!r})
receiver = fs.ReceiverNode(name={receiver["device_name"]!r})
receiver.add_component(**{receiver["component"]!r})
receiver_coordinates = [
    [x, {line["fixed"]["z"]!r}]
    for x in np.linspace({line["start"]!r}, {line["stop"]!r}, {line["count"]!r})
]
acquisition.add_receiver_group(
    name={receiver["name"]!r},
    device=receiver,
    coords=receiver_coordinates,
)
simulation += acquisition

simulation += fs.Discretization(**{setup["discretization"]!r})
simulation += fs.SolverConfig(**{setup["solver"]!r})

vtk_output = fs.VtkOutput.domain(**{vtk!r})
job = fs.FrequencyDomainJob(
    name={job["name"]!r},
    simulation=simulation,
    f_list={job["f_list"]!r},
    outputs=[vtk_output],
)
validation = fs.validate_job(job)
validation.raise_for_errors()
"""
    return source
