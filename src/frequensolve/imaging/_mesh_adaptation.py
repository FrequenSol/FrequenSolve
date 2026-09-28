"""Explicit native mesh adaptation at control-layout boundaries."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from ._artifacts import ControlVectorFile
from .controls import ControlSpace, MeshParameters
from .jobs import MeshAdaptationJob


def adapt_meshes(
    problem: Any,
    space: ControlSpace,
    state: Any,
    averaging_wavelengths: float,
    mesh_keys: set[str],
) -> tuple[ControlSpace, ControlVectorFile]:
    """Size from the full accepted material; retain its reference and transforms."""
    from ._backend import fingerprint

    previous = problem.full_space.specs
    changed = {
        key: spec
        for key, spec in space.specs.items()
        if isinstance(spec, MeshParameters) and key in mesh_keys
    }
    if not changed:
        raise ValueError(
            "mesh adaptation requires an explicitly changed MeshParameters block"
        )
    for key, spec in changed.items():
        old = previous.get(key)
        if not isinstance(old, MeshParameters) or any(
            getattr(old, name) != getattr(spec, name)
            for name in ("prop", "subdomain", "transform", "id")
        ):
            raise ValueError(
                "mesh adaptation must retain each block's property, material, transform and id"
            )
    identity = fingerprint(
        source=problem.identity(),
        state=state.values,
        controls={k: repr(v) for k, v in changed.items()},
        averaging_wavelengths=averaging_wavelengths,
        algorithm="material-slowness-gauss5-nodal-transfer-v1",
    )
    directory = problem.backend.staging_dir("mesh_adaptation", identity.split(":")[-1])
    definitions = {}
    for key, spec in changed.items():
        definition = spec.property_space(key).to_fs()
        definition["artifact"] = str(directory / (spec.block_id(key) + ".h5"))
        definitions[spec.block_id(key)] = definition
    full = state.space.without_support()
    problem._sync_simulation(state)
    source = problem._linearize_job(full, None, gradient=True)
    job = MeshAdaptationJob(
        source,
        input_vector=ControlVectorFile(
            {
                b.name: state.values[full.full_slices[b.name]]
                for b in full.resolved_blocks
                if b.name.startswith("model.")
            },
            control_spaces={
                b.name: b.basis_identity
                for b in full.resolved_blocks
                if b.name.startswith("model.") and b.basis_identity
            },
        ),
        property_spaces=definitions,
        source_identity=identity,
        frequency=max(v["frequency"] for v in definitions.values()),
        averaging_wavelengths=averaging_wavelengths,
        name=problem.backend.job_name("mesh_adaptation"),
    )
    problem.backend.run(job, postprocess_only=True)
    report = json.loads(job.result.read_text())
    if (
        report.get("schema") != "fs-control-mesh-adaptation-result-1"
        or report.get("source_identity") != identity
    ):
        raise ValueError("mesh adaptation returned another source state")
    specs = dict(space.specs)
    for key, spec in changed.items():
        specs[key] = dataclasses.replace(
            spec, artifact=report["artifacts"][spec.block_id(key)]
        )
    return ControlSpace(**specs), ControlVectorFile.read(job.gradient_file())
