# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Material update cores expressed in the authoritative parent control layout."""

from pathlib import Path

import h5py
import numpy as np

from frequensolve.units import ureg

from .controls import (
    _axis_names,
    _BindContext,
    _lattice_support,
    _profile_support,
    _sample_grid,
)


def core_masks(problem, prepared, index):
    """Keep complete basis supports intersecting this patch's requested roots."""
    patch = prepared.geometry["patches"][index]
    coverage = {item["space"]: item for item in patch.get("material_coverage", ())}
    roots = {item["root"]: item for item in prepared.geometry["roots"]}
    selected = [roots[root] for root in patch["core_roots"]]
    simulation = problem._patch_runtime.parent.simulation
    ctx = _BindContext(simulation)
    native_per_m = ctx.length(1 * ureg.m)[0]
    boxes = [
        {
            axis: (float(lo) * native_per_m, float(hi) * native_per_m)
            for axis, lo, hi in zip(
                _axis_names(simulation.dimension),
                root["sampled_lower"],
                root["sampled_upper"],
            )
        }
        for root in selected
    ]
    model = simulation.model.to_fs(simulation.export_context())
    masks = {}
    for block in problem.space.resolved_blocks:
        mask = np.zeros(block.size, bool)
        if block.kind == "mesh":
            space = block.control.space
            # Property-space names are independent of user block keys.
            if space not in coverage:
                space = block.name.removeprefix("model.")
            if space not in coverage:
                raise ValueError(
                    f"Patch core lacks canonical coverage for {block.name!r}"
                )
            definition = model["property_spaces"][space]
            artifact = Path(definition["artifact"])
            if not artifact.is_absolute():
                artifact = Path(simulation.project_path) / artifact
            with h5py.File(artifact, "r") as h5:
                identity = h5["property_space/identity"][()]
                if isinstance(identity, bytes):
                    identity = identity.decode()
                identity = str(identity).strip()
                if identity != coverage[space]["basis_identity"]:
                    raise ValueError("Patch core material basis identity changed")
                ranges = np.asarray(
                    h5["property_space/material_ranges"], dtype=np.int64
                )
            material = problem._shared.manifest.block(block.name).binding[1]
            if (
                ranges.ndim != 2
                or ranges.shape[1] != 2
                or not 1 <= material <= len(ranges)
            ):
                raise ValueError("Invalid parent material coefficient ranges")
            offset, count = map(int, ranges[material - 1])
            if count != block.size:
                raise ValueError(
                    "Patch core coefficient count differs from its parent block"
                )
            for key in coverage[space]["control_ids"]:
                slot = int(key) - offset - 1
                if 0 <= slot < count:
                    mask[slot] = True
        elif block.kind == "profile":
            control = block.control
            system = control.coordinate_system or "global"
            for box in boxes:
                if ctx.is_global(system):
                    extent = box[control.axis]
                else:
                    samples = _sample_grid(
                        {axis: np.linspace(lo, hi, 9) for axis, (lo, hi) in box.items()}
                    )
                    mapped = ctx.model._coordinate_system_samples(
                        ctx.coordinate_system(system), samples
                    )
                    values = np.asarray(mapped.coords[control.axis])
                    extent = float(values.min()), float(values.max())
                mask |= _profile_support(control, extent)
        elif block.kind == "grid":
            if not ctx.is_global(block.control.coordinate_system or "global"):
                raise ValueError(
                    "Local patch lattice updates require global coordinates"
                )
            for box in boxes:
                mask |= _lattice_support(block.control, box)
        else:
            raise ValueError("Local patch updates require material controls")
        mask &= problem.space._mask_of(block)
        masks[block.name] = mask
    return masks


def local_patch_view(problem, index, state):
    """Select assigned data and update core while fixing every exterior coefficient."""
    from copy import copy

    view = problem.restrict()
    view._patch_selection = (index,)
    view._shared = copy(problem._shared)
    view._shared.state = state
    view._shared.state_provisional = False
    view._masks = core_masks(problem, problem._patch_runtime.prepared, index)
    view._masks_adopted = True
    return view
