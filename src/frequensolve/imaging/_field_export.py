"""Local mass-lumped field views and cached exact native grid sampling."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.units import ureg

from ._artifacts import ControlVectorFile, unqualified_block_name


class NativeControlField:
    """Local field view using persisted mass and a disk-backed native sampling map.

    Never calls a site's submit/run methods. Only a missing grid map launches a
    local executable; mesh rendering and subsequent grid samples are local array
    operations. Large CSR maps are read and applied in bounded row packets.
    """

    def __init__(self, linearization, vector, *, input_role="dual", local_solver=None):
        self.linearization, self.vector, self.input_role = (
            linearization,
            vector,
            input_role,
        )
        self.local_solver = local_solver

    def _coefficients(self, key):
        from .statistics import mesh_descriptor

        vector = self.vector
        block = vector.space.block(key)
        if block.kind != "mesh":
            raise ValueError(
                "Native field conversion currently requires a mesh material control"
            )
        descriptor = mesh_descriptor(vector.space, block)
        values = np.zeros(block.size)
        values[vector.space._mask_of(block)] = vector.values[
            vector.space.slices[block.name]
        ]
        if self.input_role == "dual":
            from .property_mesh import read_lumped_mass

            cache = self._cache()
            key = ("mass", descriptor["identity"], descriptor["material"])
            if key not in cache:
                cache[key] = read_lumped_mass(
                    descriptor["path"],
                    material=descriptor["material"],
                    identity=descriptor["identity"],
                )
            values /= cache[key]
        return block, descriptor, values

    def _cache(self):
        owner = self.linearization.problem._shared
        if not hasattr(owner, "_field_export_cache"):
            owner._field_export_cache = {}
        return owner._field_export_cache

    def _sampling_map(self, block, descriptor, grid):
        from frequensolve.orchestrator.sites.local.site import LocalSite

        signature = (
            descriptor["identity"],
            descriptor["material"],
            tuple(grid.x0),
            tuple(grid.x1),
            tuple(grid.n),
            tuple(grid.dims),
            grid.units,
            grid.system,
        )
        key = ("map", hashlib.sha256(repr(signature).encode()).hexdigest())
        cache = self._cache()
        if key in cache and cache[key].is_file():
            return cache[key]
        lin = self.linearization
        solver = self.local_solver or os.environ.get("FS_LOCAL_SOLVER")
        site = lin.problem.backend.site
        if solver is None and isinstance(site, LocalSite):
            solver = site.executable or site.solver
        solver = (
            shutil.which(str(solver))
            if solver
            else shutil.which(f"fs{len(grid.dims)}d_s")
        )
        if solver is None:
            raise ValueError(
                "Exact grid mapping needs a local Sauce executable; use lin.field(local_solver=...) "
                "or FS_LOCAL_SOLVER. No HPC job is submitted."
            )
        # Private local scratch avoids writing into a remote project's result layout.
        owner = lin.problem._shared
        if not hasattr(owner, "_field_export_workspace"):
            owner._field_export_workspace = tempfile.TemporaryDirectory(
                prefix="fs-field-map-"
            )
        work = Path(owner._field_export_workspace.name) / key[1]
        work.mkdir(exist_ok=True)
        payload = lin.job.to_fs()
        payload.pop("Outputs", None)
        payload.pop("kernel_derivative", None)
        # Author directly in local scratch, never in an HPC job's result layout.
        input_path = work / "input.h5"
        name = unqualified_block_name(block.name)
        ControlVectorFile(
            {name: np.zeros(block.size)},
            native=True,
            control_spaces={name: block.basis_identity},
        ).write(input_path)
        output = work / "map.h5"
        km = float((1 * ureg(grid.units or "m")).to("km").magnitude)
        payload["control_sensitivities"] = dict(
            active=[name],
            input=str(input_path),
            gradient=str(work / "field.h5"),
            FieldExport=dict(
                control=name,
                input_role="primal",
                sampling_map=True,
                output=str(output),
                grid=dict(
                    origin_km=(np.asarray(grid.x0) * km).tolist(),
                    spacing_km=[
                        (b - a) * km / (n - 1) if n > 1 else 0.0
                        for a, b, n in zip(grid.x0, grid.x1, grid.n)
                    ],
                    counts=list(map(int, grid.n)),
                ),
            ),
        )
        payload["result_path"] = str(work)
        path = work / "request.json"
        path.write_text(json.dumps(payload, default=str))
        with (work / "mapping.log").open("w") as log:
            result = subprocess.run(
                [
                    solver,
                    "-nthreads",
                    "1",
                    "-j",
                    str(path),
                    "--smooth",
                    "--result-dir",
                    str(work),
                    "--tmp-directory",
                    str(work / "tmp"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(
                f"Local field mapping failed; see {work / 'mapping.log'}. "
                "The simulation geometry and property artifacts must be available locally."
            )
        with h5py.File(output) as h5:
            if "offsets" not in h5:
                raise RuntimeError(
                    "Local solver lacks field sampling-map support; rebuild it"
                )
        cache[key] = output
        return output

    def to_grid(self, grid, key, *, frozen=np.nan, context=None):
        if context is not None:
            raise ValueError(
                "Native field sampling uses the simulation geometry, not a Python EvaluationContext"
            )
        from scipy.sparse import csr_matrix

        if not isinstance(grid, CartesianGrid):
            raise ValueError("Field sampling requires a CartesianGrid")
        axes = ["x", "z"] if len(grid.dims) == 2 else ["x", "y", "z"]
        if list(grid.dims) != axes or (grid.system or "global") != "global":
            raise ValueError(
                "Field grids require ordered global Cartesian x/z or x/y/z axes"
            )
        block, descriptor, values = self._coefficients(key)
        path = self._sampling_map(block, descriptor, grid)
        total = int(np.prod(grid.n))
        image = np.full(total, frozen, dtype=float)
        mask = self.vector.space._mask_of(block)
        inactive = None if mask.all() else (~mask).astype(float)
        with h5py.File(path) as h5:
            for first in range(0, total, 32768):
                last = min(total, first + 32768)
                offsets = np.asarray(h5["offsets"][first : last + 1], dtype=np.int64)
                start, end = int(offsets[0]), int(offsets[-1])
                indices = np.asarray(h5["indices"][start:end])
                weights = np.asarray(h5["weights"][start:end])
                operator = csr_matrix(
                    (weights, indices, offsets - start),
                    shape=(last - first, block.size),
                )
                valid = np.asarray(h5["valid"][first:last], dtype=bool)
                if inactive is not None:
                    valid &= np.asarray(abs(operator) @ inactive) == 0
                image[first:last] = np.where(valid, operator @ values, frozen)
        field = xr.DataArray(
            image.reshape(grid.shape),
            dims=grid.dims[::-1],
            coords={
                d: np.linspace(a, b, n)
                for d, a, b, n in zip(grid.dims, grid.x0, grid.x1, grid.n)
            },
            name=block.prop or block.address,
        )
        field.attrs["representation"] = (
            "lumped_mass_gradient"
            if self.input_role == "dual"
            else "primal_control_field"
        )
        for dim in grid.dims:
            field.coords[dim].attrs["units"] = grid.units or "m"
        return field

    def to_mesh(self, mesh=None, key=None, *, material=None, units="m"):
        from .property_mesh import PropertyMesh

        if key is None:
            if len(self.vector.space.blocks) != 1:
                raise ValueError("Select one control for native field conversion")
            key = self.vector.space.blocks[0]
        block, descriptor, values = self._coefficients(key)
        if mesh is None:
            mesh = descriptor["path"]
            material = descriptor["material"] if material is None else material
        if isinstance(mesh, (str, Path)) and Path(mesh).suffix.lower() in {
            ".h5",
            ".hdf5",
        }:
            mesh = PropertyMesh.read(mesh, material=material or descriptor["material"])
        if isinstance(mesh, PropertyMesh):
            if block.basis_identity != mesh.control_identity(block.transform):
                raise ValueError(
                    "PropertyMesh identity does not match the control basis"
                )
            values[~self.vector.space._mask_of(block)] = np.nan
            return mesh.to_mesh(values, name=block.prop or block.address, units=units)
        full = self.vector.space.to_sauce_vector(self.vector)
        full[self.vector.space.full_slices[block.name]] = values
        primal = self.vector.space.from_sauce_vector(full)
        return primal.to_mesh(mesh, key, material=material, units=units)

    def to_property(self, grid, key, *, frozen=np.nan):
        from frequensolve.model.property import Property

        return Property(self.to_grid(grid, key, frozen=frozen))
