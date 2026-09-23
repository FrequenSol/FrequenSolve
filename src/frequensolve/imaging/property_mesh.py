"""Read the adapted leaf mesh and constrained basis of a Sauce property space."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["PropertyMesh"]

# Native element kind -> (VTK linear cell type, vertex count).
_CELLS = {1: (12, 8), 2: (10, 4), 3: (13, 6), 4: (14, 5), 5: (9, 4), 6: (5, 3)}


def _text(dataset):
    value = dataset[()]
    return value.decode().rstrip(" \0") if isinstance(value, bytes) else str(value)


@dataclass
class PropertyMesh:
    """A material's frozen property leaves and vertex-to-coefficient map.

    Points are Cartesian metres, with 2D ``(x, z)`` embedded as ``(x, 0, z)``.
    Vertices are kept separate per cell to preserve topology/material sides.
    ``basis @ coefficients`` applies the solver's hanging-node constraints.
    Geometry is a snapshot at artifact creation: curved cells are represented
    by their linear vertex geometry. Solver property-mesh output uses current
    geometry and physical properties, also represented with linear cells.
    """

    points: np.ndarray
    cells: np.ndarray
    cell_types: np.ndarray
    basis: csr_matrix
    identity: str
    material: int
    dimension: int

    @classmethod
    def read(cls, path: str | Path, *, material: int) -> "PropertyMesh":
        """Read one one-based material group from a property-space HDF5 file.

        Requires an artifact containing ``visualization_kind`` and
        ``visualization_points_m`` (written by current Sauce builds).
        Coefficient order is the material-local order used by control vectors.
        """
        points, cells, kinds, rows, columns, weights = [], [], [], [], [], []
        with h5py.File(path, "r") as h5:
            group = h5["property_space"]
            if _text(group["schema"]) != "fs-property-space-1":
                raise ValueError("Unsupported property-space schema")
            if _text(group["basis"]) != "continuous-material-h1-linear-v1":
                raise ValueError("Unsupported property-space basis")
            dimension = int(np.asarray(group["header"])[0])
            if dimension not in (2, 3):
                raise ValueError("Property mesh must be 2D or 3D")
            ranges = np.asarray(group["material_ranges"], dtype=int)
            if ranges.ndim != 2 or ranges.shape[1] != 2:
                raise ValueError("Invalid property material ranges")
            if material < 1 or material > len(ranges):
                raise ValueError("material must name a one-based material group")
            offset, count = ranges[material - 1]
            identity = _text(group["identity"])
            for root in group["roots"].values():
                if int(root["meta"][0]) != material:
                    continue
                if (
                    "visualization_points_m" not in root
                    or "visualization_kind" not in root
                ):
                    raise ValueError(
                        "This artifact has no leaf visualization geometry; regenerate it "
                        "with a current Sauce build or use solver property-mesh VTU output"
                    )
                xyz = np.asarray(root["visualization_points_m"])
                leaf_kinds = np.asarray(root["visualization_kind"], dtype=int)
                ptr = np.asarray(root["offset"], dtype=int) - 1
                indices = np.asarray(root["index"], dtype=int) - 1
                global_ids = np.asarray(root["global_ids"], dtype=int) - offset - 1
                weight = np.asarray(root["weight"], dtype=float)
                if (
                    xyz.shape != (len(leaf_kinds), 8, dimension)
                    or len(ptr) != 8 * len(leaf_kinds) + 1
                    or ptr[0] != 0
                    or ptr[-1] != len(indices)
                    or len(indices) != len(weight)
                    or np.any(np.diff(ptr) < 0)
                    or np.any(indices < 0)
                    or np.any(indices >= len(global_ids))
                    or np.any(global_ids < 0)
                    or np.any(global_ids >= count)
                    or not np.isfinite(xyz).all()
                    or not np.isfinite(weight).all()
                ):
                    raise ValueError("Invalid property-mesh geometry or constraint map")
                for leaf, kind in enumerate(leaf_kinds):
                    if kind not in _CELLS:
                        raise ValueError(f"Unsupported property cell kind {kind}")
                    vtk_kind, nv = _CELLS[kind]
                    start = len(points)
                    cells.extend([nv, *range(start, start + nv)])
                    kinds.append(vtk_kind)
                    for vertex in range(nv):
                        p = xyz[leaf, vertex]
                        points.append([p[0], 0.0, p[1]] if dimension == 2 else p)
                        lo, hi = ptr[leaf * 8 + vertex : leaf * 8 + vertex + 2]
                        rows.extend([start + vertex] * (hi - lo))
                        columns.extend(global_ids[indices[lo:hi]])
                        weights.extend(weight[lo:hi])
        if not points:
            raise ValueError("The selected material has no property-mesh cells")
        basis = csr_matrix((weights, (rows, columns)), shape=(len(points), count))
        if not np.allclose(np.asarray(basis.sum(axis=1)).ravel(), 1.0, atol=1e-12):
            raise ValueError("Property nodal constraints do not preserve constants")
        return cls(
            np.asarray(points),
            np.asarray(cells),
            np.asarray(kinds, dtype=np.uint8),
            basis,
            identity,
            material,
            dimension,
        )

    def control_identity(self, transform: str) -> str:
        """Return the registry identity of this material's transformed basis."""
        descriptor = {
            "schema": "sauce-parameterized-property-identity-1",
            "control": f"{self.identity}/material/{self.material}",
            "transform": transform,
        }
        payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def to_mesh(self, coefficients: Any, *, name: str = "control", units: str = "m"):
        """Return a PyVista mesh with constrained coefficient values at vertices."""
        from frequensolve.imaging.controls import _pyvista
        from frequensolve.units import ureg

        values = np.asarray(coefficients, dtype=float).reshape(-1)
        if len(values) != self.basis.shape[1]:
            raise ValueError(
                f"Expected {self.basis.shape[1]} coefficients; got {len(values)}"
            )
        factor = float((1.0 * ureg.m).to(units).magnitude)
        mesh = _pyvista().UnstructuredGrid(
            self.cells, self.cell_types, self.points * factor
        )
        mesh.point_data[name] = self.basis @ values
        mesh.field_data["geometry_units"] = [units]
        mesh.field_data["property_space_identity"] = [self.identity]
        mesh.field_data["material"] = [self.material]
        return mesh
