"""Read the adapted leaf mesh and constrained basis of a Sauce property space."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["PropertyMesh", "read_reference_vertices"]


def read_lumped_mass(path: str | Path, *, material: int, identity: str) -> np.ndarray:
    """Read exact native support volumes for one immutable constrained basis."""
    with h5py.File(path, "r") as h5:
        group = h5["property_space"]
        if _text(group["identity"]) != identity:
            raise ValueError("Lumped mass belongs to a different property basis")
        if "lumped_mass" not in group or "lumped_mass_units" not in group:
            raise ValueError(
                "Property artifact has no native lumped mass; regenerate it with the current Sauce build"
            )
        if _text(group["lumped_mass_units"]) != "km**D":
            raise ValueError("Unsupported lumped-mass units")
        ranges = group["material_ranges"]
        if not 1 <= material <= len(ranges):
            raise ValueError("Invalid lumped-mass material group")
        offset, count = map(int, ranges[material - 1])
        mass = np.asarray(group["lumped_mass"][offset : offset + count], dtype=float)
        if len(mass) != count or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("Invalid native lumped-mass vector")
    mass.flags.writeable = False
    return mass


# Native element kind -> (VTK linear cell type, vertex count).
_CELLS = {1: (12, 8), 2: (10, 4), 3: (13, 6), 4: (14, 5), 5: (9, 4), 6: (5, 3)}


def _text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode().rstrip(" \0") if isinstance(value, bytes) else str(value)


def _roots(group: h5py.Group, schema: str, dimension: int) -> Iterator[Any]:
    """Expose legacy per-root and packed v2 constraints through one layout."""
    if schema == "fs-property-space-1":
        yield from group["roots"].values()
        return
    meta = np.asarray(group["root_meta"], dtype=int)
    bounds = np.asarray(group["root_offsets"], dtype=int)
    if (
        bounds.shape != (len(meta) + 1, 4)
        or np.any(bounds[0] != 0)
        or np.any(np.diff(bounds, axis=0) < 0)
    ):
        raise ValueError("Invalid packed property root offsets")
    for info, lo, hi in zip(meta, bounds[:-1], bounds[1:]):
        if info[0] == 0:
            continue
        p0, e0, g0, l0 = map(int, lo)
        p1, e1, g1, l1 = map(int, hi)
        if l1 <= l0 or (p1 - p0) % (l1 - l0):
            raise ValueError("Invalid packed property root geometry")
        slots = (p1 - p0) // (l1 - l0)
        yield {
            "meta": info,
            "visualization_points_m": np.asarray(
                group["visualization_points_m"][p0:p1]
            ).reshape(l1 - l0, slots, dimension),
            "visualization_kind": group["visualization_kind"][l0:l1],
            "offset": np.asarray(group["offset"][p0 : p1 + 1]) - e0,
            "index": group["index"][e0:e1],
            "weight": group["weight"][e0:e1],
            "global_ids": group["global_ids"][g0:g1],
        }


def read_reference_vertices(root: h5py.Group) -> np.ndarray:
    """Decode a root's lossless dyadic vertices to (nodes, corner slots, dimension)."""
    dataset = root["vertices"]
    if "encoding" not in dataset.attrs:
        return np.asarray(dataset, dtype=float)
    encoding = np.asarray(dataset.attrs["encoding"]).reshape(-1)[0]
    if isinstance(encoding, bytes):
        encoding = encoding.decode().rstrip(" \0")
    if encoding != "dyadic-reference-i32-v1":
        raise ValueError("Unsupported property vertex encoding")
    words = np.asarray(dataset)
    if (
        words.ndim != 1
        or words.dtype.kind != "i"
        or words.dtype.itemsize != 4
        or len(words) < 5
    ):
        raise ValueError("Invalid property vertex payload")
    version, dimension, corners, nodes, power = map(int, words[:5])
    if (
        version != 1
        or dimension not in (2, 3)
        or corners != 2**dimension
        or nodes < 1
        or not 0 <= power <= 52
    ):
        raise ValueError("Invalid property vertex header")
    count, width = dimension * corners * nodes, power + 1
    if len(words) != 5 + (count * width + 31) // 32:
        raise ValueError("Invalid property vertex payload length")
    # Keep the expanded bit matrix bounded even for deeply refined meshes.
    values = np.empty(count, dtype=float)
    weights = np.uint64(1) << np.arange(width, dtype=np.uint64)
    for first in range(0, count, 8192):
        last = min(first + 8192, count)
        start_bit, end_bit = first * width, last * width
        block = words[5 + start_bit // 32 : 5 + (end_bit + 31) // 32]
        bits = np.unpackbits(block.astype("<i4").view(np.uint8), bitorder="little")
        bits = bits[start_bit % 32 : start_bit % 32 + (last - first) * width]
        decoded = bits.reshape(last - first, width).astype(np.uint64) @ weights
        if np.any(decoded > 2**power):
            raise ValueError("Property reference vertex outside unit cell")
        values[first:last] = np.ldexp(decoded.astype(float), -power)
    return values.reshape(nodes, corners, dimension)


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
        points: list[Any] = []
        weights: list[float] = []
        cells, kinds, rows, columns = [], [], [], []
        with h5py.File(path, "r") as h5:
            group = h5["property_space"]
            schema = _text(group["schema"])
            if schema not in {"fs-property-space-1", "fs-property-space-2"}:
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
            for root in _roots(group, schema, dimension):
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
                slots = xyz.shape[1] if xyz.ndim == 3 else 0
                if (
                    slots not in (2**dimension, 8)
                    or xyz.shape != (len(leaf_kinds), slots, dimension)
                    or len(ptr) != slots * len(leaf_kinds) + 1
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
                        lo, hi = ptr[leaf * slots + vertex : leaf * slots + vertex + 2]
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

    def to_mesh(
        self, coefficients: Any, *, name: str = "control", units: str = "m"
    ) -> Any:
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
