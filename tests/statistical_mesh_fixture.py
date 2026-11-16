"""Small complete native property meshes for statistical API tests."""

from pathlib import Path

import h5py
import numpy as np


def write_mesh(path: Path, *, refined=False, curved=False):
    root = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    if refined:
        nodes = np.array(
            [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        )
        vertices = np.stack([root, nodes[[0, 1, 2, 3]], nodes[[1, 4, 5, 2]]])
        indices = np.array([1, 2, 3, 4, 2, 5, 6, 3])
    else:
        nodes = root
        vertices = root[None, :, :]
        indices = np.arange(1, 5)
    identity = "fixture-refined" if refined else "fixture-coarse"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("property_space")
        g["schema"] = np.bytes_("fs-property-space-1")
        g["basis"] = np.bytes_("continuous-material-h1-linear-v1")
        g["identity"] = np.bytes_(identity)
        g["initial_identity"] = np.bytes_("fixture-geometry")
        g["header"] = np.array([2, len(nodes), 8 if refined else 4], dtype=np.int32)
        g["material_ranges"] = np.array([[0, len(nodes)]], dtype=np.int32)
        g["metres_per_native_unit"] = np.array([1.0])
        r = g.create_group("roots/1")
        r["meta"] = np.array([1, 5, 1, 5], dtype=np.int32)
        r["physical_vertices"] = root
        r["linear_geometry"] = np.array([not curved], dtype=np.int32)
        r["codec"] = np.bytes_("preorder-bitpack-i32-v1")
        r["n_bits"] = np.array([6 if refined else 2], dtype=np.int64)
        r["words"] = np.array([1 if refined else 0], dtype=np.int32)
        r["vertices"] = vertices
        r["offset"] = np.arange(1, len(indices) + 2, dtype=np.int32)
        r["index"] = indices.astype(np.int32)
        r["weight"] = np.ones(len(indices))
        r["global_ids"] = np.arange(1, len(nodes) + 1, dtype=np.int64)
        r["canonical_roots"] = np.ones(len(nodes), dtype=np.int32)
        leaves = vertices[1:] if refined else vertices
        r["visualization_points_m"] = leaves
        r["visualization_kind"] = np.full(len(leaves), 5, dtype=np.int32)
    return (
        dict(
            path=str(path),
            identity=identity,
            roots=np.array([1], dtype=np.int32),
            material=1,
            size=len(nodes),
            metres_per_native_unit=1.0,
        ),
        nodes,
    )
