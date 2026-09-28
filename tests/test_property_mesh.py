import h5py
import numpy as np
import pytest

from frequensolve import VtkOutput
from frequensolve import imaging as im


@pytest.fixture
def adapted_artifact(tmp_path):
    # One coarse quad abuts two fine quads. The mid-edge vertex (0.5, 0.5)
    # depends equally on the coarse edge endpoints, not an independent DOF.
    masters = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1], [1, 0], [1, 0.5], [1, 1]])
    leaves = [
        masters[[0, 1, 2, 3]],
        np.array([[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]]),
        np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]]),
    ]
    points = np.zeros((3, 8, 2))
    pointers, indices, weights = [1], [], []
    for leaf, vertices in enumerate(leaves):
        points[leaf, :4] = vertices
        for v in range(8):
            if v < 4:
                hits = np.flatnonzero(np.all(masters == vertices[v], axis=1))
                if len(hits):
                    indices.append(int(hits[0]) + 1)
                    weights.append(1.0)
                else:
                    indices.extend([2, 3])
                    weights.extend([0.5, 0.5])
            pointers.append(len(indices) + 1)
    path = tmp_path / "property.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("property_space")
        g["schema"] = np.bytes_("fs-property-space-1")
        g["basis"] = np.bytes_("continuous-material-h1-linear-v1")
        g["identity"] = np.bytes_("test-adapted-space")
        g["header"] = [2, 12, 4]
        g["material_ranges"] = [[0, 5], [5, 7]]
        r = g.create_group("roots/1")
        r["meta"] = [2, 5, 1, 5]
        r["visualization_kind"] = [5, 5, 5]
        r["visualization_points_m"] = points
        r["offset"] = pointers
        r["index"] = indices
        r["weight"] = weights
        r["global_ids"] = np.arange(6, 13)
    return path, masters


def test_adapted_property_mesh_applies_hanging_constraints(adapted_artifact):
    pytest.importorskip("pyvista")
    path, masters = adapted_artifact
    geometry = im.PropertyMesh.read(path, material=2)
    coefficients = 2 + 3 * masters[:, 0] - 4 * masters[:, 1]
    mesh = geometry.to_mesh(coefficients, name="vp", units="km")
    assert mesh.n_cells == 3 and mesh.n_points == 12
    assert geometry.basis.shape == (12, 7)
    assert np.count_nonzero(np.diff(geometry.basis.indptr) == 2) == 2
    expected = 2 + 3000 * mesh.points[:, 0] - 4000 * mesh.points[:, 2]
    np.testing.assert_allclose(mesh["vp"], expected)
    # VTK roundtrip retains the leaf topology and evaluated values.
    mesh.save(path.with_suffix(".vtu"))
    from frequensolve.plotting.vtu import read_vtu

    restored = read_vtu(path.with_suffix(".vtu"))
    np.testing.assert_array_equal(restored.cells, mesh.cells)
    np.testing.assert_allclose(restored["vp"], expected)
    values = np.arange(7.0)
    values[1] = np.nan
    masked = geometry.to_mesh(values)
    assert np.isnan(masked["control"][[1, 4, 7, 8]]).all()


def test_property_mesh_rejects_missing_geometry_and_invalid_mapping(adapted_artifact):
    path, _ = adapted_artifact
    with h5py.File(path, "a") as h5:
        h5["property_space/roots/1/index"][0] = 99
    with pytest.raises(ValueError, match="constraint map"):
        im.PropertyMesh.read(path, material=2)
    with h5py.File(path, "a") as h5:
        del h5["property_space/roots/1/visualization_points_m"]
    with pytest.raises(ValueError, match="regenerate"):
        im.PropertyMesh.read(path, material=2)


def test_planar_plot_sampling_preserves_bilinear_quads(adapted_artifact):
    pytest.importorskip("pyvista")
    from frequensolve.plotting.vtu import _rasterize_planar_field

    path, masters = adapted_artifact
    geometry = im.PropertyMesh.read(path, material=2)
    # Positive everywhere, with a cross term that triangle interpolation cannot
    # reproduce. Also crosses the coarse/fine interface and its hanging vertex.
    coefficients = 1 + masters[:, 0] + 2 * masters[:, 1] + 3 * np.prod(masters, axis=1)
    mesh = geometry.to_mesh(coefficients)
    raster = _rasterize_planar_field(mesh, "control", resolution=23)
    x, _, z = raster.cell_centers().points.T
    np.testing.assert_allclose(raster["control"], 1 + x + 2 * z + 3 * x * z, atol=1e-12)
    assert np.isfinite(raster["control"]).all()
    assert raster["control"].min() > 0
    with pytest.raises(ValueError, match="resolution"):
        _rasterize_planar_field(mesh, "control", resolution=0)


@pytest.mark.parametrize("distributed", [False, True])
def test_vector_mesh_checks_basis_identity(adapted_artifact, tmp_path, distributed):
    pytest.importorskip("pyvista")
    from tests.imaging_fakes import layered_simulation

    path, _ = adapted_artifact
    geometry = im.PropertyMesh.read(path, material=2)
    space = im.ControlSpace(
        vp=im.MeshParameters("vp", "sediment", frequency=8.0, epw=2.0)
    ).bind(layered_simulation(tmp_path / "project"))
    manifest = im.ControlRegistryManifest.from_dict(
        {
            "schema": "fs-control-registry-1",
            "fingerprint": "sha256:" + "0" * 64,
            "blocks": [
                {
                    "id": 1,
                    "name": "model.vp",
                    "binding": [1, 1, 1],
                    "layout": [1, 4 if distributed else 7, 2, 1],
                    "units": "",
                    "actions": 3,
                    "transform": 0,
                    "scaling": [0.0, 1.0, -1e300, 1e300],
                    "basis_identity": geometry.control_identity("identity"),
                    "distributed": distributed,
                    "global_dofs": 7,
                }
            ],
            "active_blocks": [1],
            "active_offsets": [1],
        }
    )
    vector = space.with_manifest(manifest).ones()
    np.testing.assert_allclose(vector.to_mesh(geometry, "vp")["vp"], 1)
    state = im.ControlState(vector.space, np.ones(7))
    state_file = state.to_file()
    state_file.control_spaces["model.vp"] = "another-basis"
    with pytest.raises(ValueError, match="different control basis"):
        im.ControlState.from_file(state_file, vector.space)
    vector_file = vector.to_file(native=True)
    vector_file.control_spaces["vp"] = "another-basis"
    with pytest.raises(ValueError, match="different control basis"):
        im.ControlVector.from_file(vector_file, vector.space)
    geometry.identity = "a-different-mesh-with-the-same-size"
    with pytest.raises(ValueError, match="identity"):
        vector.to_mesh(geometry, "vp")


def test_property_mesh_output_request_roundtrip():
    output = VtkOutput.property_mesh(
        "velocity", subdomain="sediment", properties=["vp"]
    )
    payload = output.to_fs()
    assert "execute_on" not in payload
    assert payload["target"] == {
        "kind": "property_mesh",
        "space": "velocity",
        "subdomain": "sediment",
    }
    assert VtkOutput.from_fs(payload).to_fs() == payload
    output = VtkOutput.property_mesh("velocity", properties=["vp"], execute_on="final")
    payload = output.to_fs()
    assert payload["execute_on"] == "final"
    restored = VtkOutput.from_fs(payload)
    assert restored.extra["execute_on"] == "final"
    assert restored.to_fs() == payload
    with pytest.raises(ValueError, match="property"):
        VtkOutput.property_mesh("velocity", fields=["pressure"]).to_fs()


def test_mesh_basis_identity_survives_state_and_direction_roundtrip(tmp_path):
    state = im.ControlStateFile(
        {"model.vp": [1.0, 2.0]}, control_spaces={"model.vp": "sha256:mesh-basis"}
    )
    saved = im.ControlStateFile.read(state.write(tmp_path / "state.h5"))
    assert saved.control_spaces == state.control_spaces
    direction = saved.restrict(["model.vp"])
    direction.state_fingerprint = "state"
    direction.control_registry_fingerprint = "registry"
    reread = im.ControlVectorFile.read(direction.write(tmp_path / "direction.h5"))
    assert reread.control_spaces == state.control_spaces
    assert saved.with_update(reread).control_spaces == state.control_spaces
    reread.control_spaces["model.vp"] = "sha256:other-mesh"
    with pytest.raises(ValueError, match="different control basis"):
        saved.with_update(reread)


def test_control_space_identity_keys_are_normalized():
    # Native Fortran HDF5 strings may carry fixed-width trailing padding.
    state = im.ControlStateFile({"vp": [1]}, control_spaces={"vp": "basis \0"})
    assert state.control_spaces == {"model.vp": "basis"}
    with pytest.raises(ValueError, match="unknown block"):
        im.ControlStateFile({"vp": [1]}, control_spaces={"rho": "basis"})


def test_packed_v2_preserves_root_constraints(adapted_artifact, tmp_path):
    path, _ = adapted_artifact
    expected = im.PropertyMesh.read(path, material=2)
    packed = tmp_path / "packed.h5"
    with h5py.File(path) as src, h5py.File(packed, "w") as dst:
        src.copy("property_space", dst)
        g = dst["property_space"]
        root = g["roots/1"]
        # Two copies exercise nonzero offsets in every packed array.
        points = root["visualization_points_m"][:].reshape(-1, 2)
        index, weight, ids = (root[k][:] for k in ("index", "weight", "global_ids"))
        ptr = root["offset"][:]
        n, e, m, l = len(points), len(index), len(ids), 3
        g["root_meta"] = [root["meta"][:], root["meta"][:]]
        g["root_offsets"] = [[0, 0, 0, 0], [n, e, m, l], [2 * n, 2 * e, 2 * m, 2 * l]]
        g["visualization_points_m"] = np.tile(points, (2, 1))
        g["visualization_kind"] = np.tile(root["visualization_kind"][:], 2)
        g["offset"] = np.r_[ptr[:-1], ptr + e]
        g["index"], g["weight"], g["global_ids"] = (
            np.tile(a, 2) for a in (index, weight, ids)
        )
        del g["roots"]
        del g["schema"]
        g["schema"] = np.bytes_("fs-property-space-2")
    actual = im.PropertyMesh.read(packed, material=2)
    np.testing.assert_array_equal(actual.points, np.tile(expected.points, (2, 1)))
    np.testing.assert_allclose(
        actual.basis.toarray(), np.tile(expected.basis.toarray(), (2, 1))
    )
    assert actual.identity == expected.identity


@pytest.mark.parametrize(
    "dimension,power,nodes", [(2, 0, 1), (2, 7, 1100), (3, 52, 350)]
)
def test_reference_vertex_decoder_handles_word_and_block_boundaries(
    tmp_path, dimension, power, nodes
):
    from frequensolve.imaging.property_mesh import read_reference_vertices

    shape = (nodes, 2**dimension, dimension)
    rng = np.random.default_rng(93)
    values = rng.integers(0, 2**power + 1, size=shape, dtype=np.uint64)
    values.flat[0], values.flat[-1] = 0, 2**power
    width = power + 1
    bits = (
        ((values.ravel()[:, None] >> np.arange(width, dtype=np.uint64)) & 1)
        .astype(np.uint8)
        .ravel()
    )
    bits = np.pad(bits, (0, (-len(bits)) % 32))
    packed = np.packbits(bits, bitorder="little").view("<i4")
    path = tmp_path / "vertices.h5"
    with h5py.File(path, "w") as root:
        root["vertices"] = np.r_[
            np.array([1, dimension, 2**dimension, nodes, power], dtype=np.int32), packed
        ]
        root["vertices"].attrs["encoding"] = "dyadic-reference-i32-v1"
        decoded = read_reference_vertices(root)
    np.testing.assert_array_equal(decoded, np.ldexp(values.astype(float), -power))
