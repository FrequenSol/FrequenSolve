"""Receiver-block artifacts, physical scaling, and shared-probe behavior."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging.receiver_blocks import block_probe_request
from tests.test_imaging_controls import _layered_simulation, _registry_block


@pytest.fixture
def mesh_space(tmp_path):
    simulation = _layered_simulation(tmp_path)
    bound = im.ControlSpace(
        vp=im.MeshParameters("vp", "sediment", 8, 2, artifact="material.h5"),
        rho=im.MeshParameters("rho", "sediment", 8, 2, artifact="material.h5"),
    ).bind(simulation)
    blocks = []
    for i, name in enumerate(("vp", "rho")):
        blocks.append(
            {
                **_registry_block(
                    i + 1, "model." + name, 2, offset=1 + 2 * i, components=1
                ),
                "basis_identity": "sha256:" + "a" * 64,
            }
        )
    return bound.with_manifest(
        im.ControlRegistryManifest.from_dict(
            {
                "schema": "fs-control-registry-1",
                "fingerprint": "sha256:" + "b" * 64,
                "blocks": blocks,
                "active_blocks": [1, 2],
                "active_offsets": [1, 3],
            }
        )
    )


def test_physical_block_solve_and_correlations(mesh_space):
    matrices = np.array([[[4.0, 3.0], [3.0, 9.0]], [[2.0, -1.0], [-1.0, 3.0]]])
    blocks = im.ReceiverBlocks(("vp", "rho"), matrices, {}, mesh_space)
    gradient = mesh_space.pack({"vp": [1.0, 2.0], "rho": [3.0, 4.0]})
    update = blocks.solve(gradient, scales={"vp": 2.0, "rho": 5.0}, damping=0.1)
    scaling = np.array([2.0, 5.0])
    scaled = matrices * scaling[None, :, None] * scaling[None, None, :]
    ridge = 0.1 * np.diagonal(scaled, axis1=1, axis2=2).max()
    expected = np.stack(
        [
            scaling * np.linalg.solve(m + ridge * np.eye(2), scaling * r)
            for m, r in zip(scaled, ([1.0, 3.0], [2.0, 4.0]))
        ]
    )
    np.testing.assert_allclose(update.values, expected.T.ravel())
    np.testing.assert_allclose(blocks.correlations[:, 0, 1], [0.5, -1 / np.sqrt(6)])
    with pytest.raises(ValueError, match="positive"):
        blocks.solve(gradient, damping=0)
    with pytest.raises(ValueError, match="exactly once"):
        blocks.solve(gradient, scales={"vp": 2})


def test_frequency_reduction_and_basis_validation(mesh_space, tmp_path):
    matrices = np.array([[[4.0, 3.0], [3.0, 9.0]], [[2.0, -1.0], [-1.0, 3.0]]])
    for task, multiplier in ((1, 1.0), (2, 2.0)):
        with h5py.File(tmp_path / f"blocks_{task}.h5", "w") as h:
            h["schema"] = "fs-receiver-blocks-1"
            for i, name in enumerate(("vp", "rho")):
                h[f"control_spaces/{name}"] = mesh_space.block(name).basis_identity
                for j, other in enumerate(("vp", "rho")):
                    h[f"blocks/{name}/{other}"] = multiplier * matrices[:, i, j]
    job = SimpleNamespace(
        receiver_diagonal={"blocks": {"controls": ["vp", "rho"]}},
        receiver_blocks_file=lambda task: tmp_path / f"blocks_{task}.h5",
    )
    lin = SimpleNamespace(
        job=job, space=mesh_space, frequency_weights=[0.25, 0.75], task_factors=[{}, {}]
    )
    result = im.ReceiverBlocks.from_linearization(lin)
    np.testing.assert_allclose(result.matrices, 1.75 * matrices)
    assert not result.matrices.flags.writeable
    with h5py.File(tmp_path / "blocks_2.h5", "r+") as h:
        h["control_spaces/rho"][()] = "wrong"
    with pytest.raises(ValueError, match="basis mismatch"):
        im.ReceiverBlocks.from_linearization(lin)


def test_block_request_resolves_aliases_and_rejects_duplicates(mesh_space):
    assert block_probe_request(
        {"controls": ["vp", "model.rho"], "probes": 4}, mesh_space
    ) == {"probes": 4, "blocks": {"controls": ["vp", "rho"]}}
    with pytest.raises(ValueError, match="distinct"):
        block_probe_request({"controls": ["vp", "model.vp"]}, mesh_space)
