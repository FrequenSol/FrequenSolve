# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

import frequensolve as fs
from frequensolve import imaging as im


def test_sharing_normalization_and_transpose_at_edges():
    # Unequal edge apertures make the row-normalized map nonsymmetric.
    p = im.ReceiverPool.from_coordinates([0.0, 1.0, 2.0], 1.01)
    assert p.nnz == 7
    np.testing.assert_allclose(p.apply([2.0, 4.0, 8.0]), [3.0, 14 / 3, 6.0])
    np.testing.assert_allclose(p.scatter([1.0, 0.0, 0.0]), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(p.scatter([0.0, 1.0, 0.0]), [1 / 3] * 3)
    np.testing.assert_allclose(p.apply(np.ones(3)), 1)
    assert p.nnz < p.sample_count * p.output_count


def test_irregular_coordinates_units_quadrature_and_group_isolation():
    xyz = fs.Q_([[0, 0], [0.1, 0], [0.3, 0], [0, 0], [0.1, 0]], "km")
    p = im.ReceiverPool.from_coordinates(
        xyz, fs.Q_(150, "m"), groups=[0, 0, 0, 1, 1], sample_weights=[1, 3, 2, 1, 1]
    )
    np.testing.assert_allclose(p.apply([0, 4, 8, 100, 200]), [3, 3, 8, 150, 150])
    p.validate_groups([0, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="mix groups"):
        p.validate_groups([0, 1, 0, 1, 1])
    h = im.ReceiverPool.from_coordinates([0.0, 1.0, 2.0], 2.0, taper="hann")
    np.testing.assert_allclose(h.apply([0.0, 1.0, 2.0]), [1 / 3, 1.0, 5 / 3])


def test_explicit_rectangular_map_and_immutable_owned_weights():
    weights = csr_matrix([[1.0, 3.0, 0.0], [0.0, 2.0, 2.0]])
    centers = np.array([0, 2])
    p = im.ReceiverPool(weights, centers=centers)
    weights.data[:] = 100
    centers[:] = 1
    np.testing.assert_allclose(p.apply([0.0, 4.0, 8.0]), [3.0, 6.0])
    np.testing.assert_array_equal(p.centers, [0, 2])
    with pytest.raises(ValueError, match="input shape"):
        p.apply(np.zeros(2))
    with pytest.raises(ValueError, match="dual shape"):
        p.scatter(np.zeros(3))


@pytest.mark.parametrize(
    "weights",
    [
        [[0, 0], [1, 1]],
        [[1, -1], [0, 1]],
        [[1, np.nan], [0, 1]],
        [[1, np.inf], [0, 1]],
        [[1j, 0], [0, 1]],
    ],
)
def test_invalid_pool_weights(weights):
    with pytest.raises(ValueError):
        im.ReceiverPool(weights)


@pytest.mark.parametrize(
    "options",
    [
        dict(radius=0),
        dict(radius=np.nan),
        dict(radius=1, groups=[0]),
        dict(radius=1, taper="bad"),
        dict(radius=1, sample_weights=[-1, 1]),
        dict(radius=1, sample_weights=[1j, 1]),
        dict(radius=1, sample_weights=[0, 0]),
    ],
)
def test_invalid_neighborhoods(options):
    with pytest.raises(ValueError):
        im.ReceiverPool.from_coordinates([0.0, 1.0], **options)


def test_quantity_and_center_validation():
    with pytest.raises(ValueError, match="both carry"):
        im.ReceiverPool.from_coordinates([0.0, 1.0], fs.Q_(1, "m"))
    with pytest.raises(ValueError, match="rectangular"):
        im.ReceiverPool([[1, 1]])
    with pytest.raises(ValueError, match="centers"):
        im.ReceiverPool(eye(2), centers=[0.0, 1.0])


def test_canonical_keys_components_terms_shots_and_reordering():
    jets = im.SpectralJetLinearization.__new__(im.SpectralJetLinearization)
    jets.keys = [
        ("p", 2, 2, 1),
        ("p", 1, 1, 2),
        ("p", 1, 2, 1),
        ("p", 1, 1, 1),
        ("v", 1, 1, 1),
        ("p", 2, 1, 1),
    ]
    jets.groups = np.array([0, 1, 2, 2, 3, 0])
    coords = {"p": fs.Q_([0, 0.1], "km"), "v": fs.Q_([0.0], "km")}
    pool = jets.receiver_pool(coords, fs.Q_(200, "m"))
    np.testing.assert_allclose(
        pool.apply([2.0, 100.0, 4.0, 6.0, 1000.0, 8.0]),
        [5.0, 100.0, 5.0, 5.0, 1000.0, 5.0],
    )
    jets.keys = jets.keys[::-1]
    with pytest.raises(ValueError, match="sample keys"):
        jets._validate_pool(pool)
    with pytest.raises(ValueError, match="mapping"):
        jets.receiver_pool([0, 1], 1)
    with pytest.raises(ValueError, match="term ID"):
        jets.receiver_pool({"p": [0, 1]}, 1)
    with pytest.raises(ValueError, match="canonical receiver"):
        jets.receiver_pool({"p": [0], "v": [0]}, 1)


def test_nonphysical_coordinates_and_invalid_partitions_are_rejected():
    with pytest.raises(ValueError, match="real"):
        im.ReceiverPool.from_coordinates([1j, 1], 1)
    with pytest.raises(ValueError, match="groups"):
        im.ReceiverPool.from_coordinates([0, 1], 1, groups=[0, np.nan])
    with pytest.raises(ValueError, match="groups"):
        im.SpectralAttribute("phase_slope").linearize(
            np.ones((2, 2)), np.ones((2, 2)), groups=[0, np.nan]
        )
