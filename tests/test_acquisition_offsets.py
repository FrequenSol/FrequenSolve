"""Horizontal acquisition distances in the SDK's x/z and x/y/z convention."""

import numpy as np
import pytest

from frequensolve import Acquisition, SourceEncoding
from frequensolve.seismic import ReceiverNode


def _acquisition(sources, receivers, *, encoding=None):
    acquisition = Acquisition(source_encoding=encoding)
    acquisition.add_sources(kind="scalar", coords=sources)
    acquisition.add_receiver_group(
        "receivers", ReceiverNode(name="pressure"), np.asarray(receivers, dtype=float)
    )
    return acquisition


def test_2d_horizontal_offsets_exclude_depth_and_are_unsigned():
    acquisition = _acquisition(
        [[10.0, 100.0]], [[13.0, 104.0], [7.0, 0.0], [10.0, -500.0]]
    )
    np.testing.assert_array_equal(acquisition.offsets(1, "receivers"), [3.0, 3.0, 0.0])


def test_3d_horizontal_offsets_use_xy_and_exclude_depth():
    acquisition = _acquisition(
        [[10.0, 20.0, 100.0]],
        [[13.0, 24.0, 104.0], [7.0, 16.0, 0.0], [10.0, 20.0, -500.0]],
    )
    np.testing.assert_array_equal(acquisition.offsets(1, "receivers"), [5.0, 5.0, 0.0])


def test_offsets_use_encoded_source_field_reference():
    acquisition = _acquisition(
        [[0.0, 100.0], [10.0, 100.0]],
        [[5.0, 0.0], [8.0, 200.0]],
        encoding=SourceEncoding.dense([[1.0, -1.0]], names=["dipole"]),
    )
    np.testing.assert_array_equal(acquisition.source_coords(1), [5.0, 100.0])
    np.testing.assert_array_equal(acquisition.offsets(1, "receivers"), [0.0, 3.0])


@pytest.mark.parametrize("dimension", [2, 3])
def test_offsets_accept_empty_receiver_group(dimension):
    acquisition = _acquisition([np.zeros(dimension)], np.empty((0, dimension)))
    actual = acquisition.offsets(1, "receivers")
    assert actual.shape == (0,)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("source_dimension,receiver_dimension", [(2, 3), (3, 2)])
def test_offsets_reject_mismatched_dimensions(source_dimension, receiver_dimension):
    acquisition = _acquisition(
        [np.zeros(source_dimension)], [np.zeros(receiver_dimension)]
    )
    with pytest.raises(ValueError, match="matching 2D or 3D"):
        acquisition.offsets(1, "receivers")
