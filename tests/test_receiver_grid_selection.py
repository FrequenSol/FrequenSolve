"""Flat receiver selection agrees with the public grid coordinate table."""

import numpy as np
import pytest

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.seismic.receivers import CoordsGrid


@pytest.fixture(params=[[2, 3], [2, 3, 4]])
def coordinates(request):
    n = request.param
    return CoordsGrid(
        grid=CartesianGrid(
            n=n,
            x0=[10.0 * i for i in range(len(n))],
            dx=[float(i + 1) for i in range(len(n))],
        )
    )


def test_every_scalar_flat_index_matches_coordinate_table(coordinates):
    expected = coordinates.get()
    for i in range(len(expected)):
        np.testing.assert_array_equal(coordinates.get(i), expected[i : i + 1])
        np.testing.assert_array_equal(
            coordinates.get(i - len(expected)), expected[i : i + 1]
        )


def test_flat_index_list_preserves_order_duplicates_and_negative_indices(coordinates):
    expected = coordinates.get()
    selected = [1, -1, 0, 1, -len(expected)]
    np.testing.assert_array_equal(coordinates.get(selected), expected[selected])


def test_empty_flat_index_list_preserves_dimension(coordinates):
    actual = coordinates.get([])
    assert actual.shape == (0, len(coordinates.grid.n))
    assert actual.dtype == np.float64


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("boundary", ["past-end", "before-start"])
def test_out_of_range_flat_indices_are_rejected(coordinates, as_list, boundary):
    size = coordinates.size
    index = int(size) if boundary == "past-end" else -int(size) - 1
    with pytest.raises(IndexError, match="receiver index.*out of range"):
        coordinates.get([index] if as_list else index)


def test_tensor_slices_keep_grid_coordinate_semantics(coordinates):
    slices = [slice(1, 2)] + [slice(None)] * (len(coordinates.grid.n) - 1)
    np.testing.assert_array_equal(
        coordinates.get(slices), coordinates.grid.get_coords(slices)
    )


def test_numpy_integer_flat_indices_are_accepted(coordinates):
    expected = coordinates.get()
    np.testing.assert_array_equal(coordinates.get(np.int64(1)), expected[1:2])
    np.testing.assert_array_equal(coordinates.get([np.int64(1)]), expected[[1]])


@pytest.mark.parametrize("selection", [[1, 2.5], [slice(None), 1], [1, slice(None)]])
def test_mixed_flat_and_tensor_selection_is_rejected(coordinates, selection):
    with pytest.raises(ValueError, match="integer indices or tensor slices"):
        coordinates.get(selection)


def test_flat_slice_retains_public_table_order(coordinates):
    np.testing.assert_array_equal(
        coordinates.get(slice(None, None, -2)), coordinates.get()[::-2]
    )
