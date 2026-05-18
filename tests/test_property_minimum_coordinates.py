"""Test the _ensure_minimum_coordinates function."""

import numpy as np
import pytest
import xarray as xr

from frequensolve.model.property import _ensure_minimum_coordinates


def test_scalar_dataarray():
    """Test that scalar DataArrays are returned unchanged."""
    da = xr.DataArray(5.0)
    result = _ensure_minimum_coordinates(da)
    assert result == da


def test_single_dimension_single_point():
    """Test expanding a single dimension with one coordinate point."""
    da = xr.DataArray(data=np.array([1.0]), coords={"x": np.array([0.0])}, dims=["x"])
    result = _ensure_minimum_coordinates(da)

    assert len(result.coords["x"]) == 2
    assert result.coords["x"].values[0] == 0.0
    assert result.coords["x"].values[1] == 1.0
    assert result.values[0] == 1.0
    assert result.values[1] == 1.0


def test_multiple_dimensions_single_point():
    """Test expanding multiple dimensions with single coordinate points."""
    da = xr.DataArray(
        data=np.array([[1.0]]),
        coords={"x": np.array([0.0]), "y": np.array([0.0])},
        dims=["x", "y"],
    )
    result = _ensure_minimum_coordinates(da)

    assert len(result.coords["x"]) == 2
    assert len(result.coords["y"]) == 2
    assert result.coords["x"].values[0] == 0.0
    assert result.coords["x"].values[1] == 1.0
    assert result.coords["y"].values[0] == 0.0
    assert result.coords["y"].values[1] == 1.0

    # Check that data is duplicated correctly
    assert result.values[0, 0] == 1.0
    assert result.values[0, 1] == 1.0
    assert result.values[1, 0] == 1.0
    assert result.values[1, 1] == 1.0


def test_mixed_dimensions():
    """Test expanding only dimensions with single points while preserving others."""
    da = xr.DataArray(
        data=np.array([[1.0, 2.0]]),
        coords={"x": np.array([0.0]), "y": np.array([0.0, 1.0])},
        dims=["x", "y"],
    )
    result = _ensure_minimum_coordinates(da)

    assert len(result.coords["x"]) == 2
    assert len(result.coords["y"]) == 2
    assert result.coords["x"].values[0] == 0.0
    assert result.coords["x"].values[1] == 1.0
    assert result.coords["y"].values[0] == 0.0
    assert result.coords["y"].values[1] == 1.0

    # Check that data is duplicated correctly
    assert result.values[0, 0] == 1.0
    assert result.values[0, 1] == 2.0
    assert result.values[1, 0] == 1.0
    assert result.values[1, 1] == 2.0


def test_no_expansion_needed():
    """Test that DataArrays with multiple points in all dimensions are unchanged."""
    da = xr.DataArray(
        data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        coords={"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0])},
        dims=["x", "y"],
    )
    result = _ensure_minimum_coordinates(da)

    # Should be unchanged
    assert result.equals(da)


def test_three_dimensions():
    """Test expanding a three-dimensional array."""
    da = xr.DataArray(
        data=np.array([[[1.0]]]),
        coords={"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])},
        dims=["x", "y", "z"],
    )
    result = _ensure_minimum_coordinates(da)

    assert len(result.coords["x"]) == 2
    assert len(result.coords["y"]) == 2
    assert len(result.coords["z"]) == 2
    assert result.shape == (2, 2, 2)

    # All values should be 1.0
    assert np.all(result.values == 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
