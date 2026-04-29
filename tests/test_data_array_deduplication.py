#!/usr/bin/env python3
"""
Tests for data array deduplication functionality.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import the functions to test
from frequensolve.util.data_file import (
    DataArrayFile,
    check_data_exists,
    hash_array_blake3,
    hash_dataarray_blake3,
    save_data_if_new,
)

try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


class TestDataArrayDeduplication:
    """Test class for data array deduplication functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_hash_array_blake3(self):
        """Test that array hashing works correctly."""
        # Create test arrays
        arr1 = np.random.rand(10, 10)
        arr2 = np.random.rand(10, 10)
        arr3 = np.random.rand(10, 10)

        # Same array should have same hash
        hash1 = hash_array_blake3(arr1)
        hash2 = hash_array_blake3(arr1)
        assert hash1 == hash2

        # Different arrays should have different hashes
        hash3 = hash_array_blake3(arr2)
        assert hash1 != hash3
        assert hash2 != hash3

        # Array with same values should have same hash
        arr_copy = arr1.copy()
        hash4 = hash_array_blake3(arr_copy)
        assert hash1 == hash4

    @pytest.mark.skipif(not XARRAY_AVAILABLE, reason="xarray not available")
    def test_hash_dataarray_blake3(self):
        """Test that DataArray hashing works correctly."""
        # Create test DataArrays
        coords = {"x": np.linspace(0, 1, 5), "y": np.linspace(0, 1, 5)}
        da1 = xr.DataArray(np.random.rand(5, 5), dims=["x", "y"], coords=coords)
        da2 = xr.DataArray(np.random.rand(5, 5), dims=["x", "y"], coords=coords)

        # Same DataArray should have same hash
        hash1 = hash_dataarray_blake3(da1)
        hash2 = hash_dataarray_blake3(da1)
        assert hash1 == hash2

        # Different DataArrays should have different hashes
        hash3 = hash_dataarray_blake3(da2)
        assert hash1 != hash3

        # DataArray with same values and coords should have same hash
        da_copy = da1.copy()
        hash4 = hash_dataarray_blake3(da_copy)
        assert hash1 == hash4

    def test_dataarrayfile_numpy(self):
        """Test DataArrayFile with numpy arrays."""
        # Create test data
        data1 = np.random.rand(20, 20)
        data2 = np.random.rand(20, 20)

        # Create DataArrayFile
        file_path = self.test_dir / "test.npy"
        data_file = DataArrayFile(file_path)

        # Test initial state
        assert not data_file.is_already_saved(data1)
        assert not file_path.exists()

        # Test saving
        saved_path = data_file.save_if_new(data1)
        assert saved_path == file_path
        assert file_path.exists()

        # Test that same data returns existing path
        saved_path = data_file.save_if_new(data1)
        assert saved_path == file_path

        # Test that different data is saved
        saved_path = data_file.save_if_new(data2)
        assert saved_path == file_path

        # Test loading
        loaded_data = data_file.load()
        assert np.array_equal(data2, loaded_data)

    @pytest.mark.skipif(not XARRAY_AVAILABLE, reason="xarray not available")
    def test_dataarrayfile_xarray(self):
        """Test DataArrayFile with xarray DataArrays."""
        # Create test data
        coords = {"x": np.linspace(0, 1, 10), "y": np.linspace(0, 1, 10)}
        da1 = xr.DataArray(np.random.rand(10, 10), dims=["x", "y"], coords=coords)
        da2 = xr.DataArray(np.random.rand(10, 10), dims=["x", "y"], coords=coords)

        # Create DataArrayFile
        file_path = self.test_dir / "test.zarr"
        data_file = DataArrayFile(file_path)

        # Test initial state
        assert not data_file.is_already_saved(da1)
        assert not file_path.exists()

        # Test saving
        saved_path = data_file.save_if_new(da1)
        assert saved_path == file_path
        assert file_path.exists()

        # Test that same data returns existing path
        saved_path = data_file.save_if_new(da1)
        assert saved_path == file_path

        # Test that different data is saved
        saved_path = data_file.save_if_new(da2)
        assert saved_path == file_path

        # Test loading
        loaded_data = data_file.load()
        assert da2.equals(loaded_data)

    def test_save_data_if_new(self):
        """Test the convenience function save_data_if_new."""
        data = np.random.rand(15, 15)
        file_path = self.test_dir / "convenience_test.npy"

        # First save should return the path
        saved_path = save_data_if_new(data, file_path)
        assert saved_path == file_path
        assert file_path.exists()

        # Second save should return the same path
        saved_path = save_data_if_new(data, file_path)
        assert saved_path == file_path

    def test_check_data_exists(self):
        """Test the convenience function check_data_exists."""
        data1 = np.random.rand(15, 15)
        data2 = np.random.rand(15, 15)
        file_path = self.test_dir / "exists_test.npy"

        # Initially data doesn't exist
        assert not check_data_exists(data1, file_path)

        # Save the data
        save_data_if_new(data1, file_path)

        # Now data1 exists but data2 doesn't
        assert check_data_exists(data1, file_path)
        assert not check_data_exists(data2, file_path)

    def test_different_formats(self):
        """Test saving in different formats."""
        data = np.random.rand(10, 10)

        formats = [
            ("numpy", ".npy"),
            ("hdf5", ".h5"),
            ("binary", ".bin"),
            ("netcdf", ".nc"),
        ]

        for format_name, extension in formats:
            file_path = self.test_dir / f"format_test{extension}"

            # Test saving
            saved_path = save_data_if_new(data, file_path, format=format_name)
            assert saved_path == file_path
            assert file_path.exists()

            # Test that it returns the same path
            saved_path = save_data_if_new(data, file_path, format=format_name)
            assert saved_path == file_path

    def test_manifest_creation(self):
        """Test that manifest file is created and updated correctly."""
        data = np.random.rand(10, 10)
        file_path = self.test_dir / "manifest_test.npy"

        # Save data
        save_data_if_new(data, file_path)

        # Check that manifest file exists
        manifest_file = self.test_dir / "data_manifest.json"
        assert manifest_file.exists()

        # Check manifest contents
        import json

        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        assert "manifest_test.npy" in manifest
        file_info = manifest["manifest_test.npy"]
        assert "hash" in file_info
        assert "format" in file_info
        assert "shape" in file_info
        assert "dtype" in file_info

    def test_error_handling(self):
        """Test error handling for unsupported data types."""
        data_file = DataArrayFile(self.test_dir / "error_test.npy")

        # Test with unsupported data type
        with pytest.raises(TypeError):
            data_file.compute_hash("not an array")

        # Test with unsupported format
        data = np.random.rand(5, 5)
        with pytest.raises(ValueError):
            data_file.save(data, format="unsupported_format")

    def test_find_existing_file(self):
        """Test finding existing files with the same data."""
        data1 = np.random.rand(10, 10)
        data2 = np.random.rand(10, 10)

        # Save data1 to first file
        file1_path = self.test_dir / "data1.npy"
        data_file1 = DataArrayFile(file1_path)
        saved_path1 = data_file1.save_if_new(data1)
        assert saved_path1 == file1_path

        # Try to save data1 to a different file - should return the original path
        file2_path = self.test_dir / "data1_different.npy"
        data_file2 = DataArrayFile(file2_path)
        saved_path2 = data_file2.save_if_new(data1)
        assert saved_path2 == file1_path  # Should return the original file path

        # Save data2 to a new file
        file3_path = self.test_dir / "data2.npy"
        data_file3 = DataArrayFile(file3_path)
        saved_path3 = data_file3.save_if_new(data2)
        assert saved_path3 == file3_path

        # Test find_existing_file method
        existing_file = data_file1.find_existing_file(data1)
        assert existing_file == file1_path

        existing_file = data_file1.find_existing_file(data2)
        assert existing_file == file3_path

        # Test with non-existent data
        data3 = np.random.rand(10, 10)
        existing_file = data_file1.find_existing_file(data3)
        assert existing_file is None


if __name__ == "__main__":
    pytest.main([__file__])
