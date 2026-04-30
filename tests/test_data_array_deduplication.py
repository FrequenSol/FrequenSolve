import os

import numpy as np
import xarray as xr

from frequensolve.util.data_file import (
    DataArrayFile,
    check_data_exists,
    hash_array_blake3,
    hash_dataarray_blake3,
    save_data_if_new,
)


def test_hash_array_blake3_includes_payload_shape_and_dtype():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert hash_array_blake3(values) == hash_array_blake3(values.copy())
    assert hash_array_blake3(values) != hash_array_blake3(values.astype(np.float64))
    assert hash_array_blake3(values) != hash_array_blake3(values.reshape(4, 3))


def test_hash_dataarray_blake3_includes_coordinates_and_attrs():
    data = xr.DataArray(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        dims=("x", "z"),
        coords={"x": [0.0, 0.5, 1.0], "z": [0.0, 1.0]},
        attrs={"units": "km/s"},
    )

    assert hash_dataarray_blake3(data) == hash_dataarray_blake3(data.copy())

    changed_coords = data.assign_coords(x=[0.0, 0.25, 1.0])
    changed_attrs = data.copy()
    changed_attrs.attrs["units"] = "m/s"

    assert hash_dataarray_blake3(data) != hash_dataarray_blake3(changed_coords)
    assert hash_dataarray_blake3(data) != hash_dataarray_blake3(changed_attrs)


def test_save_data_if_new_skips_unchanged_target_without_manifest(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "model.npy"

    assert save_data_if_new(data, path) == path
    first_mtime = path.stat().st_mtime_ns

    assert save_data_if_new(data.copy(), path) == path
    assert path.stat().st_mtime_ns == first_mtime
    assert not (tmp_path / "data_manifest.json").exists()

    changed = data + 1
    assert save_data_if_new(changed, path) == path
    assert np.array_equal(np.load(path), changed)


def test_data_array_file_roundtrips_xarray_hdf5_without_manifest(tmp_path):
    data = xr.DataArray(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        dims=("x", "z"),
        coords={"x": [0.0, 0.5, 1.0], "z": [0.0, 1.0]},
        attrs={"units": "km/s"},
    )
    file = DataArrayFile(tmp_path / "model.h5")

    file.save_if_new(data, format="hdf5")
    loaded = file.load()

    assert isinstance(loaded, xr.DataArray)
    assert loaded.equals(data)
    assert loaded.attrs == data.attrs
    assert check_data_exists(data, file.path)
    assert not (tmp_path / "data_manifest.json").exists()


def test_data_array_file_supports_legacy_constructor_order(tmp_path):
    data = np.arange(5, dtype=np.float32)
    file = DataArrayFile(data, tmp_path / "legacy.npy")

    assert file.save_if_new() == tmp_path / "legacy.npy"
    assert check_data_exists(data, file.path)


def test_binary_dedup_uses_explicit_load_shape(tmp_path):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "model.bin"

    save_data_if_new(
        data,
        path,
        format="binary",
        load_kwargs={"shape": data.shape, "dtype": np.float32},
    )
    first_mtime = os.stat(path).st_mtime_ns

    save_data_if_new(
        data.copy(),
        path,
        format="binary",
        load_kwargs={"shape": data.shape, "dtype": np.float32},
    )

    assert os.stat(path).st_mtime_ns == first_mtime
