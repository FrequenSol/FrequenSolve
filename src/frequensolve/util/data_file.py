"""Small file-backed array helpers.

The simulation export path now deduplicates bulk data inside HDF5 datasets via
``fs_hash`` attributes. This module remains as a lightweight utility for ad hoc
array files, but intentionally does not maintain directory manifests.
"""

from __future__ import annotations

import json
import mmap
from pathlib import Path
from typing import Any, Optional, Union

import blake3
import h5py
import numpy as np
import xarray as xr

__all__ = [
    "DataArrayFile",
    "check_data_exists",
    "hash_array_blake3",
    "hash_dataarray_blake3",
    "hash_file_blake3",
    "save_data_if_new",
]


ArrayLike = Union[np.ndarray, xr.DataArray]


def _hash_update_json(hasher: blake3.blake3, value: Any) -> None:
    hasher.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def hash_file_blake3(path: Union[str, Path]) -> str:
    """Compute a BLAKE3 digest for a file without creating manifest state."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    hasher = blake3.blake3()
    if path.stat().st_size == 0:
        return hasher.hexdigest()

    with path.open("rb") as handle:
        mm = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            hasher.update(mm)
        finally:
            mm.close()
    return hasher.hexdigest()


def hash_array_blake3(array: np.ndarray) -> str:
    """Hash a NumPy array's dtype, shape, and byte payload."""

    values = np.ascontiguousarray(array)
    hasher = blake3.blake3()
    _hash_update_json(
        hasher,
        {
            "kind": "ndarray",
            "dtype": str(values.dtype),
            "shape": values.shape,
        },
    )
    hasher.update(memoryview(values).cast("B"))
    return hasher.hexdigest()


def hash_dataarray_blake3(data: xr.DataArray) -> str:
    """Hash an xarray DataArray's values, dimensions, coordinates, and attrs."""

    values = np.ascontiguousarray(data.values)
    hasher = blake3.blake3()
    _hash_update_json(
        hasher,
        {
            "kind": "dataarray",
            "dtype": str(values.dtype),
            "shape": values.shape,
            "dims": list(data.dims),
            "attrs": dict(data.attrs),
        },
    )
    hasher.update(memoryview(values).cast("B"))

    for dim in data.dims:
        coord = np.ascontiguousarray(data.coords[dim].values)
        _hash_update_json(
            hasher,
            {
                "coord": dim,
                "dtype": str(coord.dtype),
                "shape": coord.shape,
                "attrs": dict(data.coords[dim].attrs),
            },
        )
        hasher.update(memoryview(coord).cast("B"))
    return hasher.hexdigest()


class DataArrayFile:
    """Save/load NumPy or xarray arrays and skip unchanged target rewrites."""

    def __init__(
        self,
        path_or_data: Union[str, Path, ArrayLike],
        path: Optional[Union[str, Path]] = None,
    ):
        # Backward compatibility: older code used DataArrayFile(data, path).
        if path is None:
            self.path = Path(path_or_data)
            self._data: Optional[ArrayLike] = None
        else:
            self.path = Path(path)
            self._data = path_or_data

    def compute_hash(self, data: ArrayLike) -> str:
        if isinstance(data, np.ndarray):
            return hash_array_blake3(data)
        if isinstance(data, xr.DataArray):
            return hash_dataarray_blake3(data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _data_or_raise(self, data: Optional[ArrayLike]) -> ArrayLike:
        data = self._data if data is None else data
        if data is None:
            raise ValueError("data must be provided")
        return data

    def is_already_saved(self, data: Optional[ArrayLike] = None, **load_kwargs) -> bool:
        """Return True when the target file already stores the same payload."""

        data = self._data_or_raise(data)
        if not self.path.exists():
            return False

        try:
            existing = self.load(**load_kwargs)
        except Exception:
            return False
        return self.compute_hash(existing) == self.compute_hash(data)

    def find_if_exists(
        self, data: Optional[ArrayLike] = None, **load_kwargs
    ) -> Optional[Path]:
        """Compatibility shim: manifests are gone, so only the target path is checked."""

        return self.path if self.is_already_saved(data, **load_kwargs) else None

    find_existing_file = find_if_exists

    def save_if_new(
        self,
        data: Optional[ArrayLike] = None,
        *,
        format: str = "auto",
        load_kwargs: Optional[dict[str, Any]] = None,
        **save_kwargs,
    ) -> Path:
        data = self._data_or_raise(data)
        self._data = data
        if self.is_already_saved(data, **dict(load_kwargs or {})):
            return self.path
        return self.save(data, format=format, **save_kwargs)

    def save(
        self,
        data: Optional[ArrayLike] = None,
        *,
        format: str = "auto",
        **kwargs,
    ) -> Path:
        data = self._data_or_raise(data)
        self._data = data
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if format == "auto":
            format = self._detect_format_from_extension()

        if format == "numpy":
            self._save_numpy(data, **kwargs)
        elif format == "xarray":
            self._save_xarray(data, **kwargs)
        elif format == "netcdf":
            self._save_netcdf(data, **kwargs)
        elif format == "hdf5":
            self._save_hdf5(data, **kwargs)
        elif format == "binary":
            self._save_binary(data, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return self.path

    def _detect_format_from_extension(self) -> str:
        ext = self.path.suffix.lower()
        if ext in {".npy", ".npz"}:
            return "numpy"
        if ext == ".zarr":
            return "xarray"
        if ext in {".nc", ".netcdf"}:
            return "netcdf"
        if ext in {".h5", ".hdf5"}:
            return "hdf5"
        if ext in {".bin", ""}:
            return "binary"
        return "numpy"

    def _save_numpy(self, data: ArrayLike, **kwargs) -> None:
        np.save(
            self.path, data.values if isinstance(data, xr.DataArray) else data, **kwargs
        )

    def _save_xarray(self, data: ArrayLike, **kwargs) -> None:
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data)
        data.to_zarr(self.path, **kwargs)

    def _save_netcdf(self, data: ArrayLike, **kwargs) -> None:
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data)
        data.to_netcdf(self.path, **kwargs)

    def _save_hdf5(self, data: ArrayLike, **kwargs) -> None:
        dataset_name = kwargs.pop("dataset_name", "data")
        with h5py.File(self.path, "w") as h5:
            if isinstance(data, xr.DataArray):
                dset = h5.create_dataset(dataset_name, data=data.values, **kwargs)
                dset.attrs["dims"] = list(data.dims)
                for dim in data.dims:
                    dset.attrs[dim] = data.coords[dim].values
                if data.attrs:
                    dset.attrs["attrs_json"] = json.dumps(data.attrs, default=str)
            else:
                h5.create_dataset(dataset_name, data=data, **kwargs)

    def _save_binary(self, data: ArrayLike, **kwargs) -> None:
        dtype = kwargs.pop("dtype", np.float32)
        values = data.values if isinstance(data, xr.DataArray) else data
        np.asarray(values, dtype=dtype).tofile(self.path)

    def load(self, **kwargs) -> ArrayLike:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        format = self._detect_format_from_extension()
        if format == "numpy":
            return self._load_numpy(**kwargs)
        if format == "xarray":
            return self._load_xarray(**kwargs)
        if format == "netcdf":
            return self._load_netcdf(**kwargs)
        if format == "hdf5":
            return self._load_hdf5(**kwargs)
        if format == "binary":
            return self._load_binary(**kwargs)
        raise ValueError(f"Unsupported format: {format}")

    def _load_numpy(self, **kwargs) -> np.ndarray:
        return np.load(self.path, **kwargs)

    def _load_xarray(self, **kwargs) -> xr.DataArray:
        ds = xr.open_zarr(self.path, **kwargs)
        if isinstance(ds, xr.DataArray):
            return ds
        return ds[list(ds.data_vars)[0]]

    def _load_netcdf(self, **kwargs) -> xr.DataArray:
        ds = xr.open_dataset(self.path, **kwargs)
        return ds[list(ds.data_vars)[0]]

    def _load_hdf5(self, **kwargs) -> ArrayLike:
        dataset_name = kwargs.pop("dataset_name", "data")
        with h5py.File(self.path, "r") as h5:
            dset = h5[dataset_name]
            values = dset[:]
            if "dims" not in dset.attrs:
                return values

            dims = [
                dim.decode() if isinstance(dim, bytes) else str(dim)
                for dim in dset.attrs["dims"]
            ]
            coords = {dim: dset.attrs[dim] for dim in dims if dim in dset.attrs}
            attrs = {}
            if "attrs_json" in dset.attrs:
                attrs = json.loads(dset.attrs["attrs_json"])
            return xr.DataArray(values, dims=dims, coords=coords, attrs=attrs)

    def _load_binary(self, **kwargs) -> np.ndarray:
        shape = kwargs.get("shape")
        dtype = kwargs.get("dtype", np.float32)
        if shape is None:
            raise ValueError("shape must be provided for binary files")
        return np.fromfile(self.path, dtype=dtype).reshape(shape)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataArrayFile):
            return NotImplemented
        return self.path == other.path

    def __repr__(self) -> str:
        return f"<DataArrayFile {self.path!r}>"


def save_data_if_new(
    data: ArrayLike,
    path: Union[str, Path],
    *,
    format: str = "auto",
    load_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Path:
    """Save ``data`` unless ``path`` already contains the same payload."""

    return DataArrayFile(path).save_if_new(
        data,
        format=format,
        load_kwargs=load_kwargs,
        **kwargs,
    )


def check_data_exists(
    data: ArrayLike,
    path: Union[str, Path],
    *,
    load_kwargs: Optional[dict[str, Any]] = None,
) -> bool:
    """Return True when ``path`` already contains ``data``."""

    return DataArrayFile(path).is_already_saved(data, **dict(load_kwargs or {}))
