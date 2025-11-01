import json
import mmap
from pathlib import Path
from typing import Any, Dict, Optional, Union

import blake3
import h5py
import numpy as np
import xarray as xr

__all__ = ["DataArrayFile", "save_data_if_new", "check_data_exists"]


def hash_file_blake3(path):
    """
    Compute a blake3 hex digest of the file at `path` using a single mmap'd read.
    Requires: pip install blake3
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        h = blake3.blake3()
        h.update(mm)
        mm.close()
    return h.hexdigest()


def hash_array_blake3(array: np.ndarray) -> str:
    """
    Compute a blake3 hex digest of a numpy array.

    Args:
        array: The numpy array to hash

    Returns:
        str: The blake3 hex digest of the array
    """
    h = blake3.blake3()
    h.update(array.tobytes())
    return h.hexdigest()


def hash_dataarray_blake3(da: "xr.DataArray") -> str:
    """
    Compute a blake3 hex digest of an xarray DataArray.

    Args:
        da: The xarray DataArray to hash

    Returns:
        str: The blake3 hex digest of the DataArray
    """
    h = blake3.blake3()
    h.update(da.values.tobytes())
    for dim in da.dims:
        coord_values = da.coords[dim].values
        h.update(coord_values.tobytes())
    return h.hexdigest()


class File:
    def __init__(self, path):
        self.path = Path(path)
        self._hash = None
        self._mtime = None

        manifest = self.manifest
        if self.path.name in manifest:
            self._hash = manifest[self.path.name]
            self._mtime = manifest[self.path.name]["mtime"]

    def _compute_hash(self):
        with self.path.open("rb") as f:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            h = blake3.blake3()
            h.update(mm)
            mm.close()
        return h.hexdigest()

    @property
    def manifest(self):
        parent = self.path.parent
        file = parent / "manifest.json"
        manifest = {}
        if file.exists():
            with open(file, "r") as f:
                manifest = json.load(f)
        return manifest

    @property
    def hash(self):
        """Get a file's hash if not already computed"""
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        mtime = self.path.stat().st_mtime

        if self._hash is None or mtime != self._mtime:
            self._hash = self._compute_hash()
            self._mtime = mtime

            # Update manifest
            manifest = self.path.parent / "manifest.json"
            manifest_data = {}
            if manifest.exists():
                with open(manifest, "r") as f:
                    manifest_data = json.load(f)
            manifest_data[self.path.name] = {"hash": self._hash, "mtime": mtime}
            with open(manifest, "w") as f:
                json.dump(manifest_data, f, indent=3)
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, File):
            return NotImplemented
        return self.hash == other.hash

    def __repr__(self):
        # show just a prefix of the blake3 digest
        return f"<File {self.path!r} blake3={self.hash[:8]}…>"


# TODO: If a file's mtime has been updated, open file and recompute hash?
class DataArrayFile:
    """
    A class for managing data array files with deduplication capabilities.

    This class extends the File class functionality to work with data arrays
    (numpy arrays and xarray DataArrays) and provides methods to check if
    data has already been saved to avoid redundant saves.
    """

    def __init__(
        self, data: Optional[Union[np.ndarray, "xr.DataArray"]], path: Union[str, Path]
    ):
        """
        Initialize a DataArrayFile.

        Args:
            path: The file path where the data array will be saved
            data: Optional data array to associate with this file
        """
        self.path = Path(path)
        self._data = data
        self._data_hash = self.compute_hash(data)
        self._hash = None
        self._manifest = None

    @property
    def manifest(self) -> Dict[str, Any]:
        """Get the manifest for this file's directory."""
        if self._manifest is None:
            manifest_file = self.path.parent / "data_manifest.json"
            self._manifest = {}
            if manifest_file.exists():
                with open(manifest_file, "r") as f:
                    self._manifest = json.load(f)
        return self._manifest

    def _update_manifest(self, data_hash: str, metadata: Dict[str, Any] = None):
        """Update the manifest with file information."""
        manifest_file = self.path.parent / "data_manifest.json"
        manifest_data = self.manifest.copy()

        file_info = {
            "hash": data_hash,
            "mtime": self.path.stat().st_mtime if self.path.exists() else None,
            "size": self.path.stat().st_size if self.path.exists() else None,
        }

        if metadata:
            file_info.update(metadata)

        manifest_data[self.path.name] = file_info

        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f, indent=3)
        self._manifest = manifest_data

    def compute_hash(self, data: Union[np.ndarray, "xr.DataArray"]) -> str:
        """
        Compute the hash of a data array.

        Args:
            data: The data array to hash

        Returns:
            str: The blake3 hex digest
        """
        if isinstance(data, np.ndarray):
            return hash_array_blake3(data)
        elif isinstance(data, xr.DataArray):
            return hash_dataarray_blake3(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def is_already_saved(self) -> bool:
        """
        Check if the data array has already been saved to this file.

        Args:
            data: The data array to check

        Returns:
            bool: True if the data is already saved, False otherwise
        """
        if not self.path.exists():
            return False

        data_hash = self._data_hash
        manifest = self.manifest

        if self.path.name in manifest:
            stored_hash = manifest[self.path.name].get("hash")
            return stored_hash == data_hash

        return False

    def find_if_exists(self) -> Optional[Path]:
        """
        Find an existing file in the manifest that contains the same data.

        Args:
            data: The data array to search for

        Returns:
            Optional[Path]: Path to existing file with same data, or None if not found
        """
        data_hash = self._data_hash
        manifest = self.manifest

        for filename, info in manifest.items():
            stored_hash = info.get("hash")
            if stored_hash == data_hash:
                return self.path.parent / filename

        return None

    def save_if_new(self, format: str = "auto", **kwargs) -> Path:
        """
        Save the data array only if it hasn't been saved before.

        Args:
            data: The data array to save
            format: The format to save in ("auto", "numpy", "xarray", "netcdf", "hdf5", "binary")
            **kwargs: Additional arguments passed to the save function

        Returns:
            Path: The path where the data is saved (either existing or newly created)
        """
        if self.is_already_saved():
            return self.path

        # Check if any other file in the manifest contains the same data
        existing_file = self.find_if_exists()
        if existing_file is not None:
            return existing_file

        # Data doesn't exist anywhere, save it to the current path
        self.save(format=format, **kwargs)
        return self.path

    def save(self, format: str = "auto", **kwargs) -> Path:
        """
        Save a data array to file.

        Args:
            data: The data array to save
            format: The format to save in ("auto", "numpy", "xarray", "hdf5", "binary")
            **kwargs: Additional arguments passed to the save function

        Returns:
            Path: The path where the file was saved
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if format == "auto":
            format = self._detect_format_from_extension()
        data_hash = self._data_hash

        # Save based on format
        if format == "numpy":
            self._save_numpy(self._data, **kwargs)
        elif format == "xarray":
            self._save_xarray(self._data, **kwargs)
        elif format == "netcdf":
            self._save_netcdf(self._data, **kwargs)
        elif format == "hdf5":
            self._save_hdf5(self._data, **kwargs)
        elif format == "binary":
            self._save_binary(self._data, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Update manifest
        metadata = {
            "format": format,
            "dtype": str(np.single),
            "shape": self._data.shape if hasattr(self._data, "shape") else None,
        }
        if isinstance(self._data, xr.DataArray):
            metadata.update(
                {
                    "dims": list(self._data.dims),
                    "coords": {
                        dim: self._data.coords[dim].shape for dim in self._data.dims
                    },
                }
            )

        self._update_manifest(data_hash, metadata)

        return self.path

    def _detect_format_from_extension(self) -> str:
        """Detect the format based on file extension."""
        ext = self.path.suffix.lower()
        if ext in [".npy", ".npz"]:
            return "numpy"
        elif ext in [".zarr"]:
            return "xarray"
        elif ext in [".nc", ".netcdf"]:
            return "netcdf"
        elif ext in [".h5", ".hdf5"]:
            return "hdf5"
        elif ext in [".bin", ""]:
            return "binary"
        else:
            return "numpy"  # default

    def _save_numpy(self, data: Union[np.ndarray, "xr.DataArray"], **kwargs):
        """Save data using numpy."""
        if isinstance(data, xr.DataArray):
            np.save(self.path, data.values, **kwargs)
        else:
            np.save(self.path, data, **kwargs)

    def _save_xarray(self, data: xr.DataArray, **kwargs):
        """Save data using xarray."""
        data.to_zarr(self.path, **kwargs)

    def _save_netcdf(self, data: xr.DataArray, **kwargs):
        """Save data using xarray to netCDF format."""
        if isinstance(data, xr.DataArray):
            data.to_netcdf(self.path, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _save_hdf5(self, data: Union[np.ndarray, "xr.DataArray"], **kwargs):
        """Save data using h5py."""
        dataset_name = kwargs.pop("dataset_name", "data")

        with h5py.File(self.path, "w") as f:
            if isinstance(data, xr.DataArray):
                dset = f.create_dataset(dataset_name, data=data.values, **kwargs)
                dset.attrs["dims"] = list(data.dims)
                for dim in data.dims:
                    dset.attrs[dim] = data.coords[dim].values.tolist()
            else:
                f.create_dataset(dataset_name, data=data, **kwargs)

    def _save_binary(self, data: Union[np.ndarray, "xr.DataArray"], **kwargs):
        """Save data as binary file."""
        if isinstance(data, xr.DataArray):
            data.values.astype(np.single).tofile(self.path)
        else:
            data.astype(np.single).tofile(self.path)

    def load(self, **kwargs) -> Union[np.ndarray, "xr.DataArray"]:
        """
        Load data from file.

        Args:
            **kwargs: Additional arguments passed to the load function

        Returns:
            The loaded data array
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        format = self._detect_format_from_extension()

        if format == "numpy":
            return self._load_numpy(**kwargs)
        elif format == "xarray":
            return self._load_xarray(**kwargs)
        elif format == "netcdf":
            return self._load_netcdf(**kwargs)
        elif format == "hdf5":
            return self._load_hdf5(**kwargs)
        elif format == "binary":
            return self._load_binary(**kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _load_numpy(self, **kwargs) -> np.ndarray:
        """Load data using numpy."""
        return np.load(self.path, **kwargs)

    def _load_xarray(self, **kwargs) -> "xr.DataArray":
        """Load data using xarray."""
        return xr.open_zarr(self.path, **kwargs)

    def _load_netcdf(self, **kwargs) -> Union[np.ndarray, "xr.DataArray"]:
        """Load data using xarray from netCDF format."""
        ds = xr.open_dataset(self.path, **kwargs)
        return ds[list(ds.data_vars)[0]]

    def _load_hdf5(self, **kwargs) -> Union[np.ndarray, "xr.DataArray"]:
        """Load data using h5py."""
        dataset_name = kwargs.pop("dataset_name", "data")

        with h5py.File(self.path, "r") as f:
            data = f[dataset_name][:]

            # Check if we have coordinate information
            if "dims" in f[dataset_name].attrs:
                dims = f[dataset_name].attrs["dims"]
                coords = {}
                for dim in dims:
                    if dim in f[dataset_name].attrs:
                        coords[dim] = f[dataset_name].attrs[dim]
                return xr.DataArray(data, dims=dims, coords=coords)

            return data

    def _load_binary(self, **kwargs) -> np.ndarray:
        """Load data from binary file."""
        shape = kwargs.get("shape")
        dtype = kwargs.get("dtype", np.float32)

        if shape is None:
            raise ValueError("shape must be provided for binary files")

        return np.fromfile(self.path, dtype=dtype).reshape(shape)

    def __eq__(self, other):
        if not isinstance(other, DataArrayFile):
            return NotImplemented
        return self.path == other.path

    def __repr__(self):
        return f"<DataArrayFile {self.path!r}>"


def save_data_if_new(
    data: Union[np.ndarray, "xr.DataArray"],
    path: Union[str, Path],
    format: str = "auto",
    **kwargs,
) -> Path:
    """
    Convenience function to save a data array only if it hasn't been saved before.

    This is a simple wrapper around DataArrayFile.save_if_new() for quick use.
    If the same data already exists in any file in the manifest, returns that file's path.

    Args:
        data: The data array to save
        path: The file path where to save the data
        format: The format to save in ("auto", "numpy", "xarray", "netcdf", "hdf5", "binary")
        **kwargs: Additional arguments passed to the save function

    Returns:
        Path: The path where the data is saved (either existing or newly created)

    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 100)
        >>> saved_path = save_data_if_new(data, "my_data.npy")
        >>> print(f"Data saved to: {saved_path}")
    """
    data_file = DataArrayFile(data, path)
    return data_file.save_if_new(format=format, **kwargs)


def check_data_exists(
    data: Union[np.ndarray, "xr.DataArray"], path: Union[str, Path]
) -> bool:
    """
    Check if a data array has already been saved to a file.

    Args:
        data: The data array to check
        path: The file path to check

    Returns:
        bool: True if the data is already saved, False otherwise

    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 100)
        >>> exists = check_data_array_exists(data, "my_data.npy")
        >>> print(f"Data already exists: {exists}")
    """
    data_file = DataArrayFile(path)
    return data_file.is_already_saved(data)
