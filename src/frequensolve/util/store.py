"""HDF5 backing store for arrays materialized during solver export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import blake3
import h5py
import numpy as np
import xarray as xr

__all__ = ["HDF5Reference", "SimulationStore", "hash_dataarray_payload"]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _h5_attr_value(value: Any) -> Any:
    """Convert Python/NumPy values to HDF5 attribute-safe values."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return np.asarray(value, dtype=string_dtype)
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
        flat = value.ravel()
        if all(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in flat):
            return value.astype(string_dtype)
    return value


def _hash_update_json(hasher, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
    hasher.update(payload)


def hash_dataarray_payload(
    data: xr.DataArray,
    attrs: Optional[Mapping[str, Any]] = None,
    dtype: Optional[Any] = None,
) -> str:
    """Hash data and metadata that affect solver-facing array meaning.

    Args:
        data: Data array to hash, including dimensions and coordinates.
        attrs: Additional attributes that will be written with the stored
            dataset.
        dtype: Optional dtype used for the hashed data payload.

    Returns:
        Hex digest of the deterministic BLAKE3 hash.
    """
    values = np.ascontiguousarray(data.values)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    hasher = blake3.blake3()
    _hash_update_json(hasher, {"dtype": str(values.dtype), "shape": values.shape})
    hasher.update(memoryview(values).cast("B"))
    _hash_update_json(hasher, {"dims": list(data.dims)})
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
    _hash_update_json(
        hasher, {"attrs": dict(data.attrs), "extra_attrs": dict(attrs or {})}
    )
    return hasher.hexdigest()


@dataclass(frozen=True)
class HDF5Reference:
    """Reference to an array stored in a simulation HDF5 input store.

    Args:
        file: HDF5 file path.
        dataset: Dataset path inside ``file``.
        hash: BLAKE3 content hash without the ``blake3:`` prefix.
        project_path: Optional project root used to emit project-relative
            locators.
    """

    file: Path
    dataset: str
    hash: str
    project_path: Optional[Path] = None

    @property
    def clean_dataset(self) -> str:
        """Return the dataset path without leading or trailing slashes."""

        return self.dataset.strip("/")

    def locator(self) -> str:
        """Return a ``file:dataset`` locator for this HDF5 reference.

        Returns:
            Project-relative locator when possible, otherwise an absolute file
            locator.
        """

        file = self.file
        if self.project_path is not None:
            try:
                file = file.resolve().relative_to(self.project_path.resolve())
            except Exception:
                pass
        return f"{file}:{self.clean_dataset}"

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this HDF5 reference for solver input.

        Returns:
            Property/file payload containing locator, format, dataset, and
            content hash.
        """

        return {
            "file": self.locator(),
            "format": "hdf5",
            "dataset": self.clean_dataset,
            "hash": f"blake3:{self.hash}",
        }


class SimulationStore:
    """Single HDF5 store for local simulation input arrays.

    Args:
        path: HDF5 file path.
        project_path: Optional project root used when serializing references.
    """

    def __init__(self, path: Path, project_path: Optional[Path] = None):
        """Create an HDF5 simulation store.

        Args:
            path: HDF5 file path.
            project_path: Optional project root used for relative references.
        """

        self.path = Path(path)
        self.project_path = (
            Path(project_path).resolve() if project_path is not None else None
        )

    def put_dataarray(
        self,
        dataset: str,
        data: xr.DataArray,
        *,
        attrs: Optional[Mapping[str, Any]] = None,
        compression: Optional[str] = None,
        dtype: Optional[Any] = np.float32,
    ) -> HDF5Reference:
        """Write a data array to the store and return its reference.

        Args:
            dataset: Dataset path inside the HDF5 file.
            data: Data array to write.
            attrs: Additional HDF5 attributes to write on the dataset.
            compression: Optional HDF5 compression filter.
            dtype: Optional dtype used for stored values. ``None`` preserves
                the input dtype.

        Returns:
            ``HDF5Reference`` for the stored dataset. Existing datasets are
            reused when their stored hash matches.
        """

        dataset = dataset.strip("/")
        attrs = dict(attrs or {})
        hash_dtype = dtype
        if dtype is not None and np.dtype(dtype) == np.dtype(np.float32):
            hash_dtype = None
        digest = hash_dataarray_payload(data, attrs=attrs, dtype=hash_dtype)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        values = np.ascontiguousarray(data.values)
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        with h5py.File(self.path, "a") as h5:
            if dataset in h5:
                dset = h5[dataset]
                if dset.attrs.get("fs_hash") == f"blake3:{digest}":
                    return HDF5Reference(self.path, dataset, digest, self.project_path)
                del h5[dataset]

            dset = h5.create_dataset(dataset, data=values, compression=compression)
            dset.attrs["fs_hash"] = f"blake3:{digest}"
            dset.attrs["fs_hash_algorithm"] = "blake3"
            dset.attrs["dims"] = _h5_attr_value(list(data.dims))
            for dim in data.dims:
                dset.attrs[dim] = _h5_attr_value(np.asarray(data.coords[dim].values))

            axis_units = []
            for dim in data.dims:
                axis_units.append(data.coords[dim].attrs.get("units", ""))
            if any(axis_units):
                dset.attrs["axis_units"] = _h5_attr_value(axis_units)

            for key, value in attrs.items():
                if value is None:
                    continue
                dset.attrs[key] = _h5_attr_value(_json_default(value))

        return HDF5Reference(self.path, dataset, digest, self.project_path)

    def put_array_chunks(
        self,
        dataset: str,
        shape: Sequence[int],
        chunk_iter: Iterable[np.ndarray],
        *,
        attrs: Optional[Mapping[str, Any]] = None,
        dims: Optional[Sequence[str]] = None,
        coords: Optional[Mapping[str, Any]] = None,
        compression: Optional[str] = None,
        dtype: Optional[Any] = np.float64,
    ) -> HDF5Reference:
        """Write an array from row chunks and return its store reference.

        This is intended for generated arrays that are too large to first
        materialize as a single NumPy or xarray object.
        """

        dataset = dataset.strip("/")
        attrs = dict(attrs or {})
        dims = list(dims or [])
        coords = dict(coords or {})
        shape = tuple(int(value) for value in shape)
        dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        hasher = blake3.blake3()
        _hash_update_json(hasher, {"dtype": str(dtype), "shape": shape})

        with h5py.File(self.path, "a") as h5:
            if dataset in h5:
                del h5[dataset]

            dset = h5.create_dataset(
                dataset,
                shape=shape,
                dtype=dtype,
                compression=compression,
            )
            offset = 0
            for chunk in chunk_iter:
                values = np.ascontiguousarray(chunk, dtype=dtype)
                if values.ndim != len(shape) or values.shape[1:] != shape[1:]:
                    raise ValueError(
                        f"Chunk shape {values.shape} is incompatible with {shape}"
                    )
                stop = offset + values.shape[0]
                if stop > shape[0]:
                    raise ValueError("Chunk iterator produced too many rows")
                dset[offset:stop, ...] = values
                hasher.update(memoryview(values).cast("B"))
                offset = stop
            if offset != shape[0]:
                raise ValueError(
                    f"Chunk iterator produced {offset} rows, expected {shape[0]}"
                )

            _hash_update_json(hasher, {"dims": dims})
            for dim in dims:
                if dim not in coords:
                    continue
                coord = np.ascontiguousarray(coords[dim])
                _hash_update_json(
                    hasher,
                    {
                        "coord": dim,
                        "dtype": str(coord.dtype),
                        "shape": coord.shape,
                        "attrs": {},
                    },
                )
                hasher.update(memoryview(coord).cast("B"))
            _hash_update_json(hasher, {"attrs": {}, "extra_attrs": attrs})
            digest = hasher.hexdigest()

            dset.attrs["fs_hash"] = f"blake3:{digest}"
            dset.attrs["fs_hash_algorithm"] = "blake3"
            if dims:
                dset.attrs["dims"] = _h5_attr_value(dims)
            for dim, coord in coords.items():
                dset.attrs[dim] = _h5_attr_value(np.asarray(coord))
            for key, value in attrs.items():
                if value is None:
                    continue
                dset.attrs[key] = _h5_attr_value(_json_default(value))

        return HDF5Reference(self.path, dataset, digest, self.project_path)
