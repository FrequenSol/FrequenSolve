from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
    data: xr.DataArray, attrs: Optional[Mapping[str, Any]] = None
) -> str:
    """Hash data and metadata that affect the solver-facing meaning of an array."""
    values = np.ascontiguousarray(data.values)
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
    file: Path
    dataset: str
    hash: str
    project_path: Optional[Path] = None

    @property
    def clean_dataset(self) -> str:
        return self.dataset.strip("/")

    def locator(self) -> str:
        file = self.file
        if self.project_path is not None:
            try:
                file = file.resolve().relative_to(self.project_path.resolve())
            except Exception:
                pass
        return f"{file}:{self.clean_dataset}"

    def to_fs(self) -> Dict[str, Any]:
        return {
            "file": self.locator(),
            "format": "hdf5",
            "dataset": self.clean_dataset,
            "hash": f"blake3:{self.hash}",
        }


class SimulationStore:
    """Single HDF5 store for local simulation input arrays."""

    def __init__(self, path: Path, project_path: Optional[Path] = None):
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
    ) -> HDF5Reference:
        dataset = dataset.strip("/")
        attrs = dict(attrs or {})
        digest = hash_dataarray_payload(data, attrs=attrs)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        values = np.ascontiguousarray(data.values).astype(np.float32, copy=False)
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
