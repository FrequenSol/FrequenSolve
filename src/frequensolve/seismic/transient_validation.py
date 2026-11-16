"""Early validation of physical-time acquisition operators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _real(values: Any, label: str) -> None:
    values = np.asarray(values)
    if not np.isfinite(values).all() or (
        np.iscomplexobj(values) and np.any(values.imag != 0)
    ):
        raise ValueError(
            f"Transient {label} must be finite and real; use convolution filters for phase/delay encoding"
        )


def _table(file: str | Path, dataset: str, root: str | Path, label: str) -> None:
    path = Path(file)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        return  # External catalogs are independently validated by the native reader.
    with h5py.File(path, "r") as h5:
        data = h5[dataset]
        if data.ndim != 3 or data.shape[-1] != 2:
            raise ValueError(
                f"Transient {label} requires static complex-pair matrix storage"
            )
        stride = max(
            1,
            (8 * 1024 * 1024) // max(1, np.prod(data.shape[1:]) * data.dtype.itemsize),
        )
        for start in range(0, data.shape[0], stride):
            block = data[start : start + stride]
            if not np.isfinite(block).all() or np.any(block[..., 1] != 0):
                raise ValueError(
                    f"Transient {label} must be finite and real; use convolution filters"
                )


def validate_transient_acquisition(
    acquisition: Any, *, dt: float, project_path: str | Path
) -> None:
    from .sources import _complex_value

    encoding = acquisition.source_encoding
    geometry = acquisition.source_geometry
    if encoding is not None:
        if (
            encoding.encoding_type == "FrequencyDense"
            or encoding.frequencies_dataset is not None
        ):
            raise ValueError(
                "Transient source encoding cannot use frequency-dependent weights; use convolution filters"
            )
        if encoding.weights is not None:
            _real(encoding.weights, "source weights")
        elif encoding.encoding_type == "Named":
            for field in encoding.fields:
                _real(
                    [_complex_value(value) for value in field.terms.values()],
                    "source weights",
                )
        elif encoding.encoding_type == "HDF5Dense":
            _table(encoding.file, encoding.dataset, project_path, "source weights")
        if encoding.filters is not None:
            if geometry is None:
                raise ValueError("Source convolution requires source geometry")
            encoding.filters.resolve(project_path)
            _real(encoding.filters.taps, "source filter taps")
            if encoding.filters.shape[:2] != (
                encoding.field_count,
                geometry.point_count,
            ):
                raise ValueError(
                    "Source filter dimensions must match encoded fields and physical sources"
                )
    for group in acquisition.receiver_groups:
        device = group.device
        for obj in [device, *device.components]:
            if (
                getattr(obj, "transfer", None) is not None
                or getattr(obj, "response", None) is not None
            ):
                raise ValueError(
                    "Transient receivers cannot use spectral transfer/response functions; use convolution filters"
                )
            weight = getattr(obj, "weight", None)
            if weight is not None:
                _real(_complex_value(weight), "receiver weight")
        if hasattr(device, "_authored_weight_blocks"):
            for weights, _ in device._authored_weight_blocks():
                _real(weights, "receiver weights")
            if device.weight_table is not None:
                _table(
                    device.weight_table.file,
                    device.weight_table.dataset,
                    project_path,
                    "receiver weights",
                )
        filters = getattr(device, "filters", None)
        if filters is not None:
            filters.resolve(project_path)
            _real(filters.taps, "receiver filter taps")
            if filters.shape[2] != group.size:
                raise ValueError("Receiver filter element count must match coordinates")
            if not np.isclose(filters.dt, dt, rtol=1e-10, atol=0):
                raise ValueError(
                    "Receiver convolution dt must match the integration timestep"
                )
            if filters.shape[:2] != (device.encoding_count, len(device.components)):
                raise ValueError(
                    "Receiver filter dimensions differ from encoding/component counts"
                )
            if device.reduction not in {"sum", "mean"}:
                raise ValueError("Receiver convolution requires sum or mean reduction")
            if group.sampling is not None and group.sampling.kind not in {
                "DenseAllToAll",
                "Dense",
            }:
                raise ValueError("Receiver convolution requires dense sampling")
