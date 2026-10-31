"""Reconstruct real point-source pressure from independent cylindrical acoustic modes."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np

__all__ = ["reconstruct_azimuthal_pressure"]


def reconstruct_azimuthal_pressure(modes: Mapping, delta_theta=0.0):
    """Sum equally forced modes as P0 + 2 sum(Pm cos(m*delta_theta)).

    ``modes`` maps every integer order 0..M to an equally shaped real array,
    or to xarray DataArrays with identical dimensions and coordinates. Each
    simulation uses the same physical source strength at the same (r,z).
    ``delta_theta`` is the receiver angle minus source angle in radians, either
    a scalar or a one-dimensional array. An angle array adds a leading angle
    axis (named ``azimuth`` for xarray). Existing sample/source axes are retained.

    Inputs must contain signed pressure, never magnitudes or energies. Apply
    source-specific angular offsets separately when sources have different
    azimuths. No temporal Fourier transform is performed.
    """
    if not isinstance(modes, Mapping) or not modes:
        raise ValueError(
            "modes must map consecutive integer orders 0..M to pressure arrays"
        )
    if any(isinstance(m, bool) or not isinstance(m, Integral) or m < 0 for m in modes):
        raise ValueError("Mode orders must be nonnegative integers")
    orders = sorted(modes)
    if orders != list(range(len(orders))):
        raise ValueError("Provide every angular mode from 0 through M")
    angles = np.asarray(delta_theta)
    if (
        angles.ndim > 1
        or angles.size == 0
        or not np.issubdtype(angles.dtype, np.number)
        or np.iscomplexobj(angles)
    ):
        raise ValueError(
            "delta_theta must be a real scalar or nonempty vector in radians"
        )
    if not np.all(np.isfinite(angles)):
        raise ValueError("Angles must be finite")
    values = [modes[m] for m in orders]
    labeled = hasattr(values[0], "dims") and hasattr(values[0], "coords")
    if any((hasattr(v, "dims") and hasattr(v, "coords")) != labeled for v in values):
        raise ValueError("Do not mix labeled and unlabeled modal arrays")
    if labeled:
        import xarray as xr

        first = values[0]
        for value in values[1:]:
            if value.dims != first.dims or not value.coords.equals(first.coords):
                raise ValueError(
                    "Modal sample coordinates and dimensions must match exactly"
                )
        if "azimuth" in first.dims or "azimuth" in first.coords:
            raise ValueError(
                "Modal inputs must not already contain an azimuth coordinate"
            )
    arrays = [np.asarray(v) for v in values]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Modal pressure arrays must have matching shapes")
    if any(
        not np.issubdtype(a.dtype, np.number)
        or np.iscomplexobj(a)
        or not np.all(np.isfinite(a))
        for a in arrays
    ):
        raise ValueError("Modal pressure must contain finite real signed samples")
    weights = np.cos(angles[..., None] * np.asarray(orders))
    weights[..., 1:] *= 2.0
    result = np.tensordot(weights, np.stack(arrays), axes=([-1], [0]))
    if not labeled:
        return result
    dims = first.dims
    coords = dict(first.coords)
    if angles.ndim:
        dims = ("azimuth",) + dims
        coords["azimuth"] = angles
    output = xr.DataArray(
        result, dims=dims, coords=coords, name=first.name, attrs=first.attrs.copy()
    )
    output.attrs.update(
        azimuthal_mode_max=orders[-1], angular_reconstruction="point-source pressure"
    )
    if angles.ndim:
        output.coords["azimuth"].attrs.update(
            units="rad", long_name="receiver angle minus source angle"
        )
    else:
        output.attrs["delta_theta_radians"] = float(angles)
    return output
