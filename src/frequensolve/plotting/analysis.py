"""Numerical analysis helpers for trace arrays."""

from __future__ import annotations

import numpy as np
import xarray as xr

from frequensolve.seismic.trace_geometry import (
    as_trace_array,
    coordinate_values,
    receiver_offsets,
    require_dims,
    sampling_rate,
    select_time,
    time_limit,
    trace_values,
)

__all__ = [
    "hilbert_envelope",
    "pick_first_arrivals",
    "window_first_arrivals",
    "compute_timelag",
    "compute_nrms",
    "phase_velocity_transform",
]


def hilbert_envelope(signal: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the analytic-signal envelope along ``axis``."""

    from scipy.signal import hilbert

    return np.abs(hilbert(np.asarray(signal), axis=axis))


def pick_first_arrivals(
    signal: np.ndarray,
    threshold_ratio: float = 0.2,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Pick first samples whose envelope exceeds a fraction of trace peak."""

    values = np.asarray(signal)
    envelope = hilbert_envelope(values, axis=axis)
    if axis != 0:
        envelope = np.moveaxis(envelope, axis, 0)
    threshold = threshold_ratio * np.nanmax(envelope, axis=0)
    return np.argmax(envelope >= threshold, axis=0)


def window_first_arrivals(
    reference: xr.DataArray | np.ndarray,
    signal: xr.DataArray | np.ndarray,
    sampling_rate_hz: float,
    *,
    threshold_ratio: float = 0.2,
    window_length: float = 0.01,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Window ``signal`` around first arrivals picked from ``reference``."""

    from scipy.signal.windows import tukey

    ref = (
        trace_values(reference)
        if isinstance(reference, xr.DataArray)
        else np.asarray(reference)
    )
    values = (
        trace_values(signal) if isinstance(signal, xr.DataArray) else np.asarray(signal)
    )
    if ref.shape != values.shape:
        raise ValueError("reference and signal must have the same shape")
    if ref.ndim != 2:
        raise ValueError("first-arrival windowing expects a 2D time/receiver array")

    picks = pick_first_arrivals(ref, threshold_ratio=threshold_ratio, axis=0)
    half_width = max(int(round(window_length * sampling_rate_hz / 2.0)), 1)
    window = tukey(2 * half_width, alpha=alpha)
    out = np.zeros_like(values)

    for receiver, pick in enumerate(np.ravel(picks)):
        start = int(pick) - half_width
        stop = int(pick) + half_width
        w_start = max(0, -start)
        w_stop = len(window) - max(0, stop - values.shape[0])
        start = max(start, 0)
        stop = min(stop, values.shape[0])
        if stop > start:
            out[start:stop, receiver] = (
                values[start:stop, receiver] * window[w_start:w_stop]
            )

    return out, np.asarray(picks, dtype=int)


def _aligned_time_pair(
    baseline: xr.DataArray,
    monitor: xr.DataArray,
    *,
    T_max: float | None = None,
    Tf: float | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    baseline = select_time(as_trace_array(baseline), time_limit(T_max, Tf))
    monitor = select_time(as_trace_array(monitor), time_limit(T_max, Tf))
    require_dims(baseline, "time", "receiver")
    require_dims(monitor, "time", "receiver")
    return xr.align(
        baseline.transpose("time", "receiver"),
        monitor.transpose("time", "receiver"),
        join="exact",
    )


def compute_timelag(
    baseline: xr.DataArray,
    monitor: xr.DataArray,
    *,
    threshold: float = 0.1,
    window_length: float = 0.01,
    max_lag: float | None = None,
    T_max: float | None = None,
    Tf: float | None = None,
) -> xr.DataArray:
    """Estimate receiver-by-receiver time lag in milliseconds."""

    baseline, monitor = _aligned_time_pair(baseline, monitor, T_max=T_max, Tf=Tf)
    rate = sampling_rate(baseline)
    base_window, _ = window_first_arrivals(
        baseline,
        baseline,
        rate,
        threshold_ratio=threshold,
        window_length=window_length,
    )
    monitor_window, _ = window_first_arrivals(
        baseline,
        monitor,
        rate,
        threshold_ratio=threshold,
        window_length=window_length,
    )

    max_shift = None if max_lag is None else max(int(round(max_lag * rate)), 1)
    lag = np.zeros(baseline.sizes["receiver"], dtype=np.float32)
    n_time = base_window.shape[0]
    for i in range(baseline.sizes["receiver"]):
        corr = np.correlate(base_window[:, i], monitor_window[:, i], mode="full")
        shifts = np.arange(-n_time + 1, n_time)
        if max_shift is not None:
            keep = np.abs(shifts) <= max_shift
            corr = corr[keep]
            shifts = shifts[keep]
        lag[i] = -float(shifts[int(np.argmax(corr))]) / rate * 1000.0

    out = xr.DataArray(
        lag,
        dims=("receiver",),
        coords={
            "receiver": baseline.coords.get("receiver", np.arange(1, lag.size + 1))
        },
        name="timelag",
    )
    out.attrs["units"] = "ms"
    return out


def compute_nrms(
    reference: xr.DataArray,
    baseline: xr.DataArray,
    monitor: xr.DataArray,
    *,
    threshold: float = 0.2,
    window_length: float = 0.01,
    T_max: float | None = None,
    Tf: float | None = None,
) -> xr.DataArray:
    """Compute normalized RMS difference around reference first arrivals."""

    reference = select_time(as_trace_array(reference), time_limit(T_max, Tf))
    baseline = select_time(as_trace_array(baseline), time_limit(T_max, Tf))
    monitor = select_time(as_trace_array(monitor), time_limit(T_max, Tf))
    require_dims(reference, "time", "receiver")
    reference, baseline, monitor = xr.align(
        reference.transpose("time", "receiver"),
        baseline.transpose("time", "receiver"),
        monitor.transpose("time", "receiver"),
        join="exact",
    )
    rate = sampling_rate(reference)
    baseline_window, _ = window_first_arrivals(
        reference,
        baseline,
        rate,
        threshold_ratio=threshold,
        window_length=window_length,
    )
    monitor_window, _ = window_first_arrivals(
        reference,
        monitor,
        rate,
        threshold_ratio=threshold,
        window_length=window_length,
    )
    diff = baseline_window - monitor_window
    rms1 = np.sqrt(np.mean(np.square(baseline_window), axis=0))
    rms2 = np.sqrt(np.mean(np.square(monitor_window), axis=0))
    rmsd = np.sqrt(np.mean(np.square(diff), axis=0))
    denominator = (rms1 + rms2) / 2.0
    nrms = np.divide(
        100.0 * rmsd,
        denominator,
        out=np.zeros_like(rmsd, dtype=float),
        where=denominator > 0,
    )
    out = xr.DataArray(
        nrms,
        dims=("receiver",),
        coords={
            "receiver": reference.coords.get("receiver", np.arange(1, nrms.size + 1))
        },
        name="nrms",
    )
    out.attrs["units"] = "%"
    return out


def phase_velocity_transform(
    trace: xr.DataArray,
    *,
    c_min: float = 0.5,
    c_max: float = 6.0,
    n_c: int = 500,
    units: str = "",
    smooth: float | None = None,
) -> xr.DataArray:
    """Compute a quick frequency/phase-velocity diagnostic transform."""

    trace = as_trace_array(trace)
    require_dims(trace, "frequency", "receiver")
    trace = trace.transpose("frequency", "receiver")

    data = trace_values(trace, complex_=True).copy()
    offset, _ = receiver_offsets(trace, units=units)
    data[:, offset < 0] = 0.0

    frequencies = coordinate_values(trace, "frequency", require_numeric=True)
    velocities = np.linspace(c_min, c_max, n_c)

    taper = np.ones(offset.size, dtype=float)
    if offset.size > 4:
        from scipy.ndimage import gaussian_filter1d

        width = max(offset.size // 8, 1)
        taper[:] = 0.0
        taper[width : offset.size - width] = 1.0
        taper = gaussian_filter1d(taper, sigma=max(offset.size // 16, 1))

    transform = np.empty((frequencies.size, velocities.size), dtype=np.float32)
    for i, frequency in enumerate(frequencies):
        phase = np.exp(1j * 2.0 * np.pi * frequency * offset[:, None] / velocities)
        transform[i, :] = np.abs(data[i, :] @ (phase * taper[:, None]))

    if smooth is not None and smooth > 0:
        from scipy.ndimage import gaussian_filter

        transform = gaussian_filter(transform, sigma=float(smooth))

    out = xr.DataArray(
        transform,
        dims=("frequency", "phase_velocity"),
        coords={"frequency": frequencies, "phase_velocity": velocities},
        name="phase_velocity_transform",
    )
    if "frequency" in trace.coords:
        out.coords["frequency"].attrs.update(trace.coords["frequency"].attrs)
    out.coords["frequency"].attrs.setdefault("units", "Hz")
    if units:
        out.coords["phase_velocity"].attrs["units"] = f"{units}/s"
    return out
