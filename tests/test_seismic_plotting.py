import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr

from frequensolve.seismic.analysis import compute_nrms, compute_timelag
from frequensolve.seismic.animate import animate_gather
from frequensolve.seismic.plotting import (
    plot_cf,
    plot_gather,
    plot_gather_diff,
    plot_timelag,
    plot_xf,
)


def _time_trace(n_time=64, n_receiver=8):
    time = np.linspace(0.0, 0.63, n_time)
    receiver = np.linspace(0.0, 0.7, n_receiver)
    data = np.zeros((n_time, n_receiver), dtype=float)
    for i in range(n_receiver):
        center = 12 + i
        data[:, i] = np.exp(-0.5 * ((np.arange(n_time) - center) / 3.0) ** 2)
    trace = xr.DataArray(
        data,
        dims=("time", "receiver"),
        coords={"time": time, "receiver": receiver},
    )
    trace.coords["time"].attrs["units"] = "s"
    trace.coords["receiver"].attrs["units"] = "km"
    return trace


def _frequency_trace(n_frequency=24, n_receiver=8):
    frequency = np.linspace(0.0, 30.0, n_frequency)
    receiver = np.linspace(0.0, 0.7, n_receiver)
    phase = np.exp(1j * 2.0 * np.pi * frequency[:, None] * receiver[None, :] / 2.0)
    return xr.DataArray(
        phase,
        dims=("frequency", "receiver"),
        coords={"frequency": frequency, "receiver": receiver},
    )


def test_plot_gather_returns_figure_and_axis():
    fig, ax = plot_gather(_time_trace(), A=1.0)

    assert fig is ax.figure
    assert len(ax.images) == 1


def test_plot_gather_diff_returns_three_axes():
    trace = _time_trace()
    fig, axes = plot_gather_diff(trace, 0.8 * trace)

    assert len(axes) == 3
    assert all(axis.figure is fig for axis in axes)


def test_frequency_plots_accept_dataarrays():
    trace = _frequency_trace()

    fig_xf, ax_xf = plot_xf(trace)
    fig_cf, ax_cf = plot_cf(trace, n_c=16)

    assert fig_xf is ax_xf.figure
    assert fig_cf is ax_cf.figure
    assert len(ax_cf.images) == 1


def test_timelag_and_nrms_helpers_return_per_receiver_arrays():
    trace = _time_trace()
    shifted = trace.shift(time=1, fill_value=0.0)

    lag = compute_timelag(trace, shifted, window_length=0.12)
    nrms = compute_nrms(trace, trace, shifted, window_length=0.12)
    fig, ax = plot_timelag(trace, shifted, window_length=0.12)

    assert lag.dims == ("receiver",)
    assert nrms.dims == ("receiver",)
    assert lag.shape == (trace.sizes["receiver"],)
    assert nrms.shape == (trace.sizes["receiver"],)
    assert lag.attrs["units"] == "ms"
    assert nrms.attrs["units"] == "%"
    assert fig is ax.figure


def test_animate_gather_returns_animation():
    trace = _time_trace(n_time=4, n_receiver=4)

    animation = animate_gather(trace, grid_shape=(2, 2))

    assert animation is not None
