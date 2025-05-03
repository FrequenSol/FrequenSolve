"""Plotting functions for seismic data.

This module is primarily for plotting time-domain data, or at least
uniformly sampled in frequency.
"""

from warnings import warn

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .shot_record import ShotRecord  # noqa

__all__ = [
    "plot_gather",
    "animate_gather",
    "plot_gather_diff",
    "plot_xf",
    "plot_cf",
    "plot_timelag",
    "compute_timelag",
    "compute_nrms",
]

try:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
except:
    pass


# --------------------------------------------
# Time-domain Plot Functions
# --------------------------------------------
def plot_gather(shot: ShotRecord, **kwargs):
    """Plot a 2D shot gather for a time-domain shot.

    Renders a simple 2D image of amplitude over (receiver X-position) vs. time.
    Useful for visualizing wavefield arrivals at multiple receivers. The plot shows
    amplitude variations using a specified colormap, with time increasing downward
    and receiver position along the horizontal axis.

    Args:
       shot (ShotRecord):     A ShotRecord object of type='TD' containing time-domain data.

    Keyword Args:
       A (float):       Amplitude scaling for display (default 1).
       units (str):     Length units for labeling X axis (default "km").
       cmap (str):      Matplotlib colormap name (default "Greys").
       figsize (tuple): Figure size (width, height) (default (8,8)).
       fontsize (int):  Font size for labels/ticks (default 14).
       Tf (float):      Cutoff time in seconds. If None, uses full time range.
       save (str):      If provided, saves the figure to this path.

    Raises:
       AssertionError:  If shot is not a time-domain shot (type != "TD").
    """
    assert shot.type == "TD"

    A = kwargs.get("A", 1)
    units = kwargs.get("units", "km")
    cmap = kwargs.get("cmap", "Greys")
    figsize = kwargs.get("figsize", (5, 5))
    fontsize = kwargs.get("fontsize", 12)
    aspect = kwargs.get("aspect", "auto")

    plt.rcParams.update({"font.size": fontsize})

    Tf = kwargs.get("Tf", None)
    nTf, Tf = shot.sampling.cutoff(Tf)

    group = shot.receiver_group

    x_min, x_max = group.coordinates.bounds
    idir = -1
    x0 = 0
    x1 = 0
    while x0 == x1:
        idir += 1
        x0 = x_min[idir]
        x1 = x_max[idir]
        xlabel = f"{['X', 'Y', 'Z'][idir]} ({units})"
        if idir == len(x_min):
            x0 = 0
            x1 = np.shape(group.coordinates)[-1]
            break

    fig = plt.figure(1, figsize=figsize)
    plt.clf()

    plt.title(f"Shot {shot.number}: {shot.field}")
    plt.xlabel(xlabel)
    plt.ylabel("Time (s)")

    if isinstance(aspect, (int, float)):
        aspect *= (x1 - x0) / Tf
    # Plot gather as an image
    plt.imshow(
        shot.data[:nTf, :],
        origin="upper",
        cmap=cmap,
        extent=[x0, x1, Tf, 0],
        vmin=-A,
        vmax=A,
        aspect=aspect,
        interpolation="bilinear",
    )
    plt.tight_layout()

    if "save" in kwargs:
        file = kwargs["save"]
        plt.savefig(file, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def animate_gather(shot: ShotRecord, **kwargs):
    """Animate a shot gather on a 2D grid (time-domain).

    This function assumes the receiver group data can be reshaped into a 2D grid
    (e.g. for snapshot-like visualization in time).

    Args:
       shot (ShotRecord): A ShotRecord object of type='TD'.

    Keyword Args:
       A (float):        Amplitude scaling for display (default 1).
       cmap (str):       Matplotlib colormap (default "RdGy").
       interval (int):   Animation interval in milliseconds (default 50).
       units (str):      Length units for labeling X, Z axes (default "km").
       figsize (tuple):  Figure size (width, height) (default (8,8)).
       fontsize (int):   Font size for labels/ticks (default 10).
       Tf (float):       Cutoff time in seconds.
       save (str):       Path (with extension) to save figure to.

    Raises:
       ValueError: If the associated receiver group is not grid-based.
    """
    assert shot.type == "TD"

    def plot_frame(u, i, fig, frames):
        frames.append(
            [
                plt.imshow(
                    u[i, :, :].transpose(),
                    origin="upper",
                    cmap=cmap,
                    vmin=-A,
                    vmax=A,
                    # extent=[x0[0], x1[0], x1[1], x0[1]],
                )
            ]
        )

    Tf = kwargs.get("Tf", None)
    nTf, Tf = shot.sampling.cutoff(Tf)

    A = kwargs.get("A", 1)
    units = kwargs.get("units", "km")
    interval = kwargs.get("interval", 50)
    cmap = kwargs.get("cmap", "Greys")
    figsize = kwargs.get("figsize", (5, 5))
    fontsize = kwargs.get("fontsize", 12)

    plt.rcParams.update({"font.size": fontsize})

    Tf = kwargs.get("Tf", None)
    nTf, Tf = shot.sampling.cutoff(Tf)

    group = shot.receiver_group

    try:
        x0 = group.grid.x0
        x1 = group.grid.x1
        n = group.grid.n
        shot.data = shot.data[:nTf, :].reshape((nTf, n[0], n[1]))
    except OSError as e:
        raise ValueError(
            "animate_gather currently only works with 'grid' receivers"
        ) from e

    frames = []
    fig = plt.figure(1, figsize=figsize)

    plt.xlabel(f"X ({units})")
    plt.ylabel(f"Z ({units})")

    ax = plt.gca()
    ax.set_axis_off()

    for i in range(nTf):
        plot_frame(shot.data, i, fig, frames)

    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)
    if "save" in kwargs:
        file = kwargs["save"]
        if file.endswith(".mp4"):
            file = file.replace(".mp4", "")
        ani.save(f"{file}.mp4")
    else:
        plt.show()

    del (frames, fig)


def plot_gather_diff(shot1: ShotRecord, shot2: ShotRecord, **kwargs):
    """Plot the difference between two time-domain shot gathers side-by-side.

    Shows the "baseline", "perturbed", and "difference" in a 3-panel figure.

    Args:
       shot1 (ShotRecord):     First TD ShotRecord (baseline).
       shot2 (ShotRecord):     Second TD ShotRecord (perturbed).

    Keyword Args:
       A (float):        Amplitude scaling for display (default 1.0).
       units (str):      Length units for labeling X axis (default "km").
       cmap (str):       Matplotlib colormap (default "Greys").
       figsize (tuple):  Figure size (width, height) (default (15,4)).
       fontsize (int):   Font size for labels/ticks (default 12).
       Tf (float):       Cutoff time in seconds.
       save (str):       Path (with extension) to save figure to.

    Raises:
       AssertionError:   If shots are not time-domain or time lengths differ.
    """
    assert shot1.type == "TD" and shot2.type == "TD"
    assert (
        shot1.sampling.nTime == shot2.sampling.nTime
        and shot1.sampling.T == shot2.sampling.T
    )

    A = kwargs.get("A", 1)
    units = kwargs.get("units", "km")
    cmap = kwargs.get("cmap", "Greys")
    figsize = kwargs.get("figsize", (10, 4))
    fontsize = kwargs.get("fontsize", 12)
    plt.rcParams.update({"font.size": fontsize})

    dpi = kwargs.pop("dpi", None)
    flip = kwargs.pop("flip", False)
    aspect = kwargs.get("aspect", "auto")
    wspace = kwargs.pop("wspace", 0.2)
    stack = kwargs.pop("stack", "horizontal")

    Cdiff = kwargs.pop("amplify_diff", 1.0)
    title1 = kwargs.get("title1", "Baseline")
    title2 = kwargs.get("title2", "Perturbed")
    title3 = kwargs.get("title3", "Difference")
    if Cdiff != 1.0:
        title3 = f"{title3} ({Cdiff}x amplified)"

    Tf = kwargs.get("Tf", None)
    nTf, Tf = shot1.sampling.cutoff(Tf)

    sgroup = shot1.source_group
    rgroup = shot1.receiver_group

    x_min, x_max = rgroup.coordinates.bounds
    idir = -1
    x0 = 0
    x1 = 0
    while x0 == x1:
        idir += 1
        x0 = x_min[idir]
        x1 = x_max[idir]
        xlabel = f"{['X', 'Depth'][idir]} ({units})"
        if idir == len(x_min):
            x0 = 0
            x1 = np.shape(rgroup.coordinates)[-1]
            break
    tlabel = "Time (s)"

    if isinstance(aspect, (int, float)):
        aspect *= (x1 - x0) / Tf

    if stack == "horizontal":
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, sharey=True)
        if flip:
            axes_y = [ax1]
            axes_x = [ax1, ax2, ax3]
        else:
            axes_y = [ax1]
            axes_x = [ax1, ax2, ax3]
    elif stack == "vertical":
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        if flip:
            axes_x = [ax3]
            axes_y = [ax1, ax2, ax3]
        else:
            axes_x = [ax3]
            axes_y = [ax1, ax2, ax3]
    plt.subplots_adjust(wspace=wspace)

    ax1.set_title(title1)
    ax2.set_title(title2)
    ax3.set_title(title3)

    # Plot data
    if flip:
        for ax in axes_y:
            ax.set_ylabel(xlabel)
        for ax in axes_x:
            ax.set_xlabel(tlabel)

        kwargs_imshow = dict(
            origin="upper",
            cmap=cmap,
            extent=[Tf, 0, x1, x0],
            vmin=-A,
            vmax=A,
            aspect=aspect,
        )

        # Flip the data horizontally by reversing the time axis
        ax1.imshow(shot1.data[nTf:1:-1, :].T, **kwargs_imshow)
        ax2.imshow(shot2.data[nTf:1:-1, :].T, **kwargs_imshow)
        ax3.imshow(
            Cdiff * (shot1.data[nTf:1:-1, :].T - shot2.data[nTf:1:-1, :].T),
            **kwargs_imshow,
        )
    else:
        for ax in axes_y:
            ax.set_ylabel(tlabel)
        for ax in axes_x:
            ax.set_xlabel(xlabel)

        kwargs_imshow = dict(
            origin="upper",
            cmap=cmap,
            extent=[x0, x1, Tf, 0],
            vmin=-A,
            vmax=A,
            aspect=aspect,
        )
        ax1.imshow(shot1.data[:nTf, :], **kwargs_imshow)
        ax2.imshow(shot2.data[:nTf, :], **kwargs_imshow)
        ax3.imshow(Cdiff * (shot1.data[:nTf, :] - shot2.data[:nTf, :]), **kwargs_imshow)

    if "save" in kwargs:
        file = kwargs["save"]
        plt.savefig(
            file, bbox_inches="tight", **({"dpi": dpi} if dpi is not None else {})
        )
        plt.close()
    else:
        plt.show()
    del fig


def hilbert_envelope(x: np.ndarray, axis: int = 0):
    """Compute the Hilbert envelope of a signal."""
    try:
        from pyfftw.interfaces import numpy_fft as fft
    except:
        warn("pyfftw not found, using numpy for FFT (slow)")
        import numpy.fft as fft

    N = x.shape[axis]

    Xf = fft.fft(x, N, axis=axis)
    h = np.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = fft.ifft(Xf * h, axis=axis)
    return np.abs(x)


def pick_first_arrivals(
    signal: np.ndarray,
    threshold_ratio: float,
):
    """Window and extract first arrivals from a pair of seismic signals.

    Uses the Hilbert transform to identify first arrivals and apply a smooth window
    around them. This is useful for isolating primary arrivals before cross-correlation
    or other analysis.

    Args:
       signal (np.ndarray):          Primary time series to analyze for first arrival.
       signal2 (np.ndarray):         Secondary time series to window (same length as signal).
       sampling_rate (float):        Sampling rate in samples per second (default 1.0).
       threshold_ratio (float):      Fraction of peak envelope amplitude for first-arrival detection (default 0.2).
       window_length (float):        Total length of the time window in seconds (default 0.01).
       window_type ('tukey' or 'gaussian'):  Type of window to apply. (default 'tukey')
       alpha (float, optional):      Shape parameter for Tukey window (default 0.5).

    Returns:
       tuple[np.ndarray, np.ndarray]: A tuple containing:
          - smoothed1: The windowed primary signal
          - smoothed2: The windowed secondary signal

    Raises:
       ValueError: If window_type is not 'tukey' or 'gaussian'.
       ValueError: If signals have different lengths.
    """

    # Step 1: Envelope
    envelope = hilbert_envelope(signal)

    # Step 2: Find first arrival index for each trace
    threshold = threshold_ratio * np.max(envelope, axis=0)
    first_arrivals = np.argmax(envelope > threshold, axis=0)

    return first_arrivals


def window_first_arrivals(
    signal: np.ndarray,
    signal2: np.ndarray,
    sampling_rate: float,
    threshold_ratio: float,
    window_length: float,
    alpha: float = 0.5,
):
    """Pick and window first arrivals from a pair of seismic signals."""
    from scipy.signal.windows import tukey

    ifirst = pick_first_arrivals(signal, threshold_ratio)

    def full_window(signal, ifirst, wl):
        """Apply a simple window, centered around first threshold crossing."""

        # Step 3: Window length in samples
        hwl = int(wl * sampling_rate / 2)
        window = tukey(2 * hwl, alpha=alpha)

        out = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            i1 = ifirst[i] - hwl
            i2 = ifirst[i] + hwl
            j1 = 0
            j2 = len(window)
            if i1 < 0:
                j1 = -i1
                i1 = 0
            if i2 > len(signal):
                j2 -= i2 - len(signal)
                i2 = len(signal) - 1
            out[i1:i2, i] = window[j1:j2] * signal[i1:i2, i]
        return out

    def small_window(signal, ifirst, wl):
        """Apply a simple window, centered around first threshold crossing."""

        # Step 3: Window length in samples
        hwl = int(wl * sampling_rate / 2)
        window = tukey(2 * hwl, alpha=alpha)

        out = np.zeros((2 * hwl, signal.shape[1]), dtype=np.single)
        for i in range(signal.shape[1]):
            i1 = ifirst[i] - hwl
            i2 = ifirst[i] + hwl
            j1 = 0
            j2 = len(window)
            if i1 < 0:
                j1 = -i1
                i1 = 0
            if i2 > len(signal):
                j2 -= i2 - len(signal)
                i2 = len(signal) - 1
            out[j1:j2, i] = window[j1:j2] * signal[i1:i2, i]
        return out

    def apply_window(sig, sig2, ifirst):
        """Apply a simple window, centered around maximum amplitude of simple window."""
        small1 = small_window(sig, ifirst, window_length)
        small2 = small_window(sig2, ifirst, window_length)

        return small1, small2

    # Step 5: Apply window
    smoothed1, smoothed2 = apply_window(signal, signal2, ifirst)

    # fig = plt.figure(figsize=(10, 5))
    # A = 0.1*np.max(np.abs(smoothed1))

    # plt.subplot(121)
    # plt.title(f"Baseline")
    # plt.ylabel("Time (s)")
    # plt.imshow(
    #     signal,
    #     origin="upper",
    #     cmap="Greys",
    #     vmin=-A,
    #     vmax=A,
    #     aspect="auto",
    #     interpolation="nearest",
    # )
    # plt.plot(ifirst, "r-")

    # plt.subplot(122)
    # plt.title(f"Monitor")
    # plt.imshow(
    #     smoothed1,
    #     origin="upper",
    #     cmap="Greys",
    #     vmin=-A,
    #     vmax=A,
    #     aspect="auto",
    #     interpolation="nearest",
    # )

    # plt.show()
    return smoothed1, smoothed2


def compute_nrms(
    base1: ShotRecord,
    shot1: ShotRecord,
    base2: ShotRecord,
    shot2: ShotRecord,
    threshold=0.2,
    **kwargs,
):
    """Compute time lag between two time-domain shots.

    Performs cross-correlation between corresponding traces in two shots to estimate
    time shifts. Useful for analyzing velocity perturbations or timing differences
    between baseline and monitor surveys.

    Args:
       shot1 (ShotRecord):     First TD ShotRecord (baseline).
       shot2 (ShotRecord):     Second TD ShotRecord (monitor/perturbed).

    Keyword Args:
       A (float):        Amplitude scaling for display (default 1.0).
       units (str):      Length units for labeling X axis (default "km").
       cmap (str):       Matplotlib colormap (default "Greys").
       figsize (tuple):  Figure size (width, height) in inches (default (8,8)).
       fontsize (int):   Font size for labels/ticks (default 14).
       Tf (float):       Cutoff time in seconds.
       save (str):       If provided, saves figure to this path.
       max_lag (float):  Maximum time lag to consider in seconds (default 1.0).

    Raises:
       AssertionError:   If shots are not time-domain or have different geometries.
       ValueError:       If shots have incompatible sampling.
    """

    sgroup = base1.source_group
    rgroup = base1.receiver_group

    nT = base1.sampling.nTime
    n_recv = rgroup.size
    rate = (nT - 1) / base1.sampling.T  # samples/second

    Tmax = kwargs.get("Tmax", None)
    nTmax, Tmax = base1.sampling.cutoff(Tmax)

    b1, win1 = window_first_arrivals(
        base1.data[:nTmax, :],
        shot1.data[:nTmax, :],
        rate,
        threshold_ratio=threshold,
        window_length=0.005,
    )

    b2, win2 = window_first_arrivals(
        base2.data[:nTmax, :],
        shot2.data[:nTmax, :],
        rate,
        threshold_ratio=threshold,
        window_length=0.005,
    )
    wind = win1 - win2

    rms1 = np.sqrt(np.mean(np.square(win1), axis=0))
    rms2 = np.sqrt(np.mean(np.square(win2), axis=0))
    rmsd = np.sqrt(np.mean(np.square(wind), axis=0))
    nrms = 100 * rmsd / ((rms1 + rms2) / 2)

    return nrms


def compute_timelag(shot1: ShotRecord, shot2: ShotRecord, threshold=0.1, **kwargs):
    """Compute time lag between two time-domain shots.

    Performs cross-correlation between corresponding traces in two shots to estimate
    time shifts. Useful for analyzing velocity perturbations or timing differences
    between baseline and monitor surveys.

    Args:
       shot1 (ShotRecord):     First TD ShotRecord (baseline).
       shot2 (ShotRecord):     Second TD ShotRecord (monitor/perturbed).

    Keyword Args:
       A (float):        Amplitude scaling for display (default 1.0).
       units (str):      Length units for labeling X axis (default "km").
       cmap (str):       Matplotlib colormap (default "Greys").
       figsize (tuple):  Figure size (width, height) in inches (default (8,8)).
       fontsize (int):   Font size for labels/ticks (default 14).
       Tf (float):       Cutoff time in seconds.
       save (str):       If provided, saves figure to this path.
       max_lag (float):  Maximum time lag to consider in seconds (default 1.0).

    Raises:
       AssertionError:   If shots are not time-domain or have different geometries.
       ValueError:       If shots have incompatible sampling.
    """

    assert shot1.type == "TD" and shot2.type == "TD"
    assert (
        shot1.sampling.nTime == shot2.sampling.nTime
        and shot1.sampling.T == shot2.sampling.T
    )

    sgroup = shot1.source_group
    rgroup = shot1.receiver_group

    nT = shot1.sampling.nTime
    n_recv = rgroup.size
    rate = (nT - 1) / shot1.sampling.T  # samples/second

    Tmax = kwargs.get("Tmax", None)
    nTmax, Tmax = shot1.sampling.cutoff(Tmax)

    win1, win2 = window_first_arrivals(
        shot1.data[:nTmax, :],
        shot2.data[:nTmax, :],
        rate,
        threshold_ratio=threshold,
        window_length=0.005,
    )

    nW = np.shape(win1)[0]

    # Compute lag time per receiver
    lag = np.zeros((n_recv), dtype=np.single)
    for i in range(n_recv):
        tr1 = win1[:, i]
        tr2 = win2[:, i]
        cor = np.correlate(tr1, tr2, "full")
        icor = np.argmax(cor) - (nW - 1)
        lag[i] = -(icor / rate * 1000)  # ms

    # Clip lag values
    lag[lag < 0] = 0

    return lag


def plot_timelag(shot1: ShotRecord, shot2: ShotRecord, threshold=0.1, **kwargs):
    """Plot time lag analysis between two time-domain shots.

    Performs cross-correlation between corresponding traces in two shots to estimate
    time shifts. Useful for analyzing velocity perturbations or timing differences
    between baseline and monitor surveys.

    Args:
       shot1 (ShotRecord):     First TD ShotRecord (baseline).
       shot2 (ShotRecord):     Second TD ShotRecord (monitor/perturbed).

    Keyword Args:
       A (float):        Amplitude scaling for display (default 1.0).
       units (str):      Length units for labeling X axis (default "km").
       cmap (str):       Matplotlib colormap (default "Greys").
       figsize (tuple):  Figure size (width, height) in inches (default (8,8)).
       fontsize (int):   Font size for labels/ticks (default 14).
       Tf (float):       Cutoff time in seconds.
       save (str):       If provided, saves figure to this path.
       max_lag (float):  Maximum time lag to consider in seconds (default 1.0).

    Raises:
       AssertionError:   If shots are not time-domain or have different geometries.
       ValueError:       If shots have incompatible sampling.
    """
    assert shot1.type == "TD" and shot2.type == "TD"
    assert (
        shot1.sampling.nTime == shot2.sampling.nTime
        and shot1.sampling.T == shot2.sampling.T
    )

    A = kwargs.get("A", 1)
    units = kwargs.get("units", "km")
    cmap = kwargs.get("cmap", "Greys")
    figsize = kwargs.get("figsize", (5, 5))
    fontsize = kwargs.get("fontsize", 12)

    sgroup = shot1.source_group
    rgroup = shot1.receiver_group

    # TODO; check that source/receiver positions align

    x_min, x_max = rgroup.coordinates.bounds
    idir = -1
    x0 = 0
    x1 = 0
    while x0 == x1:
        idir += 1
        x0 = x_min[idir]
        x1 = x_max[idir]
        # TODO: different for 2D; make getting axis limits a member of shot
        xlabel = f"{['X', 'Y', 'Z'][idir]} ({units})"
        if idir == len(x_min):
            x0 = 0
            x1 = np.shape(rgroup.coordinates)[-1]
            break

    x_s = sgroup.source.coordinates
    i_r = np.argmin(np.abs(rgroup.coordinates[:, idir] - x_s[idir]))

    plt.rcParams.update({"font.size": fontsize})

    Tf = kwargs.get("Tf", None)
    nTf, Tf = shot1.sampling.cutoff(Tf)

    Tmax = kwargs.get("Tmax", None)
    nTmax, Tmax = shot1.sampling.cutoff(Tmax)

    nT = shot1.sampling.nTime
    n_recv = rgroup.size
    rate = (nT - 1) / shot1.sampling.T  # samples/second

    win1, win2 = window_first_arrivals(
        shot1.data[:nTmax, :],
        shot2.data[:nTmax, :],
        rate,
        threshold_ratio=threshold,
        window_length=0.005,
    )

    # Compute lag time per receiver
    lag = np.zeros((n_recv), dtype=np.single)
    for i in range(n_recv):
        tr1 = win1[:, i]
        tr2 = win2[:, i]
        cor = np.argmax(np.correlate(tr1, tr2, "full"))
        cor -= nT - 1
        lag[i] = -(cor / rate * 1000)  # ms

    print("lag: ", lag[i_r], max(lag))

    x = np.linspace(x0, x1, n_recv)

    fig = plt.figure(1, figsize=figsize)
    plt.clf()
    plt.title(f"Shot {shot1.number}: {shot1.field}")
    plt.xlabel(xlabel)
    plt.ylabel("Time (s)")

    A = 0.5 * np.max(np.abs(win1))
    plt.imshow(
        win1[:nTf, :],
        origin="upper",
        cmap=cmap,
        extent=[x0, x1, Tf, 0],
        vmin=-A,
        vmax=A,
        aspect="auto",
        interpolation="nearest",
    )

    plt.tight_layout()
    plt.show()

    # fig = plt.figure(1, figsize=figsize)
    # plt.clf()

    # plt.plot(x, lag)
    # ax = plt.gca()
    # ax.set_xlim([x0, x1])
    # ax.set_ylim([0, 1.0])  # may adjust for range of lag
    # plt.xlabel(xlabel)
    # plt.ylabel("Δt (ms)")

    # if "save" in kwargs:
    #     file = kwargs["save"]
    #     plt.savefig(file, bbox_inches="tight")
    #     plt.close()
    # else:
    #     plt.show()
    # plt.close()

    return i_r, lag


# --------------------------------------------
# Frequency-domain Plot Functions
# --------------------------------------------
def plot_xf(shot: ShotRecord, **kwargs):
    """Plot a frequency-domain shot gather.

    Creates a 2D plot showing amplitude vs frequency and receiver position.
    Useful for analyzing frequency content at different receiver locations.

    Args:
       shot (ShotRecord):      A ShotRecord object of type='FD' containing frequency-domain data.

    Keyword Args:
       A (float):        Amplitude scaling for display (default 1).
       units (str):      Length units for labeling X axis (default "km").
       cmap (str):       Matplotlib colormap name (default "RdYlBu_r").
       figsize (tuple):  Figure size (width, height) in inches(default (8,8)).
       fontsize (int):   Font size for labels/ticks (default 14).
       save (str):       If provided, saves the figure to this path.

    Raises:
       AssertionError:   If shot is not a frequency-domain shot (type != "FD").
    """
    assert shot.type == "FD"

    A = kwargs.get("A", 1)
    units = kwargs.get("units", "km")
    cmap = kwargs.get("cmap", "Greys")
    figsize = kwargs.get("figsize", (5, 5))
    fontsize = kwargs.get("fontsize", 12)

    plt.rcParams.update({"font.size": fontsize})

    f_min = shot.sampling.f_min
    f_max = shot.sampling.f_max
    nf = shot.sampling.nfreq

    rgroup = shot.receiver_group

    x_min, x_max = rgroup.coordinates.bounds
    x0 = x_min[0]
    x1 = x_max[0]
    xlabel = f"X ({units})"
    if x0 == x1:
        x0 = x_min[1]
        x1 = x_max[1]
        xlabel = f"Depth ({units})"

    # Plot
    plt.ylabel("f (Hz)")
    plt.imshow(
        shot.data[:, :].real,
        origin="lower",
        cmap=cmap,
        extent=[x0, x1, 0, f_max],
        aspect="auto",
    )

    if "save" in kwargs:
        file = kwargs["save"]
        plt.savefig(file, bbox_inches="tight")
    else:
        plt.show()


def plot_cf(shot: ShotRecord, **kwargs):
    """Create a CF (phase velocity vs. frequency) plot using a smooth-windowed Radon transform.

    For a frequency-domain shot, estimates wave speed distribution by testing different
    velocities (c) for a given frequency (f). Uses a windowed Radon transform approach
    to identify dominant wave speeds at each frequency.

    Args:
       shot (ShotRecord):      A frequency-domain ShotRecord object.

    Keyword Args:
       A (float):        Amplitude scaling for color max (default 1).
       units (str):      Label for distance ("km" or "m").
       cmap (str):       Matplotlib colormap (default "RdYlBu_r").
       figsize (tuple):  Figure size (width, height) (default (8,8)).
       fontsize (int):   Font size for labels/ticks (default 14).
       c_min (float):    Minimum wave speed for transform (default 0.5).
       c_max (float):    Maximum wave speed for transform (default 6.0).
       n_c (int):        Number of wave speed samples (default 500).
       save (str):       File path to save figure.

    Raises:
       AssertionError:   If shot is not a frequency-domain shot (type != "FD").
       ValueError:       If receiver geometry is incompatible with transform.
    """
    assert shot.type == "FD"
    from scipy.ndimage import gaussian_filter1d

    A = kwargs.get("A", 1)
    units = kwargs.get("units", "km")
    cmap = kwargs.get("cmap", "RdYlBu_r")
    figsize = kwargs.get("figsize", (5, 5))
    fontsize = kwargs.get("fontsize", 14)

    symm = kwargs.get("symm", False)
    c_min = kwargs.get("c_min", 0.5)
    c_max = kwargs.get("c_max", 6.0)
    n_c = kwargs.get("n_c", 500)

    sgroup = shot.source_group
    rgroup = shot.receiver_group

    x_min, x_max = rgroup.coordinates.bounds
    idir = -1
    x0 = 0
    x1 = 0
    while x0 == x1:
        idir += 1
        x0 = x_min[idir]
        x1 = x_max[idir]
        xlabel = f"{['X', 'Y', 'Z'][idir]} ({units})"
        if idir == len(x_min):
            x0 = 0
            x1 = np.shape(rgroup.coordinates)[-1]
            break

    n1 = rgroup.size
    xl = rgroup.coordinates[:, idir] - sgroup.source.coordinates[idir]

    shot.data[:, xl < 0] = 0

    f_max = shot.sampling.f_max
    nf = shot.sampling.nfreq

    fl = np.linspace(0, f_max, nf)
    cl = np.linspace(c_min, c_max, n_c)
    cf = np.zeros((nf, n_c), dtype=np.single)

    # Define window function for spatial damping
    w = np.zeros((n1), dtype=np.single)
    i1 = n1 // 8
    i2 = n1 - i1
    w[i1:i2] = 1
    w = gaussian_filter1d(w, n1 // 16)

    # Evaluate radon-like transform
    for ifreq, f in enumerate(fl):
        for ic, c in enumerate(cl):
            v = np.exp(1j * f * 2 * np.pi * xl / c) * w
            cf[ifreq, ic] = np.abs(np.dot(shot.data[ifreq, :], v))

    plt.xlabel("f (Hz)")
    plt.ylabel(f"c ({units}/s)")
    plt.imshow(
        cf[:, :].transpose(),
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=A * np.max(cf),
        extent=[0, f_max, c_min, c_max],
        aspect="auto",
    )

    if "save" in kwargs:
        file = kwargs["save"]
        plt.savefig(file, bbox_inches="tight")
    else:
        plt.show()
