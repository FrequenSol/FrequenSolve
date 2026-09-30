"""Frozen receiver/shot polynomial windows for spectral FWI."""

import os
from math import comb
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import numpy as np

__all__ = ["write_receiver_window"]


def write_receiver_window(
    path: str | Path,
    delays: Any,
    *,
    coefficients: Any = (0, 0, 0, 0, 1),
) -> str:
    """Write coefficients of ``p(t - delays)`` and return a solver dataset path.

    ``delays`` is a finite, nonnegative seconds array of shape
    ``(global_receiver, source_field)``. It may be an HDF5 dataset: preparation
    streams bounded blocks rather than loading a survey into memory. A column
    must match a solver source field; physical arrival picks normally require
    unencoded shots. ``coefficients`` contains ascending powers of seconds,
    through degree four. The default is ``(t-delay)**4``.

    Delays can come from explicit picks or eikonal receiver times. They are
    frozen objective data, not differentiated model-dependent quantities.
    Instantaneous traveltime estimates need screening before use: they are not
    necessarily first-arrival times. This polynomial has an early-time lobe;
    it is not a causal mute. Laplace damping remains a frequency-job setting.

    Creates a new file (never overwrites an existing objective). The solver
    reads only its local receivers and active source batch from this artifact.
    """
    coefficients = np.asarray(coefficients)
    if (
        coefficients.ndim != 1
        or not 1 <= coefficients.size <= 5
        or np.iscomplexobj(coefficients)
        or not np.isfinite(coefficients).all()
        or coefficients[-1] == 0
    ):
        raise ValueError(
            "coefficients must be 1–5 finite real terms with nonzero highest term"
        )
    if not hasattr(delays, "shape"):
        delays = np.asarray(delays)
    if len(delays.shape) != 2 or min(delays.shape) <= 0:
        raise ValueError("delays must have shape (global_receiver, source_field)")
    destination = Path(path).resolve()
    receivers, sources = delays.shape
    degree = coefficients.size - 1
    # Bounded preparation workspace, independent of total acquisition size.
    chunks = (1, min(receivers, 256), min(sources, 64))
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(
        dir=destination.parent, prefix=".receiver-window-"
    ) as scratch:
        temporary = Path(scratch) / "window.h5"
        with h5py.File(temporary, "x") as handle:
            result = handle.create_dataset(
                "coefficients",
                (degree + 1, receivers, sources),
                dtype="f8",
                chunks=chunks,
            )
            handle.attrs["polynomial"] = "p(t-delay); ascending powers of seconds"
            handle.attrs["delay_units"] = "s"
            for r in range(0, receivers, chunks[1]):
                for s in range(0, sources, chunks[2]):
                    selection = (slice(r, r + chunks[1]), slice(s, s + chunks[2]))
                    delay = np.asarray(delays[selection])
                    if (
                        np.iscomplexobj(delay)
                        or not np.isfinite(delay).all()
                        or np.any(delay < 0)
                    ):
                        raise ValueError(
                            "receiver delays must be finite nonnegative seconds"
                        )
                    delay = np.asarray(delay, dtype=float)
                    shifted = np.zeros((degree + 1, *delay.shape))
                    with np.errstate(over="raise", invalid="raise"):
                        for k in range(degree + 1):
                            for n in range(k, degree + 1):
                                shifted[k] += (
                                    coefficients[n] * comb(n, k) * (-delay) ** (n - k)
                                )
                    result[(slice(None), *selection)] = shifted
        # Publish only a complete file, atomically refusing a concurrent overwrite.
        os.link(temporary, destination)
    return f"{destination}:/coefficients"
