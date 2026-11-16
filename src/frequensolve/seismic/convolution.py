"""Real causal FIR banks for native transient acquisition arrays."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from frequensolve.util.mixins import ExportContext

__all__ = ["ConvolutionFilters"]


class ConvolutionFilters:
    """Dimensionless discrete FIR taps, with no implicit dt factor.

    ``taps[..., m]`` multiplies the input at ``t - m*dt``. Source banks
    have shape (encoding, source, tap); receiver banks have shape
    (encoding, component, receiver, tap). All clocks are physical seconds.
    External banks remain lazy until validation or numerical evaluation.
    """

    dt: float
    shape: tuple[int, ...]

    def __init__(self, taps: Any, *, dt: float) -> None:
        if (
            not isinstance(dt, Real)
            or isinstance(dt, (bool, np.bool_))
            or not isfinite(float(dt))
            or dt <= 0
        ):
            raise ValueError("Convolution filter dt must be finite and positive")
        values = np.asarray(taps)
        if np.iscomplexobj(values):
            if np.any(values.imag != 0):
                raise ValueError("Convolution filters require real taps")
            values = values.real
        if values.ndim not in (3, 4) or any(n < 1 for n in values.shape):
            raise ValueError(
                "Filters require (encoding, source, tap) or (encoding, component, receiver, tap)"
            )
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("Convolution taps must be finite real numbers")
        self.dt = float(dt)
        self._taps: np.ndarray | None = np.array(
            values, dtype=np.float64, order="C", copy=True
        )
        self._taps.flags.writeable = False
        self.file: str | Path | None = None
        self.dataset: str | None = None
        self.shape = self._taps.shape

    @property
    def taps(self) -> np.ndarray:
        if self._taps is None:
            with h5py.File(self.file, "r") as h5:
                dataset = h5[self.dataset]
                expected = (int(np.prod(self.shape[:-2])), *self.shape[-2:])
                if dataset.shape != expected:
                    raise ValueError(
                        "Convolution filter table shape differs from metadata"
                    )
                loaded = type(self)(dataset[()].reshape(self.shape), dt=self.dt)
            self._taps = loaded.taps
        return self._taps

    def to_fs(
        self,
        ctx: ExportContext | None = None,
        *,
        path: str = "inputs/acquisition/convolution",
    ) -> dict[str, Any]:
        values = self.taps
        # Flatten encoding/component together for the native bank.
        native = values.reshape((-1, *values.shape[-2:]))
        payload: dict[str, Any] = {"dt": self.dt, "shape": list(values.shape)}
        store = getattr(ctx, "store", None)
        if store is None:
            payload["taps"] = native.ravel().tolist()
        else:
            ref = store.put_array_chunks(
                path,
                native.shape,
                lambda: (native,),
                dims=("channel", "element", "tap"),
                dtype=np.float64,
                attrs={"fs_kind": "convolution_filters"},
            )
            payload.update(ref.to_fs())
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> ConvolutionFilters:
        shape = tuple(data["shape"])
        dt = data["dt"]
        if (
            len(shape) not in (3, 4)
            or any(
                not isinstance(n, Integral) or isinstance(n, bool) or n < 1
                for n in shape
            )
            or not isinstance(dt, Real)
            or isinstance(dt, bool)
            or not isfinite(float(dt))
            or dt <= 0
        ):
            raise ValueError("Invalid convolution filter metadata")
        if "taps" in data:
            return cls(np.asarray(data["taps"]).reshape(data["shape"]), dt=data["dt"])
        result = cls.__new__(cls)
        result.dt = float(data["dt"])
        result.shape = tuple(data["shape"])
        result.file, result.dataset = data["file"], data["dataset"]
        result._taps = None
        return result

    def resolve(self, root: str | Path) -> None:
        if self.file is not None and not Path(self.file).is_absolute():
            self.file = Path(root) / self.file

    def frequency_response(self, frequencies: Any) -> np.ndarray:
        """Evaluate the forward FIR response with exp(-i*2*pi*f*t)."""
        f = np.asarray(frequencies, dtype=complex)
        if not np.isfinite(f).all():
            raise ValueError("Frequencies must be finite")
        phase = np.exp(-2j * np.pi * f[..., None] * np.arange(self.shape[-1]) * self.dt)
        return np.tensordot(phase, self.taps, axes=([-1], [-1]))

    def apply(self, samples: Any) -> np.ndarray:
        """Offline source/receiver matrix convolution using bounded FFT blocks.

        Inputs have shape (..., input, time); outputs (..., encoding, time).
        For receiver banks, apply one component at a time via a three-axis bank.
        The full causal tail is retained. This reference is also useful for
        comparing native arrays with independently recorded point traces.
        """
        from scipy.signal import oaconvolve

        x = np.asarray(samples)
        if len(self.shape) != 3 or x.ndim < 2 or x.shape[-2] != self.shape[-2]:
            raise ValueError(
                "apply requires a three-axis bank and (..., input, time) samples"
            )
        if np.iscomplexobj(x) or not np.isfinite(x).all() or x.shape[-1] == 0:
            raise ValueError(
                "Samples must be finite real values with a nonempty time axis"
            )
        result = np.zeros(
            (*x.shape[:-2], self.shape[0], x.shape[-1] + self.shape[-1] - 1)
        )
        for output, bank in enumerate(self.taps):
            for source, kernel in enumerate(bank):
                nonzero = np.flatnonzero(kernel)
                if not nonzero.size:
                    continue
                if nonzero.size <= 8:
                    for lag in nonzero:
                        result[..., output, lag : lag + x.shape[-1]] += (
                            kernel[lag] * x[..., source, :]
                        )
                else:
                    result[..., output, :] += oaconvolve(
                        x[..., source, :],
                        kernel.reshape((1,) * (x.ndim - 2) + (-1,)),
                        axes=-1,
                    )
        return result
