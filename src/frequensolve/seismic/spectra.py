"""Explicit sampled signals and complex acquisition transfer functions.

The transform is sum(x[n] exp(-2 pi i (f + i laplace) t[n])). ``laplace``
is the nonpositive imaginary frequency in Hz used by Sauce, not a decay rate
in radians/second. ``integral`` normalization additionally multiplies by dt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from xarray import DataArray

__all__ = [
    "SampledWavelet",
    "TabulatedSpectrum",
    "GainDelay",
    "TransferFunction",
    "SpectralResponse",
]


def _readonly(values: ArrayLike, dtype: DTypeLike) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    # Immutable backing storage also prevents callers re-enabling WRITEABLE.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _axis(values: ArrayLike, name: str, *, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric values")
    axis = _readonly(raw, np.float64)
    if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional axis")
    if nonnegative and np.any(axis < 0):
        raise ValueError(f"{name} must be nonnegative")
    return axis


def _coordinates(
    frequencies: ArrayLike, laplace: float
) -> tuple[np.ndarray, np.ndarray]:
    f = _axis(frequencies, "frequencies", nonnegative=True)
    laplace = float(laplace)
    if not np.isfinite(laplace) or laplace > 0:
        raise ValueError("laplace must be a nonpositive imaginary frequency in Hz")
    return f, f + 1j * laplace


def _finite(values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("Spectral evaluation overflowed or produced nonfinite values")
    return values


class TransferFunction:
    """Evaluate a complex transfer function H(f), with Y(f) = H(f) X(f).

    Magnitude describes gain and argument describes phase. ``at_frequencies``
    also evaluates dH/df in physical Hz. A normalized sampled impulse response
    uses the same protocol; an emitted source waveform is a SourceSignature.
    ``SpectralResponse`` is a compatibility alias for this class.
    """

    units = "1"

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__}

    def at_frequencies(
        self, frequencies: ArrayLike, *, laplace: float = 0.0, derivative: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def __mul__(self, other: object) -> TransferFunction:
        if not isinstance(other, TransferFunction):
            return NotImplemented
        return _ProductResponse(self, other)


@dataclass(frozen=True)
class GainDelay(TransferFunction):
    """Transfer function for constant complex gain and delay in seconds.

    Positive delay is later.

    A constant phase offset can be represented explicitly by a complex gain.
    A complex gain need not represent a real impulse response at DC/Nyquist.
    """

    gain: complex = 1.0
    delay: float = 0.0

    def __post_init__(self) -> None:
        gain, delay = complex(self.gain), float(self.delay)
        if not np.isfinite(gain) or not np.isfinite(delay):
            raise ValueError("gain and delay must be finite")
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "delay", delay)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "GainDelay",
            "gain": [self.gain.real, self.gain.imag],
            "delay_seconds": self.delay,
        }

    def at_frequencies(
        self, frequencies: ArrayLike, *, laplace: float = 0.0, derivative: bool = False
    ) -> np.ndarray:
        _, z = _coordinates(frequencies, laplace)
        with np.errstate(over="ignore", invalid="ignore"):
            value = self.gain * np.exp(-2j * np.pi * z * self.delay)
            if derivative:
                value *= -2j * np.pi * self.delay
        return _finite(value)


@dataclass(frozen=True)
class _ProductResponse(TransferFunction):
    left: TransferFunction
    right: TransferFunction

    def __post_init__(self) -> None:
        if self.left.units != "1" or self.right.units != "1":
            raise ValueError(
                "Compose dimensionless responses; normalize physical samples first"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "Product",
            "left": self.left.describe(),
            "right": self.right.describe(),
        }

    def at_frequencies(
        self, frequencies: ArrayLike, *, laplace: float = 0.0, derivative: bool = False
    ) -> np.ndarray:
        left = self.left.at_frequencies(frequencies, laplace=laplace)
        right = self.right.at_frequencies(frequencies, laplace=laplace)
        if derivative:
            value = self.left.at_frequencies(
                frequencies, derivative=True, laplace=laplace
            ) * right + left * self.right.at_frequencies(
                frequencies, derivative=True, laplace=laplace
            )
        else:
            value = left * right
        return _finite(value)


@dataclass(frozen=True, init=False)
class SampledWavelet(TransferFunction):
    """An immutable recording with an explicit clock and Fourier normalization.

    ``samples`` contains every sample (no duplicated endpoint convention).
    ``normalization='dft'`` matches Sauce trace-store Fourier extraction;
    ``'integral'`` approximates the continuous transform by multiplying by dt.
    No peak normalization, window, phase conversion, or resampling is implicit.
    """

    samples: np.ndarray
    dt: float
    t0: float
    normalization: str
    units: str

    def __init__(
        self,
        samples: ArrayLike,
        *,
        dt: float,
        t0: float = 0.0,
        normalization: str = "dft",
        units: str = "1",
    ) -> None:
        samples = _axis(samples, "samples")
        dt, t0 = float(dt), float(t0)
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
            raise ValueError("dt must be positive and t0 finite, in seconds")
        if normalization not in {"dft", "integral"}:
            raise ValueError("normalization must be 'dft' or 'integral'")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("units must be a nonempty unit expression")
        for name, value in dict(
            samples=samples, dt=dt, t0=t0, normalization=normalization, units=units
        ).items():
            object.__setattr__(self, name, value)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        return self

    def describe(self) -> dict[str, Any]:
        return {
            "type": "SampledWavelet",
            "dt_seconds": self.dt,
            "t0_seconds": self.t0,
            "sample_count": len(self.samples),
            "normalization": self.normalization,
            "units": self.units,
            "samples_sha256": hashlib.sha256(self.samples.tobytes()).hexdigest(),
        }

    @property
    def times(self) -> np.ndarray:
        return self.t0 + np.arange(len(self.samples)) * self.dt

    @property
    def signal(self) -> np.ndarray:
        return self.samples

    @property
    def frequencies(self) -> np.ndarray:
        return np.fft.rfftfreq(len(self.samples), self.dt)

    @property
    def spectrum(self) -> np.ndarray:
        return self.at_frequencies(self.frequencies)

    @property
    def center(self) -> float:
        # Recorded time origins are physical, never an automatic display shift.
        return 0.0

    @property
    def scale(self) -> float:
        return 1.0

    def relative_to(self, strength: float, *, units: str) -> SampledWavelet:
        """Divide physical samples by a reference source strength in given units.

        The corresponding Sauce source amplitude must use this same strength.
        This explicit conversion prevents applying the physical amplitude twice.
        """
        from frequensolve.units import ureg

        strength = float(strength)
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("reference strength must be finite and positive")
        factor = (1.0 * ureg(self.units)).to(units).magnitude / strength
        return SampledWavelet(
            self.samples * factor,
            dt=self.dt,
            t0=self.t0,
            normalization=self.normalization,
        )

    def at_frequencies(
        self,
        frequencies: ArrayLike,
        *,
        laplace: float = 0.0,
        derivative: bool = False,
        batch_bytes: int = 64 * 1024**2,
    ) -> np.ndarray:
        f, z = _coordinates(frequencies, laplace)
        if np.any(f > 0.5 / self.dt):
            raise ValueError("Requested frequency exceeds recording Nyquist")
        if not isinstance(batch_bytes, int) or batch_bytes <= 0:
            raise ValueError("batch_bytes must be a positive integer")
        count = len(self.samples)
        bins = np.rint(f * count * self.dt).astype(np.int64)
        # Tight alignment: never silently approximate a requested physical frequency.
        aligned = np.all(
            np.abs(f - bins / (count * self.dt))
            <= 8 * np.finfo(float).eps * np.maximum(1, f)
        )
        time = self.times
        work = self.samples if not derivative else self.samples * (-2j * np.pi * time)
        if aligned and laplace == 0:
            # The time moment is real: use an FFT for its exact derivative too.
            fft_samples = self.samples * time if derivative else self.samples
            values = np.fft.rfft(fft_samples)[bins] * np.exp(-2j * np.pi * f * self.t0)
            if derivative:
                values *= -2j * np.pi
        else:
            values = np.empty(len(f), dtype=np.complex128)
            width = max(1, batch_bytes // max(1, count * 32))
            for first in range(0, len(f), width):
                last = min(first + width, len(f))
                with np.errstate(over="ignore", invalid="ignore"):
                    kernel = np.exp(-2j * np.pi * time[:, None] * z[None, first:last])
                    values[first:last] = np.einsum(
                        "t,tf->f", work, kernel, optimize=False
                    )
        if self.normalization == "integral":
            values *= self.dt
        return _finite(values)

    @classmethod
    def from_xarray(
        cls, data: DataArray, *, normalization: str = "dft"
    ) -> SampledWavelet:
        if data.dims != ("time",) or "time" not in data.coords:
            raise ValueError(
                "Select one recording with a time coordinate before import"
            )
        times = _axis(data.time.values, "time")
        time_units = data.time.attrs.get("units", "s")
        if time_units != "s":
            from frequensolve.units import ureg

            times = times * (1.0 * ureg(time_units)).to("s").magnitude
        if len(times) < 2:
            raise ValueError("At least two time coordinates are required to infer dt")
        dt = times[1] - times[0]
        if dt <= 0 or not np.allclose(np.diff(times), dt, rtol=1e-7, atol=1e-12):
            raise ValueError("Recording time coordinates must be uniformly increasing")
        return cls(
            data.values,
            dt=dt,
            t0=times[0],
            normalization=normalization,
            units=data.attrs.get("units", "1"),
        )

    @classmethod
    def from_npy(
        cls,
        file: str | Path,
        *,
        dt: float,
        t0: float = 0.0,
        normalization: str = "dft",
        units: str = "1",
    ) -> SampledWavelet:
        return cls(
            np.load(Path(file), allow_pickle=False),
            dt=dt,
            t0=t0,
            normalization=normalization,
            units=units,
        )

    @classmethod
    def from_segy(
        cls, file: str | Path, *, trace: int = 0, normalization: str = "dft", units: str
    ) -> SampledWavelet:
        """Read one zero-based trace, with header interval/delay and explicit units."""
        from frequensolve._optional import optional_dependency_error

        try:
            import segyio
        except ImportError as exc:
            raise optional_dependency_error(
                "SEG-Y wavelets",
                extra="seismic-io",
                dependencies=("segyio",),
                error=exc,
            ) from exc
        with segyio.open(str(file), ignore_geometry=True) as handle:
            if not isinstance(trace, int) or not 0 <= trace < handle.tracecount:
                raise ValueError("trace must be a valid zero-based SEG-Y trace index")
            header = handle.header[trace]
            dt = (
                header[segyio.TraceField.TRACE_SAMPLE_INTERVAL]
                or handle.bin[segyio.BinField.Interval]
            )
            return cls(
                handle.trace[trace],
                dt=dt * 1e-6,
                t0=header[segyio.TraceField.DelayRecordingTime] * 1e-3,
                normalization=normalization,
                units=units,
            )

    def plot(self, ax_time: Any = None, ax_freq: Any = None) -> tuple[Any, Any]:
        """Plot the original recording and its Fourier amplitude on its own grid."""
        import matplotlib.pyplot as plt

        if ax_time is None:
            _, ax_time = plt.subplots()
        if ax_freq is None:
            _, ax_freq = plt.subplots()
        ax_time.plot(self.times, self.samples)
        ax_time.set(xlabel="Time [s]", ylabel=f"Amplitude [{self.units}]")
        ax_freq.plot(self.frequencies, np.abs(self.spectrum))
        ax_freq.set(xlabel="Frequency [Hz]", ylabel=f"Amplitude ({self.normalization})")
        return ax_time.figure, ax_freq.figure


@dataclass(frozen=True, init=False)
class TabulatedSpectrum(TransferFunction):
    """Complex spectrum at one explicit damping coordinate.

    Exact lookup is default. ``interpolation='linear'`` interpolates real and
    imaginary parts and uses the same piecewise-linear derivative (right-sided
    at interior knots). Supply derivatives only for exact lookup. Out-of-band
    evaluation and evaluation at a different damping coordinate are errors.
    """

    frequencies: np.ndarray
    values: np.ndarray
    derivatives: np.ndarray | None
    laplace: float
    interpolation: str

    def __init__(
        self,
        frequencies: ArrayLike,
        values: ArrayLike,
        *,
        derivatives: ArrayLike | None = None,
        laplace: float = 0.0,
        interpolation: str = "exact",
    ) -> None:
        f, _ = _coordinates(frequencies, laplace)
        if np.any(np.diff(f) <= 0):
            raise ValueError("Tabulated frequencies must be strictly increasing")
        if interpolation not in {"exact", "linear"}:
            raise ValueError("interpolation must be 'exact' or 'linear'")
        if interpolation == "linear" and (len(f) < 2 or derivatives is not None):
            raise ValueError(
                "Linear interpolation requires two samples and computes its own derivative"
            )
        arrays: dict[str, np.ndarray | None] = {}
        for name, value in (("values", values), ("derivatives", derivatives)):
            if value is None and name == "derivatives":
                arrays[name] = None
                continue
            if value is None:
                raise ValueError(f"{name} must be finite and match frequencies")
            array = _readonly(value, np.complex128)
            if array.shape != f.shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite and match frequencies")
            arrays[name] = array
        for name, value in dict(
            frequencies=f, laplace=float(laplace), interpolation=interpolation, **arrays
        ).items():
            object.__setattr__(self, name, value)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        return self

    def describe(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.frequencies.tobytes() + self.values.tobytes())
        if self.derivatives is not None:
            digest.update(self.derivatives.tobytes())
        return {
            "type": "TabulatedSpectrum",
            "laplace_hz": self.laplace,
            "interpolation": self.interpolation,
            "spectrum_sha256": digest.hexdigest(),
        }

    def at_frequencies(
        self, frequencies: ArrayLike, *, laplace: float = 0.0, derivative: bool = False
    ) -> np.ndarray:
        f, _ = _coordinates(frequencies, laplace)
        if float(laplace) != self.laplace:
            raise ValueError(
                "No tabulated spectrum at the requested Laplace coordinate"
            )
        if np.any(f < self.frequencies[0]) or np.any(f > self.frequencies[-1]):
            raise ValueError("Requested frequencies are outside the tabulated band")
        if self.interpolation == "linear":
            i = np.clip(
                np.searchsorted(self.frequencies, f, side="right") - 1,
                0,
                len(self.frequencies) - 2,
            )
            slope = np.diff(self.values)[i] / np.diff(self.frequencies)[i]
            return (
                slope
                if derivative
                else self.values[i] + slope * (f - self.frequencies[i])
            )
        i = np.searchsorted(self.frequencies, f)
        if np.any(self.frequencies[i] != f):
            raise ValueError("Exact spectral lookup requires matching frequencies")
        if derivative and self.derivatives is None:
            raise ValueError("Tabulated spectrum needs explicit frequency derivatives")
        if derivative:
            assert self.derivatives is not None
            return self.derivatives[i].copy()
        return self.values[i].copy()


# Preserve imports and isinstance checks made with the original public name.
SpectralResponse = TransferFunction
