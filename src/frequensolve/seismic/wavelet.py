from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.signal import chirp, correlate, hilbert
from scipy.stats import norm

from frequensolve._optional import optional_dependency_error
from frequensolve.util.fft import get_fft_backend

__all__ = [
    "WindowFunction",
    "BlackmanWindow",
    "GaussianWindow",
    "Wavelet",
    "RickerWavelet",
    "OrmsbyWavelet",
    "KlauderWavelet",
]


# ----------------------------------------------------------------------
# Window Functions
# ----------------------------------------------------------------------
class WindowFunction(ABC):
    """Abstract base class for window functions.

    This class defines the interface for window functions used in signal processing.
    """

    @abstractmethod
    def get(self, times: np.ndarray, n: int) -> np.ndarray:
        """Generate a window function of length n.

        Args:
            n (int): Number of points in the window.

        Returns:
            np.ndarray: Window array in [0,1].
        """
        pass


@dataclass
class GaussianWindow(WindowFunction):
    """Gaussian window function implementation.

    Attributes:
        sigma (float): The standard deviation of the Gaussian taper.
    """

    sigma: float

    def get(self, times: np.ndarray, n_dummy: int) -> np.ndarray:
        """Generate a normalized Gaussian taper of length n.

        Args:
            n (int): Number of points in the taper.

        Returns:
            np.ndarray: Taper array in [0,1].
        """
        n = len(times)
        vals = norm.pdf(np.linspace(0, 1, n), 0.5, self.sigma)
        mx = np.max(vals)
        vals = vals / mx
        return vals


@dataclass
class BlackmanWindow(WindowFunction):
    """Blackman window function implementation.

    Attributes:
        T (float): Duration (in seconds) of the window.
    """

    T: float

    def get(self, times, n: int) -> np.ndarray:
        """Generate a zero-phase Blackman taper matching the signal length.

        Returns:
            np.ndarray: Taper array in [0,1].
        """
        n = int(n)
        if n <= 0:
            return np.zeros(0)
        if self.T <= 0:
            return np.zeros(n)

        zero_phase = _zero_phase_times(times)
        if len(zero_phase) != n:
            zero_phase = _zero_phase_times(np.asarray(times)[:n])

        half_width = 0.5 * self.T
        vals = np.zeros(n)
        mask = np.abs(zero_phase) <= half_width
        if np.any(mask):
            u = (zero_phase[mask] + half_width) / self.T
            vals[mask] = 1.0 - (
                0.42 + 0.5 * np.cos(2 * np.pi * u) + 0.08 * np.cos(4 * np.pi * u)
            )
            peak = np.max(vals)
            if peak > 0:
                vals /= peak
        return vals

    def plot(self, times, **kwargs) -> None:
        """Plot the window function."""
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise optional_dependency_error(
                "Window plotting",
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc
        plt.plot(times, self.get(times, len(times)), **kwargs)
        plt.show()


def _zero_phase_times(times: np.ndarray) -> np.ndarray:
    """Return FFT-ordered times with zero lag at index 0."""

    n = len(times)
    if n < 2:
        return np.zeros(n)
    dt = float(times[1] - times[0])
    indices = np.arange(n, dtype=float)
    indices[indices > n // 2] -= n
    return indices * dt


def _apply_taper(signal: np.ndarray, taper: Optional[Callable[[int], np.ndarray]]):
    if taper is None:
        return signal
    window = np.asarray(taper(len(signal)))
    if window.shape != signal.shape:
        raise ValueError("Wavelet taper returned an array with the wrong shape")
    return signal * window


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(np.abs(signal))) if signal.size else 0.0
    return signal / peak if peak > 0 else signal


def _trapezoid_spectrum(
    frequencies: np.ndarray,
    corners: Union[List[float], Tuple[float, float, float, float]],
) -> np.ndarray:
    f1, f2, f3, f4 = [float(f) for f in corners]
    if not (0 <= f1 < f2 <= f3 < f4):
        raise ValueError("Ormsby frequencies must satisfy 0 <= f1 < f2 <= f3 < f4")

    amplitude = np.zeros_like(frequencies, dtype=float)
    if f2 > f1:
        mask = (frequencies >= f1) & (frequencies < f2)
        amplitude[mask] = (frequencies[mask] - f1) / (f2 - f1)
    mask = (frequencies >= f2) & (frequencies <= f3)
    amplitude[mask] = 1.0
    if f4 > f3:
        mask = (frequencies > f3) & (frequencies <= f4)
        amplitude[mask] = (f4 - frequencies[mask]) / (f4 - f3)
    return amplitude


# ----------------------------------------------------------------------
# Wavelet
# ----------------------------------------------------------------------
@dataclass(init=False)
class Wavelet:
    """Data container for wavelets in time and frequency domains.

    Attributes:
       times (np.ndarray): The time samples.
       signal (np.ndarray): The wavelet signal.
    """

    f: Union[float, List[float]]
    window: Optional[Tuple[Literal["gaussian", "blackman"], float]] = None
    f_max: float = field(init=False)
    _center: float = field(default=0.0, init=False, repr=False)
    _causal: bool = field(default=False, init=False, repr=False)
    _scale: float = field(default=1.0, init=False, repr=False)
    _times: np.ndarray = field(default=None, init=False)
    _signal: np.ndarray = field(default=None, init=False)
    _frequencies: np.ndarray = field(default=None, init=False)
    _spectrum: np.ndarray = field(default=None, init=False)

    def __init__(
        self,
        f: Union[float, List[float]],
        center: float = 0.0,
        window: Optional[Tuple[Literal["gaussian", "blackman"], float]] = None,
        causal: bool = False,
        scale: float = 1.0,
    ):
        self.f = f
        self._center = float(center)
        self.window = window
        self._causal = bool(causal)
        self._scale = float(scale)
        self._times = None
        self._signal = None
        self._frequencies = None
        self._spectrum = None

    @property
    def center(self) -> float:
        """Wavelet center time. Use ``recenter`` to update cached samples."""

        return self._center

    @property
    def causal(self) -> bool:
        """Whether this wavelet is generated in causal form."""

        return self._causal

    @property
    def scale(self) -> float:
        """Amplitude scale applied during wavelet generation."""

        return self._scale

    @property
    def times(self) -> np.ndarray:
        return self._times

    @times.setter
    def times(self, times: np.ndarray) -> None:
        """Set the time samples and invalidate cached properties.

        Args:
            times: The new time samples.
        """

        times = np.asarray(times)
        if self._times is not None and np.array_equal(times, self._times):
            return
        self._generate_for_times(times, invalidate_frequencies=True)

    def _generate_for_times(
        self, times: np.ndarray, *, invalidate_frequencies: bool
    ) -> None:
        self._times = np.asarray(times)
        self._spectrum = None
        if invalidate_frequencies:
            self._frequencies = None
        taper = Wavelet._get_window_callable(self._times, self.window)
        offset = (
            0
            if self._causal
            else np.searchsorted(self._times, self._center, side="left")
        )
        self._generate(self._times, taper)

        # if self.causal:
        #     self._signal = Wavelet._make_causal(self.signal)
        self._signal = np.roll(self._signal, shift=offset)

    def recenter(self, center: float, times: Optional[np.ndarray] = None) -> np.ndarray:
        """Set the center time and regenerate cached time/frequency samples."""

        self._center = float(center)
        if times is not None:
            self._generate_for_times(np.asarray(times), invalidate_frequencies=True)
        elif self._times is not None:
            self._generate_for_times(self._times, invalidate_frequencies=False)
        else:
            self._evaluate_initial()
        return self.signal

    @property
    def signal(self) -> np.ndarray:
        return self._signal

    @signal.setter
    def signal(self, signal: np.ndarray) -> None:
        self._signal = signal

    @property
    def spectrum(self):
        """Compute the frequency-domain representation of the wavelet.

        Returns:
           np.ndarray: The frequency-domain representation of the wavelet.
        """
        if self._spectrum is None:
            self._evaluate_initial()
            fft = get_fft_backend()
            self._spectrum = fft.rfft(self.signal).astype(np.complex64)
        return self._spectrum

    @property
    def frequencies(self):
        """Compute the frequencies for the wavelet.

        Returns:
           np.ndarray: The frequencies for the wavelet.
        """
        if self._frequencies is None:
            self._evaluate_initial()
            n = len(self._times) - 1
            dt = self.times[1] - self.times[0]
            fft = get_fft_backend()
            self._frequencies = fft.rfftfreq(n, d=dt).astype(np.float32)
        return self._frequencies

    def evaluate(self, times: np.ndarray) -> np.ndarray:
        """Evaluate the wavelet at given times."""

        # Setting times will trigger re-evaluation if needed
        self.times = times
        return self.signal

    def _evaluate_initial(self):
        if self.times is None:
            dt = 1.0 / (50.0 * self.f_max)
            T_max = 100.0 / self.f_max
            self.times = np.arange(0.0, T_max + dt, dt)

    @staticmethod
    def _get_window_callable(
        times: np.ndarray,
        window: Optional[
            Union[WindowFunction, Tuple[Literal["gaussian", "blackman"], float]]
        ] = None,
        sigma: Optional[float] = None,
    ) -> Callable[[int], np.ndarray]:
        """Get a callable that returns a window of length n."""

        if isinstance(window, tuple):
            if window[0].lower() == "gaussian":
                window = GaussianWindow(sigma=window[1] / times[-1])
            elif window[0].lower() == "blackman":
                window = BlackmanWindow(T=window[1])
            else:
                raise ValueError(f"Unknown window type: {window[0]}")
        elif sigma is not None:  # legacy support
            window = GaussianWindow(sigma=sigma / times[-1])

        if window is not None:
            return lambda n: window.get(times, n)
        else:
            return None

    @staticmethod
    def _make_causal(signal: np.ndarray) -> np.ndarray:
        """Convert a wavelet to minimum phase (causal) form using Kolmogorov factorization.

        Args:
            signal: Input time-domain signal to make causal

        Returns:
            Causal (minimum phase) version of input signal
        """
        n = len(signal)
        fft = get_fft_backend()

        # Get spectrum and add small constant for stability
        spectrum = fft.rfft(signal)
        A = np.abs(spectrum)
        eps = np.max(A) * 1e-3
        Atmp = np.clip(A, eps, np.inf)

        # Compute minimum phase using Kolmogorov method
        lnA = np.log(Atmp)
        phi = np.imag(hilbert(lnA))
        phase_min = np.exp(-1j * phi)

        # Transform back to time domain
        signal_min = fft.irfft(phase_min * A, n=n)
        signal_min = signal_min[: len(signal_min) // 2 + 1]
        return signal_min

    def plot(self, **kwargs) -> None:
        """Plot the time-domain and spectrum of the wavelet."""
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise optional_dependency_error(
                "Wavelet plotting",
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc

        self._evaluate_initial()

        f_max = kwargs.pop("f_max", self.f_max)

        # Axis limit kwargs
        Tf = kwargs.pop("T_max", 25.0 / f_max)
        if Tf > self.times[-1]:
            dt = self.times[1] - self.times[0]
            self.times = np.arange(0.0, Tf + dt, dt)
        nTf = np.searchsorted(self.times, Tf, side="left")
        nTf = np.minimum(nTf, len(self.times))

        nF = np.searchsorted(self.frequencies, f_max, side="left")
        nF = np.minimum(nF, len(self.frequencies))

        # Save kwargs
        save_time = kwargs.pop("save_time", None)
        save_freq = kwargs.pop("save_freq", None)
        dpi = kwargs.pop("dpi", None)
        y_scale = kwargs.pop("y_scale", None)

        # Formatting kwargs
        figsize = kwargs.pop("figsize", (6, 3))
        fontsize = kwargs.pop("fontsize", 10)
        plt.rcParams.update({"font.size": fontsize})

        # Plot time-domain
        plt.figure(figsize=figsize)
        times = self.times
        signal = self.signal
        if signal.shape != times.shape:
            times = times[: signal.size]
        plt.plot(times[:nTf] - self.center, signal[:nTf], **kwargs)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        if save_time:
            plt.savefig(
                save_time,
                bbox_inches="tight",
                **({"dpi": dpi} if dpi is not None else {}),
            )
            plt.close()
        else:
            plt.show()

        # Plot frequency-domain
        plt.figure(figsize=figsize)
        if y_scale == "dB":
            normalized_spectrum = np.abs(self.spectrum) / np.max(np.abs(self.spectrum))
            plt.plot(
                self.frequencies[:nF], 20 * np.log10(normalized_spectrum[:nF]), **kwargs
            )
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude [dB]")
            plt.grid(True, alpha=0.3)
            plt.yticks(np.arange(-80, 1, 20))
            plt.yticks(np.arange(-80, 1, 10), minor=True)
            plt.ylim(-80, 1)
        else:
            plt.plot(self.frequencies[:nF], np.abs(self.spectrum[:nF]))
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)

        plt.grid(True, which="minor", linestyle=":", alpha=0.3)
        if save_freq:
            plt.savefig(
                save_freq,
                bbox_inches="tight",
                **({"dpi": dpi} if dpi is not None else {}),
            )
            plt.close()
        else:
            plt.show()


class RickerWavelet(Wavelet):
    """Ricker wavelet.

    Attributes:
        f (float): Central frequency of the Ricker wavelet.
        center (float): Time shift used to place the wavelet peak in sampled
            traces. Defaults to one period, ``1 / f``.
        window (Optional[Union[WindowFunction,Tuple[Literal["gaussian", "blackman"], float]]]):
            Window function to apply to the wavelet.
        causal (bool): Whether to make the wavelet causal.
        times (Optional[np.ndarray]): Time points to sample the wavelet at. If None, the wavelet
            will be generated when needed.
    """

    def __init__(
        self,
        f: Union[float, List[float]],
        center: Optional[float] = None,
        window: Optional[
            Union[WindowFunction, Tuple[Literal["gaussian", "blackman"], float]]
        ] = None,
        causal: bool = False,
        scale: float = 1.0,
    ):
        f0 = f[0] if isinstance(f, (list, tuple, np.ndarray)) else f
        f0 = float(f0)
        if f0 <= 0.0:
            raise ValueError("Ricker wavelet frequency must be positive")
        if center is None:
            center = 1.0 / f0

        super().__init__(
            f=f0,
            center=float(center),
            window=window,
            causal=causal,
            scale=scale,
        )
        self.__post_init__()

    def __post_init__(self):
        self.f_max = 3 * self.f

    def _generate(self, times: np.ndarray, taper: Callable[[int], np.ndarray]) -> None:
        """Generate the wavelet signal."""

        tau = _zero_phase_times(times)
        arg = (np.pi * float(self.f) * tau) ** 2
        signal = (1.0 - 2.0 * arg) * np.exp(-arg)
        signal = _apply_taper(signal, taper)
        self.signal = _normalize(signal) * self.scale


@dataclass
class OrmsbyWavelet(Wavelet):
    """Ormsby wavelet"""

    def __init__(
        self,
        f: Union[List[float], Tuple[float, float, float, float]],
        center: float = 0.0,
        window: Optional[
            Union[WindowFunction, Tuple[Literal["gaussian", "blackman"], float]]
        ] = None,
        causal: bool = False,
        scale: float = 1.0,
    ):
        super().__init__(
            f=f,
            center=center,
            window=window,
            causal=causal,
            scale=scale,
        )
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization hook."""
        if len(self.f) != 4:
            raise ValueError(
                "Ormsby wavelet requires four frequencies [f1, f2, f3, f4]."
            )
        self.f_max = 1.2 * self.f[-1]

    def _generate(self, times: np.ndarray, taper: Callable[[int], np.ndarray]) -> None:
        """Generate the wavelet signal."""

        if len(times) < 2:
            self.signal = np.zeros_like(times, dtype=float)
            return
        dt = float(times[1] - times[0])
        frequencies = np.fft.rfftfreq(len(times), d=dt)
        spectrum = _trapezoid_spectrum(frequencies, self.f)
        signal = np.fft.irfft(spectrum.astype(np.complex128), n=len(times))
        signal = _apply_taper(signal, taper)
        self.signal = _normalize(signal) * self.scale


@dataclass
class KlauderWavelet(Wavelet):
    """Klauder wavelet"""

    def __init__(
        self,
        f: Union[List[float], Tuple[float, float]],
        center: float = 0.0,
        window: Optional[
            Union[WindowFunction, Tuple[Literal["gaussian", "blackman"], float]]
        ] = None,
        causal: bool = False,
        scale: float = 1.0,
    ):
        super().__init__(
            f=f,
            center=center,
            window=window,
            causal=causal,
            scale=scale,
        )
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization hook."""
        if len(self.f) != 2:
            raise ValueError("Klauder wavelet requires two frequencies [f0, f1].")

        self.f_max = 1.2 * self.f[-1]

    def _generate(self, times: np.ndarray, taper: Callable[[int], np.ndarray]) -> None:
        """Generate the wavelet signal."""

        if len(times) < 2:
            self.signal = np.zeros_like(times, dtype=float)
            return
        dt = float(times[1] - times[0])
        duration = max(float(times[-1] - times[0]), dt)
        t = np.arange(len(times), dtype=float) * dt
        sweep = chirp(
            t,
            f0=float(self.f[0]),
            t1=duration,
            f1=float(self.f[1]),
            method="linear",
        )
        sweep = _apply_taper(sweep, taper)
        signal = correlate(sweep, sweep, mode="same", method="auto")
        signal = np.roll(signal, -len(signal) // 2)
        self.signal = _normalize(signal) * self.scale
