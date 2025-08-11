from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
from pylops.utils.wavelets import klauder, ormsby, ricker
from scipy.signal import hilbert
from scipy.stats import norm

# Convert to time domain using FFT
try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.config.NUM_THREADS = 4
except:
    warn("pyfftw not found, using numpy for FFT (slow)")
    import numpy.fft as fft

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

    def get(self, times, n_dummy: int) -> np.ndarray:
        """Generate a normalized Blackman taper of length n, embedded at center of n points.

        Returns:
            np.ndarray: Taper array in [0,1].
        """
        nt = len(times)
        if nt % 2 == 0:
            n = 2 * (nt - 2) + 1
        else:
            n = 2 * (nt - 1) + 1
        dt = (times[-1] - times[0]) / (n - 1)
        N = int(self.T / dt) + 1

        vals = np.zeros(n)
        i1 = n // 2 - N // 2
        i2 = i1 + N
        i1 = max(0, i1)
        i2 = min(n, i2)

        vals[i1:i2] = 1.0 - (
            0.42
            + 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, N))
            + 0.08 * np.cos(4 * np.pi * np.linspace(0, 1, N))
        )
        return vals

    def plot(self, times, **kwargs) -> None:
        """Plot the window function."""
        import matplotlib.pyplot as plt

        plt.plot(times, self.get(times, len(times)), **kwargs)
        plt.show()


# TODO: make ability to define custom wavelets (including from file)


# ----------------------------------------------------------------------
# Wavelet
# ----------------------------------------------------------------------
@dataclass
class Wavelet:
    """Data container for wavelets in time and frequency domains.

    Attributes:
       times (np.ndarray): The time samples.
       signal (np.ndarray): The wavelet signal.
    """

    f: Union[float, List[float]]
    center: float = 0.0
    window: Optional[Tuple[Literal["gaussian", "blackman"], float]] = None
    causal: bool = False
    f_max: float = field(init=False)
    scale: float = 1.0
    _times: np.ndarray = field(default=None, init=False)
    _signal: np.ndarray = field(default=None, init=False)
    _frequencies: np.ndarray = field(default=None, init=False)
    _spectrum: np.ndarray = field(default=None, init=False)

    @property
    def times(self) -> np.ndarray:
        return self._times

    @times.setter
    def times(self, times: np.ndarray) -> None:
        """Set the time samples and invalidate cached properties.

        Args:
            times: The new time samples.
        """

        if self._times is None:
            self._times = times
        else:
            if np.array_equal(times, self._times):
                return
            else:
                self._times = times
                self._spectrum = None
                self._frequencies = None

        taper = Wavelet._get_window_callable(self._times, self.window)
        offset = (
            0 if self.causal else np.searchsorted(self._times, self.center, side="left")
        )
        self._generate(self._times, taper)

        # if self.causal:
        #     self._signal = Wavelet._make_causal(self.signal)
        self._signal = np.roll(self._signal, shift=offset)

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
            self._frequencies = fft.rfftfreq(n, d=dt).astype(np.float32)
        return self._frequencies

    def evaluate(self, times: np.ndarray) -> np.ndarray:
        """Evaluate the wavelet at given times."""

        # Setting times will trigger re-evaluation if needed
        self.times = times
        return self.signal

    def _evaluate_initial(self):
        if self.times is None:
            dt = 1.0 / (20 * self.f_max)
            self.times = np.arange(0.0, 1.0 + dt, dt)

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
        import matplotlib.pyplot as plt

        self._evaluate_initial()

        # Axis limit kwargs
        Tf = kwargs.pop("T_max", None)
        if Tf:
            nTf = np.searchsorted(self.times, Tf, side="left")
            nTf = np.minimum(nTf, len(self.times))
        else:
            nTf = len(self.times)
        f_max = kwargs.pop("f_max", self.f_max)
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
        # plt.title("Signal")
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
            # plt.title("Spectrum")
            plt.plot(
                self.frequencies[:nF], 20 * np.log10(normalized_spectrum[:nF]), **kwargs
            )
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude [dB]")
            plt.grid(True, alpha=0.3)
            plt.yticks(np.arange(-60, 1, 20))
            plt.yticks(np.arange(-60, 1, 10), minor=True)
            plt.ylim(-60, 1)
        else:
            plt.plot(self.frequencies[:nF], np.abs(self.spectrum[:nF]))
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)

        # # Set major x ticks
        # major_xticks = plt.xticks()[0]
        # if len(major_xticks) > 1:
        #     tick_spacing = major_xticks[1] - major_xticks[0]
        #     minor_xticks = np.arange(major_xticks[0], major_xticks[-1] + tick_spacing/2, tick_spacing/2)
        #     plt.xticks(minor_xticks[::2])
        #     plt.xticks(minor_xticks, minor=True)
        # else:
        #     plt.xticks(minor=True)

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


@dataclass
class RickerWavelet(Wavelet):
    """Ricker wavelet.

    Attributes:
        f (float): Central frequency of the Ricker wavelet.
        center (float): Center time of the wavelet.
        window (Optional[Union[WindowFunction,Tuple[Literal["gaussian", "blackman"], float]]]):
            Window function to apply to the wavelet.
        causal (bool): Whether to make the wavelet causal.
        times (Optional[np.ndarray]): Time points to sample the wavelet at. If None, the wavelet
            will be generated when needed.
    """

    def __post_init__(self):
        if isinstance(self.f, (list, tuple, np.ndarray)):
            self.f = self.f[0]
        self.f_max = 3 * self.f

    def _generate(self, times: np.ndarray, taper: Callable[[int], np.ndarray]) -> None:
        """Generate the wavelet signal."""

        # if self.causal:
        #     signal, _, _ = ricker(times, f0=self.f, taper=taper)
        # else:
        signal, _, i_c = ricker(times / 2.0, f0=self.f, taper=taper)
        signal = np.roll(signal[::2], shift=(-i_c // 2))
        self.signal = signal * self.scale


@dataclass
class OrmsbyWavelet(Wavelet):
    """Ormsby wavelet"""

    def __post_init__(self):
        """Post-initialization hook."""
        if len(self.f) != 4:
            raise ValueError(
                "Ormsby wavelet requires four frequencies [f1, f2, f3, f4]."
            )
        self.f_max = 1.2 * self.f[-1]

    def _generate(self, times: np.ndarray, taper: Callable[[int], np.ndarray]) -> None:
        """Generate the wavelet signal."""

        if self.causal:
            signal, _, _ = ormsby(times, f=self.f, taper=taper)
        else:
            signal, _, i_c = ormsby(times / 2.0, f=self.f, taper=taper)
            signal = np.roll(signal[::2], shift=(-i_c // 2))
        self.signal = signal * self.scale


@dataclass
class KlauderWavelet(Wavelet):
    """Klauder wavelet"""

    def __post_init__(self):
        """Post-initialization hook."""
        if len(self.f) != 2:
            raise ValueError("Klauder wavelet requires two frequencies [f0, f1].")

        self.f_max = 1.2 * self.f[-1]

    def _generate(self, times: np.ndarray, taper: Callable[[int], np.ndarray]) -> None:
        """Generate the wavelet signal."""

        # if self.causal:
        #     signal, _, _ = klauder(times, f=self.f, taper=taper)
        # else:
        signal, _, i_c = klauder(times / 2.0, f=self.f, taper=taper)
        signal = np.roll(signal[::2], shift=(-i_c // 2))
        self.signal = signal * self.scale
