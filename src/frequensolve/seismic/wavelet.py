from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal, Optional

import numpy as np
from pylops.utils.wavelets import klauder, ormsby, ricker
from scipy.signal import hilbert
from scipy.stats import norm

__all__ = [
    "TaperFunction",
    "Wavelet",
    "RickerWavelet",
    "OrmsbyWavelet",
    "KlauderWavelet",
]


# ----------------------------------------------------------------------
# TaperFunction
# ----------------------------------------------------------------------
@dataclass
class TaperFunction:
    """Wrapper for specifying a Gaussian taper function

    Attributes:
       sigma (float): The standard deviation of the Gaussian taper.
    """

    sigma: float

    def get(self, n: int) -> np.ndarray:
        """Generate a normalized Gaussian taper of length n.

        Args:
           n (int): Number of points in the taper.

        Returns:
           np.ndarray: Taper array in [0,1].
        """
        vals = norm.pdf(np.linspace(0, 1, n), 0.5, self.sigma)
        mx = np.max(vals)
        vals = vals / mx
        return vals


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

    times: np.ndarray
    signal: np.ndarray

    @property
    def spectrum(self):
        """Compute the frequency-domain representation of the wavelet.

        Returns:
           np.ndarray: The frequency-domain representation of the wavelet.
        """
        spec = np.fft.rfft(self.signal)
        return spec.astype(np.complex64)

    @property
    def frequencies(self):
        """Compute the frequencies for the wavelet.

        Returns:
           np.ndarray: The frequencies for the wavelet.
        """
        n = len(self.signal)
        dt = self.times[1] - self.times[0]
        return np.fft.rfftfreq(n, d=dt).astype(np.float32)

    @staticmethod
    def make_causal(signal: np.ndarray) -> np.ndarray:
        """Convert a wavelet to minimum phase (causal) form using Kolmogorov factorization.

        Args:
            signal: Input time-domain signal to make causal
            times: Time samples corresponding to signal

        Returns:
            Causal (minimum phase) version of input signal
        """
        n = len(signal)

        # Get spectrum and add small constant for stability
        spectrum = np.fft.rfft(signal)
        A = np.abs(spectrum)
        eps = np.max(A) * 1e-2
        Atmp = np.clip(A, eps, np.inf)

        # Compute minimum phase using Kolmogorov method
        lnA = np.log(Atmp)
        phi = np.imag(hilbert(lnA))
        phase_min = np.exp(-1j * phi)

        # Transform back to time domain
        signal_min = np.fft.irfft(phase_min * A, n=n)

        return signal_min

    def plot(self) -> None:
        """Plot the time-domain and spectrum of the wavelet."""
        import matplotlib.pyplot as plt

        # Plot time-domain
        plt.figure()
        plt.title("Wavelet")
        plt.plot(self.times, self.signal)
        plt.xlabel("Time (s)")
        plt.show()

        # Plot frequency-domain
        plt.figure()
        plt.title("Wavelet Spectrum")
        plt.plot(self.frequencies, np.abs(self.spectrum))
        plt.xlabel("Frequency (Hz)")
        plt.show()


@dataclass
class RickerWavelet(Wavelet):
    """Ricker wavelet"""

    def __init__(
        self,
        f0: float,
        times: np.ndarray,
        offset: int = 0,
        sigma: Optional[float] = None,
        causal: bool = False,
    ):
        taper = None
        if sigma is not None:
            taper_func = TaperFunction(sigma=sigma / times[-1])
            taper = lambda n: taper_func.get(n)

        signal, _, center = ricker(times / 2.0, f0=f0, taper=taper)
        if causal:
            signal = Wavelet.make_causal(signal[::2])
            signal = np.roll(signal, shift=offset)
        else:
            signal = np.roll(signal[::2], shift=(-center // 2 + offset))

        super().__init__(times=times, signal=signal)


@dataclass
class OrmsbyWavelet(Wavelet):
    """Ormsby wavelet"""

    def __init__(
        self,
        f: List[float],
        times: np.ndarray,
        offset: int = 0,
        sigma: Optional[float] = None,
        causal: bool = False,
    ):
        if len(f) < 4:
            raise ValueError(
                "Ormsby wavelet requires four frequencies [f0, f1, f2, f3]."
            )

        taper = None
        if sigma is not None:
            taper_func = TaperFunction(sigma=sigma / times[-1])
            taper = lambda n: taper_func.get(n)

        signal, _, center = ormsby(times / 2.0, f=f, taper=taper)
        if causal:
            signal = Wavelet.make_causal(signal[::2])
            signal = np.roll(signal, shift=offset)
        else:
            signal = np.roll(signal[::2], shift=(-center // 2 + offset))

        super().__init__(times=times, signal=signal)


@dataclass
class KlauderWavelet(Wavelet):
    """Klauder wavelet"""

    def __init__(
        self,
        f: List[float],
        times: np.ndarray,
        offset: int = 0,
        sigma: Optional[float] = None,
        causal: bool = False,
    ):
        if len(f) != 2:
            raise ValueError("Klauder wavelet requires two frequencies [f0, f1].")

        taper = None
        if sigma is not None:
            taper_func = TaperFunction(sigma=sigma / times[-1])
            taper = lambda n: taper_func.get(n)

        signal, _, center = klauder(times / 2.0, f=f, taper=taper)
        if causal:
            signal = Wavelet.make_causal(signal[::2])
            signal = np.roll(signal, shift=offset)
        else:
            signal = np.roll(signal[::2], shift=(-center // 2 + offset))

        super().__init__(times=times, signal=signal)
