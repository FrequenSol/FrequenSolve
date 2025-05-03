from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal, Optional
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
    pyfftw.config.NUM_THREADS = 6  # Or however many threads you want to use
except:
    warn("pyfftw not found, using numpy for FFT (slow)")
    import numpy.fft as fft

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
    f_max: float

    @cached_property
    def spectrum(self):
        """Compute the frequency-domain representation of the wavelet.

        Returns:
           np.ndarray: The frequency-domain representation of the wavelet.
        """
        spec = fft.rfft(self.signal)
        return spec.astype(np.complex64)

    @cached_property
    def frequencies(self):
        """Compute the frequencies for the wavelet.

        Returns:
           np.ndarray: The frequencies for the wavelet.
        """
        n = len(self.signal)
        dt = self.times[1] - self.times[0]
        return fft.rfftfreq(n, d=dt).astype(np.float32)

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

        return signal_min

    def plot(self, **kwargs) -> None:
        """Plot the time-domain and spectrum of the wavelet."""
        import matplotlib.pyplot as plt

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

        # Formatting kwargs
        figsize = kwargs.pop("figsize", (6, 4))
        fontsize = kwargs.pop("fontsize", 12)
        plt.rcParams.update({"font.size": fontsize})

        # Plot time-domain
        plt.figure(figsize=figsize)
        # plt.title("Signal")
        plt.plot(self.times[:nTf], self.signal[:nTf], **kwargs)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.2)
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
        # plt.title("Spectrum")
        plt.plot(self.frequencies[:nF], np.abs(self.spectrum[:nF]), **kwargs)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.2)
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

        super().__init__(times=times, signal=signal, f_max=3 * f0)


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

        super().__init__(times=times, signal=signal, f_max=1.1 * f[-1])


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
            signal, _, center = klauder(times, f=f, taper=taper)
            n = len(signal)
            signal = Wavelet.make_causal(signal)[: n // 2 + 1]
            signal = np.roll(signal, shift=offset)
        else:
            signal = np.roll(signal[::2], shift=(-center // 2 + offset))

        super().__init__(times=times, signal=signal, f_max=1.1 * f[-1])
