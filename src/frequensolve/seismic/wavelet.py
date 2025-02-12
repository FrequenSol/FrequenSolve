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

    @classmethod
    def generate(
        cls,
        kind: str,
        f_pts: List[float],
        times: np.ndarray,
        offset: int = 0,
        sigma: Optional[float] = None,
        causal: bool = True,
    ) -> "Wavelet":
        """Generate a wavelet using built-in wavelet functions (Ricker, Ormsby, Klauder).

        Args:
           kind (str): Wavelet type ('Ricker', 'Ormsby', 'Klauder').
           f_pts (List[float]): Frequencies for the wavelet (depends on the wavelet kind).
           times (np.ndarray): Time array for final wavelet.
           offset (int): Sample shift (in index units) to apply after generation.
           sigma (Optional[float]): Gaussian taper parameter (optional).

        Returns:
           Wavelet: The created Wavelet object.
        """

        # Create a taper function if requested
        taper = None
        if sigma is not None:
            taper_func = TaperFunction(sigma=sigma / times[-1])
            taper = lambda n: taper_func.get(n)

        # Generate the wavelet via pylops functions
        if kind == "Ricker":
            if len(f_pts) < 1:
                raise ValueError(
                    "Ricker wavelet requires at least one frequency (f_central)."
                )
            signal, tvals, center = ricker(times / 2.0, f0=f_pts[0], taper=taper)

        elif kind == "Ormsby":
            if len(f_pts) < 4:
                raise ValueError(
                    "Ormsby wavelet requires four frequencies [f1, f2, f3, f4]."
                )
            signal, tvals, center = ormsby(times / 2.0, f=f_pts, taper=taper)

        elif kind == "Klauder":
            if len(f_pts) < 2:
                raise ValueError("Klauder wavelet requires two frequencies [f1, f2].")
            signal, tvals, center = klauder(times / 2.0, f=f_pts, taper=taper)

        else:
            raise ValueError(
                f"Unknown wavelet kind: '{kind}'. "
                "Expected one of 'Ricker', 'Ormsby', or 'Klauder'."
            )

        signal = np.roll(signal[::2], shift=(-center // 2 + offset))
        signal = signal.astype(np.float32)
        if causal:
            n = len(signal)
            A = np.abs(np.fft.rfft(signal))
            lnA = np.log(A)
            phi = np.imag(hilbert(lnA))
            spectrum = A * np.exp(-1j * phi)
            signal = np.fft.irfft(spectrum, n=n)

        # Build the wavelet object
        return cls(times=times, signal=signal)

    @cached_property
    def spectrum(self):
        """Compute the frequency-domain representation of the wavelet.

        Returns:
           np.ndarray: The frequency-domain representation of the wavelet.
        """
        spec = np.fft.rfft(self.signal)
        return spec.astype(np.complex64)

    @cached_property
    def frequencies(self):
        """Compute the frequencies for the wavelet.

        Returns:
           np.ndarray: The frequencies for the wavelet.
        """
        n = len(self.signal)
        dt = self.times[1] - self.times[0]
        return np.fft.rfftfreq(n, d=dt).astype(np.float32)

    @staticmethod
    def make_causal(signal: np.ndarray, times: np.ndarray) -> np.ndarray:
        n = len(signal)
        A = np.abs(np.fft.rfft(signal))
        lnA = np.log(A)
        phi = np.imag(hilbert(lnA))
        spectrum = A * np.exp(-1j * phi)
        signal = np.fft.irfft(spectrum, n=n)

        return signal

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
        causal: bool = True,
    ):
        taper = None
        if sigma is not None:
            taper_func = TaperFunction(sigma=sigma / times[-1])
            taper = lambda n: taper_func.get(n)

        signal, _, center = ricker(times / 2.0, f0=f0, taper=taper)
        signal = np.roll(signal[::2], shift=(-center // 2 + offset))
        if causal:
            signal = Wavelet.make_causal(signal, times)
        signal = signal.astype(np.float32)

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
        causal: bool = True,
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
        signal = np.roll(signal[::2], shift=(-center // 2 + offset))
        if causal:
            signal = Wavelet.make_causal(signal, times)
        signal = signal.astype(np.float32)

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
        causal: bool = True,
    ):
        if len(f) != 2:
            raise ValueError("Klauder wavelet requires two frequencies [f0, f1].")

        taper = None
        if sigma is not None:
            taper_func = TaperFunction(sigma=sigma / times[-1])
            taper = lambda n: taper_func.get(n)

        signal, _, center = klauder(times / 2.0, f=f, taper=taper)
        signal = np.roll(signal[::2], shift=(-center // 2 + offset))

        if causal:
            signal = Wavelet.make_causal(signal, times)
        signal = signal.astype(np.float32)

        super().__init__(times=times, signal=signal)
