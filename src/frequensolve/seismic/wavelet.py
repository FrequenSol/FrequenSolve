import numpy as np

from dataclasses import dataclass
from typing import Optional, List

from pylops.utils.wavelets import ricker, ormsby, klauder
from scipy.stats import norm
from scipy.signal import hilbert

__all__ = ['TaperFunction', 'Wavelet']


# ----------------------------------------------------------------------
# TaperFunction
# ----------------------------------------------------------------------
@dataclass
class TaperFunction:
   """
   @class   TaperFunction
   @brief   Wrapper for specifying a Gaussian taper function
   @details The taper is defined using a normal PDF with mean=0.5 and std=sigma.
   """
   sigma: float

   def get(self, n: int) -> np.ndarray:
      """
      @brief Generate a normalized Gaussian taper of length n.
      @param n  Number of points in the taper.
      @return   Taper array in [0,1].
      """
      vals = norm.pdf(np.linspace(0, 1, n), 0.5, self.sigma)
      mx = np.max(vals)
      vals = vals / mx
      return vals


# TODO: Cleanup causal wavelet
# TODO: Add option for zero-phase wavelet
#
# ----------------------------------------------------------------------
# Wavelet
# ----------------------------------------------------------------------
@dataclass
class Wavelet:
   """
   @class Wavelet
   @brief Data container for wavelets in time and frequency domains.
   """
   times:    np.ndarray
   signal:   np.ndarray

   @classmethod
   def generate(cls,
                kind:   str,
                f_pts:  List[float],
                times:  np.ndarray,
                offset: int = 0,
                sigma:  Optional[float] = None,
                causal: bool = True) -> "Wavelet":
      """
      @brief Generate a wavelet using built-in wavelet functions (Ricker, Ormsby, Klauder).
      @param kind   Wavelet type ('Ricker', 'Ormsby', 'Klauder').
      @param f_pts  Frequencies for the wavelet (depends on the wavelet kind).
      @param times  Time array for final wavelet.
      @param offset Sample shift (in index units) to apply after generation.
      @param sigma  Gaussian taper parameter (optional).
      @return       A Wavelet object.
      """
      
      # Create a taper function if requested
      taper = None
      if sigma is not None:
         taper_func = TaperFunction(sigma = sigma / times[-1])
         taper = lambda n: taper_func.get(n)

      # Generate the wavelet via pylops functions
      if kind == "Ricker":
         if len(f_pts) < 1:
            raise ValueError("Ricker wavelet requires at least one frequency (f_central).")
         signal, tvals, center = ricker(times / 2.0, f0=f_pts[0], taper=taper)

      elif kind == "Ormsby":
         if len(f_pts) < 4:
            raise ValueError("Ormsby wavelet requires four frequencies [f1, f2, f3, f4].")
         signal, tvals, center = ormsby(times / 2.0, f=f_pts, taper=taper)

      elif kind == "Klauder":
         if len(f_pts) < 2:
            raise ValueError("Klauder wavelet requires two frequencies [f1, f2].")
         signal, tvals, center = klauder(times / 2.0, f=f_pts, taper=taper)

      else:
         raise ValueError(f"Unknown wavelet kind: '{kind}'. "
                          "Expected one of 'Ricker', 'Ormsby', or 'Klauder'.")

      signal = np.roll(signal[::2], shift=(-center // 2 + offset))
      signal = signal.astype(np.float32)
      if causal:
         n   = len(signal)
         A   = np.abs(np.fft.rfft(signal))
         lnA = np.log(A)
         phi = np.imag(hilbert(lnA))
         spectrum = A * np.exp(-1j * phi)
         signal   = np.fft.irfft(spectrum, n=n)

      # Build the wavelet object
      return cls(times=times, signal=signal)
      
      
   @property
   def spectrum(self):
      spec = np.fft.rfft(self.signal)
      return spec.astype(np.complex64)
   
   
   @property
   def frequencies(self):
      n  = len(self.signal)
      dt = self.times[1] - self.times[0]
      return np.fft.rfftfreq(n, d=dt).astype(np.float32)


   def plot(self) -> None:
      """
      @brief Plot the time-domain and spectrum of the wavelet.
      """
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
