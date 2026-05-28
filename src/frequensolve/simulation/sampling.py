"""Frequency and time sampling helpers for seismic simulations."""

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["Sampling", "DiscreteSampling", "UniformSweepSampling"]


@dataclass
class Sampling(ABC):
    """Base class for sampling parameters."""

    pass


@dataclass
class DiscreteSampling(Sampling):
    """Sampling parameters for a seismic study at discrete frequencies.

    Attributes:
       freq (List[float]): The frequencies.
    """

    f_list: List[float]

    @property
    def nfreq(self):
        """Number of explicitly requested frequencies."""

        return len(self.freqs)

    def to_fs(self, ctx=None) -> dict:
        """Serialize discrete sampling to the solver payload."""

        return {
            "f_list": self.f_list,
        }


@dataclass
class UniformSweepSampling(Sampling):
    """Sampling parameters for a seismic study at uniform frequency steps.

    Attributes:
       f_min (float): Minimum frequency (Hz).
       f_max (float): Maximum frequency (Hz).
       df (float):    Frequency spacing (Hz).
       upscale (int): Integer multiple for upscaling the time-sampling rate.
    """

    f_min: float
    f_max: float
    df: float
    t_shift: float = 0.0
    upscale: int = 1

    @property
    def t0(self):
        """First time sample after applying the configured time shift."""

        return -self.t_shift

    @property
    def T(self):
        """Base signal period implied by the frequency spacing."""

        return 1 / self.df

    @property
    def ofreq(self):
        """Frequency-index offset for the minimum requested frequency."""

        return round(self.f_min / self.df)

    @property
    def nfreq(self):
        """Number of base frequency samples from zero through ``f_max``."""

        return round(self.f_max / self.df) + 1

    @property
    def ntime(self):
        """Number of base time intervals required by the real FFT grid."""

        return int(2 * (self.nfreq - 1))

    @property
    def t_list(self):
        """Base time samples over one period."""

        return np.linspace(0, self.T, self.ntime + 1)

    @property
    def f_list(self):
        """Base frequency samples from zero through ``f_max``."""

        return np.linspace(0, self.f_max, self.nfreq)

    @property
    def dt(self):
        """Base time-sample interval."""

        return self.T / self.ntime

    # Upscaled
    @property
    def nFreq(self):
        """Upscaled number of frequency samples."""

        return self.upscale * (self.nfreq - 1) + 1

    @property
    def F_list(self):
        """Upscaled frequency samples."""

        return np.linspace(0, self.upscale * self.f_max, self.nFreq)

    @property
    def nTime(self):
        """Upscaled number of time intervals."""

        return int(2 * (self.nFreq - 1))

    @property
    def T_list(self):
        """Upscaled time samples over one period."""

        return np.linspace(0, self.T, self.nTime + 1)

    @property
    def dT(self):
        """Upscaled time-sample interval."""

        return self.T / self.nTime

    def cutoff(self, Tf: Optional[float] = None):
        """Cutoff the time-domain sampling to a specified maximum time.

        Args:
           Tf (float, optional): Maximum time (s). Defaults to None.

        Returns:
           tuple: The number of time samples and the maximum time.
        """
        if Tf:
            Tl = self.T_list
            nTf = np.searchsorted(Tl, Tf + self.t_shift, side="left")
            nTf = np.minimum(nTf + 1, self.nTime)
            return nTf, Tl[nTf] - self.t_shift
        else:
            return self.nTime, self.T

    def to_fs(self, ctx=None) -> dict:
        """Serialize uniform sweep sampling to the solver payload."""

        return {
            "f_min": self.f_min,
            "f_max": self.f_max,
            "df": self.df,
            "upscale": self.upscale,
        }

    @classmethod
    def from_fs(cls, data: dict) -> "Sampling":
        """Deserialize uniform sweep sampling from a solver payload."""

        return cls(
            f_min=data["f_min"],
            f_max=data["f_max"],
            df=data["df"],
            upscale=data.get("upscale", 1),
        )
