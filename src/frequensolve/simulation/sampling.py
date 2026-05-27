from abc import ABC
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Sampling", "UniformSweepSampling"]


@dataclass
class Sampling(ABC):
    """Base class for time/frequency sampling parameter objects."""

    pass


@dataclass
class UniformSweepSampling(Sampling):
    """Sampling parameters for a seismic study at uniform frequency steps.

    Args:
        f_min: Minimum modeled frequency in hertz.
        f_max: Maximum modeled frequency in hertz.
        df: Uniform frequency spacing in hertz.
        t_shift: Time shift applied when reporting time-domain samples.
        upscale: Integer multiple for the reconstructed time/frequency grid.
    """

    f_min: float
    f_max: float
    df: float
    t_shift: float = 0.0
    upscale: int = 1

    @property
    def t0(self):
        """Return the first reported time sample after applying ``t_shift``."""

        return -self.t_shift

    @property
    def T(self):
        """Return the base time period implied by ``df``."""

        return 1 / self.df

    @property
    def ofreq(self):
        """Return the index offset of ``f_min`` on the base frequency grid."""

        return round(self.f_min / self.df)

    @property
    def nfreq(self):
        """Return the number of base nonnegative frequency samples."""

        return round(self.f_max / self.df) + 1

    @property
    def ntime(self):
        """Return the number of base time intervals implied by the sweep."""

        return int(2 * (self.nfreq - 1))

    @property
    def t_list(self):
        """Return the base time samples over one period."""

        return np.linspace(0, self.T, self.ntime + 1)

    @property
    def f_list(self):
        """Return the base nonnegative frequency samples."""

        return np.linspace(0, self.f_max, self.nfreq)

    @property
    def dt(self):
        """Return the base time sample spacing."""

        return self.T / self.ntime

    # Upscaled
    @property
    def nFreq(self):
        """Return the number of upscaled nonnegative frequency samples."""

        return self.upscale * (self.nfreq - 1) + 1

    @property
    def F_list(self):
        """Return the upscaled nonnegative frequency samples."""

        return np.linspace(0, self.upscale * self.f_max, self.nFreq)

    @property
    def nTime(self):
        """Return the number of upscaled time intervals."""

        return int(2 * (self.nFreq - 1))

    @property
    def T_list(self):
        """Return the upscaled time samples over one period."""

        return np.linspace(0, self.T, self.nTime + 1)

    @property
    def dT(self):
        """Return the upscaled time sample spacing."""

        return self.T / self.nTime

    def cutoff(self, Tf: Optional[float] = None):
        """Return the truncated time-sampling length for a final time.

        Args:
            Tf: Maximum reported time in seconds. ``None`` keeps the full
                reconstructed period.

        Returns:
            ``(n_samples, final_time)`` for the requested cutoff.
        """
        if Tf:
            Tl = self.T_list
            nTf = np.searchsorted(Tl, Tf + self.t_shift, side="left")
            nTf = np.minimum(nTf + 1, self.nTime)
            return nTf, Tl[nTf] - self.t_shift
        else:
            return self.nTime, self.T

    def to_fs(self, ctx=None) -> dict:
        """Serialize sampling parameters for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible sampling payload.
        """

        return {
            "f_min": self.f_min,
            "f_max": self.f_max,
            "df": self.df,
            "upscale": self.upscale,
        }

    @classmethod
    def from_fs(cls, data: dict) -> "Sampling":
        """Deserialize uniform sweep sampling from solver JSON.

        Args:
            data: Serialized sampling payload.

        Returns:
            ``UniformSweepSampling`` instance.
        """

        return cls(
            f_min=data["f_min"],
            f_max=data["f_max"],
            df=data["df"],
            upscale=data.get("upscale", 1),
        )
