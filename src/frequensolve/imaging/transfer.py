"""Policies for transferring primal model coefficients between material meshes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Optional

from frequensolve.units import is_quantity

from ._artifacts import validate_mesh_transfer

__all__ = ["Transfer"]


@dataclass(frozen=True)
class Transfer:
    """Immutable material transfer policy, shared by problems and stages.

    Use :meth:`l2` for integral projection with optional variational smoothing,
    or :meth:`nodal` for interpolation. Smoothing lengths accept physical length
    quantities; bare numbers are meters. These policies act on model coefficients
    (log updates for log controls), not gradient covectors.
    """

    method: str
    smoothing_length: float = 0.0
    smoothing_wavelengths: Optional[float] = None
    frequency: Optional[float] = None

    def __post_init__(self) -> None:
        length: Any = self.smoothing_length
        if is_quantity(length):
            length = length.to("m").magnitude
        if not isinstance(length, Real):
            raise TypeError(
                "smooth must be a scalar length quantity or a number in meters"
            )
        length = float(length)
        validate_mesh_transfer(self.method, length)
        object.__setattr__(self, "smoothing_length", length)
        if self.smoothing_wavelengths is not None:
            fraction = self.smoothing_wavelengths
            if not isinstance(fraction, Real) or not isfinite(fraction) or fraction < 0:
                raise ValueError(
                    "smooth_wavelengths must be a finite nonnegative fraction"
                )
            if self.method != "l2" or length != 0:
                raise ValueError(
                    "wavelength smoothing requires L2 transfer without fixed-length smoothing"
                )
            object.__setattr__(self, "smoothing_wavelengths", float(fraction))
        if self.frequency is not None:
            if self.smoothing_wavelengths is None:
                raise ValueError("frequency requires smooth_wavelengths")
            frequency = self.frequency
            if is_quantity(frequency):
                frequency = frequency.to("Hz").magnitude
            if (
                not isinstance(frequency, Real)
                or not isfinite(frequency)
                or frequency <= 0
            ):
                raise ValueError("frequency must be a finite positive scalar in Hz")
            object.__setattr__(self, "frequency", float(frequency))

    @classmethod
    def l2(
        cls,
        *,
        smooth: Any = None,
        smooth_wavelengths: Optional[float] = None,
        frequency: Any = None,
    ) -> Transfer:
        """Project onto a finer or coarser mesh; optionally smooth by a length.

        ``Transfer.l2(smooth=100 * u.m)`` and ``Transfer.l2(smooth=100)``
        are equivalent. Alternatively, ``smooth_wavelengths=0.1`` uses a local
        length of 0.1 * accepted wavespeed / frequency. The frequency accepts Hz
        or a frequency quantity and defaults to the target mesh sizing frequency.
        The wavespeed field is frozen for the transfer. Omitting both smoothing
        options gives ordinary L2 projection.
        """
        if smooth is not None and smooth_wavelengths is not None:
            raise ValueError("choose smooth or smooth_wavelengths, not both")
        return cls(
            "l2", 0.0 if smooth is None else smooth, smooth_wavelengths, frequency
        )

    @classmethod
    def nodal(cls) -> Transfer:
        """Interpolate the existing field at the target mesh's independent nodes."""
        return cls("nodal")


def resolve_transfer(
    transfer: Optional[Transfer],
    mesh_transfer: Optional[str],
    mesh_smoothing_length: Optional[float],
) -> Transfer:
    """Normalize either public spelling without silently discarding options."""
    if transfer is not None:
        if not isinstance(transfer, Transfer):
            raise TypeError(
                "transfer must be im.Transfer.l2(...) or im.Transfer.nodal()"
            )
        if mesh_transfer is not None or mesh_smoothing_length is not None:
            raise ValueError(
                "transfer cannot be combined with mesh_transfer or mesh_smoothing_length"
            )
        return transfer
    return Transfer(
        "nodal" if mesh_transfer is None else mesh_transfer,
        0.0 if mesh_smoothing_length is None else mesh_smoothing_length,
    )
