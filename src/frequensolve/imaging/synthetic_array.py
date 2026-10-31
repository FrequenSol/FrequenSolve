# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Declarative synthetic arrays over existing receiver channels."""

from dataclasses import dataclass

import numpy as np

from .receiver_pool import ReceiverPool


@dataclass(frozen=True)
class SyntheticArray:
    """Overlapping spatial arrays used to estimate receiver attributes.

    Use :meth:`gaussian`, :meth:`hann` or :meth:`uniform`. Length quantities
    are required by these constructors. Gaussian width is its standard
    deviation; Hann and uniform widths are finite support radii. Arrays pool
    energy and cross-products, not coherent waveforms or completed attributes.
    They do not add physical receiver nodes or change acquisition devices.
    """

    kind: str
    length_m: float
    truncate: float = 3.0

    def __post_init__(self):
        if self.kind not in {"gaussian", "hann", "uniform"}:
            raise ValueError("synthetic array kind must be gaussian, hann or uniform")
        for name in ("length_m", "truncate"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, complex))
                or np.ndim(value) != 0
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, float(value))
        if self.kind != "gaussian" and self.truncate != 3.0:
            raise ValueError("truncate is only supported by Gaussian arrays")

    @staticmethod
    def _length(value):
        if not hasattr(value, "to"):
            raise TypeError("array lengths require units, e.g. fs.Q_(100, 'm')")
        return value.to("m").magnitude

    @classmethod
    def gaussian(cls, *, sigma, truncate=3.0):
        """Gaussian weights exp(-distance² / (2 sigma²)), cut at truncate*sigma."""
        return cls("gaussian", cls._length(sigma), truncate)

    @classmethod
    def hann(cls, *, radius):
        """Raised-cosine weights, falling to zero at radius."""
        return cls("hann", cls._length(radius))

    @classmethod
    def uniform(cls, *, radius):
        """Equal weights within radius, including the boundary."""
        return cls("uniform", cls._length(radius))

    @property
    def radius_m(self):
        return (
            self.length_m * self.truncate if self.kind == "gaussian" else self.length_m
        )

    def to_dict(self):
        """JSON-safe configuration, with explicit metre units."""
        return dict(kind=self.kind, length_m=self.length_m, truncate=self.truncate)

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))

    def bind(self, coordinates, *, groups=None, sample_weights=None, sample_keys=None):
        """Resolve unit-bearing sample coordinates into a reusable sparse map.

        ``groups`` partitions samples, e.g. by shot, component and survey line.
        One-dimensional coordinates can describe distance along a cable.
        """
        if not hasattr(coordinates, "to"):
            raise TypeError("synthetic array coordinates require length units")
        return ReceiverPool.from_coordinates(
            coordinates.to("m").magnitude,
            self.radius_m,
            groups=groups,
            taper=self.kind,
            sigma=self.length_m if self.kind == "gaussian" else None,
            sample_weights=sample_weights,
            sample_keys=sample_keys,
        )
