"""Calibrated, ghost-free airgun signatures mapped to acoustic volume rates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frequensolve.seismic.acquisition import Acquisition

import numpy as np

from frequensolve.seismic.spectra import (
    SampledWavelet,
    TransferFunction,
    _coordinates,
    _finite,
)
from frequensolve.units import ureg as u

__all__ = ["AirgunSignature"]


@dataclass(frozen=True)
class _AirgunVolumeRate(TransferFunction):
    signature: SampledWavelet
    density: float

    def describe(self) -> dict[str, Any]:
        return {
            "type": "AirgunVolumeRate",
            "signature": self.signature.describe(),
            "water_density_kg_m3": self.density,
            "ghost": "excluded",
            "reference_volume_rate_m3_s": 1.0,
            "reference_pressure_distance_Pa_m": 1.0,
        }

    def at_frequencies(
        self, frequencies: Any, *, laplace: Any = 0.0, derivative: bool = False
    ) -> np.ndarray:
        _, z = _coordinates(frequencies, laplace)
        if np.any(z == 0):
            raise ValueError(
                "Airgun volume-rate conversion requires nonzero complex frequency"
            )
        # r*p = rho/(4*pi) dQ/dt; return multipliers of a 1 m^3/s source.
        pressure = self.signature.at_frequencies(frequencies, laplace=laplace)
        value = pressure / z
        if derivative:
            value = (
                self.signature.at_frequencies(
                    frequencies, laplace=laplace, derivative=True
                )
                / z
                - pressure / z**2
            )
        return _finite(4 * np.pi / (self.density * 2j * np.pi) * value)


@dataclass(frozen=True, init=False)
class AirgunSignature:
    """A compact-monopole far-field pressure-distance signature.

    ``wavelet`` carries calibrated samples in pressure*length (e.g. bar*m),
    not chamber pressure. Remove the measurement's propagation delay so its
    clock is source-relative. Times and Fourier normalization are explicit in
    :class:`SampledWavelet`. ``ghost='excluded'`` is required: a measured
    ghosted array signature must be deghosted before monopole conversion.
    This represents an equivalent isotropic source, not array directivity.

    Water density defaults to 1025 kg/m^3 and should match the calibration
    medium. Frequency-domain and frequency-synthesis jobs are supported;
    this does not create a native time-marching source signature.
    """

    wavelet: SampledWavelet
    water_density: float

    def __init__(
        self,
        wavelet: SampledWavelet,
        *,
        ghost: str,
        water_density: Any = 1025 * u.kg / u.m**3,
    ) -> None:
        if not isinstance(wavelet, SampledWavelet):
            raise TypeError(
                "AirgunSignature requires a SampledWavelet with pressure-distance units"
            )
        if ghost != "excluded":
            raise ValueError(
                "AirgunSignature requires ghost='excluded'; deghost measured signatures first"
            )
        density = float(water_density.to("kg/m^3").magnitude)
        if not np.isfinite(density) or density <= 0:
            raise ValueError("water_density must be finite and positive")
        calibrated = wavelet.relative_to(1.0, units="Pa*m")
        if not np.any(calibrated.samples):
            raise ValueError("Airgun signature must be nonzero")
        calibrated = SampledWavelet(
            calibrated.samples,
            dt=wavelet.dt,
            t0=wavelet.t0,
            normalization=wavelet.normalization,
            units="Pa*m",
        )
        object.__setattr__(self, "wavelet", calibrated)
        object.__setattr__(self, "water_density", density)

    @classmethod
    def from_wavelet(
        cls,
        wavelet: SampledWavelet,
        *,
        strength: Any,
        measure: str,
        ghost: str,
        water_density: Any = 1025 * u.kg / u.m**3,
    ) -> AirgunSignature:
        """Calibrate a dimensionless shape by peak magnitude or peak-to-peak strength."""
        if wavelet.units != "1":
            raise ValueError("from_wavelet requires a dimensionless shape")
        amplitude = float(strength.to("Pa*m").magnitude)
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("strength must be finite and positive")
        if measure == "peak":
            scale = np.max(np.abs(wavelet.samples))
        elif measure == "peak_to_peak":
            scale = np.ptp(wavelet.samples)
        else:
            raise ValueError("measure must be peak or peak_to_peak")
        if scale <= 0:
            raise ValueError("Wavelet has zero amplitude under the selected measure")
        return cls(
            SampledWavelet(
                wavelet.samples * amplitude / scale,
                dt=wavelet.dt,
                t0=wavelet.t0,
                normalization=wavelet.normalization,
                units="Pa*m",
            ),
            ghost=ghost,
            water_density=water_density,
        )

    def volume_rate(self) -> TransferFunction:
        """Return the physical spectrum relative to a 1 m^3/s volume source."""
        return _AirgunVolumeRate(self.wavelet, self.water_density)

    def acquisition(
        self,
        coords: Any,
        *,
        frequencies: Any,
        laplace: Any = (0.0,),
        out_of_plane_thickness: Any = None,
    ) -> Acquisition:
        """Create acoustic volume sources with this shared calibrated signature.

        Planar 2D requires an explicit physical out-of-plane thickness: total
        Q is spread over that thickness, not fitted to a 3D trace at one range.
        Three-dimensional sources must omit it. Coordinate values retain the
        usual acquisition coordinate units. Receivers/encoding can be added to
        the returned Acquisition normally.
        """
        from frequensolve.seismic.acquisition import Acquisition
        from frequensolve.seismic.source_signature import SourceSignature
        from frequensolve.seismic.sources import SourceGeometry

        coordinates = np.atleast_2d(getattr(coords, "magnitude", coords))
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
            raise ValueError("coords must contain planar 2D or Cartesian 3D points")
        amplitude = 1.0
        if coordinates.shape[1] == 2:
            if out_of_plane_thickness is None:
                raise ValueError(
                    "Planar 2D airguns require an explicit out_of_plane_thickness"
                )
            thickness = float(out_of_plane_thickness.to("m").magnitude)
            if not np.isfinite(thickness) or thickness <= 0:
                raise ValueError("out_of_plane_thickness must be finite and positive")
            # Sauce volume injection already divides Q by a physical 1 m.
            amplitude /= thickness
        elif out_of_plane_thickness is not None:
            raise ValueError("out_of_plane_thickness is only valid for planar 2D")
        return Acquisition(
            source_geometry=SourceGeometry.points(
                kind="volume_injection",
                coords=coords,
                amplitude=amplitude * u.m**3 / u.s,
            ),
            source_signature=SourceSignature(
                self.volume_rate(), frequencies=frequencies, laplace=laplace
            ),
        )
