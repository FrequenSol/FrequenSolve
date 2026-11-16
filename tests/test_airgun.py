"""Airgun calibration, monopole conversion, and explicit planar conventions."""

import numpy as np
import pytest

from frequensolve.seismic import AirgunSignature, SampledWavelet
from frequensolve.units import ureg as u
from frequensolve.util.mixins import ExportContext


def signature():
    return AirgunSignature(
        SampledWavelet([0, 1, -0.5, 0], dt=0.01, units="bar*m"), ghost="excluded"
    )


def test_pressure_distance_and_volume_rate_spectra():
    source = signature()
    f = np.array([2.0, 4.0, 8.0])
    z = f - 0.2j
    q = source.volume_rate().at_frequencies(f, laplace=-0.2)
    expected = source.wavelet.at_frequencies(f, laplace=-0.2)
    np.testing.assert_allclose(1025 * 2j * np.pi * z * q / (4 * np.pi), expected)
    h = 1e-5
    fd = (
        source.volume_rate().at_frequencies(f + h, laplace=-0.2)
        - source.volume_rate().at_frequencies(f - h, laplace=-0.2)
    ) / (2 * h)
    np.testing.assert_allclose(
        source.volume_rate().at_frequencies(f, laplace=-0.2, derivative=True),
        fd,
        rtol=1e-9,
    )
    pa = AirgunSignature(
        SampledWavelet([0, 1e5, -5e4, 0], dt=0.01, units="Pa*m"), ghost="excluded"
    )
    np.testing.assert_allclose(
        pa.volume_rate().at_frequencies(f), source.volume_rate().at_frequencies(f)
    )


@pytest.mark.parametrize(
    "measure,expected", [("peak", 10e5), ("peak_to_peak", 10e5 / 1.5)]
)
def test_synthetic_strength(measure, expected):
    source = AirgunSignature.from_wavelet(
        SampledWavelet([0, 2, -1, 0], dt=0.01),
        strength=10 * u.bar * u.m,
        measure=measure,
        ghost="excluded",
    )
    assert max(source.wavelet.samples) == pytest.approx(expected)


def test_acquisition_exports_existing_volume_rate_contract(tmp_path):
    acq = signature().acquisition(
        [[0.2, 0.3]], frequencies=[2, 4], out_of_plane_thickness=100 * u.m
    )
    payload = acq.to_fs(ExportContext(tmp_path))
    geometry = payload["source_geometry"]
    assert geometry["kind"] == "volume_injection"
    # SourceGeometry serializes the physical amplitude through its ordinary path.
    assert geometry["defaults"]["amplitude"]["value"] == pytest.approx(0.01)
    assert payload["source_signature"]["frequency_derivative_dataset"] == "q_f"
    three = signature().acquisition([[0.2, 0.3, 0.4]], frequencies=[2])
    assert three.source_geometry.to_fs()["defaults"]["amplitude"]["value"] == 1


def test_reject_ambiguous_conventions():
    with pytest.raises(ValueError, match="deghost"):
        AirgunSignature(
            SampledWavelet([0, 1, 0], dt=0.1, units="bar*m"), ghost="included"
        )
    with pytest.raises(ValueError, match="thickness"):
        signature().acquisition([[0, 1]], frequencies=[1])
    with pytest.raises(ValueError, match="only valid"):
        signature().acquisition(
            [[0, 1, 2]], frequencies=[1], out_of_plane_thickness=u.m
        )
    with pytest.raises(ValueError, match="nonzero complex frequency"):
        signature().volume_rate().at_frequencies([0])
