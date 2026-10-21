"""Waveform-only dependencies remain available at their actual point of use."""

import subprocess
import sys

import numpy as np
import pytest


def test_waveform_dependencies_load_only_when_waveforms_are_requested():
    code = """
import sys
import numpy as np
from frequensolve.seismic.wavelet import GaussianWindow, KlauderWavelet, Wavelet
assert 'scipy.signal' not in sys.modules
assert 'scipy.stats' not in sys.modules
times = np.linspace(0, 1, 101)
GaussianWindow(sigma=0.2).get(times, len(times))
assert 'scipy.stats' in sys.modules
wavelet = KlauderWavelet(f=[5, 15])
wavelet._generate(times, None)
assert 'scipy.signal' in sys.modules
result = Wavelet._make_causal(wavelet.signal)
assert result.size == times.size // 2 + 1
assert np.all(np.isfinite(result))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("sigma", [0.1, 0.3])
def test_gaussian_window_retains_normalized_gaussian_values(sigma):
    from frequensolve.seismic.wavelet import GaussianWindow

    times = np.linspace(0, 2, 101)
    expected = np.exp(-0.5 * ((np.linspace(0, 1, 101) - 0.5) / sigma) ** 2)
    actual = GaussianWindow(sigma=sigma).get(times, len(times))
    np.testing.assert_allclose(actual, expected / expected.max(), rtol=1e-14)


@pytest.mark.parametrize("frequencies", [[5, 15], [10, 25]])
def test_klauder_wavelet_retains_scipy_sweep_autocorrelation(frequencies):
    from scipy.signal import chirp, correlate

    from frequensolve.seismic.wavelet import KlauderWavelet

    times = np.linspace(0, 1, 101)
    wavelet = KlauderWavelet(f=frequencies, scale=2.0)
    wavelet._generate(times, None)
    sweep = chirp(times, f0=frequencies[0], t1=1.0, f1=frequencies[1], method="linear")
    expected = np.roll(
        correlate(sweep, sweep, mode="same", method="auto"), -len(times) // 2
    )
    np.testing.assert_allclose(
        wavelet.signal,
        2.0 * expected / np.max(np.abs(expected)),
        rtol=1e-13,
        atol=1e-14,
    )
