import numpy as np
import numpy.fft as numpy_fft
import pytest

from frequensolve.util.fft import configure_fft, get_fft_backend


def test_fft_backend_can_force_numpy():
    try:
        configure_fft(prefer_pyfftw=False)
        assert get_fft_backend() is numpy_fft
    finally:
        configure_fft()


def test_fast_fft_extra_executes_pyfftw_backend():
    pytest.importorskip("pyfftw")
    try:
        configure_fft(prefer_pyfftw=True, threads=1)
        backend = get_fft_backend()

        assert backend.__name__.startswith("pyfftw")
        values = backend.fft(np.array([1.0, 0.0, -1.0, 0.0]))
        np.testing.assert_allclose(values, numpy_fft.fft([1.0, 0.0, -1.0, 0.0]))
    finally:
        configure_fft()


def test_fft_backend_rejects_invalid_thread_count():
    with pytest.raises(ValueError, match="threads"):
        configure_fft(threads=0)


def test_fft_backend_rejects_invalid_env_threads(monkeypatch):
    try:
        configure_fft()
        monkeypatch.setenv("FS_FFT_THREADS", "many")
        with pytest.raises(ValueError, match="FS_FFT_THREADS"):
            get_fft_backend()
    finally:
        configure_fft()
