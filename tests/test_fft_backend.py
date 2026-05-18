import numpy.fft as numpy_fft
import pytest

from frequensolve.util.fft import configure_fft, get_fft_backend


def test_fft_backend_can_force_numpy():
    try:
        configure_fft(prefer_pyfftw=False)
        assert get_fft_backend() is numpy_fft
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
