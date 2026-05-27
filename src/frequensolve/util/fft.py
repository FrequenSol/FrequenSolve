"""Lazy FFT backend selection.

The SDK keeps pyFFTW optional so authoring/import workflows do not require it, but
numerical paths should still use it automatically when it is installed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from types import ModuleType
from typing import Optional

import numpy.fft as _numpy_fft

__all__ = ["configure_fft", "get_fft_backend"]

_prefer_pyfftw = True
_threads: Optional[int] = None


def configure_fft(*, threads: Optional[int] = None, prefer_pyfftw: bool = True) -> None:
    """Configure the process-local FFT backend.

    Args:
        threads: Optional thread count for pyFFTW. If omitted, ``FS_FFT_THREADS``
            is used when set; otherwise pyFFTW keeps its own default.
        prefer_pyfftw: If true, use pyFFTW when installed. If false, force NumPy.
    """

    if threads is not None and threads < 1:
        raise ValueError("threads must be a positive integer")
    global _prefer_pyfftw, _threads
    _prefer_pyfftw = prefer_pyfftw
    _threads = threads
    _resolve_fft_backend.cache_clear()


def get_fft_backend(
    *, threads: Optional[int] = None, prefer_pyfftw: Optional[bool] = None
) -> ModuleType:
    """Return an FFT module compatible with ``numpy.fft``.

    This imports pyFFTW only when an FFT is actually needed and silently falls
    back to NumPy when pyFFTW is not installed.

    Args:
        threads: Optional pyFFTW thread count for this lookup. When omitted,
            the configured process default or ``FS_FFT_THREADS`` is used.
        prefer_pyfftw: Optional override for whether pyFFTW should be preferred
            over NumPy.

    Returns:
        Module exposing NumPy-compatible FFT functions.

    Raises:
        ValueError: If ``FS_FFT_THREADS`` is set to a non-positive integer.
    """

    selected_preference = _prefer_pyfftw if prefer_pyfftw is None else prefer_pyfftw
    selected_threads = _threads if threads is None else threads
    if selected_threads is None:
        selected_threads = _threads_from_env()
    return _resolve_fft_backend(selected_preference, selected_threads)


def _threads_from_env() -> Optional[int]:
    value = os.environ.get("FS_FFT_THREADS")
    if value is None or value == "":
        return None
    try:
        threads = int(value)
    except ValueError as exc:
        raise ValueError("FS_FFT_THREADS must be a positive integer") from exc
    if threads < 1:
        raise ValueError("FS_FFT_THREADS must be a positive integer")
    return threads


@lru_cache(maxsize=8)
def _resolve_fft_backend(prefer_pyfftw: bool, threads: Optional[int]) -> ModuleType:
    if not prefer_pyfftw:
        return _numpy_fft

    try:
        import pyfftw
        from pyfftw.interfaces import cache, numpy_fft
    except ImportError:
        return _numpy_fft

    cache.enable()
    if threads is not None:
        pyfftw.config.NUM_THREADS = threads
    return numpy_fft
