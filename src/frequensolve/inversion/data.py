"""Real vector layouts for complex frequency-domain observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["ComplexDataRealifier"]


def _real_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    """Validate one finite real vector without discarding an imaginary part."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} has size {array.size}; expected {size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ComplexDataRealifier:
    """Isometric real-vector representation of complex frequency-domain data.

    Real and imaginary values are interleaved in the final axis, matching the
    pair layout written by Sauce trace datasets. For complex ``z`` and a real
    packed dual ``y``, this map preserves
    ``dot(pack(z), y) == real(vdot(z, unpack(y)))``.
    """

    shape: tuple[int, ...]

    def __init__(self, shape: Sequence[int]):
        normalized = tuple(int(value) for value in shape)
        if not normalized or any(value < 1 for value in normalized):
            raise ValueError("complex data shape must contain positive dimensions")
        object.__setattr__(self, "shape", normalized)

    @classmethod
    def from_data(cls, data: Any) -> "ComplexDataRealifier":
        """Construct a layout from one scalar or array-valued observation."""

        shape = np.asarray(data).shape
        return cls(shape if shape else (1,))

    @property
    def complex_size(self) -> int:
        """Return the number of complex samples represented by the layout."""

        return int(np.prod(self.shape))

    @property
    def real_size(self) -> int:
        """Return the packed real-vector length."""

        return 2 * self.complex_size

    def pack(self, data: Any) -> np.ndarray:
        """Interleave real and imaginary parts into one float64 vector."""

        values = np.asarray(data, dtype=np.complex128)
        if values.shape == () and self.shape == (1,):
            values = values.reshape(1)
        if values.shape != self.shape:
            raise ValueError(
                f"complex data has shape {values.shape}; expected {self.shape}"
            )
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ValueError("complex data must contain only finite values")
        return np.stack((values.real, values.imag), axis=-1).reshape(-1)

    def unpack(self, vector: Any) -> np.ndarray:
        """Restore a complex array from an interleaved real vector."""

        values = _real_vector(vector, size=self.real_size, name="realified data vector")
        pairs = values.reshape((*self.shape, 2))
        return pairs[..., 0] + 1j * pairs[..., 1]

    def pack_jacobian(self, jacobian: Any, model_size: int) -> np.ndarray:
        """Realify a complex data-by-real-model Jacobian without densifying it."""

        model_size = int(model_size)
        if model_size < 1:
            raise ValueError("model size must be positive")
        values = np.asarray(jacobian, dtype=np.complex128)
        expected_size = self.complex_size * model_size
        if values.size != expected_size:
            raise ValueError(
                f"complex Jacobian has {values.size} values; expected {expected_size}"
            )
        values = values.reshape(self.complex_size, model_size)
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ValueError("complex Jacobian must contain only finite values")
        return np.stack((values.real, values.imag), axis=1).reshape(
            self.real_size, model_size
        )
