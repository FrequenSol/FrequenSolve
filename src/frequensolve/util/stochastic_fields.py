from typing import List, Optional, Union

import numpy as np
import xarray as xr

__all__ = ["von_karman_stochastic_field"]


def von_karman_stochastic_field(
    xarr: xr.DataArray,
    mean: float,
    std: float,
    k0: Union[float, List[float]],
    nu: float = 0.5,
    anisotropy: Optional[List[float]] = None,
    seed: Optional[int] = None,
):
    """
    Generates a stochastic field using the von Kármán spectral density.

    Parameters:
       grid (xarray.DataArray):
          Array defining the sampling grid.
       mean (float):
          Mean value of the field.
       std (float):
          Standard deviation of the field.
       k0 (float[0,inf]):
          Correlation wavenumber (characteristic scale).
          If k0 is a list, the Von Karman spectral density is summed over the list.
          (to define a stochastic field with multiple characteristic scales)
       nu (float[0,1]):
          Smoothness parameter. Values close to 1 are smooth,
          values close to 0 are rough.
       anisotropy (List[float]):
          Anisotropic stretching factor for each dimension.
       seed (int):
          Random seed (setting a value fixes the seed for reproducibility).
    """
    from frequensolve.util.fft import get_fft_backend

    fft = get_fft_backend()
    ndim = len(xarr.dims)
    n = np.zeros(ndim, dtype=int)
    L = np.zeros(ndim, dtype=np.single)
    for i, dim in enumerate(xarr.dims):
        n[i] = len(xarr[dim])
        L[i] = xarr[dim][-1] - xarr[dim][0]
    if not isinstance(k0, (list, np.ndarray)):
        k0 = [k0]
    nu = nu
    a = anisotropy
    if a is None or len(a) == 0:
        a = [1.0] * ndim
    if len(a) < ndim:
        for i in range(ndim - len(a)):
            a.append(a[-1])
    a = a
    if seed is not None:
        np.random.seed(seed)

    amplitude = _von_karman_spectral_density(n, L, k0, nu, a).astype(np.single)
    phase = np.exp(2j * np.pi * np.random.random((n))).astype(np.complex64)
    fft_field = amplitude * phase
    field = fft.ifftn(fft_field).real

    field = field - np.average(field)
    std0 = np.std(field)
    field *= std / std0
    field += mean

    return xr.DataArray(field, coords=xarr.coords, dims=xarr.dims)


def _von_karman_spectral_density(n, L, k0, nu, a):
    ndim = len(n)
    kaxes = []
    for i in range(ndim):
        ax = a[i] * np.fft.fftfreq(n[i], d=L[i] / n[i])
        kaxes.append(ax.astype(np.single))
    kgrid = np.meshgrid(*kaxes, indexing="ij")

    k = np.zeros(kgrid[0].shape, dtype=np.single)
    for kc in kgrid:
        k += kc**2
    k = np.sqrt(k)

    density = np.zeros(np.shape(k), dtype=np.single)
    for K0 in k0:
        density += 1 / (1 + (k / K0) ** 2) ** (nu + 1)
    return density
