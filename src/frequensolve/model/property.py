from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.dispersion import DispersionScaling
from frequensolve.util.stochastic_fields import von_karman_stochastic_field

__all__ = ["Property"]


# --------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------
def _dims_compatible(dims1: List[str], dims2: List[str]) -> bool:
    """Helper function to check if two xarrays have compatible grids."""
    dims1 = set(dims1)
    dims2 = set(dims2)
    return dims1 == dims2


# Helper functions
def _dims_in(dims1: List[str], dims2: List[str]) -> bool:
    """Helper function to check if all dimensions in dims1 are in dims2."""
    dims1 = set(dims1)
    dims2 = set(dims2)
    return dims1.issubset(dims2)


def _extend_single_coords(da: xr.DataArray) -> xr.DataArray:
    """Ensure each dimension has at least 2 coordinate points.

    If any dimension has only 1 coordinate point, add another coordinate
    at distance 1 from the existing one and copy the data to the new dimension.

    Args:
        da: Input DataArray

    Returns:
        DataArray with at least 2 coordinate points in each dimension
    """
    if da.ndim == 0:
        return da

    new_coords = {}
    new_data = da.values.copy()
    new_dims = list(da.dims)

    for dim in da.dims:
        coords = da.coords[dim].values
        if len(coords) == 1:
            # Add a second coordinate at distance 1 from the existing one
            new_coord_value = coords[0] + 1.0
            new_coords[dim] = np.array([coords[0], new_coord_value])

            dim_idx = da.dims.index(dim)
            new_shape = list(new_data.shape)
            new_shape[dim_idx] = 2

            expanded_data = np.zeros(new_shape, dtype=new_data.dtype)
            if dim_idx == 0:
                expanded_data[0, ...] = new_data[0, ...]
                expanded_data[1, ...] = new_data[0, ...]
            elif dim_idx == 1:
                expanded_data[:, 0, ...] = new_data[:, 0, ...]
                expanded_data[:, 1, ...] = new_data[:, 0, ...]
            elif dim_idx == 2:
                expanded_data[:, :, 0, ...] = new_data[:, :, 0, ...]
                expanded_data[:, :, 1, ...] = new_data[:, :, 0, ...]
            else:
                slices = [slice(None)] * len(new_data.shape)
                slices[dim_idx] = 0
                slices0 = slices.copy()
                expanded_data[tuple(slices)] = new_data[tuple(slices0)]
                slices[dim_idx] = 1
                expanded_data[tuple(slices)] = new_data[tuple(slices0)]

            new_data = expanded_data
        else:
            new_coords[dim] = coords

    result = xr.DataArray(
        data=new_data, coords=new_coords, dims=new_dims, attrs=da.attrs
    )
    return result


def _coords_compatible(
    coords1: Dict[str, ArrayLike],
    coords2: Dict[str, ArrayLike],
    rtol: float = 1e-06,
    atol: float = 1e-08,
) -> bool:
    """Helper function to check if two xarrays have compatible grids."""
    dims1 = set(coords1.keys())
    dims2 = set(coords2.keys())
    if not _dims_compatible(dims1, dims2):
        return False

    for dim in coords1.keys():
        if dim not in coords2:
            return False
        if len(coords1[dim]) != len(coords2[dim]):
            return False
        if not np.allclose(coords1[dim].values, coords2[dim].values, rtol=rtol):
            return False
    return True


# ----------------------------------------------------------------------------
# Property class
# ----------------------------------------------------------------------------
class Property:
    """Class defining properties.

    Allows flexible definition of properties but rigid serialization
    for compatibility with solver code.
    """

    def __init__(
        self,
        data: Union[int, float, str, Path, xr.DataArray, DispersionScaling] = 0.0,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        if isinstance(data, DispersionScaling):
            self.dispersion = data.dispersion
            data = data.property
        else:
            self.dispersion = None

        if isinstance(data, str):
            data = Path(data)
        if isinstance(data, (int, np.integer)):
            data = float(data)

        # Check if this is a remote file
        self.is_remote = False
        self.remote_path = None
        self.remote_scale = 1.0
        if isinstance(data, Path) and str(data).startswith("remote:"):
            self.is_remote = True
            self.remote_path = str(data).replace("remote:", "")
            self.darr = grid
        elif isinstance(data, (float, np.floating)):
            self.darr = xr.DataArray(data=data)
        elif isinstance(data, Path):
            self.darr = Property.read(data.resolve(), grid=grid)
        elif isinstance(data, xr.DataArray):
            self.darr = data
        else:
            raise ValueError(f"Unknown property type: {type(data)}")

        # Ensure each dimension has at least 2 coordinate points
        if not self.is_constant and not self.is_remote:
            self.darr = _extend_single_coords(self.darr)

        if scale != 1.0 and not self.is_remote:
            self.darr.values = self.darr.values * scale
        elif self.is_remote:
            self.remote_scale = scale

    @property
    def is_constant(self) -> bool:
        """Check if the property is constant."""
        if self.is_remote:
            return False
        return len(self.darr.coords) == 0

    @property
    def extrema(self):
        """Get the extreme values of the property."""
        if self.is_remote:
            raise ValueError(
                "Cannot get extrema of remote property without loading data"
            )
        min = self.darr.min(skipna=True).compute()
        max = self.darr.max(skipna=True).compute()
        return min, max

    @property
    def grid(self) -> CartesianGrid:
        """Get the grid of the property."""
        return CartesianGrid.from_xarray(self.darr)

    def get(self, grid: Optional[xr.DataArray] = None):
        """Get a property from the dataset.

        If the property is stored in the dataset attributes, a DataArray is created
        with the same dimensions and coordinates as the dataset.
        """
        if self.is_remote:
            raise ValueError(
                f"Cannot access data from remote property: {self.remote_path}"
            )

        # If coords not provided, return the property
        if grid is None:
            if self.is_constant:
                return self.darr.values
            else:
                coords = self.darr.coords
        else:
            coords = grid.coords

        # If coordinates match, return the property
        if _coords_compatible(coords, self.darr.coords):
            result = self.darr
        # Otherwise, interpolate the property onto coords
        else:
            if _dims_compatible(self.darr.dims, coords):
                # Linear interpolation for valid values
                out = self.darr.interp(coords=coords, method="linear")
                # Use nearest neighbor interpolation to fill NaNs
                if np.isnan(out.values).any():
                    nearest_interp = self.darr.interp(
                        coords=coords,
                        method="nearest",
                        kwargs={"fill_value": "extrapolate"},
                    )
                    nan_mask = np.isnan(out.values)
                    out.values[nan_mask] = nearest_interp.values[nan_mask]
                result = out

            elif self.is_constant:
                dims = grid.dims
                shape = tuple(len(coords[dim]) for dim in dims)
                result = xr.DataArray(
                    data=np.full(shape, self.darr.values), dims=dims, coords=coords
                )
            else:
                raise ValueError(
                    f"Incompatible dimensions: {coords} != {self.darr.dims}\n"
                    f"Note that in 2D the dimensions should be x and z, this is a point of confusion."
                )

        # Ensure the result has at least 2 coordinate points in each dimension
        if not self.is_constant and result.ndim > 0:
            result = _extend_single_coords(result)

        return result

    def __iadd__(self, other: Union[float, xr.DataArray]) -> None:
        """Add a scalar or DataArray to the property."""
        if self.is_remote:
            raise ValueError(
                f"Cannot perform addition on remote property: {self.remote_path}"
            )

        if isinstance(other, float):
            self.darr = self.darr + other
        elif isinstance(other, xr.DataArray):
            if self.is_constant:
                self.darr = self.darr + other
            else:
                if _coords_compatible(self.darr.coords, other.coords):
                    self.darr = self.darr + other
                else:
                    self.darr = self.darr.interp(other.coords) + other
        else:
            raise ValueError(f"Unknown type for addition: {type(other)}")

    def __add__(self, other: Union[float, xr.DataArray]) -> None:
        """Add a scalar or DataArray to the property."""
        return self.__iadd__(other)

    def write(self, file: Path):
        """Write the property to a file."""
        if self.is_remote:
            raise ValueError(f"Cannot write to remote property: {self.remote_path}")

        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.darr.values.astype(np.single).tofile(file)
        return file

    @staticmethod
    def read(file: Path, grid: Optional[xr.DataArray] = None) -> xr.DataArray:
        """Read the property from file."""
        reader = Property._get_reader(file)
        return reader(file, grid=grid)

    @staticmethod
    def _get_reader(file: Path) -> Callable:
        """Get the reader for the file."""

        if not file.exists():
            raise FileNotFoundError(f"File {file} not found")

        if file.suffix == ".bin":
            return Property._bin_reader
        if file.suffix == ".sgy" or file.suffix == ".segy":
            return Property._segy_reader
        elif file.suffix == ".h5":
            return Property._h5_reader
        elif file.suffix == ".zarr":
            return Property._zarr_reader
        elif file.suffix == ".nc":
            return Property._netcdf_reader
        elif file.suffix == "":
            return Property._bin_reader
        else:
            raise ValueError(f"Unknown file format for {file}")

    @staticmethod
    def _bin_reader(file: Path, grid: xr.DataArray) -> xr.DataArray:
        """Read a binary file."""
        dims = sorted(grid.dims)
        grid = grid.transpose(*dims[::-1])
        data = np.fromfile(file, dtype=np.float32).reshape(grid.shape)
        da = xr.DataArray(data, coords=grid.coords, dims=grid.dims)

        da = _extend_single_coords(da)

        dims = sorted(grid.dims)
        da = da.transpose(*dims)
        return da

    @staticmethod
    def _h5_reader(file: Path, **kwargs) -> xr.DataArray:
        """Read an h5 file."""
        import h5py

        fname, dset = file.split(":")

        with h5py.File(fname, "r") as f:
            if "coords" not in f:
                if "grid" not in kwargs:
                    raise ValueError(
                        "Coords not found in h5 file, must be provided via 'coords' keyword argument"
                    )
                grid = kwargs["grid"]
                coords = grid.coords
                dims = grid.dims
            else:
                dims = f["coords"].attrs["dims"]
                coords = {dim: f["coords"][dim][()] for dim in dims}
            return xr.DataArray(f[dset], coords=coords, dims=dims)

    @staticmethod
    def _netcdf_reader(file: Path, **kwargs) -> xr.DataArray:
        """Read a netcdf file."""
        grid = kwargs.pop("grid", None)
        ds = xr.open_dataset(file, **kwargs)
        da = ds[list(ds.data_vars)[0]]

        if grid is not None:
            return da.interp(coords=grid.coords)
        else:
            return da

    @staticmethod
    def _segy_reader(file: Path, **kwargs) -> xr.DataArray:
        """Read a SEGY file."""
        import segyio

        with segyio.open(file, mode="r", strict=False) as sgy:

            # TODO: Improve how L is defined (ideally it would be defined in the segy, but it's often not)
            scale = kwargs.get("scale", 1.0)
            L = kwargs.get("L", 4.5)

            dims = ["x", "z"]
            coords = {}
            nx = sgy.tracecount
            nz = len(sgy.samples)
            coords["z"] = sgy.samples / 1000.0
            coords["z"] -= coords["z"][0]
            coords["x"] = np.linspace(0, L, nx)
            # for i in range(sgy.tracecount):
            #     coords["x"][i] = sgy.bin[segyio.BinField.SourceX]

            da = xr.DataArray(dims=dims, coords=coords, data=np.zeros((nx, nz)))
            for i in range(sgy.tracecount):
                data = np.array(sgy.trace[i].data[:], dtype=np.float32)
                da.values[i, :] = data * scale

        dims = sorted(da.dims)
        da = da.transpose(*dims)
        return da

    @staticmethod
    def _zarr_reader(file: Path, **kwargs) -> xr.DataArray:
        """Read a zarr file."""
        if "grid" in kwargs:
            kwargs.remove("grid")
        return xr.open_zarr(file, **kwargs)

    def _mask(self, mask: xr.DataArray) -> None:
        """Mask property."""
        if self.is_remote:
            raise ValueError(f"Cannot mask remote property: {self.remote_path}")

        if not self.is_constant:
            self.darr = self.darr.where(mask)

    def _like(self, da: xr.DataArray) -> xr.DataArray:
        """Interpolate properties onto a new grid."""
        # if _dims_in(da.dims, self.darr.dims):
        #    return self.darr.interp(coords=da.coords)
        # else:
        if self.is_remote:
            raise ValueError(f"Cannot interpolate remote property: {self.remote_path}")

        dims1 = set(self.darr.dims)
        dims2 = set(da.dims)
        dims = dims1.intersection(dims2)
        coords = {dim: da.coords[dim] for dim in dims}
        return self.darr.interp(
            coords=coords, method="nearest", kwargs={"fill_value": "extrapolate"}
        ).broadcast_like(da)

    def stochastic_perturbation(
        self,
        std: float,
        method: str,
        type: str,
        grid: Optional[xr.DataArray] = None,
        **kwargs,
    ) -> None:
        """Perturb the dataset by a given factor.

        std (float):
           Standard deviation of the perturbation
        method (str):
           Stochasticperturbation method
        grid (xr.DataArray):
           Xarray defining the grid for the perturbation
        kwargs (Dict[str, Any]):
           Arguments to the perturbation method.
           For method == "von_karman":
              k0 (float):
                 Stochastic field correlation length
              nu (float):
                 Stochastic field smoothness parameter
                 (nu -> 0: less smooth, nu -> 1 more smoother)
              anisotropy (List[float]):
                 Anisotropic stretching factor for each dimension
              seed (int):
                 Random seed (for reproducibility)
        """
        if self.is_remote:
            raise ValueError(
                f"Cannot perform stochastic perturbation on remote property: {self.remote_path}"
            )

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        if method == "von_karman":
            k0 = kwargs.get("k0", [1.0])
            nu = kwargs.get("nu", 0.5)
            anisotropy = kwargs.get("anisotropy", [1.0] * len(self.darr.dims))
            seed = kwargs.get("seed", None)
            mean = 0.0

            if grid is None:
                grid = self.darr

            da = von_karman_stochastic_field(grid, mean, std, k0, nu, anisotropy, seed)

            if _coords_compatible(self.darr.coords, grid.coords):
                if type == "additive":
                    self.darr += da
                elif type == "multiplicative":
                    self.darr *= 1 + da
            else:
                if type == "additive":
                    self.darr = self._like(grid) + da
                elif type == "multiplicative":
                    self.darr = self._like(grid) * (1 + da)
        else:
            raise ValueError(f"Unknown perturbation method: {method}")
