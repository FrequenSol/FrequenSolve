"""Grid definitions for regular coordinate sampling and wavefield outputs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import xarray as xr

from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.mixins import TypeTaggedMixin

__all__ = ["Grid", "CartesianGrid"]


@register_class
@dataclass
class Grid(TypeTaggedMixin, ABC):
    """Base class for solver-facing coordinate grids."""

    @abstractmethod
    def get_coords(self, slices: Optional[List[slice]] = None) -> np.ndarray:
        """Gets coordinates for all grid points or a subset defined by slices."""
        pass

    @abstractmethod
    def generate_coords(
        self, slices: Optional[List[slice]] = None
    ) -> Generator[np.ndarray, None, None]:
        """Generates coordinates for all grid points or a subset defined by slices.

        Args:
           slices: Optional list of slice objects defining index ranges for each dimension.
              If None, generates coordinates for all points. Length must match grid dimensions.

        Yields:
           np.ndarray: Coordinates for each grid point.
        """
        pass

    @abstractmethod
    def to_fs(self, ctx=None) -> Dict:
        """Convert the grid to a solver payload."""
        pass

    @classmethod
    def from_fs(cls, data: Dict) -> "Grid":
        """Deserialize a registered grid subclass from a solver payload."""

        return cls.dispatch_from_fs(data, class_registry)


# ----------------------------------------------------------------------
# Cartesian Grid
# ----------------------------------------------------------------------
@register_class
@dataclass
class CartesianGrid(Grid):
    """Uniform Cartesian grid used for receivers, properties, and wavefields.

    Specify any two of ``n``, ``x1``, and ``dx`` together with ``x0``; the
    missing quantity is calculated during initialization. The stored axis order
    follows the solver convention, while :attr:`shape` returns the xarray-style
    reversed order used by sampled data arrays.

    Attributes:
        n: Number of points in each dimension.
        x0: Starting coordinates in each dimension.
        x1: Ending coordinates in each dimension.
        dx: Grid spacing in each dimension.
        dims: Coordinate dimension names. Defaults to ``["x"]``, ``["x", "z"]``,
            or ``["x", "y", "z"]`` based on dimensionality.
        units: Coordinate units.
        system: Coordinate-system name for the grid coordinates.
    """

    n: List[int] = field(default_factory=list)
    x0: List[float] = field(default_factory=list)
    x1: List[float] = field(default_factory=list)
    dx: List[float] = field(default_factory=list)
    dims: List[str] = field(default_factory=list)
    units: Optional[str] = None
    system: Optional[str] = None

    def __post_init__(self):
        """Derive missing spacing, endpoint, count, and default dimension names."""

        if len(self.x1) == 0:
            self.x1 = [x0 + (n - 1) * dx for x0, n, dx in zip(self.x0, self.n, self.dx)]
        elif len(self.dx) == 0:
            self.dx = [
                (x1 - x0) / (n - 1) for x0, x1, n in zip(self.x0, self.x1, self.n)
            ]
        elif len(self.n) == 0:
            self.n = [
                int((x1 - x0) / dx + 1) for x0, x1, dx in zip(self.x0, self.x1, self.dx)
            ]

        if len(self.dims) == 0:
            if len(self.n) == 1:
                self.dims = ["x"]
            elif len(self.n) == 2:
                self.dims = ["x", "z"]
            elif len(self.n) == 3:
                self.dims = ["x", "y", "z"]

    def __eq__(self, other):
        g1 = np.array([self.x0, self.x1, self.n])
        g2 = np.array([other.x0, other.x1, other.n])
        return np.allclose(g1, g2)

    def get_coords(self, slices: Optional[List[slice]] = None) -> np.ndarray:
        """Gets coordinates for all grid points or a subset defined by slices.

        Args:
           slices (Optional[List[slice]]): List of slice objects defining index ranges for each dimension.
              If None, returns coordinates for all points. Length must match grid dimensions.

        Returns:
           np.ndarray: Array of coordinate points [x,y] or [x,y,z] for each grid point.
              Returns empty array if grid dimensions are not 2 or 3.
        """
        if len(self.n) == 1:
            return np.array([np.linspace(self.x0[0], self.x1[0], self.n[0])])

        elif len(self.n) == 2:
            x = np.linspace(self.x0[0], self.x1[0], self.n[0])
            y = np.linspace(self.x0[1], self.x1[1], self.n[1])
            if slices is not None:
                x = x[slices[0]]
                y = y[slices[1]]
            return np.array([[y_, x_] for y_ in y for x_ in x])

        elif len(self.n) == 3:
            x = np.linspace(self.x0[0], self.x1[0], self.n[0])
            y = np.linspace(self.x0[1], self.x1[1], self.n[1])
            z = np.linspace(self.x0[2], self.x1[2], self.n[2])
            if slices is not None:
                x = x[slices[0]]
                y = y[slices[1]]
                z = z[slices[2]]
            return np.array([[z_, y_, x_] for z_ in z for y_ in y for x_ in x])
        else:
            raise ValueError("Grid must have 1, 2, or 3 dimensions")

    def as_xarray(self):
        """Return an xarray ``DataArray`` shell carrying this grid's coordinates."""
        from xarray import DataArray

        if len(self.dims) == len(self.n):
            dims = self.dims
        else:
            if len(self.n) == 1:
                dims = ["x"]
            elif len(self.n) == 2:
                dims = ["x", "z"]
            elif len(self.n) == 3:
                dims = ["x", "y", "z"]
            else:
                raise ValueError("Grid must have 1, 2, or 3 dimensions")

        coords = {
            dim: np.linspace(self.x0[i], self.x1[i], self.n[i])
            for i, dim in enumerate(dims)
        }
        return DataArray(dims=dims[::-1], coords=coords)

    @classmethod
    def from_xarray(cls, xarr: "xr.DataArray") -> "CartesianGrid":
        """Build a grid from an xarray object with uniformly spaced coordinates."""

        coords = xarr.coords
        dims = xarr.dims[::-1]

        n = [coords[dim].size for dim in dims]
        x0 = [float(coords[dim].values.min()) for dim in dims]
        x1 = [float(coords[dim].values.max()) for dim in dims]

        grid = cls(n=n, x0=x0, x1=x1, dims=dims)

        for i, dim in enumerate(dims):
            if len(coords[dim]) > 1:
                coords2 = np.linspace(grid.x0[i], grid.x1[i], grid.n[i])
                if not np.allclose(coords2, coords[dim].values):
                    raise ValueError(
                        f"Grid coordinates do not align with xarray coordinates for {dim}"
                    )
        return grid

    @property
    def dimension(self) -> int:
        """Number of coordinate dimensions in the grid."""

        return len(self.n)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Data-array shape corresponding to this grid's dimension order."""

        return tuple(self.n[::-1])

    # TODO: indexing below is confusing to align with fortran definition, should be changed
    def generate_coords(self, slices: Optional[List[slice]] = None):
        """Generates coordinates for all grid points or a subset defined by slices.

        Args:
           slices (Optional[List[slice]]): List of slice objects defining index ranges for each dimension.
              If None, yields coordinates for all points. Length must match grid dimensions.

        Yields:
           Generator[float]: Coordinate point for each grid point.
        """
        if len(self.n) == 1:
            x = np.linspace(self.x0[0], self.x1[0], self.n[0])
            if slices is not None:
                x = x[slices[0]]
            for x_ in x:
                yield [x_]

        elif len(self.n) == 2:
            x = np.linspace(self.x0[0], self.x1[0], self.n[0])
            y = np.linspace(self.x0[1], self.x1[1], self.n[1])
            if slices is not None:
                x = x[slices[0]]
                y = y[slices[1]]

            for y_ in y:
                for x_ in x:
                    yield [y_, x_]

        elif len(self.n) == 3:
            x = np.linspace(self.x0[0], self.x1[0], self.n[0])
            y = np.linspace(self.x0[1], self.x1[1], self.n[1])
            z = np.linspace(self.x0[2], self.x1[2], self.n[2])
            if slices is not None:
                x = x[slices[0]]
                y = y[slices[1]]
                z = z[slices[2]]
            for z_ in z:
                for y_ in y:
                    for x_ in x:
                        yield [z_, y_, x_]
        else:
            raise ValueError("Grid must have 1, 2, or 3 dimensions")

    def to_fs(self, ctx=None) -> Dict:
        """Converts the grid to a dictionary representation.

        Returns:
           Dict: Dictionary containing the grid parameters.
        """
        return {
            "_type": self.__class__.__name__,
            "dims": self.dims,
            "n": self.n,
            "x0": self.x0,
            "x1": self.x1,
            "dx": self.dx,
            **({"units": self.units} if self.units is not None else {}),
            **({"system": self.system} if self.system is not None else {}),
        }

    @classmethod
    def from_fs(cls, data: Dict) -> "CartesianGrid":
        """Creates a CartesianGrid instance from a dictionary.

        Args:
           data (Dict): Dictionary containing the grid parameters.

        Returns:
           CartesianGrid: A new CartesianGrid instance.
        """
        if "dims" in data:
            dims = data["dims"]
        else:
            if len(data["n"]) == 1:
                dims = ["x"]
            elif len(data["n"]) == 2:
                dims = ["x", "z"]
            elif len(data["n"]) == 3:
                dims = ["x", "y", "z"]
        return cls(
            n=data["n"],
            x0=data["x0"],
            x1=data["x1"],
            dx=data.get("dx", []),
            dims=dims,
            units=data.get("units"),
            system=data.get("system"),
        )
