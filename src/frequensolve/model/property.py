from __future__ import annotations

import copy
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.dispersion import DispersionScaling
from frequensolve.units import is_quantity, quantity_to_fs, unit_expression
from frequensolve.util.mixins import ExportContext, merge_extra
from frequensolve.util.stochastic_fields import von_karman_stochastic_field

__all__ = [
    "Property",
    "PropertyExpression",
    "PropertyMap",
    "canonical_property_name",
    "prop",
    "_ensure_minimum_coordinates",
]


PROPERTY_ALIASES = {
    "vp": "vp",
    "vs": "vs",
    "rho": "rho",
    "density": "rho",
    "qp": "qp",
    "qs": "qs",
    "vadapt": "vadapt",
    "epsilon": "epsilon",
    "gamma": "gamma",
    "delta": "delta",
    "phi": "phi",
    "theta": "theta",
}


def canonical_property_name(name: str) -> str:
    key = str(name).strip()
    normalized = key.lower()
    return PROPERTY_ALIASES.get(normalized, normalized)


def _is_hdf5_locator(value: Any) -> bool:
    text = str(value)
    if ":" not in text:
        return False
    file_part = text.split(":", 1)[0].lower()
    return file_part.endswith(".h5") or file_part.endswith(".hdf5")


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


class PropertyExpression:
    """Algebraic property expression serialized as a solver-friendly AST."""

    def __init__(self, node: Mapping[str, Any]):
        self.node = copy.deepcopy(dict(node))

    @classmethod
    def ref(cls, name: str) -> "PropertyExpression":
        return cls({"ref": canonical_property_name(name)})

    @classmethod
    def value(cls, value: Any) -> "PropertyExpression":
        if is_quantity(value):
            return cls(quantity_to_fs(value))
        if isinstance(value, xr.DataArray):
            if value.size != 1:
                raise ValueError("Expression constants must be scalar values")
            units = value.attrs.get("units")
            payload = {"value": _plain_value(value.values.item())}
            if units is not None:
                payload["units"] = unit_expression(units)
            return cls(payload)
        return cls({"value": _plain_value(value)})

    @classmethod
    def from_value(cls, value: Any) -> "PropertyExpression":
        if isinstance(value, PropertyExpression):
            return value
        if isinstance(value, Mapping):
            payload = copy.deepcopy(dict(value))
            if "expr" in payload:
                return cls.from_value(payload["expr"])
            return cls(_canonicalize_expression_refs(payload))
        return cls.value(value)

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        return copy.deepcopy(self.node)

    def depends_on(self) -> List[str]:
        out: List[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, PropertyExpression):
                node = node.node
            if isinstance(node, Mapping):
                if "ref" in node:
                    name = canonical_property_name(node["ref"])
                    if name not in out:
                        out.append(name)
                if "arg" in node:
                    visit(node["arg"])
                for child in node.get("args", []):
                    visit(child)
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                for child in node:
                    visit(child)

        visit(self.node)
        return out

    def _binary(self, op: str, other: Any) -> "PropertyExpression":
        return PropertyExpression(
            {
                "op": op,
                "args": [self.to_fs(), PropertyExpression.from_value(other).to_fs()],
            }
        )

    def _rbinary(self, op: str, other: Any) -> "PropertyExpression":
        return PropertyExpression(
            {
                "op": op,
                "args": [PropertyExpression.from_value(other).to_fs(), self.to_fs()],
            }
        )

    def __add__(self, other: Any) -> "PropertyExpression":
        return self._binary("add", other)

    def __radd__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("add", other)

    def __sub__(self, other: Any) -> "PropertyExpression":
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("sub", other)

    def __mul__(self, other: Any) -> "PropertyExpression":
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("mul", other)

    def __truediv__(self, other: Any) -> "PropertyExpression":
        return self._binary("div", other)

    def __rtruediv__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("div", other)

    def __pow__(self, other: Any) -> "PropertyExpression":
        return self._binary("pow", other)

    def __rpow__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("pow", other)

    def __neg__(self) -> "PropertyExpression":
        return PropertyExpression({"op": "neg", "arg": self.to_fs()})

    def magnitude(self, units: Any) -> "PropertyExpression":
        return PropertyExpression(
            {
                "op": "magnitude",
                "arg": self.to_fs(),
                "units": unit_expression(units),
            }
        )

    def to(self, units: Any) -> "PropertyExpression":
        return PropertyExpression(
            {
                "op": "convert",
                "arg": self.to_fs(),
                "units": unit_expression(units),
            }
        )

    def __repr__(self) -> str:
        return f"PropertyExpression({self.node!r})"


def prop(name: str) -> PropertyExpression:
    """Reference another material property in a derived property expression."""
    return PropertyExpression.ref(name)


def _canonicalize_expression_refs(node: Any) -> Any:
    if isinstance(node, Mapping):
        payload = {}
        for key, value in node.items():
            if key == "ref":
                payload[key] = canonical_property_name(value)
            elif key in {"arg", "args"}:
                payload[key] = _canonicalize_expression_refs(value)
            else:
                payload[key] = value
        return payload
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return [_canonicalize_expression_refs(item) for item in node]
    return node


class PropertyMap(MutableMapping):
    """Dict-like material property mapping with normalized property values."""

    def __init__(
        self,
        values: Optional[Mapping[str, Any]] = None,
        grid: Optional[xr.DataArray] = None,
    ):
        self._store: Dict[str, Property] = {}
        self.grid = grid
        if values:
            self.update(values)

    def __getitem__(self, key: str) -> "Property":
        return self._store[canonical_property_name(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[canonical_property_name(key)] = Property.from_value(
            value, grid=self.grid
        )

    def __delitem__(self, key: str) -> None:
        del self._store[canonical_property_name(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        try:
            return canonical_property_name(str(key)) in self._store
        except Exception:
            return False

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        file_factory: Optional[Callable[[str, "Property"], Path]] = None,
        dataset_factory: Optional[Callable[[str, "Property"], str]] = None,
    ) -> Dict[str, Any]:
        payload = {}
        for key, prop in self._store.items():
            file = None
            dataset = None
            use_store = ctx is not None and getattr(ctx, "store", None) is not None
            if (
                file_factory is not None
                and not use_store
                and not prop.is_constant
                and prop.darr is not None
            ):
                file = file_factory(key, prop)
            if (
                dataset_factory is not None
                and not prop.is_constant
                and prop.darr is not None
            ):
                dataset = dataset_factory(key, prop)
            payload[key] = prop.to_fs(ctx=ctx, file=file, dataset=dataset)
        return payload


class Property:
    """A solver property specification.

    The object can represent constants, Pint quantities, in-memory arrays,
    loaded local files, or structured file references that are only accessible
    to the solver/server.
    """

    def __init__(
        self,
        data: Union[
            int, float, str, Path, np.ndarray, xr.DataArray, DispersionScaling
        ] = 0.0,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        format: Optional[str] = None,
        absolute: bool = False,
        read: bool = True,
        **kwargs,
    ):
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        self.dispersion = None
        if isinstance(data, DispersionScaling):
            self.dispersion = data.dispersion
            data = data.property

        self.scale = float(scale)
        self.units = unit_expression(units) if units is not None else None
        self.system = system
        self.format = format
        self.absolute = bool(absolute)
        self.extra = dict(kwargs)
        self.file_path: Optional[Path] = None
        self.file_grid = grid
        self.darr: Optional[xr.DataArray] = None
        self.expression: Optional[PropertyExpression] = None
        self.is_remote = False
        self.remote_path = None
        self.remote_scale = self.scale

        if isinstance(data, PropertyExpression):
            self.expression = data
            return

        if is_quantity(data):
            q = quantity_to_fs(data)
            data = q["value"]
            self.units = q["units"]

        if isinstance(data, Mapping):
            other = self.from_value(data, grid=grid)
            self.__dict__.update(other.__dict__)
            return

        if isinstance(data, str) and _is_hdf5_locator(data):
            self.file_path = data
            self.format = self.format or "hdf5"
            self.darr = None
            return

        if isinstance(data, str):
            parsed = self._parse_legacy_string(data)
            if parsed is not None:
                data, parsed_scale, parsed_grid = parsed
                self.scale = parsed_scale
                self.remote_scale = parsed_scale
                if parsed_grid is not None:
                    self.extra.setdefault("dims", parsed_grid)
            data = Path(data) if not str(data).startswith("remote:") else data

        if isinstance(data, Path) and str(data).startswith("remote:"):
            self.is_remote = True
            self.remote_path = str(data).replace("remote:", "", 1)
            self.file_path = Path(self.remote_path)
            self.absolute = True
            self.darr = None
        elif isinstance(data, str) and data.startswith("remote:"):
            self.is_remote = True
            self.remote_path = data.replace("remote:", "", 1)
            self.file_path = Path(self.remote_path)
            self.absolute = True
            self.darr = None
        elif isinstance(data, (int, float, np.integer, np.floating)):
            self.darr = xr.DataArray(data=float(data))
        elif isinstance(data, np.ndarray):
            self.darr = xr.DataArray(data=data)
        elif isinstance(data, list):
            self.darr = xr.DataArray(data=np.asarray(data))
        elif isinstance(data, Path):
            self.file_path = data
            if read:
                self.darr = Property.read(data.resolve(), grid=grid)
                if self.scale != 1.0:
                    self.darr.values = self.darr.values * self.scale
            else:
                self.darr = None
        elif isinstance(data, xr.DataArray):
            self.darr = data
            if self.units is None and "units" in data.attrs:
                self.units = data.attrs["units"]
        else:
            raise ValueError(f"Unknown property type: {type(data)}")

        if self.scale != 1.0 and self.darr is not None and self.file_path is None:
            self.darr.values = self.darr.values * self.scale

    @classmethod
    def file(
        cls,
        path: Union[str, Path],
        *,
        scale: float = 1.0,
        units: Optional[Any] = None,
        grid: Optional[Union[CartesianGrid, xr.DataArray, Mapping[str, Any]]] = None,
        format: Optional[str] = None,
        absolute: bool = False,
        system: Optional[str] = None,
        **extra,
    ) -> "Property":
        data_arg = str(path) if _is_hdf5_locator(path) else Path(path)
        prop = cls(
            data_arg,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
            format=format,
            absolute=absolute,
            read=False,
            **extra,
        )
        prop.file_path = str(path) if _is_hdf5_locator(path) else Path(path)
        prop.is_remote = bool(absolute) or str(path).startswith("remote:")
        prop.remote_path = (
            str(path).replace("remote:", "", 1) if prop.is_remote else None
        )
        return prop

    @classmethod
    def expr(
        cls,
        expression: Any,
        *,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        **extra,
    ) -> "Property":
        prop = cls(0.0, units=units, system=system, **extra)
        prop.darr = None
        prop.file_path = None
        prop.file_grid = None
        prop.is_remote = False
        prop.remote_path = None
        prop.expression = PropertyExpression.from_value(expression)
        return prop

    @classmethod
    def from_value(cls, value: Any, grid: Optional[xr.DataArray] = None) -> "Property":
        if isinstance(value, Property):
            return value
        if isinstance(value, PropertyExpression):
            return cls.expr(value)
        if isinstance(value, Mapping):
            payload = dict(value)
            if "expr" in payload:
                payload.pop("depends_on", None)
                return cls.expr(
                    payload.pop("expr"),
                    units=payload.pop("units", None),
                    system=payload.pop("system", None),
                    **payload,
                )
            if "value" in payload and "file" not in payload:
                return cls(
                    payload.pop("value"),
                    grid=grid,
                    units=payload.pop("units", None),
                    system=payload.pop("system", None),
                    **payload,
                )
            if "file" in payload:
                prop_grid = payload.pop("grid", grid)
                return cls.file(
                    payload.pop("file"),
                    scale=payload.pop("scale", 1.0),
                    units=payload.pop("units", None),
                    grid=prop_grid,
                    format=payload.pop("format", None),
                    absolute=payload.pop("absolute", False),
                    system=payload.pop("system", None),
                    **payload,
                )
        return cls(value, grid=grid)

    @staticmethod
    def _parse_legacy_string(value: str):
        if "|" not in value:
            return None
        parts = value.split("|")
        path = parts[0]
        scale = 1.0
        dims = None
        for part in parts[1:]:
            try:
                scale = float(part)
            except ValueError:
                dims = [d for d in part]
        return path, scale, dims

    @property
    def data(self):
        return self.darr

    @property
    def is_constant(self) -> bool:
        if self.expression is not None or self.is_remote or self.darr is None:
            return False
        return len(self.darr.coords) == 0

    @property
    def extrema(self):
        if self.expression is not None:
            raise ValueError("Cannot get extrema of a derived property expression")
        if self.is_remote or self.darr is None:
            raise ValueError(
                "Cannot get extrema of a file reference without loading data"
            )
        return (
            self.darr.min(skipna=True).compute(),
            self.darr.max(skipna=True).compute(),
        )

    @property
    def grid(self) -> CartesianGrid:
        if isinstance(self.file_grid, CartesianGrid):
            return self.file_grid
        if isinstance(self.file_grid, Mapping):
            return CartesianGrid.from_dict(dict(self.file_grid))
        if isinstance(self.file_grid, xr.DataArray):
            return CartesianGrid.from_xarray(self.file_grid)
        if self.darr is None:
            raise ValueError("File-backed property requires explicit grid metadata")
        return CartesianGrid.from_xarray(self.darr)

    def get(self, grid: Optional[xr.DataArray] = None):
        if self.expression is not None:
            raise ValueError("Cannot access data from a derived property expression")
        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot access data from file property: {self.file_path}")

        if grid is None:
            if self.is_constant:
                return self.darr.values
            coords = self.darr.coords
        else:
            coords = grid.coords

        if _coords_compatible(coords, self.darr.coords):
            result = self.darr
        else:
            if self.is_constant:
                dims = grid.dims
                shape = tuple(len(coords[dim]) for dim in dims)
                result = xr.DataArray(
                    data=np.full(shape, self.darr.values), dims=dims, coords=coords
                )
            elif _dims_compatible(self.darr.dims, coords):
                coords2 = {dim: coords[dim] for dim in self.darr.dims}
                out = self.darr.interp(coords=coords2, method="linear")
                if np.isnan(out.values).any():
                    nearest_interp = self.darr.interp(
                        coords=coords2,
                        method="nearest",
                        kwargs={"fill_value": "extrapolate"},
                    )
                    nan_mask = np.isnan(out.values)
                    out.values[nan_mask] = nearest_interp.values[nan_mask]
                if len(coords.keys()) != len(out.coords.keys()):
                    out = out.broadcast_like(grid)
                result = out
            else:
                raise ValueError(
                    f"Incompatible dimensions: {coords} != {self.darr.dims}\n"
                    "Note that in 2D the dimensions should be x and z."
                )
        return result

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        file: Optional[Union[str, Path]] = None,
        dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.expression is not None:
            payload = {"expr": self.expression.to_fs(ctx)}
            depends_on = self.expression.depends_on()
            if depends_on:
                payload["depends_on"] = depends_on
        elif self.file_path is not None and self.darr is None:
            payload = self._file_payload(ctx)
        elif self.is_constant:
            value = self.get()
            payload = {"value": value.tolist() if hasattr(value, "tolist") else value}
        else:
            if (
                ctx is not None
                and getattr(ctx, "store", None) is not None
                and dataset is not None
            ):
                attrs = {"fs_kind": "property"}
                if self.units is not None:
                    attrs["units"] = self.units
                if self.system is not None:
                    attrs["system"] = self.system
                ref = ctx.store.put_dataarray(dataset, self.darr, attrs=attrs)
                payload = ref.to_fs()
            elif file is None:
                payload = {"value": self.darr.values.tolist()}
            else:
                path = Path(file)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.write(path)
                rel = ctx.relative_to_project(path) if ctx else path
                payload = {"file": rel}
                payload["grid"] = (
                    self.grid.to_fs()
                    if hasattr(self.grid, "to_fs")
                    else self.grid.__dict__()
                )

        if self.units is not None:
            payload["units"] = self.units
        if self.system is not None:
            payload["system"] = self.system
        return merge_extra(payload, self.extra, "Property")

    def _file_payload(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        path = self.file_path
        if self.is_remote:
            file_value = self.remote_path if self.remote_path is not None else str(path)
        elif isinstance(path, str) and _is_hdf5_locator(path):
            file_value = path
        elif ctx is not None:
            file_value = ctx.relative_to_project(path)
        else:
            file_value = path

        payload: Dict[str, Any] = {"file": file_value}
        if self.format is not None:
            payload["format"] = self.format
        if self.absolute:
            payload["absolute"] = True
        if self.scale != 1.0:
            payload["scale"] = self.scale
        try:
            grid = self.grid
            payload["grid"] = (
                grid.to_fs() if hasattr(grid, "to_fs") else grid.__dict__()
            )
        except ValueError:
            pass
        return payload

    def __iadd__(self, other: Union[float, xr.DataArray]) -> None:
        if self.is_remote or self.darr is None:
            raise ValueError(
                f"Cannot perform addition on file property: {self.file_path}"
            )

        if isinstance(other, float):
            self.darr = self.darr + other
        elif isinstance(other, xr.DataArray):
            if self.is_constant or _coords_compatible(self.darr.coords, other.coords):
                self.darr = self.darr + other
            else:
                self.darr = self.darr.interp(other.coords) + other
        else:
            raise ValueError(f"Unknown type for addition: {type(other)}")

    def __add__(self, other: Union[float, xr.DataArray]) -> None:
        return self.__iadd__(other)

    def write(self, file: Path):
        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot write file property: {self.file_path}")
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.darr.values.astype(np.single).tofile(file)
        return file

    @staticmethod
    def read(file: Path, grid: Optional[xr.DataArray] = None) -> xr.DataArray:
        reader = Property._get_reader(file)
        return reader(file, grid=grid)

    @staticmethod
    def _get_reader(file: Path) -> Callable:
        path = (
            Path(str(file).split(":", 1)[0]) if _is_hdf5_locator(file) else Path(file)
        )
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        suffix = path.suffix
        if suffix == ".bin" or suffix == "":
            return Property._bin_reader
        if suffix in [".sgy", ".segy"]:
            return Property._segy_reader
        if suffix in [".h5", ".hdf5"]:
            return Property._h5_reader
        if suffix == ".zarr":
            return Property._zarr_reader
        if suffix == ".nc":
            return Property._netcdf_reader
        raise ValueError(f"Unknown file format for {file}")

    @staticmethod
    def _bin_reader(file: Path, grid: xr.DataArray) -> xr.DataArray:
        data = np.fromfile(file, dtype=np.float32).reshape(grid.shape)
        return xr.DataArray(data, coords=grid.coords, dims=grid.dims)

    @staticmethod
    def _h5_reader(file: Path, **kwargs) -> xr.DataArray:
        import h5py

        fname, dset = str(file).split(":") if ":" in str(file) else (file, "data")
        with h5py.File(fname, "r") as f:
            dset_obj = f[dset]
            if "dims" in dset_obj.attrs:
                dims = list(dset_obj.attrs["dims"])
                coords = {dim: dset_obj.attrs[dim] for dim in dims}
            elif "coords" not in f:
                if "grid" not in kwargs or kwargs["grid"] is None:
                    raise ValueError(
                        "Coords not found in h5 file, must be provided via 'grid'"
                    )
                grid = kwargs["grid"]
                coords = grid.coords
                dims = grid.dims
            else:
                dims = f["coords"].attrs["dims"]
                coords = {dim: f["coords"][dim][()] for dim in dims}
            return xr.DataArray(dset_obj[()], coords=coords, dims=dims)

    @staticmethod
    def _netcdf_reader(file: Path, **kwargs) -> xr.DataArray:
        grid = kwargs.pop("grid", None)
        ds = xr.open_dataset(file, **kwargs)
        da = ds[list(ds.data_vars)[0]]
        return da.interp(coords=grid.coords) if grid is not None else da

    @staticmethod
    def _segy_reader(file: Path, **kwargs) -> xr.DataArray:
        import segyio

        with segyio.open(file, mode="r", strict=False) as sgy:
            scale = kwargs.get("scale", 1.0)
            L = kwargs.get("L", 4.5)
            dims = ["x", "z"]
            coords = {"z": sgy.samples / 1000.0, "x": np.linspace(0, L, sgy.tracecount)}
            coords["z"] -= coords["z"][0]
            da = xr.DataArray(
                dims=dims,
                coords=coords,
                data=np.zeros((sgy.tracecount, len(sgy.samples))),
            )
            for i in range(sgy.tracecount):
                da.values[i, :] = (
                    np.array(sgy.trace[i].data[:], dtype=np.float32) * scale
                )
        return da.transpose(*sorted(da.dims))

    @staticmethod
    def _zarr_reader(file: Path, **kwargs) -> xr.DataArray:
        kwargs.pop("grid", None)
        return xr.open_zarr(file, **kwargs)

    def _mask(self, mask: xr.DataArray) -> None:
        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot mask file property: {self.file_path}")
        if not self.is_constant:
            self.darr = self.darr.where(mask)

    def _like(self, da: xr.DataArray) -> xr.DataArray:
        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot interpolate file property: {self.file_path}")
        dims = set(self.darr.dims).intersection(set(da.dims))
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
        if self.is_remote or self.darr is None:
            raise ValueError(
                f"Cannot perform stochastic perturbation on file property: {self.file_path}"
            )
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if method != "von_karman":
            raise ValueError(f"Unknown perturbation method: {method}")

        k0 = kwargs.get("k0", [1.0])
        nu = kwargs.get("nu", 0.5)
        anisotropy = kwargs.get("anisotropy", [1.0] * len(self.darr.dims))
        seed = kwargs.get("seed", None)
        grid = grid if grid is not None else self.darr
        da = von_karman_stochastic_field(grid, 0.0, std, k0, nu, anisotropy, seed)

        base = (
            self.darr
            if _coords_compatible(self.darr.coords, grid.coords)
            else self._like(grid)
        )
        if type == "additive":
            self.darr = base + da
        elif type == "multiplicative":
            self.darr = base * (1 + da)


def _ensure_minimum_coordinates(da: xr.DataArray) -> xr.DataArray:
    """Ensure every dimension has at least two coordinates for interpolation."""
    if not da.dims:
        return da
    out = da
    for dim in list(out.dims):
        if len(out.coords[dim]) == 1:
            coord0 = out.coords[dim].values[0]
            coord1 = coord0 + 1
            out = xr.concat([out, out.copy(deep=True)], dim=dim)
            out = out.assign_coords({dim: [coord0, coord1]})
    return out


def _dims_compatible(dims1: List[str], dims2: List[str]) -> bool:
    dims1 = set(dims1)
    dims2 = set(dims2)
    return dims1.issubset(dims2)


def _dims_in(dims1: List[str], dims2: List[str]) -> bool:
    dims1 = set(dims1)
    dims2 = set(dims2)
    return dims1.issubset(dims2)


def _coords_compatible(
    coords1: Dict[str, ArrayLike],
    coords2: Dict[str, ArrayLike],
    rtol: float = 1e-06,
    atol: float = 1e-08,
) -> bool:
    dims1 = set(coords1.keys())
    dims2 = set(coords2.keys())
    if not _dims_compatible(dims1, dims2):
        return False
    for dim in coords1.keys():
        if dim not in coords2:
            return False
        if len(coords1[dim]) != len(coords2[dim]):
            return False
        if not np.allclose(
            coords1[dim].values, coords2[dim].values, rtol=rtol, atol=atol
        ):
            return False
    return True
