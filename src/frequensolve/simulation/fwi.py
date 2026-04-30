"""Elastic FWI problem and PyLops-compatible operator helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import xarray as xr

try:  # Prefer PyLops when it is installed; keep the package importable in lean test envs.
    from pylops import LinearOperator as _LinearOperator
except ImportError:  # pragma: no cover - exercised only when PyLops is unavailable.
    from scipy.sparse.linalg import LinearOperator as _LinearOperator

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.property import canonical_property_name
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.imaging import ImagingJob

__all__ = [
    "DEFAULT_FWI_PARAMETERS",
    "DataSpace",
    "FWIProblem",
    "FWISiteProtocol",
    "FrequenSolveJacobian",
    "ImageSpec",
    "ModelSpace",
    "build_imaging_job",
]


DEFAULT_FWI_PARAMETERS = ("vp", "vs", "rho")
_SOLVER_PROPERTY_LABELS = {"vp": "Vp", "vs": "Vs", "rho": "Rho"}


@runtime_checkable
class FWISiteProtocol(Protocol):
    """Site/solver hooks required for real FrequenSolve FWI execution.

    Implementations must call solver modes that apply the elastic Born
    forward operator, apply the adjoint-gradient contraction, and run a
    nonlinear forward solve with a model perturbation override.
    """

    def apply_fwi_jacobian(
        self, problem: "FWIProblem", model_vector: np.ndarray
    ) -> Any:
        """Apply ``J dm`` and return data-space-compatible residual traces."""
        ...

    def apply_fwi_adjoint(
        self, problem: "FWIProblem", residual_vector: np.ndarray
    ) -> Any:
        """Apply ``J.H r`` and return model-space-compatible gradients."""
        ...

    def apply_fwi_nonlinear_forward(
        self, problem: "FWIProblem", model_vector: np.ndarray
    ) -> Any:
        """Run ``F(m + dm)`` and return data-space-compatible traces."""
        ...


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    return None


def _normalize_parameters(parameters: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if parameters is None:
        parameters = DEFAULT_FWI_PARAMETERS
    out = tuple(canonical_property_name(name) for name in parameters)
    if not out:
        raise ValueError("At least one FWI parameter is required")
    if len(set(out)) != len(out):
        raise ValueError(f"FWI parameters must be unique; got {out}")
    return out


def _infer_frequencies(
    observed: Any = None, frequencies: Optional[Iterable[Any]] = None
) -> List[Any]:
    if frequencies is not None:
        return list(frequencies)
    if hasattr(observed, "f_list"):
        return list(observed.f_list)
    if isinstance(observed, TraceDataset):
        return [
            freq
            for _, freq in sorted(
                observed.metadata["f_map"].items(), key=lambda item: int(item[0])
            )
        ]
    raise ValueError(
        "frequencies must be provided when they cannot be inferred from observed"
    )


def _observed_data_path(observed: Any) -> Path:
    if observed is None:
        raise ValueError("observed data must be provided")
    if hasattr(observed, "trace_path"):
        return Path(observed.trace_path)
    if isinstance(observed, TraceDataset):
        if not observed.files:
            raise ValueError("TraceDataset has no files")
        return Path(observed.files[0]).parent
    path = _as_path(observed)
    if path is not None:
        return path
    raise TypeError(
        "observed must be a SimulationJob, TraceDataset, filesystem path, or path-like string"
    )


def _model_bounds(simulation: Any, dimension: int) -> Tuple[List[float], List[float]]:
    model = simulation.model
    if dimension == 2:
        names = ("x_limits", "z_limits")
    elif dimension == 3:
        names = ("x_limits", "y_limits", "z_limits")
    else:
        raise ValueError(f"Unsupported FWI grid dimension: {dimension}")

    x0 = []
    x1 = []
    missing = []
    for name in names:
        limits = getattr(model, name, None)
        if limits is None:
            missing.append(name)
            continue
        x0.append(float(limits[0]))
        x1.append(float(limits[1]))
    if missing:
        raise ValueError(
            "A resolution-only FWI grid requires model bounds "
            f"{', '.join(names)}; missing {', '.join(missing)}. "
            "Pass a CartesianGrid with explicit x0/x1 instead."
        )
    return x0, x1


def normalize_grid(
    simulation: Any,
    grid: Union[CartesianGrid, xr.DataArray, Mapping[str, Any], Sequence[int]],
) -> CartesianGrid:
    """Normalize user grid input to a ``CartesianGrid``."""

    if isinstance(grid, CartesianGrid):
        return grid
    if isinstance(grid, xr.DataArray):
        return CartesianGrid.from_xarray(grid)
    if isinstance(grid, Mapping):
        data = dict(grid)
        if "_type" in data:
            return CartesianGrid.from_fs(data)
        return CartesianGrid(**data)
    if isinstance(grid, Sequence) and not isinstance(grid, (str, bytes)):
        n = [int(value) for value in grid]
        x0, x1 = _model_bounds(simulation, len(n))
        return CartesianGrid(n=n, x0=x0, x1=x1)
    raise TypeError(
        "grid must be a CartesianGrid, xarray.DataArray, mapping, or resolution sequence"
    )


@dataclass(frozen=True)
class ImageSpec:
    """Typed imaging output specification that serializes to the legacy solver contract."""

    name: str
    condition: str
    property: Optional[str] = None

    @classmethod
    def fwi(cls, parameter: str, name: Optional[str] = None) -> "ImageSpec":
        parameter = canonical_property_name(parameter)
        label = _SOLVER_PROPERTY_LABELS.get(parameter, parameter)
        return cls(name=name or f"FWI_{label}", condition="FWI", property=label)

    @classmethod
    def field(cls, field: str, name: Optional[str] = None) -> "ImageSpec":
        return cls(name=name or str(field), condition=str(field))

    @classmethod
    def condition_image(cls, condition: str, name: Optional[str] = None) -> "ImageSpec":
        return cls(name=name or str(condition), condition=str(condition))

    def to_legacy_value(self) -> str:
        if self.condition == "FWI":
            return f"FWI:{self.property}"
        return self.condition


def _normalize_image_specs(
    *,
    parameters: Optional[Iterable[str]] = None,
    fields: Optional[Iterable[str]] = None,
    condition: Optional[str] = None,
    images: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for parameter in (
        _normalize_parameters(parameters) if parameters is not None else ()
    ):
        spec = ImageSpec.fwi(parameter)
        out[spec.name] = spec.to_legacy_value()
    for field_name in fields or ():
        spec = ImageSpec.field(str(field_name))
        out[spec.name] = spec.to_legacy_value()
    if condition is not None:
        spec = ImageSpec.condition_image(condition)
        out[spec.name] = spec.to_legacy_value()
    for name, value in (images or {}).items():
        if isinstance(value, ImageSpec):
            out[value.name if name is None else name] = value.to_legacy_value()
        else:
            out[str(name)] = str(value)
    if not out:
        for parameter in DEFAULT_FWI_PARAMETERS:
            spec = ImageSpec.fwi(parameter)
            out[spec.name] = spec.to_legacy_value()
    return out


@dataclass
class ModelSpace:
    """Vectorization rules for elastic FWI model perturbations."""

    grid: CartesianGrid
    parameters: Tuple[str, ...] = DEFAULT_FWI_PARAMETERS
    dtype: Any = np.complex128

    def __init__(
        self,
        grid: CartesianGrid,
        parameters: Optional[Iterable[str]] = None,
        dtype: Any = np.complex128,
    ):
        self.grid = grid
        self.parameters = _normalize_parameters(parameters)
        self.dtype = np.dtype(dtype)

    @property
    def template(self) -> xr.DataArray:
        return self.grid.as_xarray()

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.template.dims)

    @property
    def coords(self) -> Dict[str, Any]:
        return {name: self.template.coords[name] for name in self.template.dims}

    @property
    def grid_size(self) -> int:
        return int(np.prod(self.grid.shape))

    @property
    def size(self) -> int:
        return len(self.parameters) * self.grid_size

    @property
    def shape(self) -> Tuple[int]:
        return (self.size,)

    def zeros(self) -> xr.Dataset:
        return self.unpack(np.zeros(self.size, dtype=self.dtype))

    def pack(
        self, model: Union[xr.Dataset, Mapping[str, Any], np.ndarray, Sequence[Any]]
    ) -> np.ndarray:
        if hasattr(model, "raw_images"):
            return self.pack(model.raw_images)
        if isinstance(model, xr.Dataset):
            chunks = []
            for parameter in self.parameters:
                key = self._dataset_parameter_key(model, parameter)
                if key is None:
                    raise KeyError(
                        f"Model perturbation is missing parameter '{parameter}'"
                    )
                chunks.append(self._coerce_parameter(model[key], parameter))
            return np.concatenate(chunks).astype(self.dtype, copy=False)
        if isinstance(model, Mapping):
            return self.pack(xr.Dataset({key: value for key, value in model.items()}))

        vector = np.asarray(model, dtype=self.dtype).reshape(-1)
        if vector.size != self.size:
            raise ValueError(
                f"Model vector has size {vector.size}; expected {self.size}"
            )
        return vector

    def _dataset_parameter_key(
        self, dataset: xr.Dataset, parameter: str
    ) -> Optional[str]:
        if parameter in dataset:
            return parameter
        label = _SOLVER_PROPERTY_LABELS.get(parameter, parameter)
        for candidate in (label, label.lower(), f"FWI_{label}", f"FWI_{parameter}"):
            if candidate in dataset:
                return candidate
        return None

    def _coerce_parameter(self, data: Any, parameter: str) -> np.ndarray:
        da = (
            data
            if isinstance(data, xr.DataArray)
            else xr.DataArray(data, dims=self.dims)
        )
        missing = [dim for dim in self.dims if dim not in da.dims]
        if missing:
            raise ValueError(f"Parameter '{parameter}' is missing dimensions {missing}")
        da = da.transpose(*self.dims)
        if tuple(da.shape) != tuple(self.grid.shape):
            raise ValueError(
                f"Parameter '{parameter}' has shape {da.shape}; expected {self.grid.shape}"
            )
        return np.asarray(da.data).reshape(-1)

    def unpack(self, vector: Union[np.ndarray, Sequence[Any]]) -> xr.Dataset:
        vector = np.asarray(vector, dtype=self.dtype).reshape(-1)
        if vector.size != self.size:
            raise ValueError(
                f"Model vector has size {vector.size}; expected {self.size}"
            )

        data_vars = {}
        offset = 0
        for parameter in self.parameters:
            values = vector[offset : offset + self.grid_size].reshape(self.grid.shape)
            data_vars[parameter] = xr.DataArray(
                values,
                dims=self.dims,
                coords=self.coords,
                name=parameter,
            )
            offset += self.grid_size
        return xr.Dataset(data_vars)


@dataclass(frozen=True)
class _DataSegment:
    group: str
    components: Tuple[str, ...]
    sources: Tuple[int, ...]
    receivers: Tuple[int, ...]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (len(self.sources), len(self.components), len(self.receivers))

    def size(self, n_frequencies: int) -> int:
        return n_frequencies * int(np.prod(self.shape))


@dataclass
class DataSpace:
    """Vectorization rules for complex frequency-domain trace data."""

    frequencies: Tuple[Any, ...]
    segments: Tuple[_DataSegment, ...]
    dtype: Any = np.complex128

    def __init__(
        self,
        frequencies: Iterable[Any],
        segments: Iterable[_DataSegment],
        dtype: Any = np.complex128,
    ):
        self.frequencies = tuple(frequencies)
        self.segments = tuple(segments)
        self.dtype = np.dtype(dtype)
        if not self.frequencies:
            raise ValueError("DataSpace requires at least one frequency")
        if not self.segments:
            raise ValueError("DataSpace requires at least one receiver group segment")

    @classmethod
    def from_simulation(
        cls,
        simulation: Any,
        frequencies: Iterable[Any],
        dtype: Any = np.complex128,
    ) -> "DataSpace":
        sources = tuple(range(1, len(simulation.acquisition.source_groups) + 1))
        if not sources:
            raise ValueError("FWI DataSpace requires at least one source group")

        segments = []
        for group in simulation.acquisition.receiver_groups:
            components = tuple(component.name for component in group.device.components)
            if not components:
                raise ValueError(f"Receiver group '{group.name}' has no components")
            receivers = tuple(range(1, int(group.size) + 1))
            segments.append(
                _DataSegment(
                    group=group.name,
                    components=components,
                    sources=sources,
                    receivers=receivers,
                )
            )
        return cls(frequencies=frequencies, segments=segments, dtype=dtype)

    @property
    def size(self) -> int:
        return sum(segment.size(len(self.frequencies)) for segment in self.segments)

    @property
    def shape(self) -> Tuple[int]:
        return (self.size,)

    def zeros(self) -> xr.Dataset:
        return self.unpack(np.zeros(self.size, dtype=self.dtype))

    def pack(
        self, data: Union[xr.Dataset, Mapping[str, Any], np.ndarray, Sequence[Any]]
    ) -> np.ndarray:
        if isinstance(data, TraceDataset):
            return self.pack(self._dataset_from_trace_dataset(data))
        if isinstance(data, xr.Dataset):
            chunks = []
            for segment in self.segments:
                if segment.group not in data:
                    raise KeyError(
                        f"Trace data is missing receiver group '{segment.group}'"
                    )
                chunks.append(self._coerce_group(data[segment.group], segment))
            return np.concatenate(chunks).astype(self.dtype, copy=False)
        if isinstance(data, Mapping):
            chunks = []
            for segment in self.segments:
                if segment.group not in data:
                    raise KeyError(
                        f"Trace data is missing receiver group '{segment.group}'"
                    )
                chunks.append(self._coerce_group(data[segment.group], segment))
            return np.concatenate(chunks).astype(self.dtype, copy=False)

        vector = np.asarray(data, dtype=self.dtype).reshape(-1)
        if vector.size != self.size:
            raise ValueError(
                f"Data vector has size {vector.size}; expected {self.size}"
            )
        return vector

    def _coerce_group(self, data: Any, segment: _DataSegment) -> np.ndarray:
        dims = ("frequency", "source", "component", "receiver")
        expected = (len(self.frequencies), *segment.shape)
        if isinstance(data, xr.DataArray):
            da = data.rename({"shot": "source"}) if "shot" in data.dims else data
            missing = [dim for dim in dims if dim not in da.dims]
            if missing:
                raise ValueError(
                    f"Trace group '{segment.group}' is missing dimensions {missing}"
                )
            values = da.transpose(*dims).data
        else:
            values = data
        values = np.asarray(values, dtype=self.dtype)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Trace group '{segment.group}' has shape {values.shape}; expected {expected}"
            )
        return values.reshape(-1)

    def _dataset_from_trace_dataset(self, traces: TraceDataset) -> xr.Dataset:
        data_vars = {}
        for segment in self.segments:
            values = np.zeros(
                (len(self.frequencies), *segment.shape),
                dtype=self.dtype,
            )
            for isource, source in enumerate(segment.sources):
                for icomp, component in enumerate(segment.components):
                    fd = traces.fd(segment.group, component, source=source)
                    if "frequency" in fd.dims:
                        fd = fd.interp(
                            frequency=list(self.frequencies),
                            kwargs={"fill_value": 0},
                        )
                    if hasattr(fd.data, "compute"):
                        fd_values = fd.data.compute()
                    else:
                        fd_values = fd.data
                    values[:, isource, icomp, :] = np.asarray(fd_values).reshape(
                        len(self.frequencies), len(segment.receivers)
                    )
            data_vars[segment.group] = xr.DataArray(
                values,
                dims=("frequency", "source", "component", "receiver"),
                coords={
                    "frequency": list(self.frequencies),
                    "source": list(segment.sources),
                    "component": list(segment.components),
                    "receiver": list(segment.receivers),
                },
            )
        return xr.Dataset(data_vars)

    def unpack(self, vector: Union[np.ndarray, Sequence[Any]]) -> xr.Dataset:
        vector = np.asarray(vector, dtype=self.dtype).reshape(-1)
        if vector.size != self.size:
            raise ValueError(
                f"Data vector has size {vector.size}; expected {self.size}"
            )

        data_vars = {}
        offset = 0
        for segment in self.segments:
            shape = (len(self.frequencies), *segment.shape)
            n = int(np.prod(shape))
            values = vector[offset : offset + n].reshape(shape)
            data_vars[segment.group] = xr.DataArray(
                values,
                dims=("frequency", "source", "component", "receiver"),
                coords={
                    "frequency": list(self.frequencies),
                    "source": list(segment.sources),
                    "component": list(segment.components),
                    "receiver": list(segment.receivers),
                },
                name=segment.group,
            )
            offset += n
        return xr.Dataset(data_vars)


class FrequenSolveJacobian(_LinearOperator):
    """PyLops-compatible linearized FrequenSolve FWI operator."""

    def __init__(self, problem: "FWIProblem"):
        self.problem = problem
        self.shape = (problem.data_space.size, problem.model_space.size)
        self.dtype = np.dtype(np.complex128)
        self.explicit = False
        try:
            super().__init__(dtype=self.dtype, shape=self.shape)
        except TypeError:  # PyLops versions differ; direct attributes are sufficient.
            try:
                super().__init__(
                    dtype=self.dtype, dims=self.shape[1], dimsd=self.shape[0]
                )
            except TypeError:
                pass

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.problem.apply_jacobian(x)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self.problem.apply_adjoint(x)


@dataclass
class FWIProblem:
    """Bound elastic FWI problem with model/data spaces and operator constructors."""

    simulation: Any
    observed: Any
    frequencies: List[Any]
    grid: CartesianGrid
    site: Any = None
    parameters: Tuple[str, ...] = DEFAULT_FWI_PARAMETERS
    model_space: ModelSpace = field(init=False)
    data_space: DataSpace = field(init=False)
    matvec: Optional[Callable[["FWIProblem", np.ndarray], np.ndarray]] = None
    rmatvec: Optional[Callable[["FWIProblem", np.ndarray], np.ndarray]] = None
    nonlinear_forward: Optional[Callable[["FWIProblem", np.ndarray], np.ndarray]] = None

    def __init__(
        self,
        simulation: Any,
        observed: Any,
        frequencies: Optional[Iterable[Any]] = None,
        parameters: Optional[Iterable[str]] = None,
        grid: Union[
            CartesianGrid, xr.DataArray, Mapping[str, Any], Sequence[int], None
        ] = None,
        site: Any = None,
        matvec: Optional[Callable[["FWIProblem", np.ndarray], np.ndarray]] = None,
        rmatvec: Optional[Callable[["FWIProblem", np.ndarray], np.ndarray]] = None,
        nonlinear_forward: Optional[
            Callable[["FWIProblem", np.ndarray], np.ndarray]
        ] = None,
    ):
        if grid is None:
            raise ValueError("FWIProblem requires an inversion grid or resolution")
        self.simulation = simulation
        self.observed = observed
        self.frequencies = _infer_frequencies(
            observed=observed, frequencies=frequencies
        )
        self.parameters = _normalize_parameters(parameters)
        self.grid = normalize_grid(simulation, grid)
        self.site = site
        self.model_space = ModelSpace(self.grid, self.parameters)
        self.data_space = DataSpace.from_simulation(simulation, self.frequencies)
        self.matvec = matvec
        self.rmatvec = rmatvec
        self.nonlinear_forward = nonlinear_forward

    def jacobian(self) -> FrequenSolveJacobian:
        return FrequenSolveJacobian(self)

    def forward_operator(self) -> FrequenSolveJacobian:
        return self.jacobian()

    def adjoint_operator(self):
        """Return ``J.H``. This is the adjoint used by inverse solvers, not a true inverse."""

        return self.jacobian().H

    def inverse_operator(self):
        """Alias for ``adjoint_operator`` for inverse-problem workflows."""

        return self.adjoint_operator()

    def apply_jacobian(
        self, model_perturbation: Union[np.ndarray, xr.Dataset, Mapping[str, Any]]
    ) -> np.ndarray:
        vector = self.model_space.pack(model_perturbation)
        if self.matvec is not None:
            return self.data_space.pack(self.matvec(self, vector))
        if self.site is not None and hasattr(self.site, "apply_fwi_jacobian"):
            return self.data_space.pack(self.site.apply_fwi_jacobian(self, vector))
        raise NotImplementedError(
            "FrequenSolve linearized/Born forward execution is not wired for this site. "
            "Implement Site.apply_fwi_jacobian(problem, model_vector) or pass a matvec callback."
        )

    def apply_adjoint(
        self, residual: Union[np.ndarray, xr.Dataset, Mapping[str, Any]]
    ) -> np.ndarray:
        vector = self.data_space.pack(residual)
        if self.rmatvec is not None:
            return self.model_space.pack(self.rmatvec(self, vector))
        if self.site is not None and hasattr(self.site, "apply_fwi_adjoint"):
            return self.model_space.pack(self.site.apply_fwi_adjoint(self, vector))
        raise NotImplementedError(
            "FrequenSolve adjoint-gradient execution is not wired for this site. "
            "Implement Site.apply_fwi_adjoint(problem, residual_vector) or pass an rmatvec callback."
        )

    def run_nonlinear_forward(
        self, model_perturbation: Union[np.ndarray, xr.Dataset, Mapping[str, Any]]
    ) -> np.ndarray:
        vector = self.model_space.pack(model_perturbation)
        if self.nonlinear_forward is not None:
            return self.data_space.pack(self.nonlinear_forward(self, vector))
        if self.site is not None and hasattr(self.site, "apply_fwi_nonlinear_forward"):
            return self.data_space.pack(
                self.site.apply_fwi_nonlinear_forward(self, vector)
            )
        raise NotImplementedError(
            "FrequenSolve nonlinear forward-with-model-override execution is not wired for this site. "
            "Implement Site.apply_fwi_nonlinear_forward(problem, model_vector) or pass a nonlinear_forward callback."
        )

    def dot_test(
        self,
        model_perturbation: Optional[
            Union[np.ndarray, xr.Dataset, Mapping[str, Any]]
        ] = None,
        residual: Optional[Union[np.ndarray, xr.Dataset, Mapping[str, Any]]] = None,
        seed: int = 0,
        tolerance: float = 1.0e-8,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        if model_perturbation is None:
            model_perturbation = rng.standard_normal(
                self.model_space.size
            ) + 1j * rng.standard_normal(self.model_space.size)
        if residual is None:
            residual = rng.standard_normal(
                self.data_space.size
            ) + 1j * rng.standard_normal(self.data_space.size)

        dm = self.model_space.pack(model_perturbation)
        r = self.data_space.pack(residual)
        jdm = self.jacobian() @ dm
        jhr = self.jacobian().H @ r
        lhs = np.vdot(jdm, r)
        rhs = np.vdot(dm, jhr)
        error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
        return {
            "lhs": lhs,
            "rhs": rhs,
            "relative_error": float(error),
            "passed": bool(error <= tolerance),
        }

    def taylor_test(
        self,
        model_perturbation: Union[np.ndarray, xr.Dataset, Mapping[str, Any]],
        epsilons: Sequence[float] = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3),
    ) -> Dict[str, Any]:
        dm = self.model_space.pack(model_perturbation)
        f0 = self.run_nonlinear_forward(
            np.zeros(self.model_space.size, dtype=np.complex128)
        )
        jdm = self.jacobian() @ dm

        residual_norms = []
        for eps in epsilons:
            fp = self.run_nonlinear_forward(eps * dm)
            residual_norms.append(float(np.linalg.norm(fp - f0 - eps * jdm)))

        rates = []
        for i in range(1, len(epsilons)):
            if residual_norms[i] == 0 or residual_norms[i - 1] == 0:
                rates.append(np.inf)
            else:
                rates.append(
                    float(
                        np.log(residual_norms[i] / residual_norms[i - 1])
                        / np.log(epsilons[i] / epsilons[i - 1])
                    )
                )
        return {
            "epsilons": list(epsilons),
            "residual_norms": residual_norms,
            "rates": rates,
            "passed": bool(len(rates) > 0 and rates[-1] > 1.5),
        }


def build_imaging_job(
    simulation: Any,
    *,
    observed: Any,
    frequencies: Optional[Iterable[Any]] = None,
    parameters: Optional[Iterable[str]] = None,
    grid: Union[
        CartesianGrid, xr.DataArray, Mapping[str, Any], Sequence[int], None
    ] = None,
    fields: Optional[Iterable[str]] = None,
    condition: Optional[str] = None,
    images: Optional[Mapping[str, Any]] = None,
    name: str = "imaging",
    weights: Any = None,
    wavelet: Any = None,
    misfit_norm: str = "L2",
    **kwargs,
) -> ImagingJob:
    if grid is None:
        raise ValueError("imaging requires a grid or resolution")
    if "misfit_type" in kwargs:
        misfit_norm = kwargs.pop("misfit_type")

    cart_grid = normalize_grid(simulation, grid)
    specs = _normalize_image_specs(
        parameters=parameters,
        fields=fields,
        condition=condition,
        images=images,
    )
    return ImagingJob(
        name=name,
        simulation=simulation,
        data_path=_observed_data_path(observed),
        f_list=_infer_frequencies(observed=observed, frequencies=frequencies),
        resolution=list(cart_grid.n),
        grid=cart_grid,
        images=specs,
        weights=weights,
        wavelet=wavelet,
        misfit_norm=misfit_norm,
        **kwargs,
    )
