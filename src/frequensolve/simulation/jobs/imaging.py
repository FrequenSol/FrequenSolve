"""Imaging-job definitions and readers for solver imaging products."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
import xarray as xr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.representation import (
    CartesianGridRepresentation,
    VariationalSmoothing,
)
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.jobs.artifacts import TraceOutputSpec
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.outputs import JobOutputs, Output
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import value_and_units_to_fs
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import ExportContext

__all__ = [
    "ImagingJob",
    "LSRTMGradientJob",
    "LSRTMNormalJob",
    "HDF5TraceStore",
    "Misfit",
    "MisfitComparison",
    "MisfitGroup",
    "ObservedTraceDerivatives",
    "PreprocessHook",
    "ImageDatabase",
    "extract_frequencies_for_job",
]


def _positive_physical_scalar(value: Any, name: str) -> Any:
    """Validate and serialize one positive scalar with optional units."""

    payload = value_and_units_to_fs(value)
    magnitude = payload.get("value") if isinstance(payload, Mapping) else payload
    values = np.asarray(magnitude)
    if values.ndim != 0 or np.iscomplexobj(values):
        raise ValueError(f"{name} must be a positive real scalar")
    try:
        scalar = float(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return payload


def _finite_fraction(value: Any, name: str, *, allow_zero: bool = False) -> float:
    """Return one finite dimensionless filter fraction."""

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    lower_ok = scalar >= 0.0 if allow_zero else scalar > 0.0
    if not np.isfinite(scalar) or not lower_ok:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return scalar


@dataclass(kw_only=True)
class PreprocessHook:
    """One solver preprocessing hook.

    Use :meth:`trace_weight` to author arbitrary nonnegative residual weights
    with a shape that remains explicit in the solver payload.
    """

    kind: str
    stage: str
    params: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    schema: str = "fs-preprocess-hook-1"
    _weight_values: Optional[np.ndarray] = field(default=None, repr=False)

    _MAX_INLINE_TRACE_WEIGHTS: ClassVar[int] = 256

    @classmethod
    def trace_weight(
        cls,
        weights: Any,
        *,
        layout: Optional[
            Literal[
                "receiver",
                "component_receiver",
                "source_component_receiver",
                "sparse_trace",
            ]
        ] = None,
        name: Optional[str] = None,
    ) -> "PreprocessHook":
        """Create a residual-stage per-trace objective-weight hook.

        Multidimensional arrays infer layouts ``receiver`` (1-D),
        ``component_receiver`` (2-D), or ``source_component_receiver`` (3-D).
        Receiver is always the fastest-varying flattened dimension. Sparse
        trace-catalog weights require ``layout="sparse_trace"`` explicitly.
        """

        values = np.asarray(weights)
        if values.ndim < 1 or values.ndim > 3 or values.size == 0:
            raise ValueError("trace weights must be a nonempty 1-D, 2-D, or 3-D array")
        if np.iscomplexobj(values) and np.any(np.imag(values) != 0):
            raise ValueError("residual trace weights must be real")
        try:
            real_values = np.asarray(np.real(values), dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("residual trace weights must be numeric") from exc
        if not np.all(np.isfinite(real_values)) or np.any(real_values < 0):
            raise ValueError("residual trace weights must be finite and nonnegative")

        inferred = {
            1: "receiver",
            2: "component_receiver",
            3: "source_component_receiver",
        }[real_values.ndim]
        selected = inferred if layout is None else layout
        allowed = {
            "receiver",
            "component_receiver",
            "source_component_receiver",
            "sparse_trace",
        }
        if selected not in allowed:
            raise ValueError(f"unsupported trace-weight layout {selected!r}")
        expected_ndim = {
            "receiver": 1,
            "component_receiver": 2,
            "source_component_receiver": 3,
            "sparse_trace": 1,
        }[selected]
        if real_values.ndim != expected_ndim:
            raise ValueError(
                f"trace-weight layout {selected!r} requires a {expected_ndim}-D array"
            )
        return cls(
            kind="trace_weight",
            stage="residual",
            name=name,
            params={"layout": selected},
            _weight_values=np.array(real_values, dtype=np.float64, copy=True),
        )

    @classmethod
    def receiver_ar1_whiten(
        cls,
        correlation: float,
        *,
        name: Optional[str] = None,
    ) -> "PreprocessHook":
        """Create exact stationary AR(1) whitening along a dense receiver axis."""

        try:
            rho = float(correlation)
        except (TypeError, ValueError) as exc:
            raise TypeError("receiver AR(1) correlation must be numeric") from exc
        if not np.isfinite(rho) or abs(rho) >= 1.0:
            raise ValueError(
                "receiver AR(1) correlation must be finite with absolute value below one"
            )
        return cls(
            kind="receiver_ar1_whiten",
            stage="trace_pair",
            name=name,
            params={"correlation": rho},
        )

    @classmethod
    def scholte_notch(
        cls,
        phase_velocity: Optional[Any] = None,
        *,
        wavenumber: Optional[Any] = None,
        relative_half_width: float = 0.04,
        relative_taper_width: float = 0.04,
        spacing_tolerance: float = 1.0e-3,
        name: Optional[str] = None,
    ) -> "PreprocessHook":
        """Create a smooth receiver-wavenumber notch around a Scholte ridge.

        Supply exactly one ridge description: ``phase_velocity`` predicts the
        frequency-local angular wavenumber as ``2*pi*f/velocity``;
        ``wavenumber`` supplies that angular wavenumber directly. The real,
        symmetric mask is applied along a regularly sampled dense cable.
        """

        if (phase_velocity is None) == (wavenumber is None):
            raise ValueError(
                "scholte_notch requires exactly one of phase_velocity or wavenumber"
            )
        params = {
            "relative_half_width": _finite_fraction(
                relative_half_width,
                "relative_half_width",
                allow_zero=True,
            ),
            "relative_taper_width": _finite_fraction(
                relative_taper_width,
                "relative_taper_width",
            ),
            "spacing_tolerance": _finite_fraction(
                spacing_tolerance,
                "spacing_tolerance",
            ),
        }
        if params["spacing_tolerance"] >= 1.0:
            raise ValueError("spacing_tolerance must be less than one")
        if phase_velocity is not None:
            params["phase_velocity"] = _positive_physical_scalar(
                phase_velocity,
                "phase_velocity",
            )
        else:
            params["wavenumber"] = _positive_physical_scalar(
                wavenumber,
                "wavenumber",
            )
        return cls(
            kind="scholte_notch",
            stage="trace_pair",
            name=name,
            params=params,
        )

    @classmethod
    def slow_velocity_mute(
        cls,
        stop_velocity: Any,
        pass_velocity: Any,
        *,
        mode: Literal["reject_slow", "keep_slow"] = "reject_slow",
        spacing_tolerance: float = 1.0e-3,
        name: Optional[str] = None,
    ) -> "PreprocessHook":
        """Create a smooth apparent-velocity fan mute along a dense cable.

        ``reject_slow`` is zero below ``stop_velocity`` and one above
        ``pass_velocity`` with a raised-cosine transition. ``keep_slow`` uses
        the complementary mask for paired P/S experiments.
        """

        if mode not in {"reject_slow", "keep_slow"}:
            raise ValueError(f"unsupported slow-velocity mute mode {mode!r}")
        stop = _positive_physical_scalar(stop_velocity, "stop_velocity")
        passed = _positive_physical_scalar(pass_velocity, "pass_velocity")
        stop_value = stop["value"] if isinstance(stop, Mapping) else stop
        pass_value = passed["value"] if isinstance(passed, Mapping) else passed
        stop_units = stop.get("units") if isinstance(stop, Mapping) else None
        pass_units = passed.get("units") if isinstance(passed, Mapping) else None
        if stop_units == pass_units and float(pass_value) <= float(stop_value):
            raise ValueError("pass_velocity must exceed stop_velocity")
        tolerance = _finite_fraction(spacing_tolerance, "spacing_tolerance")
        if tolerance >= 1.0:
            raise ValueError("spacing_tolerance must be less than one")
        return cls(
            kind="slow_velocity_mute",
            stage="trace_pair",
            name=name,
            params={
                "stop_velocity": stop,
                "pass_velocity": passed,
                "mode": mode,
                "spacing_tolerance": tolerance,
            },
        )

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Serialize this hook, materializing authored trace weights when possible."""

        params = dict(self.params)
        if self.kind == "trace_weight" and self._weight_values is not None:
            store = getattr(ctx, "store", None) if ctx is not None else None
            if store is None:
                if self._weight_values.size > self._MAX_INLINE_TRACE_WEIGHTS:
                    raise ValueError(
                        "large trace weights require an export context with an HDF5 store"
                    )
                params["weights"] = self._weight_values.reshape(-1, order="C").tolist()
            else:
                if dataset is None:
                    raise ValueError(
                        "trace-weight HDF5 serialization requires a dataset path"
                    )
                dims = {
                    1: ("receiver",),
                    2: ("component", "receiver"),
                    3: ("source", "component", "receiver"),
                }[self._weight_values.ndim]
                if params.get("layout") == "sparse_trace":
                    dims = ("trace",)
                ref = store.put_dataarray(
                    dataset,
                    xr.DataArray(self._weight_values, dims=dims),
                    attrs={"fs_kind": "residual_trace_weights"},
                    coordinate_dims=(),
                    dtype=np.float64,
                )
                params["weights"] = {
                    "_type": "HDF5Dense",
                    **ref.to_fs(format="HDF5"),
                }

        return {
            "schema": self.schema,
            **({"name": self.name} if self.name is not None else {}),
            "kind": self.kind,
            "stage": self.stage,
            "params": params,
        }

    @classmethod
    def from_fs(cls, data: Dict[str, Any]) -> "PreprocessHook":
        """Deserialize a preprocessing hook."""

        return cls(
            schema=data.get("schema", "fs-preprocess-hook-1"),
            name=data.get("name"),
            kind=data["kind"],
            stage=data["stage"],
            params=dict(data.get("params", {})),
        )


def _preprocess_to_fs(
    hooks: Sequence[Union[PreprocessHook, Mapping[str, Any]]],
    ctx: Optional[ExportContext] = None,
    *,
    scope: str,
) -> List[Dict[str, Any]]:
    """Serialize typed and raw preprocessing hooks without changing order."""

    return [
        (
            hook.to_fs(
                ctx,
                dataset=f"inputs/imaging/trace_weights/{scope}/{index}",
            )
            if isinstance(hook, PreprocessHook)
            else dict(hook)
        )
        for index, hook in enumerate(hooks)
    ]


def _preprocess_from_fs(
    hooks: Sequence[Dict[str, Any]],
) -> List[Union[PreprocessHook, Mapping[str, Any]]]:
    """Deserialize preprocessing hook payloads."""

    return [PreprocessHook.from_fs(hook) for hook in hooks]


@dataclass(kw_only=True)
class ImageDatabase:
    """Reader for solver imaging output files.

    Args:
        path: Directory containing aggregate and per-frequency image HDF5
            files.
        parts: Number of per-frequency image parts.
        shape: Expected image-grid shape.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """

    path: Path
    parts: int
    shape: Tuple[int, ...]

    def __post_init__(self):
        """Normalize and validate the image output directory."""

        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Image path {self.path} does not exist")

    @property
    def f_list(self):
        """Return frequency values recorded in per-part image files.

        Returns:
            NumPy array with one frequency per image part.
        """

        import h5py

        f_list = np.zeros(self.parts)
        for i in range(self.parts):
            file = self.image_file(i + 1)
            with h5py.File(file, "r") as f:
                f_list[i] = f["frequency"][()]
        return f_list

    def image_file(self, part: Optional[int] = None):
        """Return the aggregate or per-frequency image HDF5 file path.

        Args:
            part: Optional one-based image part number. When omitted, the
                aggregate ``image.h5`` path is returned.

        Returns:
            Path to the requested image file.
        """

        if part is None:
            return self.path / "image.h5"
        else:
            return self.path / f"image_{part}.h5"

    def require_aggregate(self) -> Path:
        """Return the aggregate image file or raise a solver-specific error.

        Returns:
            Path to ``image.h5``.

        Raises:
            FileNotFoundError: If the aggregate file written by the solver
                smoothing/stacking workflow is missing.
        """

        file = self.image_file()
        if file.exists():
            return file
        part_files = [self.image_file(i + 1) for i in range(self.parts)]
        existing_parts = [path.name for path in part_files if path.exists()]
        detail = (
            f" Found per-frequency image shard(s): {', '.join(existing_parts)}."
            if existing_parts
            else " No per-frequency image shards were found either."
        )
        raise FileNotFoundError(
            f"Aggregate image file {file} is missing. Sauce writes image.h5 "
            "during the imaging --smooth postprocess after image_N.h5 shards "
            f"are produced.{detail}"
        )

    @property
    def raw_images(self):
        """Return images from the solver ``image/raw`` group.

        Returns:
            Dataset containing raw image volumes.
        """

        return self.read_images("raw")

    @property
    def smoothed_images(self):
        """Return images from the solver ``image/smoothed`` group.

        Returns:
            Dataset containing smoothed image volumes.
        """

        return self.read_images("smoothed")

    def read_images(self, group):
        """Read one solver image group into an ``xarray.Dataset``.

        Args:
            group: HDF5 group under ``image`` to read, such as ``"raw"`` or
                ``"smoothed"``.

        Returns:
            Dataset containing one data variable per imaged property.
        """

        import h5py
        import xarray as xr

        images = xr.Dataset()
        file = self.require_aggregate()
        with h5py.File(file, "r") as f:
            h5group = f["image"][group]
            properties = [self._decode_label(p) for p in h5group["properties"]]
            for prop in properties:
                h5data = h5group[prop]
                attrs = h5data.attrs
                x0 = np.asarray(attrs["x0"])[::-1]
                x1 = np.asarray(attrs["x1"])[::-1]
                n = np.asarray(attrs["n_grid"], dtype=int)[::-1]
                dims = [self._decode_label(dim) for dim in attrs["dims"][::-1]]
                axis_units = self._axis_units(attrs, dims)
                coords = {}
                for i, dim in enumerate(dims):
                    coords[dim] = np.linspace(x0[i], x1[i], n[i])
                im = xr.DataArray(
                    data=h5data[:].reshape(n),
                    dims=dims,
                    coords=coords,
                )
                for dim, units in zip(dims, axis_units):
                    if units:
                        im.coords[dim].attrs["units"] = units
                units = self._decode_attr(attrs.get("units"))
                if units:
                    im.attrs["units"] = units
                for attr_name in ("coordinate_system", "value_scale", "value_storage"):
                    value = self._decode_attr(attrs.get(attr_name))
                    if value is not None:
                        im.attrs[attr_name] = value
                images[prop] = im
        return images

    @staticmethod
    def _decode_label(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return value.decode("utf-8")
        return str(value)

    @classmethod
    def _decode_attr(cls, value):
        if value is None:
            return None
        if isinstance(value, (bytes, np.bytes_)):
            return cls._decode_label(value)
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return cls._decode_attr(value.item())
            decoded = [cls._decode_attr(item) for item in value.tolist()]
            return decoded[0] if len(decoded) == 1 else decoded
        if isinstance(value, (list, tuple)):
            decoded = [cls._decode_attr(item) for item in value]
            return decoded[0] if len(decoded) == 1 else decoded
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _axis_units(cls, attrs, dims):
        units = cls._decode_attr(attrs.get("axis_units"))
        if units is None:
            return [None] * len(dims)
        if isinstance(units, str):
            return [units] * len(dims)
        return list(units)[::-1]


@dataclass(frozen=True, kw_only=True)
class HDF5TraceStore:
    """One named receiver trace product in an HDF5 trace store.

    A ``dataset`` may be a regular HDF5 dataset or a receiver-group name in a
    packed FrequenSolve trace root.  Sauce resolves the latter for each active
    frequency, so the reference remains valid for multi-frequency jobs.
    """

    file: Union[str, Path]
    dataset: Optional[str] = None
    missing: Optional[Literal["error", "zero", "zeros", "warn", "warning"]] = None
    source_basis: Optional[Literal["source_encoding", "source_geometry"]] = None

    def __post_init__(self) -> None:
        """Validate the native trace-store descriptor fields."""

        file = str(self.file).strip()
        if not file:
            raise ValueError("HDF5 trace-store file must be non-empty")
        object.__setattr__(self, "file", Path(file))
        if self.dataset is not None:
            dataset = str(self.dataset).strip().strip("/")
            if not dataset:
                raise ValueError("HDF5 trace-store dataset must be non-empty")
            object.__setattr__(self, "dataset", dataset)
        if self.missing is not None:
            missing = str(self.missing).strip().lower()
            if missing not in {"error", "zero", "zeros", "warn", "warning"}:
                raise ValueError("unsupported HDF5 trace-store missing policy")
            object.__setattr__(self, "missing", missing)
        if self.source_basis is not None:
            source_basis = str(self.source_basis).strip().lower()
            if source_basis not in {"source_encoding", "source_geometry"}:
                raise ValueError("unsupported HDF5 trace-store source basis")
            object.__setattr__(self, "source_basis", source_basis)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this reference using Sauce's HDF5 trace-store contract."""

        return {
            "_type": "HDF5TraceStore",
            "file": self.file,
            **({"dataset": self.dataset} if self.dataset is not None else {}),
            **({"missing": self.missing} if self.missing is not None else {}),
            **(
                {"source_basis": self.source_basis}
                if self.source_basis is not None
                else {}
            ),
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "HDF5TraceStore":
        """Deserialize Sauce's HDF5 trace-store descriptor."""

        allowed = {"_type", "file", "dataset", "missing", "source_basis"}
        unknown = sorted(set(data).difference(allowed))
        if unknown:
            raise ValueError(
                f"unsupported HDF5 trace-store option(s): {', '.join(unknown)}"
            )
        if data.get("_type") not in {"HDF5TraceStore", "SeismicStore"}:
            raise ValueError("expected an HDF5TraceStore or SeismicStore descriptor")
        return cls(
            file=data["file"],
            dataset=data.get("dataset"),
            missing=data.get("missing"),
            source_basis=data.get("source_basis"),
        )


@dataclass(frozen=True, kw_only=True)
class ObservedTraceDerivatives:
    """Explicit observed spectral derivatives needed by a comparison operator."""

    df: Union[str, Path, HDF5TraceStore]

    def __post_init__(self) -> None:
        """Normalize one supported df trace reference."""

        if isinstance(self.df, HDF5TraceStore):
            return
        if not isinstance(self.df, (str, Path)):
            raise TypeError("df must be a path or HDF5TraceStore")
        value = str(self.df).strip()
        if not value:
            raise ValueError("df trace reference must be non-empty")
        object.__setattr__(self, "df", Path(value))

    @classmethod
    def packed(
        cls,
        trace_root: Union[str, Path],
        *,
        receiver_group: str,
        source_basis: Literal["source_encoding", "source_geometry"] = "source_encoding",
    ) -> "ObservedTraceDerivatives":
        """Reference the first df channel for a packed FrequenSolve trace root."""

        group = str(receiver_group).strip()
        if not group:
            raise ValueError("receiver_group must be non-empty")
        trace_file = Path(trace_root)
        if trace_file.suffix.lower() not in {".h5", ".hdf5"}:
            trace_file = trace_file / "traces.h5"
        return cls(
            df=HDF5TraceStore(
                file=trace_file,
                dataset=f"{group}_df",
                source_basis=source_basis,
            )
        )

    @overload
    @classmethod
    def from_value(
        cls, value: Union["ObservedTraceDerivatives", str, Path, Mapping[str, Any]]
    ) -> "ObservedTraceDerivatives": ...

    @overload
    @classmethod
    def from_value(cls, value: None) -> None: ...

    @classmethod
    def from_value(
        cls,
        value: Optional[
            Union[
                "ObservedTraceDerivatives",
                str,
                Path,
                Mapping[str, Any],
            ]
        ],
    ) -> Optional["ObservedTraceDerivatives"]:
        """Normalize a public derivative reference or its serialized mapping."""

        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls(df=value)
        if not isinstance(value, Mapping):
            raise TypeError("observed_derivatives must be a derivative reference")
        if set(value) != {"df"}:
            raise ValueError("observed_derivatives must contain only df")
        df = value["df"]
        if isinstance(df, Mapping):
            df = HDF5TraceStore.from_fs(df)
        return cls(df=df)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize explicit observed derivatives for one receiver group."""

        return {
            "df": self.df.to_fs() if isinstance(self.df, HDF5TraceStore) else self.df
        }

    def resolved(
        self, resolve_path: Callable[[Any], Any]
    ) -> "ObservedTraceDerivatives":
        """Return an equivalent derivative reference with local paths resolved."""

        if isinstance(self.df, HDF5TraceStore):
            return ObservedTraceDerivatives(
                df=HDF5TraceStore(
                    file=resolve_path(self.df.file),
                    dataset=self.df.dataset,
                    missing=self.df.missing,
                    source_basis=self.df.source_basis,
                )
            )
        return ObservedTraceDerivatives(df=resolve_path(self.df))


def _derivative_input_fingerprint(
    derivatives: ObservedTraceDerivatives,
    fingerprint: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Fingerprint the file that owns one observed df reference."""

    df = derivatives.df
    if hasattr(df, "file"):
        return fingerprint(df.file)
    return fingerprint(df)


@dataclass(kw_only=True)
class MisfitGroup:
    """Observed and simulated trace paths for one receiver group misfit.

    Args:
        name: Receiver group name.
        observed: Trace-store root for observed receiver data, or
            ``None`` to request solver-side zero data.
        observed_derivatives: Explicit observed derivative traces required by
            spectral comparisons such as instantaneous travel-time FWI.
        simulated: Trace-store root for simulated receiver data.
    """

    name: str = ""
    observed: Optional[Union[str, Path]] = None
    observed_derivatives: Optional[
        Union[ObservedTraceDerivatives, str, Path, Mapping[str, Any]]
    ] = None
    simulated: Union[str, Path] = ""
    preprocess: List[Union[PreprocessHook, Mapping[str, Any]]] = field(
        default_factory=list
    )

    def __post_init__(self):
        self.observed = None if self.observed is None else Path(self.observed)
        self.observed_derivatives = ObservedTraceDerivatives.from_value(
            self.observed_derivatives
        )
        self.simulated = Path(self.simulated)

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        *,
        project_relative: bool = False,
        preprocess_scope: Optional[str] = None,
    ) -> Dict:
        """Serialize the receiver-group misfit path mapping.

        Args:
            ctx: Optional export context accepted for API consistency.
            project_relative: Accepted for API consistency; path conversion is
                handled by ``ImagingJob``.

        Returns:
            JSON-compatible receiver-group misfit payload.
        """

        return {
            "name": self.name,
            "observed": self.observed,
            **(
                {
                    "observed_derivatives": ObservedTraceDerivatives.from_value(
                        self.observed_derivatives
                    ).to_fs()
                }
                if self.observed_derivatives is not None
                else {}
            ),
            "simulated": self.simulated,
            **(
                {
                    "preprocess": _preprocess_to_fs(
                        self.preprocess,
                        ctx,
                        scope=preprocess_scope or f"receiver_groups/{self.name}",
                    )
                }
                if self.preprocess
                else {}
            ),
        }

    def add_trace_weights(
        self,
        weights: Any,
        *,
        layout: Optional[
            Literal[
                "receiver",
                "component_receiver",
                "source_component_receiver",
                "sparse_trace",
            ]
        ] = None,
        name: Optional[str] = None,
    ) -> PreprocessHook:
        """Append arbitrary residual weights for this receiver group."""

        hook = PreprocessHook.trace_weight(weights, layout=layout, name=name)
        self.preprocess.append(hook)
        return hook

    def add_receiver_ar1_whitening(
        self,
        correlation: float,
        *,
        name: Optional[str] = None,
    ) -> PreprocessHook:
        """Append stationary neighbor-covariance whitening for this group."""

        hook = PreprocessHook.receiver_ar1_whiten(correlation, name=name)
        self.preprocess.append(hook)
        return hook

    def add_scholte_notch(
        self,
        phase_velocity: Optional[Any] = None,
        *,
        wavenumber: Optional[Any] = None,
        relative_half_width: float = 0.04,
        relative_taper_width: float = 0.04,
        spacing_tolerance: float = 1.0e-3,
        name: Optional[str] = None,
    ) -> PreprocessHook:
        """Append a smooth Scholte-ridge notch for this receiver group."""

        hook = PreprocessHook.scholte_notch(
            phase_velocity,
            wavenumber=wavenumber,
            relative_half_width=relative_half_width,
            relative_taper_width=relative_taper_width,
            spacing_tolerance=spacing_tolerance,
            name=name,
        )
        self.preprocess.append(hook)
        return hook

    def add_slow_velocity_mute(
        self,
        stop_velocity: Any,
        pass_velocity: Any,
        *,
        mode: Literal["reject_slow", "keep_slow"] = "reject_slow",
        spacing_tolerance: float = 1.0e-3,
        name: Optional[str] = None,
    ) -> PreprocessHook:
        """Append a smooth apparent-velocity fan mute for this receiver group."""

        hook = PreprocessHook.slow_velocity_mute(
            stop_velocity,
            pass_velocity,
            mode=mode,
            spacing_tolerance=spacing_tolerance,
            name=name,
        )
        self.preprocess.append(hook)
        return hook

    @classmethod
    def from_fs(cls, data: Dict) -> "MisfitGroup":
        """Deserialize a receiver-group misfit path mapping.

        Args:
            data: Serialized misfit-group payload.

        Returns:
            ``MisfitGroup`` with trace-store roots restored.
        """

        return cls(
            name=data["name"],
            observed=data["observed"],
            observed_derivatives=data.get("observed_derivatives"),
            simulated=data["simulated"],
            preprocess=_preprocess_from_fs(data.get("preprocess", [])),
        )


@dataclass(frozen=True, kw_only=True)
class MisfitComparison:
    """Receiver-data comparison operator used before the misfit norm.

    ``phase_derivative`` compares the stabilized instantaneous phase-frequency
    slope, which is the frequency-domain instantaneous travel-time attribute.
    Sauce consumes base and first real-frequency-derivative trace stores for
    that comparison.
    """

    kind: Literal["waveform", "phase_derivative"] = "waveform"
    derivative_axis: Literal["frequency"] = "frequency"
    source_derivative: Literal["frozen", "total"] = "frozen"
    relative_amplitude_floor: float = 0.01

    def __post_init__(self) -> None:
        """Normalize and validate one solver comparison configuration."""

        kind = str(self.kind).strip().lower()
        if kind not in {"waveform", "phase_derivative"}:
            raise ValueError("comparison kind must be 'waveform' or 'phase_derivative'")
        axis = str(self.derivative_axis).strip().lower()
        if axis != "frequency":
            raise ValueError(
                "phase-derivative comparison requires derivative_axis='frequency'"
            )
        source_derivative = str(self.source_derivative).strip().lower()
        if source_derivative not in {"frozen", "total"}:
            raise ValueError("source_derivative must be 'frozen' or 'total'")
        try:
            floor = float(self.relative_amplitude_floor)
        except (TypeError, ValueError) as exc:
            raise TypeError("relative_amplitude_floor must be numeric") from exc
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("relative_amplitude_floor must be finite and positive")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "derivative_axis", axis)
        object.__setattr__(self, "source_derivative", source_derivative)
        object.__setattr__(self, "relative_amplitude_floor", floor)

    @classmethod
    def phase_derivative(
        cls,
        *,
        source_derivative: Literal["frozen", "total"] = "frozen",
        relative_amplitude_floor: float = 0.01,
    ) -> "MisfitComparison":
        """Create an instantaneous travel-time phase-derivative comparison."""

        return cls(
            kind="phase_derivative",
            source_derivative=source_derivative,
            relative_amplitude_floor=relative_amplitude_floor,
        )

    @overload
    @classmethod
    def from_value(
        cls, value: Union["MisfitComparison", Mapping[str, Any]]
    ) -> "MisfitComparison": ...

    @overload
    @classmethod
    def from_value(cls, value: None) -> None: ...

    @classmethod
    def from_value(
        cls,
        value: Optional[Union["MisfitComparison", Mapping[str, Any]]],
    ) -> Optional["MisfitComparison"]:
        """Normalize a comparison object or its serialized mapping."""

        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("comparison must be a MisfitComparison or mapping")
        allowed = {
            "kind",
            "derivative_axis",
            "source_derivative",
            "relative_amplitude_floor",
        }
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(f"unsupported comparison option(s): {', '.join(unknown)}")
        return cls(**dict(value))

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the comparison using Sauce's imaging contract."""

        if self.kind == "waveform":
            return {"kind": "waveform"}
        return {
            "kind": self.kind,
            "derivative_axis": self.derivative_axis,
            "source_derivative": self.source_derivative,
            "relative_amplitude_floor": self.relative_amplitude_floor,
        }


@dataclass(kw_only=True)
class Misfit:
    """Misfit norm and receiver groups used by an imaging job.

    Args:
        norm: Misfit norm name. Currently the solver-facing contract supports
            ``"L2"``.
        comparison: Optional receiver-data comparison operator. The
            ``phase_derivative`` operator is the instantaneous travel-time
            comparison and requires first frequency-derivative trace stores.
        receiver_groups: Receiver-group misfit path mappings.
    """

    norm: Literal["L2"] = "L2"
    comparison: Optional[Union[MisfitComparison, Mapping[str, Any]]] = None
    receiver_groups: List[MisfitGroup] = field(default_factory=list)
    preprocess: List[Union[PreprocessHook, Mapping[str, Any]]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """Normalize the optional comparison operator."""

        self.comparison = MisfitComparison.from_value(self.comparison)

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict:
        """Serialize the imaging misfit configuration.

        Args:
            ctx: Optional export context forwarded to receiver groups.
            project_relative: Accepted for API consistency; path conversion is
                handled by ``ImagingJob``.

        Returns:
            JSON-compatible misfit payload.
        """

        return {
            "norm": self.norm,
            **(
                {"comparison": MisfitComparison.from_value(self.comparison).to_fs()}
                if self.comparison is not None
                else {}
            ),
            **(
                {
                    "preprocess": _preprocess_to_fs(
                        self.preprocess,
                        ctx,
                        scope="misfit",
                    )
                }
                if self.preprocess
                else {}
            ),
            "receiver_groups": [
                group.to_fs(
                    ctx,
                    preprocess_scope=f"receiver_groups/{index}",
                )
                for index, group in enumerate(self.receiver_groups)
            ],
        }

    @classmethod
    def from_fs(cls, data: Dict) -> "Misfit":
        """Deserialize an imaging misfit configuration.

        Args:
            data: Serialized misfit payload.

        Returns:
            ``Misfit`` with receiver groups restored.
        """

        return cls(
            norm=data["norm"],
            comparison=data.get("comparison"),
            preprocess=_preprocess_from_fs(data.get("preprocess", [])),
            receiver_groups=[
                MisfitGroup.from_fs(group) for group in data["receiver_groups"]
            ],
        )


@register_class
@dataclass(kw_only=True)
class ImagingJob(BaseJob):
    """Reverse-time migration or FWI-imaging job configuration.

    Args:
        name: Job name used in project paths and serialized payloads.
        simulation: Seismic simulation used to compute simulated data.
        data_path: Optional directory containing observed trace data. When
            ``None``, the solver receives ``null`` and uses zero data to compute
            sensitivity kernels.
        f_list: Frequencies to image.
        resolution: Optional image-grid resolution used when ``grid`` is not
            provided.
        grid: Optional explicit Cartesian image grid.
        images: Mapping from image name to solver imaging condition.
        weights: Optional per-frequency weights.
        wavelet: Optional wavelet whose spectrum is sampled for weights.
        misfit_norm: Misfit norm name.
        comparison: Optional receiver-data comparison operator. Use
            :meth:`MisfitComparison.phase_derivative` for instantaneous
            travel-time FWI. Supply ``observed_derivatives`` with base and
            first ``df`` observed traces.
        observed_derivatives: Explicit observed first frequency derivatives.
            :meth:`ObservedTraceDerivatives.packed` references the ``_df``
            channel of a packed FrequenSolve trace root.
        keep_forward: Keep forward wavefields after imaging.
        keep_adjoint: Keep adjoint wavefields after imaging.
        keep_unstacked: Keep per-source or per-frequency image contributions.
        regularization: Optional smoothing/regularization payload.
        save_path: Optional image-output directory.
        reassemble_adjoint: Request adjoint reassembly from stored pieces.
        outputs: Optional output request or output collection for this job.
        k_list: Optional signed physical Fourier wavenumbers for 2.5D jobs.
        k_weights: Optional quadrature weights paired with ``k_list``.
        k_units: Optional units for ``k_list`` and ``k_weights``.
        **kwargs: Extra imaging payload fields preserved on export.

    Raises:
        FileNotFoundError: If a non-null ``data_path`` does not exist.
        ValueError: If weights do not match frequencies or a grid cannot be
            inferred from the simulation model.
    """

    misfit: Misfit = field(default_factory=Misfit)
    data_path: Optional[Union[str, Path]]
    save_path: Path
    grid: CartesianGrid = field(default_factory=CartesianGrid)
    representation: CartesianGridRepresentation = field(init=False)
    keep_forward: bool = False
    keep_adjoint: bool = False
    keep_unstacked: bool = False
    images: dict = field(default_factory=dict)
    weights: Optional[List[float]] = field(default_factory=list)
    reassemble_adjoint: bool = False

    def __init__(
        self,
        name: str,
        simulation: SeismicSimulation,
        data_path: Optional[Union[str, Path]] = None,
        f_list: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
        grid: Optional[CartesianGrid] = None,
        representation: Optional[CartesianGridRepresentation] = None,
        images: Optional[dict] = None,
        weights: Optional[List[float]] = None,
        wavelet: Optional[Wavelet] = None,
        misfit_norm: Literal["L2"] = "L2",
        comparison: Optional[Union[MisfitComparison, Mapping[str, Any]]] = None,
        observed_derivatives: Optional[
            Union[ObservedTraceDerivatives, str, Path, Mapping[str, Any]]
        ] = None,
        keep_forward: bool = False,
        keep_adjoint: bool = False,
        keep_unstacked: bool = False,
        regularization: Optional[Union[VariationalSmoothing, Mapping[str, Any]]] = None,
        save_path: Optional[Union[str, Path]] = None,
        reassemble_adjoint: bool = False,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
        k_list: Optional[Iterable[float]] = None,
        k_weights: Optional[Iterable[float]] = None,
        k_units: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create an imaging job.

        The job compares simulated traces against observed data at
        ``data_path`` and writes image products to ``save_path`` or the job
        result directory.
        """

        if "misfit_type" in kwargs:
            misfit_norm = kwargs.pop("misfit_type")
        if f_list is None:
            raise ValueError("f_list is required for an imaging job")
        simulation.save()
        super().__init__(
            name=name,
            simulation=simulation,
            f_list=list(f_list),
            workflow="RTM",
            outputs=JobOutputs(outputs),
            k_list=None if k_list is None else list(k_list),
            k_weights=None if k_weights is None else list(k_weights),
            k_units=k_units,
        )

        self.regularization: Union[VariationalSmoothing, Dict[str, Any]]
        f_sim = self.trace_outputs.path

        self.data_path = None if data_path is None else Path(data_path)
        if self.data_path is not None and not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")

        if save_path is not None:
            self.save_path = Path(save_path).resolve()
        else:
            self.save_path = self._result_path / "imaging"
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

        if wavelet is not None:
            self.wavelet = wavelet
            frequencies = self.wavelet.frequencies
            spectrum = abs(self.wavelet.spectrum)
            self.weights = []
            for f in self.f_list:
                idx = abs(frequencies - f).argmin()
                self.weights.append(float(spectrum[idx]))
        elif weights is not None:
            self.weights = np.asarray(weights, dtype=float).reshape(-1).tolist()
            if len(self.weights) != len(self.f_list):
                raise ValueError(
                    "Imaging weights must contain one value per frequency; "
                    f"got {len(self.weights)} weights for {len(self.f_list)} frequencies"
                )
        else:
            self.weights = None

        if isinstance(regularization, VariationalSmoothing):
            if regularization.derivative_order != 1:
                raise ValueError(
                    "Cartesian image smoothing uses mixed first-order FEM fields"
                )
            if regularization.input_role != "primal":
                raise ValueError(
                    "Cartesian image samples require smoothing input_role='primal'"
                )
            self.regularization = regularization
        elif regularization is not None:
            self.regularization = dict(regularization)
        else:
            self.regularization = {
                "type": "TV",
                "lambda": 1.0,
                "epsilon": 1.0,
                "iterations": 5,
            }

        self.kwargs = kwargs
        self.images = dict(images or {})
        self.keep_forward = keep_forward
        self.keep_adjoint = keep_adjoint
        self.keep_unstacked = keep_unstacked
        self.misfit = Misfit(norm=misfit_norm, comparison=comparison)
        self.observed_derivatives = ObservedTraceDerivatives.from_value(
            observed_derivatives
        )
        if (
            self.misfit.comparison is not None
            and MisfitComparison.from_value(self.misfit.comparison).kind
            == "phase_derivative"
            and self.observed_derivatives is None
        ):
            raise ValueError(
                "phase-derivative comparison requires observed_derivatives; "
                "use ObservedTraceDerivatives.packed(...) for packed traces"
            )
        for receiver_group in simulation.acquisition.receiver_groups:
            self.misfit.receiver_groups.append(
                MisfitGroup(
                    name=receiver_group.name,
                    observed=self.data_path,
                    observed_derivatives=self.observed_derivatives,
                    simulated=f_sim,
                )
            )

        if grid is not None and representation is not None:
            raise ValueError("provide either grid or representation, not both")
        if representation is not None:
            if not isinstance(representation, CartesianGridRepresentation):
                raise TypeError(
                    "Sauce imaging currently requires a CartesianGridRepresentation"
                )
            self.grid = representation.grid
        elif grid is not None:
            self.grid = grid
        # TODO: this will only work for a layered model right now
        elif simulation.model.dimension == 2:
            if resolution is None:
                raise ValueError("resolution is required when grid is not provided")
            x_limits = simulation.model.x_limits
            z_limits = simulation.model.z_limits
            x0 = [x_limits[0], z_limits[0]]
            x1 = [x_limits[1], z_limits[1]]
            assert (
                len(resolution) == 2
            ), "For 2D models, resolution must be a length 2 array"
            self.grid = CartesianGrid(x0=x0, x1=x1, n=resolution)
        elif simulation.model.dimension == 3:
            if resolution is None:
                raise ValueError("resolution is required when grid is not provided")
            x_limits = simulation.model.x_limits
            y_limits = simulation.model.y_limits
            z_limits = simulation.model.z_limits
            x0 = [x_limits[0], y_limits[0], z_limits[0]]
            x1 = [x_limits[1], y_limits[1], z_limits[1]]
            assert (
                len(resolution) == 3
            ), "For 3D models, resolution must be a length 3 array"
            self.grid = CartesianGrid(x0=x0, x1=x1, n=resolution)
        else:
            raise ValueError(f"Unknown model dimension: {simulation.model.dimension}")
        self.representation = CartesianGridRepresentation(self.grid)
        self.reassemble_adjoint = reassemble_adjoint

    def image_file(self, part: Optional[int] = None) -> Path:
        """Return the local aggregate or per-frequency image file path."""

        if part is None:
            return self.save_path / "image.h5"
        return self.save_path / f"image_{part}.h5"

    def requires_postprocess(self) -> bool:
        """Return true because images are stacked and smoothed after tasks."""

        return True

    def postprocess_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate or task-local image path."""

        return self.image_file(part)

    def load_images(self) -> ImageDatabase:
        """Open imaging results that are already present locally.

        This method never contacts an execution site. Remote results must be
        fetched or copied into ``save_path`` before calling it.

        Returns:
            Reader for the aggregate and per-frequency image files.

        Raises:
            FileNotFoundError: If the aggregate ``image.h5`` is unavailable.
        """

        images = ImageDatabase(
            path=self._local_image_path,
            parts=self.n_tasks,
            shape=self.grid.shape,
        )
        images.require_aggregate()
        return images

    def image_output_exists(self) -> bool:
        """Return whether the aggregate image product exists locally."""

        return self.image_file().is_file()

    def postprocess_output_exists(self) -> bool:
        """Return whether the final stacked image exists locally."""

        return self.image_output_exists()

    def image_part_outputs_exist(self) -> bool:
        """Return whether every per-frequency image shard exists locally."""

        return all(
            self.image_file(part).is_file() for part in range(1, self.n_tasks + 1)
        )

    def postprocess_part_outputs_exist(self) -> bool:
        """Return whether every per-frequency image shard exists locally."""

        return self.image_part_outputs_exist()

    def needs_image_smoothing(self) -> bool:
        """Return whether local shards are present but ``image.h5`` is missing."""

        return self.needs_postprocess()

    def is_run_current(self) -> bool:
        """Return whether the imaging run and aggregate image are current."""

        return super().is_run_current() and self.image_output_exists()

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict:
        """Serialize this imaging job to the solver job contract.

        Args:
            ctx: Optional export context accepted for API consistency.
            project_relative: When true, emit project-relative paths where
                possible.

        Returns:
            JSON-compatible imaging job payload.
        """

        images = []
        for key, value in self.images.items():
            tmp = value.split(":")
            if tmp[0] == "FWI":
                cond = tmp[0]
                prop = tmp[1]
            else:
                cond = value
                prop = None
            images.append(
                {
                    "name": key,
                    "IC": cond,
                    **({"property": prop} if prop is not None else {}),
                }
            )

        imaging = {
            "data_path": self._export_path(
                self.data_path, project_relative=project_relative
            ),
            "save_path": self._export_path(
                self.save_path, project_relative=project_relative
            ),
            "misfit": self._misfit_to_fs(ctx, project_relative=project_relative),
            "grid": self.representation.to_fs(ctx),
            "keep_forward": self.keep_forward,
            "keep_adjoint": self.keep_adjoint,
            "keep_unstacked": self.keep_unstacked,
            "images": images,
            "weights": self.weights,
            "reassemble_adjoint": self.reassemble_adjoint,
            "Smoothing": self._smoothing_to_fs(),
            **self.kwargs,
        }
        return {
            **super().to_fs(ctx, project_relative=project_relative),
            "Image": imaging,
        }

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash observed derivative inputs for every receiver group."""

        derivatives = {
            group.name: _derivative_input_fingerprint(
                group.observed_derivatives, self._path_content_fingerprint
            )
            for group in self.misfit.receiver_groups
            if group.observed_derivatives is not None
        }
        return {"observed_derivatives": derivatives} if derivatives else {}

    def _smoothing_to_fs(self) -> Dict[str, Any]:
        """Serialize smoothing for the Cartesian-image FEM implementation."""

        if not isinstance(self.regularization, VariationalSmoothing):
            payload = dict(self.regularization)
            if (
                "illumination_normalization" not in payload
                and "normalize_illumination" not in payload
            ):
                payload["illumination_normalization"] = "none"
            return payload
        config = self.regularization
        payload = config.to_fs()
        payload.pop("input_role")
        if config.reference_wavelength is not None:
            payload.pop("reference_wavelength")
            if config.kind == "tgv":
                payload["alpha1"], payload["alpha2"] = config.resolved_tgv_weights()
            elif config.kind in {"tikhonov", "tv"}:
                payload["alpha"] = config.resolved_alpha()
        return payload

    @classmethod
    def from_fs(
        cls,
        data: Dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "ImagingJob":
        """Deserialize an imaging job from a saved job payload.

        Args:
            data: Serialized imaging job payload.
            base_path: Optional directory used to resolve relative paths.
            project_path: Optional project root used to resolve
                project-relative paths.

        Returns:
            Reconstructed ``ImagingJob``.

        Raises:
            KeyError: If the payload does not include an imaging section.
        """

        data = dict(data)
        stored_project_path = data.get("project_path")
        resolved_project_path = project_path or stored_project_path
        image_data = data.pop("Image", data.pop("Imaging", None))
        if image_data is None:
            raise KeyError("ImagingJob data must include an 'Image' section")
        grid = image_data.pop("grid", None)
        grid_obj = CartesianGrid.from_fs(grid) if grid is not None else None
        resolution = list(grid_obj.n) if grid_obj is not None else None
        misfit = Misfit.from_fs(image_data.pop("misfit"))
        images = image_data.pop("images", None) or []
        images = {
            image["name"]: (
                f"FWI:{image['property']}"
                if image.get("IC") == "FWI" and image.get("property") is not None
                else image["IC"]
            )
            for image in images
        }
        simulation_ref = data.pop("simulation")
        simulation_path = Path(simulation_ref)
        if not simulation_path.is_absolute():
            project_path = project_path or data.get("project_path")
            if project_path is None and base_path is not None:
                project_path = BaseJob._project_root_from_job_path(Path(base_path))
            if project_path is not None:
                simulation_path = Path(project_path) / simulation_path

        project_path = project_path or data.get("project_path")
        data_path = cls._resolve_saved_path(
            image_data.pop("data_path", None),
            base_path=base_path,
            project_path=resolved_project_path,
        )
        save_path = cls._resolve_saved_path(
            image_data.pop("save_path", None),
            base_path=base_path,
            project_path=resolved_project_path,
        )
        for group in misfit.receiver_groups:
            group.observed = cls._resolve_saved_path(
                group.observed,
                base_path=base_path,
                project_path=resolved_project_path,
            )
            group.simulated = cls._resolve_saved_path(
                group.simulated,
                base_path=base_path,
                project_path=resolved_project_path,
            )
            if group.observed_derivatives is not None:
                group.observed_derivatives = ObservedTraceDerivatives.from_value(
                    group.observed_derivatives
                ).resolved(
                    lambda path: cls._resolve_saved_path(
                        path,
                        base_path=base_path,
                        project_path=resolved_project_path,
                    )
                )
        if "direction" in image_data:
            image_data["direction"] = cls._resolve_saved_path(
                image_data["direction"],
                base_path=base_path,
                project_path=resolved_project_path,
            )

        job = cls(
            name=data.pop("name", None),
            simulation=SeismicSimulation.load(simulation_path),
            f_list=data.pop("f_list", None),
            data_path=data_path,
            resolution=resolution,
            grid=grid_obj,
            images=images,
            keep_forward=image_data.pop("keep_forward", None),
            keep_adjoint=image_data.pop("keep_adjoint", None),
            keep_unstacked=image_data.pop("keep_unstacked", None),
            save_path=save_path,
            regularization=image_data.pop("Smoothing", None),
            weights=image_data.pop("weights", None),
            reassemble_adjoint=image_data.pop("reassemble_adjoint", None),
            outputs=JobOutputs.from_fs(data.pop("Outputs", None)),
            k_list=data.pop("k_list", None),
            k_weights=data.pop("k_weights", None),
            k_units=data.pop("k_units", None),
            **image_data,
        )
        job.misfit = misfit
        return job

    def _remote_image_path(self, work_dir: Union[Path, str]):
        """Get local path but with version number."""
        rel_path = self.save_path.relative_to(self.project_path)
        remote = Path(work_dir) / rel_path
        return remote

    @property
    def _local_image_path(self):
        return self.save_path

    def _export_path(
        self,
        path: Optional[Union[str, Path]],
        *,
        project_relative: bool = False,
    ):
        if path is None:
            return None
        path = Path(path)
        if not project_relative:
            return path
        try:
            return path.resolve().relative_to(self._project_path())
        except ValueError:
            return path

    def _misfit_to_fs(self, ctx=None, *, project_relative: bool = False) -> Dict:
        payload = self.misfit.to_fs(ctx)
        if not project_relative:
            return payload
        for group in payload["receiver_groups"]:
            group["observed"] = self._export_path(
                group["observed"], project_relative=True
            )
            group["simulated"] = self._export_path(
                group["simulated"], project_relative=True
            )
            derivatives = group.get("observed_derivatives")
            if derivatives is not None:
                df = derivatives["df"]
                if isinstance(df, Mapping):
                    df = dict(df)
                    derivatives["df"] = df
                    df["file"] = self._export_path(df["file"], project_relative=True)
                else:
                    derivatives["df"] = self._export_path(df, project_relative=True)
        return payload

    @staticmethod
    def _resolve_saved_path(
        path: Union[str, Path, None],
        *,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        if path is None:
            return None
        path = Path(path)
        if path.is_absolute():
            return path
        if project_path is not None:
            return Path(project_path) / path
        if base_path is not None:
            project_root = BaseJob._project_root_from_job_path(Path(base_path))
            if project_root is not None:
                return project_root / path
            return Path(base_path).resolve() / path
        return path


@register_class
class LSRTMGradientJob(ImagingJob):
    """One-call Cartesian LSRTM residual-gradient job.

    Every Sauce frequency task solves the background and incremental forward
    problems, forms ``F0 + Jm - d`` and its L2 objective in the executable,
    and solves one incremental adjoint to write ``J.T @ (F0 + Jm - d)``.
    The Cartesian model iterate is supplied through the ordinary image
    direction buffer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create an LSRTM batch-gradient job."""

        direction = kwargs.pop("direction", None)
        zero_direction = bool(kwargs.pop("zero_direction", direction is None))
        if direction is not None and zero_direction:
            raise ValueError("direction and zero_direction are mutually exclusive")
        if direction is None and not zero_direction:
            raise ValueError(
                "LSRTMGradientJob requires direction or zero_direction=True"
            )
        if direction is not None:
            direction = Path(direction).resolve()
            if not direction.is_file():
                raise FileNotFoundError(f"LSRTM direction file not found: {direction}")
        if kwargs.pop("born_traces_only", False):
            raise ValueError("LSRTMGradientJob cannot be trace-only")
        kwargs["born_traces_only"] = False
        kwargs["gauss_newton"] = True
        super().__init__(*args, **kwargs)
        if any(group.observed is None for group in self.misfit.receiver_groups):
            raise ValueError("LSRTMGradientJob requires observed data")
        self.direction = direction
        self.workflow = "lsrtm_gradient"

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict:
        """Serialize the explicit iterate consumed by the fused workflow."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        image = payload["Image"]
        if self.direction is None:
            image["zero_direction"] = True
        else:
            image["direction"] = self._export_path(
                self.direction, project_relative=project_relative
            )
        return payload

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash the Cartesian iterate consumed by the fused gradient."""

        return {
            **super()._input_fingerprint_payload(),
            "direction": (
                {"kind": "zero"}
                if self.direction is None
                else self._path_content_fingerprint(self.direction)
            ),
        }

    @property
    def trace_outputs(self) -> TraceOutputSpec:
        """Return baseline and incremental receiver-trace products."""

        baseline = super().trace_outputs
        incremental_groups = [f"{group}_inc" for group in baseline.groups]
        incremental_components = [
            (
                f"{component.partition(':')[0]}_inc:{component.partition(':')[2]}"
                if ":" in component
                else component
            )
            for component in baseline.components
        ]
        return replace(
            baseline,
            groups=[*baseline.groups, *incremental_groups],
            components=[*baseline.components, *incremental_components],
        )


@register_class
class LSRTMNormalJob(ImagingJob):
    """Fused Cartesian Born and Gauss-Newton normal-operator job.

    Each frequency task loads the reusable background forward state, solves
    the incremental forward problem, forms the weighted ``-Jp`` receiver
    residual in Sauce, solves the incremental adjoint problem, and writes the
    ``J^T W Jp`` image. Incremental traces remain available for optimizer
    diagnostics without materializing an intermediate residual trace store.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create a fused LSRTM normal-operator job."""

        if kwargs.pop("born_traces_only", False):
            raise ValueError("LSRTMNormalJob cannot be trace-only")
        if not kwargs.pop("gauss_newton", True):
            raise ValueError("LSRTMNormalJob requires gauss_newton=True")
        kwargs["born_traces_only"] = False
        kwargs["gauss_newton"] = True
        super().__init__(*args, **kwargs)
        self.workflow = "born"

    @property
    def trace_outputs(self) -> TraceOutputSpec:
        """Return the incremental receiver-trace output specification."""

        baseline = super().trace_outputs
        return replace(
            baseline,
            groups=[f"{group}_inc" for group in baseline.groups],
            components=[
                (
                    f"{component.partition(':')[0]}_inc:{component.partition(':')[2]}"
                    if ":" in component
                    else component
                )
                for component in baseline.components
            ],
        )


def extract_frequencies_for_job(job: ImagingJob, td):
    """Extract job frequencies from time traces into solver-ready HDF5 files.

    Args:
        job: Imaging job whose frequency list and data path define the output.
        td: Time-domain trace data array.

    Returns:
        ``None``. One HDF5 trace file is written for each job frequency.
    """

    import h5py
    import xarray as xr

    if job.data_path is None:
        raise ValueError(
            "Cannot extract observed frequencies for an imaging job without data_path"
        )

    # Make frequency domain data array
    shape = list(td.shape)
    fd_dims = ["time" if d == "frequency" else d for d in td.dims]
    fd_coords = {d: td.coords[d] for d in fd_dims if d != "time"}
    fd_coords["frequency"] = job.f_list
    fd_coords["complex"] = ["real", "imag"]
    fd_coords["component"] = "v_z"
    for i, d in enumerate(td.dims):
        if d == "time":
            fd_dims[i] = "frequency"
            shape[i] = len(job.f_list)
    if "component" not in td.dims:
        shape.append(1)
        fd_dims.append("component")
    fd_dims.append("complex")
    fd_shape = [*shape, 2]
    fd = xr.DataArray(
        np.zeros(fd_shape),
        dims=fd_dims,
        coords=fd_coords,
    )
    fd = fd.transpose("frequency", "source", "component", "receiver", "complex")

    # Compute frequency domain data
    taxis = td.coords["time"]
    dt = taxis[1] - taxis[0]
    freqs = job.f_list
    for i, f in enumerate(freqs):
        exp = xr.DataArray(
            np.exp(2 * np.pi * 1j * f * td.coords["time"]),
            dims=["time"],
            coords={"time": td.coords["time"]},
        )
        tmp = (td * exp).sum(dim="time") * dt
        ctmp = np.empty([*tmp.shape, 2], dtype=np.float32)
        ctmp[:, :, 0] = tmp.real
        ctmp[:, :, 1] = tmp.imag
        fd.data[i] = ctmp.reshape(fd.data[i].shape)

    # Assumes only one receiver group
    group = job.simulation.acquisition.receiver_groups[0]

    # Write hdf5 files
    for i, f in enumerate(job.f_list):
        file = job.data_path / f"traces_{i + 1}.h5"
        with h5py.File(file, "w") as f:
            f[group.name] = fd.data[i].astype(np.float32)
