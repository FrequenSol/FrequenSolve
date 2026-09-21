"""Native material-control Born and adjoint job authoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union, overload

import h5py
import numpy as np

from frequensolve.model.representation import VariationalSmoothing
from frequensolve.simulation.jobs.artifacts import TraceOutputSpec
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.jobs.imaging import (
    MisfitComparison,
    ObservedTraceDerivatives,
    PreprocessHook,
    _derivative_input_fingerprint,
    _preprocess_from_fs,
    _preprocess_to_fs,
)
from frequensolve.simulation.outputs import JobOutputs, Output
from frequensolve.simulation.simulation import BaseSimulation
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import ExportContext

__all__ = [
    "ControlBlock",
    "ControlSpace",
    "BornControlSensitivityJob",
    "RTMControlSensitivityJob",
    "TimeReversalFocusJob",
]


@dataclass(frozen=True)
class ControlBlock:
    """One named coefficient block; individual coefficients remain positional."""

    id: str
    size: int
    coordinates: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        block_id = str(self.id).strip()
        if not block_id or "/" in block_id or block_id in {".", ".."}:
            raise ValueError("control block id must be a non-empty HDF5-safe name")
        if int(self.size) < 1:
            raise ValueError("control block size must be positive")
        object.__setattr__(self, "id", block_id)
        object.__setattr__(self, "size", int(self.size))
        if self.coordinates is not None:
            coordinates = np.asarray(self.coordinates, dtype=np.float64)
            if coordinates.ndim != 1 or coordinates.size != self.size:
                raise ValueError(
                    "control coordinates must have one value per coefficient"
                )
            if not np.all(np.isfinite(coordinates)):
                raise ValueError("control coordinates must be finite")
            object.__setattr__(self, "coordinates", np.array(coordinates, copy=True))

    @classmethod
    def from_property(cls, value: Any) -> "ControlBlock":
        """Describe the single ordered block owned by a parameterized property."""

        from frequensolve.model.parameterization import ParameterizedProperty
        from frequensolve.model.property import Property

        parameterized = Property.from_value(value)
        if not isinstance(parameterized, ParameterizedProperty):
            raise TypeError("control blocks require a ParameterizedProperty")
        coordinates = getattr(parameterized.control, "coordinates", None)
        return cls(parameterized.id, parameterized.control.size, coordinates)


class ControlSpace:
    """Deterministic vectorization and HDF5 I/O for material-control blocks."""

    dtype = np.dtype(np.float64)

    def __init__(self, blocks: Iterable[ControlBlock]):
        ordered = tuple(sorted(blocks, key=lambda block: block.id))
        if not ordered:
            raise ValueError("control space requires at least one block")
        ids = [block.id for block in ordered]
        if len(set(ids)) != len(ids):
            raise ValueError("control block ids must be unique")
        self.blocks = ordered
        self._slices: Dict[str, slice] = {}
        offset = 0
        for block in ordered:
            self._slices[block.id] = slice(offset, offset + block.size)
            offset += block.size
        self.size = offset

    @classmethod
    def from_property(cls, value: Any) -> "ControlSpace":
        """Build a one-block space directly from a parameterized property."""

        return cls([ControlBlock.from_property(value)])

    @classmethod
    def from_properties(cls, values: Iterable[Any]) -> "ControlSpace":
        """Build a deterministic space from parameterized properties."""

        if isinstance(values, Mapping):
            values = values.values()
        return cls(ControlBlock.from_property(value) for value in values)

    @property
    def shape(self) -> tuple[int]:
        """Return the one-dimensional operator shape."""

        return (self.size,)

    def zeros(self) -> np.ndarray:
        """Return a zero control vector."""

        return np.zeros(self.size, dtype=self.dtype)

    def pack(
        self, values: Union[Mapping[str, Any], Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """Pack per-block values in Sauce's lexical block order."""

        if isinstance(values, Mapping):
            missing = [block.id for block in self.blocks if block.id not in values]
            extra = sorted(set(values) - set(self._slices))
            if missing or extra:
                raise ValueError(
                    f"control blocks do not match space; missing={missing}, extra={extra}"
                )
            vector = self.zeros()
            for block in self.blocks:
                block_values = np.asarray(values[block.id])
                if np.iscomplexobj(block_values):
                    raise ValueError("material control vectors must be real-valued")
                block_values = np.asarray(block_values, dtype=self.dtype)
                if block_values.ndim != 1 or block_values.size != block.size:
                    raise ValueError(
                        f"control block {block.id!r} has shape {block_values.shape}; "
                        f"expected ({block.size},)"
                    )
                vector[self._slices[block.id]] = block_values
        else:
            vector = np.asarray(values)
            if np.iscomplexobj(vector):
                raise ValueError("material control vectors must be real-valued")
            vector = np.asarray(vector, dtype=self.dtype)
            if vector.ndim != 1 or vector.size != self.size:
                raise ValueError(
                    f"control vector has shape {vector.shape}; expected ({self.size},)"
                )
            vector = np.array(vector, copy=True)
        if not np.all(np.isfinite(vector)):
            raise ValueError("control vectors must be finite")
        return vector

    def unpack(
        self, vector: Union[Mapping[str, Any], Sequence[float], np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Unpack a vector into independent ordered coefficient blocks."""

        packed = self.pack(vector)
        return {
            block.id: np.array(packed[self._slices[block.id]], copy=True)
            for block in self.blocks
        }

    def write_hdf5(
        self,
        path: Union[str, Path],
        values: Union[Mapping[str, Any], Sequence[float], np.ndarray],
    ) -> Path:
        """Write float64 ``/controls/<block-id>`` datasets for Sauce."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blocks = self.unpack(values)
        with h5py.File(path, "w") as h5:
            controls = h5.create_group("controls")
            for block in self.blocks:
                controls.create_dataset(
                    block.id, data=blocks[block.id], dtype=np.float64
                )
        return path

    def read_hdf5(self, path: Union[str, Path]) -> np.ndarray:
        """Read and validate a Sauce material-control vector file."""

        with h5py.File(path, "r") as h5:
            if "controls" not in h5:
                raise ValueError("control-vector file has no /controls group")
            values = {
                block.id: np.asarray(h5[f"controls/{block.id}"], dtype=np.float64)
                for block in self.blocks
                if f"controls/{block.id}" in h5
            }
        return self.pack(values)


def _normalized_frequencies(
    values: Union[Sequence[Union[float, complex]], np.ndarray],
) -> list:
    """Normalize Laplace samples to the solver's negative-imaginary convention."""

    frequencies = np.asarray(values)
    if frequencies.size == 0:
        raise ValueError("control-sensitivity jobs require at least one frequency")
    if np.iscomplexobj(frequencies):
        return [complex(value.real, -abs(value.imag)) for value in frequencies]
    return frequencies.astype(float).tolist()


def _active_controls(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    """Validate and preserve one ordered active control subspace."""

    if values is None:
        return None
    active = [str(value).strip() for value in values]
    if not active or any(not value or "/" in value for value in active):
        raise ValueError("active controls must be non-empty HDF5-safe names")
    if len(set(active)) != len(active):
        raise ValueError("active controls must be unique")
    return active


def _source_taper(value: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Validate one physical source-proximity sensitivity taper."""

    if value is None:
        return None
    missing = {"d0", "d1"}.difference(value)
    if missing:
        raise ValueError("source taper requires d0 and d1")
    d0 = float(value["d0"])
    d1 = float(value["d1"])
    units = str(value.get("units", "m")).strip()
    if not np.isfinite(d0) or d0 < 0.0:
        raise ValueError("source taper d0 must be finite and nonnegative")
    if not np.isfinite(d1) or d1 <= d0:
        raise ValueError("source taper d1 must be finite and greater than d0")
    if not units:
        raise ValueError("source taper units must be non-empty")
    return {"d0": d0, "d1": d1, "units": units}


def _spatial_window(value: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Validate one axis-aligned physical sensitivity window."""

    if value is None:
        return None
    missing = {"axis", "minimum", "maximum"}.difference(value)
    if missing:
        raise ValueError("spatial window requires axis, minimum, and maximum")
    axis = str(value["axis"]).strip().lower()
    minimum = float(value["minimum"])
    maximum = float(value["maximum"])
    taper = float(value.get("taper", 0.0))
    units = str(value.get("units", "m")).strip()
    if axis not in {"x", "y", "z"}:
        raise ValueError("spatial window axis must be x, y, or z")
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(
            "spatial window bounds must be finite with maximum greater than minimum"
        )
    if not np.isfinite(taper) or taper < 0.0 or 2.0 * taper > maximum - minimum:
        raise ValueError(
            "spatial window taper must be finite, nonnegative, and no greater "
            "than half the window width"
        )
    if not units:
        raise ValueError("spatial window units must be non-empty")
    return {
        "axis": axis,
        "minimum": minimum,
        "maximum": maximum,
        "taper": taper,
        "units": units,
    }


def _job_path(
    value: Optional[Union[str, Path]],
    ctx: Optional[ExportContext],
    project_relative: bool,
) -> Optional[str]:
    """Serialize one optional job input/output path."""

    if value is None:
        return None
    path = Path(value)
    if ctx is not None and project_relative:
        return str(ctx.relative_to_project(path))
    return str(path)


def _observed_derivative_groups(
    value: Optional[
        Union[
            ObservedTraceDerivatives,
            str,
            Path,
            Mapping[str, Any],
        ]
    ],
    names: Sequence[str],
) -> Dict[str, ObservedTraceDerivatives]:
    """Normalize one shared or receiver-group keyed df reference mapping."""

    if value is None:
        return {}
    if isinstance(value, ObservedTraceDerivatives) or isinstance(value, (str, Path)):
        derivative = ObservedTraceDerivatives.from_value(value)
        return {name: derivative for name in names}
    if not isinstance(value, Mapping):
        raise TypeError(
            "observed_derivatives must be a derivative reference or mapping"
        )
    if set(value) == {"df"}:
        derivative = ObservedTraceDerivatives.from_value(value)
        return {name: derivative for name in names}
    unknown = sorted(set(value).difference(names))
    if unknown:
        raise ValueError(
            "observed_derivatives contains unknown receiver group(s): "
            f"{', '.join(unknown)}"
        )
    return {
        str(name): ObservedTraceDerivatives.from_value(derivative)
        for name, derivative in value.items()
    }


def _derivatives_to_fs(
    derivatives: ObservedTraceDerivatives,
    ctx: Optional[ExportContext],
    project_relative: bool,
) -> Dict[str, Any]:
    """Serialize one df reference while preserving project-relative paths."""

    payload = derivatives.to_fs()
    df = payload["df"]
    if isinstance(df, Mapping):
        df = dict(df)
        payload["df"] = df
        df["file"] = _job_path(df["file"], ctx, project_relative)
    else:
        payload["df"] = _job_path(df, ctx, project_relative)
    return payload


@overload
def _resolve_saved_job_path(
    value: Union[str, Path],
    *,
    base_path: Optional[Union[str, Path]],
    project_path: Optional[Union[str, Path]],
    source_project: Optional[Union[str, Path]],
) -> Path: ...


@overload
def _resolve_saved_job_path(
    value: None,
    *,
    base_path: Optional[Union[str, Path]],
    project_path: Optional[Union[str, Path]],
    source_project: Optional[Union[str, Path]],
) -> None: ...


def _resolve_saved_job_path(
    value: Optional[Union[str, Path]],
    *,
    base_path: Optional[Union[str, Path]],
    project_path: Optional[Union[str, Path]],
    source_project: Optional[Union[str, Path]],
) -> Optional[Path]:
    """Resolve a saved project-relative control workflow path."""

    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        if source_project is not None and project_path is not None:
            try:
                relative = path.relative_to(Path(source_project).expanduser())
            except ValueError:
                pass
            else:
                return Path(project_path).expanduser().resolve() / relative
        return path
    if project_path is not None:
        return Path(project_path).expanduser().resolve() / path
    if base_path is not None:
        project_root = BaseJob._project_root_from_job_path(Path(base_path))
        if project_root is not None:
            return project_root / path
        return Path(base_path).expanduser().resolve() / path
    return path


@overload
def _control_path(value: Union[str, Path], simulation: BaseSimulation) -> Path: ...


@overload
def _control_path(value: None, simulation: BaseSimulation) -> None: ...


def _control_path(
    value: Optional[Union[str, Path]], simulation: BaseSimulation
) -> Optional[Path]:
    return _resolve_saved_job_path(
        value,
        base_path=None,
        project_path=getattr(simulation, "project_path", None),
        source_project=None,
    )


@register_class
class BornControlSensitivityJob(BaseJob):
    """Apply the native physical-operator JVP to one control direction."""

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: Union[Sequence[Union[float, complex]], np.ndarray],
        *,
        direction: Union[str, Path],
        current: Optional[Union[str, Path]] = None,
        active: Optional[Sequence[str]] = None,
        source_taper: Optional[Mapping[str, Any]] = None,
        spatial_window: Optional[Mapping[str, Any]] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
    ):
        super().__init__(
            name,
            simulation,
            "born",
            _normalized_frequencies(f_list),
            JobOutputs(outputs),
        )
        self.direction = _control_path(direction, simulation)
        self.current = _control_path(current, simulation)
        self.active = _active_controls(active)
        self.source_taper = _source_taper(source_taper)
        self.spatial_window = _spatial_window(spatial_window)

    @property
    def trace_outputs(self) -> TraceOutputSpec:
        """Describe the incremental receiver groups written by Sauce Born jobs."""

        baseline = super().trace_outputs
        groups = [f"{group}_inc" for group in baseline.groups]
        components = []
        for component in baseline.components:
            group, separator, name = component.partition(":")
            components.append(f"{group}_inc:{name}" if separator else component)
        return TraceOutputSpec(
            path=baseline.path,
            frequencies=baseline.frequencies,
            groups=groups,
            components=components,
            sources=baseline.sources,
            wavefields=baseline.wavefields,
        )

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize the Born workflow and direction-file contract."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        payload["control_sensitivities"] = {
            # Sauce opens control HDF5 files directly rather than through its
            # project-relative input resolver. Absolute paths remain relocatable
            # because from_fs remaps paths below the serialized source project.
            "direction": _job_path(self.direction, ctx, False),
            **({"active": self.active} if self.active is not None else {}),
            **(
                {"source_taper": self.source_taper}
                if self.source_taper is not None
                else {}
            ),
            **(
                {"spatial_window": self.spatial_window}
                if self.spatial_window is not None
                else {}
            ),
            **(
                {"current": _job_path(self.current, ctx, False)}
                if self.current is not None
                else {}
            ),
        }
        return payload

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash the current model and JVP direction consumed by this job."""

        inputs = {"direction": self._path_content_fingerprint(self.direction)}
        if self.current is not None:
            inputs["current"] = self._path_content_fingerprint(self.current)
        return inputs

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "BornControlSensitivityJob":
        """Deserialize a native control-sensitivity Born job."""

        source_project = data.get("project_path")
        resolved_project = project_path or source_project
        sim = cls._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=resolved_project,
            source_project=source_project,
        )
        config = data["control_sensitivities"]
        job = cls(
            data["name"],
            sim,
            cls._decode_frequencies(data["f_list"]),
            direction=_resolve_saved_job_path(
                config["direction"],
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            ),
            current=_resolve_saved_job_path(
                config.get("current"),
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            ),
            active=config.get("active"),
            source_taper=config.get("source_taper"),
            spatial_window=config.get("spatial_window"),
            outputs=JobOutputs.from_fs(data.get("Outputs")),
        )
        job._job_id = data.get("job_id")
        return job


@register_class
class RTMControlSensitivityJob(BaseJob):
    """Write a material control gradient, optionally including DPG Gram dependence."""

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: Union[Sequence[Union[float, complex]], np.ndarray],
        *,
        observed: Union[str, Path, Mapping[str, Union[str, Path]]],
        gradient: Union[str, Path],
        objective_file: Optional[Union[str, Path]] = None,
        gram_derivative: str = "frozen",
        current: Optional[Union[str, Path]] = None,
        active: Optional[Sequence[str]] = None,
        source_taper: Optional[Mapping[str, Any]] = None,
        spatial_window: Optional[Mapping[str, Any]] = None,
        objective: Optional[Mapping[str, Any]] = None,
        comparison: Optional[Union[MisfitComparison, Mapping[str, Any]]] = None,
        observed_derivatives: Optional[
            Union[
                ObservedTraceDerivatives,
                str,
                Path,
                Mapping[str, Any],
            ]
        ] = None,
        preprocess: Optional[Sequence[Union[PreprocessHook, Mapping[str, Any]]]] = None,
        weights: Optional[Sequence[float]] = None,
        smoothing: Optional[Union[VariationalSmoothing, Mapping[str, Any]]] = None,
        raw_gradient: Optional[Union[str, Path]] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
    ):
        super().__init__(
            name,
            simulation,
            "rtm",
            _normalized_frequencies(f_list),
            JobOutputs(outputs),
        )
        if isinstance(observed, Mapping):
            if not observed:
                raise ValueError("RTM control sensitivity requires observed data")
            self.observed = {
                str(name): _control_path(path, simulation)
                for name, path in observed.items()
            }
        else:
            self.observed = {"surface": _control_path(observed, simulation)}
        self.gradient = _control_path(gradient, simulation)
        self._objective_file = _control_path(objective_file, simulation)
        if gram_derivative not in {"frozen", "total"}:
            raise ValueError("gram_derivative must be 'frozen' or 'total'")
        self.gram_derivative = gram_derivative
        self.current = _control_path(current, simulation)
        self.active = _active_controls(active)
        self.source_taper = _source_taper(source_taper)
        self.spatial_window = _spatial_window(spatial_window)
        self.objective = dict(objective or {"kind": "l2"})
        self.comparison = MisfitComparison.from_value(comparison)
        if self.gram_derivative == "total":
            if self.source_taper is not None or self.spatial_window is not None:
                raise ValueError(
                    "total Gram derivatives do not support source_taper or spatial_window"
                )
            if (
                self.comparison is not None
                and self.comparison.kind == "phase_derivative"
            ):
                raise ValueError(
                    "total Gram derivatives do not support phase_derivative comparisons"
                )
        self.observed_derivatives = _observed_derivative_groups(
            observed_derivatives,
            tuple(self.observed),
        )
        if (
            self.comparison is not None
            and self.comparison.kind == "phase_derivative"
            and set(self.observed_derivatives) != set(self.observed)
        ):
            raise ValueError(
                "phase-derivative comparison requires observed_derivatives for "
                "every receiver group"
            )
        self.observed_derivatives = {
            name: spec.resolved(lambda path: _control_path(path, simulation))
            for name, spec in self.observed_derivatives.items()
        }
        self.preprocess = list(preprocess or [])
        if weights is None:
            self.weights = None
        else:
            frequency_weights = np.asarray(weights)
            if np.iscomplexobj(frequency_weights):
                raise ValueError("control-gradient frequency weights must be real")
            frequency_weights = np.asarray(frequency_weights, dtype=np.float64).reshape(
                -1
            )
            if frequency_weights.size != self.n_tasks or not np.all(
                np.isfinite(frequency_weights)
            ):
                raise ValueError(
                    "control-gradient weights must contain one finite value per frequency"
                )
            self.weights = frequency_weights.tolist()
        self.smoothing = VariationalSmoothing.from_value(smoothing)
        self.raw_gradient = _control_path(raw_gradient, simulation)

    def gradient_file(self, part: Optional[int] = None, *, raw: bool = False) -> Path:
        """Return the final, raw aggregate, or task-local gradient path."""

        if raw:
            if part is not None:
                raise ValueError("raw aggregate gradients do not have task parts")
            if self.raw_gradient is not None:
                return self.raw_gradient
            return self.gradient.with_name(
                f"{self.gradient.stem}_raw{self.gradient.suffix}"
            )
        if part is None:
            return self.gradient
        if int(part) < 1:
            raise ValueError("control-gradient task part must be positive")
        return self.gradient.with_name(
            f"{self.gradient.stem}_{int(part)}{self.gradient.suffix}"
        )

    def objective_file(self, part: Optional[int] = None) -> Optional[Path]:
        """Return the aggregate or one task-local scalar-objective path."""

        if self._objective_file is None:
            return None
        if part is None:
            return self._objective_file
        if int(part) < 1:
            raise ValueError("control-objective task part must be positive")
        return self._objective_file.with_name(
            f"{self._objective_file.stem}_{int(part)}{self._objective_file.suffix}"
        )

    @property
    def objective_value(self) -> float:
        """Read the aggregated scalar objective written by Sauce."""

        path = self.objective_file()
        if path is None:
            raise ValueError("this RTM job does not request a scalar objective")
        with h5py.File(path, "r") as h5:
            return float(h5["value"][()])

    def requires_postprocess(self) -> bool:
        """Return true because RTM gradients are finalized after frequency tasks."""

        return True

    def postprocess_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate or task-local control-gradient path."""

        return self.gradient_file(part)

    def postprocess_fetch_files(self) -> list[Path]:
        """Return both the exact aggregate covector and final Riesz result."""

        files = [self.gradient_file(raw=True), self.gradient_file()]
        objective = self.objective_file()
        if objective is not None:
            files.append(objective)
        return files

    def postprocess_output_exists(self) -> bool:
        """Return whether the final control gradient exists locally."""

        objective = self.objective_file()
        return self.gradient_file().is_file() and (
            objective is None or objective.is_file()
        )

    def postprocess_part_outputs_exist(self) -> bool:
        """Return whether every frequency-gradient shard exists locally."""

        gradients_exist = all(
            self.gradient_file(part).is_file() for part in range(1, self.n_tasks + 1)
        )
        if not gradients_exist:
            return False
        if self._objective_file is None:
            return True
        for part in range(1, self.n_tasks + 1):
            objective = self.objective_file(part)
            if objective is None or not objective.is_file():
                return False
        return True

    def is_run_current(self) -> bool:
        """Return whether the run and final aggregated gradient are current."""

        return super().is_run_current() and self.postprocess_output_exists()

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize the RTM control-gradient and observed-data contracts."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        payload["control_sensitivities"] = {
            "gradient": _job_path(self.gradient, ctx, False),
            **(
                {"gram_derivative": self.gram_derivative}
                if self.gram_derivative != "frozen"
                else {}
            ),
            **(
                {"objective": _job_path(self._objective_file, ctx, False)}
                if self._objective_file is not None
                else {}
            ),
            **({"active": self.active} if self.active is not None else {}),
            **(
                {"source_taper": self.source_taper}
                if self.source_taper is not None
                else {}
            ),
            **(
                {"spatial_window": self.spatial_window}
                if self.spatial_window is not None
                else {}
            ),
            **(
                {"current": _job_path(self.current, ctx, False)}
                if self.current is not None
                else {}
            ),
            **({"weights": self.weights} if self.weights is not None else {}),
            **(
                {"Smoothing": self.smoothing.to_fs(include_illumination=False)}
                if self.smoothing is not None
                else {}
            ),
            **(
                {"raw_gradient": _job_path(self.raw_gradient, ctx, False)}
                if self.raw_gradient is not None
                else {}
            ),
        }
        payload["Image"] = {
            "schema": "fs-imaging-1",
            "name": self.name,
            "misfit": {
                "objective": dict(self.objective),
                **(
                    {"comparison": self.comparison.to_fs()}
                    if self.comparison is not None
                    else {}
                ),
                "preprocess": {
                    # A native control VJP is the linear transpose used by
                    # matrix-free J.T J products.  In particular, it must not
                    # inherit image defaults such as illumination
                    # normalization, which depend on the modeled fields and
                    # would make the returned covector nonlinear.
                    "include_defaults": False,
                    "hooks": _preprocess_to_fs(
                        self.preprocess,
                        ctx,
                        scope="misfit",
                    ),
                },
                "receiver_groups": [
                    {
                        "name": name,
                        "observed": _job_path(path, ctx, project_relative),
                        **(
                            {
                                "observed_derivatives": _derivatives_to_fs(
                                    self.observed_derivatives[name],
                                    ctx,
                                    project_relative,
                                )
                            }
                            if name in self.observed_derivatives
                            else {}
                        ),
                    }
                    for name, path in self.observed.items()
                ],
            },
        }
        return payload

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash the current model and observed data consumed by this job."""

        inputs = {
            "observed": {
                name: self._path_content_fingerprint(path)
                for name, path in sorted(self.observed.items())
            },
            "observed_derivatives": {
                name: _derivative_input_fingerprint(
                    derivatives,
                    self._path_content_fingerprint,
                )
                for name, derivatives in sorted(self.observed_derivatives.items())
            },
        }
        if self.current is not None:
            inputs["current"] = self._path_content_fingerprint(self.current)
        return inputs

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "RTMControlSensitivityJob":
        """Deserialize a native control-sensitivity RTM job."""

        source_project = data.get("project_path")
        resolved_project = project_path or source_project
        sim = cls._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=resolved_project,
            source_project=source_project,
        )
        config = data["control_sensitivities"]
        misfit = data["Image"]["misfit"]
        observed = {
            group["name"]: _resolve_saved_job_path(
                group["observed"],
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            )
            for group in misfit["receiver_groups"]
        }
        observed_derivatives = {}
        for group in misfit["receiver_groups"]:
            derivatives = ObservedTraceDerivatives.from_value(
                group.get("observed_derivatives")
            )
            if derivatives is not None:
                observed_derivatives[group["name"]] = derivatives.resolved(
                    lambda path: _resolve_saved_job_path(
                        path,
                        base_path=base_path,
                        project_path=resolved_project,
                        source_project=source_project,
                    )
                )
        job = cls(
            data["name"],
            sim,
            cls._decode_frequencies(data["f_list"]),
            observed=observed,
            gradient=_resolve_saved_job_path(
                config["gradient"],
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            ),
            objective_file=_resolve_saved_job_path(
                config.get("objective"),
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            ),
            current=_resolve_saved_job_path(
                config.get("current"),
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            ),
            active=config.get("active"),
            source_taper=config.get("source_taper"),
            spatial_window=config.get("spatial_window"),
            objective=misfit.get("objective"),
            gram_derivative=config.get("gram_derivative", "frozen"),
            comparison=misfit.get("comparison"),
            observed_derivatives=observed_derivatives or None,
            preprocess=_preprocess_from_fs(
                misfit.get("preprocess", {}).get("hooks", [])
            ),
            weights=config.get("weights"),
            smoothing=config.get("Smoothing"),
            raw_gradient=_resolve_saved_job_path(
                config.get("raw_gradient"),
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            ),
            outputs=JobOutputs.from_fs(data.get("Outputs")),
        )
        job._job_id = data.get("job_id")
        return job


@register_class
class TimeReversalFocusJob(RTMControlSensitivityJob):
    """Evaluate acoustic time-reversal focusing and its exact control gradient.

    The objective backpropagates observed data directly, so evaluating it does
    not require a modeled forward wavefield. Its exact gradient performs one
    additional solve per frequency with a regularized inverse-distance pressure
    functional centered on each encoded source field. Receiver coefficients are
    conjugated for time reversal while the attenuating complex frequency is
    retained unchanged.
    """

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: Union[Sequence[Union[float, complex]], np.ndarray],
        *,
        observed: Union[str, Path, Mapping[str, Union[str, Path]]],
        gradient: Union[str, Path],
        objective_file: Union[str, Path],
        softening: float,
        distance_power: float = 1.0,
        current: Optional[Union[str, Path]] = None,
        active: Optional[Sequence[str]] = None,
        spatial_window: Optional[Mapping[str, Any]] = None,
        weights: Optional[Sequence[float]] = None,
        preprocess: Optional[Sequence[Union[PreprocessHook, Mapping[str, Any]]]] = None,
        smoothing: Optional[Union[VariationalSmoothing, Mapping[str, Any]]] = None,
        raw_gradient: Optional[Union[str, Path]] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
    ):
        super().__init__(
            name,
            simulation,
            f_list,
            observed=observed,
            gradient=gradient,
            current=current,
            active=active,
            spatial_window=spatial_window,
            objective={"kind": "l2"},
            preprocess=preprocess,
            weights=weights,
            smoothing=smoothing,
            raw_gradient=raw_gradient,
            outputs=outputs,
        )
        self.workflow = "focus"
        self._focus_objective_file = _control_path(objective_file, simulation)
        self.softening = float(softening)
        self.distance_power = float(distance_power)
        if not np.isfinite(self.softening) or self.softening <= 0:
            raise ValueError("focus softening must be finite and positive")
        if not np.isfinite(self.distance_power) or self.distance_power <= 0:
            raise ValueError("focus distance_power must be finite and positive")

    def focus_objective_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate or one task-local scalar-objective path."""

        if part is None:
            return self._focus_objective_file
        if int(part) < 1:
            raise ValueError("focus-objective task part must be positive")
        return self._focus_objective_file.with_name(
            f"{self._focus_objective_file.stem}_{int(part)}"
            f"{self._focus_objective_file.suffix}"
        )

    @property
    def objective_value(self) -> float:
        """Read the aggregated scalar objective written by Sauce."""

        with h5py.File(self.focus_objective_file(), "r") as h5:
            return float(h5["value"][()])

    def postprocess_fetch_files(self) -> list[Path]:
        """Return finalized objective and gradient artifacts."""

        return [*super().postprocess_fetch_files(), self.focus_objective_file()]

    def postprocess_output_exists(self) -> bool:
        """Return whether both final focus products exist locally."""

        return (
            super().postprocess_output_exists()
            and self.focus_objective_file().is_file()
        )

    def postprocess_part_outputs_exist(self) -> bool:
        """Return whether every gradient and objective shard exists locally."""

        return super().postprocess_part_outputs_exist() and all(
            self.focus_objective_file(part).is_file()
            for part in range(1, self.n_tasks + 1)
        )

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize the linear focus functional and its bulk-data inputs."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        payload["focus"] = {
            "objective": _job_path(self._focus_objective_file, ctx, False),
            "softening": self.softening,
            "distance_power": self.distance_power,
        }
        return payload

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "TimeReversalFocusJob":
        """Deserialize a time-reversal focus job."""

        if (
            data.get("control_sensitivities", {}).get("gram_derivative", "frozen")
            != "frozen"
        ):
            raise ValueError(
                "total Gram derivatives do not support time-reversal focus"
            )

        source_project = data.get("project_path")
        resolved_project = project_path or source_project
        common = RTMControlSensitivityJob.from_fs(
            data,
            base_path=base_path,
            project_path=project_path,
        )
        focus = data["focus"]
        job = cls(
            data["name"],
            common.simulation,
            common.f_list,
            observed=common.observed,
            gradient=common.gradient,
            objective_file=_resolve_saved_job_path(
                focus["objective"],
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            ),
            softening=focus["softening"],
            distance_power=focus.get("distance_power", 1.0),
            current=common.current,
            active=common.active,
            spatial_window=common.spatial_window,
            weights=common.weights,
            preprocess=common.preprocess,
            smoothing=common.smoothing,
            raw_gradient=common.raw_gradient,
            outputs=common.outputs,
        )
        job._job_id = data.get("job_id")
        return job
