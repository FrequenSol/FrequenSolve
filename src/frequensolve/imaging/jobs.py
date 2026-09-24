"""Low-level Sauce imaging jobs.

Four job classes cover every Sauce imaging workflow:

- :class:`FWIOperatorJob` — ``fwi_operator`` actions over the shared control
  registry (calibrate, linearize, jvp, vjp, normal, wri, solve), including
  auxiliary model extension and joint reflectivity.
- :class:`ControlGradientJob` — native ``rtm`` / ``born`` / ``focus`` control
  sensitivities.
- :class:`ImageKernelJob` — Cartesian image kernels (``rtm`` / ``born`` /
  ``lsrtm_gradient`` with ``Imaging.grid``).
- :class:`SmoothJob` — Sauce's ``--smooth`` postprocess run on its own over an
  existing job's per-task gradient parts or one explicit vector
  (``control_sensitivities.input``).

Path conventions: input files (directions, duals, baselines) resolve relative
to the project root; output files resolve relative to the job result directory.
Task-suffixed outputs follow Sauce's ``<stem>_<task><ext>`` rule.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.imaging._artifacts import (
    ControlVectorFile,
    ImageSet,
    SmoothingConfig,
    qualified_block_name,
    unqualified_block_name,
)
from frequensolve.simulation.jobs.artifacts import TraceOutputSpec
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.outputs import JobOutputs, Output
from frequensolve.simulation.simulation import BaseSimulation
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import ExportContext

__all__ = [
    "FWIOperatorJob",
    "ControlGradientJob",
    "ImageKernelJob",
    "ImageSpec",
    "SmoothJob",
    "FWI_ACTIONS",
]

FWI_ACTIONS: Tuple[str, ...] = (
    "calibrate",
    "linearize",
    "jvp",
    "vjp",
    "normal",
    "wri",
    "solve",
)
_STATE_ACTIONS = frozenset({"linearize", "jvp", "vjp", "normal", "solve"})
_SOURCE_CONTROL_ACTIONS = frozenset({"linearize", "jvp", "vjp", "normal"})
_KERNEL_RESIDUALS_FWI = ("derivative", "window")
_KERNEL_RESIDUALS_CONTROL = ("derivative", "window", "jet")
_REFLECTIVITY_PARAMETERIZATIONS = ("vp_ip", "vp_vs_ip", "ip_is_rho")
_FIELD_RETENTION = ("none", "forward", "adjoint", "all")
_IMAGE_WORKFLOWS = ("rtm", "born", "lsrtm_gradient")
_CONTROL_KINDS = ("rtm", "born", "focus")
_FOCUS_KINDS = ("trfwi", "weft")
_DEFAULT_MISFIT: Dict[str, Any] = {"objective": {"kind": "l2"}}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalized_frequencies(
    values: Union[Sequence[Union[float, complex]], np.ndarray],
) -> list:
    """Normalize Laplace samples to the solver's negative-imaginary convention."""

    frequencies = np.asarray(values)
    if frequencies.size == 0:
        raise ValueError("imaging jobs require at least one frequency")
    if np.iscomplexobj(frequencies):
        return [complex(value.real, -abs(value.imag)) for value in frequencies]
    return frequencies.astype(float).tolist()


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


def _resolve_saved_job_path(
    value: Optional[Union[str, Path]],
    *,
    base_path: Optional[Union[str, Path]],
    project_path: Optional[Union[str, Path]],
    source_project: Optional[Union[str, Path]],
) -> Optional[Path]:
    """Resolve a saved project-relative workflow path."""

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


def _input_path(
    value: Optional[Union[str, Path]], simulation: BaseSimulation
) -> Optional[Path]:
    """Resolve an input path relative to the simulation's project root."""

    return _resolve_saved_job_path(
        value,
        base_path=None,
        project_path=getattr(simulation, "project_path", None),
        source_project=None,
    )


def _output_path(value: Optional[Union[str, Path]], result_dir: Path) -> Optional[Path]:
    """Resolve an output path relative to the job result directory."""

    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(result_dir) / path


def _task_path(path: Path, task: Optional[int]) -> Path:
    """Return Sauce's ``<stem>_<task><ext>`` sibling of ``path``."""

    if task is None:
        return path
    if int(task) < 1:
        raise ValueError("task numbers are one-based and positive")
    return path.with_name(f"{path.stem}_{int(task)}{path.suffix}")


def _raw_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_raw{path.suffix}")


def _active_controls(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Validate one ordered unqualified material-control subspace."""

    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError("active controls must be a sequence of names, not a string")
    active = [str(value).strip() for value in values]
    if not active or any(not value or "/" in value for value in active):
        raise ValueError("active controls must be non-empty HDF5-safe names")
    if len(set(active)) != len(active):
        raise ValueError("active controls must be unique")
    return active


def _qualified_active(values: Sequence[str]) -> List[str]:
    """Validate one ordered qualified registry subspace (may be empty)."""

    if isinstance(values, str):
        raise TypeError("active blocks must be a sequence of names, not a string")
    active = [qualified_block_name(value) for value in values]
    if len(set(active)) != len(active):
        raise ValueError("active blocks must be unique")
    return active


def _source_taper(value: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
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


def _spatial_window(value: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
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


def _frequency_weights(
    values: Optional[Sequence[float]], n_tasks: int, *, label: str
) -> Optional[List[float]]:
    """Validate one nonnegative per-frequency weight list."""

    if values is None:
        return None
    weights = np.asarray(values)
    if np.iscomplexobj(weights):
        raise ValueError(f"{label} weights must be real")
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        weights.size != n_tasks
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError(
            f"{label} weights must contain one finite nonnegative value per frequency"
        )
    return weights.tolist()


def _misfit_to_fs(
    misfit: Any,
    ctx: Optional[ExportContext],
    *,
    project_relative: bool,
) -> Optional[Dict[str, Any]]:
    """Serialize a misfit mapping or an object exposing ``to_fs``."""

    if misfit is None:
        return None
    if isinstance(misfit, Mapping):
        return copy.deepcopy(dict(misfit))
    to_fs = getattr(misfit, "to_fs", None)
    if to_fs is None or not callable(to_fs):
        raise TypeError("misfit must be a mapping or an object with a to_fs method")
    kwargs: Dict[str, Any] = {}
    try:
        parameters = inspect.signature(to_fs).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "project_relative" in parameters:
        kwargs["project_relative"] = project_relative
    if "ctx" in parameters:
        kwargs["ctx"] = ctx
    return dict(to_fs(**kwargs))


def _misfit_comparison_kinds(misfit: Optional[Mapping[str, Any]]) -> List[str]:
    """Return every comparison kind named by a serialized misfit."""

    if not misfit:
        return []
    kinds = []
    comparison = misfit.get("comparison")
    if isinstance(comparison, Mapping) and comparison.get("kind"):
        kinds.append(str(comparison["kind"]))
    for term in misfit.get("objective_terms") or ():
        comparison = term.get("comparison") if isinstance(term, Mapping) else None
        if isinstance(comparison, Mapping) and comparison.get("kind"):
            kinds.append(str(comparison["kind"]))
    return kinds


def _kernel_derivative(
    value: Optional[Mapping[str, Any]], *, residuals: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Validate a ``kernel_derivative`` request against the job-level contract."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("kernel_derivative must be a mapping")
    allowed = {
        "order",
        "axis",
        "residual",
        "window",
        "order_weights",
        "source_derivative",
    }
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(
            f"unsupported kernel_derivative option(s): {', '.join(unknown)}"
        )
    payload: Dict[str, Any] = {}
    if "order" in value:
        order = int(value["order"])
        if order < 0 or order > 4:
            raise ValueError("kernel_derivative order must be between 0 and 4")
        payload["order"] = order
    if "axis" in value:
        axis = str(value["axis"]).strip().lower()
        if axis not in {"fourier", "laplace"}:
            raise ValueError("kernel_derivative axis must be 'fourier' or 'laplace'")
        payload["axis"] = axis
    residual = str(value.get("residual", "")).strip().lower()
    if not residual:
        raise ValueError(
            "kernel_derivative requires residual in "
            f"{{{', '.join(residuals)}}} for this workflow"
        )
    if residual not in residuals:
        raise ValueError(
            f"kernel_derivative residual {residual!r} is not allowed here; "
            f"choose one of {', '.join(residuals)}"
        )
    payload["residual"] = residual
    if "window" in value:
        window = [float(v) for v in value["window"]]
        if not 1 <= len(window) <= 5 or window[-1] == 0.0:
            raise ValueError(
                "kernel_derivative window needs 1 to 5 coefficients with a nonzero last term"
            )
        payload["window"] = window
    elif residual == "window":
        raise ValueError(
            "kernel_derivative residual 'window' requires window coefficients"
        )
    if "order_weights" in value:
        weights = [float(v) for v in value["order_weights"]]
        if (
            not 1 <= len(weights) <= 5
            or any(w < 0 for w in weights)
            or weights[-1] <= 0
        ):
            raise ValueError(
                "kernel_derivative order_weights must be 1 to 5 nonnegative values "
                "with a positive last weight"
            )
        payload["order_weights"] = weights
    if "source_derivative" in value:
        policy = str(value["source_derivative"]).strip().lower()
        if policy not in {"frozen", "total"}:
            raise ValueError(
                "kernel_derivative source_derivative must be frozen or total"
            )
        payload["source_derivative"] = policy
    return payload


def _observed_groups(
    observed: Optional[Union[str, Path, Mapping[str, Union[str, Path]]]],
    simulation: BaseSimulation,
    *,
    fallback: str = "surface",
) -> Optional[Dict[str, Optional[Path]]]:
    """Normalize observed data into ``receiver group -> path`` (project-resolved)."""

    if observed is None:
        return None
    if isinstance(observed, Mapping):
        if not observed:
            raise ValueError("observed data mapping must not be empty")
        return {
            str(name): _input_path(path, simulation) for name, path in observed.items()
        }
    path = _input_path(observed, simulation)
    names = _receiver_group_names(simulation) or [fallback]
    return {name: path for name in names}


def _receiver_group_names(simulation: BaseSimulation) -> List[str]:
    acquisition = getattr(simulation, "acquisition", None)
    groups = getattr(acquisition, "receiver_groups", None) or ()
    return [str(group.name) for group in groups]


def _incremental_trace_outputs(
    baseline: TraceOutputSpec, *, keep_baseline: bool
) -> TraceOutputSpec:
    incremental_groups = [f"{group}_inc" for group in baseline.groups]
    incremental_components = []
    for component in baseline.components:
        group, separator, name = component.partition(":")
        incremental_components.append(f"{group}_inc:{name}" if separator else component)
    if keep_baseline:
        return replace(
            baseline,
            groups=[*baseline.groups, *incremental_groups],
            components=[*baseline.components, *incremental_components],
        )
    return replace(
        baseline, groups=incremental_groups, components=incremental_components
    )


def _min_support(value: Any) -> Optional[float]:
    """Validate ``fwi_operator.controls.min_support`` (a non-negative number)."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("min_support must be a number")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("min_support must be a finite non-negative number")
    return threshold


def _postprocess_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


class _ImagingJobBase(BaseJob):
    """Shared serialization behavior for the imaging job family."""

    _EXTRA_FINGERPRINT_FIELDS: ClassVar[Tuple[str, ...]] = (
        "Imaging",
        "fwi_operator",
        "kernel_derivative",
    )

    @staticmethod
    def _fingerprint_job_payload(
        job_data: Dict[str, Any],
        *,
        include_frequencies: bool = False,
    ) -> Dict[str, Any]:
        """Select solver-relevant fields, including the new imaging blocks."""

        payload = BaseJob._fingerprint_job_payload(
            job_data, include_frequencies=include_frequencies
        )
        for field in _ImagingJobBase._EXTRA_FINGERPRINT_FIELDS:
            if field in job_data and field not in payload:
                payload[field] = job_data[field]
        return payload

    @staticmethod
    def effective_output_request_payload_from_fs(
        job_payload: Mapping[str, Any],
        *,
        simulation_payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract the output request, including the new imaging blocks."""

        request = BaseJob.effective_output_request_payload_from_fs(
            job_payload, simulation_payload=simulation_payload
        )
        for field in _ImagingJobBase._EXTRA_FINGERPRINT_FIELDS:
            if field in job_payload and field not in request:
                request[field] = job_payload[field]
        return request

    @classmethod
    def _load_context(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]],
        project_path: Optional[Union[str, Path]],
    ):
        """Load the simulation and return ``(simulation, resolve)``."""

        source_project = data.get("project_path")
        resolved_project = project_path or source_project
        simulation = cls._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=resolved_project,
            source_project=source_project,
        )

        def resolve(value: Optional[Union[str, Path]]) -> Optional[Path]:
            return _resolve_saved_job_path(
                value,
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            )

        return simulation, resolve

    @staticmethod
    def _finish_load(job: "BaseJob", data: Mapping[str, Any]) -> None:
        job.preserve_task_outputs = data.get("preserve_task_outputs", False)
        if not isinstance(job.preserve_task_outputs, bool):
            raise TypeError("preserve_task_outputs must be a boolean")
        job._job_id = data.get("job_id")


# ---------------------------------------------------------------------------
# FWIOperatorJob
# ---------------------------------------------------------------------------


def _validate_wri(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("wri must be a mapping")
    allowed = {
        "penalty",
        "data_scale",
        "receiver_group",
        "receiver_groups",
        "curvature",
    }
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported wri option(s): {', '.join(unknown)}")
    if "penalty" not in value:
        raise ValueError("wri requires a positive penalty")
    penalty = float(value["penalty"])
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("wri penalty must be finite and positive")
    payload: Dict[str, Any] = {"penalty": penalty}
    if "data_scale" in value:
        scale = value["data_scale"]
        if isinstance(scale, str):
            if scale.strip().lower() != "auto":
                raise ValueError("wri data_scale must be 'auto' or a positive number")
            payload["data_scale"] = "auto"
        else:
            scale = float(scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("wri data_scale must be 'auto' or a positive number")
            payload["data_scale"] = scale
    if "receiver_group" in value and "receiver_groups" in value:
        raise ValueError(
            "wri receiver_group and receiver_groups are mutually exclusive"
        )
    if "receiver_group" in value:
        group = str(value["receiver_group"]).strip()
        if not group:
            raise ValueError("wri receiver_group must be non-empty")
        payload["receiver_group"] = group
    if "receiver_groups" in value:
        groups = [str(g).strip() for g in value["receiver_groups"]]
        if not groups or any(not g for g in groups) or len(set(groups)) != len(groups):
            raise ValueError("wri receiver_groups must be unique non-empty names")
        payload["receiver_groups"] = groups
    if "curvature" in value:
        curvature = str(value["curvature"]).strip().lower()
        if curvature not in {"fixed_wavefield", "joint_schur"}:
            raise ValueError("wri curvature must be fixed_wavefield or joint_schur")
        payload["curvature"] = curvature
    return payload


def _validate_source_controls(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("source_controls must be a mapping")
    allowed = {"location_method", "reference_step"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported source_controls option(s): {', '.join(unknown)}")
    payload: Dict[str, Any] = {}
    if "location_method" in value:
        method = str(value["location_method"]).strip().lower()
        if method not in {"analytic", "local_fd4"}:
            raise ValueError(
                "source_controls location_method must be analytic or local_fd4"
            )
        payload["location_method"] = method
    if "reference_step" in value:
        step = float(value["reference_step"])
        if not 1e-6 <= step <= 1e-2:
            raise ValueError(
                "source_controls reference_step must be within [1e-6, 1e-2]"
            )
        payload["reference_step"] = step
    return payload


def _validate_scale(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"value", "units"}:
        raise ValueError(f"{name} must be a {{value, units}} mapping")
    scale = float(value["value"])
    units = str(value["units"]).strip()
    if not np.isfinite(scale) or scale <= 0.0 or not units:
        raise ValueError(f"{name} requires a positive value and non-empty units")
    return {"value": scale, "units": units}


def _validate_extension_solver(
    value: Mapping[str, Any], result_dir: Path
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("extension solver must be a mapping")
    allowed = {
        "damping",
        "lag_penalty",
        "lag_scale",
        "field_scales",
        "relative_tolerance",
        "absolute_tolerance",
        "max_iterations",
        "cache_mb",
        "require_convergence",
        "solution",
        "report",
        "workspace_mb",
        "offset_penalty",
        "offset_scale",
        "max_outer_iterations",
        "max_line_search",
        "gradient_relative_tolerance",
        "gradient_absolute_tolerance",
    }
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(
            f"unsupported extension solver option(s): {', '.join(unknown)}"
        )
    missing = sorted({"damping", "solution", "report"}.difference(value))
    if missing:
        raise ValueError(f"extension solver requires {', '.join(missing)}")
    payload: Dict[str, Any] = {}
    damping = float(value["damping"])
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("extension solver damping must be finite and positive")
    payload["damping"] = damping
    for key in ("lag_penalty", "offset_penalty"):
        if key in value:
            penalty = float(value[key])
            if not np.isfinite(penalty) or penalty < 0.0:
                raise ValueError(
                    f"extension solver {key} must be finite and nonnegative"
                )
            payload[key] = penalty
            scale_key = key.replace("penalty", "scale")
            if penalty > 0.0 and scale_key not in value:
                raise ValueError(f"extension solver {key} > 0 requires {scale_key}")
    for key in ("lag_scale", "offset_scale"):
        if key in value:
            payload[key] = _validate_scale(value[key], f"extension solver {key}")
    if "field_scales" in value:
        scales = [float(v) for v in value["field_scales"]]
        if not scales or any(not np.isfinite(s) or s <= 0.0 for s in scales):
            raise ValueError("extension solver field_scales must be positive")
        payload["field_scales"] = scales
    for key in (
        "relative_tolerance",
        "absolute_tolerance",
        "cache_mb",
        "workspace_mb",
        "gradient_relative_tolerance",
        "gradient_absolute_tolerance",
    ):
        if key in value:
            number = float(value[key])
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"extension solver {key} must be finite and nonnegative"
                )
            payload[key] = number
    for key in ("max_iterations", "max_outer_iterations", "max_line_search"):
        if key in value:
            count = int(value[key])
            if count < 0:
                raise ValueError(f"extension solver {key} must be nonnegative")
            payload[key] = count
    if "require_convergence" in value:
        payload["require_convergence"] = bool(value["require_convergence"])
    payload["solution"] = _output_path(value["solution"], result_dir)
    payload["report"] = _output_path(value["report"], result_dir)
    return payload


def _validate_extension(
    value: Mapping[str, Any], simulation: BaseSimulation, result_dir: Path
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("extension must be a mapping")
    allowed = {"fields", "direction", "covector", "manifest", "solver"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported extension option(s): {', '.join(unknown)}")
    fields = list(value.get("fields") or [])
    if not fields:
        raise ValueError("extension requires at least one field")
    payload_fields = []
    seen = set()
    for field in fields:
        if not isinstance(field, Mapping) or not field.get("control"):
            raise ValueError("every extension field requires a control ID")
        control = unqualified_block_name(field["control"])
        if control in seen:
            raise ValueError(f"duplicate extension control {control!r}")
        seen.add(control)
        has_lags = "lags" in field
        has_offsets = "offsets" in field
        if has_lags == has_offsets:
            raise ValueError("extension fields need exactly one of lags or offsets")
        entry: Dict[str, Any] = {"control": control}
        if has_lags:
            lags = field["lags"]
            missing = sorted({"count", "origin", "spacing", "units"}.difference(lags))
            if missing:
                raise ValueError(f"extension lags require {', '.join(missing)}")
            count = int(lags["count"])
            spacing = float(lags["spacing"])
            units = str(lags["units"]).strip()
            if count < 1 or not np.isfinite(spacing) or spacing <= 0.0 or not units:
                raise ValueError(
                    "extension lags need count >= 1, positive spacing, units"
                )
            entry["lags"] = {
                "count": count,
                "origin": float(lags["origin"]),
                "spacing": spacing,
                "units": units,
            }
        else:
            offsets = field["offsets"]
            missing = sorted({"half_offsets", "units"}.difference(offsets))
            if missing:
                raise ValueError(f"extension offsets require {', '.join(missing)}")
            half_offsets = [
                [float(v) for v in vector] for vector in offsets["half_offsets"]
            ]
            if not half_offsets or any(not 2 <= len(v) <= 3 for v in half_offsets):
                raise ValueError("extension half_offsets must be 2D or 3D vectors")
            units = str(offsets["units"]).strip()
            if not units:
                raise ValueError("extension offsets require units")
            entry_offsets: Dict[str, Any] = {
                "half_offsets": half_offsets,
                "units": units,
            }
            if "packet_mb" in offsets:
                packet = float(offsets["packet_mb"])
                if not np.isfinite(packet) or packet <= 0.0:
                    raise ValueError("extension offsets packet_mb must be positive")
                entry_offsets["packet_mb"] = packet
            if "artifact" in offsets:
                entry_offsets["artifact"] = _input_path(offsets["artifact"], simulation)
            entry["offsets"] = entry_offsets
        payload_fields.append(entry)
    payload: Dict[str, Any] = {"fields": payload_fields}
    if value.get("direction") is not None:
        payload["direction"] = _input_path(value["direction"], simulation)
    if value.get("covector") is not None:
        payload["covector"] = _output_path(value["covector"], result_dir)
    if value.get("manifest") is not None:
        payload["manifest"] = _output_path(value["manifest"], result_dir)
    if value.get("solver") is not None:
        payload["solver"] = _validate_extension_solver(value["solver"], result_dir)
    return payload


def _validate_reflectivity(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("reflectivity must be a mapping")
    allowed = {"parameterization", "workspace_mb", "fields"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported reflectivity option(s): {', '.join(unknown)}")
    parameterization = str(value.get("parameterization", "")).strip().lower()
    if parameterization not in _REFLECTIVITY_PARAMETERIZATIONS:
        raise ValueError(
            "reflectivity parameterization must be one of "
            + ", ".join(_REFLECTIVITY_PARAMETERIZATIONS)
        )
    fields = list(value.get("fields") or [])
    if not fields:
        raise ValueError("reflectivity requires at least one field")
    max_axis = 2 if parameterization == "vp_ip" else 3
    payload_fields = []
    names = set()
    for field in fields:
        if not isinstance(field, Mapping):
            raise TypeError("reflectivity fields must be mappings")
        missing = sorted({"name", "layer", "axis"}.difference(field))
        if missing:
            raise ValueError(f"reflectivity fields require {', '.join(missing)}")
        name = str(field["name"]).strip()
        if not name or not (name[0].isalpha() and name.replace("_", "").isalnum()):
            raise ValueError(
                f"reflectivity field name {name!r} must match [A-Za-z][A-Za-z0-9_]*"
            )
        if name in names:
            raise ValueError(f"duplicate reflectivity field {name!r}")
        names.add(name)
        layer = int(field["layer"])
        axis = int(field["axis"])
        if layer < 1:
            raise ValueError("reflectivity layer is one-based and positive")
        if not 1 <= axis <= max_axis:
            raise ValueError(
                f"reflectivity axis must be between 1 and {max_axis} for {parameterization}"
            )
        has_basis = field.get("basis") is not None
        has_control = field.get("control") is not None
        if has_basis == has_control:
            raise ValueError("reflectivity fields need exactly one of basis or control")
        entry: Dict[str, Any] = {"name": name, "layer": layer, "axis": axis}
        if has_basis:
            entry["basis"] = unqualified_block_name(field["basis"])
        else:
            control = field["control"]
            control_payload = (
                control.to_fs()
                if hasattr(control, "to_fs")
                else copy.deepcopy(dict(control))
            )
            entry["control"] = control_payload
        payload_fields.append(entry)
    payload: Dict[str, Any] = {
        "parameterization": parameterization,
        "fields": payload_fields,
    }
    if "workspace_mb" in value:
        workspace = float(value["workspace_mb"])
        if not np.isfinite(workspace) or workspace <= 0.0:
            raise ValueError("reflectivity workspace_mb must be positive")
        payload["workspace_mb"] = workspace
    return payload


def _validate_reduced_normal(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("reduced_normal must be a mapping")
    allowed = {"relative_tolerance", "absolute_tolerance", "max_iterations"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported reduced_normal option(s): {', '.join(unknown)}")
    payload: Dict[str, Any] = {}
    for key in ("relative_tolerance", "absolute_tolerance"):
        if key in value:
            number = float(value[key])
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"reduced_normal {key} must be finite and nonnegative")
            payload[key] = number
    if "max_iterations" in value:
        count = int(value["max_iterations"])
        if count < 0:
            raise ValueError("reduced_normal max_iterations must be nonnegative")
        payload["max_iterations"] = count
    return payload


@register_class
class FWIOperatorJob(_ImagingJobBase):
    """One task-local ``fwi_operator`` action over the shared control registry.

    Args:
        name: Job name.
        simulation: Simulation whose model and acquisition define the registry.
        f_list: One frequency per task.
        action: One of ``calibrate``, ``linearize``, ``jvp``, ``vjp``,
            ``normal``, ``wri``, ``solve``.
        active: Ordered qualified block names (``model.<id>``,
            ``source.<i>.<quantity>``, ``reflectivity.<name>``; bare material
            IDs are qualified as ``model.<id>``). Required (may be empty) for
            state actions; must be ``None`` for ``calibrate`` and ``wri``.
        state: ``fs-objective-linearization-4`` stem: an output for
            ``linearize`` (result-relative), an input otherwise.
        covector: ``fs-control-vector-1`` output stem (``model_covector`` for
            ``wri``).
        direction: ``fs-control-vector-1`` input (``model_direction`` for
            ``wri``).
        objective_vector: ``fs-objective-vector-4`` output (``jvp``) or input
            (``vjp``, optional ``solve`` target).
        objective: Optional scalar objective report output.
        control_state: ``fs-control-state-1`` baseline input.
        state_output: ``fs-control-state-1`` baseline output (exact name in a
            single-task job, ``<stem>_<task><ext>`` per task otherwise).
        manifest: ``fs-control-registry-1`` output (same naming rule).
        min_support: Optional ``fwi_operator.controls.min_support`` relative
            support threshold (Sauce default ``1e-2``) used for the
            ``/support/<block>`` bitmasks of ``state_output`` and covectors.
        support_measure: Also export the quantized ``/support_measure/<block>``
            diagnostic (``fwi_operator.controls.support_measure``).
        model_gradient: Request the reduced background covector after ``solve``.
        cache_receiver_state: Allow the state to cache receiver values.
        source_controls: ``fwi_operator.source_controls`` mapping.
        balance: ``fs-objective-balance-1`` output for ``calibrate``.
        wri: ``fwi_operator.wri`` mapping (``penalty`` required).
        extension: ``fwi_operator.extension`` mapping.
        reflectivity: ``fwi_operator.reflectivity`` mapping.
        reduced_normal: ``fwi_operator.reduced_normal`` mapping (may be empty).
        misfit: ``Imaging.misfit`` mapping or object with ``to_fs``.
        kernel_derivative: Optional total spectral kernel derivative request.
        smoothing: Optional :class:`SmoothingConfig` applied by the ``smooth``
            postprocess to the covector task parts.
        weights: Optional per-frequency weights for that postprocess.
        control_active: Unqualified names for ``control_sensitivities.active``.
        gram_derivative: Optional ``control_sensitivities.gram_derivative``.
        outputs: Optional output requests.
    """

    supports_trace_packing: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: Union[Sequence[Union[float, complex]], np.ndarray],
        *,
        action: str,
        active: Optional[Sequence[str]] = None,
        state: Optional[Union[str, Path]] = None,
        covector: Optional[Union[str, Path]] = None,
        direction: Optional[Union[str, Path]] = None,
        objective_vector: Optional[Union[str, Path]] = None,
        objective: Optional[Union[str, Path]] = None,
        control_state: Optional[Union[str, Path]] = None,
        state_output: Optional[Union[str, Path]] = None,
        manifest: Optional[Union[str, Path]] = None,
        min_support: Optional[float] = None,
        support_measure: bool = False,
        model_gradient: bool = False,
        cache_receiver_state: bool = True,
        source_controls: Optional[Mapping[str, Any]] = None,
        balance: Optional[Union[str, Path]] = None,
        wri: Optional[Mapping[str, Any]] = None,
        extension: Optional[Mapping[str, Any]] = None,
        reflectivity: Optional[Mapping[str, Any]] = None,
        reduced_normal: Optional[Mapping[str, Any]] = None,
        misfit: Any = None,
        kernel_derivative: Optional[Mapping[str, Any]] = None,
        smoothing: Optional[Union[SmoothingConfig, Mapping[str, Any]]] = None,
        weights: Optional[Sequence[float]] = None,
        control_active: Optional[Sequence[str]] = None,
        gram_derivative: Optional[str] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
        preserve_task_outputs: bool = False,
        k_list: Optional[Iterable[float]] = None,
        k_weights: Optional[Iterable[float]] = None,
        k_units: Optional[str] = None,
    ) -> None:
        super().__init__(
            name,
            simulation,
            "fwi_operator",
            _normalized_frequencies(f_list),
            JobOutputs(outputs),
            k_list=None if k_list is None else list(k_list),
            k_weights=None if k_weights is None else list(k_weights),
            k_units=k_units,
            preserve_task_outputs=preserve_task_outputs,
        )
        action = str(action).strip().lower()
        if action not in FWI_ACTIONS:
            raise ValueError(
                f"fwi_operator action must be one of {', '.join(FWI_ACTIONS)}"
            )
        self.action = action
        result_dir = self._result_path
        self.active = None if active is None else _qualified_active(active)
        state_is_output = action == "linearize"
        self.state = (
            _output_path(state, result_dir)
            if state_is_output
            else _input_path(state, simulation)
        )
        self.covector = _output_path(covector, result_dir)
        self.direction = _input_path(direction, simulation)
        self.objective_vector = (
            _output_path(objective_vector, result_dir)
            if action == "jvp"
            else _input_path(objective_vector, simulation)
        )
        self.objective = _output_path(objective, result_dir)
        self.control_state = _input_path(control_state, simulation)
        self.state_output = _output_path(state_output, result_dir)
        self.manifest = _output_path(manifest, result_dir)
        self.min_support = _min_support(min_support)
        self.support_measure = _postprocess_bool(support_measure, "support_measure")
        self.model_gradient = _postprocess_bool(model_gradient, "model_gradient")
        self.cache_receiver_state = _postprocess_bool(
            cache_receiver_state, "cache_receiver_state"
        )
        self.source_controls = (
            None
            if source_controls is None
            else _validate_source_controls(source_controls)
        )
        self.balance = _output_path(balance, result_dir)
        self.wri = None if wri is None else _validate_wri(wri)
        self.extension = (
            None
            if extension is None
            else _validate_extension(extension, simulation, result_dir)
        )
        self.reflectivity = (
            None if reflectivity is None else _validate_reflectivity(reflectivity)
        )
        self.reduced_normal = (
            None if reduced_normal is None else _validate_reduced_normal(reduced_normal)
        )
        self.misfit = misfit
        self.kernel_derivative = _kernel_derivative(
            kernel_derivative, residuals=_KERNEL_RESIDUALS_FWI
        )
        self.smoothing = SmoothingConfig.from_value(smoothing)
        self.weights = _frequency_weights(
            weights, self.n_tasks, label="control-gradient"
        )
        self.control_active = _active_controls(control_active)
        if gram_derivative is not None and gram_derivative not in {"frozen", "total"}:
            raise ValueError("gram_derivative must be 'frozen' or 'total'")
        self.gram_derivative = gram_derivative
        self._validate()

    # -- validation -----------------------------------------------------------

    def _validate(self) -> None:
        action = self.action
        has_extension = self.extension is not None
        if has_extension and self.reflectivity is not None:
            raise ValueError("extension and reflectivity are mutually exclusive")
        if self.reduced_normal is not None and not has_extension:
            raise ValueError("reduced_normal requires a model extension")
        if self.reduced_normal is not None and action != "solve":
            raise ValueError("reduced_normal requires action = solve")
        if self.model_gradient and action != "solve":
            raise ValueError("model_gradient requires action = solve with an extension")
        if self.model_gradient and self.reduced_normal is not None:
            raise ValueError("choose model_gradient or reduced_normal, not both")
        if self.source_controls is not None and action not in _SOURCE_CONTROL_ACTIONS:
            raise ValueError(
                "source_controls support linearize, jvp, vjp and normal only"
            )
        if has_extension and action not in _STATE_ACTIONS:
            raise ValueError(
                "extension supports linearize, jvp, vjp, normal and solve only"
            )
        if (
            self.min_support is not None or self.support_measure
        ) and action not in _STATE_ACTIONS:
            raise ValueError(
                "min_support and support_measure apply to fwi_operator.controls "
                "of linearize, jvp, vjp, normal and solve only"
            )
        if has_extension and self._uses_control_sensitivities():
            raise ValueError(
                "extension jobs cannot carry control_sensitivities "
                "(smoothing, weights, control_active, gram_derivative)"
            )
        if self.reflectivity is not None and action not in _STATE_ACTIONS - {"solve"}:
            raise ValueError(
                "reflectivity supports linearize, jvp, vjp and normal only"
            )

        if action in _STATE_ACTIONS:
            if self.active is None:
                raise ValueError(
                    f"action {action!r} requires active (ordered qualified blocks)"
                )
            if self.state is None:
                raise ValueError(f"action {action!r} requires a state path")
            if self.balance is not None:
                raise ValueError("balance is only written by calibrate")
            if self.wri is not None:
                raise ValueError("wri options are only used by action = wri")
        else:
            for label in (
                "active",
                "state",
                "control_state",
                "state_output",
                "manifest",
            ):
                if getattr(self, label) is not None:
                    raise ValueError(
                        f"{label} belongs to the joint control registry and is not "
                        f"accepted by action {action!r}"
                    )
            if self.objective_vector is not None:
                raise ValueError(
                    f"objective_vector is not accepted by action {action!r}"
                )

        if action == "calibrate":
            if self.balance is None:
                raise ValueError("calibrate requires a balance output path")
            if self.direction is not None or self.covector is not None:
                raise ValueError("calibrate does not take direction or covector")
            if self._uses_control_sensitivities():
                raise ValueError("calibrate does not compute control sensitivities")
        elif action == "wri":
            if self.wri is None:
                raise ValueError("action wri requires wri options with a penalty")
            if self.covector is None:
                raise ValueError("wri requires a covector (model_covector) output")
            if "curvature" in self.wri:
                if self.direction is None:
                    raise ValueError(
                        "wri curvature requires a direction (model_direction)"
                    )
            elif self.direction is not None:
                raise ValueError("wri direction requires wri.curvature")
            if self.balance is not None:
                raise ValueError("balance is only written by calibrate")
        elif action == "linearize":
            if self.direction is not None:
                raise ValueError("linearize does not take a direction")
            if self.objective_vector is not None:
                raise ValueError("linearize does not take an objective_vector")
            if has_extension and self.covector is not None:
                raise ValueError(
                    "extension linearize returns extension.covector; a physical "
                    "covector requires solve with model_gradient or reduced_normal"
                )
        elif action == "jvp":
            if self.objective_vector is None:
                raise ValueError("jvp requires an objective_vector output")
            if self.covector is not None:
                raise ValueError("jvp does not write a covector")
            if has_extension:
                if self.extension.get("direction") is None:  # type: ignore[union-attr]
                    raise ValueError("extension jvp requires extension.direction")
                if self.direction is not None:
                    raise ValueError(
                        "extension jvp takes extension.direction, not direction"
                    )
            else:
                if self.direction is None:
                    raise ValueError("jvp requires a direction input")
                if not self.active:
                    raise ValueError("jvp requires a nonempty active subspace")
        elif action == "vjp":
            if self.objective_vector is None:
                raise ValueError("vjp requires an objective_vector input")
            if self.direction is not None:
                raise ValueError("vjp does not take a direction")
            if has_extension:
                if self.extension.get("covector") is None:  # type: ignore[union-attr]
                    raise ValueError("extension vjp requires extension.covector")
                if self.covector is not None:
                    raise ValueError(
                        "extension vjp writes extension.covector, not covector"
                    )
            elif self.covector is None:
                raise ValueError("vjp requires a covector output")
        elif action == "normal":
            if self.objective_vector is not None:
                raise ValueError("normal does not take an objective_vector")
            if has_extension:
                missing = [
                    key
                    for key in ("direction", "covector")
                    if self.extension.get(key) is None  # type: ignore[union-attr]
                ]
                if missing:
                    raise ValueError(
                        f"extension normal requires extension.{' and extension.'.join(missing)}"
                    )
                if self.direction is not None or self.covector is not None:
                    raise ValueError(
                        "extension normal uses extension.direction/covector"
                    )
            else:
                if self.direction is None or self.covector is None:
                    raise ValueError("normal requires direction and covector")
                if not self.active:
                    raise ValueError("normal requires a nonempty active subspace")
        elif action == "solve":
            if not has_extension:
                raise ValueError("solve requires an extension")
            if self.extension.get("solver") is None:  # type: ignore[union-attr]
                raise ValueError("solve requires extension.solver")
            if self.extension.get("covector") is not None:  # type: ignore[union-attr]
                raise ValueError(
                    "solve writes extension.solver.solution, not extension.covector"
                )
            reduced = self.model_gradient or self.reduced_normal is not None
            if reduced:
                if self.covector is None:
                    raise ValueError(
                        "solve with model_gradient or reduced_normal requires a covector output"
                    )
                if self.objective_vector is not None:
                    raise ValueError(
                        "reduced background actions require the observed-data target"
                    )
                if not self.active:
                    raise ValueError(
                        "reduced background actions require active material blocks"
                    )
            elif self.covector is not None:
                raise ValueError(
                    "physical extension covectors require model_gradient or reduced_normal"
                )
            if self.reduced_normal is not None and self.direction is None:
                raise ValueError("reduced_normal requires a physical control direction")
            if self.reduced_normal is None and self.direction is not None:
                raise ValueError("solve takes a direction only with reduced_normal")

        if has_extension and action != "solve" and self.extension.get("solver") is not None:  # type: ignore[union-attr]
            raise ValueError("extension.solver requires action = solve")

        if self.smoothing is not None:
            if self.covector is None:
                raise ValueError("smoothing requires a covector output to postprocess")
            if self.control_active is None:
                model_blocks = [
                    name for name in self.active or () if name.startswith("model.")
                ]
                if len(model_blocks) != len(self.active or ()):
                    raise ValueError(
                        "smoothing applies to model.* blocks only; pass control_active "
                        "to select the smoothed material controls"
                    )
                self.control_active = [
                    unqualified_block_name(name) for name in model_blocks
                ] or None
        if (
            self.weights is not None
            and self.smoothing is None
            and self.covector is None
        ):
            raise ValueError("weights require a smoothed covector postprocess")

    def _uses_control_sensitivities(self) -> bool:
        return any(
            value is not None
            for value in (
                self.smoothing,
                self.weights,
                self.control_active,
                self.gram_derivative,
            )
        )

    # -- output paths ---------------------------------------------------------

    def state_file(self, task: Optional[int] = None) -> Path:
        """Return the objective-state manifest stem or one task's manifest."""

        if self.state is None:
            raise ValueError("this job has no objective state path")
        return _task_path(self.state, task)

    def covector_file(self, task: Optional[int] = None, *, raw: bool = False) -> Path:
        """Return the covector stem, one task part, or the raw smoothing aggregate."""

        if self.covector is None:
            raise ValueError("this job has no covector output")
        if raw:
            if task is not None:
                raise ValueError("raw aggregate covectors do not have task parts")
            return _raw_path(self.covector)
        return _task_path(self.covector, task)

    def objective_vector_file(self, task: Optional[int] = None) -> Path:
        """Return the objective-vector manifest stem or one task's manifest."""

        if self.objective_vector is None:
            raise ValueError("this job has no objective vector path")
        return _task_path(self.objective_vector, task)

    def report_file(self, task: Optional[int] = None) -> Path:
        """Return the ``fs-objective-report-1`` stem or one task's report.

        State actions (``linearize``, ``jvp``, ``vjp``, ``normal``, ``solve``)
        receive their report beside the objective state manifest as
        ``<state stem>_<task>_report.json``; Sauce does not read
        ``fwi_operator.objective`` for them.  ``calibrate`` and ``wri`` use the
        ``objective`` path.
        """

        if self.action in _STATE_ACTIONS and self.state is not None:
            stem = _task_path(self.state, task)
            return stem.with_name(f"{stem.stem}_report.json")
        if self.objective is None:
            raise ValueError("this job has no objective report path")
        return _task_path(self.objective, task)

    def balance_file(self, task: Optional[int] = None) -> Path:
        """Return the balance artifact stem or one task's artifact."""

        if self.balance is None:
            raise ValueError("this job has no balance output")
        return _task_path(self.balance, task)

    def _multitask_output(self, path: Path, task: Optional[int]) -> Path:
        """Return a ``controls`` export of ``task``.

        Sauce keeps the exact configured path in a single-task job and adds
        the ``_<task>`` suffix when ``f_list`` has more than one entry.
        """

        if task is None or self.n_tasks <= 1:
            return path
        return _task_path(path, task)

    def state_output_file(self, task: Optional[int] = None) -> Path:
        """Return the ``fs-control-state-1`` export (configured path or one task's)."""

        if self.state_output is None:
            raise ValueError("this job has no state_output path")
        return self._multitask_output(self.state_output, task)

    def manifest_file(self, task: Optional[int] = None) -> Path:
        """Return the ``fs-control-registry-1`` export (configured path or one task's)."""

        if self.manifest is None:
            raise ValueError("this job has no manifest path")
        return self._multitask_output(self.manifest, task)

    def task_input(self, path: Union[str, Path], task: int) -> Path:
        """Return the file Sauce reads for per-task operator input ``path``.

        With more than one frequency task, the inputs ``state`` (jvp, vjp,
        normal, solve), ``direction``, ``objective_vector`` (vjp),
        ``extension.direction`` and ``model_direction`` resolve to the
        task-suffixed sibling ``<stem>_<task><ext>`` when it exists, and to
        the exact path otherwise; a single-task job uses the exact path.
        """

        path = Path(path)
        if self.n_tasks > 1:
            candidate = _task_path(path, task)
            if candidate.is_file():
                return candidate
        return path

    def _extension_path(self, *keys: str) -> Path:
        value: Any = self.extension
        for key in keys:
            value = None if value is None else value.get(key)
        if value is None:
            raise ValueError(f"this job has no extension.{'.'.join(keys)} path")
        return Path(value)

    def extension_solution_file(self, task: Optional[int] = None) -> Path:
        """Return the solved-taps stem or one task's ``fs-extension-vector-1``."""

        return _task_path(self._extension_path("solver", "solution"), task)

    def extension_report_file(self, task: Optional[int] = None) -> Path:
        """Return the inner-solve report stem or one task's ``fs-extension-solve-1``."""

        return _task_path(self._extension_path("solver", "report"), task)

    def extension_covector_file(self, task: Optional[int] = None) -> Path:
        """Return the extension covector stem or one task part."""

        return _task_path(self._extension_path("covector"), task)

    def extension_manifest_file(self) -> Path:
        """Return the exact ``fs-model-extension-1`` descriptor path."""

        return self._extension_path("manifest")

    # -- postprocess ----------------------------------------------------------

    def requires_postprocess(self) -> bool:
        """Return whether the covector parts are smoothed after the tasks."""

        return self.smoothing is not None

    def postprocess_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate or one task-local covector path."""

        return self.covector_file(part)

    def postprocess_fetch_files(self) -> List[Path]:
        """Return both the exact aggregate covector and the smoothed result."""

        return [self.covector_file(raw=True), self.covector_file()]

    def postprocess_output_exists(self) -> bool:
        """Return whether the smoothed covector exists locally."""

        return self.covector_file().is_file()

    def postprocess_part_outputs_exist(self) -> bool:
        """Return whether every task covector part exists locally."""

        return all(
            self.covector_file(part).is_file() for part in range(1, self.n_tasks + 1)
        )

    def is_run_current(self) -> bool:
        """Return whether the run and, when smoothed, the aggregate are current."""

        current = super().is_run_current()
        if current and self.requires_postprocess():
            return self.postprocess_output_exists()
        return current

    # -- serialization --------------------------------------------------------

    def _imaging_payload(
        self, ctx: Optional[ExportContext], *, project_relative: bool
    ) -> Dict[str, Any]:
        imaging: Dict[str, Any] = {}
        misfit = _misfit_to_fs(self.misfit, ctx, project_relative=project_relative)
        if misfit is not None:
            imaging["misfit"] = misfit
        return imaging

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize the ``fwi_operator`` request."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        imaging = self._imaging_payload(ctx, project_relative=project_relative)
        payload["Imaging"] = imaging
        if self.kernel_derivative is not None:
            payload["kernel_derivative"] = dict(self.kernel_derivative)
            payload["Image"] = copy.deepcopy(imaging)

        op: Dict[str, Any] = {"action": self.action}
        if self.action in _STATE_ACTIONS:
            op["state"] = _job_path(self.state, ctx, False)
            controls: Dict[str, Any] = {"active": list(self.active or [])}
            if self.control_state is not None:
                controls["state"] = _job_path(self.control_state, ctx, False)
            if self.state_output is not None:
                controls["state_output"] = _job_path(self.state_output, ctx, False)
            if self.manifest is not None:
                controls["manifest"] = _job_path(self.manifest, ctx, False)
            if self.min_support is not None:
                controls["min_support"] = self.min_support
            if self.support_measure:
                controls["support_measure"] = True
            op["controls"] = controls
        if self.action == "wri":
            op["model_covector"] = _job_path(self.covector, ctx, False)
            if self.direction is not None:
                op["model_direction"] = _job_path(self.direction, ctx, False)
            op["wri"] = copy.deepcopy(self.wri)
        else:
            if self.direction is not None:
                op["direction"] = _job_path(self.direction, ctx, False)
            if self.covector is not None:
                op["covector"] = _job_path(self.covector, ctx, False)
            if self.objective_vector is not None:
                op["objective_vector"] = _job_path(self.objective_vector, ctx, False)
        if self.objective is not None:
            op["objective"] = _job_path(self.objective, ctx, False)
        if self.balance is not None:
            op["balance"] = _job_path(self.balance, ctx, False)
        if self.model_gradient:
            op["model_gradient"] = True
        if not self.cache_receiver_state:
            op["cache_receiver_state"] = False
        if self.source_controls is not None:
            op["source_controls"] = dict(self.source_controls)
        if self.extension is not None:
            op["extension"] = self._extension_to_fs(ctx)
        if self.reflectivity is not None:
            op["reflectivity"] = copy.deepcopy(self.reflectivity)
        if self.reduced_normal is not None:
            op["reduced_normal"] = dict(self.reduced_normal)
        payload["fwi_operator"] = op

        if self._uses_control_sensitivities() or self.action == "wri":
            # WRI always carries control_sensitivities: Sauce reads its material
            # subspace there and rejects curvature requests without the block.
            sensitivities: Dict[str, Any] = {}
            if self.control_active is not None:
                sensitivities["active"] = list(self.control_active)
            if self.gram_derivative is not None:
                sensitivities["gram_derivative"] = self.gram_derivative
            if self.smoothing is not None:
                # The --smooth postprocess aggregates <gradient>_<task>.h5
                # parts; bind it to the covector task parts.
                sensitivities["gradient"] = _job_path(self.covector, ctx, False)
                sensitivities["Smoothing"] = self.smoothing.to_control_fs()
            if self.weights is not None:
                sensitivities["weights"] = list(self.weights)
            payload["control_sensitivities"] = sensitivities
        return payload

    def _extension_to_fs(self, ctx: Optional[ExportContext]) -> Dict[str, Any]:
        extension = copy.deepcopy(self.extension or {})
        for field in extension["fields"]:
            offsets = field.get("offsets")
            if offsets and offsets.get("artifact") is not None:
                offsets["artifact"] = _job_path(offsets["artifact"], ctx, False)
        for key in ("direction", "covector", "manifest"):
            if extension.get(key) is not None:
                extension[key] = _job_path(extension[key], ctx, False)
        solver = extension.get("solver")
        if solver:
            solver["solution"] = _job_path(solver["solution"], ctx, False)
            solver["report"] = _job_path(solver["report"], ctx, False)
        return extension

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash the direction, objective dual, baseline and extension inputs."""

        inputs: Dict[str, Any] = {}
        if self.direction is not None:
            inputs["direction"] = self._resolved_input_fingerprint(self.direction)
        if self.objective_vector is not None and self.action != "jvp":
            inputs["objective_vector"] = self._task_input_fingerprints(
                self.objective_vector
            )
        if self.control_state is not None:
            inputs["control_state"] = self._path_content_fingerprint(self.control_state)
        if self.extension is not None and self.extension.get("direction") is not None:
            inputs["extension_direction"] = self._resolved_input_fingerprint(
                Path(self.extension["direction"])
            )
        return inputs

    def _resolved_input_fingerprint(self, path: Path) -> Any:
        """Hash a per-task operator input as each task resolves it.

        One file when every task reads the exact path, otherwise one hash per
        task of the file :meth:`task_input` resolves.
        """

        resolved = [self.task_input(path, task) for task in range(1, self.n_tasks + 1)]
        if all(item == Path(path) for item in resolved):
            return self._path_content_fingerprint(path)
        return {
            str(task): self._path_content_fingerprint(item)
            for task, item in enumerate(resolved, start=1)
        }

    def _task_input_fingerprints(self, stem: Path) -> Dict[str, Any]:
        """Hash per-task manifests, falling back to one shared file."""

        if stem.exists() and not any(
            _task_path(stem, task).exists() for task in range(1, self.n_tasks + 1)
        ):
            return {"shared": self._path_content_fingerprint(stem)}
        return {
            str(task): self._path_content_fingerprint(_task_path(stem, task))
            for task in range(1, self.n_tasks + 1)
        }

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "FWIOperatorJob":
        """Deserialize an ``fwi_operator`` job."""

        simulation, resolve = cls._load_context(data, base_path, project_path)
        op = data["fwi_operator"]
        action = str(op["action"])
        controls = op.get("controls") or {}
        imaging = data.get("Imaging") or data.get("Image") or {}
        sensitivities = data.get("control_sensitivities") or {}
        extension = copy.deepcopy(op.get("extension"))
        if extension is not None:
            for key in ("direction", "covector", "manifest"):
                if extension.get(key) is not None:
                    extension[key] = resolve(extension[key])
            solver = extension.get("solver")
            if solver:
                solver["solution"] = resolve(solver["solution"])
                solver["report"] = resolve(solver["report"])
            for field in extension.get("fields", ()):
                offsets = field.get("offsets")
                if offsets and offsets.get("artifact") is not None:
                    offsets["artifact"] = resolve(offsets["artifact"])
        job = cls(
            data["name"],
            simulation,
            cls._decode_frequencies(data["f_list"]),
            action=action,
            active=controls.get("active") if "controls" in op else None,
            state=resolve(op.get("state")),
            covector=resolve(
                op.get("model_covector") if action == "wri" else op.get("covector")
            ),
            direction=resolve(
                op.get("model_direction") if action == "wri" else op.get("direction")
            ),
            objective_vector=resolve(op.get("objective_vector")),
            objective=resolve(op.get("objective")),
            control_state=resolve(controls.get("state")),
            state_output=resolve(controls.get("state_output")),
            manifest=resolve(controls.get("manifest")),
            min_support=controls.get("min_support"),
            support_measure=bool(controls.get("support_measure", False)),
            model_gradient=bool(op.get("model_gradient", False)),
            cache_receiver_state=bool(op.get("cache_receiver_state", True)),
            source_controls=op.get("source_controls"),
            balance=resolve(op.get("balance")),
            wri=op.get("wri"),
            extension=extension,
            reflectivity=op.get("reflectivity"),
            reduced_normal=op.get("reduced_normal"),
            misfit=imaging.get("misfit"),
            kernel_derivative=data.get("kernel_derivative"),
            smoothing=sensitivities.get("Smoothing"),
            weights=sensitivities.get("weights"),
            control_active=sensitivities.get("active"),
            gram_derivative=sensitivities.get("gram_derivative"),
            outputs=JobOutputs.from_fs(data.get("Outputs")),
            k_list=data.get("k_list"),
            k_weights=data.get("k_weights"),
            k_units=data.get("k_units"),
        )
        cls._finish_load(job, data)
        return job


# ---------------------------------------------------------------------------
# ControlGradientJob
# ---------------------------------------------------------------------------


def _validate_focus(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("focus must be a mapping with softening and optional kind")
    allowed = {"softening", "kind"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported focus option(s): {', '.join(unknown)}")
    if "softening" not in value:
        raise ValueError("focus requires a positive softening length (km)")
    softening = float(value["softening"])
    if not np.isfinite(softening) or softening <= 0.0:
        raise ValueError("focus softening must be finite and positive")
    kind = str(value.get("kind", "trfwi")).strip().lower()
    if kind not in _FOCUS_KINDS:
        raise ValueError(f"focus kind must be one of {', '.join(_FOCUS_KINDS)}")
    return {"softening": softening, "kind": kind}


@register_class
class ControlGradientJob(_ImagingJobBase):
    """Native control sensitivities: ``rtm`` VJP, ``born`` JVP or ``focus``.

    Args:
        name: Job name.
        simulation: Simulation owning the parameterized material controls.
        f_list: One frequency per task.
        kind: ``"rtm"``, ``"born"`` or ``"focus"``.
        observed: Observed data path (all receiver groups) or
            ``group -> path`` mapping. Required for ``rtm`` and ``focus``.
        gradient: Aggregate control-gradient output (``rtm``/``focus``).
        direction: Native control direction input (``born``).
        objective_file: Scalar objective output (``control_sensitivities.objective``
            for ``rtm``, ``focus.objective`` for ``focus``).
        gram_derivative: ``"frozen"`` or ``"total"`` DPG Gram dependence.
        current: Optional control-coefficient override file.
        active: Optional ordered unqualified control subspace.
        source_taper: Optional radial source-proximity taper.
        spatial_window: Optional axis-aligned sensitivity window.
        misfit: ``Imaging.misfit`` mapping or object with ``to_fs``; receiver
            groups are completed from ``observed``.
        weights: Optional per-frequency postprocess weights.
        smoothing: Optional :class:`SmoothingConfig` Riesz map.
        raw_gradient: Optional exact aggregate output before smoothing.
        focus: ``{"softening": km, "kind": "trfwi"|"weft"}`` for ``focus``.
        kernel_derivative: Optional spectral kernel derivative (``rtm`` only).
        outputs: Optional output requests.
    """

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: Union[Sequence[Union[float, complex]], np.ndarray],
        *,
        kind: str,
        observed: Optional[Union[str, Path, Mapping[str, Union[str, Path]]]] = None,
        gradient: Optional[Union[str, Path]] = None,
        direction: Optional[Union[str, Path]] = None,
        objective_file: Optional[Union[str, Path]] = None,
        gram_derivative: str = "frozen",
        current: Optional[Union[str, Path]] = None,
        active: Optional[Sequence[str]] = None,
        source_taper: Optional[Mapping[str, Any]] = None,
        spatial_window: Optional[Mapping[str, Any]] = None,
        misfit: Any = None,
        weights: Optional[Sequence[float]] = None,
        smoothing: Optional[Union[SmoothingConfig, Mapping[str, Any]]] = None,
        raw_gradient: Optional[Union[str, Path]] = None,
        focus: Optional[Mapping[str, Any]] = None,
        kernel_derivative: Optional[Mapping[str, Any]] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
        preserve_task_outputs: bool = False,
        k_list: Optional[Iterable[float]] = None,
        k_weights: Optional[Iterable[float]] = None,
        k_units: Optional[str] = None,
    ) -> None:
        kind = str(kind).strip().lower()
        if kind not in _CONTROL_KINDS:
            raise ValueError(f"kind must be one of {', '.join(_CONTROL_KINDS)}")
        super().__init__(
            name,
            simulation,
            kind,
            _normalized_frequencies(f_list),
            JobOutputs(outputs),
            k_list=None if k_list is None else list(k_list),
            k_weights=None if k_weights is None else list(k_weights),
            k_units=k_units,
            preserve_task_outputs=preserve_task_outputs,
        )
        self.kind = kind
        result_dir = self._result_path
        self.observed = _observed_groups(observed, simulation)
        self.gradient = _output_path(gradient, result_dir)
        self.direction = _input_path(direction, simulation)
        self._objective_file = _output_path(objective_file, result_dir)
        if gram_derivative not in {"frozen", "total"}:
            raise ValueError("gram_derivative must be 'frozen' or 'total'")
        self.gram_derivative = gram_derivative
        self.current = _input_path(current, simulation)
        self.active = _active_controls(active)
        self.source_taper = _source_taper(source_taper)
        self.spatial_window = _spatial_window(spatial_window)
        self.misfit = misfit
        self.weights = _frequency_weights(
            weights, self.n_tasks, label="control-gradient"
        )
        self.smoothing = SmoothingConfig.from_value(smoothing)
        self.raw_gradient = _output_path(raw_gradient, result_dir)
        self.focus = None if focus is None else _validate_focus(focus)
        self.kernel_derivative = _kernel_derivative(
            kernel_derivative, residuals=_KERNEL_RESIDUALS_CONTROL
        )
        self._validate()

    def _validate(self) -> None:
        kind = self.kind
        if kind == "born":
            if self.direction is None:
                raise ValueError("born control sensitivities require a direction")
            forbidden = {
                "observed": self.observed,
                "gradient": self.gradient,
                "objective_file": self._objective_file,
                "weights": self.weights,
                "smoothing": self.smoothing,
                "raw_gradient": self.raw_gradient,
                "focus": self.focus,
                "kernel_derivative": self.kernel_derivative,
            }
            present = [key for key, value in forbidden.items() if value is not None]
            if present:
                raise ValueError(
                    f"born control sensitivities do not take {', '.join(present)}"
                )
        else:
            if self.direction is not None:
                raise ValueError(
                    f"{kind} control sensitivities do not take a direction"
                )
            if not self.observed:
                raise ValueError(f"{kind} control sensitivities require observed data")
            if self.gradient is None:
                raise ValueError(
                    f"{kind} control sensitivities require a gradient output"
                )
        if kind == "focus":
            if self.focus is None:
                raise ValueError(
                    "focus jobs require focus={'softening': km, 'kind': ...}"
                )
            if self._objective_file is None:
                raise ValueError(
                    "focus jobs require objective_file for focus.objective"
                )
            if self.source_taper is not None:
                raise ValueError("time-reversal focus does not support source_taper")
            if self.gram_derivative != "frozen":
                raise ValueError(
                    "total Gram derivatives do not support time-reversal focus"
                )
            if self.kernel_derivative is not None:
                raise ValueError("kernel_derivative requires workflow = rtm")
        elif self.focus is not None:
            raise ValueError("focus options require kind = 'focus'")
        if self.gram_derivative == "total":
            if self.source_taper is not None or self.spatial_window is not None:
                raise ValueError(
                    "total Gram derivatives do not support source_taper or spatial_window"
                )
            if "phase_derivative" in _misfit_comparison_kinds(
                _misfit_to_fs(self.misfit, None, project_relative=False)
            ):
                raise ValueError(
                    "total Gram derivatives do not support phase_derivative comparisons"
                )

    # -- output paths ---------------------------------------------------------

    def gradient_file(self, task: Optional[int] = None, *, raw: bool = False) -> Path:
        """Return the final, raw aggregate, or task-local gradient path."""

        if self.gradient is None:
            raise ValueError("born control sensitivities do not write a gradient")
        if raw:
            if task is not None:
                raise ValueError("raw aggregate gradients do not have task parts")
            return (
                self.raw_gradient
                if self.raw_gradient is not None
                else _raw_path(self.gradient)
            )
        return _task_path(self.gradient, task)

    def objective_file(self, task: Optional[int] = None) -> Optional[Path]:
        """Return the aggregate or one task-local scalar-objective path."""

        if self._objective_file is None:
            return None
        return _task_path(self._objective_file, task)

    @property
    def objective_value(self) -> float:
        """Read the aggregated scalar objective written by Sauce."""

        import h5py

        path = self.objective_file()
        if path is None:
            raise ValueError("this job does not request a scalar objective")
        with h5py.File(path, "r") as h5:
            return float(h5["value"][()])

    @property
    def trace_outputs(self) -> TraceOutputSpec:
        """Describe receiver traces; Born jobs write incremental groups only."""

        baseline = super().trace_outputs
        if self.kind == "born":
            return _incremental_trace_outputs(baseline, keep_baseline=False)
        return baseline

    # -- postprocess ----------------------------------------------------------

    def requires_postprocess(self) -> bool:
        """Return whether gradients are aggregated after the frequency tasks."""

        return self.kind != "born"

    def postprocess_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate or task-local control-gradient path."""

        return self.gradient_file(part)

    def postprocess_fetch_files(self) -> List[Path]:
        """Return the raw aggregate, final gradient and any objective."""

        files = [self.gradient_file(raw=True), self.gradient_file()]
        objective = self.objective_file()
        if objective is not None:
            files.append(objective)
        return files

    def postprocess_output_exists(self) -> bool:
        """Return whether the final gradient (and objective) exist locally."""

        objective = self.objective_file()
        return self.gradient_file().is_file() and (
            objective is None or objective.is_file()
        )

    def postprocess_part_outputs_exist(self) -> bool:
        """Return whether every gradient (and objective) shard exists locally."""

        for part in range(1, self.n_tasks + 1):
            if not self.gradient_file(part).is_file():
                return False
            objective = self.objective_file(part)
            if objective is not None and not objective.is_file():
                return False
        return True

    def is_run_current(self) -> bool:
        """Return whether the run and the aggregated products are current."""

        current = super().is_run_current()
        if current and self.requires_postprocess():
            return self.postprocess_output_exists()
        return current

    # -- serialization --------------------------------------------------------

    def _misfit_payload(
        self, ctx: Optional[ExportContext], *, project_relative: bool
    ) -> Dict[str, Any]:
        misfit = _misfit_to_fs(self.misfit, ctx, project_relative=project_relative)
        if misfit is None:
            misfit = copy.deepcopy(_DEFAULT_MISFIT)
        if "objective_terms" not in misfit and "objective" not in misfit:
            misfit["objective"] = {"kind": "l2"}
        if "preprocess" not in misfit:
            # A native control VJP is the linear transpose used by matrix-free
            # products; it must not inherit image defaults such as
            # illumination normalization.
            misfit["preprocess"] = {"include_defaults": False, "hooks": []}
        groups = list(misfit.get("receiver_groups") or [])
        by_name = {str(group.get("name")): dict(group) for group in groups}
        ordered = []
        for name, path in (self.observed or {}).items():
            group = by_name.pop(name, {"name": name})
            if group.get("observed") is None:
                group["observed"] = _job_path(path, ctx, project_relative)
            ordered.append(group)
        ordered.extend(by_name.values())
        misfit["receiver_groups"] = ordered
        return misfit

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize the control-sensitivity request."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        sensitivities: Dict[str, Any] = {}
        if self.kind == "born":
            sensitivities["direction"] = _job_path(self.direction, ctx, False)
        else:
            sensitivities["gradient"] = _job_path(self.gradient, ctx, False)
        if self.gram_derivative != "frozen":
            sensitivities["gram_derivative"] = self.gram_derivative
        if self.kind == "rtm" and self._objective_file is not None:
            sensitivities["objective"] = _job_path(self._objective_file, ctx, False)
        if self.active is not None:
            sensitivities["active"] = list(self.active)
        if self.source_taper is not None:
            sensitivities["source_taper"] = dict(self.source_taper)
        if self.spatial_window is not None:
            sensitivities["spatial_window"] = dict(self.spatial_window)
        if self.current is not None:
            sensitivities["current"] = _job_path(self.current, ctx, False)
        if self.weights is not None:
            sensitivities["weights"] = list(self.weights)
        if self.smoothing is not None:
            sensitivities["Smoothing"] = self.smoothing.to_control_fs()
        if self.raw_gradient is not None:
            sensitivities["raw_gradient"] = _job_path(self.raw_gradient, ctx, False)
        payload["control_sensitivities"] = sensitivities
        if self.kind != "born":
            imaging = {
                "schema": "fs-imaging-1",
                "name": self.name,
                "misfit": self._misfit_payload(ctx, project_relative=project_relative),
            }
            payload["Imaging"] = imaging
            if self.kernel_derivative is not None:
                payload["kernel_derivative"] = dict(self.kernel_derivative)
                payload["Image"] = copy.deepcopy(imaging)
        if self.kind == "focus":
            payload["focus"] = {
                "objective": _job_path(self._objective_file, ctx, False),
                "softening": self.focus["softening"],  # type: ignore[index]
                "kind": self.focus["kind"],  # type: ignore[index]
            }
        return payload

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash the observed data, direction and current-control inputs."""

        inputs: Dict[str, Any] = {}
        if self.observed:
            inputs["observed"] = {
                name: self._path_content_fingerprint(path)
                for name, path in sorted(self.observed.items())
                if path is not None
            }
        if self.direction is not None:
            inputs["direction"] = self._path_content_fingerprint(self.direction)
        if self.current is not None:
            inputs["current"] = self._path_content_fingerprint(self.current)
        return inputs

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "ControlGradientJob":
        """Deserialize a control-sensitivity job."""

        simulation, resolve = cls._load_context(data, base_path, project_path)
        kind = str(data["workflow"]).lower()
        sensitivities = data.get("control_sensitivities") or {}
        imaging = data.get("Imaging") or data.get("Image") or {}
        misfit = copy.deepcopy(imaging.get("misfit"))
        observed = None
        if misfit and misfit.get("receiver_groups"):
            observed = {
                str(group["name"]): resolve(group.get("observed"))
                for group in misfit["receiver_groups"]
                if group.get("observed") is not None
            }
            for group in misfit["receiver_groups"]:
                group.pop("observed", None)
        focus = data.get("focus")
        objective_file = (
            focus.get("objective")
            if kind == "focus" and focus
            else sensitivities.get("objective")
        )
        job = cls(
            data["name"],
            simulation,
            cls._decode_frequencies(data["f_list"]),
            kind=kind,
            observed=observed or None,
            gradient=resolve(sensitivities.get("gradient")),
            direction=resolve(sensitivities.get("direction")),
            objective_file=resolve(objective_file),
            gram_derivative=sensitivities.get("gram_derivative", "frozen"),
            current=resolve(sensitivities.get("current")),
            active=sensitivities.get("active"),
            source_taper=sensitivities.get("source_taper"),
            spatial_window=sensitivities.get("spatial_window"),
            misfit=misfit,
            weights=sensitivities.get("weights"),
            smoothing=sensitivities.get("Smoothing"),
            raw_gradient=resolve(sensitivities.get("raw_gradient")),
            focus=(
                {"softening": focus["softening"], "kind": focus.get("kind", "trfwi")}
                if kind == "focus" and focus
                else None
            ),
            kernel_derivative=data.get("kernel_derivative"),
            outputs=JobOutputs.from_fs(data.get("Outputs")),
            k_list=data.get("k_list"),
            k_weights=data.get("k_weights"),
            k_units=data.get("k_units"),
        )
        cls._finish_load(job, data)
        return job


# ---------------------------------------------------------------------------
# ImageKernelJob
# ---------------------------------------------------------------------------


class ImageSpec:
    """One Cartesian image request: imaging condition plus optional property."""

    __slots__ = ("condition", "property", "sources")

    def __init__(
        self,
        condition: str,
        property: Optional[str] = None,
        sources: Optional[Sequence[int]] = None,
    ) -> None:
        self.condition = str(condition).strip()
        if not self.condition:
            raise ValueError("image condition must be non-empty")
        self.property = None if property is None else str(property).strip() or None
        if sources is not None:
            ids = [int(s) for s in sources]
            if not ids or any(s < 1 for s in ids) or len(set(ids)) != len(ids):
                raise ValueError("image sources must be unique one-based ids")
            self.sources: Optional[List[int]] = ids
        else:
            self.sources = None

    @classmethod
    def from_value(cls, value: Any) -> "ImageSpec":
        """Normalize a string, tuple, mapping or spec-like object."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, (tuple, list)):
            if not 1 <= len(value) <= 3:
                raise ValueError("image tuples are (IC[, property[, sources]])")
            return cls(*value)
        if isinstance(value, Mapping):
            condition = value.get("IC", value.get("condition"))
            if condition is None:
                raise ValueError("image mappings require an IC")
            return cls(condition, value.get("property"), value.get("sources"))
        to_fs = getattr(value, "to_fs", None)
        if callable(to_fs):
            return cls.from_value(to_fs())
        raise TypeError("images must be strings, tuples, mappings or ImageSpec objects")

    def to_fs(self, name: str) -> Dict[str, Any]:
        """Serialize this request as one ``Imaging.images`` entry."""

        return {
            "name": str(name),
            "IC": self.condition,
            **({"property": self.property} if self.property is not None else {}),
            **({"sources": list(self.sources)} if self.sources is not None else {}),
        }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ImageSpec) and (
            (self.condition, self.property, self.sources)
            == (other.condition, other.property, other.sources)
        )

    def __repr__(self) -> str:
        return f"ImageSpec({self.condition!r}, property={self.property!r}, sources={self.sources!r})"


@register_class
class ImageKernelJob(_ImagingJobBase):
    """Cartesian image kernels on ``Imaging.grid``.

    Args:
        name: Job name.
        simulation: Seismic simulation used to model the data.
        f_list: One frequency per task.
        grid: Cartesian image grid (or its serialized mapping).
        images: ``image name -> ImageSpec`` (or string IC, ``(IC, property)``
            tuple, or mapping with ``IC``/``property``/``sources``).
        observed: ``None`` for zero-data sensitivity kernels, one path for
            every receiver group, or a ``group -> path`` mapping.
        misfit: ``Imaging.misfit`` mapping or object with ``to_fs``.
        weights: Optional per-frequency weights.
        wavelet: Optional wavelet whose spectrum supplies ``weights``.
        smoothing: Optional :class:`SmoothingConfig` or ``Imaging.Smoothing``
            mapping applied while stacking shards.
        keep_forward, keep_adjoint, keep_unstacked: Legacy retention flags.
        field_retention: Explicit ``none``/``forward``/``adjoint``/``all``.
        direction: Cartesian direction file for incremental workflows.
        zero_direction: Use an explicit zero direction instead.
        gauss_newton: Drive the Born adjoint from the linearized residual.
        born_traces_only: Stop a Born workflow after incremental traces.
        workflow: ``"rtm"``, ``"born"`` or ``"lsrtm_gradient"``.
        save_path: Image output directory (default ``<results>/imaging``).
        kernel_derivative: Optional spectral kernel derivative (``rtm`` only).
        outputs: Optional output requests.
        **extra: Additional ``Imaging`` fields passed through verbatim.
    """

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: Union[Sequence[Union[float, complex]], np.ndarray],
        *,
        grid: Union[CartesianGrid, Mapping[str, Any]],
        images: Mapping[str, Any],
        observed: Optional[Union[str, Path, Mapping[str, Union[str, Path]]]] = None,
        misfit: Any = None,
        weights: Optional[Sequence[float]] = None,
        wavelet: Any = None,
        smoothing: Optional[Union[SmoothingConfig, Mapping[str, Any]]] = None,
        keep_forward: bool = False,
        keep_adjoint: bool = False,
        keep_unstacked: bool = False,
        field_retention: Optional[str] = None,
        direction: Optional[Union[str, Path]] = None,
        zero_direction: bool = False,
        gauss_newton: bool = False,
        born_traces_only: bool = False,
        workflow: str = "rtm",
        save_path: Optional[Union[str, Path]] = None,
        kernel_derivative: Optional[Mapping[str, Any]] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
        preserve_task_outputs: bool = False,
        k_list: Optional[Iterable[float]] = None,
        k_weights: Optional[Iterable[float]] = None,
        k_units: Optional[str] = None,
        **extra: Any,
    ) -> None:
        workflow = str(workflow).strip().lower()
        if workflow not in _IMAGE_WORKFLOWS:
            raise ValueError(f"workflow must be one of {', '.join(_IMAGE_WORKFLOWS)}")
        super().__init__(
            name,
            simulation,
            workflow,
            _normalized_frequencies(f_list),
            JobOutputs(outputs),
            k_list=None if k_list is None else list(k_list),
            k_weights=None if k_weights is None else list(k_weights),
            k_units=k_units,
            preserve_task_outputs=preserve_task_outputs,
        )
        if isinstance(grid, Mapping):
            grid = CartesianGrid.from_fs(grid)
        if not isinstance(grid, CartesianGrid):
            raise TypeError("image kernels require a CartesianGrid")
        self.grid = grid
        if not images:
            raise ValueError("image kernels require at least one image")
        self.images: Dict[str, ImageSpec] = {
            str(key): ImageSpec.from_value(value) for key, value in dict(images).items()
        }
        self.observed = _observed_groups(observed, simulation)
        self.misfit = misfit
        if wavelet is not None:
            if weights is not None:
                raise ValueError("provide either weights or wavelet, not both")
            frequencies = np.asarray(wavelet.frequencies)
            spectrum = np.abs(np.asarray(wavelet.spectrum))
            weights = [
                float(spectrum[int(np.abs(frequencies - np.real(f)).argmin())])
                for f in self.f_list
            ]
        self.weights = _frequency_weights(weights, self.n_tasks, label="imaging")
        if isinstance(smoothing, Mapping):
            self.smoothing: Optional[Union[SmoothingConfig, Dict[str, Any]]] = dict(
                smoothing
            )
        else:
            self.smoothing = SmoothingConfig.from_value(smoothing)
        self.keep_forward = _postprocess_bool(keep_forward, "keep_forward")
        self.keep_adjoint = _postprocess_bool(keep_adjoint, "keep_adjoint")
        self.keep_unstacked = _postprocess_bool(keep_unstacked, "keep_unstacked")
        if field_retention is not None:
            field_retention = str(field_retention).strip().lower()
            if field_retention not in _FIELD_RETENTION:
                raise ValueError(
                    f"field_retention must be one of {', '.join(_FIELD_RETENTION)}"
                )
        self.field_retention = field_retention
        self.direction = _input_path(direction, simulation)
        self.zero_direction = _postprocess_bool(zero_direction, "zero_direction")
        self.gauss_newton = _postprocess_bool(gauss_newton, "gauss_newton")
        self.born_traces_only = _postprocess_bool(born_traces_only, "born_traces_only")
        self.save_path = (
            Path(save_path).resolve()
            if save_path is not None
            else self._result_path / "imaging"
        )
        self.kernel_derivative = _kernel_derivative(
            kernel_derivative, residuals=_KERNEL_RESIDUALS_CONTROL
        )
        self.extra = dict(extra)
        self._validate()

    def _validate(self) -> None:
        workflow = self.workflow
        has_direction = self.direction is not None
        if has_direction and self.zero_direction:
            raise ValueError("direction and zero_direction are mutually exclusive")
        if workflow == "rtm":
            if has_direction or self.zero_direction:
                raise ValueError("rtm image kernels do not take a direction")
            if self.gauss_newton or self.born_traces_only:
                raise ValueError(
                    "gauss_newton and born_traces_only apply to Born workflows"
                )
        else:
            if not has_direction and not self.zero_direction:
                raise ValueError(
                    f"{workflow} requires direction or zero_direction=True"
                )
            if self.kernel_derivative is not None:
                raise ValueError("kernel_derivative requires workflow = rtm")
        if workflow == "lsrtm_gradient":
            if self.born_traces_only:
                raise ValueError("lsrtm_gradient cannot be trace-only")
            self.gauss_newton = True
            if not self.observed or any(
                path is None for path in self.observed.values()
            ):
                raise ValueError("lsrtm_gradient requires observed data")
        if self.smoothing is not None and isinstance(self.smoothing, SmoothingConfig):
            if self.smoothing.derivative_order != 1:
                raise ValueError(
                    "Cartesian image smoothing uses mixed first-order FEM fields"
                )

    # -- image products -------------------------------------------------------

    def image_file(self, task: Optional[int] = None) -> Path:
        """Return the committed aggregate or per-task image file path."""

        from frequensolve.simulation.artifact_catalog import load_artifact_catalog
        from frequensolve.simulation.artifact_contract import ArtifactRequest

        catalog = load_artifact_catalog(
            self._result_path,
            tasks=() if task is None else (task,),
            operations=("smooth",) if task is None else (),
        )
        request = ArtifactRequest(role="image", retention="durable")
        records = catalog.select(
            request, **({"operation": "smooth"} if task is None else {"task": task})
        )
        if len(records) != 1:
            raise FileNotFoundError(
                f"Expected one committed image for task {task}; found {len(records)}"
            )
        record = records[0]
        if record.path.stat().st_size != record.bytes:
            raise ValueError(f"Image size changed: {record.path}")
        return record.path

    def load_images(self) -> ImageSet:
        """Open locally present imaging results as an :class:`ImageSet`."""

        files: Dict[Optional[int], Path] = {None: self.image_file()}
        for task in range(1, self.n_tasks + 1):
            try:
                files[task] = self.image_file(task)
            except FileNotFoundError:
                pass
        images = ImageSet(
            path=self.save_path,
            parts=self.n_tasks,
            shape=self.grid.shape,
            artifact_files=files,
            frequencies=tuple(self.f_list),
        )
        images.require_aggregate()
        return images

    @property
    def trace_outputs(self) -> TraceOutputSpec:
        """Return baseline and/or incremental receiver-trace products."""

        baseline = super().trace_outputs
        if self.workflow == "born":
            return _incremental_trace_outputs(baseline, keep_baseline=False)
        if self.workflow == "lsrtm_gradient":
            return _incremental_trace_outputs(baseline, keep_baseline=True)
        return baseline

    # -- postprocess ----------------------------------------------------------

    def requires_postprocess(self) -> bool:
        """Return true because images are stacked and smoothed after tasks."""

        return True

    def postprocess_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate or task-local image path."""

        return self.image_file(part)

    def image_output_exists(self) -> bool:
        """Return whether the aggregate image product exists locally."""

        try:
            return self.image_file().is_file()
        except (FileNotFoundError, ValueError):
            return False

    def postprocess_output_exists(self) -> bool:
        """Return whether the final stacked image exists locally."""

        return self.image_output_exists()

    def postprocess_part_outputs_exist(self) -> bool:
        """Return whether every per-frequency image shard exists locally."""

        try:
            return all(
                self.image_file(part).is_file() for part in range(1, self.n_tasks + 1)
            )
        except (FileNotFoundError, ValueError):
            return False

    def is_run_current(self) -> bool:
        """Return whether the imaging run and aggregate image are current."""

        return super().is_run_current() and self.image_output_exists()

    # -- serialization --------------------------------------------------------

    def _misfit_payload(
        self, ctx: Optional[ExportContext], *, project_relative: bool
    ) -> Dict[str, Any]:
        misfit = _misfit_to_fs(self.misfit, ctx, project_relative=project_relative)
        if misfit is None:
            misfit = copy.deepcopy(_DEFAULT_MISFIT)
        groups = list(misfit.get("receiver_groups") or [])
        by_name = {str(group.get("name")): dict(group) for group in groups}
        names = _receiver_group_names(self.simulation) or list(self.observed or {})
        ordered = []
        for name in names:
            group = by_name.pop(name, {"name": name})
            if "observed" not in group:
                path = (self.observed or {}).get(name)
                group["observed"] = self._export_path(
                    path, project_relative=project_relative
                )
            ordered.append(group)
        ordered.extend(by_name.values())
        misfit["receiver_groups"] = ordered
        return misfit

    def _smoothing_payload(self) -> Optional[Dict[str, Any]]:
        if self.smoothing is None:
            return None
        if isinstance(self.smoothing, SmoothingConfig):
            return self.smoothing.to_image_fs()
        payload = dict(self.smoothing)
        if (
            "illumination_normalization" not in payload
            and "normalize_illumination" not in payload
        ):
            payload["illumination_normalization"] = "none"
        return payload

    def _data_path(self, *, project_relative: bool) -> Optional[Any]:
        paths = {path for path in (self.observed or {}).values() if path is not None}
        if len(paths) == 1:
            return self._export_path(
                next(iter(paths)), project_relative=project_relative
            )
        return None

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize the Cartesian image request."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        imaging: Dict[str, Any] = {
            "schema": "fs-imaging-1",
            "name": self.name,
            "data_path": self._data_path(project_relative=project_relative),
            "save_path": self._export_path(
                self.save_path, project_relative=project_relative
            ),
            "misfit": self._misfit_payload(ctx, project_relative=project_relative),
            "grid": self.grid.to_fs(ctx),
            "keep_forward": self.keep_forward,
            "keep_adjoint": self.keep_adjoint,
            "keep_unstacked": self.keep_unstacked,
            "images": [spec.to_fs(name) for name, spec in self.images.items()],
        }
        if self.field_retention is not None:
            imaging["field_retention"] = self.field_retention
        if self.weights is not None:
            imaging["weights"] = list(self.weights)
        smoothing = self._smoothing_payload()
        if smoothing is not None:
            imaging["Smoothing"] = smoothing
        if self.workflow != "rtm":
            if self.direction is not None:
                imaging["direction"] = self._export_path(
                    self.direction, project_relative=project_relative
                )
            else:
                imaging["zero_direction"] = True
            imaging["gauss_newton"] = self.gauss_newton
            imaging["born_traces_only"] = self.born_traces_only
        imaging.update(copy.deepcopy(self.extra))
        payload["Imaging"] = imaging
        if self.kernel_derivative is not None:
            payload["kernel_derivative"] = dict(self.kernel_derivative)
            payload["Image"] = copy.deepcopy(imaging)
        return payload

    def _export_path(
        self, path: Optional[Union[str, Path]], *, project_relative: bool = False
    ) -> Optional[Any]:
        if path is None:
            return None
        path = Path(path)
        if not project_relative:
            return path
        try:
            return path.resolve().relative_to(self._project_path())
        except ValueError:
            return path

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash the explicit Cartesian direction consumed by incremental kernels."""

        inputs: Dict[str, Any] = {}
        if self.direction is not None:
            inputs["direction"] = self._path_content_fingerprint(self.direction)
        elif self.workflow != "rtm":
            inputs["direction"] = {"kind": "zero"}
        return inputs

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "ImageKernelJob":
        """Deserialize a Cartesian image job."""

        simulation, resolve = cls._load_context(data, base_path, project_path)
        imaging = copy.deepcopy(data.get("Imaging") or data.get("Image"))
        if imaging is None:
            raise KeyError("ImageKernelJob data must include an 'Imaging' section")
        misfit = imaging.pop("misfit", None) or {}
        groups = misfit.get("receiver_groups") or []
        observed = {
            str(group["name"]): resolve(group.get("observed"))
            for group in groups
            if group.get("observed") is not None
        }
        for group in groups:
            group.pop("observed", None)
        images = {
            image["name"]: ImageSpec(
                image.get("IC", "up_down"), image.get("property"), image.get("sources")
            )
            for image in imaging.pop("images", [])
        }
        for key in ("schema", "name", "data_path"):
            imaging.pop(key, None)
        direction = imaging.pop("direction", None)
        job = cls(
            data["name"],
            simulation,
            cls._decode_frequencies(data["f_list"]),
            grid=CartesianGrid.from_fs(imaging.pop("grid")),
            images=images,
            observed=observed or None,
            misfit=misfit,
            weights=imaging.pop("weights", None),
            smoothing=imaging.pop("Smoothing", None),
            keep_forward=imaging.pop("keep_forward", False),
            keep_adjoint=imaging.pop("keep_adjoint", False),
            keep_unstacked=imaging.pop("keep_unstacked", False),
            field_retention=imaging.pop("field_retention", None),
            direction=resolve(direction),
            zero_direction=imaging.pop("zero_direction", False),
            gauss_newton=imaging.pop("gauss_newton", False),
            born_traces_only=imaging.pop("born_traces_only", False),
            workflow=data["workflow"],
            save_path=resolve(imaging.pop("save_path", None)),
            kernel_derivative=data.get("kernel_derivative"),
            outputs=JobOutputs.from_fs(data.get("Outputs")),
            k_list=data.get("k_list"),
            k_weights=data.get("k_weights"),
            k_units=data.get("k_units"),
            **imaging,
        )
        cls._finish_load(job, data)
        return job


# ---------------------------------------------------------------------------
# SmoothJob
# ---------------------------------------------------------------------------


@register_class
class SmoothJob(_ImagingJobBase):
    """Run Sauce's ``--smooth`` postprocess on an existing job's task parts.

    The job carries the source job's payload with ``Smoothing`` and
    ``weights`` installed (``control_sensitivities`` for control gradients,
    ``Imaging`` for Cartesian images). Sites run it as a postprocess-only
    submission (``postprocess_only=True``): no frequency tasks are executed.

    ``input_vector`` smooths one explicit control vector instead of the source
    job's parts through ``control_sensitivities.input``: Sauce copies that file
    to ``raw_gradient``, writes the smoothed ``gradient`` and ignores
    ``weights`` and the scalar objective parts. ``f_list`` stays the source
    job's list because Sauce scales wavelength-relative smoothing by its
    largest frequency. A :class:`ControlVectorFile` is written once in native
    layout to ``<gradient>_input.h5``; a path is used in place (native
    ``/controls/<id>`` or joint ``/controls/model.<id>`` layouts). Blocks
    outside ``model.*`` are left out of ``active`` and ignored by Sauce.

    Args:
        source_job: :class:`FWIOperatorJob`, :class:`ControlGradientJob` or
            :class:`ImageKernelJob` whose parts are smoothed.
        smoothing: :class:`SmoothingConfig` or Sauce smoothing mapping.
        weights: Optional per-frequency weights (parts aggregation only).
        input_vector: Optional ``fs-control-vector-1`` (or native) file or
            :class:`ControlVectorFile` to smooth on its own.
        gradient: Output path for the smoothed vector when ``input_vector`` is
            given (default ``<results>/smoothed.h5``).
        name: Job name (default ``<source name>_smooth``).
    """

    postprocess_only: ClassVar[bool] = True
    supports_trace_packing: ClassVar[bool] = False

    def __init__(
        self,
        source_job: BaseJob,
        *,
        smoothing: Union[SmoothingConfig, Mapping[str, Any]],
        weights: Optional[Sequence[float]] = None,
        input_vector: Optional[Union[str, Path, ControlVectorFile]] = None,
        gradient: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
    ) -> None:
        if isinstance(source_job, ImageKernelJob):
            mode = "image"
        elif isinstance(source_job, (FWIOperatorJob, ControlGradientJob)):
            mode = "control"
        else:
            raise TypeError(
                "SmoothJob requires an FWIOperatorJob, ControlGradientJob or ImageKernelJob"
            )
        if smoothing is None:
            raise ValueError("SmoothJob requires a smoothing configuration")
        if mode == "control":
            if isinstance(source_job, ControlGradientJob) and source_job.kind == "born":
                raise ValueError(
                    "born control sensitivities have no gradient parts to smooth"
                )
            if isinstance(source_job, FWIOperatorJob) and source_job.covector is None:
                raise ValueError(
                    "the source fwi_operator job writes no covector to smooth"
                )
        if input_vector is not None and mode != "control":
            raise ValueError("input_vector smoothing applies to control vectors only")
        if input_vector is not None and weights is not None:
            raise ValueError(
                "weights are ignored when input_vector is given; omit them"
            )
        super().__init__(
            name or f"{source_job.name}_smooth",
            source_job.simulation,
            source_job.workflow,
            _normalized_frequencies(list(source_job.f_list)),
            source_job.outputs,
            k_list=source_job.k_list,
            k_weights=source_job.k_weights,
            k_units=source_job.k_units,
        )
        self.source_job = source_job
        self.mode = mode
        if mode == "image":
            self.smoothing: Union[SmoothingConfig, Dict[str, Any]] = (
                dict(smoothing)
                if isinstance(smoothing, Mapping)
                else SmoothingConfig.from_value(smoothing)
            )
        else:
            config = SmoothingConfig.from_value(smoothing)
            assert config is not None
            self.smoothing = config
        self.weights = _frequency_weights(weights, self.n_tasks, label="smoothing")
        self.input_vector: Optional[Path] = None
        self.gradient: Optional[Path] = None
        self.control_active: Optional[List[str]] = None
        if input_vector is not None:
            self.gradient = _output_path(gradient or "smoothed.h5", self._result_path)
            self.input_vector = self._stage_input_vector(input_vector)
        elif gradient is not None:
            raise ValueError("gradient is only used with input_vector")

    def _stage_input_vector(self, vector: Union[str, Path, ControlVectorFile]) -> Path:
        """Return the ``control_sensitivities.input`` path, writing it if needed.

        A :class:`ControlVectorFile` is written once, in native layout, to
        ``<gradient>_input.h5`` under the result directory. A path is read to
        derive ``active`` and otherwise used as-is.
        """

        assert self.gradient is not None
        if isinstance(vector, ControlVectorFile):
            source = vector
            source_path = self.gradient.with_name(
                f"{self.gradient.stem}_input{self.gradient.suffix}"
            )
        else:
            source_path = Path(vector)
            if not source_path.is_absolute():
                source_path = _input_path(source_path, self.simulation)
            source = ControlVectorFile.read(source_path)
        model_blocks = {
            unqualified_block_name(name): values
            for name, values in source.blocks.items()
            if source.native or name.startswith("model.")
        }
        if not model_blocks:
            raise ValueError("input_vector has no model.* control blocks to smooth")
        self.control_active = list(model_blocks)
        if isinstance(vector, ControlVectorFile):
            ControlVectorFile(model_blocks, native=True).write(source_path)
        return source_path

    # -- output paths ---------------------------------------------------------

    def gradient_file(self, task: Optional[int] = None, *, raw: bool = False) -> Path:
        """Return the smoothed aggregate, raw aggregate, or one input part."""

        if self.gradient is not None:
            stem = self.gradient
            if task is not None:
                raise ValueError("an explicit input vector has no task parts")
            return _raw_path(stem) if raw else stem
        source = self.source_job
        if isinstance(source, FWIOperatorJob):
            return source.covector_file(task, raw=raw)
        if isinstance(source, ControlGradientJob):
            return source.gradient_file(task, raw=raw)
        raise ValueError("image smoothing has no control-gradient file")

    # -- postprocess ----------------------------------------------------------

    def requires_postprocess(self) -> bool:
        """Return true: a smooth job is nothing but the postprocess."""

        return True

    def postprocess_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate, one task-local input part, or the explicit input."""

        if self.mode == "image":
            return self.source_job.postprocess_file(part)
        if part is not None and self.input_vector is not None:
            return self.input_vector
        return self.gradient_file(part)

    def postprocess_fetch_files(self) -> List[Path]:
        """Return the finalized smoothed products."""

        if self.mode == "image":
            return self.source_job.postprocess_fetch_files()
        return [self.gradient_file(raw=True), self.gradient_file()]

    def postprocess_output_exists(self) -> bool:
        """Return whether the smoothed product exists locally."""

        if self.mode == "image":
            return self.source_job.postprocess_output_exists()
        return self.gradient_file().is_file()

    def postprocess_part_outputs_exist(self) -> bool:
        """Return whether the explicit input or every input part exists locally."""

        if self.mode == "image":
            return self.source_job.postprocess_part_outputs_exist()
        if self.input_vector is not None:
            return self.input_vector.is_file()
        return all(
            self.gradient_file(part).is_file() for part in range(1, self.n_tasks + 1)
        )

    def is_run_current(self) -> bool:
        """Return whether the smoothed product is current."""

        return not self.needs_postprocess() and self.postprocess_output_exists()

    # -- serialization --------------------------------------------------------

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize the source payload with the smoothing request installed."""

        source_payload = self.source_job.to_fs(ctx, project_relative=project_relative)
        for key in ("artifact_contract", "job_id"):
            source_payload.pop(key, None)
        base = super().to_fs(ctx, project_relative=project_relative)
        payload = {**copy.deepcopy(source_payload), **base}
        payload["smooth_source"] = source_payload
        if self.mode == "image":
            imaging = payload.setdefault("Imaging", {})
            imaging["Smoothing"] = (
                self.smoothing.to_image_fs()
                if isinstance(self.smoothing, SmoothingConfig)
                else dict(self.smoothing)
            )
            if self.weights is not None:
                imaging["weights"] = list(self.weights)
            if "Image" in payload:
                payload["Image"] = copy.deepcopy(imaging)
        else:
            sensitivities = payload.setdefault("control_sensitivities", {})
            assert isinstance(self.smoothing, SmoothingConfig)
            sensitivities["Smoothing"] = self.smoothing.to_control_fs()
            if self.weights is not None:
                sensitivities["weights"] = list(self.weights)
            if self.gradient is not None:
                sensitivities["input"] = _job_path(self.input_vector, ctx, False)
                sensitivities["gradient"] = _job_path(self.gradient, ctx, False)
                sensitivities.pop("raw_gradient", None)
                sensitivities.pop("weights", None)
                sensitivities["active"] = list(self.control_active or [])
            elif "gradient" not in sensitivities:
                sensitivities["gradient"] = _job_path(self.gradient_file(), ctx, False)
        return payload

    def _input_fingerprint_payload(self) -> Dict[str, Any]:
        """Hash the explicit input vector or the source task parts."""

        if self.input_vector is not None:
            return {"input_vector": self._path_content_fingerprint(self.input_vector)}
        if self.mode == "image":
            return {}
        return {
            "parts": {
                str(task): self._path_content_fingerprint(self.gradient_file(task))
                for task in range(1, self.n_tasks + 1)
            }
        }

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "SmoothJob":
        """Deserialize a smooth job together with its embedded source job."""

        source = BaseJob.from_fs(
            dict(data["smooth_source"]), base_path=base_path, project_path=project_path
        )
        if isinstance(source, ImageKernelJob):
            smoothing = (data.get("Imaging") or data.get("Image") or {}).get(
                "Smoothing"
            )
            weights = (data.get("Imaging") or {}).get("weights")
            if weights == source.weights:
                weights = None
            input_vector = None
            gradient = None
        else:
            sensitivities = data.get("control_sensitivities") or {}
            smoothing = sensitivities.get("Smoothing")
            input_vector = sensitivities.get("input")
            weights = None if input_vector is not None else sensitivities.get("weights")
            gradient = (
                sensitivities.get("gradient") if input_vector is not None else None
            )
        source_project = data.get("project_path")
        resolved_project = project_path or source_project

        def resolve(value: Optional[Union[str, Path]]) -> Optional[Path]:
            return _resolve_saved_job_path(
                value,
                base_path=base_path,
                project_path=resolved_project,
                source_project=source_project,
            )

        job = cls(
            source,
            smoothing=smoothing,
            weights=weights,
            input_vector=resolve(input_vector),
            gradient=resolve(gradient),
            name=data["name"],
        )
        cls._finish_load(job, data)
        return job
