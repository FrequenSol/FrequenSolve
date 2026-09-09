"""Frequency-independent acoustic ray-tracing jobs.

Sauce exposes ray tracing as ``workflow: "raytrace"`` with an
``fs-ray-tracing-1`` configuration and an authoritative ``fs-rays-1`` HDF5
result.  The workflow is a single, serial solver task and does not use
``f_list`` or the trace-packing phase.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.jobs.run_state import SkipPolicy, TaskRunPlan
from frequensolve.simulation.outputs import JobOutputs
from frequensolve.simulation.simulation import BaseSimulation
from frequensolve.util.class_registry import register_class

__all__ = [
    "RayLaunch",
    "RaySources",
    "RayTracingConfig",
    "RayTracingJob",
]


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(value))


def _unknown(data: Mapping[str, Any], allowed: Iterable[str], name: str) -> None:
    fields = sorted(set(data) - set(allowed))
    if fields:
        raise ValueError(f"Unknown {name} field(s): {', '.join(fields)}")


def _required(data: Mapping[str, Any], fields: Iterable[str], name: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"{name} requires: {', '.join(missing)}")


def _choice(value: Any, choices: Iterable[str], name: str) -> str:
    result = str(value)
    allowed = tuple(choices)
    if result not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    return result


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value: Any, name: str) -> None:
    scalar = value.get("value") if isinstance(value, Mapping) else value
    try:
        valid = float(scalar) > 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be positive")


def _output_path(value: Any, name: str) -> str:
    path = Path(str(value))
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a non-empty relative path")
    return path.as_posix()


class RaySources:
    """Factories for ``fs-ray-tracing-1`` source selections."""

    @staticmethod
    def acquisition(
        names: Optional[Iterable[str]] = None,
        *,
        incident_domain: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Select all or named physical sources from the acquisition."""

        result: Dict[str, Any] = {"kind": "acquisition"}
        if names is not None:
            result["names"] = [str(name) for name in names]
        if incident_domain is not None:
            result["incident_domain"] = int(incident_domain)
        return result

    @staticmethod
    def explicit(
        points: Iterable[Mapping[str, Any]],
        *,
        incident_domain: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Select explicitly supplied named coordinate points."""

        result: Dict[str, Any] = {
            "kind": "explicit",
            "points": [copy.deepcopy(dict(point)) for point in points],
        }
        if incident_domain is not None:
            result["incident_domain"] = int(incident_domain)
        return result


class RayLaunch:
    """Factories for deterministic ray launch-direction rules."""

    @staticmethod
    def fan2d(
        count: int,
        angle_min: float,
        angle_max: float,
        *,
        angle_units: str = "deg",
        include_endpoints: bool = True,
    ) -> Dict[str, Any]:
        """Launch a two-dimensional angular fan."""

        return {
            "kind": "fan2d",
            "count": int(count),
            "angle_min": float(angle_min),
            "angle_max": float(angle_max),
            "angle_units": angle_units,
            "include_endpoints": bool(include_endpoints),
        }

    @staticmethod
    def sphere(count: int) -> Dict[str, Any]:
        """Launch a deterministic three-dimensional sphere."""

        return {"kind": "sphere", "count": int(count)}

    @staticmethod
    def hemisphere(count: int, axis: Any) -> Dict[str, Any]:
        """Launch a hemisphere centered on ``axis``."""

        return {"kind": "hemisphere", "count": int(count), "axis": axis}

    @staticmethod
    def cone(
        count: int,
        axis: Any,
        half_angle: float,
        *,
        angle_units: str = "deg",
    ) -> Dict[str, Any]:
        """Launch directions inside a cone centered on ``axis``."""

        return {
            "kind": "cone",
            "count": int(count),
            "axis": axis,
            "half_angle": float(half_angle),
            "angle_units": angle_units,
        }

    @staticmethod
    def explicit(directions: Iterable[Any]) -> Dict[str, Any]:
        """Launch explicitly supplied nonzero directions."""

        return {"kind": "explicit", "directions": copy.deepcopy(list(directions))}


class RayTracingConfig:
    """Validated authoring object for Sauce's ``fs-ray-tracing-1`` contract.

    Only ``launch`` is required by the Python constructor. Other sections use
    conservative public-contract defaults and remain fully overrideable.
    ``integrator`` may omit a physical termination limit, in which case its
    ``max_steps`` bound still terminates each ray.
    """

    schema = "fs-ray-tracing-1"

    def __init__(
        self,
        launch: Mapping[str, Any],
        *,
        sources: Optional[Mapping[str, Any]] = None,
        execution: Optional[Mapping[str, Any]] = None,
        integrator: Optional[Mapping[str, Any]] = None,
        branching: Optional[Mapping[str, Any]] = None,
        weights: Optional[Mapping[str, Any]] = None,
        receivers: Optional[Mapping[str, Any]] = None,
        boundaries: Optional[Mapping[str, Any]] = None,
        path_storage: Optional[Mapping[str, Any]] = None,
        output: Optional[Mapping[str, Any]] = None,
        failure_policy: Optional[Mapping[str, Any]] = None,
    ):
        payload: Dict[str, Any] = {
            "schema": self.schema,
            "physics": "acoustic",
            "sources": _mapping(
                sources if sources is not None else RaySources.acquisition(),
                "sources",
            ),
            "launch": _mapping(launch, "launch"),
            "integrator": {
                "method": "dopri5",
                "parameter": "travel_time",
                "rel_tol": 1.0e-8,
                "abs_tol": 1.0e-10,
                "max_reject": 16,
                "max_steps": 100000,
                **_mapping(integrator or {}, "integrator"),
            },
            "branching": {
                "mode": "transmit",
                "max_interactions_per_ray": 16,
                "max_total_rays": 100000,
                **_mapping(branching or {}, "branching"),
            },
            "weights": {
                "angular": {"kind": "generated", "normalize": True},
                "interface": "energy",
                "q_attenuation": False,
                "store_q_integral": False,
                **_mapping(weights or {}, "weights"),
            },
            "receivers": {"enabled": False, **_mapping(receivers or {}, "receivers")},
            "boundaries": {
                "default": "terminate",
                **_mapping(boundaries or {}, "boundaries"),
            },
            "path_storage": {
                "mode": "adaptive",
                "max_points_per_ray": 20000,
                "max_storage_bytes": 1073741824,
                "store": ["p", "root_id", "region_id"],
                **_mapping(path_storage or {}, "path_storage"),
            },
            "output": {
                "directory": "rays",
                "hdf5_file": "rays.h5",
                "write_vtp": True,
                "vtp_file": "rays.vtp",
                "overwrite": False,
                **_mapping(output or {}, "output"),
            },
            "failure_policy": {
                "invalid_source": "error",
                "invalid_material": "error",
                "geometry_failure": "error",
                "step_failure": "terminate_ray",
                "branch_budget": "truncate",
                "path_budget": "truncate",
                "max_failed_rays": 100,
                "max_failed_fraction": 0.05,
                **_mapping(failure_policy or {}, "failure_policy"),
            },
        }
        if execution is not None:
            payload["execution"] = _mapping(execution, "execution")
        if not payload["output"]["write_vtp"]:
            payload["output"].pop("vtp_file", None)
        self._validate(payload)
        self._data = payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "RayTracingConfig":
        """Load an exact solver-style ray-tracing configuration."""

        payload = _mapping(data, "RayTracing")
        cls._validate(payload)
        result = cls.__new__(cls)
        result._data = payload
        return result

    def to_fs(self) -> Dict[str, Any]:
        """Return an isolated JSON-compatible solver payload."""

        return copy.deepcopy(self._data)

    def with_updates(self, **sections: Mapping[str, Any]) -> "RayTracingConfig":
        """Return a copy with complete top-level sections replaced."""

        payload = self.to_fs()
        for name, value in sections.items():
            if name not in payload and name != "execution":
                raise ValueError(f"Unknown RayTracing section: {name}")
            payload[name] = _mapping(value, name)
        return self.from_fs(payload)

    @classmethod
    def _validate(cls, payload: Mapping[str, Any]) -> None:
        top = {
            "schema",
            "physics",
            "sources",
            "launch",
            "execution",
            "integrator",
            "branching",
            "weights",
            "receivers",
            "boundaries",
            "path_storage",
            "output",
            "failure_policy",
        }
        _unknown(payload, top, "RayTracing")
        _required(payload, top - {"execution"}, "RayTracing")
        if payload["schema"] != cls.schema:
            raise ValueError(f"RayTracing.schema must be {cls.schema!r}")
        if payload["physics"] != "acoustic":
            raise ValueError("fs-ray-tracing-1 supports acoustic physics only")
        cls._validate_sources(_mapping(payload["sources"], "sources"))
        cls._validate_launch(_mapping(payload["launch"], "launch"))
        if "execution" in payload:
            cls._validate_execution(_mapping(payload["execution"], "execution"))
        cls._validate_integrator(_mapping(payload["integrator"], "integrator"))
        cls._validate_branching(_mapping(payload["branching"], "branching"))
        cls._validate_weights(_mapping(payload["weights"], "weights"))
        cls._validate_receivers(_mapping(payload["receivers"], "receivers"))
        cls._validate_boundaries(_mapping(payload["boundaries"], "boundaries"))
        cls._validate_storage(_mapping(payload["path_storage"], "path_storage"))
        cls._validate_output(_mapping(payload["output"], "output"))
        cls._validate_failures(_mapping(payload["failure_policy"], "failure_policy"))

    @staticmethod
    def _validate_sources(data: Mapping[str, Any]) -> None:
        _unknown(data, {"kind", "names", "points", "incident_domain"}, "sources")
        _required(data, {"kind"}, "sources")
        kind = _choice(data["kind"], ("acquisition", "explicit"), "sources.kind")
        if "incident_domain" in data:
            _integer(data["incident_domain"], "sources.incident_domain", minimum=0)
        if kind == "acquisition":
            if "points" in data:
                raise ValueError("acquisition sources cannot define points")
            if "names" in data and not data["names"]:
                raise ValueError("sources.names cannot be empty")
        else:
            _required(data, {"points"}, "explicit sources")
            if "names" in data:
                raise ValueError("explicit sources cannot define names")
            RayTracingConfig._validate_named_points(data["points"], "sources.points")

    @staticmethod
    def _validate_named_points(points: Any, name: str) -> None:
        if not isinstance(points, list) or not points:
            raise ValueError(f"{name} must be a non-empty list")
        names = []
        for index, point in enumerate(points):
            item = _mapping(point, f"{name}[{index}]")
            _unknown(item, {"name", "coordinates"}, f"{name}[{index}]")
            _required(item, {"name", "coordinates"}, f"{name}[{index}]")
            names.append(str(item["name"]))
        if any(not value for value in names) or len(set(names)) != len(names):
            raise ValueError(f"{name} names must be non-empty and unique")

    @staticmethod
    def _validate_launch(data: Mapping[str, Any]) -> None:
        kind = _choice(
            data.get("kind"),
            ("explicit", "fan2d", "sphere", "hemisphere", "cone"),
            "launch.kind",
        )
        allowed = {
            "explicit": {"kind", "directions"},
            "fan2d": {
                "kind",
                "count",
                "angle_min",
                "angle_max",
                "angle_units",
                "include_endpoints",
            },
            "sphere": {"kind", "count"},
            "hemisphere": {"kind", "count", "axis"},
            "cone": {"kind", "count", "axis", "half_angle", "angle_units"},
        }[kind]
        _unknown(data, allowed, "launch")
        required = allowed - {"include_endpoints"}
        _required(data, required, f"{kind} launch")
        if kind == "explicit":
            if not isinstance(data["directions"], list) or not data["directions"]:
                raise ValueError("launch.directions must be a non-empty list")
            return
        _integer(data["count"], "launch.count", minimum=1)
        if "angle_units" in data:
            _choice(data["angle_units"], ("deg", "rad"), "launch.angle_units")
        if kind == "cone":
            _positive(data["half_angle"], "launch.half_angle")

    @staticmethod
    def _validate_execution(data: Mapping[str, Any]) -> None:
        _unknown(data, {"backend", "precision", "allow_cpu_fallback"}, "execution")
        _required(data, {"backend", "precision"}, "execution")
        backend = _choice(
            data["backend"], ("cpu", "auto", "metal"), "execution.backend"
        )
        precision = _choice(
            data["precision"],
            ("auto", "float32", "float64"),
            "execution.precision",
        )
        if precision == "float32" and backend == "cpu":
            raise ValueError("float32 execution requires backend='auto' or 'metal'")
        if precision == "float64" and backend == "metal":
            raise ValueError("float64 execution requires backend='auto' or 'cpu'")
        if data.get("allow_cpu_fallback") and precision != "auto":
            raise ValueError("allow_cpu_fallback requires precision='auto'")

    @staticmethod
    def _validate_integrator(data: Mapping[str, Any]) -> None:
        allowed = {
            "method",
            "parameter",
            "rel_tol",
            "abs_tol",
            "h_min",
            "h_max",
            "safety",
            "min_factor",
            "max_factor",
            "max_reject",
            "max_steps",
            "tau_max",
            "arc_length_max",
            "event_tol",
            "hamiltonian_tol",
            "hamiltonian_failure_tol",
        }
        _unknown(data, allowed, "integrator")
        _required(data, {"method", "parameter"}, "integrator")
        if data["method"] != "dopri5" or data["parameter"] != "travel_time":
            raise ValueError(
                "ray integration requires dopri5 parameterized by travel_time"
            )
        for field in (
            "rel_tol",
            "abs_tol",
            "h_min",
            "h_max",
            "safety",
            "min_factor",
            "max_factor",
            "tau_max",
            "arc_length_max",
            "event_tol",
            "hamiltonian_tol",
            "hamiltonian_failure_tol",
        ):
            if field in data:
                _positive(data[field], f"integrator.{field}")
        if "max_reject" in data:
            _integer(data["max_reject"], "integrator.max_reject", minimum=0)
        if "max_steps" in data:
            _integer(data["max_steps"], "integrator.max_steps", minimum=1)

    @staticmethod
    def _validate_branching(data: Mapping[str, Any]) -> None:
        allowed = {
            "mode",
            "max_interactions_per_ray",
            "max_rays_per_source",
            "max_total_rays",
            "max_branch_depth",
            "critical_tolerance",
            "minimum_energy",
        }
        _unknown(data, allowed, "branching")
        _required(
            data,
            {"mode", "max_interactions_per_ray", "max_total_rays"},
            "branching",
        )
        mode = _choice(data["mode"], ("transmit", "reflect", "both"), "branching.mode")
        _integer(
            data["max_interactions_per_ray"],
            "branching.max_interactions_per_ray",
            minimum=0,
        )
        _integer(data["max_total_rays"], "branching.max_total_rays", minimum=1)
        if mode == "both":
            _required(
                data,
                {"max_rays_per_source", "max_branch_depth"},
                "branching mode 'both'",
            )
        if "max_rays_per_source" in data:
            _integer(
                data["max_rays_per_source"], "branching.max_rays_per_source", minimum=1
            )
        if "max_branch_depth" in data:
            _integer(data["max_branch_depth"], "branching.max_branch_depth", minimum=0)

    @staticmethod
    def _validate_weights(data: Mapping[str, Any]) -> None:
        _unknown(
            data,
            {"angular", "interface", "q_attenuation", "store_q_integral"},
            "weights",
        )
        _required(
            data,
            {"angular", "interface", "q_attenuation", "store_q_integral"},
            "weights",
        )
        angular = _mapping(data["angular"], "weights.angular")
        _unknown(angular, {"kind", "values", "normalize"}, "weights.angular")
        _required(angular, {"kind"}, "weights.angular")
        kind = _choice(
            angular["kind"],
            ("generated", "uniform", "explicit"),
            "weights.angular.kind",
        )
        if kind == "explicit":
            if not isinstance(angular.get("values"), list) or not angular["values"]:
                raise ValueError("explicit angular weights require non-empty values")
        elif "values" in angular:
            raise ValueError("weights.angular.values is valid only for kind='explicit'")
        _choice(
            data["interface"],
            ("none", "pressure", "energy", "both"),
            "weights.interface",
        )
        if data["store_q_integral"] and not data["q_attenuation"]:
            raise ValueError("store_q_integral=true requires q_attenuation=true")

    @staticmethod
    def _validate_receivers(data: Mapping[str, Any]) -> None:
        _unknown(
            data, {"enabled", "kind", "groups", "points", "capture_radius"}, "receivers"
        )
        _required(data, {"enabled"}, "receivers")
        if not data["enabled"]:
            return
        _required(data, {"kind", "capture_radius"}, "enabled receivers")
        kind = _choice(data["kind"], ("acquisition", "explicit"), "receivers.kind")
        _positive(data["capture_radius"], "receivers.capture_radius")
        if kind == "acquisition":
            if not isinstance(data.get("groups"), list) or not data["groups"]:
                raise ValueError("acquisition receivers require non-empty groups")
            if "points" in data:
                raise ValueError("acquisition receivers cannot define points")
        else:
            if "groups" in data:
                raise ValueError("explicit receivers cannot define groups")
            RayTracingConfig._validate_named_points(
                data.get("points"), "receivers.points"
            )

    @staticmethod
    def _validate_boundaries(data: Mapping[str, Any]) -> None:
        _unknown(data, {"default", "rules"}, "boundaries")
        _required(data, {"default"}, "boundaries")
        choices = ("terminate", "pressure_release", "rigid")
        _choice(data["default"], choices, "boundaries.default")
        for index, rule in enumerate(data.get("rules", [])):
            item = _mapping(rule, f"boundaries.rules[{index}]")
            _unknown(item, {"boundary", "policy"}, f"boundaries.rules[{index}]")
            _required(item, {"boundary", "policy"}, f"boundaries.rules[{index}]")
            _choice(item["policy"], choices, f"boundaries.rules[{index}].policy")

    @staticmethod
    def _validate_storage(data: Mapping[str, Any]) -> None:
        _unknown(
            data,
            {"mode", "interval", "max_points_per_ray", "max_storage_bytes", "store"},
            "path_storage",
        )
        _required(
            data, {"mode", "max_points_per_ray", "max_storage_bytes"}, "path_storage"
        )
        mode = _choice(
            data["mode"],
            ("adaptive", "uniform_tau", "uniform_arc_length", "events_only"),
            "path_storage.mode",
        )
        if mode.startswith("uniform_"):
            _required(data, {"interval"}, "uniform path storage")
            _positive(data["interval"], "path_storage.interval")
        elif "interval" in data:
            raise ValueError("path_storage.interval is valid only for uniform modes")
        _integer(
            data["max_points_per_ray"], "path_storage.max_points_per_ray", minimum=2
        )
        _integer(data["max_storage_bytes"], "path_storage.max_storage_bytes", minimum=1)
        allowed_store = {"eta", "p", "direction", "root_id", "region_id", "step_index"}
        store = data.get("store", [])
        if (
            not isinstance(store, list)
            or len(set(store)) != len(store)
            or not set(store) <= allowed_store
        ):
            raise ValueError(
                "path_storage.store contains unsupported or duplicate fields"
            )

    @staticmethod
    def _validate_output(data: Mapping[str, Any]) -> None:
        _unknown(
            data,
            {"directory", "hdf5_file", "write_vtp", "vtp_file", "overwrite"},
            "output",
        )
        _required(data, {"directory", "hdf5_file", "write_vtp"}, "output")
        _output_path(data["directory"], "output.directory")
        _output_path(data["hdf5_file"], "output.hdf5_file")
        if data["write_vtp"]:
            _required(data, {"vtp_file"}, "output with write_vtp=true")
            _output_path(data["vtp_file"], "output.vtp_file")
        elif "vtp_file" in data:
            raise ValueError("output.vtp_file requires write_vtp=true")

    @staticmethod
    def _validate_failures(data: Mapping[str, Any]) -> None:
        fields = {
            "invalid_source",
            "invalid_material",
            "geometry_failure",
            "step_failure",
            "branch_budget",
            "path_budget",
            "max_failed_rays",
            "max_failed_fraction",
        }
        _unknown(data, fields, "failure_policy")
        _required(data, fields, "failure_policy")
        _choice(
            data["invalid_source"],
            ("error", "skip_source"),
            "failure_policy.invalid_source",
        )
        for field in ("invalid_material", "geometry_failure", "step_failure"):
            _choice(data[field], ("error", "terminate_ray"), f"failure_policy.{field}")
        for field in ("branch_budget", "path_budget"):
            _choice(data[field], ("error", "truncate"), f"failure_policy.{field}")
        _integer(data["max_failed_rays"], "failure_policy.max_failed_rays", minimum=0)
        fraction = float(data["max_failed_fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "failure_policy.max_failed_fraction must be between 0 and 1"
            )


@register_class
class RayTracingJob(BaseJob):
    """Single-task isotropic acoustic ray-tracing workflow.

    Sauce currently supports this public workflow in 2D and 3D on one MPI
    rank. The job is frequency independent and writes ``fs-rays-1`` HDF5.
    """

    supports_trace_packing = False
    frequency_independent = True
    max_ranks_per_task = 1

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        config: Optional[Union[RayTracingConfig, Mapping[str, Any]]] = None,
        *,
        launch: Optional[Mapping[str, Any]] = None,
        sources: Optional[Mapping[str, Any]] = None,
        **config_sections: Mapping[str, Any],
    ):
        if config is not None and (
            launch is not None or sources is not None or config_sections
        ):
            raise ValueError("Pass either config or launch/section arguments, not both")
        if config is None:
            if launch is None:
                raise ValueError("RayTracingJob requires a launch rule")
            config = RayTracingConfig(launch, sources=sources, **config_sections)
        elif isinstance(config, Mapping):
            config = RayTracingConfig.from_fs(config)
        if not isinstance(config, RayTracingConfig):
            raise TypeError("config must be a RayTracingConfig or mapping")
        super().__init__(name, simulation, "raytrace", [0.0], JobOutputs())
        self.ray_tracing = config
        self._validate_simulation_support()

    def _validate_simulation_support(self) -> None:
        physics = str(getattr(self.simulation, "physics", "")).lower()
        if physics != "acoustic":
            raise ValueError("fs-ray-tracing-1 supports acoustic simulations only")
        dimension = int(getattr(self.simulation, "dimension", 0))
        if dimension not in {2, 3}:
            raise ValueError("Ray tracing supports only 2D and 3D simulations")
        launch_kind = self.ray_tracing.to_fs()["launch"]["kind"]
        if dimension == 2 and launch_kind in {"sphere", "hemisphere", "cone"}:
            raise ValueError(f"{launch_kind} ray launches require a 3D simulation")
        if dimension == 3 and launch_kind == "fan2d":
            raise ValueError("fan2d ray launches require a 2D simulation")

    def validate_outputs(self) -> None:
        """Validate the ray contract instead of frequency-domain outputs."""

        self.ray_tracing._validate(self.ray_tracing.to_fs())
        self._validate_simulation_support()

    def to_fs(self, ctx=None, *, project_relative: bool = False) -> Dict[str, Any]:
        """Serialize a Sauce ``raytrace`` job without ``f_list`` or ``Outputs``."""

        payload = super().to_fs(ctx, project_relative=project_relative)
        payload.pop("f_list", None)
        payload.pop("Outputs", None)
        payload["RayTracing"] = self.ray_tracing.to_fs()
        return payload

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "RayTracingJob":
        """Deserialize a saved ray-tracing job."""

        simulation = BaseJob._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=project_path or data.get("project_path"),
            source_project=data.get("project_path"),
        )
        job = cls(
            name=str(data["name"]),
            simulation=simulation,
            config=RayTracingConfig.from_fs(data["RayTracing"]),
        )
        job._job_id = data.get("job_id")
        return job

    @property
    def output_directory(self) -> Path:
        """Directory containing the ray manifest and numerical products."""

        return self._result_path / self.ray_tracing.to_fs()["output"]["directory"]

    @property
    def ray_manifest_file(self) -> Path:
        """Path to the ``fs-rays-1`` manifest."""

        return self.output_directory / "manifest.json"

    @property
    def ray_hdf5_file(self) -> Path:
        """Path to the authoritative indexed ray HDF5 file."""

        return self.output_directory / self.ray_tracing.to_fs()["output"]["hdf5_file"]

    @property
    def result_manifest_file(self) -> Path:
        """Generic frequency-independent product manifest path."""

        return self.ray_manifest_file

    @property
    def result_hdf5_file(self) -> Path:
        """Generic frequency-independent authoritative HDF5 path."""

        return self.ray_hdf5_file

    @property
    def ray_vtp_file(self) -> Optional[Path]:
        """Path to the optional derived VTP product."""

        output = self.ray_tracing.to_fs()["output"]
        if not output["write_vtp"]:
            return None
        return self.output_directory / output["vtp_file"]

    @property
    def results(self):
        """Open the typed ``fs-rays-1`` result handle for this job."""

        from frequensolve.seismic.rays import RayResults

        return RayResults.from_job(self)

    def plot(self, **kwargs):
        """Plot retained ray paths from this job's local results."""

        return self.results.plot(**kwargs)

    def results_exist(self) -> bool:
        """Return whether a readable non-failed ray product exists."""

        if not self.ray_manifest_file.is_file() or not self.ray_hdf5_file.is_file():
            return False
        try:
            manifest = json.loads(self.ray_manifest_file.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("schema") == "fs-rays-1" and manifest.get("status") in {
            "complete",
            "partial",
        }

    def _remove_result_products(self) -> bool:
        removed = False
        for path in (self.ray_manifest_file, self.ray_hdf5_file, self.ray_vtp_file):
            if path is None:
                continue
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                pass
        return removed

    def is_task_current(self, task: int, *, state=None, manifest=None) -> bool:
        """Return whether the one frequency-independent task is current."""

        if task != 1 or not self.results_exist():
            return False
        state = self.run_state() if state is None else state
        if state.get("fingerprint") != self.fingerprint():
            return False
        if state.get("status") not in {"completed", "skipped"}:
            return False
        expected = self.task_fingerprint(1)
        for record in self._state_task_records(state):
            if self._task_number_from_record(record) != 1:
                continue
            if record.get("fingerprint") not in {None, expected}:
                continue
            return self._normalized_task_status(record.get("status")) == "succeeded"
        return False

    def current_tasks(self):
        """Return ``[1]`` when the ray product matches this job definition."""

        return [1] if self.is_task_current(1) else []

    def is_run_current(self) -> bool:
        """Return whether the local ray product and run state are current."""

        return self.is_task_current(1)

    def plan_tasks(
        self,
        *,
        skip_policy=None,
        reuse: bool = False,
        force: bool = False,
        apply: bool = False,
        residual=None,
        ignore_solver_options=None,
    ) -> TaskRunPlan:
        """Plan the single ray-tracing task without trace-shard reuse."""

        policy = SkipPolicy.from_value(
            skip_policy,
            residual=residual,
            ignore_solver_options=ignore_solver_options,
            reuse=False,
        )
        force = bool(force or policy.force)
        current = not force and self.is_task_current(1)
        records = []
        if current:
            records.append(
                {
                    "task": 1,
                    "status": "current",
                    "duration_seconds": 0.0,
                    "fingerprint": self.task_fingerprint(1),
                    "compatibility_fingerprint": self.task_policy_fingerprint(
                        1, SkipPolicy.compatible()
                    ),
                    "path": str(self.ray_hdf5_file),
                }
            )
        removed = False
        if apply and not current:
            removed = self._remove_result_products()
        return TaskRunPlan(
            {
                "pending_indices": [] if current else [0],
                "strict_current_tasks": [1] if current else [],
                "current_tasks": [1] if current else [],
                "reused_tasks": [],
                "accepted_tasks": [],
                "accepted_failed_tasks": [],
                "skipped_task_records": records,
                "skip_policy": policy,
                "removed_stale_outputs": removed,
            }
        )

    def task_run_plan(self, **kwargs) -> TaskRunPlan:
        """Apply and return the single-task execution plan."""

        return self.plan_tasks(apply=True, **kwargs)

    def write_run_state(self, status: str = "completed", **extra) -> Path:
        """Write run state keyed to the ray HDF5 product."""

        self._result_path.mkdir(parents=True, exist_ok=True)
        extra = dict(extra)
        task_results = self._as_records(extra.pop("tasks", None))
        result = dict(self._task_record_by_task(task_results).get(1, {}))
        task_status = self._normalized_task_status(result.get("status"))
        product_exists = self.results_exist()
        if task_status is None:
            task_status = (
                "succeeded"
                if status in {"completed", "skipped"} and product_exists
                else "not_run"
            )
        elif task_status == "succeeded" and not product_exists:
            task_status = "failed"
        row = {
            "task": 1,
            "frequency": None,
            "status": task_status,
            "complete": task_status == "succeeded" and product_exists,
            "fingerprint": self.task_fingerprint(1),
            "compatibility_fingerprint": self.task_policy_fingerprint(
                1, SkipPolicy.compatible()
            ),
            "path": str(self.ray_hdf5_file),
            "exists": product_exists,
        }
        for key in (
            "duration_seconds",
            "returncode",
            "n_ranks",
            "ranks",
            "threads_per_rank",
            "n_threads",
            "threads",
            "outputs_manifest",
            "artifacts",
        ):
            if key in result:
                row[key] = result[key]
        payload = {
            "schema": "frequensolve-python-run-1",
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self.fingerprint(),
            "fingerprint_payload": self.fingerprint_payload(),
            "tasks": [row],
            "task_summary": {
                "total": 1,
                "complete": int(row["complete"]),
                "succeeded": int(task_status == "succeeded"),
                "failed": int(task_status == "failed"),
                "not_run": int(task_status == "not_run"),
            },
            "outputs": {
                "rays": [
                    {
                        "path": str(self.ray_manifest_file),
                        "exists": self.ray_manifest_file.is_file(),
                    },
                    {
                        "path": str(self.ray_hdf5_file),
                        "exists": self.ray_hdf5_file.is_file(),
                    },
                ]
            },
        }
        if task_results:
            payload["task_results"] = task_results
        payload.update(extra)
        self._write_json_file(self.run_state_file, payload)
        self._write_solver_run_manifest_summary(payload)
        return self.run_state_file

    def frequency_status(self):
        """Return the single workflow status row with no synthetic frequency."""

        state = self.run_state()
        current = self.is_task_current(1, state=state)
        records = self._state_task_records(state)
        status = "succeeded" if current else "not_run"
        metadata = dict(records[0]) if records else {}
        if (
            records
            and self._normalized_task_status(records[0].get("status")) == "failed"
        ):
            status = "failed"
        return [
            {
                "task": 1,
                "frequency": None,
                "status": status,
                "trace_file": self.ray_hdf5_file,
                "result_file": self.ray_hdf5_file,
                "trace_exists": self.ray_hdf5_file.is_file(),
                "current": current,
                "duration_seconds": metadata.get("duration_seconds"),
                "metadata": metadata,
            }
        ]
