"""Frequency-independent acoustic Eikonal first-arrival jobs.

Sauce exposes the workflow as ``workflow: "eikonal"`` with an
``fs-eikonal-1`` configuration and an authoritative
``fs-eikonal-output-1`` HDF5 result. Version 1 is a single real64 solver task
on one MPI rank and does not use ``f_list`` or trace packing.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import numpy as np

from frequensolve.geometry.frame import coordinate_value_to_fs
from frequensolve.seismic.eikonal import EikonalResults
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.simulation.jobs.run_state import SkipPolicy, TaskRunPlan
from frequensolve.simulation.outputs import JobOutputs
from frequensolve.simulation.simulation import BaseSimulation
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import ExportContext
from frequensolve.util.physics import canonical_dimension

__all__ = [
    "EikonalConfig",
    "EikonalJob",
    "EikonalReceivers",
    "EikonalSources",
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
    if isinstance(value, Mapping):
        item = _mapping(value, name)
        _unknown(item, {"value", "units"}, name)
        _required(item, {"value", "units"}, name)
        if not isinstance(item["units"], str) or not item["units"]:
            raise ValueError(f"{name}.units must be a non-empty string")
        value = item["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        scalar = math.nan
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _output_path(value: Any, name: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a non-empty relative path")
    return path.as_posix()


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [str(item) for item in value]
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} values must be non-empty and unique")
    return result


def _point_coordinates(value: Any, name: str, dimension: Optional[int] = None) -> Any:
    """Normalize a finite physical-coordinate vector for an explicit point."""

    payload = coordinate_value_to_fs(value)
    raw = payload.get("value") if isinstance(payload, Mapping) else payload
    array = np.asarray(raw)
    if (
        array.ndim != 1
        or array.dtype.kind not in {"i", "u", "f"}
        or array.size not in {2, 3}
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must contain two or three finite numeric coordinates")
    if dimension is not None and array.size != dimension:
        raise ValueError(
            f"{name} must have {dimension} coordinates for this simulation"
        )
    return payload


def _named_points(points: Any, name: str) -> None:
    if not isinstance(points, list) or not points:
        raise ValueError(f"{name} must be a non-empty list")
    names = []
    for index, point in enumerate(points):
        item = _mapping(point, f"{name}[{index}]")
        _unknown(item, {"name", "coordinates"}, f"{name}[{index}]")
        _required(item, {"name", "coordinates"}, f"{name}[{index}]")
        _point_coordinates(item["coordinates"], f"{name}[{index}].coordinates")
        names.append(str(item["name"]))
    if any(not item for item in names) or len(set(names)) != len(names):
        raise ValueError(f"{name} names must be non-empty and unique")


class EikonalSources:
    """Factories for ``fs-eikonal-1`` source selections."""

    @staticmethod
    def acquisition(
        names: Optional[Iterable[str]] = None,
        *,
        incidence_slot: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Select all or named physical acquisition sources."""

        result: Dict[str, Any] = {"kind": "acquisition"}
        if names is not None:
            result["names"] = [str(name) for name in names]
        if incidence_slot is not None:
            result["incidence_slot"] = int(incidence_slot)
        return result

    @staticmethod
    def explicit(
        points: Iterable[Mapping[str, Any]],
        *,
        incidence_slot: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Use explicit named source coordinates."""

        result: Dict[str, Any] = {
            "kind": "explicit",
            "points": [copy.deepcopy(dict(point)) for point in points],
        }
        if incidence_slot is not None:
            result["incidence_slot"] = int(incidence_slot)
        return result


class EikonalReceivers:
    """Factories for optional ``fs-eikonal-1`` receiver selections."""

    @staticmethod
    def disabled() -> Dict[str, Any]:
        """Retain only vertex fields and no receiver products."""

        return {"enabled": False}

    @staticmethod
    def acquisition(groups: Iterable[str]) -> Dict[str, Any]:
        """Select receiver groups from the simulation acquisition."""

        return {
            "enabled": True,
            "kind": "acquisition",
            "groups": [str(group) for group in groups],
        }

    @staticmethod
    def explicit(points: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """Use explicit named receiver coordinates."""

        return {
            "enabled": True,
            "kind": "explicit",
            "points": [copy.deepcopy(dict(point)) for point in points],
        }


class EikonalConfig:
    """Validated authoring object for Sauce's ``fs-eikonal-1`` contract."""

    schema = "fs-eikonal-1"

    def __init__(
        self,
        *,
        sources: Optional[Mapping[str, Any]] = None,
        receivers: Optional[Mapping[str, Any]] = None,
        solver: Optional[Mapping[str, Any]] = None,
        products: Optional[Mapping[str, Any]] = None,
        output: Optional[Mapping[str, Any]] = None,
    ):
        product_data = {
            "field": True,
            "characteristics": False,
            **_mapping(products or {}, "products"),
        }
        if product_data.get("characteristics"):
            product_data.setdefault("max_points_per_characteristic", 10000)
            product_data.setdefault("max_characteristic_points", 1000000)
        payload: Dict[str, Any] = {
            "schema": self.schema,
            "physics": "acoustic",
            "sources": _mapping(
                sources if sources is not None else EikonalSources.acquisition(),
                "sources",
            ),
            "receivers": _mapping(
                receivers if receivers is not None else EikonalReceivers.disabled(),
                "receivers",
            ),
            "solver": _mapping(solver or {}, "solver"),
            "products": product_data,
            "output": {
                "directory": "eikonal",
                "hdf5_file": "first_arrivals.h5",
                "overwrite": False,
                **_mapping(output or {}, "output"),
            },
        }
        self._validate(payload)
        self._data = payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "EikonalConfig":
        """Load an exact solver-style Eikonal configuration."""

        payload = _mapping(data, "Eikonal")
        cls._validate(payload)
        result = cls.__new__(cls)
        result._data = payload
        return result

    def to_fs(self) -> Dict[str, Any]:
        """Return an isolated JSON-compatible solver payload."""

        payload = copy.deepcopy(self._data)
        for key in ("directory", "hdf5_file"):
            payload["output"][key] = _output_path(
                payload["output"][key], f"output.{key}"
            )
        for section in ("sources", "receivers"):
            for index, point in enumerate(payload[section].get("points", [])):
                point["coordinates"] = _point_coordinates(
                    point["coordinates"], f"{section}.points[{index}].coordinates"
                )
        return payload

    def with_updates(self, **sections: Mapping[str, Any]) -> "EikonalConfig":
        """Return a copy with complete top-level sections replaced."""

        payload = self.to_fs()
        for name, value in sections.items():
            if name not in {"sources", "receivers", "solver", "products", "output"}:
                raise ValueError(f"Unknown Eikonal section: {name}")
            payload[name] = _mapping(value, name)
        return self.from_fs(payload)

    @classmethod
    def _validate(cls, payload: Mapping[str, Any]) -> None:
        top = {
            "schema",
            "physics",
            "sources",
            "receivers",
            "solver",
            "products",
            "output",
        }
        _unknown(payload, top, "Eikonal")
        _required(payload, top, "Eikonal")
        if payload["schema"] != cls.schema:
            raise ValueError(f"Eikonal.schema must be {cls.schema!r}")
        if payload["physics"] != "acoustic":
            raise ValueError("fs-eikonal-1 supports acoustic physics only")
        cls._validate_sources(_mapping(payload["sources"], "sources"))
        receivers = _mapping(payload["receivers"], "receivers")
        cls._validate_receivers(receivers)
        cls._validate_solver(_mapping(payload["solver"], "solver"))
        cls._validate_products(
            _mapping(payload["products"], "products"),
            receivers_enabled=receivers["enabled"],
        )
        cls._validate_output(_mapping(payload["output"], "output"))

    @staticmethod
    def _validate_sources(data: Mapping[str, Any]) -> None:
        _unknown(data, {"kind", "names", "points", "incidence_slot"}, "sources")
        _required(data, {"kind"}, "sources")
        kind = _choice(data["kind"], ("acquisition", "explicit"), "sources.kind")
        if "incidence_slot" in data:
            _integer(data["incidence_slot"], "sources.incidence_slot", minimum=1)
        if kind == "acquisition":
            if "points" in data:
                raise ValueError("acquisition sources cannot define points")
            if "names" in data:
                _string_list(data["names"], "sources.names")
        else:
            _required(data, {"points"}, "explicit sources")
            if "names" in data:
                raise ValueError("explicit sources cannot define names")
            _named_points(data["points"], "sources.points")

    @staticmethod
    def _validate_receivers(data: Mapping[str, Any]) -> None:
        _unknown(data, {"enabled", "kind", "groups", "points"}, "receivers")
        _required(data, {"enabled"}, "receivers")
        if not isinstance(data["enabled"], bool):
            raise ValueError("receivers.enabled must be boolean")
        if not data["enabled"]:
            if set(data) != {"enabled"}:
                raise ValueError(
                    "disabled receivers cannot define kind, groups, or points"
                )
            return
        _required(data, {"kind"}, "enabled receivers")
        kind = _choice(data["kind"], ("acquisition", "explicit"), "receivers.kind")
        if kind == "acquisition":
            _required(data, {"groups"}, "acquisition receivers")
            if "points" in data:
                raise ValueError("acquisition receivers cannot define points")
            _string_list(data["groups"], "receivers.groups")
        else:
            _required(data, {"points"}, "explicit receivers")
            if "groups" in data:
                raise ValueError("explicit receivers cannot define groups")
            _named_points(data["points"], "receivers.points")

    @staticmethod
    def _validate_solver(data: Mapping[str, Any]) -> None:
        allowed = {
            "abs_tolerance",
            "rel_tolerance",
            "tie_tolerance",
            "max_waves",
            "max_updates",
            "minimum_full_stencil_quality",
        }
        _unknown(data, allowed, "solver")
        for field in ("abs_tolerance", "rel_tolerance", "tie_tolerance"):
            if field in data:
                _positive(data[field], f"solver.{field}")
        for field in ("max_waves", "max_updates"):
            if field in data:
                _integer(data[field], f"solver.{field}", minimum=1)
        if "minimum_full_stencil_quality" in data:
            if isinstance(data["minimum_full_stencil_quality"], bool) or not isinstance(
                data["minimum_full_stencil_quality"], (int, float)
            ):
                raise ValueError(
                    "solver.minimum_full_stencil_quality must be between 0 and 1"
                )
            try:
                quality = float(data["minimum_full_stencil_quality"])
            except (TypeError, ValueError):
                quality = math.nan
            if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
                raise ValueError(
                    "solver.minimum_full_stencil_quality must be between 0 and 1"
                )

    @staticmethod
    def _validate_products(data: Mapping[str, Any], *, receivers_enabled: bool) -> None:
        allowed = {
            "field",
            "characteristics",
            "max_points_per_characteristic",
            "max_characteristic_points",
        }
        _unknown(data, allowed, "products")
        _required(data, {"field", "characteristics"}, "products")
        for field in ("field", "characteristics"):
            if not isinstance(data[field], bool):
                raise ValueError(f"products.{field} must be boolean")
        if not receivers_enabled and (not data["field"] or data["characteristics"]):
            raise ValueError(
                "disabled receivers require field=True and characteristics=False"
            )
        if data["characteristics"]:
            _required(
                data,
                {"max_points_per_characteristic"},
                "characteristic products",
            )
        for field in (
            "max_points_per_characteristic",
            "max_characteristic_points",
        ):
            if field in data:
                _integer(data[field], f"products.{field}", minimum=2)

    @staticmethod
    def _validate_output(data: Mapping[str, Any]) -> None:
        _unknown(data, {"directory", "hdf5_file", "overwrite"}, "output")
        _required(data, {"directory", "hdf5_file"}, "output")
        _output_path(data["directory"], "output.directory")
        _output_path(data["hdf5_file"], "output.hdf5_file")
        if "overwrite" in data and not isinstance(data["overwrite"], bool):
            raise ValueError("output.overwrite must be boolean")


@register_class
class EikonalJob(BaseJob):
    """Single-task isotropic acoustic first-arrival workflow."""

    supports_trace_packing = False
    frequency_independent = True
    max_ranks_per_task = 1

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        config: Optional[Union[EikonalConfig, Mapping[str, Any]]] = None,
        *,
        sources: Optional[Mapping[str, Any]] = None,
        receivers: Optional[Mapping[str, Any]] = None,
        **config_sections: Mapping[str, Any],
    ):
        if config is not None and (
            sources is not None or receivers is not None or config_sections
        ):
            raise ValueError("Pass either config or section arguments, not both")
        if config is None:
            config = EikonalConfig(
                sources=sources,
                receivers=receivers,
                **config_sections,
            )
        elif isinstance(config, Mapping):
            config = EikonalConfig.from_fs(config)
        if not isinstance(config, EikonalConfig):
            raise TypeError("config must be an EikonalConfig or mapping")
        super().__init__(name, simulation, "eikonal", [0.0], JobOutputs())
        self.eikonal = config
        self._validate_simulation_support()

    def _validate_simulation_support(self) -> None:
        physics = str(getattr(self.simulation, "physics", "")).lower()
        if physics != "acoustic":
            raise ValueError("fs-eikonal-1 supports acoustic simulations only")
        dimension = canonical_dimension(getattr(self.simulation, "dimension", 0))
        if dimension not in {2, 3}:
            raise ValueError("Eikonal supports only full-dimensional 2D and 3D")
        config = self.eikonal.to_fs()
        self._validate_acquisition_selectors(config)
        for section in ("sources", "receivers"):
            for index, point in enumerate(config[section].get("points", [])):
                _point_coordinates(
                    point["coordinates"],
                    f"{section}.points[{index}].coordinates",
                    int(dimension),
                )
        if bool(getattr(self.simulation, "axisymmetric", False)):
            raise ValueError("Eikonal does not support axisymmetric simulations")

    def _validate_acquisition_selectors(self, config: Mapping[str, Any]) -> None:
        """Check selector names when a local acquisition catalog is available."""

        acquisition = getattr(self.simulation, "acquisition", None)
        if acquisition is None:
            return
        sources = config["sources"]
        if sources["kind"] == "acquisition" and "names" in sources:
            known = set(acquisition.source_point_names())
            if known:
                missing = sorted(set(sources["names"]) - known)
                if missing:
                    raise ValueError(
                        f"Unknown Eikonal acquisition source names: {missing}"
                    )
        receivers = config["receivers"]
        if receivers.get("kind") == "acquisition":
            known = {group.name for group in acquisition.receiver_groups}
            if known:
                missing = sorted(set(receivers["groups"]) - known)
                if missing:
                    raise ValueError(
                        f"Unknown Eikonal acquisition receiver groups: {missing}"
                    )

    def validate_outputs(self) -> None:
        """Validate the Eikonal contract instead of frequency outputs."""

        self.eikonal._validate(self.eikonal.to_fs())
        self._validate_simulation_support()

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        """Serialize a Sauce ``eikonal`` job without ``f_list`` or ``Outputs``."""

        self._validate_simulation_support()
        payload = super().to_fs(ctx, project_relative=project_relative)
        payload.pop("f_list", None)
        payload.pop("Outputs", None)
        payload["Eikonal"] = self.eikonal.to_fs()
        return payload

    @classmethod
    def from_fs(
        cls,
        data: Mapping[str, Any],
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "EikonalJob":
        """Deserialize a saved Eikonal job."""

        simulation = BaseJob._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=project_path or data.get("project_path"),
            source_project=data.get("project_path"),
        )
        job = cls(
            name=str(data["name"]),
            simulation=simulation,
            config=EikonalConfig.from_fs(data["Eikonal"]),
        )
        job._job_id = data.get("job_id")
        return job

    @property
    def output_directory(self) -> Path:
        """Directory containing the Eikonal manifest and HDF5 product."""

        return self._result_path / self.eikonal.to_fs()["output"]["directory"]

    @property
    def eikonal_manifest_file(self) -> Path:
        """Path to the ``fs-eikonal-output-1`` manifest."""

        return self.output_directory / "manifest.json"

    @property
    def eikonal_hdf5_file(self) -> Path:
        """Path to the authoritative Eikonal HDF5 product."""

        return self.output_directory / self.eikonal.to_fs()["output"]["hdf5_file"]

    @property
    def result_manifest_file(self) -> Path:
        """Generic frequency-independent product manifest path."""

        return self.eikonal_manifest_file

    @property
    def result_hdf5_file(self) -> Path:
        """Generic frequency-independent authoritative HDF5 path."""

        return self.eikonal_hdf5_file

    @property
    def results(self) -> EikonalResults:
        """Open the typed ``fs-eikonal-output-1`` result handle."""

        return EikonalResults.from_job(self)

    def plot(self, **kwargs: Any) -> Any:
        """Plot the retained first-arrival field and characteristics."""

        return self.results.plot(**kwargs)

    def results_exist(self) -> bool:
        """Return whether a complete, readable Eikonal product exists."""

        if (
            not self.eikonal_manifest_file.is_file()
            or not self.eikonal_hdf5_file.is_file()
        ):
            return False
        try:
            results = self.results
            return (
                results.status == "complete"
                and results.hdf5_file == self.eikonal_hdf5_file.resolve()
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return False

    def _remove_result_products(self) -> bool:
        removed = False
        for path in (self.eikonal_manifest_file, self.eikonal_hdf5_file):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                pass
        return removed

    def is_task_current(
        self,
        task: int,
        *,
        state: Optional[Mapping[str, Any]] = None,
        manifest: Optional[Mapping[str, Any]] = None,
    ) -> bool:
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

    def current_tasks(self) -> list[int]:
        """Return ``[1]`` when the Eikonal product matches this job."""

        return [1] if self.is_task_current(1) else []

    def is_run_current(self) -> bool:
        """Return whether the local Eikonal product and run state are current."""

        return self.is_task_current(1)

    def plan_tasks(
        self,
        *,
        skip_policy: Any = None,
        reuse: bool = False,
        force: bool = False,
        apply: bool = False,
        residual: Optional[float] = None,
        ignore_solver_options: Optional[bool] = None,
    ) -> TaskRunPlan:
        """Plan the single Eikonal task without trace-shard reuse."""

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
                    "path": str(self.eikonal_hdf5_file),
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

    def task_run_plan(self, **kwargs: Any) -> TaskRunPlan:
        """Apply and return the single-task execution plan."""

        return self.plan_tasks(apply=True, **kwargs)

    def write_run_state(self, status: str = "completed", **extra: Any) -> Path:
        """Write run state keyed to the Eikonal HDF5 product."""

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
        row: Dict[str, Any] = {
            "task": 1,
            "frequency": None,
            "status": task_status,
            "complete": task_status == "succeeded" and product_exists,
            "fingerprint": self.task_fingerprint(1),
            "compatibility_fingerprint": self.task_policy_fingerprint(
                1, SkipPolicy.compatible()
            ),
            "path": str(self.eikonal_hdf5_file),
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
                "eikonal": [
                    {
                        "path": str(self.eikonal_manifest_file),
                        "exists": self.eikonal_manifest_file.is_file(),
                    },
                    {
                        "path": str(self.eikonal_hdf5_file),
                        "exists": self.eikonal_hdf5_file.is_file(),
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

    def frequency_status(self) -> Any:
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
                "trace_file": self.eikonal_hdf5_file,
                "result_file": self.eikonal_hdf5_file,
                "trace_exists": self.eikonal_hdf5_file.is_file(),
                "current": current,
                "duration_seconds": metadata.get("duration_seconds"),
                "metadata": metadata,
            }
        ]
