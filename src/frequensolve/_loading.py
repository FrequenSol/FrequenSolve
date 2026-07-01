from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

__all__ = ["load"]


def load(source: Any, *, kind: Optional[str] = None, **kwargs) -> Any:
    """Load a saved FrequenSolve object using its persisted contract.

    Args:
        source: JSON file, directory containing one JSON file, existing job
            object, trace file, or object accepted by an explicit loader.
        kind: Optional loader hint. Supported values are ``"job"``,
            ``"project"``, ``"simulation"``, ``"survey"``, and ``"traces"``.
        **kwargs: Extra keyword arguments forwarded to the selected loader.

    Returns:
        Loaded project, job, simulation, survey, or trace dataset.
    """

    normalized_kind = _normalize_kind(kind)
    if normalized_kind is not None:
        return _load_with_kind(source, normalized_kind, **kwargs)

    if not isinstance(source, (str, Path)) and hasattr(source, "job_file"):
        from frequensolve.simulation.jobs import BaseJob

        return BaseJob.load(source, **kwargs)

    path = _json_file_from_source(source)
    data = _read_json(path)

    if _is_job_payload(data):
        from frequensolve.simulation.jobs import BaseJob

        return BaseJob.load(path, **kwargs)
    if _is_project_payload(data):
        from frequensolve.project import Project

        return Project.load(path)
    if _is_simulation_payload(data):
        from frequensolve.simulation import BaseSimulation

        return BaseSimulation.load(path, **kwargs)

    raise ValueError(f"Could not infer FrequenSolve object type from {path}")


def _normalize_kind(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    text = str(kind).strip().lower().replace("-", "_")
    aliases = {
        "jobs": "job",
        "sim": "simulation",
        "sims": "simulation",
        "simulations": "simulation",
        "projects": "project",
        "survey": "survey",
        "surveys": "survey",
        "trace": "traces",
        "traces": "traces",
        "trace_dataset": "traces",
    }
    return aliases.get(text, text)


def _load_with_kind(source: Any, kind: str, **kwargs) -> Any:
    if kind == "job":
        from frequensolve.simulation.jobs import BaseJob

        return BaseJob.load(source, **kwargs)
    if kind == "project":
        from frequensolve.project import Project

        return Project.load(source)
    if kind == "simulation":
        from frequensolve.simulation import BaseSimulation

        return BaseSimulation.load(source, **kwargs)
    if kind == "survey":
        from frequensolve.seismic import Survey

        return Survey.load(source, **kwargs)
    if kind == "traces":
        from frequensolve.seismic import TraceDataset

        return TraceDataset.open(source, **kwargs)
    raise ValueError(
        "kind must be one of 'job', 'project', 'simulation', 'survey', or 'traces'"
    )


def _json_file_from_source(source: Union[str, Path]) -> Path:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        named = path / f"{path.name}.json"
        if named.exists():
            return named.resolve()
        json_files = sorted(path.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0].resolve()
        if not json_files:
            raise FileNotFoundError(f"No JSON file found in {path}")
        names = ", ".join(file.name for file in json_files)
        raise ValueError(
            f"Multiple JSON files found in {path}; specify one explicitly: {names}"
        )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to load JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _is_job_payload(data: dict[str, Any]) -> bool:
    schema = str(data.get("schema", ""))
    if schema.startswith("fs-job-"):
        return True
    return {"workflow", "f_list", "Outputs", "simulation"}.issubset(data)


def _is_project_payload(data: dict[str, Any]) -> bool:
    return "version" in data and "simulations" in data and "workflow" not in data


def _is_simulation_payload(data: dict[str, Any]) -> bool:
    if "_type" not in data:
        return False
    return "Model" in data or "Mesh" in data or "Acquisition" in data
