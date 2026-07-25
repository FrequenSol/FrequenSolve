"""Safe, package-native core services for the FrequenSolve MCP server.

This module deliberately has no dependency on an MCP SDK.  It turns the
packaged simulation-knowledge catalog into bounded JSON payloads and builds the
one supported starter scenario with the public FrequenSolve authoring classes.
It never saves, submits, runs, or calls a site.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Union, cast

from frequensolve._version import get_versions
from frequensolve.knowledge.catalog import (
    SimulationKnowledgeCatalog,
    load_simulation_knowledge,
)
from frequensolve.mcp_server.rendering import render_starter_python_source
from frequensolve.mesh.boundary_conditions import BoundaryCondition
from frequensolve.model.layered.model import LayeredModel
from frequensolve.project.project import Project
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverNode
from frequensolve.simulation.discretization import Discretization
from frequensolve.simulation.jobs.forward import FrequencyDomainJob
from frequensolve.simulation.outputs import VtkOutput
from frequensolve.simulation.solver import SolverConfig
from frequensolve.validation.api import validate_job

MCP_CONTRACT = "frequensolve-simulation-assistant/v1"
DRAFT_CONTRACT = "frequensolve-simulation-draft/v1"
STARTER_SCENARIO_ID = "known-small-2d-acoustic"
STARTER_FREQUENCY_HZ = 10.0

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RESOURCE_NAMES = {
    "identity",
    "contracts",
    "catalog",
    "public-api",
    "physics",
    "authoring-rules",
    "validation-codes",
    "examples",
    "glossary",
}
_MAX_DRAFT_NODES = 4096
_MAX_DRAFT_DEPTH = 16
_MAX_DRAFT_STRING = 1024
_MAX_QUERY_LENGTH = 128

JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]

__all__ = [
    "DRAFT_CONTRACT",
    "MCP_CONTRACT",
    "STARTER_SCENARIO_ID",
    "CoreInputError",
    "build_simulation_draft",
    "create_simulation_draft",
    "explain_validation",
    "find_vetted_example",
    "identity_payload",
    "preview_simulation",
    "render_starter_python",
    "resource_payload",
    "validate_simulation_draft",
]


class CoreInputError(ValueError):
    """A bounded public error containing only a stable safe code."""

    __slots__ = ("code",)

    def __init__(self, code: str):
        if not _SAFE_CODE_RE.fullmatch(code):
            code = "core_input_error"
        self.code = code
        super().__init__(code)


def identity_payload() -> dict[str, JsonValue]:
    """Return exact installed package, catalog, solver, and contract identities."""

    catalog = load_simulation_knowledge()
    identities = catalog.identities
    versioneer = get_versions()
    full_revision = versioneer.get("full-revisionid")
    if not isinstance(full_revision, str) or not full_revision:
        full_revision = None
    return {
        "mcp_contract": MCP_CONTRACT,
        "draft_contract": DRAFT_CONTRACT,
        "package": {
            "name": "frequensolve",
            "version": identities.package_version,
            "declared_release": identities.declared_package_release,
            "full_revisionid": full_revision,
            "dirty": bool(versioneer.get("dirty", False)),
        },
        "catalog": {
            "schema": identities.catalog_schema,
            "version": identities.catalog_version,
            "authoring_rules_schema": identities.authoring_rules_schema,
        },
        "solver": {
            "compatibility_schema": identities.compatibility_schema,
            "preferred_release": identities.preferred_frequensolver_release,
            "preferred_commit": identities.preferred_frequensolver_commit,
            "validation_profile": identities.solver_validation_profile,
        },
        "contracts": [
            {
                "name": contract.name,
                "identity": contract.identity,
                "owner": contract.owner,
                "source_revision": contract.source_revision,
            }
            for contract in identities.contracts
        ],
    }


def create_simulation_draft(
    *,
    project_name: str = "project",
    simulation_name: str = "known_small_2d_acoustic",
    frequency_hz: Union[int, float] = 10.0,
    receiver_count: int = 101,
) -> dict[str, JsonValue]:
    """Create the constrained known-small 2D acoustic draft.

    All scientific defaults come from the packaged starter catalog.  The first
    beta keeps its evaluated 10 Hz frequency fixed; callers may change only
    safe names and receiver count.
    """

    project_slug = _safe_slug(project_name, "invalid_project_name")
    simulation_slug = _safe_slug(simulation_name, "invalid_simulation_name")
    frequency = _starter_frequency(frequency_hz)
    count = _receiver_count(receiver_count)

    scenario = load_simulation_knowledge().get_starter_scenario(STARTER_SCENARIO_ID)
    if scenario.physics != "acoustic" or scenario.dimension != 2:
        raise RuntimeError("validated starter scenario identity changed")
    return {
        "schema": DRAFT_CONTRACT,
        "scenario_id": STARTER_SCENARIO_ID,
        "project_name": project_slug,
        "simulation_name": simulation_slug,
        "job_name": _job_name(frequency),
        "physics": scenario.physics,
        "dimension": scenario.dimension,
        "frequency_hz": frequency,
        "receiver_count": count,
    }


def find_vetted_example(query: str) -> dict[str, JsonValue]:
    """Return the closest deterministic vetted-example match."""

    if not isinstance(query, str):
        raise CoreInputError("invalid_example_query")
    normalized = query.strip().casefold()
    if (
        not normalized
        or len(normalized) > _MAX_QUERY_LENGTH
        or any(ord(character) < 32 for character in normalized)
    ):
        raise CoreInputError("invalid_example_query")

    catalog = load_simulation_knowledge()
    query_tokens = set(_QUERY_TOKEN_RE.findall(normalized))
    ranked: list[tuple[int, int, str, Any]] = []
    for example in catalog.examples:
        exact_values = {
            example.id.casefold(),
            example.title.casefold(),
            (example.scenario_id or "").casefold(),
        }
        exact = int(normalized in exact_values)
        searchable = " ".join(
            (
                example.id,
                example.title,
                example.summary,
                example.scenario_id or "",
            )
        ).casefold()
        tokens = set(_QUERY_TOKEN_RE.findall(searchable))
        overlap = len(query_tokens.intersection(tokens))
        ranked.append((exact, overlap, example.id, example))
    _exact, overlap, _example_id, example = max(
        ranked,
        key=lambda item: (item[0], item[1], -len(item[2]), item[2]),
    )
    return {
        "query": normalized,
        "match": {
            "id": example.id,
            "title": example.title,
            "summary": example.summary,
            "source_path": example.source_path,
            "tested_by": list(example.tested_by),
            "scenario_id": example.scenario_id,
        },
        "match_basis": "exact" if _exact else ("terms" if overlap else "fallback"),
    }


def build_simulation_draft(
    draft: Mapping[str, object],
    project_path: Union[str, Path],
) -> dict[str, JsonValue]:
    """Build the draft with real package objects without saving or running it."""

    normalized = _normalize_draft(draft)
    root = _safe_project_path(project_path)
    try:
        _project, job = _build_job(normalized, root)
    except CoreInputError:
        raise
    except Exception:
        raise CoreInputError("draft_build_failed") from None

    receiver_count = job.simulation.acquisition.receiver_groups[0].size
    return {
        "draft_contract": DRAFT_CONTRACT,
        "scenario_id": STARTER_SCENARIO_ID,
        "project_name": normalized["project_name"],
        "simulation_name": job.simulation.name,
        "job_name": job.name,
        "physics": job.simulation.physics,
        "dimension": job.simulation.dimension,
        "frequency_count": len(job.f_list),
        "receiver_count": receiver_count,
        "output_kinds": _output_kinds(job),
    }


def validate_simulation_draft(
    draft: Mapping[str, object],
    project_path: Union[str, Path],
) -> dict[str, JsonValue]:
    """Run the real package validator and return allowlisted diagnostics."""

    normalized = _normalize_draft(draft)
    root = _safe_project_path(project_path)
    try:
        _project, job = _build_job(normalized, root)
        report = validate_job(job)
    except CoreInputError:
        raise
    except Exception:
        raise CoreInputError("draft_validation_failed") from None

    catalog = load_simulation_knowledge()
    diagnostics = [_diagnostic_payload(catalog, issue.code) for issue in report.issues]
    return {
        "draft_contract": DRAFT_CONTRACT,
        "valid": report.ok,
        "error_count": len(report.errors),
        "warning_count": len(report.warnings),
        "diagnostics": diagnostics,
    }


def preview_simulation(
    draft: Mapping[str, object],
) -> dict[str, JsonValue]:
    """Return exact setup counts and explicitly bounded assumptions."""

    normalized = _normalize_draft(draft)
    frequencies: list[JsonValue] = [float(normalized["frequency_hz"])]
    output_kinds: list[JsonValue] = ["receiver-traces", "vtk-domain"]
    assumptions: list[JsonValue] = [
        "One solver task is created for each modeled frequency.",
        "The starter has one inline scalar source and no source encoding.",
        "Receiver traces use the package default and the draft adds one domain VTK output.",
        "No runtime, cost, convergence, or production-sizing estimate is made.",
        "This preview does not save, submit, or run the simulation.",
    ]
    return {
        "draft_contract": DRAFT_CONTRACT,
        "scenario_id": STARTER_SCENARIO_ID,
        "frequencies_hz": frequencies,
        "task_count": len(frequencies),
        "source_field_count": 1,
        "receiver_count": int(normalized["receiver_count"]),
        "output_kinds": output_kinds,
        "assumptions": assumptions,
    }


def render_starter_python(draft: Mapping[str, object]) -> str:
    """Render deterministic package-native Python without executing it."""

    normalized = _normalize_draft(draft)
    return render_starter_python_source(_setup_from_draft(normalized))


def explain_validation(code: str) -> dict[str, JsonValue]:
    """Return the cataloged plain-language explanation for one stable code."""

    if (
        not isinstance(code, str)
        or len(code) > _MAX_QUERY_LENGTH
        or not _SAFE_CODE_RE.fullmatch(code)
    ):
        raise CoreInputError("invalid_validation_code")
    try:
        entry = load_simulation_knowledge().explain_validation(code)
    except KeyError:
        raise CoreInputError("validation_code_not_found") from None
    return {
        "code": entry.code,
        "severity": entry.severity,
        "path": entry.path,
        "explanation": entry.explanation,
        "remediation": entry.remediation,
    }


def resource_payload(name: str) -> dict[str, JsonValue]:
    """Return one fixed MCP resource payload backed by the packaged catalog."""

    if not isinstance(name, str):
        raise CoreInputError("resource_not_found")
    normalized = name.strip().casefold()
    if normalized not in _RESOURCE_NAMES:
        raise CoreInputError("resource_not_found")
    catalog = load_simulation_knowledge()
    if normalized == "identity":
        return identity_payload()
    if normalized == "contracts":
        identity = identity_payload()
        return {
            "mcp_contract": MCP_CONTRACT,
            "draft_contract": DRAFT_CONTRACT,
            "draft_constraints": {
                "scenario_id": STARTER_SCENARIO_ID,
                "project_name": {
                    "format": "safe-slug",
                    "minimum_length": 1,
                    "maximum_length": 64,
                },
                "simulation_name": {
                    "format": "safe-slug",
                    "minimum_length": 1,
                    "maximum_length": 64,
                },
                "job_name": "derived-from-frequency",
                "physics": "acoustic",
                "dimension": 2,
                "frequency_hz": {
                    "constant": STARTER_FREQUENCY_HZ,
                    "finite": True,
                },
                "receiver_count": {"minimum": 1, "maximum": 1001},
            },
            "solver_contracts": identity["contracts"],
        }
    if normalized == "catalog":
        return _catalog_payload(catalog)

    entries: JsonValue
    if normalized == "public-api":
        entries = _trusted_json_copy([asdict(entry) for entry in catalog.public_api])
    elif normalized == "physics":
        entries = _trusted_json_copy(
            [asdict(entry) for entry in catalog.physics_entries]
        )
    elif normalized == "authoring-rules":
        entries = _trusted_json_copy(asdict(catalog.authoring_rules))
    elif normalized == "validation-codes":
        entries = _trusted_json_copy(
            [asdict(entry) for entry in catalog.validation_codes]
        )
    elif normalized == "examples":
        entries = {
            "vetted": _trusted_json_copy([asdict(entry) for entry in catalog.examples]),
            "starter_scenarios": _trusted_json_copy(
                [asdict(entry) for entry in catalog.starter_scenarios]
            ),
        }
    else:
        entries = _trusted_json_copy([asdict(entry) for entry in catalog.glossary])

    return {
        "catalog_schema": catalog.identities.catalog_schema,
        "catalog_version": catalog.identities.catalog_version,
        "resource": normalized,
        "entries": entries,
    }


def _catalog_payload(
    catalog: SimulationKnowledgeCatalog,
) -> dict[str, JsonValue]:
    return {
        "identities": identity_payload(),
        "authoring_rules": _trusted_json_copy(asdict(catalog.authoring_rules)),
        "public_api": _trusted_json_copy(
            [asdict(entry) for entry in catalog.public_api]
        ),
        "glossary": _trusted_json_copy([asdict(entry) for entry in catalog.glossary]),
        "physics": _trusted_json_copy(
            [asdict(entry) for entry in catalog.physics_entries]
        ),
        "validation_codes": _trusted_json_copy(
            [asdict(entry) for entry in catalog.validation_codes]
        ),
        "examples": _trusted_json_copy([asdict(entry) for entry in catalog.examples]),
        "starter_scenarios": _trusted_json_copy(
            [asdict(entry) for entry in catalog.starter_scenarios]
        ),
        "limitations": list(catalog.limitations),
    }


def _diagnostic_payload(
    catalog: SimulationKnowledgeCatalog,
    code: str,
) -> dict[str, JsonValue]:
    try:
        entry = catalog.explain_validation(code)
    except KeyError:  # pragma: no cover - package registry and catalog are synced
        return {
            "code": "validation.unknown",
            "severity": "error",
            "path": "",
            "explanation": "Package validation returned an unavailable diagnostic.",
            "remediation": "Use the installed FrequenSolve validation catalog.",
        }
    return {
        "code": entry.code,
        "severity": entry.severity,
        "path": entry.path,
        "explanation": entry.explanation,
        "remediation": entry.remediation,
    }


def _setup_from_draft(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the flat bounded draft from the packaged starter catalog."""

    scenario = load_simulation_knowledge().get_starter_scenario(STARTER_SCENARIO_ID)
    setup = _trusted_json_copy(scenario.setup)
    if not isinstance(
        setup, dict
    ):  # pragma: no cover - catalog validation guarantees it
        raise RuntimeError("validated starter setup was not an object")
    project = _required_object(setup, "project")
    simulation = _required_object(setup, "simulation")
    acquisition = _required_object(setup, "acquisition")
    receiver_group = _required_object(acquisition, "receiver_group")
    coordinate_line = _required_object(receiver_group, "coordinate_line")
    job = _required_object(setup, "job")
    project["name"] = draft["project_name"]
    simulation["name"] = draft["simulation_name"]
    coordinate_line["count"] = draft["receiver_count"]
    job["name"] = draft["job_name"]
    job["f_list"] = [draft["frequency_hz"]]
    return setup


def _build_job(
    draft: dict[str, Any],
    project_path: Path,
) -> tuple[Project, FrequencyDomainJob]:
    setup = _setup_from_draft(draft)
    project_config = dict(setup["project"])
    project_config.update(
        {
            "log_file": None,
            "log_to_console": False,
            "jupyter_logging": False,
        }
    )
    project = Project(path=project_path, **project_config)

    simulation = project.new_simulation(**dict(setup["simulation"]))

    model_config = dict(setup["model"])
    surfaces = model_config.pop("surfaces")
    layers = model_config.pop("layers")
    if model_config.pop("type") != "LayeredModel":
        raise CoreInputError("unsupported_draft")
    model = LayeredModel(**model_config)
    for index, surface in enumerate(surfaces):
        model.add_surface(**surface)
        if index < len(layers):
            model.add_layer(**layers[index])
    simulation += model

    mesh_config = dict(setup["mesh"])
    if mesh_config.pop("type") != "HexMeshGenerator":
        raise CoreInputError("unsupported_draft")
    adapt = mesh_config.pop("adapt")
    source_grading = mesh_config.pop("source_grading")
    simulation += model.hex_mesh_generator(**mesh_config)
    simulation.mesh.set_adapt(**adapt)
    simulation.mesh.set_source_grading(**source_grading)

    for boundary in setup["boundary_conditions"]:
        simulation += BoundaryCondition(**boundary)

    acquisition_config = setup["acquisition"]
    acquisition = Acquisition()
    acquisition.add_sources(**acquisition_config["source"])
    receiver_config = acquisition_config["receiver_group"]
    receiver = ReceiverNode(name=receiver_config["device_name"])
    receiver.add_component(**receiver_config["component"])
    line = receiver_config["coordinate_line"]
    if line["axis"] != "x":
        raise CoreInputError("unsupported_draft")
    count = int(line["count"])
    start = float(line["start"])
    stop = float(line["stop"])
    if count == 1:
        x_coordinates = [start]
    else:
        x_coordinates = [
            start + (stop - start) * index / (count - 1) for index in range(count)
        ]
    coordinates = [[x, line["fixed"]["z"]] for x in x_coordinates]
    acquisition.add_receiver_group(
        name=receiver_config["name"],
        device=receiver,
        coords=coordinates,
    )
    simulation += acquisition

    simulation += Discretization(**setup["discretization"])
    simulation += SolverConfig(**setup["solver"])

    job_config = dict(setup["job"])
    if job_config.pop("type") != "FrequencyDomainJob":
        raise CoreInputError("unsupported_draft")
    outputs = job_config.pop("outputs")
    vtk_outputs = [VtkOutput.domain(**vtk_config) for vtk_config in outputs["vtk"]]
    return project, FrequencyDomainJob(
        simulation=simulation,
        outputs=vtk_outputs,
        **job_config,
    )


def _output_kinds(job: FrequencyDomainJob) -> list[JsonValue]:
    kinds: list[JsonValue] = ["receiver-traces"]
    if job.outputs.vtk:
        kinds.append("vtk-domain")
    if job.outputs.wavefields:
        kinds.append("wavefield")
    return kinds


def _safe_slug(value: object, code: str) -> str:
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise CoreInputError(code)
    return value


def _starter_frequency(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoreInputError("invalid_frequency")
    frequency = float(value)
    if not math.isfinite(frequency) or frequency != STARTER_FREQUENCY_HZ:
        raise CoreInputError("invalid_frequency")
    return frequency


def _receiver_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1001:
        raise CoreInputError("invalid_receiver_count")
    return value


def _job_name(frequency: float) -> str:
    token = format(frequency, ".17g")
    token = token.replace(".", "_").replace("+", "p").replace("-", "m")
    return f"frequency_{token}hz"


def _safe_project_path(value: Union[str, Path]) -> Path:
    if not isinstance(value, (str, Path)):
        raise CoreInputError("invalid_project_path")
    path = Path(value)
    if not path.is_absolute():
        raise CoreInputError("invalid_project_path")
    try:
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise CoreInputError("invalid_project_path")
        else:
            parent = path.parent
            if (
                parent.is_symlink()
                or not parent.exists()
                or not parent.is_dir()
                or (hasattr(os, "geteuid") and parent.stat().st_uid != os.geteuid())
            ):
                raise CoreInputError("invalid_project_path")
            path.mkdir(mode=0o700, parents=False, exist_ok=False)
        return path.resolve(strict=True)
    except (FileExistsError, OSError, RuntimeError):
        raise CoreInputError("invalid_project_path") from None


def _normalize_draft(value: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoreInputError("invalid_draft")
    normalized = _bounded_json_copy(value)
    if not isinstance(normalized, dict):
        raise CoreInputError("invalid_draft")
    try:
        if set(normalized) != {
            "schema",
            "scenario_id",
            "project_name",
            "simulation_name",
            "job_name",
            "physics",
            "dimension",
            "frequency_hz",
            "receiver_count",
        }:
            raise CoreInputError("unsupported_draft")
        if normalized["schema"] != DRAFT_CONTRACT:
            raise CoreInputError("unsupported_draft")
        if normalized["scenario_id"] != STARTER_SCENARIO_ID:
            raise CoreInputError("unsupported_draft")
        expected = create_simulation_draft(
            project_name=_safe_slug(
                normalized.get("project_name"),
                "invalid_project_name",
            ),
            simulation_name=_safe_slug(
                normalized.get("simulation_name"),
                "invalid_simulation_name",
            ),
            frequency_hz=_starter_frequency(normalized.get("frequency_hz")),
            receiver_count=_receiver_count(normalized.get("receiver_count")),
        )
        if normalized != expected:
            raise CoreInputError("unsupported_draft")
    except CoreInputError:
        raise
    except Exception:
        raise CoreInputError("invalid_draft") from None
    return normalized


def _required_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise CoreInputError("invalid_draft")
    return value


def _bounded_json_copy(value: object) -> JsonValue:
    node_count = 0

    def visit(item: object, depth: int) -> JsonValue:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_DRAFT_NODES or depth > _MAX_DRAFT_DEPTH:
            raise CoreInputError("draft_too_large")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CoreInputError("invalid_draft")
            return item
        if isinstance(item, str):
            if len(item) > _MAX_DRAFT_STRING:
                raise CoreInputError("draft_too_large")
            return item
        if isinstance(item, list):
            return [visit(child, depth + 1) for child in item]
        if isinstance(item, dict):
            result: dict[str, JsonValue] = {}
            for key, child in item.items():
                if not isinstance(key, str) or len(key) > _MAX_DRAFT_STRING:
                    raise CoreInputError("invalid_draft")
                result[key] = visit(child, depth + 1)
            return result
        raise CoreInputError("invalid_draft")

    return visit(value, 0)


def _trusted_json_copy(value: object) -> JsonValue:
    """Round-trip trusted package data into canonical JSON container types."""

    return cast(
        JsonValue,
        json.loads(json.dumps(value, allow_nan=False, sort_keys=False)),
    )
