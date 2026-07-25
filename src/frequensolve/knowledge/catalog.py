"""Load the packaged simulation-knowledge catalog.

The catalog is deliberately small and deterministic.  It describes the public
FrequenSolve authoring surface that an agent may explain without importing a
solver, calling a cloud API, or copying native FrequenSolver implementation
details.  Runtime version identities are joined with the packaged catalog when
it is loaded so callers always see the exact installed package declaration.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Union

from frequensolve._version import get_versions
from frequensolve.frequensolver import (
    FrequenSolverCompatibilityManifest,
    load_frequensolver_compatibility,
)
from frequensolve.model.property import canonical_property_name
from frequensolve.simulation.physics import components_for_physics
from frequensolve.util.physics import (
    canonical_dimension,
    canonical_physics,
    model_dimension,
    physics_aliases,
    supported_dimensions_for_physics,
    supported_physics,
)

CATALOG_SCHEMA = "frequensolve-simulation-knowledge/v1"
AUTHORING_RULES_SCHEMA = "frequensolve-authoring-rules/v1"
CATALOG_RESOURCE = "simulation_knowledge_v1.json"
CATALOG_SCHEMA_RESOURCE = "simulation_knowledge_schema_v1.json"

_CATALOG_VERSION_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)
_CATALOG_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_PUBLIC_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PUBLIC_IMPORT_PATH_RE = re.compile(r"^frequensolve\.[A-Za-z_][A-Za-z0-9_]*$")
_VALIDATION_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
_PUBLIC_API_KINDS = {"class", "function"}
_PUBLIC_API_CATEGORIES = {
    "acquisition",
    "jobs",
    "knowledge",
    "loading",
    "mesh",
    "model",
    "numerics",
    "outputs",
    "project",
    "simulation",
    "validation",
}
_PINNED_SAUCE_REVISION = "a54bdda81c98780fb4b805b92cf6df6c6e8bd29a"
_EXPECTED_CONTRACT_IDENTITIES = (
    ("simulation", "fs-simulation-1", "Sauce", _PINNED_SAUCE_REVISION),
    ("acquisition", "fs-acquisition-2", "Sauce", _PINNED_SAUCE_REVISION),
    ("job", "fs-job-1", "Sauce", _PINNED_SAUCE_REVISION),
)
_REQUIRED_SETUP_SECTIONS = {
    "project",
    "simulation",
    "model",
    "mesh",
    "boundary_conditions",
    "acquisition",
    "discretization",
    "solver",
    "job",
}
_DOCUMENTED_BOUNDARY_CONDITIONS = (
    "free",
    "pml",
    "fixed",
    "impedance",
    "dirichlet",
    "neumann",
    "sealed",
    "drained",
    "symmetric",
    "axis",
    "symmetric_r",
)
_EXPECTED_MATERIAL_REQUIREMENTS = {
    "acoustic": ("cataloged", "acoustic", ("vp", "rho")),
    "acoustic_axisym": ("cataloged", "acoustic", ("vp", "rho")),
    "elastic": ("cataloged", "elastic:iso", ("vp", "vs", "rho")),
    "elastic_axisym": ("cataloged", "elastic:iso", ("vp", "vs", "rho")),
    "elastic_axisym_torsion": (
        "cataloged",
        "elastic:iso",
        ("vp", "vs", "rho"),
    ),
    "coupled": ("domain-specific", None, ()),
    "coupled_aep": ("domain-specific", None, ()),
    "coupled_axisym": ("domain-specific", None, ()),
    "coupled_axisym_torsion": ("domain-specific", None, ()),
    "poroelastic": (
        "cataloged",
        "poroelastic:iso",
        (
            "vp",
            "vs",
            "rho",
            "k_solid",
            "k_fluid",
            "rho_solid",
            "rho_fluid",
            "porosity",
            "tortuosity",
            "kappa",
            "viscosity",
        ),
    ),
    "em": ("not-cataloged", None, ()),
}

__all__ = [
    "AUTHORING_RULES_SCHEMA",
    "CATALOG_RESOURCE",
    "CATALOG_SCHEMA",
    "CATALOG_SCHEMA_RESOURCE",
    "AcquisitionAuthoringRules",
    "AuthoringRule",
    "AuthoringRules",
    "BoundaryAuthoringRules",
    "CatalogValidationError",
    "ContractIdentity",
    "DimensionAuthoringRules",
    "DiscretizationAuthoringRules",
    "FileReferenceAuthoringRules",
    "FrequencyAuthoringRules",
    "GlossaryEntry",
    "Hdf5DenseEncodingAuthoringRules",
    "ModelSurfaceAuthoringRules",
    "OutputAuthoringRules",
    "PhysicsKnowledge",
    "PmlAuthoringRules",
    "PublicApiKnowledge",
    "SimulationKnowledgeCatalog",
    "SolverAuthoringRules",
    "StarterScenario",
    "ValidationCodeKnowledge",
    "VersionIdentities",
    "VettedExample",
    "load_simulation_knowledge",
]


class CatalogValidationError(ValueError):
    """Raised when simulation-knowledge catalog data is invalid."""


@dataclass(frozen=True)
class ContractIdentity:
    """One public contract referenced by the knowledge catalog."""

    name: str
    identity: str
    owner: str
    source_revision: str


@dataclass(frozen=True)
class VersionIdentities:
    """Installed package, catalog, solver, and contract identities."""

    package_version: str
    declared_package_release: str
    catalog_schema: str
    catalog_version: str
    authoring_rules_schema: str
    compatibility_schema: str
    preferred_frequensolver_release: Optional[str]
    preferred_frequensolver_commit: Optional[str]
    solver_validation_profile: Optional[str]
    contracts: tuple[ContractIdentity, ...]


@dataclass(frozen=True)
class DimensionAuthoringRules:
    """Simulation-dimension rules exposed by the public configuration API."""

    supported: tuple[Union[int, float], ...]
    axisymmetric: tuple[Union[int, float], ...]
    model_dimension_for_2_5d: int


@dataclass(frozen=True)
class PmlAuthoringRules:
    """Public PML condition defaults and serialization behavior."""

    condition: str
    pml_wavelengths: float
    pml_exponent: float
    pml_reflectivity: float
    optional_fields: tuple[str, ...]
    fields_serialize_only_with_condition: bool


@dataclass(frozen=True)
class BoundaryAuthoringRules:
    """Documented boundary names and generated-mesh labels."""

    conditions_required: bool
    documented_conditions: tuple[str, ...]
    solver_specific_conditions_allowed: bool
    generated_2d_labels: tuple[str, ...]
    generated_3d_labels: tuple[str, ...]
    pml: PmlAuthoringRules


@dataclass(frozen=True)
class DiscretizationAuthoringRules:
    """Public discretization defaults and ownership boundaries."""

    default_payload: Mapping[str, object]
    rejected_fields: tuple[str, ...]
    mesh_adapt_order_path: str
    additional_fields_status: str


@dataclass(frozen=True)
class SolverAuthoringRules:
    """Public solver defaults and constrained choices."""

    solve_on_values: tuple[str, ...]
    precision_values: tuple[str, ...]
    default_solve_on: str
    default_max_iter: int
    default_tolerance: float
    default_precision: str
    additional_fields_status: str


@dataclass(frozen=True)
class FrequencyAuthoringRules:
    """Frequency-list and time-domain sweep constraints."""

    explicit_min_items: int
    finite_required: bool
    positive_real_required: bool
    complex_damping_imaginary_sign: str
    time_domain_spacing_fields: tuple[str, ...]
    time_domain_exactly_one_spacing_field: bool
    time_domain_positive_fields: tuple[str, ...]
    time_domain_f_max_greater_than_f_min: bool
    damping_fields: tuple[str, ...]
    damping_fields_mutually_exclusive: bool
    damping_factor_minimum: float


@dataclass(frozen=True)
class Hdf5DenseEncodingAuthoringRules:
    """Public coefficient layout for HDF5Dense source encoding."""

    coefficient_dataset_rank: int
    coefficient_axis_order: tuple[str, ...]
    real_imag_pair_size: int
    real_imag_order: tuple[str, ...]
    storage_kinds: tuple[str, ...]
    native_complex_storage_allowed: bool
    all_dimensions_non_empty: bool


@dataclass(frozen=True)
class AcquisitionAuthoringRules:
    """Current acquisition-v2 authoring rules exposed to agents."""

    hdf5_dense_source_encoding: Hdf5DenseEncodingAuthoringRules


@dataclass(frozen=True)
class FileReferenceAuthoringRules:
    """Local and remote referenced-file validation policy."""

    relative_paths_resolve_from: str
    missing_project_local_severity: str
    existing_external_files_validate_locally: bool
    remote_unverified_policy: str
    explicit_opt_in_parameter: str
    remote_unverified_requires_absolute_path: bool
    remote_unverified_requires_outside_project: bool
    remote_unverified_code: str
    remote_unverified_severity: str
    remote_warning_includes_concrete_path: bool


@dataclass(frozen=True)
class ModelSurfaceAuthoringRules:
    """Solver-visible ParaView model-surface selector rules."""

    named_aliases: tuple[str, ...]
    authored_and_expanded_horizon_names: bool
    indexed_prefix: str
    index_base: int
    case_sensitive: bool
    bottom_alias_allowed: bool
    borehole_surface_names_allowed: bool


@dataclass(frozen=True)
class OutputAuthoringRules:
    """Job-output defaults and validation constraints."""

    traces_enabled_by_default: bool
    vtk_frequency_count: int
    vtk_targets: tuple[str, ...]
    vtk_formats: tuple[str, ...]
    vtk_upscale_targets: tuple[str, ...]
    vtk_upscale_minimum: int
    vtk_upscale_maximum: int
    vtk_grid_allows_upscale: bool
    wavefield_grid_required: bool
    source_ids_one_based: bool
    complex_parts: tuple[str, ...]
    item_kinds: tuple[str, ...]
    model_surfaces: ModelSurfaceAuthoringRules


AuthoringRule = Union[
    DimensionAuthoringRules,
    BoundaryAuthoringRules,
    DiscretizationAuthoringRules,
    SolverAuthoringRules,
    FrequencyAuthoringRules,
    AcquisitionAuthoringRules,
    FileReferenceAuthoringRules,
    OutputAuthoringRules,
]


@dataclass(frozen=True)
class AuthoringRules:
    """Typed, versioned public authoring rules used by agent guidance."""

    schema: str
    dimensions: DimensionAuthoringRules
    boundary_conditions: BoundaryAuthoringRules
    discretization: DiscretizationAuthoringRules
    solver: SolverAuthoringRules
    frequencies: FrequencyAuthoringRules
    acquisition: AcquisitionAuthoringRules
    file_references: FileReferenceAuthoringRules
    outputs: OutputAuthoringRules

    def lookup(self, area: str) -> AuthoringRule:
        """Return one typed authoring-rule area by stable name."""

        rules: Mapping[str, AuthoringRule] = {
            "dimensions": self.dimensions,
            "boundary_conditions": self.boundary_conditions,
            "discretization": self.discretization,
            "solver": self.solver,
            "frequencies": self.frequencies,
            "acquisition": self.acquisition,
            "file_references": self.file_references,
            "outputs": self.outputs,
        }
        try:
            return rules[area]
        except KeyError as exc:
            raise KeyError(f"Authoring rule area {area!r} is not cataloged") from exc


@dataclass(frozen=True)
class PhysicsKnowledge:
    """Searchable public facts for one supported physics formulation."""

    id: str
    aliases: tuple[str, ...]
    summary: str
    supported_dimensions: tuple[Union[int, float], ...]
    property_requirements: str
    material_profile: Optional[str]
    required_properties: tuple[str, ...]
    output_components: tuple[str, ...]
    guided_scenario_id: Optional[str] = None


@dataclass(frozen=True)
class ValidationCodeKnowledge:
    """Stable plain-language explanation for one validation diagnostic."""

    code: str
    severity: str
    path: str
    explanation: str
    remediation: str


@dataclass(frozen=True)
class VettedExample:
    """Public example backed by documentation and deterministic tests."""

    id: str
    title: str
    summary: str
    source_path: str
    tested_by: tuple[str, ...]
    scenario_id: Optional[str] = None


@dataclass(frozen=True)
class StarterScenario:
    """Complete structured setup for one guided starter simulation."""

    id: str
    title: str
    summary: str
    physics: str
    dimension: Union[int, float]
    example_id: str
    setup: Mapping[str, object]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class PublicApiKnowledge:
    """One curated public Python API symbol supported by agent guidance."""

    id: str
    symbol: str
    import_path: str
    aliases: tuple[str, ...]
    kind: str
    category: str
    summary: str
    related_glossary: tuple[str, ...]


@dataclass(frozen=True)
class GlossaryEntry:
    """One public simulation concept with stable agent-facing terminology."""

    id: str
    term: str
    aliases: tuple[str, ...]
    definition: str
    related_api: tuple[str, ...]


@dataclass(frozen=True)
class SimulationKnowledgeCatalog:
    """Validated, version-matched FrequenSolve simulation knowledge."""

    identities: VersionIdentities
    authoring_rules: AuthoringRules
    public_api: tuple[PublicApiKnowledge, ...]
    glossary: tuple[GlossaryEntry, ...]
    physics_entries: tuple[PhysicsKnowledge, ...]
    validation_codes: tuple[ValidationCodeKnowledge, ...]
    examples: tuple[VettedExample, ...]
    starter_scenarios: tuple[StarterScenario, ...]
    limitations: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        package_version: Optional[str] = None,
        compatibility: Optional[FrequenSolverCompatibilityManifest] = None,
    ) -> "SimulationKnowledgeCatalog":
        """Validate catalog data and attach installed release identities.

        Args:
            payload: Parsed catalog JSON.
            package_version: Installed package version override for build and
                test tooling. Normal callers should omit it.
            compatibility: Packaged FrequenSolver compatibility declaration
                override. Normal callers should omit it.

        Returns:
            Validated catalog model.

        Raises:
            CatalogValidationError: If the catalog is structurally invalid or
                disagrees with the current public physics registries.
        """

        data = _mapping(payload, "catalog")
        _keys(
            data,
            required={
                "schema",
                "catalog_version",
                "contracts",
                "authoring_rules",
                "public_api",
                "glossary",
                "physics",
                "validation_codes",
                "examples",
                "starter_scenarios",
                "limitations",
            },
            path="catalog",
        )
        schema = _string(data["schema"], "catalog.schema")
        if schema != CATALOG_SCHEMA:
            raise CatalogValidationError(
                f"catalog.schema must be {CATALOG_SCHEMA!r}, got {schema!r}"
            )
        catalog_version = _string(data["catalog_version"], "catalog.catalog_version")
        if not _CATALOG_VERSION_RE.fullmatch(catalog_version):
            raise CatalogValidationError(
                "catalog.catalog_version must be canonical X.Y.Z"
            )

        contracts = tuple(
            _contract(item, f"catalog.contracts[{index}]")
            for index, item in enumerate(_list(data["contracts"], "catalog.contracts"))
        )
        _unique((item.name for item in contracts), "catalog contract names")
        _unique((item.identity for item in contracts), "catalog contract identities")
        _validate_contracts(contracts)

        authoring_rules = _authoring_rules(
            data["authoring_rules"], "catalog.authoring_rules"
        )

        public_api = tuple(
            _public_api(item, f"catalog.public_api[{index}]")
            for index, item in enumerate(
                _list(data["public_api"], "catalog.public_api")
            )
        )
        _validate_public_api(public_api)

        glossary = tuple(
            _glossary_entry(item, f"catalog.glossary[{index}]")
            for index, item in enumerate(_list(data["glossary"], "catalog.glossary"))
        )
        _validate_glossary(glossary)
        _validate_agent_knowledge_cross_references(public_api, glossary)

        physics_entries = tuple(
            _physics(item, f"catalog.physics[{index}]")
            for index, item in enumerate(_list(data["physics"], "catalog.physics"))
        )
        _validate_physics_entries(physics_entries)
        _validate_authoring_rules(authoring_rules, physics_entries)

        validation_codes = tuple(
            _validation_code(item, f"catalog.validation_codes[{index}]")
            for index, item in enumerate(
                _list(data["validation_codes"], "catalog.validation_codes")
            )
        )
        _unique(
            (item.code for item in validation_codes),
            "catalog validation codes",
        )
        _validate_authoring_diagnostics(authoring_rules, validation_codes)

        examples = tuple(
            _example(item, f"catalog.examples[{index}]")
            for index, item in enumerate(_list(data["examples"], "catalog.examples"))
        )
        _unique((item.id for item in examples), "catalog example ids")

        starter_scenarios = tuple(
            _scenario(item, f"catalog.starter_scenarios[{index}]")
            for index, item in enumerate(
                _list(data["starter_scenarios"], "catalog.starter_scenarios")
            )
        )
        _unique(
            (item.id for item in starter_scenarios),
            "catalog starter scenario ids",
        )
        _validate_cross_references(physics_entries, examples, starter_scenarios)

        loaded_compatibility = compatibility or load_frequensolver_compatibility()
        preferred = loaded_compatibility.preferred_frequensolver
        installed_version = package_version or get_versions()["version"]
        if not isinstance(installed_version, str) or not installed_version.strip():
            raise CatalogValidationError("package_version must be a non-empty string")

        identities = VersionIdentities(
            package_version=installed_version,
            declared_package_release=loaded_compatibility.package_release,
            catalog_schema=schema,
            catalog_version=catalog_version,
            authoring_rules_schema=authoring_rules.schema,
            compatibility_schema=loaded_compatibility.schema,
            preferred_frequensolver_release=(
                preferred.release if preferred is not None else None
            ),
            preferred_frequensolver_commit=(
                preferred.git_commit if preferred is not None else None
            ),
            solver_validation_profile=loaded_compatibility.validation_profile,
            contracts=contracts,
        )
        return cls(
            identities=identities,
            authoring_rules=authoring_rules,
            public_api=public_api,
            glossary=glossary,
            physics_entries=physics_entries,
            validation_codes=validation_codes,
            examples=examples,
            starter_scenarios=starter_scenarios,
            limitations=_string_tuple(data["limitations"], "catalog.limitations"),
        )

    def lookup_physics(self, name: str) -> PhysicsKnowledge:
        """Return catalog knowledge for a canonical physics name or alias."""

        try:
            canonical = canonical_physics(name)
        except ValueError as exc:
            raise KeyError(str(exc)) from exc
        for entry in self.physics_entries:
            if entry.id == canonical:
                return entry
        raise KeyError(f"Physics {canonical!r} is absent from the knowledge catalog")

    def lookup_authoring_rule(self, area: str) -> AuthoringRule:
        """Return one typed authoring-rule area by stable name."""

        return self.authoring_rules.lookup(area)

    def lookup_public_api(self, name: str) -> PublicApiKnowledge:
        """Return one curated public API entry by id, symbol, alias, or path."""

        query = name.strip().casefold()
        for entry in self.public_api:
            if query in {
                entry.id.casefold(),
                entry.symbol.casefold(),
                entry.import_path.casefold(),
                *(alias.casefold() for alias in entry.aliases),
            }:
                return entry
        raise KeyError(f"Public API symbol {name!r} is not cataloged")

    def lookup_glossary(self, term: str) -> GlossaryEntry:
        """Return one glossary entry by id, term, or alias."""

        query = term.strip().casefold()
        for entry in self.glossary:
            if query in {
                entry.id.casefold(),
                entry.term.casefold(),
                *(alias.casefold() for alias in entry.aliases),
            }:
                return entry
        raise KeyError(f"Glossary term {term!r} is not cataloged")

    def explain_validation(self, code: str) -> ValidationCodeKnowledge:
        """Return the stable explanation for a cataloged validation code."""

        for entry in self.validation_codes:
            if entry.code == code:
                return entry
        raise KeyError(f"Validation code {code!r} is not cataloged")

    def get_example(self, example_id: str) -> VettedExample:
        """Return a vetted public example by stable id."""

        for example in self.examples:
            if example.id == example_id:
                return example
        raise KeyError(f"Example {example_id!r} is not cataloged")

    def get_starter_scenario(
        self,
        scenario_id: str = "known-small-2d-acoustic",
    ) -> StarterScenario:
        """Return a complete guided starter setup by stable id."""

        for scenario in self.starter_scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise KeyError(f"Starter scenario {scenario_id!r} is not cataloged")


def load_simulation_knowledge(
    source: Optional[Union[str, Path]] = None,
) -> SimulationKnowledgeCatalog:
    """Load the packaged simulation-knowledge catalog without network access.

    Args:
        source: Optional catalog JSON path for build and validation tooling.
            Normal applications should omit it and load the packaged resource.

    Returns:
        Validated catalog carrying exact installed package and release
        identities.
    """

    if source is None:
        text = (
            files("frequensolve.knowledge")
            .joinpath(CATALOG_RESOURCE)
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(source).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogValidationError(
            f"invalid simulation knowledge JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CatalogValidationError("simulation knowledge catalog must be an object")
    return SimulationKnowledgeCatalog.from_mapping(payload)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise CatalogValidationError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogValidationError(f"{path} must be a non-empty string")
    result = value.strip()
    if "\r" in result or "\n" in result:
        raise CatalogValidationError(f"{path} must be a single-line string")
    return result


def _optional_string(value: object, path: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, path)


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    values = _list(value, path)
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(values)
    )
    if not result:
        raise CatalogValidationError(f"{path} must not be empty")
    _unique(result, path)
    return result


def _string_tuple_or_empty(value: object, path: str) -> tuple[str, ...]:
    values = _list(value, path)
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(values)
    )
    _unique(result, path)
    return result


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise CatalogValidationError(f"{path} must be a boolean")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogValidationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CatalogValidationError(f"{path} must be a finite number")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogValidationError(f"{path} must be an integer")
    return value


def _dimension_tuple(
    value: object,
    path: str,
) -> tuple[Union[int, float], ...]:
    result = []
    for index, item in enumerate(_list(value, path)):
        try:
            result.append(canonical_dimension(item))
        except ValueError as exc:
            raise CatalogValidationError(f"{path}[{index}]: {exc}") from exc
    if not result:
        raise CatalogValidationError(f"{path} must not be empty")
    _unique(result, path)
    return tuple(result)


def _json_mapping(value: object, path: str) -> Mapping[str, object]:
    data = _mapping(value, path)
    try:
        return json.loads(json.dumps(data))
    except (TypeError, ValueError) as exc:
        raise CatalogValidationError(f"{path} must be JSON-compatible") from exc


def _keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    path: str,
    optional: Optional[set[str]] = None,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise CatalogValidationError(f"{path} is missing keys: {', '.join(missing)}")
    allowed = required | (optional or set())
    extra = sorted(set(value) - allowed)
    if extra:
        raise CatalogValidationError(
            f"{path} contains unsupported keys: {', '.join(extra)}"
        )


def _unique(values: Any, path: str) -> None:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        labels = ", ".join(str(value) for value in sorted(duplicates, key=str))
        raise CatalogValidationError(f"{path} must be unique; duplicates: {labels}")


def _contract(value: object, path: str) -> ContractIdentity:
    data = _mapping(value, path)
    _keys(
        data,
        required={"name", "identity", "owner", "source_revision"},
        path=path,
    )
    return ContractIdentity(
        name=_string(data["name"], f"{path}.name"),
        identity=_string(data["identity"], f"{path}.identity"),
        owner=_string(data["owner"], f"{path}.owner"),
        source_revision=_string(data["source_revision"], f"{path}.source_revision"),
    )


def _validate_contracts(contracts: tuple[ContractIdentity, ...]) -> None:
    identities = tuple(
        (item.name, item.identity, item.owner, item.source_revision)
        for item in contracts
    )
    if identities != _EXPECTED_CONTRACT_IDENTITIES:
        raise CatalogValidationError(
            "catalog contracts must identify the pinned Sauce simulation, "
            "acquisition, and job contracts"
        )


def _authoring_rules(value: object, path: str) -> AuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "schema",
            "dimensions",
            "boundary_conditions",
            "discretization",
            "solver",
            "frequencies",
            "acquisition",
            "file_references",
            "outputs",
        },
        path=path,
    )
    schema = _string(data["schema"], f"{path}.schema")
    if schema != AUTHORING_RULES_SCHEMA:
        raise CatalogValidationError(
            f"{path}.schema must be {AUTHORING_RULES_SCHEMA!r}, got {schema!r}"
        )
    return AuthoringRules(
        schema=schema,
        dimensions=_dimension_rules(data["dimensions"], f"{path}.dimensions"),
        boundary_conditions=_boundary_rules(
            data["boundary_conditions"], f"{path}.boundary_conditions"
        ),
        discretization=_discretization_rules(
            data["discretization"], f"{path}.discretization"
        ),
        solver=_solver_rules(data["solver"], f"{path}.solver"),
        frequencies=_frequency_rules(data["frequencies"], f"{path}.frequencies"),
        acquisition=_acquisition_rules(data["acquisition"], f"{path}.acquisition"),
        file_references=_file_reference_rules(
            data["file_references"], f"{path}.file_references"
        ),
        outputs=_output_rules(data["outputs"], f"{path}.outputs"),
    )


def _dimension_rules(value: object, path: str) -> DimensionAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={"supported", "axisymmetric", "model_dimension_for_2_5d"},
        path=path,
    )
    return DimensionAuthoringRules(
        supported=_dimension_tuple(data["supported"], f"{path}.supported"),
        axisymmetric=_dimension_tuple(data["axisymmetric"], f"{path}.axisymmetric"),
        model_dimension_for_2_5d=_integer(
            data["model_dimension_for_2_5d"],
            f"{path}.model_dimension_for_2_5d",
        ),
    )


def _boundary_rules(value: object, path: str) -> BoundaryAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "conditions_required",
            "documented_conditions",
            "solver_specific_conditions_allowed",
            "generated_2d_labels",
            "generated_3d_labels",
            "pml",
        },
        path=path,
    )
    return BoundaryAuthoringRules(
        conditions_required=_boolean(
            data["conditions_required"], f"{path}.conditions_required"
        ),
        documented_conditions=_string_tuple(
            data["documented_conditions"], f"{path}.documented_conditions"
        ),
        solver_specific_conditions_allowed=_boolean(
            data["solver_specific_conditions_allowed"],
            f"{path}.solver_specific_conditions_allowed",
        ),
        generated_2d_labels=_string_tuple(
            data["generated_2d_labels"], f"{path}.generated_2d_labels"
        ),
        generated_3d_labels=_string_tuple(
            data["generated_3d_labels"], f"{path}.generated_3d_labels"
        ),
        pml=_pml_rules(data["pml"], f"{path}.pml"),
    )


def _pml_rules(value: object, path: str) -> PmlAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "condition",
            "defaults",
            "optional_fields",
            "fields_serialize_only_with_condition",
        },
        path=path,
    )
    defaults = _mapping(data["defaults"], f"{path}.defaults")
    _keys(
        defaults,
        required={"pml_wavelengths", "pml_exponent", "pml_reflectivity"},
        path=f"{path}.defaults",
    )
    return PmlAuthoringRules(
        condition=_string(data["condition"], f"{path}.condition"),
        pml_wavelengths=_number(
            defaults["pml_wavelengths"], f"{path}.defaults.pml_wavelengths"
        ),
        pml_exponent=_number(defaults["pml_exponent"], f"{path}.defaults.pml_exponent"),
        pml_reflectivity=_number(
            defaults["pml_reflectivity"], f"{path}.defaults.pml_reflectivity"
        ),
        optional_fields=_string_tuple(
            data["optional_fields"], f"{path}.optional_fields"
        ),
        fields_serialize_only_with_condition=_boolean(
            data["fields_serialize_only_with_condition"],
            f"{path}.fields_serialize_only_with_condition",
        ),
    )


def _discretization_rules(
    value: object,
    path: str,
) -> DiscretizationAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "default_payload",
            "rejected_fields",
            "mesh_adapt_order_path",
            "additional_fields_status",
        },
        path=path,
    )
    return DiscretizationAuthoringRules(
        default_payload=_json_mapping(
            data["default_payload"], f"{path}.default_payload"
        ),
        rejected_fields=_string_tuple(
            data["rejected_fields"], f"{path}.rejected_fields"
        ),
        mesh_adapt_order_path=_string(
            data["mesh_adapt_order_path"], f"{path}.mesh_adapt_order_path"
        ),
        additional_fields_status=_string(
            data["additional_fields_status"], f"{path}.additional_fields_status"
        ),
    )


def _solver_rules(value: object, path: str) -> SolverAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "solve_on_values",
            "precision_values",
            "defaults",
            "additional_fields_status",
        },
        path=path,
    )
    defaults = _mapping(data["defaults"], f"{path}.defaults")
    _keys(
        defaults,
        required={"solve_on", "max_iter", "tolerance", "precision"},
        path=f"{path}.defaults",
    )
    return SolverAuthoringRules(
        solve_on_values=_string_tuple(
            data["solve_on_values"], f"{path}.solve_on_values"
        ),
        precision_values=_string_tuple(
            data["precision_values"], f"{path}.precision_values"
        ),
        default_solve_on=_string(defaults["solve_on"], f"{path}.defaults.solve_on"),
        default_max_iter=_integer(defaults["max_iter"], f"{path}.defaults.max_iter"),
        default_tolerance=_number(defaults["tolerance"], f"{path}.defaults.tolerance"),
        default_precision=_string(defaults["precision"], f"{path}.defaults.precision"),
        additional_fields_status=_string(
            data["additional_fields_status"], f"{path}.additional_fields_status"
        ),
    )


def _frequency_rules(value: object, path: str) -> FrequencyAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "explicit_min_items",
            "finite_required",
            "positive_real_required",
            "complex_damping_imaginary_sign",
            "time_domain_spacing_fields",
            "time_domain_exactly_one_spacing_field",
            "time_domain_positive_fields",
            "time_domain_f_max_greater_than_f_min",
            "damping_fields",
            "damping_fields_mutually_exclusive",
            "damping_factor_minimum",
        },
        path=path,
    )
    return FrequencyAuthoringRules(
        explicit_min_items=_integer(
            data["explicit_min_items"], f"{path}.explicit_min_items"
        ),
        finite_required=_boolean(data["finite_required"], f"{path}.finite_required"),
        positive_real_required=_boolean(
            data["positive_real_required"], f"{path}.positive_real_required"
        ),
        complex_damping_imaginary_sign=_string(
            data["complex_damping_imaginary_sign"],
            f"{path}.complex_damping_imaginary_sign",
        ),
        time_domain_spacing_fields=_string_tuple(
            data["time_domain_spacing_fields"],
            f"{path}.time_domain_spacing_fields",
        ),
        time_domain_exactly_one_spacing_field=_boolean(
            data["time_domain_exactly_one_spacing_field"],
            f"{path}.time_domain_exactly_one_spacing_field",
        ),
        time_domain_positive_fields=_string_tuple(
            data["time_domain_positive_fields"],
            f"{path}.time_domain_positive_fields",
        ),
        time_domain_f_max_greater_than_f_min=_boolean(
            data["time_domain_f_max_greater_than_f_min"],
            f"{path}.time_domain_f_max_greater_than_f_min",
        ),
        damping_fields=_string_tuple(data["damping_fields"], f"{path}.damping_fields"),
        damping_fields_mutually_exclusive=_boolean(
            data["damping_fields_mutually_exclusive"],
            f"{path}.damping_fields_mutually_exclusive",
        ),
        damping_factor_minimum=_number(
            data["damping_factor_minimum"], f"{path}.damping_factor_minimum"
        ),
    )


def _acquisition_rules(value: object, path: str) -> AcquisitionAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={"hdf5_dense_source_encoding"},
        path=path,
    )
    return AcquisitionAuthoringRules(
        hdf5_dense_source_encoding=_hdf5_dense_encoding_rules(
            data["hdf5_dense_source_encoding"],
            f"{path}.hdf5_dense_source_encoding",
        )
    )


def _hdf5_dense_encoding_rules(
    value: object,
    path: str,
) -> Hdf5DenseEncodingAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "coefficient_dataset_rank",
            "coefficient_axis_order",
            "real_imag_pair_size",
            "real_imag_order",
            "storage_kinds",
            "native_complex_storage_allowed",
            "all_dimensions_non_empty",
        },
        path=path,
    )
    return Hdf5DenseEncodingAuthoringRules(
        coefficient_dataset_rank=_integer(
            data["coefficient_dataset_rank"],
            f"{path}.coefficient_dataset_rank",
        ),
        coefficient_axis_order=_string_tuple(
            data["coefficient_axis_order"],
            f"{path}.coefficient_axis_order",
        ),
        real_imag_pair_size=_integer(
            data["real_imag_pair_size"],
            f"{path}.real_imag_pair_size",
        ),
        real_imag_order=_string_tuple(
            data["real_imag_order"],
            f"{path}.real_imag_order",
        ),
        storage_kinds=_string_tuple(
            data["storage_kinds"],
            f"{path}.storage_kinds",
        ),
        native_complex_storage_allowed=_boolean(
            data["native_complex_storage_allowed"],
            f"{path}.native_complex_storage_allowed",
        ),
        all_dimensions_non_empty=_boolean(
            data["all_dimensions_non_empty"],
            f"{path}.all_dimensions_non_empty",
        ),
    )


def _file_reference_rules(
    value: object,
    path: str,
) -> FileReferenceAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "relative_paths_resolve_from",
            "missing_project_local_severity",
            "existing_external_files_validate_locally",
            "remote_unverified_policy",
            "explicit_opt_in_parameter",
            "remote_unverified_requires_absolute_path",
            "remote_unverified_requires_outside_project",
            "remote_unverified_code",
            "remote_unverified_severity",
            "remote_warning_includes_concrete_path",
        },
        path=path,
    )
    return FileReferenceAuthoringRules(
        relative_paths_resolve_from=_string(
            data["relative_paths_resolve_from"],
            f"{path}.relative_paths_resolve_from",
        ),
        missing_project_local_severity=_string(
            data["missing_project_local_severity"],
            f"{path}.missing_project_local_severity",
        ),
        existing_external_files_validate_locally=_boolean(
            data["existing_external_files_validate_locally"],
            f"{path}.existing_external_files_validate_locally",
        ),
        remote_unverified_policy=_string(
            data["remote_unverified_policy"],
            f"{path}.remote_unverified_policy",
        ),
        explicit_opt_in_parameter=_string(
            data["explicit_opt_in_parameter"],
            f"{path}.explicit_opt_in_parameter",
        ),
        remote_unverified_requires_absolute_path=_boolean(
            data["remote_unverified_requires_absolute_path"],
            f"{path}.remote_unverified_requires_absolute_path",
        ),
        remote_unverified_requires_outside_project=_boolean(
            data["remote_unverified_requires_outside_project"],
            f"{path}.remote_unverified_requires_outside_project",
        ),
        remote_unverified_code=_string(
            data["remote_unverified_code"],
            f"{path}.remote_unverified_code",
        ),
        remote_unverified_severity=_string(
            data["remote_unverified_severity"],
            f"{path}.remote_unverified_severity",
        ),
        remote_warning_includes_concrete_path=_boolean(
            data["remote_warning_includes_concrete_path"],
            f"{path}.remote_warning_includes_concrete_path",
        ),
    )


def _output_rules(value: object, path: str) -> OutputAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "traces_enabled_by_default",
            "vtk_frequency_count",
            "vtk_targets",
            "vtk_formats",
            "vtk_upscale_targets",
            "vtk_upscale_minimum",
            "vtk_upscale_maximum",
            "vtk_grid_allows_upscale",
            "wavefield_grid_required",
            "source_ids_one_based",
            "complex_parts",
            "item_kinds",
            "model_surfaces",
        },
        path=path,
    )
    return OutputAuthoringRules(
        traces_enabled_by_default=_boolean(
            data["traces_enabled_by_default"],
            f"{path}.traces_enabled_by_default",
        ),
        vtk_frequency_count=_integer(
            data["vtk_frequency_count"], f"{path}.vtk_frequency_count"
        ),
        vtk_targets=_string_tuple(data["vtk_targets"], f"{path}.vtk_targets"),
        vtk_formats=_string_tuple(data["vtk_formats"], f"{path}.vtk_formats"),
        vtk_upscale_targets=_string_tuple(
            data["vtk_upscale_targets"], f"{path}.vtk_upscale_targets"
        ),
        vtk_upscale_minimum=_integer(
            data["vtk_upscale_minimum"], f"{path}.vtk_upscale_minimum"
        ),
        vtk_upscale_maximum=_integer(
            data["vtk_upscale_maximum"], f"{path}.vtk_upscale_maximum"
        ),
        vtk_grid_allows_upscale=_boolean(
            data["vtk_grid_allows_upscale"], f"{path}.vtk_grid_allows_upscale"
        ),
        wavefield_grid_required=_boolean(
            data["wavefield_grid_required"], f"{path}.wavefield_grid_required"
        ),
        source_ids_one_based=_boolean(
            data["source_ids_one_based"], f"{path}.source_ids_one_based"
        ),
        complex_parts=_string_tuple(data["complex_parts"], f"{path}.complex_parts"),
        item_kinds=_string_tuple(data["item_kinds"], f"{path}.item_kinds"),
        model_surfaces=_model_surface_rules(
            data["model_surfaces"], f"{path}.model_surfaces"
        ),
    )


def _model_surface_rules(
    value: object,
    path: str,
) -> ModelSurfaceAuthoringRules:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "named_aliases",
            "authored_and_expanded_horizon_names",
            "indexed_prefix",
            "index_base",
            "case_sensitive",
            "bottom_alias_allowed",
            "borehole_surface_names_allowed",
        },
        path=path,
    )
    return ModelSurfaceAuthoringRules(
        named_aliases=_string_tuple(data["named_aliases"], f"{path}.named_aliases"),
        authored_and_expanded_horizon_names=_boolean(
            data["authored_and_expanded_horizon_names"],
            f"{path}.authored_and_expanded_horizon_names",
        ),
        indexed_prefix=_string(
            data["indexed_prefix"],
            f"{path}.indexed_prefix",
        ),
        index_base=_integer(data["index_base"], f"{path}.index_base"),
        case_sensitive=_boolean(
            data["case_sensitive"],
            f"{path}.case_sensitive",
        ),
        bottom_alias_allowed=_boolean(
            data["bottom_alias_allowed"],
            f"{path}.bottom_alias_allowed",
        ),
        borehole_surface_names_allowed=_boolean(
            data["borehole_surface_names_allowed"],
            f"{path}.borehole_surface_names_allowed",
        ),
    )


def _public_api(value: object, path: str) -> PublicApiKnowledge:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "id",
            "symbol",
            "import_path",
            "aliases",
            "kind",
            "category",
            "summary",
            "related_glossary",
        },
        path=path,
    )
    entry_id = _string(data["id"], f"{path}.id")
    if not _CATALOG_ID_RE.fullmatch(entry_id):
        raise CatalogValidationError(
            f"{path}.id must be a stable lower-case kebab identifier"
        )
    symbol = _string(data["symbol"], f"{path}.symbol")
    if not _PUBLIC_SYMBOL_RE.fullmatch(symbol):
        raise CatalogValidationError(f"{path}.symbol must be a public Python name")
    import_path = _string(data["import_path"], f"{path}.import_path")
    if (
        not _PUBLIC_IMPORT_PATH_RE.fullmatch(import_path)
        or import_path != f"frequensolve.{symbol}"
    ):
        raise CatalogValidationError(
            f"{path}.import_path must be the top-level path for its public symbol"
        )
    aliases = _string_tuple_or_empty(data["aliases"], f"{path}.aliases")
    for index, alias in enumerate(aliases):
        if not _PUBLIC_SYMBOL_RE.fullmatch(alias):
            raise CatalogValidationError(
                f"{path}.aliases[{index}] must be a public Python name"
            )
    kind = _string(data["kind"], f"{path}.kind")
    if kind not in _PUBLIC_API_KINDS:
        raise CatalogValidationError(
            f"{path}.kind must be one of: {', '.join(sorted(_PUBLIC_API_KINDS))}"
        )
    category = _string(data["category"], f"{path}.category")
    if category not in _PUBLIC_API_CATEGORIES:
        raise CatalogValidationError(
            f"{path}.category must be one of: "
            f"{', '.join(sorted(_PUBLIC_API_CATEGORIES))}"
        )
    return PublicApiKnowledge(
        id=entry_id,
        symbol=symbol,
        import_path=import_path,
        aliases=aliases,
        kind=kind,
        category=category,
        summary=_string(data["summary"], f"{path}.summary"),
        related_glossary=_string_tuple(
            data["related_glossary"], f"{path}.related_glossary"
        ),
    )


def _glossary_entry(value: object, path: str) -> GlossaryEntry:
    data = _mapping(value, path)
    _keys(
        data,
        required={"id", "term", "aliases", "definition", "related_api"},
        path=path,
    )
    entry_id = _string(data["id"], f"{path}.id")
    if not _CATALOG_ID_RE.fullmatch(entry_id):
        raise CatalogValidationError(
            f"{path}.id must be a stable lower-case kebab identifier"
        )
    return GlossaryEntry(
        id=entry_id,
        term=_string(data["term"], f"{path}.term"),
        aliases=_string_tuple_or_empty(data["aliases"], f"{path}.aliases"),
        definition=_string(data["definition"], f"{path}.definition"),
        related_api=_string_tuple(data["related_api"], f"{path}.related_api"),
    )


def _validate_public_api(entries: tuple[PublicApiKnowledge, ...]) -> None:
    if not entries:
        raise CatalogValidationError("catalog public_api must not be empty")
    _unique((entry.id for entry in entries), "catalog public API ids")
    _unique((entry.symbol for entry in entries), "catalog public API symbols")
    _unique(
        (entry.import_path for entry in entries),
        "catalog public API import paths",
    )
    lookup_owners: dict[str, str] = {}
    for entry in entries:
        for key in (entry.id, entry.symbol, entry.import_path, *entry.aliases):
            normalized = key.casefold()
            owner = lookup_owners.setdefault(normalized, entry.id)
            if owner != entry.id:
                raise CatalogValidationError(
                    "catalog public API lookup keys must be unambiguous; "
                    f"{key!r} maps to both {owner!r} and {entry.id!r}"
                )


def _validate_glossary(entries: tuple[GlossaryEntry, ...]) -> None:
    if not entries:
        raise CatalogValidationError("catalog glossary must not be empty")
    _unique((entry.id for entry in entries), "catalog glossary ids")
    lookup_owners: dict[str, str] = {}
    for entry in entries:
        for key in (entry.id, entry.term, *entry.aliases):
            normalized = key.casefold()
            owner = lookup_owners.setdefault(normalized, entry.id)
            if owner != entry.id:
                raise CatalogValidationError(
                    "catalog glossary lookup keys must be unambiguous; "
                    f"{key!r} maps to both {owner!r} and {entry.id!r}"
                )


def _validate_agent_knowledge_cross_references(
    public_api: tuple[PublicApiKnowledge, ...],
    glossary: tuple[GlossaryEntry, ...],
) -> None:
    api_ids = {entry.id for entry in public_api}
    glossary_ids = {entry.id for entry in glossary}
    for api_entry in public_api:
        unknown = sorted(set(api_entry.related_glossary) - glossary_ids)
        if unknown:
            raise CatalogValidationError(
                f"public API {api_entry.id!r} references unknown glossary entries: "
                f"{', '.join(unknown)}"
            )
    for glossary_entry in glossary:
        unknown = sorted(set(glossary_entry.related_api) - api_ids)
        if unknown:
            raise CatalogValidationError(
                f"glossary entry {glossary_entry.id!r} references unknown public API entries: "
                f"{', '.join(unknown)}"
            )


def _physics(value: object, path: str) -> PhysicsKnowledge:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "id",
            "aliases",
            "summary",
            "supported_dimensions",
            "property_requirements",
            "required_properties",
            "output_components",
        },
        optional={"material_profile", "guided_scenario_id"},
        path=path,
    )
    requirement_status = _string(
        data["property_requirements"], f"{path}.property_requirements"
    )
    if requirement_status not in {
        "cataloged",
        "domain-specific",
        "not-cataloged",
    }:
        raise CatalogValidationError(
            f"{path}.property_requirements must be cataloged, "
            "domain-specific, or not-cataloged"
        )
    required_properties = _string_tuple_or_empty(
        data["required_properties"], f"{path}.required_properties"
    )
    for index, property_name in enumerate(required_properties):
        if canonical_property_name(property_name) != property_name:
            raise CatalogValidationError(
                f"{path}.required_properties[{index}] must be canonical"
            )
    material_profile = _optional_string(
        data.get("material_profile"), f"{path}.material_profile"
    )
    if requirement_status == "cataloged":
        if not required_properties or material_profile is None:
            raise CatalogValidationError(
                f"{path} cataloged property requirements need a material "
                "profile and required properties"
            )
    elif required_properties or material_profile is not None:
        raise CatalogValidationError(
            f"{path} may only declare a material profile and required "
            "properties when property_requirements is cataloged"
        )
    return PhysicsKnowledge(
        id=_string(data["id"], f"{path}.id"),
        aliases=_string_tuple(data["aliases"], f"{path}.aliases"),
        summary=_string(data["summary"], f"{path}.summary"),
        supported_dimensions=_dimension_tuple(
            data["supported_dimensions"], f"{path}.supported_dimensions"
        ),
        property_requirements=requirement_status,
        material_profile=material_profile,
        required_properties=required_properties,
        output_components=_string_tuple(
            data["output_components"], f"{path}.output_components"
        ),
        guided_scenario_id=_optional_string(
            data.get("guided_scenario_id"), f"{path}.guided_scenario_id"
        ),
    )


def _validation_code(value: object, path: str) -> ValidationCodeKnowledge:
    data = _mapping(value, path)
    _keys(
        data,
        required={"code", "severity", "path", "explanation", "remediation"},
        path=path,
    )
    code = _string(data["code"], f"{path}.code")
    if not _VALIDATION_CODE_RE.fullmatch(code):
        raise CatalogValidationError(f"{path}.code is not a stable dotted identifier")
    severity = _string(data["severity"], f"{path}.severity")
    if severity not in {"error", "warning"}:
        raise CatalogValidationError(f"{path}.severity must be 'error' or 'warning'")
    issue_path = data["path"]
    if not isinstance(issue_path, str):
        raise CatalogValidationError(f"{path}.path must be a string")
    return ValidationCodeKnowledge(
        code=code,
        severity=severity,
        path=issue_path,
        explanation=_string(data["explanation"], f"{path}.explanation"),
        remediation=_string(data["remediation"], f"{path}.remediation"),
    )


def _example(value: object, path: str) -> VettedExample:
    data = _mapping(value, path)
    _keys(
        data,
        required={"id", "title", "summary", "source_path", "tested_by"},
        optional={"scenario_id"},
        path=path,
    )
    source_path = _repository_path(data["source_path"], f"{path}.source_path")
    tested_by = tuple(
        _repository_path(item, f"{path}.tested_by[{index}]")
        for index, item in enumerate(_list(data["tested_by"], f"{path}.tested_by"))
    )
    if not tested_by:
        raise CatalogValidationError(f"{path}.tested_by must not be empty")
    _unique(tested_by, f"{path}.tested_by")
    return VettedExample(
        id=_string(data["id"], f"{path}.id"),
        title=_string(data["title"], f"{path}.title"),
        summary=_string(data["summary"], f"{path}.summary"),
        source_path=source_path,
        tested_by=tested_by,
        scenario_id=_optional_string(data.get("scenario_id"), f"{path}.scenario_id"),
    )


def _scenario(value: object, path: str) -> StarterScenario:
    data = _mapping(value, path)
    _keys(
        data,
        required={
            "id",
            "title",
            "summary",
            "physics",
            "dimension",
            "example_id",
            "setup",
            "limitations",
        },
        path=path,
    )
    dimension_value = data["dimension"]
    try:
        dimension = canonical_dimension(dimension_value)
    except ValueError as exc:
        raise CatalogValidationError(f"{path}.dimension: {exc}") from exc
    physics = _string(data["physics"], f"{path}.physics")
    try:
        canonical = canonical_physics(physics)
    except ValueError as exc:
        raise CatalogValidationError(f"{path}.physics: {exc}") from exc
    if canonical != physics:
        raise CatalogValidationError(
            f"{path}.physics must use canonical physics id {canonical!r}"
        )
    if dimension not in supported_dimensions_for_physics(physics):
        raise CatalogValidationError(
            f"{path}.dimension {dimension!r} is unsupported for physics " f"{physics!r}"
        )
    setup = _mapping(data["setup"], f"{path}.setup")
    missing_sections = sorted(_REQUIRED_SETUP_SECTIONS - set(setup))
    extra_sections = sorted(set(setup) - _REQUIRED_SETUP_SECTIONS)
    if missing_sections:
        raise CatalogValidationError(
            f"{path}.setup is missing sections: {', '.join(missing_sections)}"
        )
    if extra_sections:
        raise CatalogValidationError(
            f"{path}.setup contains unsupported sections: {', '.join(extra_sections)}"
        )
    _validate_starter_setup(setup, physics=physics, dimension=dimension, path=path)
    # Round-trip through JSON to isolate the immutable model from caller-owned
    # mappings and to reject non-machine-readable values.
    try:
        normalized_setup = json.loads(json.dumps(setup))
    except (TypeError, ValueError) as exc:
        raise CatalogValidationError(f"{path}.setup must be JSON-compatible") from exc
    return StarterScenario(
        id=_string(data["id"], f"{path}.id"),
        title=_string(data["title"], f"{path}.title"),
        summary=_string(data["summary"], f"{path}.summary"),
        physics=physics,
        dimension=dimension,
        example_id=_string(data["example_id"], f"{path}.example_id"),
        setup=normalized_setup,
        limitations=_string_tuple(data["limitations"], f"{path}.limitations"),
    )


def _validate_starter_setup(
    setup: Mapping[str, object],
    *,
    physics: str,
    dimension: Union[int, float],
    path: str,
) -> None:
    setup_path = f"{path}.setup"
    project = _required_mapping(
        setup["project"],
        f"{setup_path}.project",
        {"name", "pretty_name", "load_if_exists"},
    )
    _string(project["name"], f"{setup_path}.project.name")
    _string(project["pretty_name"], f"{setup_path}.project.pretty_name")
    _boolean(project["load_if_exists"], f"{setup_path}.project.load_if_exists")

    simulation = _required_mapping(
        setup["simulation"],
        f"{setup_path}.simulation",
        {"name", "physics", "dimension", "units"},
    )
    _string(simulation["name"], f"{setup_path}.simulation.name")
    setup_physics = _string(simulation["physics"], f"{setup_path}.simulation.physics")
    if setup_physics != physics:
        raise CatalogValidationError(
            f"{setup_path}.simulation.physics must match {path}.physics"
        )
    try:
        setup_dimension = canonical_dimension(simulation["dimension"])
    except ValueError as exc:
        raise CatalogValidationError(
            f"{setup_path}.simulation.dimension: {exc}"
        ) from exc
    if setup_dimension != dimension:
        raise CatalogValidationError(
            f"{setup_path}.simulation.dimension must match {path}.dimension"
        )
    units = _mapping(simulation["units"], f"{setup_path}.simulation.units")
    for unit_name, unit_value in units.items():
        _string(unit_value, f"{setup_path}.simulation.units.{unit_name}")

    model = _required_mapping(
        setup["model"],
        f"{setup_path}.model",
        {"type", "name", "dimension", "x_limits", "surfaces", "layers"},
    )
    model_type = _string(model["type"], f"{setup_path}.model.type")
    if model_type != "LayeredModel":
        raise CatalogValidationError(f"{setup_path}.model.type must be 'LayeredModel'")
    _string(model["name"], f"{setup_path}.model.name")
    setup_model_dimension = _integer(
        model["dimension"], f"{setup_path}.model.dimension"
    )
    if setup_model_dimension != model_dimension(dimension):
        raise CatalogValidationError(
            f"{setup_path}.model.dimension must match the simulation model dimension"
        )
    _number_list(
        model["x_limits"],
        f"{setup_path}.model.x_limits",
        minimum_items=2,
    )
    surfaces = _non_empty_list(model["surfaces"], f"{setup_path}.model.surfaces")
    if len(surfaces) < 2:
        raise CatalogValidationError(
            f"{setup_path}.model.surfaces must contain at least 2 items"
        )
    for index, surface in enumerate(surfaces):
        surface_data = _required_mapping(
            surface,
            f"{setup_path}.model.surfaces[{index}]",
            {"name", "depth"},
        )
        _string(surface_data["name"], f"{setup_path}.model.surfaces[{index}].name")
        _number(surface_data["depth"], f"{setup_path}.model.surfaces[{index}].depth")
    layers = _non_empty_list(model["layers"], f"{setup_path}.model.layers")
    for index, layer in enumerate(layers):
        layer_data = _required_mapping(
            layer,
            f"{setup_path}.model.layers[{index}]",
            {"name", "properties"},
        )
        _string(layer_data["name"], f"{setup_path}.model.layers[{index}].name")
        properties = _mapping(
            layer_data["properties"],
            f"{setup_path}.model.layers[{index}].properties",
        )
        if not properties:
            raise CatalogValidationError(
                f"{setup_path}.model.layers[{index}].properties must not be empty"
            )
        for property_name, property_value in properties.items():
            _number(
                property_value,
                f"{setup_path}.model.layers[{index}].properties.{property_name}",
            )

    mesh = _required_mapping(
        setup["mesh"],
        f"{setup_path}.mesh",
        {"type", "n", "adapt", "source_grading"},
    )
    mesh_type = _string(mesh["type"], f"{setup_path}.mesh.type")
    if mesh_type != "HexMeshGenerator":
        raise CatalogValidationError(
            f"{setup_path}.mesh.type must be 'HexMeshGenerator'"
        )
    mesh_counts = _non_empty_list(mesh["n"], f"{setup_path}.mesh.n")
    if len(mesh_counts) < 2:
        raise CatalogValidationError(
            f"{setup_path}.mesh.n must contain at least 2 items"
        )
    for index, value in enumerate(mesh_counts):
        _positive_integer(value, f"{setup_path}.mesh.n[{index}]")
    adapt = _required_mapping(
        mesh["adapt"],
        f"{setup_path}.mesh.adapt",
        {"elems_per_wave", "order", "f_low", "f_high"},
    )
    _positive_number(adapt["elems_per_wave"], f"{setup_path}.mesh.adapt.elems_per_wave")
    _positive_integer(adapt["order"], f"{setup_path}.mesh.adapt.order")
    _positive_number(adapt["f_low"], f"{setup_path}.mesh.adapt.f_low")
    _positive_number(adapt["f_high"], f"{setup_path}.mesh.adapt.f_high")
    source_grading = _required_mapping(
        mesh["source_grading"],
        f"{setup_path}.mesh.source_grading",
        {"d1", "d0", "factor"},
        optional={"power"},
    )
    _positive_number(source_grading["d1"], f"{setup_path}.mesh.source_grading.d1")
    _nonnegative_number(source_grading["d0"], f"{setup_path}.mesh.source_grading.d0")
    _positive_number(
        source_grading["factor"], f"{setup_path}.mesh.source_grading.factor"
    )
    if "power" in source_grading:
        _positive_number(
            source_grading["power"], f"{setup_path}.mesh.source_grading.power"
        )

    boundaries = _non_empty_list(
        setup["boundary_conditions"], f"{setup_path}.boundary_conditions"
    )
    for index, boundary in enumerate(boundaries):
        boundary_data = _required_mapping(
            boundary,
            f"{setup_path}.boundary_conditions[{index}]",
            {"conditions", "boundaries"},
            optional={
                "name",
                "pml_wavelengths",
                "pml_exponent",
                "pml_reflectivity",
                "pml_constant",
            },
        )
        _string_tuple(
            boundary_data["conditions"],
            f"{setup_path}.boundary_conditions[{index}].conditions",
        )
        _string_tuple(
            boundary_data["boundaries"],
            f"{setup_path}.boundary_conditions[{index}].boundaries",
        )
        if "name" in boundary_data:
            _string(
                boundary_data["name"],
                f"{setup_path}.boundary_conditions[{index}].name",
            )
        for field_name in (
            "pml_wavelengths",
            "pml_exponent",
            "pml_reflectivity",
            "pml_constant",
        ):
            if field_name in boundary_data:
                _positive_number(
                    boundary_data[field_name],
                    f"{setup_path}.boundary_conditions[{index}].{field_name}",
                )

    acquisition = _required_mapping(
        setup["acquisition"],
        f"{setup_path}.acquisition",
        {"source", "receiver_group"},
    )
    source = _required_mapping(
        acquisition["source"],
        f"{setup_path}.acquisition.source",
        {"kind", "coords"},
    )
    _string(source["kind"], f"{setup_path}.acquisition.source.kind")
    source_coordinates = _non_empty_list(
        source["coords"], f"{setup_path}.acquisition.source.coords"
    )
    for index, coordinates in enumerate(source_coordinates):
        _number_list(
            coordinates,
            f"{setup_path}.acquisition.source.coords[{index}]",
            minimum_items=2,
            maximum_items=2,
        )
    receiver = _required_mapping(
        acquisition["receiver_group"],
        f"{setup_path}.acquisition.receiver_group",
        {"name", "device_name", "component", "coordinate_line"},
    )
    _string(receiver["name"], f"{setup_path}.acquisition.receiver_group.name")
    _string(
        receiver["device_name"],
        f"{setup_path}.acquisition.receiver_group.device_name",
    )
    component = _required_mapping(
        receiver["component"],
        f"{setup_path}.acquisition.receiver_group.component",
        {"name", "field"},
        optional={"direction"},
    )
    _string(
        component["name"],
        f"{setup_path}.acquisition.receiver_group.component.name",
    )
    _string(
        component["field"],
        f"{setup_path}.acquisition.receiver_group.component.field",
    )
    if "direction" in component:
        _number_list(
            component["direction"],
            f"{setup_path}.acquisition.receiver_group.component.direction",
            minimum_items=2,
            maximum_items=3,
        )
    coordinate_line = _required_mapping(
        receiver["coordinate_line"],
        f"{setup_path}.acquisition.receiver_group.coordinate_line",
        {"axis", "start", "stop", "count", "fixed"},
    )
    if (
        _string(
            coordinate_line["axis"],
            f"{setup_path}.acquisition.receiver_group.coordinate_line.axis",
        )
        != "x"
    ):
        raise CatalogValidationError(
            f"{setup_path}.acquisition.receiver_group.coordinate_line.axis "
            "must be 'x'"
        )
    _number(
        coordinate_line["start"],
        f"{setup_path}.acquisition.receiver_group.coordinate_line.start",
    )
    _number(
        coordinate_line["stop"],
        f"{setup_path}.acquisition.receiver_group.coordinate_line.stop",
    )
    count = _integer(
        coordinate_line["count"],
        f"{setup_path}.acquisition.receiver_group.coordinate_line.count",
    )
    if count < 1:
        raise CatalogValidationError(
            f"{setup_path}.acquisition.receiver_group.coordinate_line.count "
            "must be positive"
        )
    fixed = _required_mapping(
        coordinate_line["fixed"],
        f"{setup_path}.acquisition.receiver_group.coordinate_line.fixed",
        {"z"},
    )
    _number(
        fixed["z"],
        f"{setup_path}.acquisition.receiver_group.coordinate_line.fixed.z",
    )

    _required_mapping(
        setup["discretization"],
        f"{setup_path}.discretization",
        set(),
    )
    solver = _required_mapping(
        setup["solver"],
        f"{setup_path}.solver",
        {"tolerance", "grids"},
        optional={"solve_on", "max_iter", "precision"},
    )
    _positive_number(solver["tolerance"], f"{setup_path}.solver.tolerance")
    _positive_integer(solver["grids"], f"{setup_path}.solver.grids")
    if "solve_on" in solver:
        _enum_string(
            solver["solve_on"],
            f"{setup_path}.solver.solve_on",
            {"final", "all"},
        )
    if "max_iter" in solver:
        _positive_integer(solver["max_iter"], f"{setup_path}.solver.max_iter")
    if "precision" in solver:
        _enum_string(
            solver["precision"],
            f"{setup_path}.solver.precision",
            {"single", "double"},
        )

    job = _required_mapping(
        setup["job"],
        f"{setup_path}.job",
        {"type", "name", "f_list", "outputs"},
    )
    job_type = _string(job["type"], f"{setup_path}.job.type")
    if job_type != "FrequencyDomainJob":
        raise CatalogValidationError(
            f"{setup_path}.job.type must be 'FrequencyDomainJob'"
        )
    _string(job["name"], f"{setup_path}.job.name")
    frequencies = _non_empty_list(job["f_list"], f"{setup_path}.job.f_list")
    for index, frequency in enumerate(frequencies):
        _positive_number(frequency, f"{setup_path}.job.f_list[{index}]")
    outputs = _required_mapping(job["outputs"], f"{setup_path}.job.outputs", {"vtk"})
    vtk_outputs = _non_empty_list(outputs["vtk"], f"{setup_path}.job.outputs.vtk")
    for index, vtk_output in enumerate(vtk_outputs):
        vtk_data = _required_mapping(
            vtk_output,
            f"{setup_path}.job.outputs.vtk[{index}]",
            {"name", "fields", "properties", "upscale"},
            optional={"path", "show_pml", "format", "order", "parts"},
        )
        vtk_path = f"{setup_path}.job.outputs.vtk[{index}]"
        _string(vtk_data["name"], f"{vtk_path}.name")
        _string_tuple(vtk_data["fields"], f"{vtk_path}.fields")
        _string_tuple(vtk_data["properties"], f"{vtk_path}.properties")
        upscale = _integer(vtk_data["upscale"], f"{vtk_path}.upscale")
        if upscale < 0 or upscale > 2:
            raise CatalogValidationError(f"{vtk_path}.upscale must be from 0 to 2")
        if "path" in vtk_data:
            _repository_path(vtk_data["path"], f"{vtk_path}.path")
        if "show_pml" in vtk_data:
            _boolean(vtk_data["show_pml"], f"{vtk_path}.show_pml")
        if "format" in vtk_data:
            _enum_string(
                vtk_data["format"],
                f"{vtk_path}.format",
                {"vtu", "xdmf", "xmf", "vtr"},
            )
        if "order" in vtk_data:
            _positive_integer(vtk_data["order"], f"{vtk_path}.order")
        if "parts" in vtk_data:
            parts = vtk_data["parts"]
            if isinstance(parts, str):
                _enum_string(
                    parts,
                    f"{vtk_path}.parts",
                    {"real", "imag", "abs"},
                )
            else:
                parsed_parts = _string_tuple(parts, f"{vtk_path}.parts")
                if not set(parsed_parts).issubset({"real", "imag", "abs"}):
                    raise CatalogValidationError(
                        f"{vtk_path}.parts contains unsupported values"
                    )


def _required_mapping(
    value: object,
    path: str,
    required: set[str],
    *,
    optional: Optional[set[str]] = None,
) -> Mapping[str, object]:
    data = _mapping(value, path)
    missing = sorted(required - set(data))
    if missing:
        raise CatalogValidationError(f"{path} is missing keys: {', '.join(missing)}")
    extra = sorted(set(data) - required - (optional or set()))
    if extra:
        raise CatalogValidationError(
            f"{path} contains unsupported keys: {', '.join(extra)}"
        )
    return data


def _non_empty_list(value: object, path: str) -> list[object]:
    result = _list(value, path)
    if not result:
        raise CatalogValidationError(f"{path} must not be empty")
    return result


def _number_list(
    value: object,
    path: str,
    *,
    minimum_items: int,
    maximum_items: Optional[int] = None,
) -> list[float]:
    values = _list(value, path)
    if len(values) < minimum_items:
        raise CatalogValidationError(
            f"{path} must contain at least {minimum_items} items"
        )
    if maximum_items is not None and len(values) > maximum_items:
        raise CatalogValidationError(
            f"{path} must contain at most {maximum_items} items"
        )
    return [_number(item, f"{path}[{index}]") for index, item in enumerate(values)]


def _positive_number(value: object, path: str) -> float:
    result = _number(value, path)
    if result <= 0:
        raise CatalogValidationError(f"{path} must be positive")
    return result


def _nonnegative_number(value: object, path: str) -> float:
    result = _number(value, path)
    if result < 0:
        raise CatalogValidationError(f"{path} must be nonnegative")
    return result


def _positive_integer(value: object, path: str) -> int:
    result = _integer(value, path)
    if result < 1:
        raise CatalogValidationError(f"{path} must be positive")
    return result


def _enum_string(value: object, path: str, allowed: set[str]) -> str:
    result = _string(value, path)
    if result not in allowed:
        raise CatalogValidationError(
            f"{path} must be one of: {', '.join(sorted(allowed))}"
        )
    return result


def _repository_path(value: object, path: str) -> str:
    raw = _string(value, path)
    if "\\" in raw:
        raise CatalogValidationError(f"{path} must use repository-relative POSIX paths")
    parsed = PurePosixPath(raw)
    if parsed.is_absolute() or ".." in parsed.parts or raw.startswith(".codex/"):
        raise CatalogValidationError(
            f"{path} must be a public repository-relative path"
        )
    return raw


def _validate_physics_entries(entries: tuple[PhysicsKnowledge, ...]) -> None:
    by_id = {entry.id: entry for entry in entries}
    _unique((entry.id for entry in entries), "catalog physics ids")
    expected_ids = supported_physics()
    if tuple(entry.id for entry in entries) != expected_ids:
        raise CatalogValidationError(
            "catalog physics ids must match supported_physics() in stable order"
        )
    all_aliases: list[str] = []
    for entry in entries:
        try:
            canonical = canonical_physics(entry.id)
        except ValueError as exc:
            raise CatalogValidationError(
                f"catalog physics id {entry.id!r} is unsupported"
            ) from exc
        if canonical != entry.id:
            raise CatalogValidationError(
                f"catalog physics id {entry.id!r} is not canonical"
            )
        expected_aliases = physics_aliases(entry.id)
        if entry.aliases != expected_aliases:
            raise CatalogValidationError(
                f"catalog aliases for {entry.id!r} must match physics_aliases()"
            )
        expected_dimensions = supported_dimensions_for_physics(entry.id)
        if entry.supported_dimensions != expected_dimensions:
            raise CatalogValidationError(
                f"catalog dimensions for {entry.id!r} must match "
                "supported_dimensions_for_physics()"
            )
        expected_components = tuple(
            components_for_physics(entry.id).allowed_components()
        )
        if entry.output_components != expected_components:
            raise CatalogValidationError(
                f"catalog output components for {entry.id!r} must match "
                "components_for_physics()"
            )
        expected_materials = _EXPECTED_MATERIAL_REQUIREMENTS[entry.id]
        actual_materials = (
            entry.property_requirements,
            entry.material_profile,
            entry.required_properties,
        )
        if actual_materials != expected_materials:
            raise CatalogValidationError(
                f"catalog material requirements for {entry.id!r} must match "
                "the public material profile"
            )
        all_aliases.extend(entry.aliases)
    if len(by_id) != len(entries):
        raise CatalogValidationError("catalog physics ids must be unique")
    _unique(all_aliases, "catalog physics aliases")


def _validate_authoring_rules(
    rules: AuthoringRules,
    physics_entries: tuple[PhysicsKnowledge, ...],
) -> None:
    dimensions = rules.dimensions
    if dimensions.supported != (2, 2.5, 3):
        raise CatalogValidationError(
            "authoring dimensions must match canonical_dimension()"
        )
    if dimensions.axisymmetric != (2,):
        raise CatalogValidationError(
            "axisymmetric authoring dimensions must contain only 2"
        )
    if dimensions.model_dimension_for_2_5d != 2:
        raise CatalogValidationError(
            "2.5D simulations must declare a 2D model dimension"
        )

    boundary = rules.boundary_conditions
    if not boundary.conditions_required:
        raise CatalogValidationError("boundary conditions must require conditions")
    if not boundary.solver_specific_conditions_allowed:
        raise CatalogValidationError(
            "boundary conditions must allow solver-specific condition names"
        )
    if boundary.documented_conditions != _DOCUMENTED_BOUNDARY_CONDITIONS:
        raise CatalogValidationError("documented boundary conditions are invalid")
    if (
        boundary.pml.condition != "pml"
        or boundary.pml.condition not in boundary.documented_conditions
    ):
        raise CatalogValidationError(
            "PML condition must be present in documented boundary conditions"
        )
    if boundary.pml.optional_fields != ("pml_constant",):
        raise CatalogValidationError("PML optional fields are invalid")
    if boundary.generated_2d_labels != ("x_min", "x_max", "z_min", "z_max"):
        raise CatalogValidationError("generated 2D boundary labels are invalid")
    if boundary.generated_3d_labels != (
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    ):
        raise CatalogValidationError("generated 3D boundary labels are invalid")
    if (
        boundary.pml.pml_wavelengths != 0.5
        or boundary.pml.pml_exponent != 3.0
        or boundary.pml.pml_reflectivity != 0.01
    ):
        raise CatalogValidationError("PML defaults are invalid")
    if not boundary.pml.fields_serialize_only_with_condition:
        raise CatalogValidationError(
            "PML fields must serialize only when the PML condition is present"
        )

    discretization = rules.discretization
    if discretization.default_payload:
        raise CatalogValidationError("default discretization payload must be empty")
    if discretization.rejected_fields != ("order",):
        raise CatalogValidationError(
            "discretization must reject the relocated order field"
        )
    if discretization.mesh_adapt_order_path != "mesh.adapt.order":
        raise CatalogValidationError(
            "discretization order ownership must point to mesh.adapt.order"
        )
    if discretization.additional_fields_status != "contract-dependent":
        raise CatalogValidationError(
            "additional discretization fields must be contract-dependent"
        )

    solver = rules.solver
    if solver.solve_on_values != ("final", "all"):
        raise CatalogValidationError("solver solve_on values are invalid")
    if solver.precision_values != ("single", "double"):
        raise CatalogValidationError("solver precision values are invalid")
    if (
        solver.default_solve_on != "final"
        or solver.default_max_iter != 300
        or solver.default_tolerance != 1.0e-4
        or solver.default_precision != "single"
    ):
        raise CatalogValidationError("default solver settings are invalid")
    if solver.additional_fields_status != "contract-dependent":
        raise CatalogValidationError(
            "additional solver fields must be contract-dependent"
        )

    frequencies = rules.frequencies
    if frequencies.explicit_min_items != 1:
        raise CatalogValidationError("explicit frequency lists require one item")
    if not frequencies.finite_required or not frequencies.positive_real_required:
        raise CatalogValidationError(
            "frequencies must be finite with a positive real part"
        )
    if frequencies.complex_damping_imaginary_sign != "nonpositive":
        raise CatalogValidationError(
            "complex frequency damping must use a nonpositive imaginary part"
        )
    if frequencies.time_domain_spacing_fields != ("df", "T_max"):
        raise CatalogValidationError("time-domain spacing fields are invalid")
    if frequencies.time_domain_positive_fields != ("df", "T_max"):
        raise CatalogValidationError("time-domain positive fields are invalid")
    if (
        not frequencies.time_domain_exactly_one_spacing_field
        or not frequencies.time_domain_f_max_greater_than_f_min
    ):
        raise CatalogValidationError("time-domain sweep constraints are incomplete")
    if frequencies.damping_fields != ("damping_factor", "laplace"):
        raise CatalogValidationError("time-domain damping fields are invalid")
    if (
        not frequencies.damping_fields_mutually_exclusive
        or frequencies.damping_factor_minimum != 1.0
    ):
        raise CatalogValidationError("time-domain damping constraints are invalid")

    encoding = rules.acquisition.hdf5_dense_source_encoding
    if (
        encoding.coefficient_dataset_rank != 3
        or encoding.coefficient_axis_order != ("encoded_field", "source", "real_imag")
        or encoding.real_imag_pair_size != 2
        or encoding.real_imag_order != ("real", "imag")
        or encoding.storage_kinds != ("integer", "floating")
        or encoding.native_complex_storage_allowed
        or not encoding.all_dimensions_non_empty
    ):
        raise CatalogValidationError(
            "HDF5Dense coefficient rules must match the acquisition-v2 "
            "paired real/imag layout"
        )

    files = rules.file_references
    if (
        files.relative_paths_resolve_from != "project_root"
        or files.missing_project_local_severity != "error"
        or not files.existing_external_files_validate_locally
        or files.remote_unverified_policy != "slurm-or-explicit-opt-in"
        or files.explicit_opt_in_parameter != "allow_unverified_remote_files"
        or not files.remote_unverified_requires_absolute_path
        or not files.remote_unverified_requires_outside_project
        or files.remote_unverified_code != "files.remote_unverified"
        or files.remote_unverified_severity != "warning"
        or files.remote_warning_includes_concrete_path
    ):
        raise CatalogValidationError(
            "referenced-file rules must match the package's local/remote "
            "validation policy"
        )

    outputs = rules.outputs
    if not outputs.traces_enabled_by_default or outputs.vtk_frequency_count != 1:
        raise CatalogValidationError("output default/frequency rules are invalid")
    if outputs.vtk_targets != ("volume", "surface", "grid"):
        raise CatalogValidationError("VTK targets are invalid")
    if outputs.vtk_formats != ("vtu", "xdmf", "xmf", "vtr"):
        raise CatalogValidationError("VTK formats are invalid")
    if outputs.vtk_upscale_targets != ("volume", "surface"):
        raise CatalogValidationError("VTK upscale targets are invalid")
    if (
        outputs.vtk_upscale_minimum != 0
        or outputs.vtk_upscale_maximum != 2
        or outputs.vtk_grid_allows_upscale
    ):
        raise CatalogValidationError("VTK upscale constraints are invalid")
    if not outputs.wavefield_grid_required or not outputs.source_ids_one_based:
        raise CatalogValidationError("wavefield/source-id rules are invalid")
    if outputs.complex_parts != ("real", "imag", "abs"):
        raise CatalogValidationError("canonical complex output parts are invalid")
    if outputs.item_kinds != ("field", "property", "info"):
        raise CatalogValidationError("VTK item kinds are invalid")
    model_surfaces = outputs.model_surfaces
    if (
        model_surfaces.named_aliases != ("top",)
        or not model_surfaces.authored_and_expanded_horizon_names
        or model_surfaces.indexed_prefix != "surface_"
        or model_surfaces.index_base != 1
        or model_surfaces.case_sensitive
        or model_surfaces.bottom_alias_allowed
        or model_surfaces.borehole_surface_names_allowed
    ):
        raise CatalogValidationError(
            "ParaView model-surface rules must match solver-visible horizons"
        )

    for physics in physics_entries:
        if not set(physics.supported_dimensions).issubset(dimensions.supported):
            raise CatalogValidationError(
                f"physics {physics.id!r} uses an uncataloged dimension"
            )


def _validate_authoring_diagnostics(
    rules: AuthoringRules,
    validation_codes: tuple[ValidationCodeKnowledge, ...],
) -> None:
    by_code = {entry.code: entry for entry in validation_codes}
    remote_code = rules.file_references.remote_unverified_code
    entry = by_code.get(remote_code)
    if entry is None:
        raise CatalogValidationError(
            f"file-reference diagnostic {remote_code!r} is not cataloged"
        )
    if entry.severity != rules.file_references.remote_unverified_severity:
        raise CatalogValidationError(
            f"file-reference diagnostic {remote_code!r} has the wrong severity"
        )


def _validate_cross_references(
    physics_entries: tuple[PhysicsKnowledge, ...],
    examples: tuple[VettedExample, ...],
    scenarios: tuple[StarterScenario, ...],
) -> None:
    physics_by_id = {entry.id: entry for entry in physics_entries}
    examples_by_id = {entry.id: entry for entry in examples}
    scenarios_by_id = {entry.id: entry for entry in scenarios}

    for scenario in scenarios:
        try:
            canonical = canonical_physics(scenario.physics)
        except ValueError as exc:
            raise CatalogValidationError(
                f"scenario {scenario.id!r} references unsupported physics"
            ) from exc
        if canonical not in physics_by_id:
            raise CatalogValidationError(
                f"scenario {scenario.id!r} references uncataloged physics {canonical!r}"
            )
        if scenario.example_id not in examples_by_id:
            raise CatalogValidationError(
                f"scenario {scenario.id!r} references unknown example "
                f"{scenario.example_id!r}"
            )
        example = examples_by_id[scenario.example_id]
        if example.scenario_id != scenario.id:
            raise CatalogValidationError(
                f"scenario {scenario.id!r} and example {example.id!r} "
                "must reference each other"
            )

    for physics in physics_entries:
        if (
            physics.guided_scenario_id is not None
            and physics.guided_scenario_id not in scenarios_by_id
        ):
            raise CatalogValidationError(
                f"physics {physics.id!r} references unknown guided scenario "
                f"{physics.guided_scenario_id!r}"
            )
        if physics.guided_scenario_id is not None:
            scenario = scenarios_by_id[physics.guided_scenario_id]
            if canonical_physics(scenario.physics) != physics.id:
                raise CatalogValidationError(
                    f"physics {physics.id!r} guided scenario uses "
                    f"{scenario.physics!r}"
                )

    for example in examples:
        if (
            example.scenario_id is not None
            and example.scenario_id not in scenarios_by_id
        ):
            raise CatalogValidationError(
                f"example {example.id!r} references unknown scenario "
                f"{example.scenario_id!r}"
            )
        if example.scenario_id is not None:
            scenario = scenarios_by_id[example.scenario_id]
            if scenario.example_id != example.id:
                raise CatalogValidationError(
                    f"example {example.id!r} and scenario {scenario.id!r} "
                    "must reference each other"
                )
