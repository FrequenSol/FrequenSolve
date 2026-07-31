"""Local STDIO simulation-assistant MCP server."""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Union

import anyio
import anyio.to_process
from pydantic import BaseModel, ConfigDict, Field, model_validator

from frequensolve._version import get_versions
from frequensolve.knowledge import load_simulation_knowledge
from frequensolve.mcp_server import artifacts, cloud, core
from frequensolve.mcp_server._sdk_v2 import (
    CLOUD_READ_ONLY_TOOL_ANNOTATIONS,
    READ_ONLY_TOOL_ANNOTATIONS,
    SafeMCPServer,
)

MCP_SERVER_VERSION = "2.0.0"
RESPONSE_SCHEMA = "frequensolve-simulation-assistant-response/v1"
MAX_RESPONSE_BYTES = 131_072
SAFE_NAME_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
SAFE_ROOT_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,31}$"
SAFE_RELATIVE_JSON_PATTERN = (
    r"^[A-Za-z0-9_-][A-Za-z0-9_. -]*" r"(?:/[A-Za-z0-9_-][A-Za-z0-9_. -]*)*\.json$"
)
SAFE_CODE_PATTERN = r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$"
SAFE_PROFILE_PATTERN = r"^[a-z][a-z0-9_-]{0,31}$"
SAFE_RESULT_AFTER_PATTERN = (
    r"^(?![/\\])(?![A-Za-z][A-Za-z0-9+.-]*:)"
    r"(?!.*(?:^|/)\.\.?(?:/|$))[^\x00-\x1f\x7f\\]+$"
)

SERVER_INSTRUCTIONS = (
    "FrequenSolve setup assistant. Use it only to explain, draft, validate, "
    "render, inspect, or preview supported simulation inputs. It never submits, "
    "runs, uploads, changes, or deletes simulations. Do not invent package APIs, "
    "solver fields, file paths, or Cloud operations. Guided generation is limited "
    "to the cataloged known-small 2D acoustic scenario. Read artifacts only by an "
    "explicit allowed-root ID and relative path. Cloud tools use only the selected "
    "cached user profile and five fixed read queries. They never expose raw "
    "GraphQL, credentials, result contents, tenant selectors, or write actions."
)


class ClosedModel(BaseModel):
    """Base model for every public protocol object."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class ContractIdentity(ClosedModel):
    name: str
    identity: str
    owner: str
    source_revision: str


class AssistantIdentity(ClosedModel):
    package_version: str
    source_revision: str
    source_dirty: bool
    mcp_server_version: str
    mcp_contract: str
    draft_contract: str
    catalog_schema: str
    catalog_version: str
    authoring_rules_schema: str
    compatibility_schema: str
    preferred_frequensolver_release: str | None
    preferred_frequensolver_commit: str | None
    solver_validation_profile: str | None
    cloud_read_contract_id: str
    cloud_read_contract_version: str
    contracts: tuple[ContractIdentity, ...]


class ToolDiagnostic(ClosedModel):
    code: str
    severity: Literal["error", "warning", "info"]
    path: str = ""
    explanation: str
    remediation: str


class ToolEnvelope(ClosedModel):
    schema_: Literal["frequensolve-simulation-assistant-response/v1"] = Field(
        default=RESPONSE_SCHEMA,
        alias="schema",
    )
    ok: bool
    identity: AssistantIdentity
    media_type: Literal["application/json", "text/x-python"]
    payload: str
    diagnostics: tuple[ToolDiagnostic, ...] = ()


class DraftDocument(ClosedModel):
    schema_: Literal["frequensolve-simulation-draft/v1"] = Field(alias="schema")
    scenario_id: Literal["known-small-2d-acoustic"]
    project_name: str = Field(
        min_length=1,
        max_length=64,
        pattern=SAFE_NAME_PATTERN,
    )
    simulation_name: str = Field(
        min_length=1,
        max_length=64,
        pattern=SAFE_NAME_PATTERN,
    )
    job_name: str = Field(
        min_length=1,
        max_length=64,
        pattern=SAFE_NAME_PATTERN,
    )
    physics: Literal["acoustic"]
    dimension: Literal[2]
    frequency_hz: float = Field(ge=10.0, le=10.0)
    receiver_count: int = Field(ge=1, le=1001)


class FindExampleRequest(ClosedModel):
    query: str = Field(
        default="known small 2D acoustic",
        min_length=1,
        max_length=128,
        pattern=r"^[^\x00-\x1f\x7f]+$",
    )


class CreateDraftRequest(ClosedModel):
    project_name: str = Field(
        default="project",
        min_length=1,
        max_length=64,
        pattern=SAFE_NAME_PATTERN,
    )
    simulation_name: str = Field(
        default="known_small_2d_acoustic",
        min_length=1,
        max_length=64,
        pattern=SAFE_NAME_PATTERN,
    )
    frequency_hz: float = Field(default=10.0, ge=10.0, le=10.0)
    receiver_count: int = Field(default=101, ge=1, le=1001)


class DraftSource(ClosedModel):
    kind: Literal["draft"]
    draft: DraftDocument


class ArtifactReference(ClosedModel):
    root_id: str = Field(
        min_length=1,
        max_length=32,
        pattern=SAFE_ROOT_ID_PATTERN,
    )
    relative_path: str = Field(
        min_length=1,
        max_length=512,
        pattern=SAFE_RELATIVE_JSON_PATTERN,
    )


class ArtifactSource(ClosedModel):
    kind: Literal["artifact"]
    artifact: ArtifactReference


SimulationSource = Annotated[
    Union[DraftSource, ArtifactSource],
    Field(discriminator="kind"),
]


class ValidateRequest(ClosedModel):
    source: SimulationSource


class RenderRequest(ClosedModel):
    draft: DraftDocument


class InspectRequest(ClosedModel):
    artifact: ArtifactReference


class PreviewRequest(ClosedModel):
    source: SimulationSource


class ExplainRequest(ClosedModel):
    codes: list[
        Annotated[
            str,
            Field(
                min_length=1,
                max_length=128,
                pattern=SAFE_CODE_PATTERN,
            ),
        ]
    ] = Field(min_length=1, max_length=20)


class CloudCheckReadinessRequest(ClosedModel):
    pass


class CloudListSimulationsRequest(ClosedModel):
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED"] | None = (
        None
    )
    project_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    limit: int = Field(default=10, ge=1, le=25)
    cursor: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
    )


class CloudGetSimulationRequest(ClosedModel):
    simulation_id: str = Field(
        min_length=1,
        max_length=256,
    )
    view: Literal["summary", "diagnostics"] = "summary"
    limit: int | None = Field(default=None, ge=1, le=50)
    cursor: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
    )

    @model_validator(mode="after")
    def _paging_requires_diagnostics(self) -> "CloudGetSimulationRequest":
        if self.view == "summary" and (
            self.limit is not None or self.cursor is not None
        ):
            raise ValueError("Paging is only valid for the diagnostics view.")
        return self


class CloudListResultArtifactsRequest(ClosedModel):
    simulation_id: str = Field(
        min_length=1,
        max_length=256,
    )
    limit: int = Field(default=25, ge=1, le=50)
    after: str | None = Field(
        default=None,
        min_length=1,
        max_length=768,
        json_schema_extra={"pattern": SAFE_RESULT_AFTER_PATTERN},
    )

    @model_validator(mode="after")
    def _after_is_a_safe_relative_cursor(
        self,
    ) -> "CloudListResultArtifactsRequest":
        if (
            self.after is not None
            and re.fullmatch(
                SAFE_RESULT_AFTER_PATTERN,
                self.after,
            )
            is None
        ):
            raise ValueError("The result cursor must be a safe relative path.")
        return self


def build_server(
    *,
    allowed_roots: Mapping[str, str | Path] | None = None,
    cloud_profile: str | None = None,
    operation_timeout_seconds: float = 15.0,
    max_concurrency: int = 2,
) -> SafeMCPServer:
    """Build the finite local MCP surface."""

    if (
        isinstance(operation_timeout_seconds, bool)
        or not isinstance(operation_timeout_seconds, (int, float))
        or not math.isfinite(operation_timeout_seconds)
        or not 1.0 <= operation_timeout_seconds <= 60.0
    ):
        raise ValueError("operation timeout must be a finite value from 1 to 60")
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or not 1 <= max_concurrency <= 4
    ):
        raise ValueError("max concurrency must be an integer from 1 to 4")
    if cloud_profile is not None and (
        not isinstance(cloud_profile, str)
        or re.fullmatch(SAFE_PROFILE_PATTERN, cloud_profile) is None
    ):
        raise ValueError("cloud profile must be a safe site profile name")

    roots = artifacts.normalize_allowed_roots(allowed_roots or {})
    identity = _identity()
    limiter = anyio.CapacityLimiter(max_concurrency)
    server = SafeMCPServer(
        name="FrequenSolve Simulation Assistant",
        version=MCP_SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
    )

    resource_documents = _resource_documents(roots)
    for resource_name, payload in resource_documents.items():
        uri = f"frequensolve://simulation-assistant/{resource_name}"
        _register_json_resource(
            server,
            uri=uri,
            name=f"frequensolve_{resource_name.replace('-', '_')}",
            payload=payload,
        )

    prompt_text = {
        "start_2d_acoustic": (
            "Start with the known-small 2D acoustic scenario. Read the identity, "
            "physics, authoring-rules, and examples resources. Call "
            "create_simulation_draft, validate_simulation_setup, "
            "render_starter_python, and preview_simulation in that order. Do not "
            "save, submit, run, upload, or invent fields."
        ),
        "review_simulation_setup": (
            "Review a structured draft or explicitly rooted saved artifact. Use "
            "validate_simulation_setup and explain_validation. Treat package "
            "diagnostics as authoritative and do not run or change anything."
        ),
        "prepare_simulation_run": (
            "Prepare a simulation for a later user-approved run by validating and "
            "previewing it. Report exact task count, frequencies, assumptions, and "
            "expected output kinds. This server cannot submit or run it."
        ),
        "debug_validation": (
            "Use validate_simulation_setup, then pass the returned stable codes to "
            "explain_validation. Fix only the draft or source code outside this "
            "server; never bypass validation or reveal unrestricted file paths."
        ),
        "monitor_cloud_simulation": (
            "Use only the fixed Cloud read tools. Check readiness, list the signed-in "
            "user's simulations, then request either the summary or stored "
            "diagnostics for one returned simulation ID. List result metadata only "
            "after success. Never submit, cancel, download, mutate, or invent a raw "
            "GraphQL request."
        ),
    }
    for prompt_name, text in prompt_text.items():
        _register_prompt(server, name=prompt_name, text=text)

    @server.tool(
        name="find_vetted_example",
        description="Find the closest vetted package example without network access.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def find_vetted_example(request: FindExampleRequest) -> ToolEnvelope:
        return await _safe_call(
            identity,
            "application/json",
            lambda: core.find_vetted_example(request.query),
        )

    @server.tool(
        name="create_simulation_draft",
        description="Create the bounded known-small 2D acoustic draft.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def create_simulation_draft(request: CreateDraftRequest) -> ToolEnvelope:
        return await _safe_call(
            identity,
            "application/json",
            lambda: core.create_simulation_draft(
                project_name=request.project_name,
                simulation_name=request.simulation_name,
                frequency_hz=request.frequency_hz,
                receiver_count=request.receiver_count,
            ),
        )

    @server.tool(
        name="validate_simulation_setup",
        description="Validate a draft or supported rooted artifact with FrequenSolve.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def validate_simulation_setup(request: ValidateRequest) -> ToolEnvelope:
        if isinstance(request.source, DraftSource):
            draft = request.source.draft.model_dump(mode="json", by_alias=True)
            with tempfile.TemporaryDirectory(prefix="frequensolve-mcp-draft-") as temp:
                return await _safe_process_call(
                    identity,
                    "application/json",
                    core.validate_simulation_draft,
                    draft,
                    str(Path(temp) / "project"),
                    limiter=limiter,
                    timeout_seconds=operation_timeout_seconds,
                )
        return await _artifact_call(
            identity,
            request.source.artifact,
            roots,
            mode="validate",
            limiter=limiter,
            timeout_seconds=operation_timeout_seconds,
        )

    @server.tool(
        name="render_starter_python",
        description="Render deterministic starter Python; never execute or save it.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def render_starter_python(request: RenderRequest) -> ToolEnvelope:
        return await _safe_call(
            identity,
            "text/x-python",
            lambda: core.render_starter_python(
                request.draft.model_dump(mode="json", by_alias=True)
            ),
        )

    @server.tool(
        name="inspect_simulation_artifact",
        description="Inspect one supported JSON artifact under an allowed root.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def inspect_simulation_artifact(request: InspectRequest) -> ToolEnvelope:
        return await _artifact_call(
            identity,
            request.artifact,
            roots,
            mode="inspect",
            limiter=limiter,
            timeout_seconds=operation_timeout_seconds,
        )

    @server.tool(
        name="preview_simulation",
        description="Preview exact frequencies, task count, assumptions, and outputs.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def preview_simulation(request: PreviewRequest) -> ToolEnvelope:
        if isinstance(request.source, DraftSource):
            return await _safe_process_call(
                identity,
                "application/json",
                core.preview_simulation,
                request.source.draft.model_dump(mode="json", by_alias=True),
                limiter=limiter,
                timeout_seconds=operation_timeout_seconds,
            )
        return await _artifact_call(
            identity,
            request.source.artifact,
            roots,
            mode="preview",
            limiter=limiter,
            timeout_seconds=operation_timeout_seconds,
        )

    @server.tool(
        name="explain_validation",
        description="Explain stable package validation codes in plain language.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def explain_validation(request: ExplainRequest) -> ToolEnvelope:
        return await _safe_call(
            identity,
            "application/json",
            lambda: {
                "items": [core.explain_validation(code) for code in request.codes]
            },
        )

    @server.tool(
        name="cloud_check_readiness",
        description="Read the signed-in user's safe Cloud readiness summary.",
        annotations=CLOUD_READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def cloud_check_readiness(
        request: CloudCheckReadinessRequest,
    ) -> ToolEnvelope:
        del request
        return await _safe_cloud_process_call(
            identity,
            cloud_profile,
            "getCloudReadiness",
            {},
            limiter=limiter,
            timeout_seconds=operation_timeout_seconds,
        )

    @server.tool(
        name="cloud_list_simulations",
        description="List a bounded page of the signed-in user's simulations.",
        annotations=CLOUD_READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def cloud_list_simulations(
        request: CloudListSimulationsRequest,
    ) -> ToolEnvelope:
        arguments: dict[str, Any] = {"limit": request.limit}
        if request.status is not None:
            arguments["status"] = request.status
        if request.project_name is not None:
            arguments["projectName"] = request.project_name
        if request.cursor is not None:
            arguments["cursor"] = request.cursor
        return await _safe_cloud_process_call(
            identity,
            cloud_profile,
            "listMySimulations",
            arguments,
            limiter=limiter,
            timeout_seconds=operation_timeout_seconds,
        )

    @server.tool(
        name="cloud_get_simulation",
        description="Read one owned simulation summary or its stored diagnostics.",
        annotations=CLOUD_READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def cloud_get_simulation(
        request: CloudGetSimulationRequest,
    ) -> ToolEnvelope:
        arguments: dict[str, Any] = {"simulationId": request.simulation_id}
        operation = "getMySimulation"
        if request.view == "diagnostics":
            operation = "listMySimulationDiagnostics"
            arguments["limit"] = request.limit if request.limit is not None else 25
            if request.cursor is not None:
                arguments["cursor"] = request.cursor
        return await _safe_cloud_process_call(
            identity,
            cloud_profile,
            operation,
            arguments,
            limiter=limiter,
            timeout_seconds=operation_timeout_seconds,
        )

    @server.tool(
        name="cloud_list_result_artifacts",
        description="List safe relative result metadata for one owned simulation.",
        annotations=CLOUD_READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    async def cloud_list_result_artifacts(
        request: CloudListResultArtifactsRequest,
    ) -> ToolEnvelope:
        arguments: dict[str, Any] = {
            "simulationId": request.simulation_id,
            "limit": request.limit,
        }
        if request.after is not None:
            arguments["after"] = request.after
        return await _safe_cloud_process_call(
            identity,
            cloud_profile,
            "listMySimulationResultArtifacts",
            arguments,
            limiter=limiter,
            timeout_seconds=operation_timeout_seconds,
        )

    server.finalize_safety_contract(
        tool_names={
            "cloud_check_readiness",
            "cloud_get_simulation",
            "cloud_list_result_artifacts",
            "cloud_list_simulations",
            "find_vetted_example",
            "create_simulation_draft",
            "validate_simulation_setup",
            "render_starter_python",
            "inspect_simulation_artifact",
            "preview_simulation",
            "explain_validation",
        },
        prompt_names=set(prompt_text),
        resource_uris={
            f"frequensolve://simulation-assistant/{name}" for name in resource_documents
        },
    )
    return server


def _identity() -> AssistantIdentity:
    catalog = load_simulation_knowledge()
    versions = get_versions()
    identities = catalog.identities
    return AssistantIdentity(
        package_version=identities.package_version,
        source_revision=str(versions["full-revisionid"]),
        source_dirty=bool(versions["dirty"]),
        mcp_server_version=MCP_SERVER_VERSION,
        mcp_contract="frequensolve-simulation-assistant/v1",
        draft_contract="frequensolve-simulation-draft/v1",
        catalog_schema=identities.catalog_schema,
        catalog_version=identities.catalog_version,
        authoring_rules_schema=identities.authoring_rules_schema,
        compatibility_schema=identities.compatibility_schema,
        preferred_frequensolver_release=identities.preferred_frequensolver_release,
        preferred_frequensolver_commit=identities.preferred_frequensolver_commit,
        solver_validation_profile=identities.solver_validation_profile,
        cloud_read_contract_id=cloud.CONTRACT_ID,
        cloud_read_contract_version=cloud.CONTRACT_VERSION,
        contracts=tuple(
            ContractIdentity(
                name=item.name,
                identity=item.identity,
                owner=item.owner,
                source_revision=item.source_revision,
            )
            for item in identities.contracts
        ),
    )


def _resource_documents(roots: Mapping[str, str]) -> dict[str, Any]:
    names = (
        "identity",
        "contracts",
        "catalog",
        "public-api",
        "physics",
        "authoring-rules",
        "validation-codes",
        "examples",
        "glossary",
    )
    documents: dict[str, Any] = {name: core.resource_payload(name) for name in names}
    documents["allowed-roots"] = {
        "schema": "frequensolve-mcp-allowed-roots/v1",
        "root_ids": sorted(roots),
        "policy": (
            "Artifact tools accept only a configured root ID and POSIX relative "
            "JSON path. Absolute paths, traversal, symlinks, and external data "
            "references are rejected."
        ),
    }
    documents["cloud-read-contract"] = {
        "schema": "frequensolve-mcp-cloud-read-contract/v1",
        "contract_id": cloud.CONTRACT_ID,
        "contract_version": cloud.CONTRACT_VERSION,
        "classification": "customer-self-read-only",
        "profile_selection": "configured-at-server-startup",
        "tools": [
            "cloud_check_readiness",
            "cloud_list_simulations",
            "cloud_get_simulation",
            "cloud_list_result_artifacts",
        ],
        "forbidden": [
            "accountId",
            "userId",
            "token",
            "query",
            "mutation",
            "resultContents",
        ],
    }
    return documents


def _register_json_resource(
    server: SafeMCPServer,
    *,
    uri: str,
    name: str,
    payload: Any,
) -> None:
    text = _canonical_json(payload)
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("A fixed MCP resource exceeds the response budget")

    async def read_fixed_resource() -> str:
        return text

    read_fixed_resource.__name__ = name
    server.resource(
        uri,
        name=name,
        description="Version-matched FrequenSolve simulation knowledge.",
        mime_type="application/json",
    )(read_fixed_resource)


def _register_prompt(server: SafeMCPServer, *, name: str, text: str) -> None:
    async def render_fixed_prompt() -> str:
        return text

    render_fixed_prompt.__name__ = name
    server.prompt(
        name=name,
        description="Safe deterministic FrequenSolve simulation guidance.",
    )(render_fixed_prompt)


async def _safe_call(
    identity: AssistantIdentity,
    media_type: Literal["application/json", "text/x-python"],
    operation: Any,
) -> ToolEnvelope:
    try:
        result = operation()
        return _success(identity, media_type, result)
    except Exception:
        return _failure(
            identity,
            "mcp.input.invalid",
            "The requested setup is outside the supported bounded contract.",
            "Use the published schema and the known-small 2D acoustic scenario.",
        )


async def _safe_process_call(
    identity: AssistantIdentity,
    media_type: Literal["application/json", "text/x-python"],
    function: Any,
    *args: Any,
    limiter: anyio.CapacityLimiter,
    timeout_seconds: float,
) -> ToolEnvelope:
    try:
        with anyio.fail_after(timeout_seconds):
            result = await anyio.to_process.run_sync(
                function,
                *args,
                cancellable=True,
                limiter=limiter,
            )
        return _success(identity, media_type, result)
    except TimeoutError:
        return _failure(
            identity,
            "mcp.execution.timeout",
            "The bounded validation worker timed out.",
            "Reduce the input size or inspect the setup in smaller steps.",
        )
    except Exception:
        return _failure(
            identity,
            "mcp.setup.invalid",
            "FrequenSolve could not validate the supported setup.",
            "Review the structured diagnostics and the version-matched catalog.",
        )


async def _safe_cloud_process_call(
    identity: AssistantIdentity,
    profile: str | None,
    operation: str,
    arguments: Mapping[str, Any],
    *,
    limiter: anyio.CapacityLimiter,
    timeout_seconds: float,
) -> ToolEnvelope:
    try:
        with anyio.fail_after(timeout_seconds):
            result = await anyio.to_process.run_sync(
                cloud.execute_cloud_operation,
                profile,
                operation,
                dict(arguments),
                cancellable=True,
                limiter=limiter,
            )
        return _success(
            identity,
            "application/json",
            result,
            contract_validated_json=True,
        )
    except TimeoutError:
        return _failure(
            identity,
            "cloud.upstream_unavailable",
            "FrequenSol Cloud could not complete the read request.",
            "Retry the same bounded read request later.",
        )
    except cloud.CloudReadError as exc:
        code = str(exc.code)
        return _failure(
            identity,
            f"cloud.{code.casefold()}",
            exc.safe_message,
            _cloud_remediation(code),
        )
    except Exception:
        return _failure(
            identity,
            "cloud.upstream_unavailable",
            "FrequenSol Cloud could not complete the read request.",
            "Check the selected Cloud profile and retry later.",
        )


def _cloud_remediation(code: str) -> str:
    return {
        "CLOUD_SUPPORT_REQUIRED": (
            "Install Cloud support with: pip install 'frequensolve[mcp,cloud]'"
        ),
        "CLOUD_CONFIGURATION_REQUIRED": (
            "Configure a non-interactive AWS site profile and cache its public "
            "FrequenSol Cloud configuration."
        ),
        "AUTHENTICATION_REQUIRED": (
            "Sign in through the selected FrequenSolve Cloud site profile, then retry."
        ),
        "INVALID_INPUT": "Use the published bounded Cloud tool schema.",
        "NOT_FOUND_OR_ACCESS_DENIED": (
            "Choose a simulation returned by cloud_list_simulations for this user."
        ),
        "RESULTS_NOT_AVAILABLE": (
            "Wait for the owned simulation to produce safe result metadata."
        ),
        "RESPONSE_LIMIT_EXCEEDED": "Request a smaller page.",
        "UPSTREAM_UNAVAILABLE": "Retry the same bounded read request later.",
    }.get(code, "Check the selected Cloud profile and retry later.")


async def _artifact_call(
    identity: AssistantIdentity,
    reference: ArtifactReference,
    roots: Mapping[str, str],
    *,
    mode: str,
    limiter: anyio.CapacityLimiter,
    timeout_seconds: float,
) -> ToolEnvelope:
    with tempfile.TemporaryDirectory(prefix="frequensolve-mcp-artifact-") as temp:
        return await _safe_process_call(
            identity,
            "application/json",
            artifacts.inspect_or_validate_artifact,
            roots,
            reference.root_id,
            reference.relative_path,
            mode,
            temp,
            limiter=limiter,
            timeout_seconds=timeout_seconds,
        )


def _success(
    identity: AssistantIdentity,
    media_type: Literal["application/json", "text/x-python"],
    result: Any,
    *,
    contract_validated_json: bool = False,
) -> ToolEnvelope:
    if contract_validated_json:
        if media_type != "application/json" or not isinstance(result, Mapping):
            return _failure(
                identity,
                "cloud.upstream_unavailable",
                "FrequenSol Cloud could not complete the read request.",
                "Retry the same bounded read request later.",
            )
        payload = _canonical_json(result)
    elif isinstance(result, str):
        payload = _redact_text(result)
    else:
        payload = _canonical_json(_redact_json_value(result))
    if len(payload.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return _failure(
            identity,
            "mcp.output.too_large",
            "The bounded tool result exceeded the output limit.",
            "Request a smaller result or use a narrower operation.",
        )
    return ToolEnvelope(
        ok=True,
        identity=identity,
        media_type=media_type,
        payload=payload,
    )


def _failure(
    identity: AssistantIdentity,
    code: str,
    explanation: str,
    remediation: str,
) -> ToolEnvelope:
    return ToolEnvelope(
        ok=False,
        identity=identity,
        media_type="application/json",
        payload="{}",
        diagnostics=(
            ToolDiagnostic(
                code=code,
                severity="error",
                explanation=explanation,
                remediation=remediation,
            ),
        ),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b[a-z0-9_-]*(?:api[_-]?key|secret[_-]?access[_-]?key|"
        r"access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token)"
        r"\s*[=:]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)(?:password|secret|token|credential)\s*[=:]\s*[^\s,;]+"),
    re.compile(
        r"\b(?:sk-(?:proj|svcacct)-[A-Za-z0-9_-]{8,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    re.compile(r"(?i)https?://[^/\s:@]+:[^/\s@]+@"),
)
_PROHIBITED_URI = re.compile(r"(?i)\b(?:file|ftp|gs|s3|scp|ssh)://[^\s'\"`]+")
_ABSOLUTE_PATH = re.compile(r"(?<![:/A-Za-z0-9_.-])/(?!/)(?:[^/\s'\"`]+/)*[^/\s'\"`]+")
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\(?:[^\\\s'\"`]+\\)*[^\\\s'\"`]*")


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted-secret>"
                if _is_sensitive_key(str(key))
                else _redact_json_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _is_sensitive_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    return any(
        marker in normalized
        for marker in (
            "authorization",
            "password",
            "secret",
            "token",
            "credential",
            "apikey",
            "accesskey",
            "privatekey",
        )
    )


def _redact_text(value: str) -> str:
    result = _PROHIBITED_URI.sub("<redacted-uri>", value)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("<redacted-secret>", result)
    result = _WINDOWS_PATH.sub("<redacted-path>", result)
    result = _ABSOLUTE_PATH.sub("<redacted-path>", result)
    return result
