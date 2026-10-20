"""Finite local MCP workflow tools. Paid authority exists only in Cloud review."""

from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlencode, urlsplit

import anyio
from mcp.server.context import ServerRequestContext
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

from frequensolve.mcp_server._sdk_v2 import SafeMCPServer

from . import CONTRACT
from .analysis import analyze_pressure
from .client import WorkflowClient, create_project, read_local, relative, write_local
from .notebooks import create_notebook, replay_notebook
from .results import download_results, public_results, snapshot_from_bundle
from .storage import PrivateStore, canonical, directory

TOOLS = (
    "workflow_discover",
    "workflow_create_project",
    "workflow_plan",
    "workflow_results",
)
RESOURCE = "frequensolve://workflow/capabilities"
PROMPT = "curated_3d_workflow"
REVIEW_TOOL = "workflow_review"


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Empty(Closed):
    pass


class ProjectRequest(Closed):
    root_id: str
    relative_path: str
    display_name: str = Field(min_length=1, max_length=128)


class ResourceEdit(Closed):
    frequency_index: int = Field(ge=0, le=95)
    reset: bool | None = None
    nodes: int | None = Field(default=None, ge=1)
    mpi_ranks: int | None = Field(default=None, ge=1)
    cpus_per_rank: int | None = Field(default=None, ge=1)
    memory_mib_per_node: int | None = Field(default=None, ge=1)
    wall_time_seconds: int | None = Field(default=None, ge=60)


class CursorKey(Closed):
    scope: str
    id: str


class PlanCursor(Closed):
    phase: Literal["plans", "preparations"]
    key: CursorKey


class PlanRequest(Closed):
    action: Literal[
        "prepare", "create", "get", "list", "revise", "validate", "quote", "credits"
    ]
    root_id: str
    local_project: str | None = Field(
        default=None, description="Local project path; also supplies identity on reads."
    )
    project_key: str | None = Field(
        default=None,
        description="Owned project key, required for reads without local_project.",
    )
    project_id: str | None = None
    plan_id: str | None = None
    cluster_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=0)
    revision: int | None = Field(default=None, ge=1)
    intent: str | None = Field(default=None, min_length=1, max_length=100)
    edits: list[ResourceEdit] | None = Field(default=None, max_length=96)
    cursor: PlanCursor | None = Field(
        default=None,
        description="Reuse only a cursor returned by this same list operation.",
    )

    @model_validator(mode="after")
    def required_identity(self) -> PlanRequest:
        if self.action in ("get", "list", "credits"):
            if not self.local_project and not self.project_key:
                raise ValueError(
                    "Supply local_project or project_key on every plan read"
                )
        elif not self.local_project:
            raise ValueError("This action requires the existing local_project path")
        if self.action not in ("prepare", "create", "list") and not self.plan_id:
            raise ValueError("Supply the saved plan_id for this action")
        return self


class ReviewRequest(Closed):
    action: Literal["preview", "request_input", "recover"]
    root_id: str
    local_project: str
    plan_id: str
    intent: str | None = Field(default=None, min_length=1, max_length=100)
    expected_fingerprint: str | None = Field(
        default=None, pattern=r"^sha256:[a-f0-9]{64}$"
    )

    @model_validator(mode="after")
    def required_review_identity(self) -> ReviewRequest:
        if self.action != "preview" and not self.intent:
            raise ValueError("Retain a durable local review intent")
        if self.action == "request_input" and not self.expected_fingerprint:
            raise ValueError("Use the exact fingerprint returned by preview")
        return self


class ResultRequest(Closed):
    action: Literal["list", "download", "analyze", "notebook"]
    root_id: str
    project_key: str | None = None
    plan_id: str | None = None
    execution_id: str | None = None
    frequency_indices: list[int] = Field(default_factory=lambda: [0, 1], max_length=2)
    destination: str | None = None
    bundle: str | None = None
    reference_bundle: str | None = None
    frequency_index: int = Field(default=0, ge=0, le=1)


class Reply(Closed):
    version: str = CONTRACT
    ok: bool
    payload: str = "{}"
    diagnostic: str | None = None


def build_server(
    client: WorkflowClient,
    roots: dict[str, Path],
    cloud_url: str,
    *,
    interactive_review: bool = False,
) -> SafeMCPServer:
    origin = urlsplit(cloud_url)
    if (
        origin.scheme not in ("http", "https")
        or origin.username
        or origin.password
        or origin.path not in ("", "/")
        or origin.query
        or origin.fragment
    ):
        raise ValueError("Use an exact Sandbox product origin")
    if origin.scheme == "http" and origin.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Human review requires HTTPS or local Sandbox")
    cloud_url = cloud_url.rstrip("/")
    tools = TOOLS + (REVIEW_TOOL,) if interactive_review else TOOLS
    limiter = anyio.CapacityLimiter(2)
    server = SafeMCPServer(
        name="FrequenSolve scientific workflow",
        version="1.0.0",
        instructions="Restricted curated 3D acoustic workflow. Discover, prepare, review and analyze owned work. "
        "All paid stages require human review in Cloud. Never invent results or submit using other credentials. "
        "Retain intent IDs on uncertainty; intentional new work requires a new intent. Capacity may be unavailable. "
        "Selected local data may be sent to the user's chosen model provider. Notebook replay is analysis only.",
        allowed_tool_names=tools,
        allowed_resource_uris=(RESOURCE,),
        allowed_prompt_names=(PROMPT,),
    )

    @server.resource(
        RESOURCE,
        name="workflow-capabilities",
        description="Versioned restricted workflow contract and explicit allowed root IDs.",
        mime_type="application/json",
    )
    async def capabilities() -> str:
        from frequensolve._version import get_versions

        manifest = json.loads(
            Path(__file__)
            .with_name("contracts")
            .joinpath("workflow_v1.json")
            .read_bytes()
        )
        versions = get_versions()
        return canonical(
            {
                **manifest,
                "allowed_root_ids": sorted(roots),
                "native_review": {
                    "enabled": interactive_review,
                    "mode": "rehearsal_only",
                    "paid_authority": False,
                    "requires": "elicitation.form",
                    "evidence_origin": "client_reported_unattested",
                },
                "package_version": versions["version"],
                "source_revision": versions["full-revisionid"],
                "source_dirty": versions["dirty"],
            }
        ).decode()

    @server.prompt(
        name=PROMPT,
        description="Guide the qualified 3D acoustic scientific workflow with human paid-stage reviews.",
    )
    async def workflow_prompt() -> str:
        return (
            "Discover the accessible Sandbox CPU cluster and report unknown capacity. Create a fresh curated 3D acoustic 2/20 Hz project. "
            "Prepare a plan with a stable intent, open its Cloud review link for human planning approval, and read saved progress. "
            "Inspect and revise resources using an exact revision and a new intent. Validate and obtain separate human approval if compute is required. "
            "Quote execution and open Cloud review for the human to run the exact plan. Never authorize with another credential or retry failed solver work. "
            "Read progress and complete Credit history, retrieve selected immutable native outputs, compare compatible pressure data, and cleanly replay an analysis-only notebook. "
            "Retain saved IDs on uncertainty and describe local execution evidence separately from scientific correctness."
        )

    def root(root_id: str) -> Path:
        if root_id not in roots:
            raise ValueError("Unknown allowed root ID")
        with directory(roots[root_id]):
            pass
        return roots[root_id]

    def review(value: dict) -> dict:
        head = value.get("head")
        if head:
            value = {
                **value,
                "review_url": cloud_url
                + "/agent-review?"
                + urlencode(
                    {
                        "project_key": head["owner"]["project_key"],
                        "plan_id": head["plan_id"],
                    }
                ),
                "paid_authority": False,
            }
        return value

    async def bounded(function: Callable[..., Any], *args: Any) -> Reply:
        try:
            with anyio.fail_after(60):
                value = await anyio.to_thread.run_sync(
                    function, *args, limiter=limiter, abandon_on_cancel=True
                )
            if len(canonical(value)) > 120000:
                raise ValueError(
                    "Result exceeds the bounded response limit; use local artifacts"
                )
            return Reply(ok=True, payload=canonical(value).decode())
        except Exception:
            return Reply(
                ok=False,
                diagnostic="Workflow operation could not be confirmed. Retain its intent and saved IDs; check the explicit connection, local artifacts or Cloud review. Do not create replacement paid work.",
            )

    @server.tool(
        name="workflow_discover",
        description="Discover the accessible qualified Sandbox CPU cluster, permissions, hardware, limits and observed health. Unknown capacity remains unavailable.",
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        structured_output=True,
    )
    async def discover(request: Empty) -> Reply:
        return await bounded(client.call, "discover", {})

    @server.tool(
        name="workflow_create_project",
        description="Create a new local immutable project and unchanged curated 3D double-precision acoustic 2/20 Hz example beneath an allowed root. Never overwrite an existing destination.",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def project(request: ProjectRequest) -> Reply:
        return await bounded(
            create_project,
            root(request.root_id),
            relative(request.relative_path),
            request.display_name,
        )

    def plan_operation(request: PlanRequest) -> dict:
        from frequensolve.project_identity import ProjectIdentity

        project = root(request.root_id)
        arguments: dict[str, Any]
        if request.local_project:
            project = project / relative(request.local_project)
            identity = ProjectIdentity.from_dict(
                json.loads(read_local(project, ".frequensolve/project-identity.json"))
            )
            arguments = {
                "project_key": identity.storage_prefix,
                "project_id": identity.project_id,
            }
        elif request.action in ("get", "list", "credits") and request.project_key:
            arguments = {
                "project_key": request.project_key,
                **({"project_id": request.project_id} if request.project_id else {}),
            }
        else:
            raise ValueError(
                "Preparation and edits require a local immutable project; reads require an owned project key"
            )
        if request.action in ("prepare", "create"):
            if request.action == "create" and request.intent:
                journal = client.journal(project)
                saved = journal.inspect(request.intent)
                if saved is not None:
                    if saved["request"]["operation"] != "plan_create":
                        raise ValueError("Intent belongs to a different operation")
                    if (
                        request.cluster_id
                        and request.cluster_id
                        != saved["request"]["arguments"]["cluster_id"]
                    ):
                        raise ValueError("Intent belongs to a different cluster")
                    return review(
                        journal.perform(
                            request.intent,
                            "plan_create",
                            saved["request"]["arguments"],
                            client.call,
                            reconcile=True,
                        )
                    )
            prepared = client.prepare(project)
            if request.action == "prepare":
                return {
                    key: prepared[key]
                    for key in ("project_id", "project_key", "session_id", "job_path")
                }
            if not request.intent:
                raise ValueError("Create needs a durable intent name")
            if not request.cluster_id:
                raise ValueError("Select the eligible cluster ID returned by discovery")
            value = client.journal(project).perform(
                request.intent,
                "plan_create",
                {
                    **arguments,
                    "session_id": prepared["session_id"],
                    "job_path": prepared["job_path"],
                    "cluster_id": request.cluster_id,
                    "partition": "cpu",
                    "policy": {
                        "max_nodes_per_frequency": 2,
                        "max_concurrent_frequencies": 2,
                        "max_running_cpus": 64,
                        "max_wall_time_seconds": 1800,
                        "max_layout_evaluations": 4,
                    },
                },
                client.call,
            )
            return review(value)
        if request.action != "list":
            if not request.plan_id:
                raise ValueError("Select a saved plan ID")
            arguments["plan_id"] = request.plan_id
        if request.action == "get" and request.revision is not None:
            arguments["revision"] = request.revision
        if request.action == "list" and request.cursor is not None:
            arguments["cursor"] = request.cursor.model_dump()
        if request.action in ("revise", "validate", "quote"):
            if request.expected_revision is None:
                raise ValueError("Use the exact saved revision")
            arguments["expected_revision"] = request.expected_revision
        if request.action == "revise":
            if not request.edits:
                raise ValueError("Provide explicit bounded resource edits")
            arguments["edits"] = [
                edit.model_dump(exclude_none=True) for edit in request.edits
            ]
        operation = (
            "credit_history"
            if request.action == "credits"
            else "plan_" + request.action
        )
        if request.action in ("revise", "validate"):
            if not request.intent:
                raise ValueError("Mutations need a durable intent name")
            return review(
                client.journal(project).perform(
                    request.intent, operation, arguments, client.call
                )
            )
        if request.action == "credits":
            return credit_scan(client, project, arguments)
        return review(client.call(operation, arguments))

    @server.tool(
        name="workflow_plan",
        description="Prepare inputs, create plan-only requests, read/edit/validate exact saved revisions, request quotes, or scan Credit history. Every call needs local_project, or project_key for reads. Repeating create with the same intent reconciles the original input preparation; it cannot approve compute. Only reuse returned cursors. Paid stages always require separate human Cloud review.",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        structured_output=True,
    )
    async def plan(request: PlanRequest) -> Reply:
        return await bounded(plan_operation, request)

    def result_operation(request: ResultRequest) -> dict:
        allowed = root(request.root_id)
        if request.action in ("list", "download"):
            if not all((request.project_key, request.plan_id, request.execution_id)):
                raise ValueError("Select exact owned project, plan and execution IDs")
            args = {
                "project_key": request.project_key,
                "plan_id": request.plan_id,
                "execution_id": request.execution_id,
            }
            if request.action == "list":
                return public_results(client.call("plan_results", args))
            if not request.destination:
                raise ValueError("Choose a new result bundle destination")
            return download_results(
                client, args, request.frequency_indices, allowed, request.destination
            )
        if not request.bundle:
            raise ValueError("Select a local verified result bundle")
        value = snapshot_from_bundle(allowed, request.bundle, request.frequency_index)
        reference = (
            snapshot_from_bundle(
                allowed, request.reference_bundle, request.frequency_index
            )
            if request.reference_bundle
            else None
        )
        if request.action == "analyze":
            return analyze_pressure(value, reference)
        if not request.destination:
            raise ValueError("Choose a new notebook bundle directory")
        destination = relative(request.destination)
        with directory(allowed / Path(destination).parent):
            pass
        # Snapshot inputs are bounded and contain no credentials or remote operations.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="fs-notebook-input-") as temp:
            first = Path(temp) / "pressure.json"
            first.write_bytes(canonical(value))
            second = None
            if reference:
                second = Path(temp) / "reference.json"
                second.write_bytes(canonical(reference))
            notebook = create_notebook(Path(temp) / "bundle", first, second)
            receipt = replay_notebook(notebook)
            with directory(allowed / Path(destination).parent) as fd:
                os.mkdir(Path(destination).name, mode=0o700, dir_fd=fd)
            # Publish only generated files through anchored no-overwrite operations.
            for file in sorted(
                notebook.parent.iterdir(), key=lambda item: item.name == "replay.json"
            ):
                write_local(allowed, destination + "/" + file.name, file.read_bytes())
        return {
            "relative_path": destination + "/analysis.ipynb",
            "report_path": destination + "/report.html",
            "replay_status": receipt["status"],
            "assurance": "locally rerun; no service attestation",
            "report_sha256": receipt.get("report_sha256"),
        }

    @server.tool(
        name="workflow_results",
        description="List or selectively download exact native results; analyze verified pressure snapshots; generate and cleanly rerun an editable analysis-only notebook. Keeps selected frequencies independent and never launches compute.",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
        structured_output=True,
    )
    async def results(request: ResultRequest) -> Reply:
        return await bounded(result_operation, request)

    @server.tool(
        name=REVIEW_TOOL,
        description="Preview an authoritative saved proposal, request structured user input through the MCP client's form, or recover local review evidence. Opt-in no-spend rehearsal only: never authorizes or launches paid work. No approval decision may be supplied in tool arguments. Retain the same intent on retry; client responses are not attested human approval.",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        structured_output=True,
        with_context=True,
    )
    async def native_review(
        request: ReviewRequest, context: ServerRequestContext[dict[str, Any]]
    ) -> Reply:
        from frequensolve.project_identity import ProjectIdentity

        from .review import ReviewEvidence, digest, proposal, request_input

        try:
            project = root(request.root_id) / relative(request.local_project)
            identity = ProjectIdentity.from_dict(
                json.loads(read_local(project, ".frequensolve/project-identity.json"))
            )
            arguments = {
                "project_key": identity.storage_prefix,
                "project_id": identity.project_id,
                "plan_id": request.plan_id,
            }

            async def fetch() -> dict:
                with anyio.fail_after(60):
                    value = await anyio.to_thread.run_sync(
                        lambda: client.call("plan_get", arguments),
                        limiter=limiter,
                        abandon_on_cancel=True,
                    )
                return proposal(value, identity.storage_prefix, request.plan_id)

            if request.action == "preview":
                snapshot = await fetch()
                result = {
                    "proposal": snapshot,
                    "proposal_fingerprint": digest(snapshot),
                    "mode": "rehearsal_only",
                    "paid_authority": False,
                }
            else:
                assert request.intent is not None
                journal = await anyio.to_thread.run_sync(client.journal, project)
                evidence = ReviewEvidence(journal)
                if request.action == "recover":
                    result = await anyio.to_thread.run_sync(
                        evidence.recover, request.intent
                    )
                else:
                    assert request.expected_fingerprint is not None
                    snapshot = await fetch()

                    async def elicit(message: str, schema: dict) -> dict:
                        capabilities = context.session.client_capabilities
                        if (
                            not capabilities
                            or not capabilities.elicitation
                            or capabilities.elicitation.form is None
                        ):
                            raise ValueError(
                                "This client does not advertise form elicitation"
                            )
                        response = await context.session.elicit_form(
                            message, schema, related_request_id=context.request_id
                        )
                        return response.model_dump(mode="json")

                    result = await request_input(
                        evidence,
                        request.intent,
                        snapshot,
                        request.expected_fingerprint,
                        fetch,
                        elicit,
                    )
            result = {
                **result,
                "review_url": cloud_url
                + "/agent-review?"
                + urlencode(
                    {"project_key": identity.storage_prefix, "plan_id": request.plan_id}
                ),
            }
            if len(canonical(result)) > 120000:
                raise ValueError("Review response exceeds its bounded limit")
            return Reply(ok=True, payload=canonical(result).decode())
        except Exception:
            return Reply(
                ok=False,
                diagnostic="Native review could not be confirmed. Preview the current proposal or recover the same local intent. Paid work still requires Cloud review.",
            )

    server.finalize_safety_contract(
        tool_names=tools, prompt_names=(PROMPT,), resource_uris=(RESOURCE,)
    )
    return server


def credit_scan(client: WorkflowClient, project: Path, arguments: dict) -> dict:
    """Resume bounded receipt pages; never total an incomplete scan."""
    credentials = client.auth.credentials()
    identity = hashlib.sha256(
        canonical(
            {
                **arguments,
                "subject": credentials["subject"],
                "account_id": credentials["account_id"],
                "resource": client.auth.connection.resource,
            }
        )
    ).hexdigest()
    store = PrivateStore(project / ".frequensolve" / "workflow")
    with store.locked("credits-" + identity) as record:
        saved = record.read() or {
            "cursor": None,
            "observations": {},
            "complete": False,
            "seen": [],
        }
        if saved["complete"]:
            # A new request refreshes current provider-settlement observations.
            saved = {"cursor": None, "observations": {}, "complete": False, "seen": []}
        for _ in range(5):
            response = client.call(
                "credit_history",
                {
                    **arguments,
                    **({"cursor": saved["cursor"]} if saved["cursor"] else {}),
                },
            )["credit_history"]
            for item in response["observations"]:
                old = saved["observations"].get(item["operation_id"])
                phases = ["quoted", "admitting", "queued", "settling", "settled"]
                if (
                    old
                    and old["phase"] == item["phase"] == "settled"
                    and old["captured_credits"] != item["captured_credits"]
                ):
                    raise ValueError("Settlement evidence conflicts")
                if old is None or (phases.index(item["phase"]), item["observed_at"]) > (
                    phases.index(old["phase"]),
                    old["observed_at"],
                ):
                    saved["observations"][item["operation_id"]] = item
            saved["cursor"] = response["cursor"]
            if saved["cursor"] is None:
                saved["complete"] = True
            elif saved["cursor"] in saved["seen"]:
                raise ValueError("Credit history repeated a cursor")
            else:
                saved["seen"].append(saved["cursor"])
            record.write(saved)
            if saved["complete"]:
                break
        observations = list(saved["observations"].values())
        return {
            "observations": observations,
            "history_complete": saved["complete"],
            "continue_scan": not saved["complete"],
            "captured_credits": (
                str(
                    sum(
                        (
                            Decimal(item["captured_credits"])
                            for item in observations
                            if item["captured_credits"] is not None
                        ),
                        Decimal(0),
                    )
                )
                if saved["complete"]
                else None
            ),
            "settlement_pending": any(
                item["reconciliation"] == "pending" for item in observations
            ),
            "evidence": "service settlement receipts; not an account balance delta or a fresh provider balance",
        }
