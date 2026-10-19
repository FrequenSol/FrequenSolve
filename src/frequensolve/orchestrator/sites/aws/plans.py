"""Authenticated saved-plan workflow; numerical sizing stays on compute workers."""

from __future__ import annotations

import json
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from frequensolve.execution_plan import PLAN_VERSION, PlanError, positive
from frequensolve.orchestrator.sites.base import JobStatus, RunHandle

if TYPE_CHECKING:
    from .aws import AWSSite
    from .plan_results import ExecutionResults


def sizing_policy(value: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(value or {})
    allowed = {
        "max_nodes_per_frequency",
        "max_concurrent_frequencies",
        "max_running_cpus",
        "max_wall_time_seconds",
        "max_layout_evaluations",
    }
    if policy.keys() - allowed:
        raise PlanError("Unsupported automatic sizing settings")
    policy.setdefault("max_layout_evaluations", 8)
    for key, number in policy.items():
        positive(number, key)
    if policy["max_layout_evaluations"] > 8:
        raise PlanError(
            "At most eight layout evaluations may be authorized per operation"
        )
    if "max_wall_time_seconds" in policy and (
        policy["max_wall_time_seconds"] < 60 or policy["max_wall_time_seconds"] % 60
    ):
        raise PlanError("Slurm time ceilings must use whole-minute increments")
    return policy


def request(client: Any, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    query = "mutation Plan($input: AWSJSON!) { requestMyExecutionPlan(input: $input) }"
    response = client.execute(
        query, {"input": json.dumps({"action": action, **payload})}
    )
    result = response.get("requestMyExecutionPlan")
    # AWSJSON may retain the Lambda's JSON string as one encoded JSON value.
    # Accept the direct and deployed envelopes, with no unbounded decoding.
    for _ in range(2):
        if not isinstance(result, str):
            break
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            raise PlanError(
                "Cloud returned an invalid execution-plan response"
            ) from None
    if not isinstance(result, dict):
        raise PlanError("Cloud returned an invalid execution-plan response")
    if result.get("error"):
        raise PlanError(str(result.get("message") or result["error"]))
    if result.get("plan") is not None and result["plan"].get("version") != PLAN_VERSION:
        raise PlanError(
            "Execution plan version is unsupported; use a compatible client"
        )
    return result


def check_quote(quote: dict[str, Any], maximum_credits: str) -> None:
    if not isinstance(maximum_credits, str):
        raise PlanError("maximum_credits must be an explicit decimal string")
    try:
        maximum, quoted = Decimal(maximum_credits), Decimal(quote["maximum_credits"])
    except (InvalidOperation, KeyError, TypeError, ValueError):
        raise PlanError(
            "A finite Credit authorization and valid quote are required"
        ) from None
    if not maximum.is_finite() or not quoted.is_finite() or maximum < 0 or quoted < 0:
        raise PlanError("Invalid Credit authorization")
    if quoted > maximum:
        raise PlanError(
            f"Quoted maximum ({quoted} Credits) exceeds your authorization ({maximum})"
        )
    if not quote.get("fingerprint"):
        raise PlanError("Credit quote has no fingerprint")


@dataclass
class PlanHandle:
    """A saved planning operation and its immutable reviewed revisions.

    Quotes may be inspected without starting compute. Authorization is bounded by
    an explicit Credit maximum, never by an unattended interactive prompt.
    """

    site: AWSSite
    data: dict[str, Any]
    job: Any = None
    _keys: dict[str, str] = field(default_factory=dict, repr=False)
    _execute_requests: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    _pending_planning_maximum: str | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return self.data["head"]["plan_id"]

    @property
    def revision(self) -> int:
        return self.data["head"]["revision"]

    @property
    def status(self) -> str:
        return self.data["head"]["status"]

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.data.get("plan", {}).get("rows", [])

    @property
    def progress(self) -> dict[str, Any] | None:
        """Last observed controller diagnostics; refresh explicitly for new data.

        Status cannot establish allocation start, trusted sizing or settled usage.
        Older compatible servers may return no progress snapshot.
        """
        value = self.data.get("progress")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise PlanError("Cloud returned invalid planning progress")
        counts = value.get("counts")
        fields = {
            "ready",
            "intent",
            "uncertain",
            "submitted",
            "completed",
            "failed",
            "canceled",
            "not_started",
        }
        if (
            value.get("purpose") not in {"planning", "execution"}
            or not isinstance(counts, dict)
            or counts.keys() != fields
            or any(type(number) is not int or number < 0 for number in counts.values())
            or type(value.get("total")) is not int
            or not 1 <= value["total"] <= (8 if value["purpose"] == "planning" else 96)
            or sum(counts.values()) != value["total"]
            or type(value.get("completed")) is not int
            or value["completed"]
            != sum(
                counts[name]
                for name in ("completed", "failed", "canceled", "not_started")
            )
            or not isinstance(value.get("message"), str)
            or not isinstance(value.get("observed_at"), str)
            or not isinstance(value.get("stage"), str)
        ):
            raise PlanError("Cloud returned incomplete planning progress")
        return deepcopy(value)

    def _call(self, action: str, **values: Any) -> dict[str, Any]:
        if self.data.get("historical") and action != "get":
            raise PlanError(
                "This saved revision is read-only; open the current plan before editing or executing"
            )
        payload = {
            "plan_id": self.id,
            "project_key": self.data["head"]["owner"]["project_key"],
            **(
                {"project_id": self.data["head"]["owner"]["project_id"]}
                if "project_id" in self.data["head"]["owner"]
                else {}
            ),
            **values,
        }
        result = request(self.site.graphql_client, action, payload)
        if "head" in result:
            head = result["head"]
            if (
                not isinstance(head, dict)
                or head.get("plan_id") != self.id
                or not isinstance(head.get("owner"), dict)
                or head["owner"].get("project_key")
                != self.data["head"]["owner"]["project_key"]
                or head["owner"].get("project_id")
                != self.data["head"]["owner"].get("project_id")
            ):
                raise PlanError(
                    "Cloud returned a different saved plan identity; refresh the original plan"
                )
            if "revision" in values and (
                head.get("revision") != values["revision"]
                or head.get("status") != "historical"
                or result.get("historical", {}).get("revision") != values["revision"]
            ):
                raise PlanError(
                    "Saved-revision history requires a compatible Cloud response"
                )
            self.data = result
        return result

    def refresh(self) -> "PlanHandle":
        self._call(
            "get",
            **({"revision": self.revision} if self.data.get("historical") else {}),
        )
        if self.status in {"canceled", "failed"}:
            self._pending_planning_maximum = None
        elif (
            self._pending_planning_maximum is not None
            and self.status == "awaiting_authorization"
            and self.revision == 0
            and self.data.get("quote", {}).get("purpose") in {"planning", "automatic"}
        ):
            self.authorize(self._pending_planning_maximum)
        return self

    def authorize(self, maximum_credits: str) -> "PlanHandle":
        if self.status == "preparing":
            # Remember the user's bound while free source copying runs. This
            # creates no hold and authorizes no unquoted compute. Refresh will
            # check the actual planning quote against the SAME bound first.
            check_quote(
                {"maximum_credits": "0", "fingerprint": "preparation-only"},
                maximum_credits,
            )
            self._pending_planning_maximum = maximum_credits
            return self
        quote = self.data.get("quote")
        if not isinstance(quote, dict):
            raise PlanError("Refresh the planning quote before authorizing compute")
        check_quote(quote, maximum_credits)
        self._call(
            "authorize",
            expected_revision=self.revision,
            idempotency_key=self._keys.setdefault(
                f"authorize-{quote['fingerprint']}", uuid.uuid4().hex
            ),
            accepted_quote_fingerprint=quote["fingerprint"],
            maximum_credits=maximum_credits,
        )
        self._pending_planning_maximum = None
        return self

    def wait(
        self, timeout: float | None = None, poll_interval: float = 5
    ) -> "PlanHandle":
        if poll_interval <= 0 or (timeout is not None and timeout <= 0):
            raise ValueError("Wait intervals must be positive")
        started = time.monotonic()
        while True:
            self.refresh()
            if self.status in {
                "review",
                "incomplete",
                "failed",
                "canceled",
                "executing",
                "executed",
                "historical",
            }:
                return self
            if self.status == "awaiting_authorization":
                raise PlanError("Planning needs a bounded Credit authorization")
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(
                    "Planning is still in progress; the saved operation was not canceled"
                )
            time.sleep(
                min(poll_interval, max(0, timeout - (time.monotonic() - started)))
                if timeout
                else poll_interval
            )

    def edit(
        self, edits: list[dict[str, Any]], *, idempotency_key: str | None = None
    ) -> "PlanHandle":
        self._call(
            "revise",
            expected_revision=self.revision,
            edits=edits,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
        )
        return self

    def validate(self, maximum_credits: str | None = None) -> "PlanHandle":
        if (
            self.status == "awaiting_authorization"
            and self.data.get("quote", {}).get("purpose") == "validation"
        ):
            return (
                self.authorize(maximum_credits) if maximum_credits is not None else self
            )
        result = self._call(
            "validate",
            expected_revision=self.revision,
            idempotency_key=self._keys.setdefault(
                f"validate-{self.revision}", uuid.uuid4().hex
            ),
        )
        if result.get("requires_compute") and maximum_credits is not None:
            self.authorize(maximum_credits)
        return self

    def results(self, *, execution_id: str | None = None) -> ExecutionResults:
        """Inspect one settled execution without changing this plan or running work."""
        from .plan_results import ExecutionResults

        run = execution_id or self.data["head"].get("execution_id")
        if not isinstance(run, str):
            raise PlanError("Choose a completed saved execution before reading results")
        response = request(
            self.site.graphql_client,
            "results",
            {
                "plan_id": self.id,
                "project_key": self.data["head"]["owner"]["project_key"],
                **(
                    {"project_id": self.data["head"]["owner"]["project_id"]}
                    if "project_id" in self.data["head"]["owner"]
                    else {}
                ),
                "execution_id": run,
            },
        )
        return ExecutionResults(
            self.site,
            response,
            plan_id=self.id,
            execution_id=run,
            project=self.data["head"]["owner"]["project_key"],
        )

    def quote_execution(self) -> dict[str, Any]:
        """Inspect a fresh execution quote without reserving Credits or compute."""
        result = self._call("quote_execution", expected_revision=self.revision)
        quote = result.get("quote")
        if (
            not isinstance(quote, dict)
            or quote.get("purpose", "execution") != "execution"
        ):
            raise PlanError("Cloud returned no execution quote")
        maximum = quote.get("maximum_credits")
        if not isinstance(maximum, str):
            raise PlanError("maximum_credits must be an explicit decimal string")
        check_quote(quote, maximum)
        return deepcopy(quote)

    def execute(
        self, maximum_credits: str, *, idempotency_key: str | None = None
    ) -> RunHandle:
        return self.site.execute_plan(
            self, maximum_credits=maximum_credits, idempotency_key=idempotency_key
        )

    def cancel(self) -> "PlanHandle":
        execution_id = self.data["head"].get("execution_id")
        if self.status == "executing" and not execution_id:
            raise PlanError("Refresh the accepted execution identity before canceling")
        self._call(
            "cancel",
            expected_revision=self.revision,
            idempotency_key=self._keys.setdefault(
                f"cancel-{self.revision}-{execution_id or self.data['head'].get('operation_id', '')}",
                uuid.uuid4().hex,
            ),
            **(
                {"expected_execution_id": execution_id}
                if self.status == "executing"
                else {}
            ),
        )
        return self

    def export(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        return target


class PlanExecutionPaused(PlanError):
    """Automatic work paused with a retained plan, never a fabricated failed run."""

    def __init__(self, plan: PlanHandle) -> None:
        self.plan = plan
        reason = plan.data.get("automatic", {}).get("message")
        super().__init__(
            f"Plan {plan.id} is saved with status {plan.status}. "
            + (
                reason
                or "Review its validation and Credit quote before continuing; no frequency execution was started."
            )
        )


class AutomaticRunHandle(RunHandle):
    """One awaitable handle across planning and the later real execution identity.

    id remains None until Cloud publishes an actual Simulation id. Polling never
    accepts another quote or submits a solver; PlanHandle only retains the user's
    already supplied bound for the initial paid planning operation.
    """

    def __init__(self, plan: PlanHandle) -> None:
        self.plan = plan

        def cancel_planning(_run: RunHandle) -> None:
            plan.cancel()

        super().__init__(
            site=plan.site,
            job=plan.job,
            mode="aws",
            check=False,
            backend={"planId": plan.id, "planRevision": plan.revision},
            _status_fn=self._planning_status,
            _cancel_fn=cancel_planning,
            _logs_fn=self._planning_logs,
        )

    def _planning_logs(self, _run: RunHandle, **_options: Any) -> NoReturn:
        raise PlanError(
            f"Plan {self.plan.id} has no execution logs yet; inspect the saved planning progress"
        )

    def fetch(self) -> Any:
        if self.id is None:
            raise PlanError(
                f"Plan {self.plan.id} has not published an execution; no solver outputs are available"
            )
        return super().fetch()

    def _planning_status(self, _run: RunHandle) -> JobStatus:
        self.plan.refresh()
        self.backend["planRevision"] = self.plan.revision
        run_id = self.plan.data["head"].get("execution_id")
        if run_id:
            accepted = cast("AWSSite", self.site)._run_handle_from_plan(
                self.plan, run_id
            )
            # Adopt the ordinary Cloud run behavior only once the real identity
            # exists. Preserve this object's identity for wait/watch/cancel users.
            for name in (
                "id",
                "job",
                "mode",
                "backend",
                "_logs_fn",
                "_status_fn",
                "_cancel_fn",
                "_pending_fetch_fn",
                "_finalize_fn",
                "_wait_fn",
                "_wait_async_fn",
                "_timeout_fn",
                "_generic_wait",
            ):
                setattr(self, name, getattr(accepted, name))
            return self.status()
        if (
            self.plan.status == "review"
            and self.plan.data.get("automatic", {}).get("phase") == "awaiting_execution"
        ):
            return JobStatus(
                state="pending",
                message=f"Plan {self.plan.id} is validated; awaiting execution admission",
                raw={
                    "stage": "execution_admission",
                    "plan_id": self.plan.id,
                    "revision": self.plan.revision,
                },
            )
        if self.plan.status in {"review", "incomplete", "awaiting_authorization"}:
            raise PlanExecutionPaused(self.plan)
        if self.plan.status in {"failed", "canceled"}:
            return JobStatus(
                state=self.plan.status,
                return_code=1,
                message=f"Planning {self.plan.status}; saved plan {self.plan.id} retains its evidence and charges",
                raw={
                    "stage": "planning",
                    "plan_id": self.plan.id,
                    "revision": self.plan.revision,
                },
            )
        progress = self.plan.progress
        return JobStatus(
            state="pending" if self.plan.status == "preparing" else "running",
            message=(
                f"{progress['message']}; {progress['completed']} of {progress['total']} evaluations closed"
                if progress
                else f"{self.plan.status.capitalize()} resources for plan {self.plan.id}"
            ),
            raw={
                "stage": "planning",
                "plan_id": self.plan.id,
                "revision": self.plan.revision,
                **({"progress": progress} if progress else {}),
            },
        )


class ExecutionPlanningMixin:
    """Cloud plan operations share authentication and input staging with AWSSite."""

    @property
    def _planning_site(self) -> AWSSite:
        # This mixin belongs to AWSSite; keep its host dependencies explicit.
        return cast("AWSSite", self)

    def plan(
        self,
        job: Any,
        *,
        policy: dict[str, Any] | None = None,
        maximum_credits: str | None = None,
        run_after_planning: bool = False,
        idempotency_key: str | None = None,
        **compute_options: Any,
    ) -> PlanHandle:
        from .execution_profile import resolve_compute_options

        if type(run_after_planning) is not bool:
            raise PlanError("run_after_planning must be a boolean")
        policy = sizing_policy(policy)
        if maximum_credits is not None:
            check_quote(
                {"maximum_credits": "0", "fingerprint": "unquoted"}, maximum_credits
            )
        if getattr(self, "_explicit_execution_resources", False):
            raise PlanError(
                "Automatic sizing cannot silently replace explicit execution_resources. Use a named profile without fixed nodes/mpi_ranks settings."
            )
        selectors = resolve_compute_options(
            self._planning_site.execution_profile.graphql_arguments(), compute_options
        )
        self._planning_site.prepare_job(job, sync_project=True, validate=True)
        identity = self._planning_site._ensure_project_identity(
            job.project_path,
            self._planning_site._authored_project_metadata(job)[1]
            or self._planning_site._authored_project_metadata(job)[0]
            or job.project_path.name,
        )
        project = self._planning_site.project_storage_key(job.project_path)
        if getattr(job.simulation, "_project", None) is None:
            self._planning_site._sync_loaded_job_inputs(job, project)
        self._planning_site._prepare_run_snapshot(job)
        local_job, remote_job = job.save_for_remote(
            self.__class__.__name__, project, include_job_id=False
        )
        key = self._planning_site.sync_s3(local_job, remote_job)
        local_simulation, remote_simulation = job.save_simulation_for_remote(
            self.__class__.__name__, project
        )
        self._planning_site.sync_s3(local_simulation, remote_simulation)
        # Retain interpretation AFTER staging, before the asynchronous request.
        # Later authoring/staging edits must not replace these scientific bytes.
        frozen_job = self._planning_site._prepare_run_snapshot(job)
        frozen_job._cloud_saved_plan_inputs = True
        result = request(
            self._planning_site.graphql_client,
            "create",
            {
                "idempotency_key": idempotency_key or uuid.uuid4().hex,
                "job_file_key": str(key),
                **({"project_id": identity.project_id} if identity else {}),
                "policy": policy,
                "run_after_planning": run_after_planning,
                **{
                    name: selectors[name]
                    for name in ("cluster_id", "partition", "allow_unverified_compute")
                    if name in selectors
                },
            },
        )
        if identity is not None and (
            result.get("head", {}).get("owner", {}).get("project_id")
            != identity.project_id
            or result.get("head", {}).get("owner", {}).get("project_key")
            != identity.storage_prefix
        ):
            raise PlanError(
                "Cloud returned a different immutable project for the saved plan"
            )
        handle = PlanHandle(self._planning_site, result, frozen_job)
        if maximum_credits is not None:
            handle.authorize(maximum_credits)
        return handle

    def submit_automatically(self, job: Any, **options: Any) -> RunHandle:
        """Submit through the same saved-plan stages with one bounded authorization.

        This opt-in policy does not reinterpret legacy explicit MPI topology.
        Cloud pauses the saved plan if execution needs additional authorization.
        """
        allowed = {
            "sizing_policy",
            "maximum_credits",
            "cluster_id",
            "partition",
            "allow_unverified_compute",
        }
        if set(options) - allowed:
            raise PlanError(
                "Unsupported automatic submission options: "
                + ", ".join(sorted(set(options) - allowed))
            )
        if "maximum_credits" not in options:
            raise PlanError(
                "Automatic submission requires a bounded maximum_credits authorization; use site.plan(job) to inspect a planning quote first"
            )
        handle = self.plan(
            job,
            policy=options.pop("sizing_policy", None),
            maximum_credits=options.pop("maximum_credits"),
            run_after_planning=True,
            **options,
        )
        run_id = handle.data["head"].get("execution_id")
        if not run_id:
            return AutomaticRunHandle(handle)
        return self._run_handle_from_plan(handle, run_id)

    def _run_handle_from_plan(self, plan: PlanHandle, run_id: str) -> RunHandle:
        source = getattr(plan, "job", None)
        if source is not None:
            # Opened saved plans may also attach the original authored job. The
            # pinned descriptor fingerprint remains unchanged by run scoping.
            prepare = getattr(self, "_prepare_run_snapshot", None)
            if callable(prepare):
                source = prepare(source)
                source._cloud_saved_plan_inputs = True
            handle = self._planning_site._make_run_handle(
                source,
                run_id,
                backend={"planId": plan.id, "planRevision": plan.revision},
            )
        else:
            handle = RunHandle(
                site=self._planning_site,
                job=None,
                id=run_id,
                mode="aws",
                backend={"planId": plan.id, "planRevision": plan.revision},
                _status_fn=self._planning_site._poll_run,
            )
        # Bind cancellation to THIS accepted run and revision. Neither a mutable
        # authoring job nor later reuse of the user's PlanHandle can redirect it.
        accepted = PlanHandle(self._planning_site, deepcopy(plan.data), source)

        def cancel_accepted(_run: RunHandle) -> None:
            accepted.cancel()

        handle._cancel_fn = cancel_accepted
        return handle

    def list_plans(
        self, *, project_key: str, cursor: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Read one page of owned plan/preparation summaries without paid work.

        Pass the returned opaque cursor unchanged to fetch the next page. Open
        an item with ``open_plan`` to inspect its complete current revision.
        """
        if (
            not isinstance(project_key, str)
            or not 1 <= len(project_key) <= 1024
            or any(ord(char) < 32 or ord(char) == 127 for char in project_key)
        ):
            raise PlanError("Invalid saved-plan project key")
        if cursor is not None:
            try:
                valid_cursor = (
                    isinstance(cursor, dict)
                    and set(cursor) == {"phase", "key"}
                    and cursor["phase"] in {"preparations", "plans"}
                    and isinstance(cursor["key"], dict)
                    and set(cursor["key"]) == {"scope", "id"}
                    and all(
                        isinstance(value, str) and 0 < len(value) <= 1024
                        for value in cursor["key"].values()
                    )
                    and len(json.dumps(cursor).encode()) <= 4096
                )
            except (TypeError, ValueError):
                valid_cursor = False
            if not valid_cursor:
                raise PlanError("Invalid saved-plan page cursor")
        result = request(
            self._planning_site.graphql_client,
            "list",
            {
                "project_key": project_key,
                **({"cursor": deepcopy(cursor)} if cursor is not None else {}),
            },
        )
        items = result.get("items")
        if not isinstance(items, list) or len(items) > 150:
            raise PlanError("Cloud returned an invalid saved-plan page")
        for head in items:
            if (
                not isinstance(head, dict)
                or not isinstance(head.get("owner"), dict)
                or head["owner"].get("project_key") != project_key
                or not isinstance(head.get("plan_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", head["plan_id"])
                or type(head.get("revision")) is not int
                or not 0 <= head["revision"] <= 2**53 - 1
                or not isinstance(head.get("status"), str)
            ):
                raise PlanError(
                    "Cloud returned another project or invalid plan identity"
                )
        return deepcopy(result)

    def open_plan(
        self,
        plan_id: str,
        *,
        project_key: str,
        job: Any = None,
        revision: int | None = None,
    ) -> PlanHandle:
        if revision is not None:
            positive(revision, "saved revision")
        result = request(
            self._planning_site.graphql_client,
            "get",
            {
                "plan_id": plan_id,
                "project_key": project_key,
                **({"revision": revision} if revision is not None else {}),
            },
        )
        head = result.get("head", {})
        if (
            head.get("plan_id") != plan_id
            or head.get("owner", {}).get("project_key") != project_key
        ):
            raise PlanError("Cloud returned a different saved plan identity")
        if revision is not None and (
            head.get("revision") != revision
            or head.get("status") != "historical"
            or result.get("historical", {}).get("revision") != revision
        ):
            raise PlanError(
                "Saved-revision history requires a compatible Cloud response"
            )
        return PlanHandle(self._planning_site, result, job)

    def execute_plan(
        self,
        plan: PlanHandle,
        *,
        maximum_credits: str,
        idempotency_key: str | None = None,
    ) -> RunHandle:
        # No new key means recover the accepted run. A deliberately new key on
        # a completed plan requests a fresh quote/authorization and new identity;
        # it never retries an accepted failed solver job automatically.
        if plan.site is not self:
            raise PlanError("Open this plan using the intended authenticated site")
        if plan.data.get("historical"):
            raise PlanError(
                "This saved revision is read-only; open the current plan before executing"
            )
        key = idempotency_key or plan._keys.setdefault(
            f"execute-{plan.revision}", uuid.uuid4().hex
        )
        cached = plan._execute_requests.get(key)
        if cached:
            if cached["maximum_credits"] != maximum_credits:
                raise PlanError(
                    "A retried execution must reuse the same authorization; inspect the accepted operation first"
                )
            result = plan._call("execute", **cached)
        elif (
            plan.status in {"executing", "executed"}
            and plan.data["head"].get("execution_id")
            and (idempotency_key is None or plan.status == "executing")
        ):
            if idempotency_key is not None:
                raise PlanError(
                    "This plan already has an active execution; wait for physical and financial completion before an intentional rerun"
                )
            result = plan.data
        else:
            quote = plan.quote_execution()
            check_quote(quote, maximum_credits)
            payload = {
                "expected_revision": plan.revision,
                "accepted_quote_fingerprint": quote["fingerprint"],
                "maximum_credits": maximum_credits,
                "idempotency_key": key,
            }
            # Retain BEFORE the transport call. A lost response must replay the
            # same execution intent, never request a new quote or solver run.
            plan._execute_requests[key] = payload
            result = plan._call("execute", **payload)
        run_id = result["head"]["execution_id"]
        return self._run_handle_from_plan(plan, run_id)
