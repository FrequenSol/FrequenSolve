"""Authenticated saved-plan workflow; numerical sizing stays on compute workers."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from frequensolve.execution_plan import PLAN_VERSION, PlanError, positive


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
    if isinstance(result, str):
        result = json.loads(result)
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

    site: Any
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

    def _call(self, action: str, **values: Any) -> dict[str, Any]:
        payload = {
            "plan_id": self.id,
            "project_key": self.data["head"]["owner"]["project_key"],
            **values,
        }
        result = request(self.site.graphql_client, action, payload)
        if "head" in result:
            self.data = result
        return result

    def refresh(self) -> "PlanHandle":
        self._call("get")
        if self.status in {"canceled", "failed"}:
            self._pending_planning_maximum = None
        elif (
            self._pending_planning_maximum is not None
            and self.status == "awaiting_authorization"
            and self.revision == 0
            and self.data.get("quote", {}).get("purpose") == "planning"
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

    def execute(self, maximum_credits: str, *, idempotency_key: str | None = None):
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


class ExecutionPlanningMixin:
    """Cloud plan operations share authentication and input staging with AWSSite."""

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
            self.execution_profile.graphql_arguments(), compute_options
        )
        self.prepare_job(job, sync_project=True, validate=True)
        project = job.project_path.name
        if getattr(job.simulation, "_project", None) is None:
            self._sync_loaded_job_inputs(job, project)
        self._prepare_run_snapshot(job)
        local_job, remote_job = job.save_for_remote(
            self.__class__.__name__, project, include_job_id=False
        )
        key = self.sync_s3(local_job, remote_job)
        local_simulation, remote_simulation = job.save_simulation_for_remote(
            self.__class__.__name__, project
        )
        self.sync_s3(local_simulation, remote_simulation)
        result = request(
            self.graphql_client,
            "create",
            {
                "idempotency_key": idempotency_key or uuid.uuid4().hex,
                "job_file_key": str(key),
                "policy": policy,
                "run_after_planning": run_after_planning,
                **{
                    name: selectors[name]
                    for name in ("cluster_id", "partition", "allow_unverified_compute")
                    if name in selectors
                },
            },
        )
        handle = PlanHandle(self, result, job)
        if maximum_credits is not None:
            handle.authorize(maximum_credits)
        return handle

    def submit_automatically(self, job: Any, **options: Any):
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
            policy=options.pop("sizing_policy"),
            maximum_credits=options.pop("maximum_credits"),
            run_after_planning=True,
            **options,
        )
        run_id = handle.data["head"].get("execution_id")
        if not run_id:
            raise PlanError(
                f"Plan {handle.id} is saved with status {handle.status}; inspect it before execution"
            )
        return self._make_run_handle(
            job, run_id, backend={"planId": handle.id, "planRevision": handle.revision}
        )

    def open_plan(
        self, plan_id: str, *, project_key: str, job: Any = None
    ) -> PlanHandle:
        result = request(
            self.graphql_client, "get", {"plan_id": plan_id, "project_key": project_key}
        )
        return PlanHandle(self, result, job)

    def execute_plan(
        self,
        plan: PlanHandle,
        *,
        maximum_credits: str,
        idempotency_key: str | None = None,
    ):
        from frequensolve.orchestrator.sites.base import RunHandle

        if plan.site is not self:
            raise PlanError("Open this plan using the intended authenticated site")
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
        elif plan.status in {"executing", "executed"} and plan.data["head"].get(
            "execution_id"
        ):
            if idempotency_key is not None:
                raise PlanError(
                    "This plan already has an execution; create a new plan revision for an intentional rerun"
                )
            result = plan.data
        else:
            quote = plan._call("quote_execution", expected_revision=plan.revision)[
                "quote"
            ]
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
        if plan.job is not None:
            return self._make_run_handle(
                plan.job,
                run_id,
                backend={"planId": plan.id, "planRevision": plan.revision},
            )
        return RunHandle(
            site=self,
            job=None,
            id=run_id,
            mode="aws",
            backend={"planId": plan.id, "planRevision": plan.revision},
            _status_fn=self._poll_run,
            _cancel_fn=lambda run: self.cancel_job(str(run.id)),
        )
