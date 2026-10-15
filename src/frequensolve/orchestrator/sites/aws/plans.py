"""Authenticated saved-plan workflow; numerical sizing stays on compute workers."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from frequensolve.execution_plan import PlanError, positive


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
    return result


def check_quote(quote: dict[str, Any], maximum_credits: str) -> None:
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
        return self

    def authorize(self, maximum_credits: str) -> "PlanHandle":
        quote = self.data.get("quote")
        if not isinstance(quote, dict):
            raise PlanError("Refresh the planning quote before authorizing compute")
        check_quote(quote, maximum_credits)
        self._call(
            "authorize",
            expected_revision=self.revision,
            idempotency_key=self._keys.setdefault("authorize", uuid.uuid4().hex),
            accepted_quote_fingerprint=quote["fingerprint"],
            maximum_credits=maximum_credits,
        )
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
        self._call(
            "cancel",
            expected_revision=self.revision,
            idempotency_key=self._keys.setdefault("cancel", uuid.uuid4().hex),
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
    ) -> PlanHandle:
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
        selectors = self.execution_profile.graphql_arguments()
        result = request(
            self.graphql_client,
            "create",
            {
                "idempotency_key": idempotency_key or uuid.uuid4().hex,
                "job_file_key": str(key),
                "policy": sizing_policy(policy),
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
        quote = plan._call("quote_execution", expected_revision=plan.revision)["quote"]
        check_quote(quote, maximum_credits)
        result = plan._call(
            "execute",
            expected_revision=plan.revision,
            accepted_quote_fingerprint=quote["fingerprint"],
            maximum_credits=maximum_credits,
            idempotency_key=idempotency_key
            or plan._keys.setdefault(f"execute-{plan.revision}", uuid.uuid4().hex),
        )
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
