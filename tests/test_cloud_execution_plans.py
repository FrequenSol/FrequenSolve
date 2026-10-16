"""Saved-plan requests authorize bounded work and preserve retry identities."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from frequensolve.execution_plan import PlanError
from frequensolve.orchestrator.sites.aws.plans import (
    AutomaticRunHandle,
    ExecutionPlanningMixin,
    PlanExecutionPaused,
    PlanHandle,
    check_quote,
    sizing_policy,
)


class Client:
    def __init__(self):
        self.calls = []
        self.response = None

    def execute(self, _query, values):
        self.calls.append(json.loads(values["input"]))
        return {"requestMyExecutionPlan": json.dumps(self.response)}


def handle():
    client = Client()
    data = {
        "head": {
            "plan_id": "fake-plan",
            "revision": 1,
            "status": "awaiting_authorization",
            "owner": {"project_key": "fake-project"},
        },
        "quote": {"maximum_credits": "12.50", "fingerprint": "fake-fingerprint"},
    }
    client.response = deepcopy(data)
    return PlanHandle(SimpleNamespace(graphql_client=client), data), client


def test_quote_can_be_inspected_without_authorizing_compute():
    plan, client = handle()
    assert plan.status == "awaiting_authorization"
    assert client.calls == []
    with pytest.raises(PlanError, match="exceeds"):
        plan.authorize("12.49")
    assert client.calls == []


def test_source_preparation_defers_authorization_until_a_matching_quote_exists():
    plan, client = handle()
    plan.data = {"head": {**plan.data["head"], "revision": 0, "status": "preparing"}}
    client.response = deepcopy(plan.data)
    assert plan.authorize("20") is plan
    assert client.calls == []
    plan.refresh()
    assert [call["action"] for call in client.calls] == ["get"]
    client.response = {
        "head": {**plan.data["head"], "status": "awaiting_authorization"},
        "quote": {
            "purpose": "planning",
            "maximum_credits": "12.50",
            "fingerprint": "fake-fingerprint",
        },
    }
    plan.refresh()
    assert [call["action"] for call in client.calls] == ["get", "get", "authorize"]
    assert client.calls[-1]["maximum_credits"] == "20"
    plan.refresh()
    assert client.calls[-1]["action"] == "get"


def test_preparation_cannot_expand_authorization_or_authorize_validation():
    plan, client = handle()
    plan.data = {"head": {**plan.data["head"], "revision": 0, "status": "preparing"}}
    with pytest.raises(PlanError):
        plan.authorize("Infinity")
    plan.authorize("1")
    client.response = {
        "head": {**plan.data["head"], "status": "awaiting_authorization"},
        "quote": {
            "purpose": "planning",
            "maximum_credits": "12.50",
            "fingerprint": "fake-fingerprint",
        },
    }
    with pytest.raises(PlanError, match="exceeds"):
        plan.refresh()
    assert [call["action"] for call in client.calls] == ["get"]
    client.response["quote"]["purpose"] = "validation"
    plan.refresh()
    assert [call["action"] for call in client.calls] == ["get", "get"]


def test_authorization_is_bound_to_quote_and_reuses_request_identity():
    plan, client = handle()
    plan.authorize("20")
    plan.authorize("20")
    assert client.calls[0] == client.calls[1]
    assert client.calls[0]["accepted_quote_fingerprint"] == "fake-fingerprint"
    assert client.calls[0]["expected_revision"] == 1


@pytest.mark.parametrize("maximum", ["NaN", "Infinity", "-1", "unknown"])
def test_nonfinite_and_invalid_authorizations_are_rejected(maximum):
    plan, client = handle()
    with pytest.raises(PlanError):
        plan.authorize(maximum)
    assert not client.calls


def test_validation_does_not_silently_start_paid_compute():
    plan, client = handle()
    client.response = {**plan.data, "requires_compute": True}
    plan.validate()
    assert [call["action"] for call in client.calls] == ["validate"]
    plan.validate(maximum_credits="20")
    assert [call["action"] for call in client.calls] == [
        "validate",
        "validate",
        "authorize",
    ]


def test_plan_edits_cannot_change_revision_without_server_response():
    plan, client = handle()
    client.response = {
        **plan.data,
        "head": {**plan.data["head"], "revision": 2, "status": "review"},
    }
    plan.edit(
        [{"frequency_index": 0, "nodes": 2, "mpi_ranks": 2}],
        idempotency_key="fake-edit",
    )
    assert plan.revision == 2
    assert client.calls[0]["expected_revision"] == 1
    assert client.calls[0]["idempotency_key"] == "fake-edit"


def test_exports_metadata_without_credentials(tmp_path):
    plan, _ = handle()
    output = plan.export(tmp_path / "plan.json")
    assert json.loads(output.read_text()) == plan.data


@pytest.mark.parametrize("foreign", ["plan", "project"])
def test_status_response_cannot_retarget_a_saved_plan_handle(foreign):
    plan, client = handle()
    retained = deepcopy(plan.data)
    if foreign == "plan":
        client.response["head"]["plan_id"] = "another-plan"
    else:
        client.response["head"]["owner"]["project_key"] = "another-project"
    with pytest.raises(PlanError, match="different saved plan"):
        plan.refresh()
    assert plan.data == retained
    assert client.calls[-1]["plan_id"] == "fake-plan"
    assert client.calls[-1]["project_key"] == "fake-project"


def test_wait_does_not_execute_a_ready_plan():
    plan, client = handle()
    client.response["head"]["status"] = "review"
    assert plan.wait(timeout=1).status == "review"
    assert [call["action"] for call in client.calls] == ["get"]


def test_execution_cancellation_pins_the_run_and_reuses_the_exact_request():
    plan, client = handle()
    plan.data["head"].update(status="executing", execution_id="fake-run")
    client.response = deepcopy(plan.data)
    plan.cancel()
    plan.cancel()
    assert client.calls[0] == client.calls[1]
    assert client.calls[0]["expected_execution_id"] == "fake-run"
    plan.data["head"].pop("execution_id")
    with pytest.raises(PlanError, match="execution identity"):
        plan.cancel()
    assert len(client.calls) == 2


def test_policy_limits_are_optional_but_bounded_and_typed():
    assert sizing_policy() == {"max_layout_evaluations": 8}
    assert (
        sizing_policy({"max_nodes_per_frequency": 16})["max_nodes_per_frequency"] == 16
    )
    for policy in (
        {"nodes": 2},
        {"max_layout_evaluations": 9},
        {"max_running_cpus": True},
    ):
        with pytest.raises(PlanError):
            sizing_policy(policy)
    with pytest.raises(PlanError):
        check_quote({"maximum_credits": "2"}, "3")


def test_invalid_planning_policy_and_fixed_topology_are_rejected_before_staging():
    class Site(ExecutionPlanningMixin):
        _explicit_execution_resources = True

        def prepare_job(self, *args, **kwargs):
            raise AssertionError("Invalid policy must not stage inputs")

    site = Site()
    with pytest.raises(PlanError, match="fixed nodes"):
        site.plan(object())
    with pytest.raises(PlanError, match="eight"):
        site.plan(object(), policy={"max_layout_evaluations": 9})
    with pytest.raises(PlanError, match="boolean"):
        site.plan(object(), run_after_planning="yes")
    with pytest.raises(PlanError, match="maximum_credits"):
        site.submit_automatically(object(), sizing_policy={})


def test_automatic_submission_uses_saved_planning_and_one_bounded_authorization():
    class Site(ExecutionPlanningMixin):
        def plan(self, job, **kwargs):
            self.options = kwargs
            return SimpleNamespace(
                id="fake-plan",
                revision=0,
                status="planning",
                data={"head": {"execution_id": "fake-run"}},
                job=job,
            )

        def _make_run_handle(self, job, run_id, **kwargs):
            return SimpleNamespace(id=run_id, backend=kwargs["backend"])

    site = Site()
    run = site.submit_automatically(
        object(),
        sizing_policy={"max_nodes_per_frequency": 2},
        maximum_credits="100",
        partition="cpu",
    )
    assert run.id == "fake-run"
    assert site.options == {
        "policy": {"max_nodes_per_frequency": 2},
        "maximum_credits": "100",
        "run_after_planning": True,
        "partition": "cpu",
    }


def pending_automatic_run():
    from frequensolve.orchestrator.sites.base import JobStatus, RunHandle

    class Site(ExecutionPlanningMixin):
        def __init__(self):
            self.graphql_client = Client()
            self.bound = []

        def plan(self, job, **_options):
            data = {
                "head": {
                    "plan_id": "fake-plan",
                    "revision": 0,
                    "status": "preparing",
                    "operation_id": "fake-planning",
                    "owner": {"project_key": "fake-project"},
                }
            }
            self.graphql_client.response = deepcopy(data)
            return PlanHandle(self, data, job)

        def _make_run_handle(self, job, run_id, *, backend):
            self.bound.append(run_id)
            return RunHandle(
                site=self,
                job=job,
                id=run_id,
                mode="aws",
                backend=backend,
                _status_fn=lambda _run: JobStatus(
                    state="completed", return_code=0, job_id=run_id
                ),
                _cancel_fn=lambda _run: (_ for _ in ()).throw(
                    AssertionError("Legacy cancellation must not handle saved plans")
                ),
            )

    site = Site()
    return site, site.submit_automatically(object(), maximum_credits="100")


def test_automatic_handle_waits_without_a_fake_simulation_or_another_authorization():
    site, run = pending_automatic_run()
    assert isinstance(run, AutomaticRunHandle)
    assert run.id is None
    assert run.status().state == "pending"
    assert run.backend == {"planId": "fake-plan", "planRevision": 0}
    site.graphql_client.response["head"]["status"] = "planning"
    assert run.status().state == "running"
    assert run.id is None
    assert site.bound == []
    assert all(call["action"] == "get" for call in site.graphql_client.calls)
    with pytest.raises(PlanError, match="no solver outputs"):
        run.fetch()
    with pytest.raises(PlanError, match="no execution logs"):
        run.logs()


def test_automatic_handle_adopts_the_real_run_once_and_pins_its_cancellation():
    site, run = pending_automatic_run()
    site.graphql_client.response["head"].update(
        status="executing", revision=1, execution_id="fake-run"
    )
    assert run.wait(timeout=1, poll_interval=0.01, check=False).successful
    assert run.id == "fake-run"
    assert site.bound == ["fake-run"]
    run.plan.data["head"].update(execution_id="a-later-run", revision=3)
    run.cancel()
    assert site.graphql_client.calls[-1]["action"] == "cancel"
    assert site.graphql_client.calls[-1]["expected_execution_id"] == "fake-run"
    assert site.graphql_client.calls[-1]["expected_revision"] == 1


@pytest.mark.parametrize("status", ["review", "incomplete", "awaiting_authorization"])
def test_automatic_handle_exposes_a_saved_pause_without_executing_or_supplying_another_quote(
    status,
):
    site, run = pending_automatic_run()
    site.graphql_client.response["head"].update(status=status, revision=1)
    with pytest.raises(PlanExecutionPaused) as paused:
        run.status()
    assert paused.value.plan is run.plan
    assert run.id is None
    assert site.bound == []
    assert [call["action"] for call in site.graphql_client.calls] == ["get"]


def test_automatic_handle_cancels_planning_and_reports_pre_execution_failure_as_planning():
    site, run = pending_automatic_run()
    run.cancel()
    assert site.graphql_client.calls[-1]["action"] == "cancel"
    assert "expected_execution_id" not in site.graphql_client.calls[-1]
    site.graphql_client.response["head"]["status"] = "failed"
    status = run.status()
    assert status.state == "failed"
    assert status.raw["stage"] == "planning"
    assert status.job_id is None
    assert run.id is None
    assert site.bound == []


def test_automatic_handle_waits_for_server_continuation_without_requesting_another_quote():
    site, run = pending_automatic_run()
    site.graphql_client.response["head"].update(status="review", revision=1)
    site.graphql_client.response["automatic"] = {
        "phase": "awaiting_execution",
        "maximum_credits": "100",
    }
    assert run.status().raw["stage"] == "execution_admission"
    assert run.id is None
    assert [call["action"] for call in site.graphql_client.calls] == ["get"]
    site.graphql_client.response["automatic"] = {
        "phase": "paused",
        "maximum_credits": "100",
        "message": "Pricing changed; accept a fresh quote.",
    }
    with pytest.raises(PlanExecutionPaused, match="Pricing changed"):
        run.status()
    assert site.bound == []


def test_preparation_authorizes_the_complete_automatic_quote_with_the_original_user_bound():
    plan, client = handle()
    plan.data = {"head": {**plan.data["head"], "revision": 0, "status": "preparing"}}
    plan.authorize("100")
    client.response = {
        "head": {**plan.data["head"], "status": "awaiting_authorization"},
        "quote": {
            "purpose": "automatic",
            "fingerprint": "fake-combined-quote",
            "maximum_credits": "80",
            "planning_maximum_credits": "10",
            "execution_maximum_credits": "70",
        },
    }
    plan.refresh()
    assert [call["action"] for call in client.calls] == ["get", "authorize"]
    assert client.calls[-1]["accepted_quote_fingerprint"] == "fake-combined-quote"
    assert client.calls[-1]["maximum_credits"] == "100"


def test_new_validation_quote_receives_a_new_authorization_identity():
    plan, client = handle()
    plan.authorize("20")
    previous = client.calls[-1]["idempotency_key"]
    plan.data["quote"]["fingerprint"] = "new-validation-quote"
    client.response = deepcopy(plan.data)
    plan.authorize("20")
    assert client.calls[-1]["idempotency_key"] != previous


def test_lost_execution_response_replays_the_exact_intent_without_requoting():
    class LostResponseClient(Client):
        lost = False

        def execute(self, _query, values):
            payload = json.loads(values["input"])
            self.calls.append(payload)
            if payload["action"] == "quote_execution":
                return {
                    "requestMyExecutionPlan": json.dumps(
                        {
                            **data,
                            "quote": {
                                "maximum_credits": "10",
                                "fingerprint": "fake-quote",
                            },
                        }
                    )
                }
            if not self.lost:
                self.lost = True
                raise ConnectionError("accepted response lost")
            return {
                "requestMyExecutionPlan": json.dumps(
                    {
                        **data,
                        "head": {
                            **data["head"],
                            "status": "executing",
                            "execution_id": "fake-run",
                        },
                    }
                )
            }

    class Site(ExecutionPlanningMixin):
        def _make_run_handle(self, job, run_id, **kwargs):
            return SimpleNamespace(id=run_id)

    site = Site()
    site.graphql_client = LostResponseClient()
    data = {
        "head": {
            "plan_id": "fake-plan",
            "revision": 1,
            "status": "review",
            "owner": {"project_key": "fake-project"},
        }
    }
    plan = PlanHandle(site, deepcopy(data), object())
    with pytest.raises(ConnectionError):
        site.execute_plan(plan, maximum_credits="20")
    assert site.execute_plan(plan, maximum_credits="20").id == "fake-run"
    assert [call["action"] for call in site.graphql_client.calls] == [
        "quote_execution",
        "execute",
        "execute",
    ]
    assert site.graphql_client.calls[1] == site.graphql_client.calls[2]
    with pytest.raises(PlanError, match="same authorization"):
        site.execute_plan(plan, maximum_credits="30")


def test_completed_plan_requires_explicit_new_key_for_an_intentional_rerun():
    class Site(ExecutionPlanningMixin):
        def _run_handle_from_plan(self, plan, run_id):
            return SimpleNamespace(id=run_id)

    terminal = {
        "head": {
            "plan_id": "fake-plan",
            "revision": 1,
            "status": "executed",
            "owner": {"project_key": "fake-project"},
            "execution_id": "fake-old-run",
        },
        "execution": {"phase": "settled", "outcome": "FAILED"},
    }

    class RerunClient(Client):
        def execute(self, _query, values):
            payload = json.loads(values["input"])
            self.calls.append(payload)
            if payload["action"] == "quote_execution":
                response = {
                    **terminal,
                    "quote": {"maximum_credits": "10", "fingerprint": "fresh-quote"},
                }
            else:
                response = {
                    "head": {
                        **terminal["head"],
                        "status": "executing",
                        "execution_id": "fake-new-run",
                    }
                }
            return {"requestMyExecutionPlan": json.dumps(response)}

    site = Site()
    site.graphql_client = RerunClient()
    plan = PlanHandle(site, deepcopy(terminal), object())
    assert plan.execute("20").id == "fake-old-run"
    assert site.graphql_client.calls == []
    assert (
        plan.execute("20", idempotency_key="intentional-new-run").id == "fake-new-run"
    )
    assert [call["action"] for call in site.graphql_client.calls] == [
        "quote_execution",
        "execute",
    ]
    accepted = deepcopy(site.graphql_client.calls[-1])
    assert accepted["accepted_quote_fingerprint"] == "fresh-quote"
    assert (
        plan.execute("20", idempotency_key="intentional-new-run").id == "fake-new-run"
    )
    assert site.graphql_client.calls[-1] == accepted
    with pytest.raises(PlanError, match="active execution"):
        plan.execute("20", idempotency_key="another-new-run")
    assert len(site.graphql_client.calls) == 3


def test_execution_quote_is_inspectable_without_authorizing_or_dispatching_and_is_a_copy():
    plan, client = handle()
    client.response["quote"] = {
        "purpose": "execution",
        "maximum_credits": "12.5",
        "fingerprint": "fresh-quote",
    }
    result = plan.quote_execution()
    assert [call["action"] for call in client.calls] == ["quote_execution"]
    assert client.calls[0]["expected_revision"] == 1
    result["maximum_credits"] = "999"
    assert plan.data["quote"]["maximum_credits"] == "12.5"
    assert plan._execute_requests == {}
    client.response["quote"]["purpose"] = "planning"
    with pytest.raises(PlanError, match="no execution quote"):
        plan.quote_execution()


def test_historical_plan_handle_reads_the_exact_revision_and_cannot_mutate_or_execute():
    class Site(ExecutionPlanningMixin):
        pass

    site = Site()
    site.graphql_client = Client()
    plan, _client = handle()
    historical = deepcopy(plan.data)
    historical["head"]["status"] = "historical"
    historical["historical"] = {"revision": 1, "current_revision": 3}
    site.graphql_client.response = historical
    saved = site.open_plan(plan.id, project_key="fake-project", revision=1)
    assert saved.status == "historical"
    saved.refresh()
    assert all(call.get("revision") == 1 for call in site.graphql_client.calls)
    before = len(site.graphql_client.calls)
    with pytest.raises(PlanError, match="read-only"):
        saved.edit([{"frequency_index": 0, "nodes": 2}])
    with pytest.raises(PlanError, match="read-only"):
        saved.validate()
    with pytest.raises(PlanError, match="read-only"):
        saved.cancel()
    with pytest.raises(PlanError, match="read-only"):
        saved.quote_execution()
    with pytest.raises(PlanError, match="read-only"):
        site.execute_plan(saved, maximum_credits="100", idempotency_key="new-run")
    assert len(site.graphql_client.calls) == before
    assert saved.wait(timeout=1).status == "historical"


def test_open_saved_revision_rejects_incompatible_server_or_other_plan_before_authorization():
    class Site(ExecutionPlanningMixin):
        pass

    site = Site()
    site.graphql_client = Client()
    plan, _client = handle()
    site.graphql_client.response = deepcopy(plan.data)
    with pytest.raises(PlanError, match="compatible Cloud"):
        site.open_plan(plan.id, project_key="fake-project", revision=1)
    with pytest.raises(PlanError, match="different saved plan"):
        site.open_plan("another-plan", project_key="fake-project")
    before = len(site.graphql_client.calls)
    with pytest.raises(PlanError):
        site.open_plan(plan.id, project_key="fake-project", revision=0)
    assert len(site.graphql_client.calls) == before


def test_progress_is_explicitly_refreshed_copied_and_separate_from_accepted_authority():
    plan, client = handle()
    assert plan.progress is None
    progress = {
        "purpose": "planning",
        "stage": "awaiting_compute_results",
        "completed": 1,
        "total": 2,
        "counts": {
            "ready": 0,
            "submitted": 1,
            "intent": 0,
            "uncertain": 0,
            "completed": 1,
            "failed": 0,
            "canceled": 0,
            "not_started": 0,
        },
        "message": "Waiting for compute or results",
        "observed_at": "2026-10-16T00:00:00.000Z",
    }
    plan.data["progress"] = deepcopy(progress)
    snapshot = plan.progress
    snapshot["counts"]["submitted"] = 0
    assert plan.progress == progress
    assert client.calls == []
    assert plan.revision == 1
    plan.data["progress"]["total"] = 3
    with pytest.raises(PlanError, match="incomplete planning progress"):
        _ = plan.progress


def test_automatic_planning_status_reports_observed_counts_without_submitting_again():
    site, run = pending_automatic_run()
    site.graphql_client.response["head"]["status"] = "planning"
    site.graphql_client.response["progress"] = {
        "purpose": "planning",
        "stage": "awaiting_compute_results",
        "completed": 0,
        "total": 1,
        "counts": {
            "ready": 0,
            "submitted": 1,
            "intent": 0,
            "uncertain": 0,
            "completed": 0,
            "failed": 0,
            "canceled": 0,
            "not_started": 0,
        },
        "message": "Waiting for compute or results",
        "observed_at": "2026-10-16T00:00:00.000Z",
    }
    status = run.status()
    assert status.message == "Waiting for compute or results; 0 of 1 evaluations closed"
    assert status.raw["progress"]["counts"]["submitted"] == 1
    assert run.id is None and site.bound == []
    assert all(call["action"] == "get" for call in site.graphql_client.calls)
