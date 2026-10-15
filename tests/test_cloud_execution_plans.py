"""Saved-plan requests authorize bounded work and preserve retry identities."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from frequensolve.execution_plan import PlanError
from frequensolve.orchestrator.sites.aws.plans import (
    ExecutionPlanningMixin,
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


def test_wait_does_not_execute_a_ready_plan():
    plan, client = handle()
    client.response["head"]["status"] = "review"
    assert plan.wait(timeout=1).status == "review"
    assert [call["action"] for call in client.calls] == ["get"]


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
