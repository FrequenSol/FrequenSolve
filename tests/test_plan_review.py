"""Public saved-plan review is deterministic and never starts paid work."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frequensolve.execution_plan import PLAN_VERSION, PlanError
from frequensolve.orchestrator.sites.aws.plans import PlanHandle
from frequensolve.plan_review import (
    compare_plans,
    format_diff,
    format_review,
    review_plan,
)


def saved():
    def request(nodes, ranks, cpus, memory):
        return {
            "layout": {
                "partition": "cpu",
                "node_type": "fake.arm",
                "nodes": nodes,
                "ranks_per_node": ranks,
                "cpus_per_rank": cpus,
            },
            "memory_mib_per_node": memory,
            "wall_time_seconds": 900,
        }

    rows = []
    for index, resource in enumerate([request(1, 1, 4, 1024), request(2, 2, 4, 8192)]):
        rows.append(
            {
                "frequency_index": index,
                "frequency": str(index + 1),
                "automatic": deepcopy(resource),
                "requested": resource,
                "overrides": {},
                "validation": {
                    "status": "valid",
                    "minimum_memory_mib_per_node": resource["memory_mib_per_node"],
                },
                "reason": "Fewest feasible qualified nodes",
                "estimated_runtime_seconds": 12345,
            }
        )
    return {
        "head": {
            "plan_id": "fake-plan",
            "revision": 3,
            "status": "review",
            "owner": {
                "account_id": "fake-account",
                "user_id": "fake-user",
                "project_key": "fake-project",
            },
        },
        "plan": {
            "version": PLAN_VERSION,
            "plan_id": "fake-plan",
            "revision": 3,
            "rows": rows,
            "limits": {
                "max_nodes_per_frequency": 2,
                "max_concurrent_frequencies": 1,
                "max_running_cpus": 16,
                "max_wall_time_seconds": 900,
                "max_layout_evaluations": 8,
            },
            "context": {
                "input_digest": "sha256:fake-input",
                "solver_digest": "sha256:fake-solver",
            },
        },
        "quote": {
            "purpose": "planning",
            "maximum_credits": "12.50",
            "fingerprint": "fake-quote",
        },
    }


def test_mixed_resources_keep_per_node_and_distributed_totals_separate():
    view = review_plan(saved())
    first, second = [row["requested"] for row in view["rows"]]
    assert (first["mpi_ranks"], first["total_cpus"], first["total_memory_mib"]) == (
        1,
        4,
        1024,
    )
    assert (second["mpi_ranks"], second["total_cpus"], second["total_memory_mib"]) == (
        4,
        16,
        16384,
    )
    assert second["memory_mib_per_node"] == 8192
    assert view["limits"]["max_running_cpus"] == 16  # not the sum, 20
    text = format_review(saved())
    assert "8192 MiB (8 GiB)" in text and "16384 MiB (16 GiB)" in text
    assert "not summed job requests" in text
    assert "12345" not in text  # legacy runtime placeholder is not a prediction
    assert "Recorded planning quote: up to 12.50 Credits" in text
    assert "not a charge or fresh authorization" in text
    assert "execution quote" not in text


def test_summary_snapshot_diff_and_recovery_never_refresh_or_authorize():
    client = Mock()
    client.execute.side_effect = AssertionError("Review must never contact Cloud")
    plan = PlanHandle(SimpleNamespace(graphql_client=client), saved())
    plan._pending_planning_maximum = "100"
    before = deepcopy(plan.data)
    snapshot = plan.snapshot()
    snapshot["plan"]["rows"][0]["requested"]["layout"]["nodes"] = 2
    assert plan.data == before
    plan.summary(details=True)
    plan.diff(before)
    guidance = plan.recovery()
    assert "site.open_plan('fake-plan', project_key='fake-project')" in guidance
    assert "idempotency key" in guidance and "refresh()/wait()" in guidance
    assert plan.data == before and plan._pending_planning_maximum == "100"
    client.execute.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        "preparing",
        "awaiting_authorization",
        "planning",
        "incomplete",
        "failed",
        "canceled",
    ],
)
def test_unallocated_or_preparing_plans_do_not_invent_resources_or_free_compute(status):
    data = saved()
    data["head"]["status"] = status
    data.pop("plan")
    data.pop("quote")
    view = review_plan(data)
    assert view["rows"] == [] and view["recorded_quote"] is None
    text = format_review(data)
    assert "Frequency allocations: unavailable" in text
    assert "Credit quote: unavailable (not zero)" in text


def test_missing_request_does_not_substitute_automatic_sizing():
    data = saved()
    data["plan"]["rows"][1]["requested"] = None
    data["plan"]["rows"][1]["validation"] = {
        "status": "invalid",
        "reason": "No qualified layout fits",
    }
    view = review_plan(data)
    assert view["rows"][1]["requested"] is None
    text = format_review(data, details=True)
    assert "unavailable" in text and "No qualified layout fits" in text


def test_historical_quotes_and_allocations_are_labelled_read_only():
    data = saved()
    data["head"]["status"] = "historical"
    assert "Historical revision: read-only" in format_review(data)
    data["historical"] = {"revision": 3}
    assert "omit revision=" in PlanHandle(None, data).recovery()


def test_recorded_captures_are_not_inferred_from_quotes_or_summed_as_lifetime_cost():
    data = saved()
    data.pop("quote")
    data["planning"] = {
        "purpose": "validation",
        "phase": "settled",
        "maximum_credits": "20",
        "captured_credits": "7.25",
    }
    data["execution"] = {"phase": "settled", "captured_credits": "9.50"}
    text = format_review(data)
    assert (
        "Latest recorded validation operation: settled; reported captured Credits: 7.25"
        in text
    )
    assert (
        "Latest recorded execution operation: settled; reported captured Credits: 9.50"
        in text
    )
    assert (
        "not a lifetime total" in text
        and "Credit quote: unavailable (not zero)" in text
    )
    data["execution"]["phase"] = "settling"
    with pytest.raises(PlanError, match="recorded settled"):
        format_review(data)


def test_automatic_quote_breakdown_preserves_decimal_ceilings():
    data = saved()
    data["quote"] = {
        "purpose": "automatic",
        "maximum_credits": "30.0001",
        "planning_maximum_credits": "10.0001",
        "execution_maximum_credits": "20",
    }
    text = format_review(data)
    assert "automatic quote: up to 30.0001" in text
    assert "planning_maximum_credits: 10.0001" in text
    assert "execution_maximum_credits: 20" in text


def test_same_amount_with_a_different_quote_identity_is_still_a_visible_change():
    before, after = saved(), saved()
    after["quote"]["fingerprint"] = "a-new-quote"
    assert "recorded_quote.fingerprint: fake-quote -> a-new-quote" in format_diff(
        before, after
    )


@pytest.mark.parametrize(
    "field", ["nodes", "ranks_per_node", "cpus_per_rank", "node_type", "partition"]
)
def test_layout_changes_are_visible_without_claiming_paid_validation_is_certain(field):
    before = saved()
    after = deepcopy(before)
    after["head"]["revision"] += 1
    after["plan"]["revision"] += 1
    shape = after["plan"]["rows"][0]["requested"]["layout"]
    shape[field] = shape[field] + 1 if isinstance(shape[field], int) else "other"
    after["plan"]["rows"][0]["validation"] = {"status": "needs_validation"}
    diff = compare_plans(before, after)
    assert diff["changes"][0]["fields"]["requested"]["after"][field] == shape[field]
    text = format_diff(before, after)
    assert "Layout changed" in text and "paid sizing may be needed" in text
    assert "plan.validate() without a Credit maximum" in text
    assert before == saved()


def test_reservations_overrides_reset_validation_and_server_compute_response():
    before = saved()
    after = deepcopy(before)
    row = after["plan"]["rows"][0]
    row["requested"]["memory_mib_per_node"] = 2048
    row["requested"]["wall_time_seconds"] = 1200
    row["overrides"] = {"memory_mib_per_node": 2048, "wall_time_seconds": 1200}
    row["validation"] = {"status": "needs_validation"}
    text = format_diff(before, after)
    assert (
        "Reservation-only change" in text
        and "server structural/evidence checks" in text
    )
    assert "1024 -> 2048" in text and "900 -> 1200" in text
    after["requires_compute"] = False
    after["plan"]["rows"][0]["validation"] = {"status": "valid"}
    assert review_plan(after)["requires_compute"] is False
    assert "Server validation response requires compute: false" in format_review(after)
    after["plan"]["rows"][0] = deepcopy(before["plan"]["rows"][0])
    reset = format_diff(
        {
            **before,
            "plan": {**before["plan"], "rows": [row, before["plan"]["rows"][1]]},
        },
        after,
    )
    assert "overrides" in reset


@pytest.mark.parametrize(
    "field", ["account_id", "user_id", "project_key", "project_id"]
)
def test_comparisons_reject_other_owners_or_bindings(field):
    before, after = saved(), saved()
    after["head"]["owner"][field] = "different"
    with pytest.raises(PlanError, match="exact owner"):
        compare_plans(before, after)


def test_comparisons_reject_other_plans_changed_frequency_identity_and_reverse_revisions():
    before, after = saved(), saved()
    after["head"]["plan_id"] = after["plan"]["plan_id"] = "another-plan"
    with pytest.raises(PlanError, match="same plan"):
        compare_plans(before, after)
    after = saved()
    after["plan"]["rows"][0]["frequency"] = "20"
    with pytest.raises(PlanError, match="Frequency identity"):
        compare_plans(before, after)
    after = saved()
    after["head"]["revision"] = 2
    with pytest.raises(PlanError, match="snapshot is newer"):
        compare_plans(before, after)


def test_comparison_matches_indices_not_row_order_and_exposes_changed_context():
    before, after = saved(), saved()
    after["plan"]["rows"].reverse()
    assert compare_plans(before, after)["changes"] == []
    after["plan"]["context"]["input_digest"] = "sha256:changed-input"
    text = format_diff(before, after)
    assert "context.input_digest: sha256:fake-input -> sha256:changed-input" in text


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1", 10, None])
def test_invalid_recorded_quotes_fail_without_float_or_zero_coercion(amount):
    data = saved()
    data["quote"]["maximum_credits"] = amount
    with pytest.raises(PlanError, match="Credit quote"):
        review_plan(data)


def test_malformed_and_incompatible_rows_fail_clearly():
    data = saved()
    data["plan"]["version"] = "future.v2"
    with pytest.raises(PlanError, match="incompatible"):
        review_plan(data)
    data = saved()
    data["plan"]["rows"].append(data["plan"]["rows"][0])
    with pytest.raises(PlanError, match="duplicate"):
        review_plan(data)
    data = saved()
    data["plan"]["rows"][0]["requested"]["layout"]["nodes"] = True
    with pytest.raises(PlanError, match="positive integer"):
        review_plan(data)


def test_labels_cannot_inject_terminal_commands_or_extra_rows():
    data = saved()
    data["plan"]["rows"][0]["reason"] = "value\x1b[2J\nforged row\u202e"
    text = format_review(data, details=True)
    assert "\x1b" not in text and "\u202e" not in text
    assert "\nforged row" not in text


def test_copy_recovery_preserves_existing_files_and_claim_even_with_load_if_exists(
    tmp_path,
):
    from frequensolve.project.project import Project

    source = Project(name="source", path=tmp_path / "source")
    source.enable_identity()
    target = Project.copy(source.path, tmp_path / "target")
    claim = target.path / ".fs-copy.lock"
    claim.write_text("failed copy")
    original = (target.path / ".frequensolve/project-identity.json").read_bytes()
    with pytest.raises(FileExistsError, match="different new, empty destination"):
        Project.copy(source.path, target.path, load_if_exists=True)
    assert claim.read_text() == "failed copy"
    assert (
        target.path / ".frequensolve/project-identity.json"
    ).read_bytes() == original
