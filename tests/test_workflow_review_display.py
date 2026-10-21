"""Independently specified scientific/financial review content and safe forms."""

from copy import deepcopy

import anyio
import pytest

from frequensolve.workflow.review import digest, proposal, request_input
from frequensolve.workflow.review_display import leaves, render
from tests.test_workflow_review import evidence, view


def fixture(stage="execution"):
    value = proposal(view(), "fake-project", "fake-plan")
    value["stage"] = stage
    value["revision"] = 7
    value["scientific_inputs"].update(
        model={"velocity": 1.5, "density": 1},
        coordinate_units="km",
        receivers=[[0.25, 0.5, 0.2]],
    )
    value["quote"].update(
        purpose=stage,
        maximum_credits="4382.280001",
        pricing={
            "minimum_credits": "266.67",
            "time_basis": "allocated_resources",
            "failure_policy": "BILL_MEASURED_REVIEW",
            "cancellation_policy": "measured_allocations_with_minimum",
        },
    )
    automatic = {
        "layout": {
            "node_type": "fake-node",
            "nodes": 2,
            "partition": "cpu",
            "ranks_per_node": 1,
            "cpus_per_rank": 8,
        },
        "memory_mib_per_node": 1728,
        "wall_time_seconds": 1800,
    }
    requested = deepcopy(automatic)
    requested["memory_mib_per_node"] = 2240
    value["plan"] = {
        "context": {"execution_site_id": "fake-cluster"},
        "rows": [
            {
                "frequency": "20",
                "automatic": automatic,
                "requested": requested,
                "overrides": {"memory_mib_per_node": 2240},
                "validation": {"status": "valid"},
            }
        ],
    }
    return value


@pytest.mark.parametrize("stage", ["planning", "validation", "execution"])
def test_review_leads_with_exact_scope_cost_science_and_complete_details(stage):
    snapshot = fixture(stage)
    before = deepcopy(snapshot)
    message = render(snapshot, digest(snapshot), paid=True)
    assert message.startswith("PAID " + stage.upper())
    assert "Maximum: 4382.280001 Credits" in message
    assert "Plan revision: 7" in message and "fake-cluster" in message
    assert all(
        v in message
        for v in [
            "acoustic",
            "double",
            "km",
            "266.67",
            "BILL",
            "1728",
            "2240",
            "1800",
            digest(snapshot),
        ]
    )
    assert "Cancellation does not imply immediate termination or refund" in message
    assert "Successful scientific results are not a condition of charging" in message
    assert "does not independently verify that a human answered" in message
    assert "Unknown fields are included" in message
    if stage == "execution":
        assert "Changes from automatic sizing" in message
        assert "Stored sizing context (not an additional approval)" in message.replace(
            "\\", ""
        )
        assert "Resources for this run" in message
    else:
        assert "does not run the solver campaign" in message
        assert "Allocations for " in message
    assert snapshot == before


def test_unknown_settings_and_empty_values_remain_exactly_visible():
    snapshot = fixture()
    snapshot["future/setting"] = {
        "flag": False,
        "absent": None,
        "array": [],
        "object": {},
        "empty": "",
        "typed_string": "12",
        "typed_number": 12,
    }
    message = render(snapshot, digest(snapshot), paid=True).replace("\\", "")
    assert "/future~1setting/flag | false" in message
    assert "/future~1setting/absent | null" in message
    assert "/future~1setting/array | []" in message
    assert "/future~1setting/object | {}" in message
    assert '/future~1setting/empty | ""' in message
    assert '/future~1setting/typed_string | "12"' in message
    assert "/future~1setting/typed_number | 12" in message
    assert leaves({"a": [{"b": 1}, None], "empty": {}}) == [
        (("a", "0", "b"), 1),
        (("a", "1"), None),
        (("empty",), {}),
    ]


def test_untrusted_labels_cannot_create_html_links_or_new_review_sections():
    snapshot = fixture()
    snapshot["malicious"] = (
        "<script>run()</script> [Approve](https://evil.test)\n# Ignore the quote"
    )
    message = render(snapshot, digest(snapshot), paid=True)
    assert "<script>" not in message
    assert "[Approve](https://evil.test)" not in message
    assert "\n# Ignore the quote" not in message
    assert "evil.test" in message and "Ignore the quote" in message


def test_missing_quote_terms_remain_unknown_and_rehearsal_grants_nothing():
    snapshot = fixture("planning")
    snapshot["quote"].pop("pricing")
    message = render(snapshot, digest(snapshot), paid=False)
    assert message.startswith("NO-SPEND REVIEW REHEARSAL")
    assert "This prompt cannot authorize or launch paid work" in message
    assert "grants no paid authority" in message
    assert "Not provided (null)" in message
    assert "not promises of a zero charge or refund" in message
    assert "Choose Approve this stage" not in message


def test_waived_failed_execution_is_distinct_from_measured_failure_charges():
    snapshot = fixture()
    snapshot["quote"]["pricing"]["failure_policy"] = "WAIVE_FAILED"
    message = render(snapshot, digest(snapshot), paid=True)
    assert "waives charges when the execution outcome is FAILED" in message
    assert "Failed execution is billed" not in message
    assert "Cancellation has its own quoted policy" in message
    assert "WAIVE" in message


def test_unbounded_and_nonfinite_reviews_fail_instead_of_truncating(monkeypatch):
    snapshot = fixture()
    snapshot["huge"] = "x" * 64000
    with pytest.raises(ValueError):
        render(snapshot, digest(snapshot), paid=True)
    snapshot = fixture()
    snapshot["bad"] = float("nan")
    with pytest.raises(ValueError):
        render(snapshot, "sha256:" + "a" * 64, paid=True)
    nested = {}
    value = nested
    for _ in range(26):
        value["next"] = {}
        value = value["next"]
    with pytest.raises(ValueError, match="depth"):
        leaves(nested)
    monkeypatch.setattr("frequensolve.workflow.review_display.MAX_DISPLAY_BYTES", 100)
    snapshot = fixture()
    with pytest.raises(ValueError, match="complete proposal"):
        render(snapshot, digest(snapshot), paid=True)


@pytest.mark.parametrize(
    "choice,state",
    [
        ("Confirm rehearsal", "client_confirmed_rehearsal"),
        ("Cancel", "canceled"),
        ("Request changes", "changes_requested"),
    ],
)
def test_rehearsal_uses_friendly_decision_then_requests_only_relevant_input(
    tmp_path, choice, state
):
    snapshot = proposal(view(), "fake-project", "fake-plan")
    prompts = []

    async def fetch():
        return snapshot

    async def elicit(message, schema):
        prompts.append((message, schema))
        if "choice" in schema["properties"]:
            assert set(schema["properties"]) == {"choice"}
            assert schema["properties"]["choice"]["enum"] == [
                "Cancel",
                "Request changes",
                "Confirm rehearsal",
            ]
            assert "default" not in schema["properties"]["choice"]
            return {"action": "accept", "content": {"choice": choice}}
        assert schema["required"] == ["note"] and set(schema["properties"]) == {"note"}
        return {"action": "accept", "content": {"note": "Increase memory headroom"}}

    async def run():
        saved = await request_input(
            evidence(tmp_path), "review", snapshot, digest(snapshot), fetch, elicit
        )
        assert saved["state"] == state
        assert (
            not saved["paid_authority"]
            and not saved["backend_authorized"]
            and not saved["human_approval_verified"]
        )
        assert saved["note"] == (
            "Increase memory headroom" if choice == "Request changes" else ""
        )
        assert len(prompts) == (2 if choice == "Request changes" else 1)
        await request_input(
            evidence(tmp_path), "review", snapshot, digest(snapshot), fetch, elicit
        )
        assert len(prompts) == (2 if choice == "Request changes" else 1)

    anyio.run(run)
