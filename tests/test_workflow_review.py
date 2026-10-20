"""Native review evidence is useful without granting it paid authority."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import anyio
import pytest
from mcp import Client
from mcp.types import ElicitResult

from frequensolve.workflow.client import create_project
from frequensolve.workflow.journal import IntentJournal
from frequensolve.workflow.review import (
    ReviewEvidence,
    digest,
    display,
    proposal,
    request_input,
)
from frequensolve.workflow.server import REVIEW_TOOL, TOOLS, build_server
from frequensolve.workflow.storage import PrivateStore


def view(project="fake-project"):
    return {
        "head": {
            "plan_id": "fake-plan",
            "owner": {"project_key": project},
            "revision": 0,
            "status": "awaiting_authorization",
            "operation_id": "fake-sizing",
        },
        "input_digest": "sha256:" + "1" * 64,
        "scientific_inputs": {
            "dimension": 3,
            "physics": "acoustic",
            "precision": "double",
            "frequencies": [["2", "0"], ["20", "0"]],
        },
        "compute_review": {
            "execution_site_id": "fake-cluster",
            "planning_evaluations": [
                {
                    "layout": {
                        "partition": "cpu",
                        "node_type": "fake-node",
                        "nodes": 2,
                        "ranks_per_node": 1,
                        "cpus_per_rank": 8,
                    },
                    "memory_mib_per_node": 1024,
                    "wall_time_seconds": 600,
                }
            ],
        },
        "quote": {
            "purpose": "planning",
            "maximum_credits": "10.25",
            "fingerprint": "sha256:" + "2" * 64,
        },
    }


def evidence(path, user="fake-user"):
    return ReviewEvidence(IntentJournal(PrivateStore(path), {"subject": user}))


def test_complete_authoritative_display_and_bounded_proposals():
    raw = view()
    snapshot = proposal(raw, "fake-project", "fake-plan")
    text = display(snapshot)
    assert all(
        s in text
        for s in [
            "NO-SPEND",
            "10.25",
            "fake-cluster",
            "600",
            "acoustic",
            "20",
            digest(snapshot),
        ]
    )
    for changed in ["revision", "input", "target", "resources", "quote", "science"]:
        other = deepcopy(raw)
        if changed == "revision":
            other["head"]["revision"] += 1
        elif changed == "input":
            other["input_digest"] = "sha256:" + "3" * 64
        elif changed == "target":
            other["compute_review"]["execution_site_id"] = "other"
        elif changed == "resources":
            other["compute_review"]["planning_evaluations"][0]["layout"]["nodes"] = 1
        elif changed == "quote":
            other["quote"]["maximum_credits"] = "11"
        else:
            other["scientific_inputs"]["precision"] = "single"
        assert digest(proposal(other, "fake-project", "fake-plan")) != digest(snapshot)
    for changed in [
        {**raw, "compute_review": None},
        {**raw, "scientific_inputs": {}},
        {**raw, "scientific_inputs": {"huge": "x" * 64000}},
        {**raw, "quote": {**raw["quote"], "maximum_credits": "nan"}},
    ]:
        with pytest.raises(ValueError):
            proposal(changed, "fake-project", "fake-plan")
    with pytest.raises(ValueError):
        proposal(raw, "foreign-project", "fake-plan")


@pytest.mark.parametrize(
    "response,state",
    [
        (
            {"action": "accept", "content": {"choice": "confirm_rehearsal"}},
            "client_confirmed_rehearsal",
        ),
        (
            {
                "action": "accept",
                "content": {
                    "choice": "request_changes",
                    "note": "Increase memory headroom",
                },
            },
            "changes_requested",
        ),
        ({"action": "accept", "content": {"choice": "cancel"}}, "canceled"),
        ({"action": "decline", "content": None}, "declined"),
        ({"action": "cancel", "content": None}, "canceled"),
        ({"action": "accept", "content": {"approved": True}}, "incomplete"),
        ({"action": "accept", "content": {"choice": "request_changes"}}, "incomplete"),
        (
            {
                "action": "accept",
                "content": {"choice": "confirm_rehearsal", "note": "x" * 1025},
            },
            "incomplete",
        ),
    ],
)
def test_native_responses_remain_unattested_and_recover_without_reprompt(
    tmp_path, response, state
):
    snapshot = proposal(view(), "fake-project", "fake-plan")
    prompts = []

    async def fetch():
        return snapshot

    async def elicit(message, schema):
        prompts.append((message, schema))
        return response

    async def exercise():
        saved = await request_input(
            evidence(tmp_path), "intent", snapshot, digest(snapshot), fetch, elicit
        )
        assert saved["state"] == state
        assert (
            not saved["human_approval_verified"]
            and not saved["paid_authority"]
            and not saved["backend_authorized"]
        )
        assert saved["evidence_origin"] == "client_reported_unattested"
        restarted = evidence(tmp_path)
        assert restarted.recover("intent") == saved
        assert (
            await request_input(
                restarted, "intent", snapshot, digest(snapshot), fetch, elicit
            )
            == saved
        )
        assert len(prompts) == 1

    anyio.run(exercise)
    with pytest.raises(ValueError):
        evidence(tmp_path, "foreign-user").recover("intent")


def test_stale_before_or_during_review_cannot_record_confirmation(tmp_path):
    snapshot = proposal(view(), "fake-project", "fake-plan")
    changed = deepcopy(snapshot)
    changed["quote"]["maximum_credits"] = "11"

    async def fetch():
        return changed

    async def elicit(*_):
        return {"action": "accept", "content": {"choice": "confirm_rehearsal"}}

    async def exercise():
        with pytest.raises(ValueError):
            await request_input(
                evidence(tmp_path), "before", snapshot, digest(changed), fetch, elicit
            )
        result = await request_input(
            evidence(tmp_path), "during", snapshot, digest(snapshot), fetch, elicit
        )
        assert result["state"] == "stale"

    anyio.run(exercise)


@pytest.mark.parametrize("failure", ["unsupported", "timeout", "cancel_after_response"])
def test_interruption_and_unsupported_clients_leave_incomplete_evidence(
    tmp_path, failure
):
    snapshot = proposal(view(), "fake-project", "fake-plan")

    async def fetch():
        if failure == "cancel_after_response":
            raise anyio.get_cancelled_exc_class()()
        return snapshot

    async def elicit(*_):
        if failure == "unsupported":
            raise ValueError("No client form")
        if failure == "timeout":
            await anyio.sleep(1)
        return {"action": "accept", "content": {"choice": "confirm_rehearsal"}}

    async def exercise():
        if failure == "cancel_after_response":
            with pytest.raises(anyio.get_cancelled_exc_class()):
                await request_input(
                    evidence(tmp_path),
                    "interrupted",
                    snapshot,
                    digest(snapshot),
                    fetch,
                    elicit,
                )
        else:
            await request_input(
                evidence(tmp_path),
                "interrupted",
                snapshot,
                digest(snapshot),
                fetch,
                elicit,
                timeout=0.01,
            )
        assert evidence(tmp_path).recover("interrupted")["state"] == "incomplete"

    anyio.run(exercise)


def test_pending_and_concurrent_attempts_do_not_reopen_a_prompt(tmp_path):
    snapshot = proposal(view(), "fake-project", "fake-plan")
    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = list(
            pool.map(lambda _: evidence(tmp_path).begin("same", snapshot), range(2))
        )
    assert sum(created for _, created in attempts) == 1
    assert attempts[0][0] == attempts[1][0]
    changed = deepcopy(snapshot)
    changed["revision"] = 1
    with pytest.raises(ValueError):
        evidence(tmp_path).begin("same", changed)
    assert evidence(tmp_path).recover("same")["state"] == "awaiting_input"


def test_real_mcp_context_form_and_closed_arguments_never_call_paid_transport(tmp_path):
    project = create_project(tmp_path.resolve(), "fresh", "Fake native review")
    calls, forms = [], []

    class Reader:
        def call(self, operation, arguments):
            calls.append(operation)
            assert operation == "plan_get"
            return view(project["project_key"])

        def journal(self, _):
            return IntentJournal(
                PrivateStore(tmp_path / "journal"), {"subject": "fake-user"}
            )

    async def on_elicit(context, params):
        forms.append(params)
        return ElicitResult(action="accept", content={"choice": "confirm_rehearsal"})

    async def exercise():
        server = build_server(
            Reader(),
            {"pilot": tmp_path.resolve()},
            "http://127.0.0.1:5173",
            interactive_review=True,
        )
        async with Client(
            server, mode="legacy", cache=None, elicitation_callback=on_elicit
        ) as session:
            assert {t.name for t in (await session.list_tools()).tools} == set(
                TOOLS
            ) | {REVIEW_TOOL}
            request = {
                "action": "preview",
                "root_id": "pilot",
                "local_project": "fresh",
                "plan_id": "fake-plan",
            }
            preview = await session.call_tool(REVIEW_TOOL, {"request": request})
            assert preview.structured_content["ok"]
            snapshot = json.loads(preview.structured_content["payload"])
            denied = await session.call_tool(
                REVIEW_TOOL, {"request": {**request, "approved": True}}
            )
            assert denied.is_error
            result = await session.call_tool(
                REVIEW_TOOL,
                {
                    "request": {
                        **request,
                        "action": "request_input",
                        "intent": "same",
                        "expected_fingerprint": snapshot["proposal_fingerprint"],
                    }
                },
            )
            assert result.structured_content["ok"]
            receipt = json.loads(result.structured_content["payload"])
            assert (
                receipt["state"] == "client_confirmed_rehearsal"
                and not receipt["backend_authorized"]
            )
            assert len(forms) == 1 and "NO-SPEND" in forms[0].message
            assert calls == ["plan_get"] * 3

    anyio.run(exercise)
