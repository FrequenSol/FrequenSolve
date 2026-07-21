from __future__ import annotations

import pytest

from scripts.select_docker_workflow_evidence import (
    EVIDENCE_ARTIFACT,
    WORKFLOW_ID,
    WORKFLOW_PATH,
    WORKFLOW_REPOSITORY,
    select_artifact,
    select_run,
)

REQUEST_ID = "frequensolve-rc-123-1"
ACTOR = "frequensolver-builder[bot]"
HEAD_SHA = "a" * 40
CREATED_AFTER = "2026-07-21T05:00:00Z"


def _run(**overrides):
    run = {
        "id": 456,
        "display_title": f"Runtime image {REQUEST_ID}",
        "event": "workflow_dispatch",
        "path": WORKFLOW_PATH,
        "workflow_id": WORKFLOW_ID,
        "head_branch": "main",
        "head_sha": HEAD_SHA,
        "actor": {"login": ACTOR},
        "repository": {"full_name": WORKFLOW_REPOSITORY},
        "created_at": "2026-07-21T05:00:01Z",
        "status": "completed",
        "conclusion": "success",
    }
    run.update(overrides)
    return run


def _select(payload, *, require_success=False):
    return select_run(
        payload,
        request_id=REQUEST_ID,
        actor=ACTOR,
        head_sha=HEAD_SHA,
        created_after=CREATED_AFTER,
        require_success=require_success,
    )


def test_selects_only_the_exact_correlated_dispatch():
    wrong_title = _run(id=1, display_title="Runtime image another-request")
    wrong_commit = _run(id=2, head_sha="b" * 40)
    stale = _run(id=3, created_at="2026-07-21T04:59:59Z")
    expected = _run()

    assert (
        _select({"workflow_runs": [wrong_title, wrong_commit, stale, expected]})
        == expected
    )


def test_rejects_ambiguous_dispatches():
    with pytest.raises(ValueError, match="matched 2"):
        _select({"workflow_runs": [_run(id=456), _run(id=457)]})


def test_requires_completed_success_when_sealing_evidence():
    failed = _run(status="completed", conclusion="failure")

    with pytest.raises(ValueError, match="completed successfully"):
        _select({"workflow_runs": [failed]}, require_success=True)


def test_selects_one_live_evidence_artifact():
    expected = {"id": 789, "name": EVIDENCE_ARTIFACT, "expired": False}
    payload = {
        "artifacts": [
            {"id": 1, "name": EVIDENCE_ARTIFACT, "expired": True},
            {"id": 2, "name": "other", "expired": False},
            expected,
        ]
    }

    assert select_artifact(payload) == expected


def test_rejects_missing_or_duplicate_live_evidence_artifacts():
    with pytest.raises(ValueError, match="got 0"):
        select_artifact({"artifacts": []})
    with pytest.raises(ValueError, match="got 2"):
        select_artifact(
            {
                "artifacts": [
                    {"id": 1, "name": EVIDENCE_ARTIFACT, "expired": False},
                    {"id": 2, "name": EVIDENCE_ARTIFACT, "expired": False},
                ]
            }
        )
