from __future__ import annotations

import pytest

from scripts.validate_docker_dispatch_evidence import (
    SCHEMA,
    TEST_ARTIFACT,
    TEST_MARKER,
    TEST_STATUS,
    WORKFLOW_PATH,
    WORKFLOW_REPOSITORY,
    validate_dispatch_evidence,
)

RUN_ID = 123
REQUEST_ID = "frequensolve-rc-456-1-0123456789abcdef0123456789abcdef"
WORKFLOW_COMMIT = "a" * 40
SAUCE_COMMIT = "b" * 40
FS_MUMPS_COMMIT = "c" * 40
FREQUENSOLVE_COMMIT = "d" * 40


def _evidence():
    return {
        "schemaVersion": SCHEMA,
        "runId": RUN_ID,
        "runUrl": (f"https://github.com/{WORKFLOW_REPOSITORY}/actions/runs/{RUN_ID}"),
        "requestId": REQUEST_ID,
        "workflowRepository": WORKFLOW_REPOSITORY,
        "workflowPath": WORKFLOW_PATH,
        "workflowCommit": WORKFLOW_COMMIT,
        "sourceRef": "main",
        "sourceCommit": WORKFLOW_COMMIT,
        "sauceRef": "v0.1.0",
        "sauceCommit": SAUCE_COMMIT,
        "fsMumpsRef": FS_MUMPS_COMMIT,
        "fsMumpsCommit": FS_MUMPS_COMMIT,
        "frequensolveSource": "git",
        "frequensolveRef": FREQUENSOLVE_COMMIT,
        "frequensolveCommit": FREQUENSOLVE_COMMIT,
        "disablePush": True,
        "testMarker": TEST_MARKER,
        "testStatus": TEST_STATUS,
        "testArtifact": TEST_ARTIFACT,
    }


def _validate(evidence):
    validate_dispatch_evidence(
        evidence,
        run_id=RUN_ID,
        request_id=REQUEST_ID,
        workflow_commit=WORKFLOW_COMMIT,
        source_ref="main",
        sauce_ref="v0.1.0",
        sauce_commit=SAUCE_COMMIT,
        fs_mumps_commit=FS_MUMPS_COMMIT,
        frequensolve_commit=FREQUENSOLVE_COMMIT,
    )


def test_accepts_exact_bound_dispatch_evidence():
    _validate(_evidence())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requestId", "another-request"),
        ("runId", 999),
        ("workflowCommit", "f" * 40),
        ("fsMumpsCommit", "e" * 40),
        ("frequensolveCommit", "f" * 40),
        ("disablePush", False),
        ("testStatus", "failed"),
    ],
)
def test_rejects_any_changed_dispatch_identity(field, value):
    evidence = _evidence()
    evidence[field] = value

    with pytest.raises(ValueError, match=field):
        _validate(evidence)


def test_rejects_missing_or_unexpected_fields():
    evidence = _evidence()
    del evidence["sourceRef"]
    evidence["untrusted"] = "value"

    with pytest.raises(ValueError, match="missing fields.*unexpected fields"):
        _validate(evidence)
