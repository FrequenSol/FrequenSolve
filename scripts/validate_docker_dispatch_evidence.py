"""Validate evidence that binds a DockerImage dispatch to exact test inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "frequensolve-docker-dispatch-evidence/v1"
WORKFLOW_REPOSITORY = "FrequenSol/FrequenSolveDockerImage"
WORKFLOW_PATH = ".github/workflows/cicd-workflow.yml"
TEST_MARKER = "not cloud and not hpc and not interactive"
TEST_STATUS = "passed"
TEST_ARTIFACT = "frequensolve-test-evidence"


def validate_dispatch_evidence(
    evidence: dict[str, Any],
    *,
    run_id: int,
    request_id: str,
    workflow_commit: str,
    source_ref: str,
    sauce_ref: str,
    sauce_commit: str,
    fs_mumps_commit: str,
    frequensolve_commit: str,
) -> None:
    """Require an exact, successful, no-push dispatch evidence manifest."""
    expected = {
        "schemaVersion": SCHEMA,
        "runId": run_id,
        "runUrl": (f"https://github.com/{WORKFLOW_REPOSITORY}/actions/runs/{run_id}"),
        "requestId": request_id,
        "workflowRepository": WORKFLOW_REPOSITORY,
        "workflowPath": WORKFLOW_PATH,
        "workflowCommit": workflow_commit,
        "sourceRef": source_ref,
        "sourceCommit": workflow_commit,
        "sauceRef": sauce_ref,
        "sauceCommit": sauce_commit,
        "fsMumpsRef": fs_mumps_commit,
        "fsMumpsCommit": fs_mumps_commit,
        "frequensolveSource": "git",
        "frequensolveRef": frequensolve_commit,
        "frequensolveCommit": frequensolve_commit,
        "disablePush": True,
        "testMarker": TEST_MARKER,
        "testStatus": TEST_STATUS,
        "testArtifact": TEST_ARTIFACT,
    }
    mismatches = [
        f"{name} must be {value!r}, got {evidence.get(name)!r}"
        for name, value in expected.items()
        if evidence.get(name) != value
    ]
    unexpected = sorted(set(evidence) - set(expected))
    missing = sorted(set(expected) - set(evidence))
    if missing:
        mismatches.append(f"missing fields: {', '.join(missing)}")
    if unexpected:
        mismatches.append(f"unexpected fields: {', '.join(unexpected)}")
    if mismatches:
        raise ValueError("invalid Docker dispatch evidence: " + "; ".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--workflow-commit", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--sauce-ref", required=True)
    parser.add_argument("--sauce-commit", required=True)
    parser.add_argument("--fs-mumps-commit", required=True)
    parser.add_argument("--frequensolve-commit", required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Docker dispatch evidence must be an object")
        validate_dispatch_evidence(
            payload,
            run_id=args.run_id,
            request_id=args.request_id,
            workflow_commit=args.workflow_commit,
            source_ref=args.source_ref,
            sauce_ref=args.sauce_ref,
            sauce_commit=args.sauce_commit,
            fs_mumps_commit=args.fs_mumps_commit,
            frequensolve_commit=args.frequensolve_commit,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print("Docker dispatch evidence is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
