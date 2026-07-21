"""Validate the immutable evidence attached to a FrequenSolve release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.validate_docker_dispatch_evidence import (
        validate_dispatch_evidence,
    )
    from scripts.validate_heavy_test_evidence import validate_heavy_test_evidence
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/.
    from validate_docker_dispatch_evidence import validate_dispatch_evidence
    from validate_heavy_test_evidence import validate_heavy_test_evidence


SCHEMA = "frequensolve-release-evidence/v2"
DOCKER_WORKFLOW_PREFIX = (
    "FrequenSol/FrequenSolveDockerImage/.github/workflows/cicd-workflow.yml@"
)
TEST_MARKER = "not cloud and not hpc and not interactive"
TEST_STATUS = "passed"
TEST_ARTIFACT = "frequensolve-test-evidence"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FINAL_RELEASE_RE = re.compile(
    r"^v(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)$"
)


def validate_evidence(evidence: dict[str, Any], expected_commit: str) -> None:
    """Raise ``ValueError`` unless evidence proves the expected commit."""
    exact_values = {
        "schemaVersion": SCHEMA,
        "commit": expected_commit,
        "ciWorkflow": ".github/workflows/cicd-workflow.yml",
        "ciRequiredJob": "Required CI",
        "dockerTestRef": expected_commit,
        "dockerTestCommit": expected_commit,
        "dockerTestMarker": TEST_MARKER,
        "dockerTestStatus": TEST_STATUS,
        "dockerTestArtifact": TEST_ARTIFACT,
    }
    mismatches = [
        f"{name} must be {expected!r}, got {evidence.get(name)!r}"
        for name, expected in exact_values.items()
        if evidence.get(name) != expected
    ]
    docker_workflow = evidence.get("dockerWorkflow", "")
    docker_workflow_ref = docker_workflow.removeprefix(DOCKER_WORKFLOW_PREFIX)
    if not SHA_RE.fullmatch(docker_workflow_ref):
        mismatches.append(
            "dockerWorkflow must pin the DockerImage evidence workflow to a "
            "40-character commit SHA"
        )
    if evidence.get("dockerWorkflowCommit") != docker_workflow_ref:
        mismatches.append(
            "dockerWorkflowCommit must equal the commit pinned by dockerWorkflow"
        )
    for name in ("sauceCommit", "fsMumpsRef", "fsMumpsCommit"):
        if not SHA_RE.fullmatch(evidence.get(name, "")):
            mismatches.append(f"{name} must be a lowercase 40-character Git SHA")
    if evidence.get("fsMumpsRef") != evidence.get("fsMumpsCommit"):
        mismatches.append("fsMumpsRef must equal the immutable fsMumpsCommit")
    frequensolver_release = evidence.get("frequensolverRelease", "")
    if not FINAL_RELEASE_RE.fullmatch(frequensolver_release):
        mismatches.append(
            "frequensolverRelease must be an immutable final release tag vX.Y.Z"
        )
    if evidence.get("sauceRef") != frequensolver_release:
        mismatches.append("sauceRef must equal frequensolverRelease")
    if evidence.get("frequensolverVersion") != frequensolver_release:
        mismatches.append("frequensolverVersion must equal frequensolverRelease")
    if evidence.get("frequensolverGitCommit") != evidence.get("sauceCommit"):
        mismatches.append("frequensolverGitCommit must equal sauceCommit")
    if not evidence.get("frequensolverBuildId"):
        mismatches.append("frequensolverBuildId must be non-empty")
    expected_release_url = (
        "https://github.com/FrequenSol/Sauce/releases/tag/" f"{frequensolver_release}"
    )
    if evidence.get("frequensolverReleaseUrl") != expected_release_url:
        mismatches.append(
            "frequensolverReleaseUrl must identify the immutable FrequenSolver release"
        )
    for name in ("ciRunId", "dockerCallerRunId", "dockerEvidenceRunId"):
        value = evidence.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            mismatches.append(f"{name} must be a positive integer")
    if not evidence.get("ciRunUrl"):
        mismatches.append("ciRunUrl must be non-empty")
    if evidence.get("dockerImageTag") != evidence.get("dockerWorkflowCommit"):
        mismatches.append(
            "dockerImageTag must equal the pinned DockerImage workflow commit"
        )
    expected_docker_run_url = (
        "https://github.com/FrequenSol/FrequenSolve/actions/runs/"
        f"{evidence.get('dockerCallerRunId')}"
    )
    if evidence.get("dockerCallerRunUrl") != expected_docker_run_url:
        mismatches.append("dockerCallerRunUrl must identify dockerCallerRunId")
    expected_evidence_run_url = (
        "https://github.com/FrequenSol/FrequenSolveDockerImage/actions/runs/"
        f"{evidence.get('dockerEvidenceRunId')}"
    )
    if evidence.get("dockerEvidenceRunUrl") != expected_evidence_run_url:
        mismatches.append(
            "dockerEvidenceRunUrl must identify dockerEvidenceRunId in "
            "FrequenSolveDockerImage"
        )
    expected_request = (
        rf"frequensolve-rc-{evidence.get('dockerCallerRunId')}-"
        r"[1-9][0-9]*-[0-9a-f]{32}"
    )
    if not re.fullmatch(expected_request, evidence.get("dockerRequestId", "")):
        mismatches.append(
            "dockerRequestId must identify the caller run and positive run attempt"
        )
    archive_sha = evidence.get("dockerTestArchiveSha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
        mismatches.append("dockerTestArchiveSha256 must be a lowercase SHA-256")
    heavy_evidence = evidence.get("dockerTestEvidence")
    if not isinstance(heavy_evidence, dict):
        mismatches.append("dockerTestEvidence must be an object")
    else:
        try:
            validate_heavy_test_evidence(heavy_evidence, expected_commit)
        except ValueError as exc:
            mismatches.append(f"dockerTestEvidence is invalid: {exc}")
    dispatch_evidence = evidence.get("dockerDispatchEvidence")
    if not isinstance(dispatch_evidence, dict):
        mismatches.append("dockerDispatchEvidence must be an object")
    else:
        try:
            validate_dispatch_evidence(
                dispatch_evidence,
                run_id=evidence.get("dockerEvidenceRunId"),
                request_id=evidence.get("dockerRequestId", ""),
                workflow_commit=evidence.get("dockerWorkflowCommit", ""),
                source_ref="main",
                sauce_ref=evidence.get("sauceRef", ""),
                sauce_commit=evidence.get("sauceCommit", ""),
                fs_mumps_commit=evidence.get("fsMumpsCommit", ""),
                frequensolve_commit=expected_commit,
            )
        except ValueError as exc:
            mismatches.append(f"dockerDispatchEvidence is invalid: {exc}")
    if mismatches:
        raise ValueError("; ".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if not SHA_RE.fullmatch(args.commit):
        parser.error("--commit must be a lowercase 40-character Git SHA")

    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        validate_evidence(evidence, args.commit)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"release evidence is valid for {args.commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
