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


LEGACY_SCHEMA = "frequensolve-release-evidence/v2"
SCHEMA = "frequensolve-release-evidence/v3"
STANDARD_PROFILE = "standard"
SOLVER_BACKED_PROFILE = "solver-backed"
VALIDATION_PROFILES = frozenset({STANDARD_PROFILE, SOLVER_BACKED_PROFILE})
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
STANDARD_FORBIDDEN_FIELDS = frozenset(
    {
        "dockerWorkflow",
        "dockerWorkflowCommit",
        "dockerCallerRunId",
        "dockerCallerRunUrl",
        "dockerEvidenceRunId",
        "dockerEvidenceRunUrl",
        "dockerRequestId",
        "dockerImageTag",
        "dockerTestRef",
        "dockerTestCommit",
        "dockerTestMarker",
        "dockerTestStatus",
        "dockerTestArtifact",
        "dockerTestArchiveSha256",
        "dockerTestEvidence",
        "dockerDispatchEvidence",
        "frequensolverVersion",
        "frequensolverBuildId",
        "frequensolverGitCommit",
        "fsMumpsRef",
        "fsMumpsCommit",
    }
)


def release_evidence_profile(evidence: dict[str, Any]) -> str:
    """Return the explicit validation profile, including the legacy v2 mapping."""

    schema = evidence.get("schemaVersion")
    if schema == LEGACY_SCHEMA:
        return SOLVER_BACKED_PROFILE
    if schema != SCHEMA:
        raise ValueError(
            f"schemaVersion must be {SCHEMA!r} or legacy {LEGACY_SCHEMA!r}, "
            f"got {schema!r}"
        )
    profile = evidence.get("validationProfile")
    if profile not in VALIDATION_PROFILES:
        raise ValueError(
            f"validationProfile must be 'standard' or 'solver-backed', got {profile!r}"
        )
    return str(profile)


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def validate_evidence(evidence: dict[str, Any], expected_commit: str) -> None:
    """Raise ``ValueError`` unless evidence proves the expected commit."""

    mismatches: list[str] = []
    try:
        profile = release_evidence_profile(evidence)
    except ValueError as exc:
        mismatches.append(str(exc))
        profile = ""

    exact_values = {
        "commit": expected_commit,
        "ciWorkflow": ".github/workflows/cicd-workflow.yml",
        "ciRequiredJob": "Required CI",
    }
    mismatches.extend(
        f"{name} must be {expected!r}, got {evidence.get(name)!r}"
        for name, expected in exact_values.items()
        if evidence.get(name) != expected
    )

    if evidence.get("schemaVersion") == SCHEMA:
        expected_status = {
            STANDARD_PROFILE: "not-run",
            SOLVER_BACKED_PROFILE: "passed",
        }.get(profile)
        if (
            expected_status is not None
            and evidence.get("solverValidationStatus") != expected_status
        ):
            mismatches.append(
                "solverValidationStatus must be "
                f"{expected_status!r} for {profile!r}, got "
                f"{evidence.get('solverValidationStatus')!r}"
            )

    ci_run_id = evidence.get("ciRunId")
    if not _positive_integer(ci_run_id):
        mismatches.append("ciRunId must be a positive integer")
    expected_ci_url = (
        f"https://github.com/FrequenSol/FrequenSolve/actions/runs/{ci_run_id}"
    )
    if evidence.get("ciRunUrl") != expected_ci_url:
        mismatches.append("ciRunUrl must identify ciRunId in FrequenSolve")

    frequensolver_release = evidence.get("frequensolverRelease", "")
    if not isinstance(frequensolver_release, str) or not FINAL_RELEASE_RE.fullmatch(
        frequensolver_release
    ):
        mismatches.append(
            "frequensolverRelease must be an immutable final release tag vX.Y.Z"
        )
    if evidence.get("sauceRef") != frequensolver_release:
        mismatches.append("sauceRef must equal frequensolverRelease")
    if not _valid_sha(evidence.get("sauceCommit")):
        mismatches.append("sauceCommit must be a lowercase 40-character Git SHA")
    expected_release_url = (
        f"https://github.com/FrequenSol/Sauce/releases/tag/{frequensolver_release}"
    )
    if evidence.get("frequensolverReleaseUrl") != expected_release_url:
        mismatches.append(
            "frequensolverReleaseUrl must identify the immutable FrequenSolver release"
        )

    if profile == STANDARD_PROFILE:
        forbidden = sorted(STANDARD_FORBIDDEN_FIELDS.intersection(evidence))
        if forbidden:
            mismatches.append(
                "standard evidence must omit Docker and solver-backed fields: "
                + ", ".join(forbidden)
            )
    elif profile == SOLVER_BACKED_PROFILE:
        mismatches.extend(
            f"{name} must be {expected!r}, got {evidence.get(name)!r}"
            for name, expected in {
                "dockerTestRef": expected_commit,
                "dockerTestCommit": expected_commit,
                "dockerTestMarker": TEST_MARKER,
                "dockerTestStatus": TEST_STATUS,
                "dockerTestArtifact": TEST_ARTIFACT,
            }.items()
            if evidence.get(name) != expected
        )
        _validate_solver_backed_evidence(
            evidence,
            expected_commit=expected_commit,
            frequensolver_release=frequensolver_release,
            mismatches=mismatches,
        )

    if mismatches:
        raise ValueError("; ".join(mismatches))


def _validate_solver_backed_evidence(
    evidence: dict[str, Any],
    *,
    expected_commit: str,
    frequensolver_release: object,
    mismatches: list[str],
) -> None:
    """Append failures for the complete Docker and solver-backed contract."""

    docker_workflow = evidence.get("dockerWorkflow", "")
    docker_workflow_ref = (
        docker_workflow.removeprefix(DOCKER_WORKFLOW_PREFIX)
        if isinstance(docker_workflow, str)
        else ""
    )
    if not _valid_sha(docker_workflow_ref):
        mismatches.append(
            "dockerWorkflow must pin the DockerImage evidence workflow to a "
            "40-character commit SHA"
        )
    if evidence.get("dockerWorkflowCommit") != docker_workflow_ref:
        mismatches.append(
            "dockerWorkflowCommit must equal the commit pinned by dockerWorkflow"
        )
    for name in ("fsMumpsRef", "fsMumpsCommit"):
        if not _valid_sha(evidence.get(name)):
            mismatches.append(f"{name} must be a lowercase 40-character Git SHA")
    if evidence.get("fsMumpsRef") != evidence.get("fsMumpsCommit"):
        mismatches.append("fsMumpsRef must equal the immutable fsMumpsCommit")
    if evidence.get("frequensolverVersion") != frequensolver_release:
        mismatches.append("frequensolverVersion must equal frequensolverRelease")
    if evidence.get("frequensolverGitCommit") != evidence.get("sauceCommit"):
        mismatches.append("frequensolverGitCommit must equal sauceCommit")
    if not evidence.get("frequensolverBuildId"):
        mismatches.append("frequensolverBuildId must be non-empty")
    for name in ("dockerCallerRunId", "dockerEvidenceRunId"):
        value = evidence.get(name)
        if not _positive_integer(value):
            mismatches.append(f"{name} must be a positive integer")
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
    docker_request_id = evidence.get("dockerRequestId", "")
    if not isinstance(docker_request_id, str) or not re.fullmatch(
        expected_request, docker_request_id
    ):
        mismatches.append(
            "dockerRequestId must identify the caller run and positive run attempt"
        )
    archive_sha = evidence.get("dockerTestArchiveSha256", "")
    if not isinstance(archive_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", archive_sha
    ):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--print-profile",
        action="store_true",
        help="Print only the validated evidence profile.",
    )
    args = parser.parse_args()
    if not SHA_RE.fullmatch(args.commit):
        parser.error("--commit must be a lowercase 40-character Git SHA")

    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            raise ValueError("release evidence must be an object")
        validate_evidence(evidence, args.commit)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    if args.print_profile:
        print(release_evidence_profile(evidence))
    else:
        print(f"release evidence is valid for {args.commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
