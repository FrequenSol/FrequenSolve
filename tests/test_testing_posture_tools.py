import io
import tarfile

import pytest

from scripts.check_coverage_thresholds import (
    coverage_percentages,
    failed_thresholds,
)
from scripts.extract_test_evidence_archive import extract_archive
from scripts.validate_heavy_test_evidence import (
    MARKER_EXPRESSION,
)
from scripts.validate_heavy_test_evidence import SCHEMA as HEAVY_SCHEMA
from scripts.validate_heavy_test_evidence import (
    validate_heavy_test_evidence,
)
from scripts.validate_release_evidence import (
    DOCKER_WORKFLOW_PREFIX,
    SCHEMA,
    TEST_ARTIFACT,
    TEST_MARKER,
    TEST_STATUS,
    validate_evidence,
)
from scripts.verify_ci_evidence import has_required_job, run_matches

COMMIT = "a" * 40


def _heavy_evidence():
    return {
        "schemaVersion": HEAVY_SCHEMA,
        "source": {
            "repository": "FrequenSol/FrequenSolve",
            "ref": COMMIT,
            "commit": COMMIT,
        },
        "selection": {
            "markerExpression": MARKER_EXPRESSION,
            "pytestAddoptsInherited": False,
            "includesIntegration": True,
            "includesVisual": True,
            "excludedMarkers": ["cloud", "hpc", "interactive"],
        },
        "pytest": {
            "status": "passed",
            "exitCode": 0,
            "junit": {
                "path": "junit.xml",
                "present": True,
                "counts": {"tests": 501, "failures": 0, "errors": 0, "skipped": 0},
            },
            "coverage": {
                "path": "coverage.xml",
                "present": True,
                "rates": {
                    "lineRate": 0.69,
                    "branchRate": 0.52,
                    "linesValid": 100,
                    "linesCovered": 69,
                    "branchesValid": 100,
                    "branchesCovered": 52,
                },
            },
            "visual": {"path": "fig_comparison.html", "present": True},
        },
    }


def test_coverage_ratchet_calculates_line_branch_and_combined_metrics():
    report = {
        "totals": {
            "covered_lines": 69,
            "num_statements": 100,
            "covered_branches": 52,
            "num_branches": 100,
            "percent_covered": 64.5,
        }
    }

    percentages = coverage_percentages(report)

    assert percentages == {"combined": 64.5, "lines": 69.0, "branches": 52.0}
    assert (
        failed_thresholds(
            percentages, {"combined": 64.5, "lines": 69.0, "branches": 51.8}
        )
        == []
    )


def test_coverage_ratchet_reports_each_regression():
    failures = failed_thresholds(
        {"combined": 64.4, "lines": 68.9, "branches": 51.7},
        {"combined": 64.5, "lines": 69.0, "branches": 51.8},
    )

    assert len(failures) == 3
    assert all("below" in failure for failure in failures)


def test_ci_evidence_requires_exact_sha_workflow_and_stable_job():
    run = {
        "head_sha": COMMIT,
        "conclusion": "success",
        "path": ".github/workflows/cicd-workflow.yml",
        "event": "push",
    }
    jobs = [{"name": "Required CI", "conclusion": "success"}]

    assert run_matches(run, COMMIT)
    assert has_required_job(jobs)
    assert not run_matches({**run, "head_sha": "b" * 40}, COMMIT)
    assert not run_matches({**run, "event": "pull_request"}, COMMIT)
    assert run_matches({**run, "event": "workflow_dispatch"}, COMMIT)
    assert not has_required_job([{"name": "unit-test", "conclusion": "success"}])


def _release_evidence():
    return {
        "schemaVersion": SCHEMA,
        "commit": COMMIT,
        "ciRunId": 123,
        "ciRunUrl": "https://github.com/FrequenSol/FrequenSolve/actions/runs/123",
        "ciWorkflow": ".github/workflows/cicd-workflow.yml",
        "ciRequiredJob": "Required CI",
        "dockerWorkflow": f"{DOCKER_WORKFLOW_PREFIX}{'b' * 40}",
        "dockerWorkflowCommit": "b" * 40,
        "dockerCallerRunId": 456,
        "dockerImageTag": "sha-bbbbbbbbbbbb",
        "dockerTestRef": COMMIT,
        "dockerTestCommit": COMMIT,
        "dockerTestMarker": TEST_MARKER,
        "dockerTestStatus": TEST_STATUS,
        "dockerTestArtifact": TEST_ARTIFACT,
        "dockerTestArchiveSha256": "e" * 64,
        "dockerTestEvidence": _heavy_evidence(),
        "sauceRef": "main",
        "sauceCommit": "c" * 40,
        "fsMumpsRef": "main",
        "fsMumpsCommit": "d" * 40,
    }


def test_release_evidence_accepts_exact_sha_ci_and_docker_proof():
    validate_evidence(_release_evidence(), COMMIT)


def test_release_evidence_rejects_mutable_docker_workflow_ref():
    evidence = _release_evidence()
    evidence["dockerWorkflow"] = f"{DOCKER_WORKFLOW_PREFIX}main"

    try:
        validate_evidence(evidence, COMMIT)
    except ValueError as exc:
        assert "40-character commit SHA" in str(exc)
    else:
        raise AssertionError("mutable Docker workflow reference was accepted")


def test_heavy_test_evidence_requires_real_test_outputs(tmp_path):
    (tmp_path / "junit.xml").write_text("<testsuites />", encoding="utf-8")
    (tmp_path / "coverage.xml").write_text("<coverage />", encoding="utf-8")
    (tmp_path / "fig_comparison.html").write_text("visual evidence", encoding="utf-8")

    validate_heavy_test_evidence(
        _heavy_evidence(),
        COMMIT,
        evidence_root=tmp_path,
    )


def test_heavy_test_evidence_rejects_failed_pytest():
    evidence = _heavy_evidence()
    evidence["pytest"]["status"] = "failed"
    evidence["pytest"]["exitCode"] = 1
    evidence["pytest"]["junit"]["counts"]["failures"] = 1

    with pytest.raises(ValueError) as exc_info:
        validate_heavy_test_evidence(evidence, COMMIT)

    message = str(exc_info.value)
    assert "pytest.status" in message
    assert "pytest.exitCode" in message
    assert "failures" in message


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda evidence: evidence["source"].update(commit="b" * 40), "source.commit"),
        (
            lambda evidence: evidence["selection"].update(markerExpression="not cloud"),
            "markerExpression",
        ),
        (
            lambda evidence: evidence["pytest"]["coverage"]["rates"].update(
                branchRate=0.0,
                branchesValid=0,
                branchesCovered=0,
            ),
            "branch",
        ),
    ],
)
def test_heavy_test_evidence_rejects_identity_selection_and_branch_mutations(
    mutation,
    message,
):
    evidence = _heavy_evidence()
    mutation(evidence)

    with pytest.raises(ValueError, match=message):
        validate_heavy_test_evidence(evidence, COMMIT)


def test_heavy_test_evidence_rejects_missing_visual_member(tmp_path):
    (tmp_path / "junit.xml").write_text("<testsuites />", encoding="utf-8")
    (tmp_path / "coverage.xml").write_text("<coverage />", encoding="utf-8")

    with pytest.raises(ValueError, match="pytest.visual.path does not exist"):
        validate_heavy_test_evidence(
            _heavy_evidence(),
            COMMIT,
            evidence_root=tmp_path,
        )


def _write_tar_member(archive, name, content=b"evidence", *, member_type=None):
    member = tarfile.TarInfo(name)
    if member_type is not None:
        member.type = member_type
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def test_evidence_archive_extractor_copies_regular_files(tmp_path):
    archive_path = tmp_path / "evidence.tar.gz"
    destination = tmp_path / "extracted"
    with tarfile.open(archive_path, "w:gz") as archive:
        _write_tar_member(archive, "frequensolve-test-evidence.json", b"{}")
        _write_tar_member(archive, "results/junit.xml", b"<testsuites />")

    extract_archive(archive_path, destination)

    assert (destination / "frequensolve-test-evidence.json").read_bytes() == b"{}"
    assert (destination / "results/junit.xml").is_file()


@pytest.mark.parametrize(
    ("name", "member_type", "message"),
    [
        ("../outside.txt", None, "escapes the destination"),
        ("link", tarfile.SYMTYPE, "regular file or directory"),
    ],
)
def test_evidence_archive_extractor_rejects_unsafe_members(
    tmp_path,
    name,
    member_type,
    message,
):
    archive_path = tmp_path / "evidence.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _write_tar_member(archive, name, member_type=member_type)

    with pytest.raises(ValueError, match=message):
        extract_archive(archive_path, tmp_path / "extracted")

    assert not (tmp_path / "outside.txt").exists()
