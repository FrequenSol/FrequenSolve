import hashlib
import io
import json
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from frequensolve.frequensolver import load_frequensolver_compatibility
from scripts.check_coverage_thresholds import (
    coverage_percentages,
    failed_thresholds,
)
from scripts.compare_release_evidence_pair import (
    compare_release_evidence_assets,
    compare_release_evidence_pairs,
)
from scripts.extract_test_evidence_archive import extract_archive
from scripts.materialize_frequensolver_compatibility import manifest_from_evidence
from scripts.validate_frequensolver_identity import validate_identity
from scripts.validate_heavy_test_evidence import (
    MARKER_EXPRESSION,
)
from scripts.validate_heavy_test_evidence import SCHEMA as HEAVY_SCHEMA
from scripts.validate_heavy_test_evidence import (
    load_scenario_manifest,
    validate_heavy_test_evidence,
)
from scripts.validate_release_evidence import (
    DOCKER_WORKFLOW_PREFIX,
    LEGACY_SCHEMA,
    SCHEMA,
    SOLVER_BACKED_PROFILE,
    STANDARD_PROFILE,
    TEST_ARTIFACT,
    TEST_MARKER,
    TEST_STATUS,
    release_evidence_profile,
    validate_evidence,
)
from scripts.verify_ci_evidence import has_required_job, run_matches

COMMIT = "a" * 40
REPO_ROOT = Path(__file__).resolve().parents[1]


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
                "counts": {"tests": 538, "failures": 0, "errors": 0, "skipped": 0},
            },
            "coverage": {
                "path": "coverage.xml",
                "present": True,
                "rates": {
                    "lineRate": 0.7203,
                    "branchRate": 0.552,
                    "linesValid": 10000,
                    "linesCovered": 7203,
                    "branchesValid": 10000,
                    "branchesCovered": 5520,
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
    docker_request_id = "frequensolve-rc-456-1-0123456789abcdef0123456789abcdef"
    docker_dispatch_evidence = {
        "schemaVersion": "frequensolve-docker-dispatch-evidence/v1",
        "runId": 789,
        "runUrl": (
            "https://github.com/FrequenSol/FrequenSolveDockerImage/actions/runs/789"
        ),
        "requestId": docker_request_id,
        "workflowRepository": "FrequenSol/FrequenSolveDockerImage",
        "workflowPath": ".github/workflows/cicd-workflow.yml",
        "workflowCommit": "b" * 40,
        "sourceRef": "main",
        "sourceCommit": "b" * 40,
        "sauceRef": "v0.1.0",
        "sauceCommit": "c" * 40,
        "fsMumpsRef": "d" * 40,
        "fsMumpsCommit": "d" * 40,
        "frequensolveSource": "git",
        "frequensolveRef": COMMIT,
        "frequensolveCommit": COMMIT,
        "disablePush": True,
        "testMarker": TEST_MARKER,
        "testStatus": TEST_STATUS,
        "testArtifact": TEST_ARTIFACT,
    }
    return {
        "schemaVersion": SCHEMA,
        "validationProfile": SOLVER_BACKED_PROFILE,
        "solverValidationStatus": "passed",
        "commit": COMMIT,
        "ciRunId": 123,
        "ciRunUrl": "https://github.com/FrequenSol/FrequenSolve/actions/runs/123",
        "ciWorkflow": ".github/workflows/cicd-workflow.yml",
        "ciRequiredJob": "Required CI",
        "dockerWorkflow": f"{DOCKER_WORKFLOW_PREFIX}{'b' * 40}",
        "dockerWorkflowCommit": "b" * 40,
        "dockerCallerRunId": 456,
        "dockerCallerRunUrl": (
            "https://github.com/FrequenSol/FrequenSolve/actions/runs/456"
        ),
        "dockerEvidenceRunId": 789,
        "dockerEvidenceRunUrl": (
            "https://github.com/FrequenSol/FrequenSolveDockerImage/actions/runs/789"
        ),
        "dockerRequestId": docker_request_id,
        "dockerImageTag": "b" * 40,
        "dockerTestRef": COMMIT,
        "dockerTestCommit": COMMIT,
        "dockerTestMarker": TEST_MARKER,
        "dockerTestStatus": TEST_STATUS,
        "dockerTestArtifact": TEST_ARTIFACT,
        "dockerTestArchiveSha256": "e" * 64,
        "dockerTestEvidence": _heavy_evidence(),
        "dockerDispatchEvidence": docker_dispatch_evidence,
        "sauceRef": "v0.1.0",
        "sauceCommit": "c" * 40,
        "frequensolverRelease": "v0.1.0",
        "frequensolverReleaseUrl": (
            "https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"
        ),
        "frequensolverVersion": "v0.1.0",
        "frequensolverBuildId": "release-v0.1.0",
        "frequensolverGitCommit": "c" * 40,
        "fsMumpsRef": "d" * 40,
        "fsMumpsCommit": "d" * 40,
    }


def _standard_release_evidence():
    return {
        "schemaVersion": SCHEMA,
        "validationProfile": STANDARD_PROFILE,
        "solverValidationStatus": "not-run",
        "commit": COMMIT,
        "ciRunId": 123,
        "ciRunUrl": "https://github.com/FrequenSol/FrequenSolve/actions/runs/123",
        "ciWorkflow": ".github/workflows/cicd-workflow.yml",
        "ciRequiredJob": "Required CI",
        "sauceRef": "v0.1.0",
        "sauceCommit": "c" * 40,
        "frequensolverRelease": "v0.1.0",
        "frequensolverReleaseUrl": (
            "https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"
        ),
    }


def test_release_evidence_accepts_exact_sha_ci_and_docker_proof():
    validate_evidence(_release_evidence(), COMMIT)


def test_release_evidence_accepts_standard_exact_tree_ci_and_solver_identity():
    evidence = _standard_release_evidence()

    validate_evidence(evidence, COMMIT)

    assert release_evidence_profile(evidence) == STANDARD_PROFILE


def test_release_evidence_maps_legacy_v2_to_solver_backed():
    evidence = _release_evidence()
    evidence["schemaVersion"] = LEGACY_SCHEMA
    evidence.pop("validationProfile")
    evidence.pop("solverValidationStatus")

    validate_evidence(evidence, COMMIT)

    assert release_evidence_profile(evidence) == SOLVER_BACKED_PROFILE


def test_standard_release_evidence_forbids_solver_backed_fields():
    evidence = _standard_release_evidence()
    evidence["dockerEvidenceRunId"] = 789

    with pytest.raises(ValueError, match="standard evidence must omit"):
        validate_evidence(evidence, COMMIT)


def test_materialized_frequensolver_manifest_loads(tmp_path):
    manifest = manifest_from_evidence(
        _release_evidence(),
        package_release="0.3.0",
        package_commit=COMMIT,
    )
    manifest_path = tmp_path / "frequensolver_compatibility.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_frequensolver_compatibility(manifest_path)

    assert loaded.package_release == "0.3.0"
    assert loaded.preferred_frequensolver.release == "v0.1.0"
    assert loaded.evidence_run_id == 789
    assert loaded.evidence_url == (
        "https://github.com/FrequenSol/FrequenSolveDockerImage/actions/runs/789"
    )
    assert loaded.validation_profile == SOLVER_BACKED_PROFILE
    assert loaded.solver_backed


def test_standard_materialized_manifest_is_ci_backed_but_not_solver_backed(tmp_path):
    manifest = manifest_from_evidence(
        _standard_release_evidence(),
        package_release="0.3.0",
        package_commit=COMMIT,
    )
    manifest_path = tmp_path / "frequensolver_compatibility.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_frequensolver_compatibility(manifest_path)

    assert loaded.preferred_frequensolver.release == "v0.1.0"
    assert loaded.validation_profile == STANDARD_PROFILE
    assert not loaded.solver_backed
    assert loaded.evidence_run_id == 123
    assert loaded.evidence_url == (
        "https://github.com/FrequenSol/FrequenSolve/actions/runs/123"
    )


def test_release_evidence_rejects_mutable_docker_workflow_ref():
    evidence = _release_evidence()
    evidence["dockerWorkflow"] = f"{DOCKER_WORKFLOW_PREFIX}main"

    try:
        validate_evidence(evidence, COMMIT)
    except ValueError as exc:
        assert "40-character commit SHA" in str(exc)
    else:
        raise AssertionError("mutable Docker workflow reference was accepted")


def test_release_evidence_rejects_mutable_frequensolver_ref():
    evidence = _release_evidence()
    evidence["sauceRef"] = "main"
    evidence["frequensolverRelease"] = "main"
    evidence["frequensolverVersion"] = "main"

    with pytest.raises(ValueError, match="immutable final release tag"):
        validate_evidence(evidence, COMMIT)


def test_release_evidence_rejects_wrong_downstream_run_identity():
    evidence = _release_evidence()
    evidence["dockerEvidenceRunUrl"] = (
        "https://github.com/FrequenSol/FrequenSolve/actions/runs/789"
    )

    with pytest.raises(ValueError, match="dockerEvidenceRunUrl"):
        validate_evidence(evidence, COMMIT)


def test_release_evidence_requires_high_entropy_correlated_request():
    evidence = _release_evidence()
    evidence["dockerRequestId"] = "frequensolve-rc-456-1"

    with pytest.raises(ValueError, match="dockerRequestId"):
        validate_evidence(evidence, COMMIT)


def test_release_evidence_requires_bound_dispatch_manifest():
    evidence = _release_evidence()
    evidence["dockerDispatchEvidence"]["fsMumpsCommit"] = "f" * 40

    with pytest.raises(ValueError, match="dockerDispatchEvidence.*fsMumpsCommit"):
        validate_evidence(evidence, COMMIT)


def test_release_evidence_requires_exact_fs_mumps_ref():
    evidence = _release_evidence()
    evidence["fsMumpsRef"] = "main"

    with pytest.raises(ValueError, match="fsMumpsRef"):
        validate_evidence(evidence, COMMIT)


def test_release_evidence_requires_pinned_docker_image_tag():
    evidence = _release_evidence()
    evidence["dockerImageTag"] = "latest"

    with pytest.raises(ValueError, match="dockerImageTag"):
        validate_evidence(evidence, COMMIT)


def _write_release_evidence_pair(
    root,
    evidence,
    archive_content=b"sealed heavy evidence",
):
    root.mkdir()
    archive = root / "frequensolve-test-evidence.tar.gz"
    archive.write_bytes(archive_content)
    evidence = {
        **evidence,
        "dockerTestArchiveSha256": hashlib.sha256(archive_content).hexdigest(),
    }
    evidence_path = root / "release-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return evidence_path, archive


def test_release_evidence_pair_comparison_accepts_identical_sealed_assets(tmp_path):
    expected_evidence, expected_archive = _write_release_evidence_pair(
        tmp_path / "expected",
        _release_evidence(),
    )
    actual_evidence, actual_archive = _write_release_evidence_pair(
        tmp_path / "actual",
        _release_evidence(),
    )
    actual_evidence.write_text(
        json.dumps(json.loads(actual_evidence.read_text()), indent=2),
        encoding="utf-8",
    )

    compare_release_evidence_pairs(
        expected_evidence,
        expected_archive,
        actual_evidence,
        actual_archive,
    )


def test_release_evidence_pair_comparison_rejects_stale_frequensolver_metadata(
    tmp_path,
):
    expected_evidence, expected_archive = _write_release_evidence_pair(
        tmp_path / "expected",
        _release_evidence(),
    )
    stale = _release_evidence()
    stale.update(
        sauceRef="v0.0.9",
        frequensolverRelease="v0.0.9",
        frequensolverVersion="v0.0.9",
    )
    actual_evidence, actual_archive = _write_release_evidence_pair(
        tmp_path / "actual",
        stale,
    )

    with pytest.raises(ValueError, match="frequensolverRelease"):
        compare_release_evidence_pairs(
            expected_evidence,
            expected_archive,
            actual_evidence,
            actual_archive,
        )


def test_release_evidence_pair_comparison_rejects_changed_archive(tmp_path):
    expected_evidence, expected_archive = _write_release_evidence_pair(
        tmp_path / "expected",
        _release_evidence(),
    )
    actual_evidence, actual_archive = _write_release_evidence_pair(
        tmp_path / "actual",
        _release_evidence(),
        archive_content=b"different heavy evidence",
    )

    with pytest.raises(ValueError, match="frequensolve-test-evidence.tar.gz differs"):
        compare_release_evidence_pairs(
            expected_evidence,
            expected_archive,
            actual_evidence,
            actual_archive,
        )


def test_standard_release_evidence_comparison_accepts_json_only(tmp_path):
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    expected.write_text(json.dumps(_standard_release_evidence()), encoding="utf-8")
    actual.write_text(
        json.dumps(_standard_release_evidence(), indent=2),
        encoding="utf-8",
    )

    compare_release_evidence_assets(expected, actual)


def test_standard_release_evidence_comparison_rejects_heavy_archive(tmp_path):
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    archive = tmp_path / "frequensolve-test-evidence.tar.gz"
    expected.write_text(json.dumps(_standard_release_evidence()), encoding="utf-8")
    actual.write_text(json.dumps(_standard_release_evidence()), encoding="utf-8")
    archive.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="must not include a heavy evidence archive"):
        compare_release_evidence_assets(
            expected,
            actual,
            expected_archive_path=archive,
            actual_archive_path=archive,
        )


def test_materializes_package_compatibility_from_validated_release_evidence():
    manifest = manifest_from_evidence(
        _release_evidence(),
        package_release="0.3.0rc1",
        package_commit=COMMIT,
    )

    assert manifest == {
        "schema": "frequensolve-frequensolver-compatibility/v2",
        "package_release": "0.3.0rc1",
        "preferred_frequensolver": {
            "release": "v0.1.0",
            "git_commit": "c" * 40,
            "release_url": ("https://github.com/FrequenSol/Sauce/releases/tag/v0.1.0"),
        },
        "validation": {
            "profile": "solver-backed",
            "solver_backed": True,
            "run_id": 789,
            "url": (
                "https://github.com/FrequenSol/FrequenSolveDockerImage/actions/runs/789"
            ),
        },
    }


def test_frequensolver_identity_evidence_requires_exact_release_build():
    identity = {
        "schema": "frequensolver-identity-1",
        "product": "FrequenSolver",
        "version": "v0.1.0",
        "build_id": "release-v0.1.0",
        "git_commit": "c" * 40,
    }

    validate_identity(
        identity,
        expected_version="v0.1.0",
        expected_commit="c" * 40,
        expected_build_id="release-v0.1.0",
    )

    with pytest.raises(ValueError, match="git_commit"):
        validate_identity(
            {**identity, "git_commit": "d" * 40},
            expected_version="v0.1.0",
            expected_commit="c" * 40,
            expected_build_id="release-v0.1.0",
        )


@pytest.mark.parametrize("field", ["version", "build_id"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        "release\noutput",
        "release\x1foutput",
        "release-\N{LATIN SMALL LETTER E WITH ACUTE}",
    ],
    ids=["newline", "control", "non-ascii"],
)
def test_frequensolver_identity_rejects_unsafe_text_fields(field, invalid_value):
    identity = {
        "schema": "frequensolver-identity-1",
        "product": "FrequenSolver",
        "version": "v0.1.0",
        "build_id": "release-v0.1.0",
        "git_commit": "c" * 40,
    }
    identity[field] = invalid_value

    with pytest.raises(
        ValueError,
        match=rf"{field} must be a non-empty printable single-line ASCII string",
    ):
        validate_identity(
            identity,
            expected_version=identity["version"],
            expected_commit="c" * 40,
            expected_build_id=identity["build_id"],
        )


def test_frequensolver_identity_accepts_all_printable_single_line_ascii():
    printable_ascii = "".join(chr(codepoint) for codepoint in range(0x20, 0x7F))
    identity = {
        "schema": "frequensolver-identity-1",
        "product": "FrequenSolver",
        "version": printable_ascii,
        "build_id": printable_ascii,
        "git_commit": "c" * 40,
    }

    validate_identity(
        identity,
        expected_version=printable_ascii,
        expected_commit="c" * 40,
        expected_build_id=printable_ascii,
    )


def test_frequensolver_identity_cli_accepts_leading_hyphen_build_id(tmp_path):
    build_id = "-release-v0.1.0"
    identity_file = tmp_path / "frequensolver-identity.json"
    identity_file.write_text(
        json.dumps(
            {
                "schema": "frequensolver-identity-1",
                "product": "FrequenSolver",
                "version": "v0.1.0",
                "build_id": build_id,
                "git_commit": "c" * 40,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/validate_frequensolver_identity.py"),
            str(identity_file),
            "--version=v0.1.0",
            f"--commit={'c' * 40}",
            f"--build-id={build_id}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _write_heavy_test_artifacts(
    tmp_path,
    evidence,
    *,
    missing_scenario=None,
    duplicate_scenario=None,
    renamed_scenario=None,
    scenario_outcome=None,
):
    manifest = load_scenario_manifest()
    cases = []
    for scenario in manifest["requiredScenarios"]:
        if scenario["id"] == missing_scenario:
            continue
        name = scenario["junit"]["name"]
        if scenario["id"] == renamed_scenario:
            name += "_renamed"
        case = {
            "classname": scenario["junit"]["classname"],
            "name": name,
            "outcome": (
                scenario_outcome[1]
                if scenario_outcome and scenario["id"] == scenario_outcome[0]
                else None
            ),
        }
        cases.append(case)
        if scenario["id"] == duplicate_scenario:
            cases.append(dict(case))

    minimum_tests = manifest["minimums"]["tests"]
    filler_count = minimum_tests - len(cases)
    cases.extend(
        {
            "classname": "tests.test_release_baseline",
            "name": f"test_baseline_{index:04d}",
            "outcome": None,
        }
        for index in range(filler_count)
    )
    testsuites = ET.Element("testsuites")
    testsuite = ET.SubElement(testsuites, "testsuite")
    derived = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            {"classname": case["classname"], "name": case["name"]},
        )
        if case["outcome"]:
            ET.SubElement(testcase, case["outcome"])
            derived[
                {
                    "failure": "failures",
                    "error": "errors",
                    "skipped": "skipped",
                }[case["outcome"]]
            ] += 1
    testsuite.attrib.update({name: str(value) for name, value in derived.items()})
    ET.ElementTree(testsuites).write(
        tmp_path / "junit.xml", encoding="utf-8", xml_declaration=True
    )

    rates = evidence["pytest"]["coverage"]["rates"]
    ET.ElementTree(
        ET.Element(
            "coverage",
            {
                "line-rate": str(rates["lineRate"]),
                "branch-rate": str(rates["branchRate"]),
                "lines-valid": str(rates["linesValid"]),
                "lines-covered": str(rates["linesCovered"]),
                "branches-valid": str(rates["branchesValid"]),
                "branches-covered": str(rates["branchesCovered"]),
            },
        )
    ).write(tmp_path / "coverage.xml", encoding="utf-8", xml_declaration=True)
    (tmp_path / "fig_comparison.html").write_text("visual evidence", encoding="utf-8")
    evidence["pytest"]["junit"]["counts"] = derived


def test_heavy_test_evidence_requires_real_test_outputs(tmp_path):
    evidence = _heavy_evidence()
    _write_heavy_test_artifacts(tmp_path, evidence)

    validate_heavy_test_evidence(
        evidence,
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
    evidence = _heavy_evidence()
    _write_heavy_test_artifacts(tmp_path, evidence)
    (tmp_path / "fig_comparison.html").unlink()

    with pytest.raises(ValueError, match="pytest.visual.path does not exist"):
        validate_heavy_test_evidence(
            evidence,
            COMMIT,
            evidence_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"missing_scenario": "local-2d-acoustic-solve"},
            "local-2d-acoustic-solve.*found 0",
        ),
        (
            {"duplicate_scenario": "acquisition-v2-init-sizing"},
            "acquisition-v2-init-sizing.*found 2",
        ),
        (
            {"renamed_scenario": "local-2d-acoustic-gather-visual"},
            "local-2d-acoustic-gather-visual.*found 0",
        ),
        (
            {
                "scenario_outcome": (
                    "local-2d-acoustic-common-frequency-visual",
                    "skipped",
                )
            },
            "local-2d-acoustic-common-frequency-visual.*must pass",
        ),
    ],
)
def test_heavy_test_evidence_requires_each_manifest_scenario_once_and_passed(
    tmp_path,
    mutation,
    message,
):
    evidence = _heavy_evidence()
    _write_heavy_test_artifacts(tmp_path, evidence, **mutation)

    with pytest.raises(ValueError, match=message):
        validate_heavy_test_evidence(evidence, COMMIT, evidence_root=tmp_path)


def test_heavy_test_evidence_rejects_inconsistent_junit_summary(tmp_path):
    evidence = _heavy_evidence()
    _write_heavy_test_artifacts(tmp_path, evidence)
    evidence["pytest"]["junit"]["counts"]["tests"] += 1

    with pytest.raises(ValueError, match="counts.tests must match JUnit XML"):
        validate_heavy_test_evidence(evidence, COMMIT, evidence_root=tmp_path)


def test_heavy_test_evidence_rejects_inconsistent_coverage_summary(tmp_path):
    evidence = _heavy_evidence()
    _write_heavy_test_artifacts(tmp_path, evidence)
    evidence["pytest"]["coverage"]["rates"]["lineRate"] = 0.73

    with pytest.raises(ValueError, match="lineRate must match coverage XML"):
        validate_heavy_test_evidence(evidence, COMMIT, evidence_root=tmp_path)


@pytest.mark.parametrize("artifact", ["junit.xml", "coverage.xml"])
def test_heavy_test_evidence_rejects_truncated_xml(tmp_path, artifact):
    evidence = _heavy_evidence()
    _write_heavy_test_artifacts(tmp_path, evidence)
    (tmp_path / artifact).write_text("<truncated", encoding="utf-8")

    with pytest.raises(ValueError, match="not complete, valid XML"):
        validate_heavy_test_evidence(evidence, COMMIT, evidence_root=tmp_path)


def test_heavy_test_evidence_rejects_path_escape(tmp_path):
    evidence = _heavy_evidence()
    _write_heavy_test_artifacts(tmp_path, evidence)
    evidence["pytest"]["junit"]["path"] = "../junit.xml"

    with pytest.raises(ValueError, match="must stay within the evidence artifact"):
        validate_heavy_test_evidence(evidence, COMMIT, evidence_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence["pytest"]["junit"]["counts"].update(tests=537),
            "below manifest floor 538",
        ),
        (
            lambda evidence: evidence["pytest"]["coverage"]["rates"].update(
                lineRate=0.72,
                linesCovered=7200,
            ),
            "lineRate.*below manifest floor",
        ),
        (
            lambda evidence: evidence["pytest"]["coverage"]["rates"].update(
                branchRate=0.5519,
                branchesCovered=5519,
            ),
            "branchRate.*below manifest floor",
        ),
    ],
)
def test_heavy_test_evidence_rejects_below_manifest_floors(mutation, message):
    evidence = _heavy_evidence()
    mutation(evidence)

    with pytest.raises(ValueError, match=message):
        validate_heavy_test_evidence(evidence, COMMIT)


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
