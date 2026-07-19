"""Validate DockerImage's machine-readable FrequenSolve test evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "frequensolve-heavy-test-evidence/v1"
SOURCE_REPOSITORY = "FrequenSol/FrequenSolve"
MARKER_EXPRESSION = "not cloud and not hpc and not interactive"
EXCLUDED_MARKERS = {"cloud", "hpc", "interactive"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _mapping(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    errors.append(f"{name} must be an object")
    return {}


def _artifact_member(
    value: Any,
    name: str,
    evidence_root: Path | None,
    errors: list[str],
) -> None:
    section = _mapping(value, name, errors)
    path_value = section.get("path")
    if section.get("present") is not True:
        errors.append(f"{name}.present must be true")
    if not isinstance(path_value, str) or not path_value:
        errors.append(f"{name}.path must be non-empty")
        return
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{name}.path must stay within the evidence artifact")
        return
    if evidence_root is None:
        return
    member = evidence_root / path
    if not member.exists():
        errors.append(f"{name}.path does not exist: {path}")
    elif member.is_dir() and not any(child.is_file() for child in member.rglob("*")):
        errors.append(f"{name}.path contains no evidence files: {path}")


def validate_heavy_test_evidence(
    evidence: dict[str, Any],
    expected_commit: str,
    *,
    evidence_root: Path | None = None,
) -> None:
    """Raise ``ValueError`` unless heavy evidence proves the exact commit."""
    errors: list[str] = []
    if evidence.get("schemaVersion") != SCHEMA:
        errors.append(f"schemaVersion must be {SCHEMA!r}")

    source = _mapping(evidence.get("source"), "source", errors)
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "ref": expected_commit,
        "commit": expected_commit,
    }
    for name, expected in expected_source.items():
        if source.get(name) != expected:
            errors.append(f"source.{name} must be {expected!r}")

    selection = _mapping(evidence.get("selection"), "selection", errors)
    if selection.get("markerExpression") != MARKER_EXPRESSION:
        errors.append(f"selection.markerExpression must be {MARKER_EXPRESSION!r}")
    if selection.get("pytestAddoptsInherited") is not False:
        errors.append("selection.pytestAddoptsInherited must be false")
    if selection.get("includesIntegration") is not True:
        errors.append("selection.includesIntegration must be true")
    if selection.get("includesVisual") is not True:
        errors.append("selection.includesVisual must be true")
    excluded = selection.get("excludedMarkers")
    if not isinstance(excluded, list) or set(excluded) != EXCLUDED_MARKERS:
        errors.append(
            "selection.excludedMarkers must contain cloud, hpc, and interactive"
        )

    pytest_evidence = _mapping(evidence.get("pytest"), "pytest", errors)
    if pytest_evidence.get("status") != "passed":
        errors.append("pytest.status must be 'passed'")
    if pytest_evidence.get("exitCode") != 0:
        errors.append("pytest.exitCode must be 0")

    junit = _mapping(pytest_evidence.get("junit"), "pytest.junit", errors)
    _artifact_member(junit, "pytest.junit", evidence_root, errors)
    counts = _mapping(junit.get("counts"), "pytest.junit.counts", errors)
    tests = counts.get("tests")
    if not isinstance(tests, int) or isinstance(tests, bool) or tests <= 0:
        errors.append("pytest.junit.counts.tests must be a positive integer")
    for name in ("failures", "errors"):
        if counts.get(name) != 0:
            errors.append(f"pytest.junit.counts.{name} must be 0")
    skipped = counts.get("skipped")
    if not isinstance(skipped, int) or isinstance(skipped, bool) or skipped < 0:
        errors.append("pytest.junit.counts.skipped must be a non-negative integer")

    coverage = _mapping(pytest_evidence.get("coverage"), "pytest.coverage", errors)
    _artifact_member(coverage, "pytest.coverage", evidence_root, errors)
    rates = _mapping(coverage.get("rates"), "pytest.coverage.rates", errors)
    for prefix in ("lines", "branches"):
        valid = rates.get(f"{prefix}Valid")
        covered = rates.get(f"{prefix}Covered")
        if (
            not isinstance(valid, int)
            or isinstance(valid, bool)
            or valid <= 0
            or not isinstance(covered, int)
            or isinstance(covered, bool)
            or not 0 < covered <= valid
        ):
            errors.append(
                f"pytest.coverage.rates.{prefix}Valid/{prefix}Covered must "
                "describe positive measured coverage"
            )
    for name, valid_name, covered_name in (
        ("lineRate", "linesValid", "linesCovered"),
        ("branchRate", "branchesValid", "branchesCovered"),
    ):
        rate = rates.get(name)
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not 0.0 < float(rate) <= 1.0
        ):
            errors.append(
                f"pytest.coverage.rates.{name} must be greater than 0 and at most 1"
            )
        else:
            valid = rates.get(valid_name)
            covered = rates.get(covered_name)
            if (
                isinstance(valid, int)
                and not isinstance(valid, bool)
                and valid > 0
                and isinstance(covered, int)
                and not isinstance(covered, bool)
                and abs(float(rate) - covered / valid) > 1.0e-3
            ):
                errors.append(
                    f"pytest.coverage.rates.{name} does not match measured counts"
                )

    visual = pytest_evidence.get("visual")
    _artifact_member(visual, "pytest.visual", evidence_root, errors)
    if isinstance(visual, Mapping) and Path(str(visual.get("path", ""))).name != (
        "fig_comparison.html"
    ):
        errors.append("pytest.visual.path must identify fig_comparison.html")

    if errors:
        raise ValueError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if not SHA_RE.fullmatch(args.commit):
        parser.error("--commit must be a lowercase 40-character Git SHA")

    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        validate_heavy_test_evidence(
            evidence,
            args.commit,
            evidence_root=args.evidence.parent,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"heavy test evidence is valid for {args.commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
