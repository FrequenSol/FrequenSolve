"""Validate DockerImage's machine-readable FrequenSolve test evidence."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "frequensolve-heavy-test-evidence/v1"
SCENARIO_MANIFEST_SCHEMA = "frequensolve-heavy-evidence-scenarios/v1"
SCENARIO_MANIFEST_PATH = Path(__file__).with_name("heavy_evidence_scenarios.v1.json")
SOURCE_REPOSITORY = "FrequenSol/FrequenSolve"
MARKER_EXPRESSION = "not cloud and not hpc and not interactive"
EXCLUDED_MARKERS = {"cloud", "hpc", "interactive"}
REQUIRED_SCENARIO_KINDS = {"local-solve", "solver-contract", "solver-backed-visual"}
COVERAGE_RATE_TOLERANCE = 5.1e-5
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
) -> Path | None:
    section = _mapping(value, name, errors)
    path_value = section.get("path")
    if section.get("present") is not True:
        errors.append(f"{name}.present must be true")
    if not isinstance(path_value, str) or not path_value:
        errors.append(f"{name}.path must be non-empty")
        return None
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{name}.path must stay within the evidence artifact")
        return None
    if evidence_root is None:
        return None
    member = evidence_root / path
    current = evidence_root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            errors.append(f"{name}.path must not traverse a symlink: {path}")
            return None
    if not member.exists():
        errors.append(f"{name}.path does not exist: {path}")
        return None
    if not member.is_file():
        errors.append(f"{name}.path must identify a regular file: {path}")
        return None
    try:
        member.resolve(strict=True).relative_to(evidence_root.resolve(strict=True))
    except (OSError, ValueError):
        errors.append(f"{name}.path must stay within the evidence artifact")
        return None
    return member


def load_scenario_manifest(
    manifest_path: Path = SCENARIO_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load and structurally validate the versioned heavy-evidence manifest."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"heavy-evidence scenario manifest is unreadable: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("heavy-evidence scenario manifest must be an object")
    if manifest.get("schemaVersion") != SCENARIO_MANIFEST_SCHEMA:
        raise ValueError(
            "heavy-evidence scenario manifest schemaVersion must be "
            f"{SCENARIO_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("evidenceSchemaVersion") != SCHEMA:
        raise ValueError(
            f"heavy-evidence scenario manifest must target evidence schema {SCHEMA!r}"
        )

    minimums = manifest.get("minimums")
    if not isinstance(minimums, dict):
        raise ValueError("heavy-evidence scenario manifest minimums must be an object")
    tests = minimums.get("tests")
    if not isinstance(tests, int) or isinstance(tests, bool) or tests <= 0:
        raise ValueError("heavy-evidence minimum tests must be a positive integer")
    for name in ("lineRate", "branchRate"):
        rate = minimums.get(name)
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not 0.0 < float(rate) <= 1.0
        ):
            raise ValueError(f"heavy-evidence minimum {name} must be in (0, 1]")

    scenarios = manifest.get("requiredScenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("heavy-evidence requiredScenarios must be a non-empty array")
    identifiers: set[str] = set()
    selectors: set[tuple[str, str]] = set()
    kinds: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"requiredScenarios[{index}] must be an object")
        identifier = scenario.get("id")
        kind = scenario.get("kind")
        junit = scenario.get("junit")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"requiredScenarios[{index}].id must be non-empty")
        if identifier in identifiers:
            raise ValueError(f"duplicate required scenario id: {identifier}")
        if kind not in REQUIRED_SCENARIO_KINDS:
            raise ValueError(
                f"required scenario {identifier} has unsupported kind {kind!r}"
            )
        if not isinstance(junit, dict):
            raise ValueError(f"required scenario {identifier} junit must be an object")
        classname = junit.get("classname")
        name = junit.get("name")
        if not isinstance(classname, str) or not classname:
            raise ValueError(
                f"required scenario {identifier} junit.classname must be non-empty"
            )
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"required scenario {identifier} junit.name must be non-empty"
            )
        selector = (classname, name)
        if selector in selectors:
            raise ValueError(
                f"duplicate required scenario JUnit selector: {classname}::{name}"
            )
        identifiers.add(identifier)
        selectors.add(selector)
        kinds.add(str(kind))
    missing_kinds = sorted(REQUIRED_SCENARIO_KINDS - kinds)
    if missing_kinds:
        raise ValueError(
            "heavy-evidence requiredScenarios must include: " + ", ".join(missing_kinds)
        )
    return manifest


def _xml_root(path: Path, name: str, errors: list[str]) -> ET.Element | None:
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        errors.append(f"{name}.path is not complete, valid XML: {exc}")
        return None


def _element_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _junit_counts_and_scenarios(
    path: Path,
    summary: Mapping[str, Any],
    scenarios: list[dict[str, Any]],
    errors: list[str],
) -> None:
    root = _xml_root(path, "pytest.junit", errors)
    if root is None:
        return
    if _element_name(root) not in {"testsuite", "testsuites"}:
        errors.append("pytest.junit.path root must be testsuite or testsuites")
        return

    cases: list[tuple[str, str, str]] = []
    derived = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for testcase in (
        element for element in root.iter() if _element_name(element) == "testcase"
    ):
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        outcomes = [
            _element_name(child)
            for child in testcase
            if _element_name(child) in {"failure", "error", "skipped"}
        ]
        if len(outcomes) > 1:
            errors.append(
                f"pytest.junit testcase {classname}::{name} has multiple outcomes"
            )
            status = "invalid"
        elif outcomes:
            status = outcomes[0]
        else:
            status = "passed"
        derived["tests"] += 1
        if status in {"failure", "error", "skipped"}:
            key = {
                "failure": "failures",
                "error": "errors",
                "skipped": "skipped",
            }[status]
            derived[key] += 1
        cases.append((classname, name, status))

    for name, measured in derived.items():
        if summary.get(name) != measured:
            errors.append(
                f"pytest.junit.counts.{name} must match JUnit XML ({measured})"
            )

    for scenario in scenarios:
        selector = (
            scenario["junit"]["classname"],
            scenario["junit"]["name"],
        )
        matches = [case for case in cases if case[:2] == selector]
        label = f"{scenario['id']} ({selector[0]}::{selector[1]})"
        if len(matches) != 1:
            errors.append(
                f"required scenario {label} must appear exactly once in JUnit XML; "
                f"found {len(matches)}"
            )
        elif matches[0][2] != "passed":
            errors.append(
                f"required scenario {label} must pass; status was {matches[0][2]}"
            )


def _coverage_xml_rates(
    path: Path,
    summary: Mapping[str, Any],
    errors: list[str],
) -> dict[str, float | int] | None:
    root = _xml_root(path, "pytest.coverage", errors)
    if root is None:
        return None
    if _element_name(root) != "coverage":
        errors.append("pytest.coverage.path root must be coverage")
        return None
    attributes = {
        "lineRate": ("line-rate", float),
        "branchRate": ("branch-rate", float),
        "linesValid": ("lines-valid", int),
        "linesCovered": ("lines-covered", int),
        "branchesValid": ("branches-valid", int),
        "branchesCovered": ("branches-covered", int),
    }
    measured: dict[str, float | int] = {}
    for name, (attribute, converter) in attributes.items():
        try:
            measured[name] = converter(root.attrib[attribute])
        except (KeyError, ValueError):
            errors.append(f"pytest.coverage.path has invalid {attribute!r}")
            return None
    for name, value in measured.items():
        summary_value = summary.get(name)
        if isinstance(value, float):
            matches = (
                isinstance(summary_value, (int, float))
                and not isinstance(summary_value, bool)
                and abs(float(summary_value) - value) <= 1.0e-9
            )
        else:
            matches = summary_value == value
        if not matches:
            errors.append(
                f"pytest.coverage.rates.{name} must match coverage XML ({value})"
            )
    return measured


def _enforce_manifest_minimums(
    tests: Any,
    skipped: Any,
    rates: Mapping[str, Any],
    minimums: Mapping[str, Any],
    errors: list[str],
) -> None:
    minimum_tests = minimums["tests"]
    if (
        isinstance(tests, int)
        and not isinstance(tests, bool)
        and isinstance(skipped, int)
        and not isinstance(skipped, bool)
        and skipped >= 0
    ):
        passed = tests - skipped
        if passed < minimum_tests:
            errors.append(
                f"pytest.junit passed test count {passed} is below manifest floor "
                f"{minimum_tests}"
            )
    for name in ("lineRate", "branchRate"):
        value = rates.get(name)
        minimum = minimums[name]
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) < float(minimum)
        ):
            errors.append(
                f"pytest.coverage.rates.{name} {float(value):.4f} is below "
                f"manifest floor {float(minimum):.4f}"
            )


def validate_heavy_test_evidence(
    evidence: dict[str, Any],
    expected_commit: str,
    *,
    evidence_root: Path | None = None,
) -> None:
    """Raise ``ValueError`` unless heavy evidence proves the exact commit."""
    errors: list[str] = []
    manifest = load_scenario_manifest()
    scenarios = manifest["requiredScenarios"]
    minimums = manifest["minimums"]
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
    deselected = selection.get("deselectedTests", [])
    if not isinstance(deselected, list) or any(
        not isinstance(node, str) for node in deselected
    ):
        errors.append("selection.deselectedTests must be an array of strings")
        deselected = []
    for scenario in scenarios:
        junit_selector = scenario["junit"]
        node_id = (
            junit_selector["classname"].replace(".", "/")
            + ".py::"
            + junit_selector["name"]
        )
        if node_id in deselected:
            errors.append(f"required scenario {scenario['id']} must not be deselected")

    pytest_evidence = _mapping(evidence.get("pytest"), "pytest", errors)
    if pytest_evidence.get("status") != "passed":
        errors.append("pytest.status must be 'passed'")
    if pytest_evidence.get("exitCode") != 0:
        errors.append("pytest.exitCode must be 0")

    junit = _mapping(pytest_evidence.get("junit"), "pytest.junit", errors)
    junit_path = _artifact_member(junit, "pytest.junit", evidence_root, errors)
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
    if junit_path is not None:
        _junit_counts_and_scenarios(junit_path, counts, scenarios, errors)

    coverage = _mapping(pytest_evidence.get("coverage"), "pytest.coverage", errors)
    coverage_path = _artifact_member(coverage, "pytest.coverage", evidence_root, errors)
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
                and abs(float(rate) - covered / valid) > COVERAGE_RATE_TOLERANCE
            ):
                errors.append(
                    f"pytest.coverage.rates.{name} does not match measured counts"
                )
    _enforce_manifest_minimums(tests, skipped, rates, minimums, errors)
    if coverage_path is not None:
        _coverage_xml_rates(coverage_path, rates, errors)

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
