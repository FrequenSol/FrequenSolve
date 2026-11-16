"""Enforce FrequenSolve's line, branch, and combined coverage ratchet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_COMBINED = 64.5
DEFAULT_LINES = 69.0
# The measured baseline is 51.876%, conventionally reported as 52%.
DEFAULT_BRANCHES = 51.8


def coverage_percentages(report: dict[str, Any]) -> dict[str, float]:
    """Return normalized percentages from a coverage.py JSON report."""
    totals = report["totals"]
    statements = int(totals["num_statements"])
    branches = int(totals["num_branches"])
    if statements <= 0 or branches <= 0:
        raise ValueError("coverage report must contain statements and branches")

    return {
        "combined": float(totals["percent_covered"]),
        "lines": 100.0 * int(totals["covered_lines"]) / statements,
        "branches": 100.0 * int(totals["covered_branches"]) / branches,
    }


def failed_thresholds(
    percentages: dict[str, float], thresholds: dict[str, float]
) -> list[str]:
    """Describe every metric below its ratcheted threshold."""
    return [
        f"{name} coverage {percentages[name]:.3f}% is below {minimum:.3f}%"
        for name, minimum in thresholds.items()
        if percentages[name] + 1e-9 < minimum
    ]


def _risk_percentages(totals: dict[str, Any]) -> dict[str, float]:
    fields = ("num_statements", "covered_lines", "num_branches", "covered_branches")
    if any(type(totals.get(field)) is not int or totals[field] < 0 for field in fields):
        raise ValueError(
            "risk coverage requires nonnegative integer line/branch counts"
        )
    statements, lines, branches, covered_branches = (totals[field] for field in fields)
    if statements == 0 or lines > statements or covered_branches > branches:
        raise ValueError(
            "risk coverage contains empty or inconsistent executable counts"
        )
    return {
        "lines": 100.0 * lines / statements,
        "branches": 100.0 * covered_branches / branches if branches else 100.0,
        "combined": 100.0 * (lines + covered_branches) / (statements + branches),
    }


def _risk_floors(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"lines", "branches"}:
        raise ValueError("risk floors must contain exactly lines and branches")
    if any(
        type(floor) not in (int, float) or not 0 < floor <= 100
        for floor in value.values()
    ):
        raise ValueError("risk floors must be finite percentages greater than zero")
    return value


def risk_area_results(
    report: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Evaluate measured risk areas and their module minima from coverage data."""
    if policy.get("schema") != "frequensolve-risk-coverage-1":
        raise ValueError("unsupported risk coverage policy schema")
    if report.get("meta", {}).get("branch_coverage") is not True:
        raise ValueError("risk coverage requires a branch-enabled report")
    areas = policy.get("areas")
    if not isinstance(areas, list) or not areas:
        raise ValueError("risk coverage policy requires at least one area")
    results: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for area in areas:
        name = area.get("name")
        if not isinstance(name, str) or not name or name in results:
            raise ValueError("risk area names must be nonempty and unique")
        prefixes = area.get("prefixes")
        required = area.get("required_modules")
        if (
            not isinstance(prefixes, list)
            or not prefixes
            or not isinstance(required, list)
            or not required
            or any(
                not isinstance(path, str)
                or not path.startswith("src/frequensolve/")
                or ".." in path.split("/")
                for path in [*prefixes, *required]
            )
        ):
            raise ValueError(
                f"{name}: explicit package prefixes and modules are required"
            )
        floors = _risk_floors(area.get("floors"))
        module_floors = _risk_floors(area.get("module_floors"))
        files = {
            path: entry["summary"]
            for path, entry in report.get("files", {}).items()
            if any(
                path == prefix or (prefix.endswith("/") and path.startswith(prefix))
                for prefix in prefixes
            )
        }
        missing = set(required) - files.keys()
        if missing:
            raise ValueError(
                f"{name}: missing required module coverage: {sorted(missing)}"
            )
        for path, summary in files.items():
            measured = _risk_percentages(summary)
            failures.extend(
                f"{name}: {path}: {failure}"
                for failure in failed_thresholds(measured, module_floors)
            )
        totals = {
            key: sum(summary[key] for summary in files.values())
            for key in (
                "num_statements",
                "covered_lines",
                "num_branches",
                "covered_branches",
            )
        }
        results[name] = _risk_percentages(totals)
        failures.extend(
            f"{name}: {failure}" for failure in failed_thresholds(results[name], floors)
        )
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--combined", type=float, default=DEFAULT_COMBINED)
    parser.add_argument("--lines", type=float, default=DEFAULT_LINES)
    parser.add_argument("--branches", type=float, default=DEFAULT_BRANCHES)
    parser.add_argument("--risk-baseline", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    percentages = coverage_percentages(report)
    thresholds = {
        "combined": args.combined,
        "lines": args.lines,
        "branches": args.branches,
    }
    failures = failed_thresholds(percentages, thresholds)
    if args.risk_baseline:
        try:
            policy = json.loads(args.risk_baseline.read_text(encoding="utf-8"))
            areas, risk_failures = risk_area_results(report, policy)
        except (KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
        failures.extend(risk_failures)
        for name, measured in areas.items():
            print(
                name
                + ": "
                + ", ".join(
                    f"{metric}={value:.3f}%" for metric, value in measured.items()
                )
            )
    if failures:
        parser.error("; ".join(failures))

    print(
        "coverage ratchet passed: "
        + ", ".join(f"{name}={value:.3f}%" for name, value in percentages.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
