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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--combined", type=float, default=DEFAULT_COMBINED)
    parser.add_argument("--lines", type=float, default=DEFAULT_LINES)
    parser.add_argument("--branches", type=float, default=DEFAULT_BRANCHES)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    percentages = coverage_percentages(report)
    thresholds = {
        "combined": args.combined,
        "lines": args.lines,
        "branches": args.branches,
    }
    failures = failed_thresholds(percentages, thresholds)
    if failures:
        parser.error("; ".join(failures))

    print(
        "coverage ratchet passed: "
        + ", ".join(f"{name}={value:.3f}%" for name, value in percentages.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
