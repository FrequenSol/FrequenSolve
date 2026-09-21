from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import DEFAULT_HISTORY, compare_runs, list_cases, run_benchmarks


def _tags(values: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values:
        key, separator, tag_value = value.partition("=")
        if not separator or not key or not tag_value:
            raise ValueError(f"Tag must be KEY=VALUE: {value!r}")
        tags[key] = tag_value
    return tags


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.cloud",
        description="Run provider-neutral tutorial-derived Cloud benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="List benchmark cases")
    list_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--profile", required=True)
    run_parser.add_argument("--backend", choices=("batch", "slurm"), required=True)
    run_parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--case", action="append", default=[])
    run_parser.add_argument("--email")
    run_parser.add_argument("--timeout-seconds", type=int, default=10_800)
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--probe-known-bugs", action="store_true")
    run_parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Record comparable environment metadata as KEY=VALUE",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare with history")
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--baseline", type=Path)
    compare_parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    compare_parser.add_argument("--regression-percent", type=float, default=20.0)
    compare_parser.add_argument("--regression-min-seconds", type=float, default=5.0)
    compare_parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        cases = list_cases()
        if args.json:
            print(json.dumps(cases, indent=2, sort_keys=True))
        else:
            for case in cases:
                bugs = ", ".join(
                    backend for bug in case["knownBugs"] for backend in bug["backends"]
                )
                suffix = f" [known bug: {bugs}]" if bugs else ""
                print(
                    f"{case['id']} ({case['expectedSubmissions']} submissions){suffix}"
                )
        return 0
    if args.command == "run":
        run_root, summary = run_benchmarks(
            profile=args.profile,
            backend=args.backend,
            history_root=args.history_root,
            run_id=args.run_id,
            case_patterns=args.case,
            email=args.email,
            timeout_seconds=args.timeout_seconds,
            fail_fast=args.fail_fast,
            probe_known_bugs=args.probe_known_bugs,
            tags=_tags(args.tag),
        )
        print(json.dumps({"runRoot": str(run_root), **summary}, indent=2))
        return 0 if summary["successful"] else 1
    comparison = compare_runs(
        candidate=args.candidate,
        baseline=args.baseline,
        history_root=args.history_root,
        regression_percent=args.regression_percent,
        regression_min_seconds=args.regression_min_seconds,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 1 if args.fail_on_regression and comparison["regressions"] else 0
