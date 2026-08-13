#!/usr/bin/env python3
"""Run whole-package mypy and enforce the reviewed incremental baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Sequence

SCHEMA = "frequensolve-mypy-baseline-1"
BASELINE_FILE = "mypy-baseline.json"

STRICT_PATHS = (
    "scripts/check_mypy_baseline.py",
    "src/frequensolve/units.py",
    "src/frequensolve/validation/",
)

PHASES = (
    {
        "order": 1,
        "name": "public-authoring-validation-units-serialization",
        "strict_now": list(STRICT_PATHS),
        "remaining_prefixes": [
            "src/frequensolve/geometry/",
            "src/frequensolve/mesh/",
            "src/frequensolve/model/",
            "src/frequensolve/project/",
            "src/frequensolve/seismic/",
            "src/frequensolve/simulation/",
        ],
    },
    {
        "order": 2,
        "name": "core-orchestration-and-result-loading",
        "remaining_prefixes": [
            "src/frequensolve/orchestrator/sites/base.py",
            "src/frequensolve/orchestrator/sites/local/",
            "src/frequensolve/orchestrator/utils/",
            "src/frequensolve/storage.py",
        ],
    },
    {
        "order": 3,
        "name": "cloud-hpc-visual-and-optional-backends",
        "remaining_prefixes": [
            "src/frequensolve/orchestrator/sites/aws/",
            "src/frequensolve/orchestrator/sites/hpc/",
            "src/frequensolve/plotting/",
            "src/frequensolve/mcp_server/",
            "src/frequensolve/commands/",
        ],
    },
)


class Diagnostic(NamedTuple):
    file: str
    code: str


def _relative_file(value: str, root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return path.as_posix()


def parse_diagnostics(output: str, root: Path) -> Counter[Diagnostic]:
    """Parse mypy's newline-delimited JSON error output."""

    diagnostics: Counter[Diagnostic] = Counter()
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if ": note: unused section" in line:
                continue
            raise ValueError(f"Unexpected non-JSON mypy output: {line}") from exc
        if payload.get("severity") != "error":
            continue
        diagnostics[
            Diagnostic(
                file=_relative_file(str(payload["file"]), root),
                code=str(payload["code"]),
            )
        ] += 1
    return diagnostics


def _diagnostic_rows(
    diagnostics: Counter[Diagnostic],
) -> list[dict[str, object]]:
    return [
        {
            "file": diagnostic.file,
            "code": diagnostic.code,
            "count": count,
        }
        for diagnostic, count in sorted(diagnostics.items())
    ]


def _counter_from_rows(rows: Iterable[dict[str, object]]) -> Counter[Diagnostic]:
    diagnostics: Counter[Diagnostic] = Counter()
    for row in rows:
        count = row["count"]
        if not isinstance(count, int):
            raise TypeError("Mypy baseline diagnostic counts must be integers")
        diagnostic = Diagnostic(
            file=str(row["file"]),
            code=str(row["code"]),
        )
        diagnostics[diagnostic] += count
    return diagnostics


def strict_diagnostics(
    diagnostics: Counter[Diagnostic],
) -> Counter[Diagnostic]:
    """Return errors from modules already promoted to zero-error typing."""

    return Counter(
        {
            diagnostic: count
            for diagnostic, count in diagnostics.items()
            if any(
                diagnostic.file == path or diagnostic.file.startswith(path)
                for path in STRICT_PATHS
            )
        }
    )


def _run_mypy(root: Path) -> Counter[Diagnostic]:
    environment = dict(os.environ)
    environment["MYPYPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), environment.get("MYPYPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "-O", "json"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"mypy failed to produce diagnostics: {detail}")
    return parse_diagnostics(result.stdout, root)


def _load_baseline(path: Path) -> tuple[dict, Counter[Diagnostic]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported mypy baseline schema: {payload.get('schema')!r}")
    if payload.get("phases") != list(PHASES):
        raise ValueError("Mypy phase order differs from the executable plan")
    return payload, _counter_from_rows(payload.get("diagnostics", []))


def _write_baseline(path: Path, diagnostics: Counter[Diagnostic]) -> None:
    payload = {
        "schema": SCHEMA,
        "scope": "src/frequensolve",
        "strict_paths": list(STRICT_PATHS),
        "phases": list(PHASES),
        "diagnostics": _diagnostic_rows(diagnostics),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _format_delta(label: str, delta: Counter[Diagnostic]) -> list[str]:
    lines: list[str] = []
    for diagnostic, count in sorted(delta.items()):
        lines.append(f"{label} {count}x {diagnostic.file} [{diagnostic.code}]")
    return lines


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument(
        "--update",
        action="store_true",
        help="widen ceilings after reviewing diagnostics from another environment",
    )
    update_group.add_argument(
        "--replace",
        action="store_true",
        help="replace ceilings after reviewing reductions across all environments",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    baseline_path = root / BASELINE_FILE
    current = _run_mypy(root)
    promoted_errors = strict_diagnostics(current)
    if promoted_errors:
        print("Strict mypy modules regressed:", file=sys.stderr)
        print("\n".join(_format_delta("added", promoted_errors)), file=sys.stderr)
        return 1

    if args.update or args.replace:
        updated = current
        if args.update and baseline_path.exists():
            _, expected = _load_baseline(baseline_path)
            updated = expected | current
        _write_baseline(baseline_path, updated)
        print(
            f"Updated {baseline_path} with {sum(updated.values())} reviewed "
            "whole-package diagnostic ceilings"
        )
        return 0

    if not baseline_path.exists():
        print(
            f"Missing {baseline_path}; review diagnostics and run with --update",
            file=sys.stderr,
        )
        return 1

    _, expected = _load_baseline(baseline_path)
    added = current - expected
    removed = expected - current
    if added:
        lines = ["Whole-package mypy diagnostics exceed the reviewed baseline."]
        lines.extend(_format_delta("added", added))
        lines.append(
            "Fix regressions, or review the environment-specific diagnostics "
            "and widen ceilings with `python scripts/check_mypy_baseline.py "
            "--update`."
        )
        print("\n".join(lines), file=sys.stderr)
        return 1

    if removed:
        print(
            "Whole-package mypy baseline has improvement headroom; "
            "review it before lowering cross-environment ceilings:\n"
            + "\n".join(_format_delta("below", removed))
        )
    print(
        f"Whole-package mypy baseline passed: {sum(current.values())} "
        f"diagnostics; {len(STRICT_PATHS)} promoted paths remain error-free"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
