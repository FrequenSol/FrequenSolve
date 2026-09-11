from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ._shared import (
    CASE_SCHEMA,
    COMPARISON_SCHEMA,
    RUN_SCHEMA,
    append_jsonl,
    read_json,
    safe_name,
    sanitize,
    sanitize_text,
    stable_fingerprint,
    summarize_metric,
    utc_now,
    write_json,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKLOAD_ROOT = PACKAGE_ROOT / "workloads"
DEFAULT_HISTORY = Path(".benchmarks/cloud-history")
WORKER_STATUSES = {"PASS", "FAIL"}


def _validate_case_script(case: Mapping[str, Any], workload_root: Path) -> None:
    script_value = case.get("script")
    if not isinstance(script_value, str):
        raise RuntimeError(f"Benchmark case {case.get('id')!r} has no script")
    root = workload_root.resolve()
    script = (root / script_value).resolve()
    if not script.is_relative_to(root) or not script.is_file():
        raise RuntimeError(
            f"Benchmark case {case.get('id')!r} has an unsafe or missing script"
        )
    expected = case.get("scriptSha256")
    actual = hashlib.sha256(script.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Benchmark case {case.get('id')!r} does not match manifest.json; "
            "regenerate the tutorial-derived corpus"
        )


def _load_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    manifest = read_json(WORKLOAD_ROOT / "manifest.json")
    known_bugs = read_json(PACKAGE_ROOT / "known_bugs.json")
    cases = manifest.get("cases")
    bugs = known_bugs.get("knownBugs")
    if manifest.get("schema") != "frequensolve-cloud-benchmark-corpus/v1":
        raise RuntimeError("Unsupported cloud benchmark corpus schema")
    if not isinstance(cases, list) or not isinstance(bugs, list):
        raise RuntimeError("Malformed cloud benchmark corpus")
    for case in cases:
        if not isinstance(case, Mapping):
            raise RuntimeError("Malformed cloud benchmark case")
        _validate_case_script(case, WORKLOAD_ROOT)
    fingerprint_input = {
        "cases": [
            {
                "id": case["id"],
                "behaviorSha256": case["behaviorSha256"],
                "expectedSubmissions": case["expectedSubmissions"],
            }
            for case in cases
        ],
        "knownBugs": bugs,
    }
    return cases, bugs, stable_fingerprint(fingerprint_input)


def list_cases() -> list[dict[str, Any]]:
    cases, bugs, _ = _load_corpus()
    for case in cases:
        case["knownBugs"] = [bug for bug in bugs if bug["caseId"] == case["id"]]
    return cases


def _selected(
    cases: Iterable[dict[str, Any]], patterns: Iterable[str]
) -> list[dict[str, Any]]:
    filters = list(patterns)
    if not filters:
        return list(cases)
    selected = [
        case
        for case in cases
        if any(fnmatch.fnmatch(case["id"], pattern) for pattern in filters)
    ]
    if not selected:
        raise ValueError(f"No benchmark cases matched {filters!r}")
    return selected


def _known_bug(
    bugs: Iterable[dict[str, Any]], case_id: str, backend: str
) -> dict[str, Any] | None:
    return next(
        (
            bug
            for bug in bugs
            if bug["caseId"] == case_id and backend in bug["backends"]
        ),
        None,
    )


def _failure_codes(result: Mapping[str, Any]) -> set[str]:
    codes: set[str] = set()
    for submission in result.get("submissions", []):
        if not isinstance(submission, Mapping):
            continue
        diagnostics = submission.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        for source in (diagnostics.get("status"), diagnostics.get("provider")):
            if isinstance(source, Mapping) and isinstance(
                source.get("failureCode"), str
            ):
                codes.add(source["failureCode"])
    return codes


def _classify_known_bug_probe(result: dict[str, Any], bug: Mapping[str, Any]) -> None:
    result["knownBug"] = dict(bug)
    if result.get("status") == "PASS":
        result["status"] = "XPASS"
        return
    expected_codes = set(bug.get("expectedFailureCodes", []))
    actual_codes = _failure_codes(result)
    expected_failure = bug.get("expectedFailure")
    actual_failure = result.get("failure")
    structured_match = False
    if isinstance(expected_failure, Mapping) and isinstance(actual_failure, Mapping):
        expected_message = expected_failure.get("messageContains")
        expected_fields = [
            field for field in ("phase", "type") if field in expected_failure
        ]
        message_fragments = (
            [expected_message]
            if isinstance(expected_message, str)
            else expected_message if isinstance(expected_message, list) else []
        )
        structured_match = (
            bool(expected_fields or message_fragments)
            and all(
                actual_failure.get(field) == expected_failure[field]
                for field in expected_fields
            )
            and all(
                isinstance(fragment, str)
                and fragment in str(actual_failure.get("message", ""))
                for fragment in message_fragments
            )
        )
    if (
        expected_codes and expected_codes.intersection(actual_codes)
    ) or structured_match:
        result["status"] = "XFAIL"
        return
    result["status"] = "FAIL"
    result["knownBugMismatch"] = {
        "expectedFailureCodes": sorted(expected_codes),
        "actualFailureCodes": sorted(actual_codes),
        "expectedFailure": expected_failure,
        "actualFailure": actual_failure,
    }


def _identity() -> dict[str, Any]:
    import frequensolve

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "frequensolveVersion": getattr(frequensolve, "__version__", "unknown"),
        "gitRevision": revision,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


def _worker_failure(
    *,
    case: Mapping[str, Any],
    profile: str,
    backend: str,
    failure_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "schema": CASE_SCHEMA,
        "caseId": case["id"],
        "profile": profile,
        "declaredBackend": backend,
        "status": "FAIL",
        "failure": {
            "phase": "worker",
            "type": failure_type,
            "message": message,
        },
        "expectedSubmissions": case["expectedSubmissions"],
        "submissions": [],
    }


def _validated_worker_result(
    value: Any,
    *,
    case: Mapping[str, Any],
    profile: str,
    backend: str,
) -> dict[str, Any]:
    expected = {
        "schema": CASE_SCHEMA,
        "caseId": case["id"],
        "profile": profile,
        "declaredBackend": backend,
        "expectedSubmissions": case["expectedSubmissions"],
    }
    problems = []
    if not isinstance(value, Mapping):
        problems.append(f"result is {type(value).__name__}, not an object")
    else:
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                problems.append(
                    f"{key} is {value.get(key)!r}, expected {expected_value!r}"
                )
        if value.get("status") not in WORKER_STATUSES:
            problems.append(f"unsupported status {value.get('status')!r}")
        if not isinstance(value.get("submissions"), list):
            problems.append("submissions is not a list")
    if problems:
        return _worker_failure(
            case=case,
            profile=profile,
            backend=backend,
            failure_type="MalformedWorkerResult",
            message="; ".join(problems),
        )
    return dict(value)


def _performance(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    cases = list(cases)
    submissions = [
        submission
        for case in cases
        for submission in case.get("submissions", [])
        if isinstance(submission, Mapping)
    ]

    def numbers(key: str) -> list[float]:
        return [
            float(item[key])
            for item in submissions
            if isinstance(item.get(key), (int, float))
        ]

    nested: dict[str, list[float]] = {}
    for submission in submissions:
        provider_metrics = submission.get("providerMetrics")
        if not isinstance(provider_metrics, Mapping):
            continue
        for key, value in provider_metrics.items():
            values = value if isinstance(value, list) else [value]
            for sample in values:
                if isinstance(sample, (int, float)):
                    nested.setdefault(key, []).append(float(sample))
    return {
        "caseTotalSeconds": summarize_metric(
            float(case["totalSeconds"])
            for case in cases
            if case.get("status") == "PASS"
            and isinstance(case.get("totalSeconds"), (int, float))
        ),
        "submissionAcceptSeconds": summarize_metric(numbers("acceptSeconds")),
        "submissionWaitSeconds": summarize_metric(numbers("waitSeconds")),
        "submissionUserObservedSeconds": summarize_metric(
            numbers("userObservedSeconds")
        ),
        "provider": {
            key: summarize_metric(value) for key, value in sorted(nested.items())
        },
    }


def _markdown_summary(
    summary: Mapping[str, Any], cases: Iterable[Mapping[str, Any]]
) -> str:
    counts = summary["counts"]
    lines = [
        f"# Cloud benchmark {summary['runId']}",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Backend: `{summary['backend']}`",
        f"- Result: **{'PASS' if summary['successful'] else 'FAIL'}**",
        f"- Pass rate: **{summary['passRatePercent']:.2f}%** "
        f"({counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['skippedKnownBug']} known-bug skipped)",
        "",
        "| Case | Status | Seconds | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for case in cases:
        detail = ""
        if case.get("knownBug"):
            bug = case["knownBug"]
            detail = f"[{bug['issueUrl']}]({bug['issueUrl']}): {bug['reason']}"
        elif case.get("failure"):
            failure = case["failure"]
            detail = f"{failure.get('type', 'Error')}: {failure.get('message', '')}"
        seconds = case.get("totalSeconds")
        rendered_seconds = f"{seconds:.3f}" if isinstance(seconds, (int, float)) else ""
        lines.append(
            f"| `{case['caseId']}` | {case['status']} | {rendered_seconds} | {detail} |"
        )
    lines.extend(
        [
            "",
            "Pass rate excludes issue-linked `SKIP_KNOWN_BUG` cases. A run is successful only when every non-skipped case passes.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmarks(
    *,
    profile: str,
    backend: str,
    history_root: Path = DEFAULT_HISTORY,
    run_id: str | None = None,
    case_patterns: Iterable[str] = (),
    email: str | None = None,
    timeout_seconds: int = 10_800,
    fail_fast: bool = False,
    probe_known_bugs: bool = False,
    tags: Mapping[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if backend not in {"batch", "slurm"}:
        raise ValueError("backend must be batch or slurm")
    cases, bugs, corpus_fingerprint = _load_corpus()
    cases = _selected(cases, case_patterns)
    run_id = safe_name(
        run_id
        or f"{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}-{safe_name(profile)}"
    )
    run_root = history_root.resolve() / run_id
    if run_root.exists():
        raise FileExistsError(f"Benchmark run already exists: {run_root}")
    run_root.mkdir(parents=True)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    manifest = {
        "schema": RUN_SCHEMA,
        "runId": run_id,
        "profile": profile,
        "backend": backend,
        "corpusFingerprint": corpus_fingerprint,
        "selectedCases": [case["id"] for case in cases],
        "selectionFingerprint": stable_fingerprint([case["id"] for case in cases]),
        "startedAt": started_at,
        "identity": _identity(),
        "probeKnownBugs": probe_known_bugs,
        "tags": sanitize(dict(sorted((tags or {}).items()))),
    }
    write_json(run_root / "manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for case in cases:
        bug = _known_bug(bugs, case["id"], backend)
        if bug is not None and not probe_known_bugs:
            result = {
                "schema": "frequensolve-cloud-benchmark-case/v1",
                "caseId": case["id"],
                "profile": profile,
                "declaredBackend": backend,
                "status": "SKIP_KNOWN_BUG",
                "knownBug": bug,
                "expectedSubmissions": case["expectedSubmissions"],
                "submissions": [],
            }
        else:
            case_root = run_root / "cases" / safe_name(case["id"])
            case_root.mkdir(parents=True)
            command = [
                sys.executable,
                "-m",
                "benchmarks.cloud._worker",
                "--case-id",
                case["id"],
                "--run-id",
                run_id,
                "--workload",
                str(WORKLOAD_ROOT / case["script"]),
                "--result",
                str(case_root / "result.json"),
                "--workdir",
                str(case_root / "work"),
                "--profile",
                profile,
                "--backend",
                backend,
                "--expected-submissions",
                str(case["expectedSubmissions"]),
            ]
            if email:
                command.extend(["--email", email])
            environment = os.environ.copy()
            repository_root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    [
                        repository_root,
                        str(Path(repository_root) / "src"),
                        environment.get("PYTHONPATH"),
                    ],
                )
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=repository_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                (case_root / "stdout.log").write_text(
                    sanitize_text(completed.stdout), encoding="utf-8"
                )
                (case_root / "stderr.log").write_text(
                    sanitize_text(completed.stderr), encoding="utf-8"
                )
                result_path = case_root / "result.json"
                if result_path.is_file():
                    result = _validated_worker_result(
                        read_json(result_path),
                        case=case,
                        profile=profile,
                        backend=backend,
                    )
                else:
                    result = _worker_failure(
                        case=case,
                        profile=profile,
                        backend=backend,
                        failure_type="MissingWorkerResult",
                        message=(
                            "Benchmark worker exited without writing result.json "
                            f"(exit {completed.returncode})"
                        ),
                    )
                if completed.returncode != 0 and result.get("status") == "PASS":
                    result["status"] = "FAIL"
                    result["failure"] = {
                        "phase": "worker",
                        "type": "WorkerExitError",
                        "message": f"Worker exited with {completed.returncode}",
                    }
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout if error.stdout is not None else error.output
                stderr = error.stderr
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                (case_root / "stdout.log").write_text(
                    sanitize_text(stdout or ""), encoding="utf-8"
                )
                (case_root / "stderr.log").write_text(
                    sanitize_text(stderr or ""), encoding="utf-8"
                )
                result = _worker_failure(
                    case=case,
                    profile=profile,
                    backend=backend,
                    failure_type="TimeoutExpired",
                    message=f"Case exceeded {timeout_seconds} seconds",
                )
                result["failure"]["phase"] = "timeout"
            except (OSError, json.JSONDecodeError) as error:
                result = _worker_failure(
                    case=case,
                    profile=profile,
                    backend=backend,
                    failure_type=type(error).__name__,
                    message=str(error),
                )
            if bug is not None:
                _classify_known_bug_probe(result, bug)
        result = sanitize(result)
        results.append(result)
        append_jsonl(run_root / "cases.jsonl", result)
        print(
            json.dumps(
                {
                    "caseId": result["caseId"],
                    "status": result["status"],
                    "seconds": result.get("totalSeconds"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if fail_fast and result["status"] in {"FAIL", "XPASS"}:
            break
    passed = sum(result["status"] == "PASS" for result in results)
    failed = sum(result["status"] in {"FAIL", "XPASS"} for result in results)
    skipped = sum(result["status"] in {"SKIP_KNOWN_BUG", "XFAIL"} for result in results)
    denominator = passed + failed
    summary = {
        **manifest,
        "finishedAt": utc_now(),
        "totalWallSeconds": time.monotonic() - started_monotonic,
        "successful": failed == 0,
        "passRatePercent": 100.0 * passed / denominator if denominator else 100.0,
        "counts": {
            "selected": len(cases),
            "completed": len(results),
            "passed": passed,
            "failed": failed,
            "skippedKnownBug": skipped,
        },
        "performance": _performance(results),
    }
    write_json(run_root / "summary.json", summary)
    (run_root / "summary.md").write_text(
        _markdown_summary(summary, results), encoding="utf-8"
    )
    return run_root, summary


def _summary_path(path: Path) -> Path:
    return path if path.name == "summary.json" else path / "summary.json"


def _auto_baseline(
    candidate: Mapping[str, Any], history_root: Path, candidate_path: Path
) -> Path:
    candidates = []
    for path in history_root.glob("*/summary.json"):
        if path.resolve() == candidate_path.resolve():
            continue
        try:
            summary = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            summary.get("schema") == RUN_SCHEMA
            and summary.get("backend") == candidate.get("backend")
            and summary.get("profile") == candidate.get("profile")
            and summary.get("corpusFingerprint") == candidate.get("corpusFingerprint")
            and summary.get("selectionFingerprint")
            == candidate.get("selectionFingerprint")
            and summary.get("successful") is True
            and str(summary.get("finishedAt", ""))
            < str(candidate.get("finishedAt", ""))
        ):
            candidates.append((str(summary.get("finishedAt")), path))
    if not candidates:
        raise FileNotFoundError(
            "No earlier successful baseline with the same profile, backend, and corpus fingerprint"
        )
    return max(candidates)[1]


def _case_rows(run_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in (run_root / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["caseId"]] = row
    return rows


def compare_runs(
    *,
    candidate: Path,
    baseline: Path | None = None,
    history_root: Path = DEFAULT_HISTORY,
    regression_percent: float = 20.0,
    regression_min_seconds: float = 5.0,
) -> dict[str, Any]:
    candidate_summary_path = _summary_path(candidate).resolve()
    candidate_summary = read_json(candidate_summary_path)
    if candidate_summary.get("schema") != RUN_SCHEMA:
        raise RuntimeError("Candidate is not a cloud benchmark run")
    baseline_summary_path = (
        _summary_path(baseline).resolve()
        if baseline is not None
        else _auto_baseline(
            candidate_summary, history_root.resolve(), candidate_summary_path
        ).resolve()
    )
    baseline_summary = read_json(baseline_summary_path)
    compatibility = {
        "profile": baseline_summary.get("profile") == candidate_summary.get("profile"),
        "backend": baseline_summary.get("backend") == candidate_summary.get("backend"),
        "corpusFingerprint": baseline_summary.get("corpusFingerprint")
        == candidate_summary.get("corpusFingerprint"),
        "selectionFingerprint": baseline_summary.get("selectionFingerprint")
        == candidate_summary.get("selectionFingerprint"),
    }
    if not all(compatibility.values()):
        raise ValueError(f"Incompatible benchmark runs: {compatibility}")
    candidate_rows = _case_rows(candidate_summary_path.parent)
    baseline_rows = _case_rows(baseline_summary_path.parent)
    comparisons = []
    for case_id in sorted(candidate_rows.keys() & baseline_rows.keys()):
        current = candidate_rows[case_id]
        previous = baseline_rows[case_id]
        if current.get("status") != "PASS" or previous.get("status") != "PASS":
            continue
        current_seconds = current.get("totalSeconds")
        previous_seconds = previous.get("totalSeconds")
        if not isinstance(current_seconds, (int, float)) or not isinstance(
            previous_seconds, (int, float)
        ):
            continue
        delta = current_seconds - previous_seconds
        percent = 100.0 * delta / previous_seconds if previous_seconds else None
        regression = (
            delta >= regression_min_seconds
            and percent is not None
            and percent >= regression_percent
        )
        comparisons.append(
            {
                "caseId": case_id,
                "baselineSeconds": previous_seconds,
                "candidateSeconds": current_seconds,
                "deltaSeconds": delta,
                "deltaPercent": percent,
                "regression": regression,
            }
        )
    result = {
        "schema": COMPARISON_SCHEMA,
        "createdAt": utc_now(),
        "backend": candidate_summary["backend"],
        "corpusFingerprint": candidate_summary["corpusFingerprint"],
        "candidateRunId": candidate_summary["runId"],
        "baselineRunId": baseline_summary["runId"],
        "thresholds": {
            "regressionPercent": regression_percent,
            "regressionMinSeconds": regression_min_seconds,
        },
        "caseComparisons": comparisons,
        "regressions": [row for row in comparisons if row["regression"]],
    }
    write_json(candidate_summary_path.parent / "comparison.json", result)
    lines = [
        f"# Benchmark comparison: {result['candidateRunId']}",
        "",
        f"Baseline: `{result['baselineRunId']}`  ",
        f"Backend: `{result['backend']}`",
        "",
        "| Case | Baseline (s) | Candidate (s) | Delta | Regression |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparisons:
        percent = row["deltaPercent"]
        delta = "n/a" if percent is None else f"{percent:+.1f}%"
        lines.append(
            f"| `{row['caseId']}` | {row['baselineSeconds']:.3f} | "
            f"{row['candidateSeconds']:.3f} | {delta} | "
            f"{'yes' if row['regression'] else 'no'} |"
        )
    lines.append("")
    (candidate_summary_path.parent / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result
