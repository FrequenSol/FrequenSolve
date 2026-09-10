from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.cloud import _worker
from benchmarks.cloud._shared import (
    CASE_SCHEMA,
    RUN_SCHEMA,
    percentile,
    sanitize,
    stable_fingerprint,
)
from benchmarks.cloud.runner import (
    _classify_known_bug_probe,
    _validate_case_script,
    _validated_worker_result,
    compare_runs,
    list_cases,
    run_benchmarks,
)
from scripts.generate_cloud_benchmark_workloads import _behavior_sha256, generate


def test_corpus_is_standalone_and_has_issue_linked_known_bugs():
    cases = list_cases()

    assert len(cases) == 24
    assert sum(case["expectedSubmissions"] for case in cases) == 45
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["script"].endswith(".py")
        assert not case["script"].endswith(".ipynb")
        assert len(case["sourceSha256"]) == 64
        assert len(case["scriptSha256"]) == 64
        assert len(case["behaviorSha256"]) == 64
        script = Path("benchmarks/cloud/workloads") / case["script"]
        assert hashlib.sha256(script.read_bytes()).hexdigest() == case["scriptSha256"]
        for bug in case["knownBugs"]:
            assert bug["issueUrl"].startswith("https://github.com/FrequenSol/")
            assert bug["reason"]
            assert bug["backends"]


def test_case_script_validation_rejects_stale_manifest(tmp_path):
    (tmp_path / "case.py").write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match manifest"):
        _validate_case_script(
            {
                "id": "case",
                "script": "case.py",
                "scriptSha256": hashlib.sha256(b"value = 1\n").hexdigest(),
            },
            tmp_path,
        )


def test_generator_emits_executable_python_without_notebook_presentation(tmp_path):
    source = tmp_path / "tutorials"
    source.mkdir()
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "from IPython.display import display\n",
                    "import matplotlib.pyplot as plt\n",
                    "import frequensolve as fs\n",
                    "site = fs.Site(profile='local')\n",
                    "result = site.submit(job).wait()\n",
                    "display(result)\n",
                    "result.status\n",
                ],
            }
        ]
    }
    (source / "example.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    manifest = generate(source, tmp_path / "generated")
    generated = (tmp_path / "generated/example.py").read_text(encoding="utf-8")

    assert manifest["caseCount"] == 1
    assert manifest["submissionCount"] == 1
    assert "site.submit(job).wait()" in generated
    assert "IPython" not in generated
    assert "matplotlib" not in generated
    assert "display(" not in generated
    compile(generated, "example.py", "exec")


def test_behavior_hash_ignores_formatting_and_generated_source_header():
    compact = '''"""Generated Cloud benchmark workload.\nSource SHA-256: old\n"""\nvalues=[x for x in range(3)]\n'''
    reformatted = '''"""Generated Cloud benchmark workload.\nSource SHA-256: new\n"""\nvalues = [x for x in range(3)]\n'''

    assert _behavior_sha256(compact) == _behavior_sha256(reformatted)


def test_sanitizer_redacts_credentials_but_keeps_performance_authorization():
    value = sanitize(
        {
            "accessToken": "private",
            "message": "Bearer abc.def.ghi used AKIA1234567890ABCDEF",
            "creditAuthorizationWaitSeconds": 1.25,
        }
    )

    assert value == {
        "accessToken": "<redacted>",
        "message": "Bearer <redacted> used <redacted-aws-access-key>",
        "creditAuthorizationWaitSeconds": 1.25,
    }


@pytest.mark.parametrize(
    ("values", "quantile", "expected"),
    [([], 0.5, None), ([4], 0.95, 4), ([1, 2, 3], 0.5, 2), ([0, 10], 0.95, 9.5)],
)
def test_percentile_uses_linear_interpolation(values, quantile, expected):
    assert percentile(values, quantile) == expected


def test_iso_seconds_accepts_variable_fraction_precision():
    assert _worker._iso_seconds("2026-09-10T00:00:00Z", "2026-09-10T00:00:00.5Z") == 0.5


class _FakeResult:
    status = SimpleNamespace(state="completed", raw={"status": "SUCCEEDED"})

    def raise_for_status(self):
        return None


class _FakeHandle:
    id = "simulation-1"

    def __init__(self):
        self.backend = {"executionBackend": "SLURM"}

    def wait(self):
        return _FakeResult()


class _FakeGraphQL:
    def get_simulation_status_details(self, simulation_id):
        return {
            "id": simulation_id,
            "status": "SUCCEEDED",
            "executionBackend": "SLURM",
        }

    def execute(self, query, variables):
        return {
            "getSimulation": {
                "id": variables["id"],
                "status": "SUCCEEDED",
                "startTime": "2026-09-10T00:00:00Z",
                "planningStartedAt": "2026-09-10T00:00:01Z",
                "planningCompletedAt": "2026-09-10T00:00:03Z",
                "terminalAt": "2026-09-10T00:00:10Z",
                "frequencyJobs": {
                    "items": [
                        {
                            "endTime": "2026-09-10T00:00:08Z",
                            "queueWaitSeconds": 1.5,
                            "activeWorkSeconds": 4.0,
                        }
                    ],
                    "nextToken": None,
                },
                "batchAttempts": {"items": [], "nextToken": None},
                "executionAttempts": {
                    "items": [
                        {
                            "submittedAt": "2026-09-10T00:00:00Z",
                            "gatewayReceivedAt": "2026-09-10T00:00:00.5Z",
                            "queuedAt": "2026-09-10T00:00:03Z",
                            "solverStartedAt": "2026-09-10T00:00:04Z",
                        }
                    ],
                    "nextToken": None,
                },
            }
        }


class _FakeSite:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.graphql_client = _FakeGraphQL()

    def submit(self, job, **kwargs):
        assert kwargs["force"] is True
        return _FakeHandle()


class _FakeProject:
    def __init__(self, *, path, **kwargs):
        self.path = path
        self.__post_init__()

    def __post_init__(self):
        self.initialized = True


def test_recorder_captures_top_level_and_provider_metrics(monkeypatch):
    fake_module = SimpleNamespace(Site=_FakeSite, Project=_FakeProject)
    monkeypatch.setitem(sys.modules, "frequensolve", fake_module)
    recorder = _worker.Recorder("sandbox-slurm", None, "slurm", "run-1", "case-1")
    recorder.install()
    try:
        site = fake_module.Site(profile="local")
        project = fake_module.Project(path="scratch")
        result = site.submit(SimpleNamespace(name="job")).wait()
    finally:
        recorder.restore()

    assert result.status.state == "completed"
    assert Path(project.path).name.startswith("cloud-benchmark-")
    assert recorder.submissions[0]["status"] == "PASSED"
    assert recorder.submissions[0]["userObservedSeconds"] >= 0
    assert recorder.submissions[0]["providerMetrics"] == {
        "planningSeconds": 2.0,
        "serverTotalSeconds": 10.0,
        "packingAndProjectionSeconds": 2.0,
        "frequencyQueueSeconds": [1.5],
        "frequencyActiveWorkSeconds": [4.0],
        "gatewaySeconds": [0.5],
        "schedulerQueueSeconds": [1.0],
    }


def _write_run(root: Path, run_id: str, seconds: float, finished: str) -> Path:
    run = root / run_id
    run.mkdir()
    summary = {
        "schema": RUN_SCHEMA,
        "runId": run_id,
        "profile": "sandbox-batch",
        "backend": "batch",
        "corpusFingerprint": "same",
        "selectionFingerprint": "same-selection",
        "finishedAt": finished,
        "successful": True,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    case = {
        "schema": CASE_SCHEMA,
        "caseId": "acoustic",
        "status": "PASS",
        "totalSeconds": seconds,
    }
    (run / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    return run


def test_compare_auto_selects_compatible_history_and_flags_regression(tmp_path):
    baseline = _write_run(tmp_path, "baseline", 10.0, "2026-09-10T00:00:00Z")
    candidate = _write_run(tmp_path, "candidate", 16.0, "2026-09-10T01:00:00Z")

    comparison = compare_runs(candidate=candidate, history_root=tmp_path)

    assert comparison["baselineRunId"] == baseline.name
    assert comparison["candidateRunId"] == candidate.name
    assert comparison["regressions"][0]["deltaPercent"] == 60.0
    assert (candidate / "comparison.json").exists()
    assert (candidate / "comparison.md").exists()


def test_compare_rejects_different_backend(tmp_path):
    baseline = _write_run(tmp_path, "baseline", 10.0, "2026-09-10T00:00:00Z")
    candidate = _write_run(tmp_path, "candidate", 11.0, "2026-09-10T01:00:00Z")
    payload = json.loads((candidate / "summary.json").read_text())
    payload["backend"] = "slurm"
    (candidate / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Incompatible benchmark runs"):
        compare_runs(candidate=candidate, baseline=baseline)


def test_compare_rejects_different_profile(tmp_path):
    baseline = _write_run(tmp_path, "baseline", 10.0, "2026-09-10T00:00:00Z")
    candidate = _write_run(tmp_path, "candidate", 11.0, "2026-09-10T01:00:00Z")
    payload = json.loads((candidate / "summary.json").read_text())
    payload["profile"] = "hosted-dev-batch"
    (candidate / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Incompatible benchmark runs"):
        compare_runs(candidate=candidate, baseline=baseline)


def test_malformed_worker_result_becomes_explicit_failure():
    result = _validated_worker_result(
        {},
        case={"id": "acoustic", "expectedSubmissions": 1},
        profile="sandbox-batch",
        backend="batch",
    )

    assert result["caseId"] == "acoustic"
    assert result["status"] == "FAIL"
    assert result["failure"]["type"] == "MalformedWorkerResult"
    assert "submissions is not a list" in result["failure"]["message"]


def test_known_bug_only_run_is_successful_and_excluded_from_pass_rate(tmp_path):
    run_root, summary = run_benchmarks(
        profile="synthetic-slurm",
        backend="slurm",
        history_root=tmp_path,
        run_id="known-bug",
        case_patterns=["03_velocity_model_building/02_coordinate_systems"],
    )

    assert summary["successful"] is True
    assert summary["passRatePercent"] == 100.0
    assert summary["counts"] == {
        "selected": 1,
        "completed": 1,
        "passed": 0,
        "failed": 0,
        "skippedKnownBug": 1,
    }
    result = json.loads((run_root / "cases.jsonl").read_text())
    assert result["status"] == "SKIP_KNOWN_BUG"
    assert result["knownBug"]["expectedFailureCodes"] == ["SLURM_PLANNING_FAILED"]


def test_known_bug_probe_only_xfails_for_the_registered_failure_code():
    bug = {"expectedFailureCodes": ["EXPECTED"], "reason": "known"}
    matching = {
        "status": "FAIL",
        "submissions": [{"diagnostics": {"status": {"failureCode": "EXPECTED"}}}],
    }
    unrelated = {
        "status": "FAIL",
        "submissions": [{"diagnostics": {"status": {"failureCode": "UNRELATED"}}}],
    }
    passing = {"status": "PASS", "submissions": []}

    _classify_known_bug_probe(matching, bug)
    _classify_known_bug_probe(unrelated, bug)
    _classify_known_bug_probe(passing, bug)

    assert matching["status"] == "XFAIL"
    assert unrelated["status"] == "FAIL"
    assert unrelated["knownBugMismatch"]["actualFailureCodes"] == ["UNRELATED"]
    assert passing["status"] == "XPASS"


def test_corpus_fingerprint_changes_with_behavioral_identity():
    original = {"cases": [{"id": "a", "behaviorSha256": "1"}]}
    changed = {"cases": [{"id": "a", "behaviorSha256": "2"}]}
    assert stable_fingerprint(original) != stable_fingerprint(changed)
