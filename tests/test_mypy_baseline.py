import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts.check_mypy_baseline import (
    Diagnostic,
    _counter_from_rows,
    _diagnostic_rows,
    _load_baseline,
    _write_baseline,
    parse_diagnostics,
    run,
    strict_diagnostics,
)

pytestmark = pytest.mark.unit


def _mypy_line(file, code, message="diagnostic", severity="error"):
    return json.dumps(
        {
            "file": str(file),
            "line": 1,
            "column": 0,
            "message": message,
            "code": code,
            "severity": severity,
        }
    )


def test_mypy_json_diagnostics_group_by_module_and_error_class(tmp_path):
    source = tmp_path / "src/frequensolve/model.py"
    output = "\n".join(
        (
            _mypy_line(source, "arg-type", "first"),
            _mypy_line(source, "arg-type", "second"),
            _mypy_line(source, "arg-type", severity="note"),
        )
    )

    diagnostics = parse_diagnostics(output, tmp_path)

    assert diagnostics == Counter(
        {Diagnostic("src/frequensolve/model.py", "arg-type"): 2}
    )


def test_mypy_parser_tolerates_known_unused_config_status_line(tmp_path):
    output = "\n".join(
        (
            "pyproject.toml: note: unused section(s) in mypy config file: module = 'x'",
            _mypy_line(tmp_path / "src/frequensolve/model.py", "arg-type"),
        )
    )

    assert parse_diagnostics(output, tmp_path) == Counter(
        {Diagnostic("src/frequensolve/model.py", "arg-type"): 1}
    )


def test_mypy_parser_rejects_unknown_plain_text(tmp_path):
    with pytest.raises(ValueError, match="Unexpected non-JSON mypy output"):
        parse_diagnostics("mypy crashed unexpectedly", tmp_path)


def test_baseline_rows_roundtrip_deterministically():
    diagnostics = Counter(
        {
            Diagnostic("src/frequensolve/z.py", "return-value"): 1,
            Diagnostic("src/frequensolve/a.py", "no-untyped-def"): 3,
        }
    )

    rows = _diagnostic_rows(diagnostics)

    assert rows[0]["file"] == "src/frequensolve/a.py"
    assert _counter_from_rows(rows) == diagnostics


def test_strict_path_filter_covers_promoted_phase_one_and_two_modules():
    diagnostics = Counter(
        {
            Diagnostic("src/frequensolve/units.py", "arg-type"): 1,
            Diagnostic("src/frequensolve/validation/outputs.py", "assignment"): 2,
            Diagnostic(
                "src/frequensolve/orchestrator/utils/progress.py", "union-attr"
            ): 3,
            Diagnostic("src/frequensolve/storage.py", "return-value"): 1,
            Diagnostic("src/frequensolve/model/model.py", "arg-type"): 4,
        }
    )

    assert strict_diagnostics(diagnostics) == Counter(
        {
            Diagnostic("src/frequensolve/units.py", "arg-type"): 1,
            Diagnostic("src/frequensolve/validation/outputs.py", "assignment"): 2,
            Diagnostic(
                "src/frequensolve/orchestrator/utils/progress.py", "union-attr"
            ): 3,
            Diagnostic("src/frequensolve/storage.py", "return-value"): 1,
        }
    )


def test_baseline_writer_removes_promoted_diagnostic_headroom(tmp_path):
    baseline_path = tmp_path / "mypy-baseline.json"
    diagnostics = Counter(
        {
            Diagnostic(
                "src/frequensolve/orchestrator/utils/credentials.py",
                "no-untyped-def",
            ): 4,
            Diagnostic("src/frequensolve/model/model.py", "arg-type"): 2,
        }
    )

    _write_baseline(baseline_path, diagnostics)
    _, written = _load_baseline(baseline_path)

    assert written == Counter(
        {Diagnostic("src/frequensolve/model/model.py", "arg-type"): 2}
    )


def test_lazy_utility_exports_retain_consumer_types(tmp_path):
    root = Path(__file__).resolve().parents[1]
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "\n".join(
            (
                "from frequensolve.orchestrator.utils import (",
                "    Credentials,",
                "    PoolInfo,",
                "    status_text,",
                ")",
                "credentials = Credentials(username='typed-user')",
                "pool = PoolInfo()",
                "summary: str = status_text([], {})",
            )
        )
        + "\n"
    )
    environment = dict(os.environ)
    environment["MYPYPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(root / "pyproject.toml"),
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            str(consumer),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_baseline_allows_environment_with_fewer_diagnostics(monkeypatch, tmp_path):
    expected = Counter(
        {
            Diagnostic("src/frequensolve/model.py", "arg-type"): 2,
            Diagnostic("src/frequensolve/plot.py", "assignment"): 1,
        }
    )
    current = Counter({Diagnostic("src/frequensolve/model.py", "arg-type"): 1})
    monkeypatch.setattr(
        "scripts.check_mypy_baseline.Path.resolve",
        lambda _self: tmp_path / "scripts/check_mypy_baseline.py",
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline._run_mypy",
        lambda _root: current,
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline._load_baseline",
        lambda _path, **_kwargs: ({}, expected),
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline.Path.exists",
        lambda _self: True,
    )

    assert run([]) == 0


def test_baseline_rejects_environment_that_exceeds_a_ceiling(monkeypatch, tmp_path):
    expected = Counter({Diagnostic("src/frequensolve/model.py", "arg-type"): 1})
    current = Counter({Diagnostic("src/frequensolve/model.py", "arg-type"): 2})
    monkeypatch.setattr(
        "scripts.check_mypy_baseline.Path.resolve",
        lambda _self: tmp_path / "scripts/check_mypy_baseline.py",
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline._run_mypy",
        lambda _root: current,
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline._load_baseline",
        lambda _path, **_kwargs: ({}, expected),
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline.Path.exists",
        lambda _self: True,
    )

    assert run([]) == 1


def test_update_preserves_higher_ceiling_from_another_environment(
    monkeypatch, tmp_path
):
    expected = Counter(
        {
            Diagnostic("src/frequensolve/model.py", "arg-type"): 2,
            Diagnostic("src/frequensolve/plot.py", "assignment"): 1,
        }
    )
    current = Counter(
        {
            Diagnostic("src/frequensolve/model.py", "arg-type"): 1,
            Diagnostic("src/frequensolve/cloud.py", "attr-defined"): 1,
        }
    )
    written = []
    monkeypatch.setattr(
        "scripts.check_mypy_baseline.Path.resolve",
        lambda _self: tmp_path / "scripts/check_mypy_baseline.py",
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline._run_mypy",
        lambda _root: current,
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline._load_baseline",
        lambda _path, **_kwargs: ({}, expected),
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline.Path.exists",
        lambda _self: True,
    )
    monkeypatch.setattr(
        "scripts.check_mypy_baseline._write_baseline",
        lambda _path, diagnostics: written.append(diagnostics),
    )

    assert run(["--update"]) == 0
    assert written == [expected | current]
