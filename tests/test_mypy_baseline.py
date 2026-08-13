import json
from collections import Counter

import pytest

from scripts.check_mypy_baseline import (
    Diagnostic,
    _counter_from_rows,
    _diagnostic_rows,
    parse_diagnostics,
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


def test_strict_path_filter_covers_units_and_validation_only():
    diagnostics = Counter(
        {
            Diagnostic("src/frequensolve/units.py", "arg-type"): 1,
            Diagnostic("src/frequensolve/validation/outputs.py", "assignment"): 2,
            Diagnostic("src/frequensolve/model/model.py", "arg-type"): 4,
        }
    )

    assert strict_diagnostics(diagnostics) == Counter(
        {
            Diagnostic("src/frequensolve/units.py", "arg-type"): 1,
            Diagnostic("src/frequensolve/validation/outputs.py", "assignment"): 2,
        }
    )
