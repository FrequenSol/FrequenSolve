import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_optional_extra_contracts import (
    DEFAULT_MANIFEST,
    DEFAULT_PYPROJECT,
    _branch_coverage_percent,
    _coverage_percent,
    _load_toml,
    contracts_from_manifest,
    load_manifest,
    lower_bound_requirements,
    matrix_rows,
    module_coverage_failures,
    run,
    validate_manifest,
)

pytestmark = pytest.mark.unit


def _repo_contracts():
    payload = load_manifest(DEFAULT_MANIFEST)
    pyproject = _load_toml(DEFAULT_PYPROJECT)
    return payload, pyproject, validate_manifest(payload, pyproject)


def test_manifest_covers_every_advertised_runtime_extra():
    payload, pyproject, contracts = _repo_contracts()

    excluded = set(payload["excluded_project_extras"])
    project_extras = set(pyproject["project"]["optional-dependencies"])

    assert {contract.name for contract in contracts[1:]} == project_extras - excluded
    assert all(contract.imports for contract in contracts)
    assert all(contract.selectors for contract in contracts)


def test_parallel_alias_drift_fails_manifest_validation():
    payload = load_manifest(DEFAULT_MANIFEST)
    pyproject = copy.deepcopy(_load_toml(DEFAULT_PYPROJECT))
    pyproject["project"]["optional-dependencies"]["parallel"].append("unexpected>=1,<2")

    with pytest.raises(ValueError, match="parallel.*differs from.*hpc"):
        validate_manifest(payload, pyproject)


def test_manifest_contract_requires_real_behavior_selection():
    payload = load_manifest(DEFAULT_MANIFEST)
    payload["contracts"][0]["selectors"] = []

    with pytest.raises(ValueError, match="requires imports, selectors"):
        contracts_from_manifest(payload)


def test_matrix_is_derived_from_manifest_without_duplicate_workflow_list():
    _, _, contracts = _repo_contracts()

    rows = matrix_rows(contracts)

    assert rows[0] == {"contract": "base", "distribution": "sdist"}
    assert {row["contract"] for row in rows} == {
        contract.name for contract in contracts
    }


def test_lower_bound_requirements_pin_base_and_selected_extra():
    _, pyproject, contracts = _repo_contracts()
    fast_fft = next(contract for contract in contracts if contract.name == "fast-fft")

    requirements = lower_bound_requirements(pyproject, fast_fft)

    assert "numpy==1.24" in requirements
    assert "pyfftw==0.14" in requirements
    assert all("==" in requirement for requirement in requirements)


def test_runtime_dependency_without_upper_bound_is_rejected():
    payload = load_manifest(DEFAULT_MANIFEST)
    pyproject = copy.deepcopy(_load_toml(DEFAULT_PYPROJECT))
    pyproject["project"]["optional-dependencies"]["fast-fft"] = ["pyfftw>=0.14"]

    with pytest.raises(ValueError, match="lower and upper bounds"):
        validate_manifest(payload, pyproject)


def test_coverage_floor_aggregates_only_owned_package_prefixes(tmp_path):
    report = {
        "files": {
            "/tmp/site-packages/frequensolve/plotting/a.py": {
                "summary": {"num_statements": 10, "covered_lines": 6}
            },
            "src/frequensolve/plotting/b.py": {
                "summary": {"num_statements": 5, "covered_lines": 3}
            },
            "frequensolve/cloud.py": {
                "summary": {"num_statements": 100, "covered_lines": 0}
            },
        }
    }

    assert _coverage_percent(report, ("frequensolve/plotting/",)) == 60.0


def test_manifest_is_stable_json():
    payload = load_manifest(DEFAULT_MANIFEST)

    assert json.loads(json.dumps(payload))["schema"] == payload["schema"]
    assert Path(DEFAULT_MANIFEST).name == "optional-extra-contracts.json"


def test_import_verification_does_not_require_package_metadata(tmp_path):
    manifest = tmp_path / "contracts.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "frequensolve-optional-extra-contracts-2",
                "contracts": [
                    {
                        "name": "base",
                        "distribution": "sdist",
                        "imports": ["json"],
                        "selectors": ["tests/test_placeholder.py"],
                        "coverage_prefixes": ["frequensolve/example.py"],
                        "coverage_floor": 0,
                        "coverage_branch_floor": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        run(
            [
                "--manifest",
                str(manifest),
                "--pyproject",
                str(tmp_path / "missing.toml"),
                "--verify-imports",
                "base",
            ]
        )
        == 0
    )


def test_installed_package_contracts_do_not_require_hypothesis():
    # Use a fresh process: an already-loaded Hypothesis pytest plugin wraps
    # fixture registration and would invalidate an in-process missing-extra test.
    script = """
import builtins, runpy, sys
original_import = builtins.__import__
def import_without_hypothesis(name, *args, **kwargs):
    if name == "hypothesis":
        raise ModuleNotFoundError("No module named 'hypothesis'", name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_hypothesis
runpy.run_path(sys.argv[1])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).with_name("conftest.py"))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("floor", [None, 0, -1, float("nan"), 101, True])
def test_optional_contract_rejects_missing_or_disabled_branch_floor(floor):
    payload = load_manifest(DEFAULT_MANIFEST)
    payload["contracts"][0]["coverage_branch_floor"] = floor
    with pytest.raises(ValueError, match="positive branch"):
        contracts_from_manifest(payload)


def test_optional_branch_floor_uses_only_selected_package_and_rejects_line_only_report():
    report = {
        "meta": {"branch_coverage": True},
        "files": {
            "/tmp/site-packages/frequensolve/example/a.py": {
                "summary": {"num_branches": 10, "covered_branches": 4}
            },
            "src/frequensolve/example/b.py": {
                "summary": {"num_branches": 10, "covered_branches": 8}
            },
            "src/frequensolve/example_other.py": {
                "summary": {"num_branches": 100, "covered_branches": 0}
            },
        },
    }
    assert _branch_coverage_percent(report, ("frequensolve/example/",)) == 60
    report["meta"]["branch_coverage"] = False
    with pytest.raises(ValueError, match="branch-enabled"):
        _branch_coverage_percent(report, ("frequensolve/example/",))


def test_optional_branch_floor_fails_on_missing_or_corrupt_evidence():
    report = {"meta": {"branch_coverage": True}, "files": {}}
    with pytest.raises(ValueError, match="no files"):
        _branch_coverage_percent(report, ("frequensolve/example.py",))
    report["files"]["src/frequensolve/example.py"] = {
        "summary": {"num_branches": 2, "covered_branches": 3}
    }
    with pytest.raises(ValueError, match="invalid branch"):
        _branch_coverage_percent(report, ("frequensolve/example.py",))


@pytest.mark.parametrize("minimum", [0, -1, 101, True, float("nan")])
def test_optional_module_floors_cannot_be_disabled(minimum):
    payload = load_manifest(DEFAULT_MANIFEST)
    row = payload["contracts"][0]
    row["module_floors"] = {
        row["coverage_prefixes"][0]: {"lines": minimum, "branches": 5}
    }
    with pytest.raises(ValueError, match="invalid module floors"):
        contracts_from_manifest(payload)


def test_optional_module_floor_cannot_select_an_unowned_module():
    payload = load_manifest(DEFAULT_MANIFEST)
    payload["contracts"][0]["module_floors"] = {
        "frequensolve/other.py": {"lines": 10, "branches": 5}
    }
    with pytest.raises(ValueError, match="invalid module floors"):
        contracts_from_manifest(payload)


def test_optional_module_gate_rejects_zero_coverage_despite_healthy_aggregate():
    report = {
        "meta": {"branch_coverage": True},
        "files": {
            "site-packages/frequensolve/plotting/a.py": {
                "summary": {
                    "num_statements": 1000,
                    "covered_lines": 1000,
                    "num_branches": 100,
                    "covered_branches": 100,
                }
            },
            "site-packages/frequensolve/plotting/b.py": {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 0,
                    "num_branches": 2,
                    "covered_branches": 0,
                }
            },
        },
    }
    floors = {"frequensolve/plotting/b.py": {"lines": 20, "branches": 10}}
    assert _coverage_percent(report, ("frequensolve/plotting/",)) > 99
    assert len(module_coverage_failures(report, floors)) == 1
    report["files"]["site-packages/frequensolve/plotting/b.py"]["summary"].update(
        covered_lines=3, covered_branches=1
    )
    assert module_coverage_failures(report, floors) == []
    del report["files"]["site-packages/frequensolve/plotting/b.py"]
    with pytest.raises(ValueError, match="no files"):
        module_coverage_failures(report, floors)
