import builtins
import copy
import json
import runpy
from pathlib import Path

import pytest

from scripts.check_optional_extra_contracts import (
    DEFAULT_MANIFEST,
    DEFAULT_PYPROJECT,
    _coverage_percent,
    _load_toml,
    contracts_from_manifest,
    load_manifest,
    lower_bound_requirements,
    matrix_rows,
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
                "schema": "frequensolve-optional-extra-contracts-1",
                "contracts": [
                    {
                        "name": "base",
                        "distribution": "sdist",
                        "imports": ["json"],
                        "selectors": ["tests/test_placeholder.py"],
                        "coverage_prefixes": ["frequensolve/example.py"],
                        "coverage_floor": 0,
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


def test_installed_package_contracts_do_not_require_hypothesis(monkeypatch):
    original_import = builtins.__import__

    def import_without_hypothesis(name, *args, **kwargs):
        if name == "hypothesis":
            raise ModuleNotFoundError("No module named 'hypothesis'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_hypothesis)

    runpy.run_path(Path(__file__).with_name("conftest.py"))
