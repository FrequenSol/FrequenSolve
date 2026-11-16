from copy import deepcopy

import pytest

from scripts.check_coverage_thresholds import risk_area_results


def case():
    policy = {
        "schema": "frequensolve-risk-coverage-1",
        "areas": [
            {
                "name": "critical",
                "prefixes": ["src/frequensolve/critical/"],
                "required_modules": ["src/frequensolve/critical/a.py"],
                "floors": {"lines": 70, "branches": 60},
                "module_floors": {"lines": 20, "branches": 10},
            }
        ],
    }
    report = {
        "meta": {"branch_coverage": True},
        "files": {
            "src/frequensolve/critical/a.py": {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 8,
                    "num_branches": 4,
                    "covered_branches": 3,
                }
            },
        },
    }
    return report, policy


def test_risk_area_uses_executable_counts_and_exact_package_prefixes():
    report, policy = case()
    report["files"]["src/frequensolve/critical_other/b.py"] = {
        "summary": {
            "num_statements": 1000,
            "covered_lines": 0,
            "num_branches": 1000,
            "covered_branches": 0,
        }
    }
    results, failures = risk_area_results(report, policy)
    assert results["critical"] == {
        "lines": 80,
        "branches": 75,
        "combined": 100 * 11 / 14,
    }
    assert failures == []


def test_new_zero_covered_module_cannot_hide_behind_healthy_aggregate():
    report, policy = case()
    report["files"]["src/frequensolve/critical/new.py"] = {
        "summary": {
            "num_statements": 1,
            "covered_lines": 0,
            "num_branches": 1,
            "covered_branches": 0,
        }
    }
    results, failures = risk_area_results(report, policy)
    assert results["critical"]["lines"] > 70
    assert results["critical"]["branches"] == 60
    assert len(failures) == 2
    assert all("new.py" in failure for failure in failures)


def test_risk_area_rejects_a_branch_regression_even_with_healthy_lines():
    report, policy = case()
    report["files"]["src/frequensolve/critical/a.py"]["summary"]["covered_branches"] = 1
    _, failures = risk_area_results(report, policy)
    assert failures == ["critical: branches coverage 25.000% is below 60.000%"]


def test_branchless_module_does_not_invent_uncovered_branches():
    report, policy = case()
    report["files"]["src/frequensolve/critical/a.py"]["summary"].update(
        num_branches=0, covered_branches=0
    )
    results, failures = risk_area_results(report, policy)
    assert results["critical"]["branches"] == 100
    assert results["critical"]["combined"] == 80
    assert failures == []


def test_risk_area_cannot_silently_omit_a_required_module():
    report, policy = case()
    report["files"].clear()
    with pytest.raises(ValueError, match="missing required module"):
        risk_area_results(report, policy)


@pytest.mark.parametrize("meta", [{}, {"branch_coverage": False}])
def test_line_only_report_cannot_satisfy_risk_gate(meta):
    report, policy = case()
    report["meta"] = meta
    with pytest.raises(ValueError, match="branch-enabled"):
        risk_area_results(report, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("covered_lines", 11),
        ("covered_branches", 5),
        ("num_statements", 0),
        ("covered_lines", -1),
        ("covered_lines", 1.5),
        ("covered_lines", True),
    ],
)
def test_invalid_coverage_counts_fail_closed(field, value):
    report, policy = case()
    report["files"]["src/frequensolve/critical/a.py"]["summary"][field] = value
    with pytest.raises(ValueError, match="counts"):
        risk_area_results(report, policy)


@pytest.mark.parametrize("floor", [0, -1, 101, float("nan"), float("inf"), True])
def test_invalid_or_disabled_risk_floor_is_rejected(floor):
    report, policy = case()
    policy["areas"][0]["floors"]["lines"] = floor
    with pytest.raises(ValueError, match="finite percentages"):
        risk_area_results(report, policy)


def test_repeated_area_cannot_replace_an_earlier_result():
    report, policy = case()
    policy["areas"].append(deepcopy(policy["areas"][0]))
    with pytest.raises(ValueError, match="unique"):
        risk_area_results(report, policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("prefixes", []),
        ("required_modules", []),
        ("prefixes", ["../src/frequensolve/"]),
    ],
)
def test_risk_policy_requires_bounded_explicit_module_identity(key, value):
    report, policy = case()
    policy["areas"][0][key] = value
    with pytest.raises(ValueError, match="explicit package"):
        risk_area_results(report, policy)
