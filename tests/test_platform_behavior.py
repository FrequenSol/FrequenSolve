import pytest

from scripts.run_platform_behavior import (
    CORE_SELECTORS,
    build_pytest_command,
    junit_counts,
    validate_coverage_files,
    validate_install_report,
    validate_platform,
)

pytestmark = pytest.mark.unit


def test_platform_identity_rejects_architecture_fallback():
    with pytest.raises(ValueError, match="machine architecture"):
        validate_platform(
            actual_system="Linux",
            actual_machine="x86_64",
            expected_system="Linux",
            expected_machine="aarch64",
        )


def test_junit_counts_aggregate_direct_suites_without_double_counting(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite tests="4" failures="0" errors="0" skipped="1" />'
        '<testsuite tests="3" failures="0" errors="0" skipped="0" />'
        "</testsuites>",
        encoding="utf-8",
    )

    assert junit_counts(junit) == {
        "tests": 7,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
    }


def test_install_report_must_bind_the_selected_artifact(tmp_path):
    artifact = tmp_path / "frequensolve-1.2.3-py3-none-any.whl"
    artifact.touch()
    report = {
        "version": "1",
        "install": [
            {
                "download_info": {"url": artifact.as_uri()},
                "metadata": {"name": "frequensolve", "version": "1.2.3"},
                "requested": True,
            }
        ],
    }

    assert validate_install_report(report, artifact)["requested"] is True

    wrong = tmp_path / "frequensolve-9.9.9-py3-none-any.whl"
    with pytest.raises(ValueError, match="expected artifact"):
        validate_install_report(report, wrong)


def test_installed_coverage_rejects_checkout_source_paths(tmp_path):
    package_root = tmp_path / "site-packages/frequensolve"
    package_root.mkdir(parents=True)
    installed = package_root / "units.py"
    installed.touch()
    report = {
        "files": {
            str(installed): {
                "summary": {
                    "num_statements": 10,
                    "covered_lines": 8,
                    "num_branches": 4,
                    "covered_branches": 3,
                }
            }
        },
        "totals": {
            "num_statements": 10,
            "covered_lines": 8,
            "num_branches": 4,
            "covered_branches": 3,
            "percent_covered": 78.57,
        },
    }

    percentages = validate_coverage_files(report, package_root)
    assert percentages == {"combined": 78.57, "lines": 80.0, "branches": 75.0}

    report["files"][str(tmp_path / "checkout/src/frequensolve/units.py")] = report[
        "files"
    ][str(installed)]
    with pytest.raises(ValueError, match="non-installed source"):
        validate_coverage_files(report, package_root)


def test_core_pytest_command_owns_every_required_behavior_area(tmp_path):
    command = build_pytest_command(
        suite="core",
        package_root=tmp_path / "site-packages/frequensolve",
        coverage_output=tmp_path / "coverage.json",
        junit_output=tmp_path / "junit.xml",
    )

    assert command[-len(CORE_SELECTORS) :] == list(CORE_SELECTORS)
    assert any("subprocess_environment" in selector for selector in CORE_SELECTORS)
    assert any("simulation_contract" in selector for selector in CORE_SELECTORS)
    assert any("geometry_properties" in selector for selector in CORE_SELECTORS)
    assert any("validation" in selector for selector in CORE_SELECTORS)
    assert any("trace_" in selector for selector in CORE_SELECTORS)
    assert all(isinstance(argument, str) for argument in command)
