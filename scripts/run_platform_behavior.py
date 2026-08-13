#!/usr/bin/env python3
"""Run and retain a fail-closed installed-package platform contract."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

try:
    from scripts.check_coverage_thresholds import (
        DEFAULT_BRANCHES,
        DEFAULT_COMBINED,
        DEFAULT_LINES,
        coverage_percentages,
        failed_thresholds,
    )
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/.
    from check_coverage_thresholds import (
        DEFAULT_BRANCHES,
        DEFAULT_COMBINED,
        DEFAULT_LINES,
        coverage_percentages,
        failed_thresholds,
    )

SCHEMA = "frequensolve-platform-behavior/v1"
ROOT = Path(__file__).resolve().parents[1]
COVERAGE_CONFIG = ROOT / "tests/installed-package-coveragerc"
DETERMINISTIC_MARKER = (
    "not integration and not cloud and not hpc and not interactive and not visual"
)
CORE_SELECTORS = (
    "tests/test_base_package_contract.py",
    "tests/test_public_imports.py::test_top_level_import_smoke_in_clean_process",
    "tests/test_public_imports.py::test_bare_top_level_import_is_lazy_until_public_export_access",
    "tests/test_public_imports.py::test_top_level_import_does_not_load_optional_backends",
    "tests/test_public_imports.py::test_units_registry_is_lazy_but_usable",
    "tests/test_subprocess_environment.py",
    "tests/test_simulation_contract.py",
    "tests/test_geometry_properties.py",
    "tests/test_validation.py",
    "tests/test_trace_store_summary.py",
)
MINIMUM_TESTS = {"full": 1300, "core": 85}
MAXIMUM_SKIPS = {"full": 20, "core": 0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_platform(
    *,
    actual_system: str,
    actual_machine: str,
    expected_system: str,
    expected_machine: str,
) -> None:
    """Reject emulation, fallback runners, and an unexpected operating system."""

    mismatches = []
    if actual_system != expected_system:
        mismatches.append(
            f"operating system must be {expected_system!r}, got {actual_system!r}"
        )
    if actual_machine != expected_machine:
        mismatches.append(
            f"machine architecture must be {expected_machine!r}, got {actual_machine!r}"
        )
    if mismatches:
        raise ValueError("platform identity mismatch: " + "; ".join(mismatches))


def junit_counts(path: Path) -> dict[str, int]:
    """Return non-duplicated aggregate counts from pytest JUnit XML."""

    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("./testsuite"))
    if not suites:
        raise ValueError("JUnit evidence contains no test suites")
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] <= 0:
        raise ValueError("JUnit evidence contains no tests")
    return counts


def validate_coverage_files(
    report: Mapping[str, Any], package_root: Path
) -> dict[str, float]:
    """Require coverage to describe only the installed distribution."""

    files = report.get("files", {})
    if not files:
        raise ValueError("coverage evidence contains no files")
    unexpected = []
    resolved_root = package_root.resolve()
    for raw_path in files:
        path = Path(raw_path).resolve()
        if not path.is_relative_to(resolved_root):
            unexpected.append(str(path))
    if unexpected:
        raise ValueError(
            "coverage evidence includes non-installed source paths: "
            + ", ".join(sorted(unexpected)[:5])
        )
    return coverage_percentages(dict(report))


def validate_install_report(
    report: Mapping[str, Any], artifact: Path
) -> Mapping[str, Any]:
    """Bind pip's resolver report to the selected FrequenSolve artifact."""

    if str(report.get("version")) != "1":
        raise ValueError("pip install report must use schema version '1'")
    candidates = [
        row
        for row in report.get("install", [])
        if str(row.get("metadata", {}).get("name", "")).lower() == "frequensolve"
    ]
    if len(candidates) != 1:
        raise ValueError("pip install report must contain exactly one FrequenSolve row")
    download_url = str(candidates[0].get("download_info", {}).get("url", ""))
    installed_name = Path(unquote(urlparse(download_url).path)).name
    if installed_name != artifact.name:
        raise ValueError(
            f"pip installed {installed_name!r}, expected artifact {artifact.name!r}"
        )
    return candidates[0]


def build_pytest_command(
    *,
    suite: str,
    package_root: Path,
    coverage_output: Path,
    junit_output: Path,
) -> list[str]:
    """Build the deterministic pytest invocation for one platform suite."""

    if suite not in MINIMUM_TESTS:
        raise ValueError(f"unknown platform behavior suite: {suite!r}")
    selectors: Sequence[str] = () if suite == "full" else CORE_SELECTORS
    return [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--strict-markers",
        "-ra",
        "-q",
        "-m",
        DETERMINISTIC_MARKER,
        f"--junitxml={junit_output}",
        f"--cov={package_root}",
        f"--cov-config={COVERAGE_CONFIG}",
        f"--cov-report=json:{coverage_output}",
        "--cov-fail-under=0",
        *selectors,
    ]


def _installed_package_identity(commit: str) -> tuple[Path, Mapping[str, Any]]:
    import frequensolve
    from frequensolve import _version

    package_root = Path(frequensolve.__file__).resolve().parent
    if package_root.is_relative_to(ROOT.resolve()):
        raise ValueError(
            f"FrequenSolve imported from checkout instead of installed artifact: {package_root}"
        )
    version = _version.get_versions()
    if version.get("full-revisionid") != commit or version.get("dirty") is not False:
        raise ValueError(
            "installed package revision does not match requested commit: "
            f"{version.get('full-revisionid')!r}, dirty={version.get('dirty')!r}"
        )
    if importlib.metadata.version("frequensolve") != frequensolve.__version__:
        raise ValueError("installed package metadata and public version disagree")
    return package_root, version


def _write_environment_files(output_dir: Path) -> tuple[Path, Path, str]:
    import numpy as np

    from frequensolve.util.fft import get_fft_backend

    freeze_path = output_dir / "resolver-freeze.txt"
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    freeze_path.write_text(freeze.stdout, encoding="utf-8")

    numpy_path = output_dir / "numpy-config.txt"
    with numpy_path.open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream):
            np.show_config()
    return freeze_path, numpy_path, get_fft_backend().__name__


def execute(
    *,
    suite: str,
    artifact: Path,
    expected_system: str,
    expected_machine: str,
    commit: str,
    install_report_path: Path,
    output_dir: Path,
) -> int:
    """Execute one platform contract and write immutable evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    validate_platform(
        actual_system=platform.system(),
        actual_machine=platform.machine(),
        expected_system=expected_system,
        expected_machine=expected_machine,
    )
    package_root, version = _installed_package_identity(commit)
    install_report = json.loads(install_report_path.read_text(encoding="utf-8"))
    install_row = validate_install_report(install_report, artifact)
    freeze_path, numpy_path, fft_backend = _write_environment_files(output_dir)

    coverage_path = output_dir / "coverage.json"
    junit_path = output_dir / "junit.xml"
    command = build_pytest_command(
        suite=suite,
        package_root=package_root,
        coverage_output=coverage_path,
        junit_output=junit_path,
    )
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        return result.returncode

    counts = junit_counts(junit_path)
    if counts["tests"] < MINIMUM_TESTS[suite]:
        raise ValueError(
            f"{suite} suite ran {counts['tests']} tests; expected at least "
            f"{MINIMUM_TESTS[suite]}"
        )
    if counts["failures"] or counts["errors"]:
        raise ValueError(f"platform JUnit reports failures: {counts}")
    if counts["skipped"] > MAXIMUM_SKIPS[suite]:
        raise ValueError(
            f"{suite} suite skipped {counts['skipped']} tests; maximum is "
            f"{MAXIMUM_SKIPS[suite]}"
        )

    coverage_report = json.loads(coverage_path.read_text(encoding="utf-8"))
    percentages = validate_coverage_files(coverage_report, package_root)
    if suite == "full":
        thresholds = {
            "combined": DEFAULT_COMBINED,
            "lines": DEFAULT_LINES,
            "branches": DEFAULT_BRANCHES,
        }
        failures = failed_thresholds(percentages, thresholds)
        if failures:
            raise ValueError(
                "installed-package coverage ratchet failed: " + "; ".join(failures)
            )

    evidence = {
        "schemaVersion": SCHEMA,
        "commit": commit,
        "suite": suite,
        "selection": {
            "markerExpression": DETERMINISTIC_MARKER,
            "selectors": list(() if suite == "full" else CORE_SELECTORS),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "pythonImplementation": platform.python_implementation(),
        },
        "package": {
            "artifact": artifact.name,
            "artifactSha256": _sha256(artifact),
            "artifactBytes": artifact.stat().st_size,
            "version": importlib.metadata.version("frequensolve"),
            "revision": version["full-revisionid"],
            "importRoot": str(package_root),
            "pipRequested": bool(install_row.get("requested")),
        },
        "tests": {**counts, "junitSha256": _sha256(junit_path)},
        "coverage": {**percentages, "reportSha256": _sha256(coverage_path)},
        "runtime": {
            "numpyVersion": importlib.metadata.version("numpy"),
            "numpyConfigSha256": _sha256(numpy_path),
            "fftBackend": fft_backend,
            "resolverFreezeSha256": _sha256(freeze_path),
            "pipInstallReportSha256": _sha256(install_report_path),
        },
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{suite} installed-package platform contract passed: "
        f"{counts['tests']} tests, line={percentages['lines']:.3f}%, "
        f"branch={percentages['branches']:.3f}%, "
        f"combined={percentages['combined']:.3f}%"
    )
    return 0


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(MINIMUM_TESTS), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-system", required=True)
    parser.add_argument("--expected-machine", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--install-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return execute(
            suite=args.suite,
            artifact=args.artifact.resolve(),
            expected_system=args.expected_system,
            expected_machine=args.expected_machine,
            commit=args.commit,
            install_report_path=args.install_report.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(run())
