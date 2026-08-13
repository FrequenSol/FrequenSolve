#!/usr/bin/env python3
"""Validate and run built-package behavior contracts for optional extras."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from packaging.requirements import Requirement

SCHEMA = "frequensolve-optional-extra-contracts-1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/optional-extra-contracts.json"
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"


@dataclass(frozen=True)
class Contract:
    """One independently installable package behavior contract."""

    name: str
    distribution: str
    imports: tuple[str, ...]
    selectors: tuple[str, ...]
    pytest_args: tuple[str, ...]
    coverage_prefixes: tuple[str, ...]
    coverage_floor: float
    environment: Mapping[str, str]


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI path.
        import toml

        return toml.loads(path.read_text(encoding="utf-8"))
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and validate the top-level manifest schema."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"Unsupported optional-extra schema: {payload.get('schema')!r}"
        )
    return payload


def contracts_from_manifest(payload: Mapping[str, Any]) -> tuple[Contract, ...]:
    """Normalize manifest contract rows and reject ambiguous entries."""

    contracts: list[Contract] = []
    seen: set[str] = set()
    for row in payload.get("contracts", []):
        name = str(row.get("name", "")).strip()
        if not name or name in seen:
            raise ValueError(
                f"Optional-extra contract name is empty or repeated: {name!r}"
            )
        seen.add(name)
        imports = tuple(str(value) for value in row.get("imports", []))
        selectors = tuple(str(value) for value in row.get("selectors", []))
        prefixes = tuple(str(value) for value in row.get("coverage_prefixes", []))
        if not imports or not selectors or not prefixes:
            raise ValueError(
                f"Contract {name!r} requires imports, selectors, and coverage prefixes"
            )
        distribution = str(row.get("distribution", ""))
        if distribution not in {"wheel", "sdist"}:
            raise ValueError(
                f"Contract {name!r} has invalid distribution {distribution!r}"
            )
        floor = float(row.get("coverage_floor", -1))
        if not 0 <= floor <= 100:
            raise ValueError(f"Contract {name!r} has invalid coverage floor {floor}")
        contracts.append(
            Contract(
                name=name,
                distribution=distribution,
                imports=imports,
                selectors=selectors,
                pytest_args=tuple(str(value) for value in row.get("pytest_args", [])),
                coverage_prefixes=prefixes,
                coverage_floor=floor,
                environment={
                    str(key): str(value)
                    for key, value in row.get("environment", {}).items()
                },
            )
        )
    if not contracts or contracts[0].name != "base":
        raise ValueError(
            "The optional-extra manifest must begin with the base contract"
        )
    return tuple(contracts)


def _project_extras(pyproject: Mapping[str, Any]) -> Mapping[str, Sequence[str]]:
    return pyproject["project"]["optional-dependencies"]


def validate_manifest(
    payload: Mapping[str, Any], pyproject: Mapping[str, Any]
) -> tuple[Contract, ...]:
    """Validate advertised extras, aliases, selectors, and marker policy."""

    contracts = contracts_from_manifest(payload)
    contract_names = {
        contract.name for contract in contracts if contract.name != "base"
    }
    excluded = {str(value) for value in payload.get("excluded_project_extras", [])}
    extras = _project_extras(pyproject)
    expected = set(extras) - excluded
    if contract_names != expected:
        missing = sorted(expected - contract_names)
        unexpected = sorted(contract_names - expected)
        raise ValueError(
            f"Optional-extra contracts differ from package metadata; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for alias, target in payload.get("aliases", {}).items():
        if alias not in extras or target not in extras:
            raise ValueError(f"Unknown optional-extra alias {alias!r} -> {target!r}")
        if set(extras[alias]) != set(extras[target]):
            raise ValueError(f"Optional-extra alias {alias!r} differs from {target!r}")

    for contract in contracts:
        for selector in contract.selectors:
            test_path = selector.split("::", 1)[0]
            if not test_path.startswith("tests/test_") or not test_path.endswith(".py"):
                raise ValueError(
                    f"Contract {contract.name!r} has unsafe selector {selector!r}"
                )
        if contract.name != "base":
            for raw_requirement in extras[contract.name]:
                _lower_bound_requirement(raw_requirement)

    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    configured_markers = {str(row).split(":", 1)[0] for row in markers}
    approval_markers = set(payload.get("approval_markers", {}))
    if approval_markers != configured_markers:
        raise ValueError(
            "Approval-marker policy differs from pytest configuration; "
            f"policy={sorted(approval_markers)}, configured={sorted(configured_markers)}"
        )
    return contracts


def _contract(contracts: Sequence[Contract], name: str) -> Contract:
    for contract in contracts:
        if contract.name == name:
            return contract
    raise ValueError(f"Unknown optional-extra contract: {name}")


def _lower_bound_requirement(raw: str) -> str:
    requirement = Requirement(raw)
    lower_bounds = [
        item.version for item in requirement.specifier if item.operator in {">=", "=="}
    ]
    upper_bounds = [
        item for item in requirement.specifier if item.operator in {"<", "<="}
    ]
    if not lower_bounds or not upper_bounds:
        raise ValueError(
            f"Runtime requirement must declare lower and upper bounds: {raw!r}"
        )
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    marker = f"; {requirement.marker}" if requirement.marker else ""
    return f"{requirement.name}{extras}=={lower_bounds[0]}{marker}"


def lower_bound_requirements(
    pyproject: Mapping[str, Any], contract: Contract
) -> tuple[str, ...]:
    """Return exact direct lower bounds for base plus one selected extra."""

    requirements = list(pyproject["project"]["dependencies"])
    if contract.name != "base":
        requirements.extend(_project_extras(pyproject)[contract.name])
    return tuple(_lower_bound_requirement(str(raw)) for raw in requirements)


def matrix_rows(contracts: Sequence[Contract]) -> list[dict[str, str]]:
    return [
        {"contract": contract.name, "distribution": contract.distribution}
        for contract in contracts
    ]


def _collected_cases(marker_expression: str) -> int:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--strict-markers",
            "--collect-only",
            "-q",
            "-m",
            marker_expression,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 5}:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"pytest collection failed for {marker_expression!r}: {detail}"
        )
    return sum(
        1
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )


def validate_approval_markers(policy: Mapping[str, Any]) -> dict[str, int]:
    """Validate marker membership without conflating ownership and approval."""

    counts: dict[str, int] = {}
    for marker, rules in policy.items():
        count = _collected_cases(marker)
        minimum = int(rules.get("minimum_cases", 0))
        if count < minimum:
            raise ValueError(
                f"Approval marker {marker!r} collected {count} cases; expected {minimum}"
            )
        companion = rules.get("required_companion")
        if companion and _collected_cases(f"{marker} and not {companion}"):
            raise ValueError(
                f"Approval marker {marker!r} has cases without {companion!r}"
            )
        counts[str(marker)] = count
    return counts


def _verify_imports(contract: Contract) -> None:
    for module_name in contract.imports:
        importlib.import_module(module_name)


def _coverage_percent(report: Mapping[str, Any], prefixes: Sequence[str]) -> float:
    statements = covered = 0
    for file_name, entry in report.get("files", {}).items():
        normalized = Path(file_name).as_posix()
        package_index = normalized.rfind("frequensolve/")
        if package_index >= 0:
            normalized = normalized[package_index:]
        if not any(
            normalized == prefix or normalized.startswith(prefix) for prefix in prefixes
        ):
            continue
        summary = entry["summary"]
        statements += int(summary["num_statements"])
        covered += int(summary["covered_lines"])
    if statements == 0:
        raise ValueError(f"Coverage report contains no files for prefixes {prefixes}")
    return 100.0 * covered / statements


def run_contract(contract: Contract, coverage_output: Path) -> int:
    """Collect and run one contract against the currently installed package."""

    _verify_imports(contract)
    environment = dict(os.environ)
    environment.update(contract.environment)
    common = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--strict-markers",
        *contract.pytest_args,
        *contract.selectors,
    ]
    collect = subprocess.run(
        [*common, "--collect-only", "-q"],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if collect.returncode != 0:
        return collect.returncode
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        [
            *common,
            "-q",
            "--cov=frequensolve",
            f"--cov-report=json:{coverage_output}",
            "--cov-fail-under=0",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if run.returncode != 0:
        return run.returncode
    report = json.loads(coverage_output.read_text(encoding="utf-8"))
    percent = _coverage_percent(report, contract.coverage_prefixes)
    print(
        f"{contract.name} optional-extra coverage: {percent:.2f}% "
        f"(floor {contract.coverage_floor:.2f}%)"
    )
    return 0 if percent >= contract.coverage_floor else 1


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true")
    action.add_argument("--matrix", action="store_true")
    action.add_argument("--lower-bound-requirements", metavar="CONTRACT")
    action.add_argument("--run", metavar="CONTRACT")
    parser.add_argument("--coverage-output", type=Path)
    args = parser.parse_args(argv)

    payload = load_manifest(args.manifest)
    pyproject = _load_toml(args.pyproject)
    contracts = validate_manifest(payload, pyproject)
    if args.validate:
        marker_counts = validate_approval_markers(payload["approval_markers"])
        print(f"Validated {len(contracts)} package behavior contracts")
        print(
            f"Approval marker collection: {json.dumps(marker_counts, sort_keys=True)}"
        )
        return 0
    if args.matrix:
        print(json.dumps({"include": matrix_rows(contracts)}, separators=(",", ":")))
        return 0
    if args.lower_bound_requirements:
        contract = _contract(contracts, args.lower_bound_requirements)
        print("\n".join(lower_bound_requirements(pyproject, contract)))
        return 0
    contract = _contract(contracts, args.run)
    if args.coverage_output is None:
        parser.error("--run requires --coverage-output")
    return run_contract(contract, args.coverage_output)


if __name__ == "__main__":
    raise SystemExit(run())
