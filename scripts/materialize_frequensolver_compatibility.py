"""Materialize packaged FrequenSolver compatibility metadata from release evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.validate_release_evidence import (
        SOLVER_BACKED_PROFILE,
        release_evidence_profile,
        validate_evidence,
    )
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/.
    from validate_release_evidence import (
        SOLVER_BACKED_PROFILE,
        release_evidence_profile,
        validate_evidence,
    )


SCHEMA = "frequensolve-frequensolver-compatibility/v2"
PACKAGE_RELEASE_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)(?:rc[1-9][0-9]*)?$"
)


def manifest_from_evidence(
    evidence: dict[str, Any],
    *,
    package_release: str,
    package_commit: str,
) -> dict[str, Any]:
    """Return deterministic package metadata after validating sealed evidence."""

    if not PACKAGE_RELEASE_RE.fullmatch(package_release):
        raise ValueError("package_release must be canonical X.Y.Z or X.Y.ZrcN")
    validate_evidence(evidence, package_commit)
    profile = release_evidence_profile(evidence)
    solver_backed = profile == SOLVER_BACKED_PROFILE
    evidence_run_id = (
        evidence["dockerEvidenceRunId"] if solver_backed else evidence["ciRunId"]
    )
    evidence_url = (
        evidence["dockerEvidenceRunUrl"] if solver_backed else evidence["ciRunUrl"]
    )
    return {
        "schema": SCHEMA,
        "package_release": package_release,
        "preferred_frequensolver": {
            "release": evidence["frequensolverRelease"],
            "git_commit": evidence["sauceCommit"],
            "release_url": evidence["frequensolverReleaseUrl"],
        },
        "validation": {
            "profile": profile,
            "solver_backed": solver_backed,
            "run_id": evidence_run_id,
            "url": evidence_url,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-evidence", required=True, type=Path)
    parser.add_argument("--package-release", required=True)
    parser.add_argument("--package-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        evidence = json.loads(args.release_evidence.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            raise ValueError("release evidence must be an object")
        manifest = manifest_from_evidence(
            evidence,
            package_release=args.package_release,
            package_commit=args.package_commit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "materialized preferred FrequenSolver "
        f"{manifest['preferred_frequensolver']['release']} for "
        f"FrequenSolve {args.package_release}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
