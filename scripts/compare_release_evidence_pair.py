"""Compare two sealed FrequenSolve release-evidence asset sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.validate_release_evidence import (
        SOLVER_BACKED_PROFILE,
        release_evidence_profile,
    )
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/.
    from validate_release_evidence import (
        SOLVER_BACKED_PROFILE,
        release_evidence_profile,
    )

ARCHIVE_SHA_FIELD = "dockerTestArchiveSha256"


def _load_evidence(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_release_evidence_assets(
    expected_evidence_path: Path,
    actual_evidence_path: Path,
    *,
    expected_archive_path: Path | None = None,
    actual_archive_path: Path | None = None,
) -> None:
    """Raise ``ValueError`` unless both profile-aware asset sets are identical."""

    expected_evidence = _load_evidence(expected_evidence_path)
    actual_evidence = _load_evidence(actual_evidence_path)
    mismatches: list[str] = []

    if expected_evidence != actual_evidence:
        differing_fields = sorted(
            key
            for key in expected_evidence.keys() | actual_evidence.keys()
            if expected_evidence.get(key) != actual_evidence.get(key)
        )
        mismatches.append(
            "release-evidence.json differs in fields: " + ", ".join(differing_fields)
        )

    expected_profile = release_evidence_profile(expected_evidence)
    actual_profile = release_evidence_profile(actual_evidence)
    if expected_profile != actual_profile:
        mismatches.append(
            f"validation profiles differ ({expected_profile!r} != {actual_profile!r})"
        )

    archive_paths = (expected_archive_path, actual_archive_path)
    if expected_profile != SOLVER_BACKED_PROFILE:
        if any(path is not None for path in archive_paths):
            mismatches.append(
                "standard release evidence must not include a heavy evidence archive"
            )
        if mismatches:
            raise ValueError(
                "existing release evidence assets do not match the newly sealed set: "
                + "; ".join(mismatches)
            )
        return

    if expected_archive_path is None or actual_archive_path is None:
        mismatches.append(
            "solver-backed release evidence requires both heavy evidence archives"
        )
        raise ValueError(
            "existing release evidence assets do not match the newly sealed set: "
            + "; ".join(mismatches)
        )

    expected_archive_sha = _sha256(expected_archive_path)
    actual_archive_sha = _sha256(actual_archive_path)
    for label, evidence, archive_sha in (
        ("newly sealed", expected_evidence, expected_archive_sha),
        ("existing release", actual_evidence, actual_archive_sha),
    ):
        declared_archive_sha = evidence.get(ARCHIVE_SHA_FIELD)
        if declared_archive_sha != archive_sha:
            mismatches.append(
                f"{label} release-evidence.json declares "
                f"{ARCHIVE_SHA_FIELD}={declared_archive_sha!r}, but its archive "
                f"SHA-256 is {archive_sha}"
            )

    if expected_archive_sha != actual_archive_sha:
        mismatches.append(
            "frequensolve-test-evidence.tar.gz differs "
            f"(newly sealed {expected_archive_sha}, existing {actual_archive_sha})"
        )

    if mismatches:
        raise ValueError(
            "existing release evidence assets do not match the newly sealed set: "
            + "; ".join(mismatches)
        )


def compare_release_evidence_pairs(
    expected_evidence_path: Path,
    expected_archive_path: Path,
    actual_evidence_path: Path,
    actual_archive_path: Path,
) -> None:
    """Retain the legacy four-path API for solver-backed callers."""

    compare_release_evidence_assets(
        expected_evidence_path,
        actual_evidence_path,
        expected_archive_path=expected_archive_path,
        actual_archive_path=actual_archive_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-evidence", required=True, type=Path)
    parser.add_argument("--expected-archive", type=Path)
    parser.add_argument("--actual-evidence", required=True, type=Path)
    parser.add_argument("--actual-archive", type=Path)
    args = parser.parse_args()

    try:
        if (args.expected_archive is None) != (args.actual_archive is None):
            raise ValueError(
                "--expected-archive and --actual-archive must be provided together"
            )
        compare_release_evidence_assets(
            args.expected_evidence,
            args.actual_evidence,
            expected_archive_path=args.expected_archive,
            actual_archive_path=args.actual_archive,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print("existing release evidence assets match the newly sealed set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
