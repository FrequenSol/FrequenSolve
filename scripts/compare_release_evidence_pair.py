"""Compare two sealed FrequenSolve release-evidence asset pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def compare_release_evidence_pairs(
    expected_evidence_path: Path,
    expected_archive_path: Path,
    actual_evidence_path: Path,
    actual_archive_path: Path,
) -> None:
    """Raise ``ValueError`` unless both evidence asset pairs are identical."""
    expected_evidence = _load_evidence(expected_evidence_path)
    actual_evidence = _load_evidence(actual_evidence_path)
    expected_archive_sha = _sha256(expected_archive_path)
    actual_archive_sha = _sha256(actual_archive_path)

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
            "existing release evidence assets do not match the newly sealed pair: "
            + "; ".join(mismatches)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-evidence", required=True, type=Path)
    parser.add_argument("--expected-archive", required=True, type=Path)
    parser.add_argument("--actual-evidence", required=True, type=Path)
    parser.add_argument("--actual-archive", required=True, type=Path)
    args = parser.parse_args()

    try:
        compare_release_evidence_pairs(
            args.expected_evidence,
            args.expected_archive,
            args.actual_evidence,
            args.actual_archive,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print("existing release evidence assets match the newly sealed pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
