"""Validate checksum-bound FrequenSolver identity release evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "frequensolver-identity-1"
PRODUCT = "FrequenSolver"
EXPECTED_KEYS = {"schema", "product", "version", "build_id", "git_commit"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_nonempty_printable_single_line_ascii(value: Any) -> bool:
    """Return whether ``value`` satisfies the FrequenSolver identity contract."""

    return (
        isinstance(value, str)
        and bool(value)
        and all(" " <= character <= "~" for character in value)
    )


def validate_identity(
    identity: dict[str, Any],
    *,
    expected_version: str,
    expected_commit: str,
    expected_build_id: str | None = None,
) -> None:
    """Raise ``ValueError`` unless identity proves the expected release build."""

    errors: list[str] = []
    missing = sorted(EXPECTED_KEYS - set(identity))
    extra = sorted(set(identity) - EXPECTED_KEYS)
    if missing:
        errors.append(f"missing identity keys: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected identity keys: {', '.join(extra)}")
    expected = {
        "schema": SCHEMA,
        "product": PRODUCT,
        "version": expected_version,
        "git_commit": expected_commit,
    }
    if expected_build_id is not None:
        expected["build_id"] = expected_build_id
    for name, value in expected.items():
        if identity.get(name) != value:
            errors.append(f"{name} must be {value!r}, got {identity.get(name)!r}")
    if not SHA_RE.fullmatch(expected_commit):
        errors.append("expected commit must be a lowercase 40-character Git SHA")
    for name in ("version", "build_id"):
        if not _is_nonempty_printable_single_line_ascii(identity.get(name)):
            errors.append(
                f"{name} must be a non-empty printable single-line ASCII string"
            )
    if errors:
        raise ValueError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("identity", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--build-id")
    args = parser.parse_args()
    try:
        identity = json.loads(args.identity.read_text(encoding="utf-8"))
        if not isinstance(identity, dict):
            raise ValueError("identity must be an object")
        validate_identity(
            identity,
            expected_version=args.version,
            expected_commit=args.commit,
            expected_build_id=args.build_id,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"FrequenSolver identity is valid for {args.version} at {args.commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
