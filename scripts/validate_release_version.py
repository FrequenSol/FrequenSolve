#!/usr/bin/env python3
"""Validate that a FrequenSolve package release comes from its matching tag."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence

VERSION_COMPONENT = r"(?:0|[1-9][0-9]*)"
RELEASE_VERSION_RE = re.compile(
    rf"^{VERSION_COMPONENT}\.{VERSION_COMPONENT}\.{VERSION_COMPONENT}"
    r"(?:rc[1-9][0-9]*)?$"
)


def _git_output(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _versioneer_version() -> str:
    import versioneer

    return versioneer.get_version()


def _current_ref_type() -> str:
    ref_type = os.environ.get("GITHUB_REF_TYPE", "").strip()
    if ref_type:
        return ref_type
    return "tag" if _git_output(["describe", "--exact-match", "--tags", "HEAD"]) else ""


def _current_ref_name() -> str:
    ref_name = os.environ.get("GITHUB_REF_NAME", "").strip()
    if ref_name:
        return ref_name
    return _git_output(["describe", "--exact-match", "--tags", "HEAD"])


def validate_release_version(
    *, version: str, ref_type: str, ref_name: str
) -> list[str]:
    errors: list[str] = []
    normalized_version = version.strip()
    normalized_ref_type = ref_type.strip()
    normalized_ref_name = ref_name.strip()

    if normalized_ref_type != "tag":
        errors.append("release must run from a tag ref")

    if "dirty" in normalized_version or "untagged" in normalized_version:
        errors.append("version must not include dirty or untagged markers")

    if "+" in normalized_version:
        errors.append("version must not include a local version segment")

    if version != normalized_version or not RELEASE_VERSION_RE.fullmatch(version):
        errors.append("version must be canonical ASCII X.Y.Z or X.Y.ZrcN with N >= 1")

    expected_tag = f"v{normalized_version}"
    if normalized_ref_name != expected_tag:
        errors.append(f"release tag must be {expected_tag}")

    return errors


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FrequenSolve package release version and git tag."
    )
    parser.add_argument("--version", default="", help="Versioneer version override.")
    parser.add_argument("--ref-type", default="", help="GitHub ref type override.")
    parser.add_argument("--ref-name", default="", help="GitHub ref name override.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    version = args.version or _versioneer_version()
    ref_type = args.ref_type or _current_ref_type()
    ref_name = args.ref_name or _current_ref_name()

    errors = validate_release_version(
        version=version,
        ref_type=ref_type,
        ref_name=ref_name,
    )
    if errors:
        print("Invalid FrequenSolve release version:", file=sys.stderr)
        print(f"  version: {version}", file=sys.stderr)
        print(f"  ref_type: {ref_type or '<unknown>'}", file=sys.stderr)
        print(f"  ref_name: {ref_name or '<unknown>'}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"FrequenSolve release version validated: {ref_name} -> {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
