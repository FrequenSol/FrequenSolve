#!/usr/bin/env python3
"""Plan FrequenSolve release tags using PEP 440 package versions."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence

VERSION_COMPONENT = r"(?:0|[1-9][0-9]*)"
FINAL_VERSION_PATTERN = (
    rf"{VERSION_COMPONENT}\.{VERSION_COMPONENT}\.{VERSION_COMPONENT}"
)
FINAL_VERSION_RE = re.compile(rf"^{FINAL_VERSION_PATTERN}$")
FINAL_TAG_RE = re.compile(rf"^v(?P<version>{FINAL_VERSION_PATTERN})$")
RC_TAG_RE = re.compile(
    rf"^v(?P<version>{FINAL_VERSION_PATTERN})rc(?P<number>[1-9][0-9]*)$"
)


def validate_final_version(version: str) -> list[str]:
    if FINAL_VERSION_RE.fullmatch(version):
        return []
    return ["version must be canonical ASCII X.Y.Z, such as 0.2.0"]


def validate_release_candidate_tag(tag: str) -> list[str]:
    if RC_TAG_RE.fullmatch(tag):
        return []
    return ["release candidate tag must be canonical ASCII vX.Y.ZrcN with N >= 1"]


def next_release_candidate_tag(version: str, existing_tags: Sequence[str]) -> str:
    errors = validate_final_version(version)
    if errors:
        raise ValueError("; ".join(errors))

    highest = 0
    for tag in existing_tags:
        match = RC_TAG_RE.fullmatch(tag)
        if match is None or match.group("version") != version:
            continue
        highest = max(highest, int(match.group("number")))
    return f"v{version}rc{highest + 1}"


def final_tag_from_release_candidate(tag: str) -> str:
    errors = validate_release_candidate_tag(tag)
    if errors:
        raise ValueError("; ".join(errors))
    match = RC_TAG_RE.fullmatch(tag)
    assert match is not None
    return f"v{match.group('version')}"


def publication_target(tag: str, *, is_prerelease: bool) -> str:
    """Return the only package index allowed by canonical release metadata."""
    if RC_TAG_RE.fullmatch(tag):
        if not is_prerelease:
            raise ValueError(
                "release candidate tags must be published as GitHub prereleases"
            )
        return "testpypi"

    if FINAL_TAG_RE.fullmatch(tag):
        if is_prerelease:
            raise ValueError(
                "final release tags must not be published as GitHub prereleases"
            )
        return "pypi"

    raise ValueError(
        "release tag must be canonical ASCII vX.Y.Z or vX.Y.ZrcN with N >= 1"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    next_rc = subparsers.add_parser(
        "next-rc", help="Print the next release candidate tag for a final version."
    )
    next_rc.add_argument("--version", required=True)
    next_rc.add_argument(
        "--existing-tag",
        action="append",
        default=[],
        help="Existing tag to consider. May be repeated.",
    )

    final = subparsers.add_parser(
        "final-from-rc", help="Print the final release tag for a release candidate."
    )
    final.add_argument("--rc-tag", required=True)

    publication = subparsers.add_parser(
        "publication-target",
        help="Derive the allowed package index from tag and release metadata.",
    )
    publication.add_argument("--tag", required=True)
    publication.add_argument(
        "--prerelease",
        required=True,
        choices=("true", "false"),
        help="GitHub release prerelease state.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "next-rc":
            print(next_release_candidate_tag(args.version, args.existing_tag))
            return 0
        if args.command == "final-from-rc":
            print(final_tag_from_release_candidate(args.rc_tag))
            return 0
        if args.command == "publication-target":
            print(
                publication_target(
                    args.tag,
                    is_prerelease=args.prerelease == "true",
                )
            )
            return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
