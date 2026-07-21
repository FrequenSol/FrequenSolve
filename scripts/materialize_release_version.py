#!/usr/bin/env python3
"""Materialize a validated Versioneer identity into release staging."""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from types import ModuleType

try:
    from scripts.validate_release_version import validate_release_version
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/.
    from validate_release_version import validate_release_version


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ROOT = Path(__file__).resolve().parents[1]


def _load_versioneer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "frequensolve_release_versioneer", ROOT / "versioneer.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the repository Versioneer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_release_version(
    *,
    version: str,
    release_tag: str,
    revision: str,
    output: Path,
) -> None:
    """Write Versioneer's static release module after validating its identity."""

    errors = validate_release_version(
        version=version,
        ref_type="tag",
        ref_name=release_tag,
    )
    if errors:
        raise ValueError("; ".join(errors))
    if not COMMIT_RE.fullmatch(revision):
        raise ValueError("revision must be a lowercase 40-character Git commit")
    if output.resolve() == (ROOT / "src/frequensolve/_version.py").resolve():
        raise ValueError("refusing to rewrite the tracked Versioneer source")

    versions = {
        "version": version,
        "full-revisionid": revision,
        "dirty": False,
        "error": None,
        "date": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _load_versioneer().write_to_version_file(str(output), versions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        materialize_release_version(
            version=args.version,
            release_tag=args.release_tag,
            revision=args.revision,
            output=args.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"materialized FrequenSolve {args.version} from {args.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
