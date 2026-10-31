"""Atomic ownership check for a persistent remote project directory.

This module also runs on login nodes with only the Python standard library.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

MARKER = ".frequensolve-project.json"
SCHEMA = "frequensolve-remote-project-1"


def claim_project(root: Path, source: str) -> None:
    """Claim an empty/recognized project root, or verify its existing owner."""
    root = Path(root)
    if not source or not Path(source).is_absolute():
        raise ValueError("Project identity must be an absolute local project path")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / MARKER
    expected = {"schema": SCHEMA, "source_project": source}

    def verify():
        actual = json.loads(marker.read_text())
        if actual != expected:
            owner = (
                actual.get("source_project", "unknown")
                if isinstance(actual, dict)
                else "unknown"
            )
            raise ValueError(
                f"Remote project directory {root} belongs to {owner!r}, not {source!r}. "
                "Use that project's existing local directory or select a different work_dir. "
                "No project files were uploaded."
            )

    if marker.exists():
        verify()
        return

    # Older versions copied the project manifest without rewriting its source
    # project paths. Adopt only a directory whose manifests identify one owner.
    origins = set()
    manifests = False
    for file in root.glob("*.json"):
        try:
            data = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("simulations"), list):
            continue
        manifests = True
        for simulation in data["simulations"]:
            origin = (
                simulation.get("project_path") if isinstance(simulation, dict) else None
            )
            origins.add(origin if isinstance(origin, str) else None)
    populated = manifests or any(
        (root / name).exists() and any((root / name).iterdir())
        for name in ("simulations", "jobs")
    )
    if populated and origins != {source}:
        raise ValueError(
            f"Remote project directory {root} already contains project data with "
            f"unverified or different ownership. Refusing to overwrite it with {source!r}; "
            "select a different work_dir or verify the existing project metadata."
        )

    # Link a fully written file instead of exposing a partially written marker.
    # Competing clients cannot both claim the same root for different projects.
    fd, temporary = tempfile.mkstemp(prefix=".fs-project-claim-", dir=root)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(expected, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, marker)
        except FileExistsError:
            pass
        verify()
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    """Return a machine-readable claim result to the submitting client."""
    import sys

    try:
        claim_project(Path(sys.argv[1]), sys.argv[2])
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
    else:
        print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()
