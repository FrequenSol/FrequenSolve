"""Safely extract a solver-backed test-evidence archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def _member_path(member: tarfile.TarInfo) -> PurePosixPath:
    """Return a normalized, relative member path or raise ``ValueError``."""
    if "\\" in member.name:
        raise ValueError(f"archive member uses a non-portable path: {member.name!r}")
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive member escapes the destination: {member.name!r}")
    if not (member.isdir() or member.isreg()):
        raise ValueError(
            f"archive member must be a regular file or directory: {member.name!r}"
        )
    if not path.parts and not member.isdir():
        raise ValueError("archive root member must be a directory")
    return path


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract only validated regular files and directories.

    Files are copied manually after every member has been validated. This avoids
    tar link, device, and path-traversal behavior on every supported Python
    version instead of depending on the runtime's ``tarfile`` extraction filter.
    """
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        normalized: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        seen: set[PurePosixPath] = set()
        for member in members:
            path = _member_path(member)
            if path in seen:
                raise ValueError(f"archive contains a duplicate path: {member.name!r}")
            seen.add(path)
            normalized.append((member, path))

        if destination.exists() or destination.is_symlink():
            raise ValueError(f"destination must not already exist: {destination}")
        destination.mkdir(parents=True)
        root = destination.resolve()
        for member, path in normalized:
            if not path.parts:
                continue
            target = root.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name!r}")
            try:
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            finally:
                source.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        extract_archive(args.archive, args.destination)
    except (OSError, tarfile.TarError, ValueError) as exc:
        parser.error(str(exc))
    print(f"extracted validated test evidence to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
