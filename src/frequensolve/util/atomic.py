"""Durable, atomic publication helpers for local filesystem artifacts."""

from __future__ import annotations

import json
import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

__all__ = ["atomic_output_path", "atomic_write_json", "atomic_write_text"]

_DEFAULT_FILE_MODE = 0o644
_TEMPORARY_NAME_ATTEMPTS = 128


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_temporary(destination: Path, mode: int) -> tuple[int, Path]:
    try:
        existing_mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        existing_mode = None

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    for _ in range(_TEMPORARY_NAME_ATTEMPTS):
        token = secrets.token_hex(12)
        temporary = destination.with_name(
            f".{destination.name}.{token}.tmp{destination.suffix}"
        )
        try:
            descriptor = os.open(temporary, flags, mode)
        except FileExistsError:
            continue
        try:
            if existing_mode is not None:
                os.fchmod(descriptor, existing_mode)
        except BaseException:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        return descriptor, temporary
    raise FileExistsError(f"Could not create a unique temporary file for {destination}")


@contextmanager
def atomic_output_path(
    path: Union[str, Path],
    *,
    mode: int = _DEFAULT_FILE_MODE,
) -> Iterator[Path]:
    """Yield a unique temporary path and durably publish it on success.

    The temporary file is created beside the destination so ``os.replace`` is
    atomic. The completed file and its parent directory are synced before the
    context returns. An exception before replacement preserves the previously
    published destination, while any abandoned temporary file is removed.

    Args:
        path: Final artifact path.
        mode: Permissions requested for a new artifact, subject to the process
            umask. An existing destination's permissions are preserved.

    Yields:
        Empty temporary file for the caller to populate and close.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _create_temporary(destination, mode)
    try:
        os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        yield temporary
        _fsync_file(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    path: Union[str, Path],
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = _DEFAULT_FILE_MODE,
) -> Path:
    """Atomically and durably replace a text file."""

    destination = Path(path)
    with atomic_output_path(destination, mode=mode) as temporary:
        temporary.write_text(text, encoding=encoding)
    return destination


def atomic_write_json(
    path: Union[str, Path],
    payload: Any,
    *,
    trailing_newline: bool = False,
    mode: int = _DEFAULT_FILE_MODE,
    **json_kwargs: Any,
) -> Path:
    """Atomically and durably replace a JSON document."""

    destination = Path(path)
    with atomic_output_path(destination, mode=mode) as temporary:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, **json_kwargs)
            if trailing_newline:
                stream.write("\n")
    return destination
