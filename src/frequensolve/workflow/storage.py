"""Private, locked JSON records with no symlink following or partial writes."""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@contextmanager
def directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Anchor every component rather than resolve through a replaceable link."""
    path = Path(os.path.abspath(path))
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Safe workflow storage is unavailable on this platform")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


class PrivateStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @contextmanager
    def locked(self, name: str) -> Iterator["LockedRecord"]:
        import fcntl

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", name):
            raise ValueError("Invalid workflow record name")
        with directory(self.root, create=True) as fd:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("Workflow storage must be private to its owner")
            # Concurrent first creation can transiently return ENOENT on macOS.
            # Keep the anchored directory and every no-follow check on retry.
            for attempt in range(3):
                try:
                    lock = os.open(
                        name + ".lock",
                        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=fd,
                    )
                    break
                except FileNotFoundError:
                    if attempt == 2:
                        raise
            try:
                if not stat.S_ISREG(os.fstat(lock).st_mode):
                    raise ValueError("Invalid workflow lock")
                fcntl.flock(lock, fcntl.LOCK_EX)
                yield LockedRecord(fd, name + ".json")
            finally:
                os.close(lock)


class LockedRecord:
    def __init__(self, fd: int, name: str):
        self.fd, self.name = fd, name

    def read(self) -> dict | None:
        try:
            fd = os.open(self.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o077
                or info.st_uid != os.getuid()
            ):
                raise ValueError("Invalid private workflow record")
            data = stream.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("Workflow record is too large")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("Invalid workflow record")
        return value

    def write(self, value: dict) -> None:
        data = canonical(value)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("Workflow record is too large")
        name = "." + uuid4().hex
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.fd,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(name, self.name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            os.fsync(self.fd)
        finally:
            try:
                os.unlink(name, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def delete(self) -> None:
        try:
            os.unlink(self.name, dir_fd=self.fd)
            os.fsync(self.fd)
        except FileNotFoundError:
            pass
