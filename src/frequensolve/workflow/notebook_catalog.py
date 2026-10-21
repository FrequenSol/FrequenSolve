"""Bounded local registration history; no notebook execution or Cloud authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path

from .client import read_local, relative
from .notebook_binding import validate
from .storage import PrivateStore, canonical, directory

VERSION = "notebook-history.v1"
PAGE_SIZE = 25
MAX_DIRECTORY_ENTRIES = 1000
MAX_AUTOMATIC_RECORDS = 128
MAX_METADATA_BYTES = 16384


def bookmark_id(bookmark: dict) -> str:
    return hashlib.sha256(canonical(bookmark)).hexdigest()


def archive(store: PrivateStore, bookmark: dict | None) -> None:
    """Called under the existing latest-bookmark lock, only on registration."""
    if bookmark is None:
        return
    payload = {"version": VERSION, "bookmark": bookmark}
    with store.locked("notebook-" + bookmark_id(bookmark)) as record:
        previous = record.read()
        if previous is None:
            record.write(payload)
        elif previous != payload:
            raise ValueError("Notebook catalog record conflicts with its identity")


def validate_record(bookmark: dict) -> dict:
    if not isinstance(bookmark, dict) or len(canonical(bookmark)) > MAX_METADATA_BYTES:
        raise ValueError("Invalid bounded notebook metadata")
    relative(bookmark["destination"])
    files = bookmark["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= 32:
        raise ValueError("Invalid notebook file count")
    names, total = set(), 0
    for item in files:
        name = relative(item["name"])
        if "/" in name or name in names:
            raise ValueError("Invalid notebook file name")
        names.add(name)
        if (
            type(item["bytes"]) is not int
            or not 0 <= item["bytes"] <= 16 * 1024 * 1024
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid notebook file identity")
        total += item["bytes"]
    if total > 64 * 1024 * 1024:
        raise ValueError("Notebook exceeds the file-byte limit")
    status = bookmark["recorded_replay_status"]
    if not isinstance(status, str) or not 1 <= len(status) <= 64:
        raise ValueError("Invalid recorded notebook state")
    if "label" in bookmark and (
        not isinstance(bookmark["label"], str) or not 1 <= len(bookmark["label"]) <= 128
    ):
        raise ValueError("Invalid notebook label")
    return deepcopy(bookmark)


def _latest(project: Path) -> tuple[dict | None, bool]:
    try:
        value = json.loads(
            read_local(
                project,
                ".frequensolve/workflow/latest-notebook.json",
                MAX_METADATA_BYTES,
            )
        )
        bookmark_id(value)
        return value, False
    except FileNotFoundError:
        return None, False
    except (OSError, ValueError, TypeError):
        return None, True


def _index(project: Path) -> tuple[list[str], dict | None, bool]:
    latest, uncertain = _latest(project)
    keys = {bookmark_id(latest)} if latest is not None else set()
    try:
        with (
            directory(project / ".frequensolve/workflow") as fd,
            os.scandir(fd) as entries,
        ):
            for count, entry in enumerate(entries):
                if count >= MAX_DIRECTORY_ENTRIES:
                    raise ValueError("Notebook catalog exceeds the directory limit")
                found = re.fullmatch(r"notebook-([0-9a-f]{64})\.json", entry.name)
                if found:
                    keys.add(found[1])
    except FileNotFoundError:
        pass
    return sorted(keys), latest, uncertain


def _get(project: Path, key: str, latest: dict | None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Use a returned notebook selection")
    if latest is not None and bookmark_id(latest) == key:
        value = latest
    else:
        payload = json.loads(
            read_local(
                project,
                ".frequensolve/workflow/notebook-" + key + ".json",
                MAX_METADATA_BYTES + 128,
            )
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"version", "bookmark"}
            or payload.get("version") != VERSION
        ):
            raise ValueError("Unsupported notebook catalog record")
        value = payload["bookmark"]
        if bookmark_id(value) != key:
            raise ValueError("Notebook catalog identity changed")
    return {"bookmark_id": key, "bookmark": validate_record(value)}


def get(project: Path, key: str) -> dict:
    """Resolve one explicit returned selection without scanning or writing."""
    latest, _ = _latest(project)
    return _get(project, key, latest)


def page(project: Path, after: str | None = None, *, limit: int = PAGE_SIZE) -> dict:
    if after is not None and not re.fullmatch(r"[0-9a-f]{64}", after):
        raise ValueError("Use the returned notebook cursor")
    keys, latest, uncertain = _index(project)
    remaining = [key for key in keys if after is None or key > after]
    records, unreadable = [], int(uncertain)
    for key in remaining[:limit]:
        try:
            records.append(_get(project, key, latest))
        except (OSError, ValueError, TypeError, KeyError):
            unreadable += 1
    return {
        "records": records,
        "after": after,
        "next_after": remaining[limit - 1] if len(remaining) > limit else None,
        "unreadable_entries": unreadable,
        "history_complete": after is None
        and len(remaining) <= limit
        and unreadable == 0,
        "scope": "registered local notebook bookmarks; unregistered folders are not searched",
        "pagination": "Registrations may change between pages; retain earlier entries and deduplicate by notebook_id. Choice numbers refer to the current page.",
    }


def candidates(project: Path) -> dict:
    return page(project, limit=MAX_AUTOMATIC_RECORDS)


def summary(entry: dict) -> dict:
    saved = entry["bookmark"]
    binding = saved.get("binding")
    try:
        validate(binding)
    except (ValueError, TypeError):
        binding = None
    return {
        "notebook_id": entry["bookmark_id"],
        "label": saved.get("label") or Path(saved["destination"]).name,
        "folder": Path(saved["destination"]).name,
        "recorded_replay_status": saved["recorded_replay_status"],
        "frequency_indices": binding["frequency_indices"] if binding else [],
        "frequency_label": (
            ", ".join(
                f"{float(f):.6g} Hz"
                + (f" (Laplace {float(l):.6g})" if float(l) != 0 else "")
                for f, l in binding["frequency_coordinates"]
            )
            if binding
            else "Unbound older context"
        ),
        "binding_status": "bound" if binding else "unbound",
        "saved_bytes_checked": False,
    }
