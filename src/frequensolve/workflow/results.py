"""Selective authenticated downloads and portable, bounded native result bundles."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from uuid import uuid4

from .client import WorkflowClient, read_local, relative, transfer_bytes
from .native import pressure_snapshot
from .storage import canonical, directory

MAX_BYTES = 16 * 1024 * 1024


def public_results(response: dict) -> dict:
    value = response.get("results", {})
    fields = (
        "version",
        "execution_id",
        "plan_id",
        "plan_revision",
        "outcome",
        "representation",
        "plan_digest",
        "digest",
    )
    result = {name: value[name] for name in fields if name in value}
    result["frequencies"] = []
    for row in value.get("frequencies", []):
        safe = {
            key: row.get(key)
            for key in (
                "frequency_index",
                "frequency_value",
                "state",
                "requested",
                "scientific",
                "native_error",
            )
        }
        safe["actual"] = {
            key: item
            for key, item in (row.get("actual") or {}).items()
            if key != "provider_attempt_id"
        }
        safe["archive"] = (
            {
                key: row["archive"].get(key)
                for key in ("file_count", "total_bytes", "snapshot_digest")
            }
            if row.get("archive")
            else None
        )
        result["frequencies"].append(safe)
    return result


def _publish(root: Path, name: str, content: bytes) -> None:
    parts = relative(name).split("/")
    with directory(root.joinpath(*parts[:-1])) as fd:
        temporary = ".workflow-result-" + uuid4().hex
        handle = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=fd,
        )
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # A hard-link publication is atomic and cannot replace a destination.
            os.link(
                temporary,
                parts[-1],
                src_dir_fd=fd,
                dst_dir_fd=fd,
                follow_symlinks=False,
            )
            os.fsync(fd)
        finally:
            os.unlink(temporary, dir_fd=fd)


def download_results(
    client: WorkflowClient,
    arguments: dict,
    frequencies: list[int],
    root: Path,
    destination: str,
) -> dict:
    result = public_results(client.call("plan_results", arguments))
    if (
        not frequencies
        or len(frequencies) > 2
        or len(set(frequencies)) != len(frequencies)
        or any(type(index) is not int or index < 0 for index in frequencies)
    ):
        raise ValueError("Select one or two qualified frequency indices")
    selected = []
    payloads = {}
    total = 0
    for index in frequencies:
        rows = result["frequencies"]
        if (
            index >= len(rows)
            or rows[index]["frequency_index"] != index
            or not rows[index].get("archive")
        ):
            raise ValueError("Selected frequency has no retained native archive")
        row = rows[index]
        files = []
        cursor = None
        seen = set()
        while True:
            page = client.call(
                "plan_files",
                {
                    **arguments,
                    "frequency_index": index,
                    **({"cursor": cursor} if cursor is not None else {}),
                },
            )["files"]
            files.extend(page["items"])
            if len(files) > 128:
                raise ValueError("Native file listing exceeds the pilot limit")
            cursor = page["next_cursor"]
            if cursor is None:
                break
            if cursor in seen:
                raise ValueError("Native file listing repeated a cursor")
            seen.add(cursor)
        required = [
            item
            for item in files
            if item.get("schema")
            in ("fs_trace_metadata_v1", "fs_trace_payload_shard_v1")
        ]
        if (
            sum(item.get("schema") == "fs_trace_metadata_v1" for item in required) != 1
            or sum(
                item.get("schema") == "fs_trace_payload_shard_v1" for item in required
            )
            != 1
        ):
            raise ValueError("Native trace dependency closure is unsupported")
        records = []
        for item in required:
            total += item["size"]
            if total > MAX_BYTES:
                raise ValueError("Selected receiver payloads exceed 16 MiB")
            ticket = client.call(
                "result_ticket",
                {
                    **arguments,
                    "frequency_index": index,
                    "file_index": item["file_index"],
                },
            )["download"]
            if (
                ticket.get("path") != item["path"]
                or ticket.get("size") != item["size"]
                or ticket.get("execution_id") != arguments["execution_id"]
                or ticket.get("frequency_index") != index
            ):
                raise ValueError("Download ticket belongs to different native output")
            value = transfer_bytes(ticket)
            path = f"frequency-{index:06d}/" + relative(item["path"])
            payloads[path] = value
            records.append(
                {
                    "path": path,
                    "size": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                    "schema": item["schema"],
                }
            )
        selected.append({"frequency": row, "files": records})
    manifest = {
        "version": "workflow-native-bundle.v1",
        "results": result,
        "selected": selected,
        "evidence_origin": "local_record_of_authenticated_exact_version_download",
        "service_attestation": False,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", canonical(manifest))
        for path, value in payloads.items():
            archive.writestr(path, value)
    content = buffer.getvalue()
    if len(content) > MAX_BYTES:
        raise ValueError("Result bundle exceeds its total file limit")
    _publish(root, destination, content)
    return {
        "relative_path": destination,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "frequency_indices": frequencies,
        "input_evidence": manifest["evidence_origin"],
        "service_attestation": False,
    }


def snapshot_from_bundle(root: Path, path: str, frequency_index: int) -> dict:
    content = read_local(root, path, MAX_BYTES)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if (
            len(names) > 8
            or len(set(names)) != len(names)
            or sum(info.file_size for info in archive.infolist()) > MAX_BYTES
        ):
            raise ValueError("Native bundle exceeds its bounded file/byte limit")
        for name in names:
            relative(name)
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("version") != "workflow-native-bundle.v1":
            raise ValueError("Unsupported native result bundle")
        rows = [
            item
            for item in manifest["selected"]
            if item["frequency"]["frequency_index"] == frequency_index
        ]
        if len(rows) != 1:
            raise ValueError("Selected frequency is unavailable")
        files = {}
        for item in rows[0]["files"]:
            value = archive.read(relative(item["path"]))
            if (
                len(value) != item["size"]
                or hashlib.sha256(value).hexdigest() != item["sha256"]
                or item["schema"] in files
            ):
                raise ValueError("Native result bytes or dependency identity changed")
            files[item["schema"]] = value
        if set(files) != {"fs_trace_metadata_v1", "fs_trace_payload_shard_v1"}:
            raise ValueError("Native trace dependency closure is incomplete")
        result = manifest["results"]
        return pressure_snapshot(
            files["fs_trace_payload_shard_v1"],
            files["fs_trace_metadata_v1"],
            rows[0]["frequency"],
            {
                "origin": "local_bundle_receipt",
                "bundle_sha256": hashlib.sha256(content).hexdigest(),
                "input_hashes_verified": True,
                "service_attestation": False,
                "plan_id": result["plan_id"],
                "plan_revision": result["plan_revision"],
                "execution_id": result["execution_id"],
                "result_digest": result["digest"],
                "frequency_index": frequency_index,
                "files": [
                    {"sha256": item["sha256"], "schema": item["schema"]}
                    for item in rows[0]["files"]
                ],
            },
        )
