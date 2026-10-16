"""Read and download exact saved-plan native archives without compute effects."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from frequensolve.execution_plan import PlanError, digest


def _relative(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or "\\" in value
        or re.search(r"[\x00-\x1f\x7f]", value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise PlanError("Saved result paths must stay inside the accepted run")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise PlanError("Saved result has no valid content digest")
    return value


class ExecutionResults:
    """Server-authorized result index and version-pinned native files.

    Download keeps separate frequency directories. It does not claim that a
    legacy combined packed product or missing memory telemetry is available.
    """

    def __init__(self, site, response, *, plan_id, execution_id, project):
        head, value = response.get("head"), response.get("results")
        location = response.get("result_location")
        if (
            not isinstance(head, dict)
            or head.get("plan_id") != plan_id
            or head.get("execution_id") != execution_id
            or head.get("status") != "executed"
            or head.get("owner", {}).get("project_key") != project
            or not isinstance(value, dict)
            or value.get("version") != "frequency-execution-results.v1"
            or value.get("execution_id") != execution_id
            or value.get("plan_id") != plan_id
            or type(value.get("plan_revision")) is not int
            or value["plan_revision"] < 1
            or value.get("plan_revision") != head.get("revision")
            or value.get("representation") != "per-frequency-native-archives.v1"
            or value.get("combined_packed_product") is not None
            or value.get("digest")
            != digest({key: item for key, item in value.items() if key != "digest"})
            or not isinstance(location, dict)
            or set(location) != {"bucket", "result_s3_prefix"}
            or location.get("bucket") != site.config.s3_bucket
        ):
            raise PlanError("Cloud returned different or incompatible saved results")
        if not isinstance(execution_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", execution_id
        ):
            raise PlanError("Invalid saved execution identity")
        prefix = location.get("result_s3_prefix")
        if (
            not isinstance(prefix, str)
            or not prefix.endswith("/runs/" + execution_id + "/")
            or not prefix.startswith(project + "/")
            or _relative(prefix[:-1]) + "/" != prefix
        ):
            raise PlanError("Cloud returned a foreign saved result destination")
        rows = value.get("frequencies")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 96:
            raise PlanError("Saved results have incomplete frequency coverage")
        for index, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or type(row.get("frequency_index")) is not int
                or row["frequency_index"] != index
            ):
                raise PlanError("Saved results changed frequency identity")
            archive = row.get("archive")
            if archive is None:
                continue
            if (
                not isinstance(archive, dict)
                or archive.get("version") != "frequency-output-archive.v1"
                or archive.get("identity", {}).get("execution_id") != execution_id
                or archive.get("identity", {}).get("frequency_index") != index
                or archive.get("identity", {}).get("plan_digest")
                != value.get("plan_digest")
            ):
                raise PlanError("Saved native archive belongs to different work")
            if (
                type(archive.get("file_count")) is not int
                or not 3 <= archive["file_count"] <= 2003
                or type(archive.get("total_bytes")) is not int
                or not 0 <= archive["total_bytes"] <= 5 * 1024**4
            ):
                raise PlanError("Saved native archive exceeds its bounded capacity")
            snapshot = _digest(archive.get("snapshot_digest"))
            self._reference(
                archive.get("manifest"),
                location["bucket"],
                prefix
                + f"frequency-results/frequency-{index:06d}/manifests/"
                + snapshot[7:]
                + ".json",
            )
        self.site, self._data, self._location = (
            site,
            deepcopy(value),
            deepcopy(location),
        )

    @property
    def data(self):
        return deepcopy(self._data)

    @staticmethod
    def _reference(ref, bucket, key):
        if (
            not isinstance(ref, dict)
            or set(ref) != {"bucket", "key", "version_id", "digest"}
            or ref["bucket"] != bucket
            or ref["key"] != key
            or not isinstance(ref["version_id"], str)
            or not ref["version_id"]
            or ref["version_id"] == "null"
        ):
            raise PlanError("Saved archive has a foreign immutable provider reference")
        _digest(ref["digest"])
        return ref

    def _download(self, ref, *, maximum, output=None, expected_size=None):
        response = self.site.s3_client.get_object(
            Bucket=ref["bucket"], Key=ref["key"], VersionId=ref["version_id"]
        )
        body = response.get("Body")
        try:
            size = response.get("ContentLength")
            if (
                response.get("VersionId") != ref["version_id"]
                or type(size) is not int
                or not 0 <= size <= maximum
                or expected_size is not None
                and size != expected_size
                or body is None
            ):
                raise PlanError(
                    "Saved result has incomplete immutable provider evidence"
                )
            total, checksum, chunks = 0, hashlib.sha256(), []
            for block in iter(lambda: body.read(1024 * 1024), b""):
                total += len(block)
                if total > size:
                    raise PlanError("Saved result exceeds its declared size")
                checksum.update(block)
                if output is None:
                    chunks.append(block)
                else:
                    output.write(block)
            if total != size or "sha256:" + checksum.hexdigest() != ref["digest"]:
                raise PlanError(
                    "Saved result bytes differ from the accepted content digest"
                )
            return b"".join(chunks) if output is None else None
        finally:
            if body is not None:
                body.close()

    def download(self, directory: str | Path) -> Path:
        """Download all retained outputs once; never overwrite an existing run.

        Missing/failed frequencies stay in the index. The local result directory
        becomes visible only after every retained file is version/hash checked.
        """
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = root / self._data["execution_id"]
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                "Saved result directory already exists; choose another destination"
            )
        with tempfile.TemporaryDirectory(
            prefix=".fs-result-download-", dir=root
        ) as temporary:
            stage = Path(temporary) / "results"
            stage.mkdir(mode=0o700)
            for row in self._data["frequencies"]:
                archive = row.get("archive")
                if archive is None:
                    continue
                raw = self._download(archive["manifest"], maximum=4 * 1024**2)
                manifest = json.loads(raw)
                if (
                    not isinstance(manifest, dict)
                    or set(manifest)
                    != {
                        "version",
                        "identity",
                        "snapshot_digest",
                        "total_bytes",
                        "files",
                        "digest",
                    }
                    or manifest["version"] != "frequency-output-archive-manifest.v1"
                    or manifest["identity"] != archive["identity"]
                    or manifest["snapshot_digest"] != archive["snapshot_digest"]
                    or manifest["total_bytes"] != archive["total_bytes"]
                    or not isinstance(manifest["files"], list)
                    or len(manifest["files"]) != archive["file_count"]
                    or manifest["digest"]
                    != digest(
                        {key: item for key, item in manifest.items() if key != "digest"}
                    )
                ):
                    raise PlanError(
                        "Native archive differs from the accepted saved result"
                    )
                prefix = (
                    self._location["result_s3_prefix"]
                    + f"frequency-results/frequency-{row['frequency_index']:06d}/"
                )
                folder = stage / f"frequency-{row['frequency_index']:06d}"
                folder.mkdir(mode=0o700)
                seen, total = set(), 0
                for item in manifest["files"]:
                    if not isinstance(item, dict) or set(item) != {
                        "path",
                        "size",
                        "digest",
                        "kind",
                        "schema",
                        "object",
                    }:
                        raise PlanError("Native archive has an unsupported file record")
                    path, checksum = _relative(item["path"]), _digest(item["digest"])
                    if (
                        path in seen
                        or type(item["size"]) is not int
                        or not 0 <= item["size"] <= 5 * 1024**3
                    ):
                        raise PlanError(
                            "Native archive has duplicate or oversized output files"
                        )
                    seen.add(path)
                    reference = item["object"]
                    if not isinstance(reference, dict) or set(reference) != {
                        "key",
                        "version_id",
                    }:
                        raise PlanError("Native file has no exact provider reference")
                    ref = {
                        **reference,
                        "bucket": self._location["bucket"],
                        "digest": checksum,
                    }
                    self._reference(
                        ref,
                        self._location["bucket"],
                        prefix + "objects/" + checksum[7:],
                    )
                    local = folder / "files" / path
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with local.open("xb") as output:
                        self._download(
                            ref,
                            maximum=item["size"],
                            expected_size=item["size"],
                            output=output,
                        )
                        output.flush()
                        os.fsync(output.fileno())
                    total += item["size"]
                if total != archive["total_bytes"]:
                    raise PlanError("Native archive total storage changed")
                (folder / "archive.json").write_bytes(raw)
            (stage / "execution-results.json").write_text(
                json.dumps(
                    self._data,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            stage.rename(target)
        return target
