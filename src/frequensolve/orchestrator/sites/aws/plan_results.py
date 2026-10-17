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
        try:
            response = self.site.s3_client.get_object(
                Bucket=ref["bucket"], Key=ref["key"], VersionId=ref["version_id"]
            )
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") not in {
                "AccessDenied",
                "403",
            }:
                raise
            # Standard Cloud credentials may only read current objects. S3
            # returns the atomically selected VersionId with the bytes; the
            # checks below still require the exact saved version AND digest.
            # A moved alias cannot substitute another revision's results.
            response = self.site.s3_client.get_object(
                Bucket=ref["bucket"], Key=ref["key"]
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

    def _local_archive(self, directory, frequency_index):
        if type(frequency_index) is not int or not 0 <= frequency_index < len(
            self._data["frequencies"]
        ):
            raise PlanError("Choose a saved frequency index")
        root = Path(directory).expanduser().resolve()
        saved = root / "execution-results.json"
        if saved.stat().st_size > 512 * 1024 or digest(
            json.loads(saved.read_bytes())
        ) != digest(self._data):
            raise PlanError("Local results belong to a different saved execution")
        row = self._data["frequencies"][frequency_index]
        archive = row.get("archive")
        if archive is None:
            raise PlanError("This frequency has no retained native outputs")
        folder = root / f"frequency-{frequency_index:06d}"
        manifest_file = folder / "archive.json"
        if manifest_file.stat().st_size > 4 * 1024**2:
            raise PlanError("Local native archive exceeds its bounded manifest")
        raw = manifest_file.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != archive["manifest"]["digest"]:
            raise PlanError("Local native archive content changed")
        manifest = json.loads(raw)
        if (
            manifest.get("identity") != archive["identity"]
            or manifest.get("snapshot_digest") != archive["snapshot_digest"]
            or manifest.get("digest")
            != digest({key: item for key, item in manifest.items() if key != "digest"})
        ):
            raise PlanError("Local native archive lost its accepted identity")
        files = {item["path"]: item for item in manifest["files"]}
        if len(files) != archive["file_count"]:
            raise PlanError("Local native archive frequency coverage changed")
        return folder / "files", row, files

    @staticmethod
    def _verified_file(root, files, relative):
        relative = _relative(relative)
        item = files.get(relative)
        if item is None:
            raise PlanError("Native artifact was not retained by this execution")
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise PlanError("Native artifact path escaped the downloaded frequency")
        with path.open("rb") as stream:
            checksum, count = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(block)
                if count > item["size"]:
                    raise PlanError("Local native artifact size changed")
                checksum.update(block)
        if count != item["size"] or "sha256:" + checksum.hexdigest() != item["digest"]:
            raise PlanError("Local native artifact bytes changed")
        return path

    def artifact_catalog(self, directory: str | Path, *, frequency_index: int):
        """Read one retained frequency's producer-authored logical artifacts."""
        import math

        from frequensolve.simulation.artifact_contract import ArtifactCatalog

        root, row, files = self._local_archive(directory, frequency_index)
        relative = f"_fs_run/tasks/task_{frequency_index + 1:06d}/result.json"
        self._verified_file(root, files, relative)
        if files[relative]["digest"] != row["archive"]["identity"].get(
            "native_result_digest"
        ):
            raise PlanError("Native task result differs from its retained root proof")
        catalog = ArtifactCatalog.read_task_results(root, tasks=[frequency_index + 1])
        task = catalog.results.get(frequency_index + 1)
        expected = complex(*map(float, row["frequency_value"]))
        if task is None or not (
            math.isclose(
                task.partition.frequency.real,
                expected.real,
                rel_tol=1e-12,
                abs_tol=1e-10,
            )
            and math.isclose(
                task.partition.frequency.imag,
                expected.imag,
                rel_tol=1e-12,
                abs_tol=1e-10,
            )
        ):
            raise PlanError("Native artifact catalog belongs to a different frequency")
        for artifact in catalog.artifacts_for_task(frequency_index + 1):
            item = files.get(artifact.relative_path)
            if item is None or item["size"] != artifact.bytes:
                raise PlanError("Native artifact catalog refers to unretained files")
        return catalog

    def trace_dataset(self, directory: str | Path, *, frequency_index: int):
        """Open a converged receiver gather with its own layout-specific metadata.

        Frequencies stay independent. This does not merge unlike survey metadata
        or pretend that a combined packed sweep has been produced.
        """
        import math

        import h5py
        import numpy as np

        from frequensolve.seismic.trace_store import TraceStore
        from frequensolve.seismic.traces import TraceDataset
        from frequensolve.simulation.jobs.artifacts import TraceManifest

        root, row, files = self._local_archive(directory, frequency_index)
        if (row.get("scientific") or {}).get("converged") is not True:
            raise PlanError(
                "This frequency has no verified converged traces; inspect its raw artifacts"
            )
        catalog = self.artifact_catalog(directory, frequency_index=frequency_index)
        shards = catalog.query(role="simulated_traces", representation="hdf5_shard")
        metadata = catalog.query(role="traces_metadata")
        if (
            len(shards) != 1
            or len(metadata) != 1
            or shards[0].schema != "fs_trace_payload_shard_v1"
            or metadata[0].schema != "fs_trace_metadata_v1"
        ):
            raise PlanError(
                "This receiver trace family is not qualified for saved-plan loading"
            )
        shard = self._verified_file(root, files, shards[0].relative_path)
        survey = self._verified_file(root, files, metadata[0].relative_path)
        groups = TraceStore.discover_trace_groups(survey)
        real, imaginary = map(float, row["frequency_value"])
        frequencies = TraceStore._read_trace_frequencies(shard)
        laplace = TraceStore._read_trace_laplace_values(shard)
        with h5py.File(shard, "r") as data:
            task = (
                np.asarray(data["task_id"][()]).ravel()
                if "task_id" in data
                else np.array([])
            )
        if (
            not groups
            or len(frequencies) != 1
            or len(laplace) != 1
            or task.size != 1
            or task.dtype.kind not in "iu"
            or int(task[0]) != frequency_index + 1
            or not math.isclose(frequencies[0], real, rel_tol=1e-12, abs_tol=1e-10)
            or not math.isclose(laplace[0], imaginary, rel_tol=1e-12, abs_tol=1e-10)
        ):
            raise PlanError(
                "Trace payload does not match its saved frequency/task/survey"
            )
        manifest = TraceManifest(
            files=[shard],
            frequencies={frequency_index + 1: real},
            laplace={frequency_index + 1: imaginary},
            groups=groups,
            simulation=root,
            result_path=root,
            output_path=root,
            project_path=root,
            artifacts=[*metadata, *shards],
            artifact_contract="fs-task-result-2",
        )
        return TraceDataset.from_manifest(manifest)

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
