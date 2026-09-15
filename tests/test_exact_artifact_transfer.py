from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager
from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    ArtifactContractError,
    ArtifactRecord,
    ArtifactRequest,
)

_H5_STRING = h5py.string_dtype(encoding="utf-8")


def _write_stale_empty_index(result_path, fingerprint):
    path = result_path / "_fs_run/tasks.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "fs-task-index-1"
        tasks = h5.create_group("tasks")
        artifacts = h5.create_group("artifacts")
        dependencies = h5.create_group("dependencies")
        tasks.create_dataset("task_id", data=np.asarray([1], dtype=np.int64))
        tasks.create_dataset("status", data=["success"], dtype=_H5_STRING)
        tasks.create_dataset("frequency_real", data=[1.0], dtype=np.float64)
        tasks.create_dataset("frequency_imag", data=[0.0], dtype=np.float64)
        for name in (
            "fingerprint_job",
            "fingerprint_simulation",
            "fingerprint_outputs",
        ):
            tasks.create_dataset(name, data=[fingerprint], dtype=_H5_STRING)
        tasks.create_dataset("artifact_offset", data=[0], dtype=np.int64)
        tasks.create_dataset("artifact_count", data=[0], dtype=np.int64)
        tasks.create_dataset("iterations", data=[-1], dtype=np.int64)
        tasks.create_dataset("residual", data=[np.nan], dtype=np.float64)
        for name in (
            "timing_mesh",
            "timing_setup",
            "timing_assembly",
            "timing_solve_forward",
            "timing_solve_adjoint",
            "timing_imaging",
        ):
            tasks.create_dataset(name, data=[np.nan], dtype=np.float64)
        for name in (
            "id",
            "role",
            "schema",
            "representation",
            "path",
            "retention",
            "generation",
        ):
            artifacts.create_dataset(name, data=[], dtype=_H5_STRING)
        for name in ("bytes", "dependency_offset", "dependency_count"):
            artifacts.create_dataset(name, data=[], dtype=np.int64)
        dependencies.create_dataset("id", data=[], dtype=_H5_STRING)
    return path


from frequensolve.simulation.artifact_transfer import (
    collection_parts,
    fetch_artifact_payloads,
    select_transfer_artifacts,
)


def _record(
    root,
    *,
    id,
    role,
    path,
    bytes,
    representation="hdf5_shard",
    schema="payload-1",
    retention="durable",
    dependencies=(),
):
    return ArtifactRecord.from_fs(
        {
            "id": id,
            "role": role,
            "path": path,
            "bytes": bytes,
            "representation": representation,
            "schema": schema,
            "retention": retention,
            "dependencies": list(dependencies),
        },
        result_path=root,
    )


class _Catalog:
    def __init__(self, records):
        self.records = tuple(records)

    def query(self, **filters):
        return [
            record
            for record in self.records
            if all(
                value is None or getattr(record, key) == value
                for key, value in filters.items()
            )
        ]

    def select(self, request, *, task=None):
        del task
        records = self.query(
            id=request.id,
            role=request.role,
            retention=request.retention,
        )
        if request.representations:
            records = [
                record
                for record in records
                if record.representation in request.representations
            ]
        return sorted(records, key=request.preference)


def test_default_selection_excludes_auxiliary_and_transient_records(tmp_path):
    records = [
        _record(
            tmp_path,
            id="traces",
            role="simulated_traces",
            path="traces/data.h5",
            bytes=1,
        ),
        _record(
            tmp_path,
            id="forward-wavefield",
            role="wavefield",
            path="fields/forward.h5",
            bytes=2,
        ),
        _record(
            tmp_path,
            id="restart",
            role="restart_checkpoint",
            path="restart/state.h5",
            bytes=3,
            retention="cache",
        ),
        _record(
            tmp_path,
            id="scratch",
            role="debug",
            path="scratch/debug.h5",
            bytes=4,
            retention="transient",
        ),
        _record(
            tmp_path,
            id="wavefield_illumination",
            role="image",
            path="images/illumination.h5",
            bytes=5,
        ),
    ]
    catalog = _Catalog(records)

    defaults = select_transfer_artifacts(catalog)
    explicit = select_transfer_artifacts(
        catalog,
        requests=(ArtifactRequest(role="wavefield"),),
    )
    explicit_auxiliary = select_transfer_artifacts(
        catalog,
        requests=(
            ArtifactRequest(role="restart_checkpoint", retention="cache"),
            ArtifactRequest(role="debug", retention="transient"),
        ),
        include_defaults=False,
    )

    assert [record.id for record in defaults] == ["traces", "wavefield_illumination"]
    assert [record.id for record in explicit] == [
        "traces",
        "wavefield_illumination",
        "forward-wavefield",
    ]
    assert [record.id for record in explicit_auxiliary] == ["restart", "scratch"]


def test_json_catalog_selection_expands_xmf_dependency(tmp_path):
    payload = _record(
        tmp_path,
        id="visualization-data:pressure",
        role="visualization_data",
        path="vtk/pressure.h5",
        bytes=4,
        representation="hdf5",
    )
    xmf = _record(
        tmp_path,
        id="visualization:pressure:1",
        role="visualization",
        path="vtk/pressure.xmf",
        bytes=3,
        representation="xmf",
        dependencies=(payload.id,),
    )

    class JsonCatalog(_Catalog):
        results = {1: object()}

        def query(self, *, task=None, **filters):
            assert task in (None, 1)
            return super().query(**filters)

    selected = select_transfer_artifacts(
        JsonCatalog((payload, xmf)),
        requests=(ArtifactRequest(role="visualization"),),
        include_defaults=False,
    )

    assert [record.id for record in selected] == [xmf.id, payload.id]


def test_indexed_catalog_selection_expands_xmf_dependency(tmp_path):
    payload = _record(
        tmp_path,
        id="visualization-data:pressure",
        role="visualization_data",
        path="vtk/pressure.h5",
        bytes=4,
        representation="hdf5",
    )
    xmf = _record(
        tmp_path,
        id="visualization:pressure:1",
        role="visualization",
        path="vtk/pressure.xmf",
        bytes=3,
        representation="xmf",
        dependencies=(payload.id,),
    )

    class IndexedCatalog(_Catalog):
        tasks = {1: object()}

        def query(self, *, task=None, **filters):
            assert task in (None, 1)
            return super().query(**filters)

    selected = select_transfer_artifacts(
        IndexedCatalog((payload, xmf)),
        requests=(ArtifactRequest(role="visualization"),),
        include_defaults=False,
    )

    assert [record.id for record in selected] == [xmf.id, payload.id]


def test_collection_parts_are_manifest_relative_and_relocation_stable(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "relocated"
    relative = "images/generation-7/manifest.json"
    manifest = {
        "attributes": {"schema": "fs-sharded-array-1"},
        "storage": {
            "part_count": 2,
            "metadata": {"path": "metadata/layout.h5", "bytes": 7},
            "parts": [
                {"path": "parts/part-00000.h5", "bytes": 3},
                {"path": "parts/part-00001.h5", "bytes": 5},
            ],
        },
    }
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    path = first / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)
    record = _record(
        first,
        id="image:gradient",
        role="image",
        path=relative,
        bytes=len(encoded),
        representation="collection_manifest",
        schema="fs-sharded-array-1",
    )

    first_parts = collection_parts(record)
    relocated_path = second / relative
    relocated_path.parent.mkdir(parents=True)
    relocated_path.write_bytes(encoded)
    relocated = _record(
        second,
        id="image:gradient",
        role="image",
        path=relative,
        bytes=len(encoded),
        representation="collection_manifest",
        schema="fs-sharded-array-1",
    )

    assert [part.relative_path for part in first_parts] == [
        "images/generation-7/metadata/layout.h5",
        "images/generation-7/parts/part-00000.h5",
        "images/generation-7/parts/part-00001.h5",
    ]
    assert [part.relative_path for part in collection_parts(relocated)] == [
        part.relative_path for part in first_parts
    ]
    assert all(
        str(part.path).startswith(str(second)) for part in collection_parts(relocated)
    )


def test_collection_manifest_rejects_escaping_part(tmp_path):
    manifest = {
        "attributes": {"schema": "fs-sharded-array-1"},
        "storage": {
            "part_count": 1,
            "parts": [{"path": "../escape.h5", "bytes": 0}],
        },
    }
    encoded = json.dumps(manifest).encode()
    path = tmp_path / "collection/manifest.json"
    path.parent.mkdir()
    path.write_bytes(encoded)
    record = _record(
        tmp_path,
        id="image",
        role="image",
        path="collection/manifest.json",
        bytes=len(encoded),
        representation="collection_manifest",
        schema="fs-sharded-array-1",
    )

    with pytest.raises(ArtifactContractError, match="relative and normalized"):
        collection_parts(record)


def test_collection_manifest_rejects_nul_part(tmp_path):
    manifest = {
        "attributes": {"schema": "fs-sharded-array-1"},
        "storage": {
            "part_count": 1,
            "parts": [{"path": "part\0hidden.h5", "bytes": 0}],
        },
    }
    encoded = json.dumps(manifest).encode()
    path = tmp_path / "collection/manifest.json"
    path.parent.mkdir()
    path.write_bytes(encoded)
    record = _record(
        tmp_path,
        id="image",
        role="image",
        path="collection/manifest.json",
        bytes=len(encoded),
        representation="collection_manifest",
        schema="fs-sharded-array-1",
    )

    with pytest.raises(ArtifactContractError, match="portable"):
        collection_parts(record)


def test_rsync_exact_transfer_uses_null_files_from(monkeypatch, tmp_path):
    calls = []
    listed = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        option = next(value for value in argv if value.startswith("--files-from="))
        listed.extend(Path(option.split("=", 1)[1]).read_bytes().split(b"\0")[:-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    site = SimpleNamespace(
        _login_client=SimpleNamespace(
            is_proxy=lambda: True,
            get_proxy_details=lambda: ("/tmp/control.sock", "user"),
        ),
        transfer_method="rsync",
        credentials=SimpleNamespace(username="user"),
        config=SimpleNamespace(hostname="login.example.edu"),
        verbose=False,
    )
    manager = SlurmTransferManager(site)
    monkeypatch.setattr(manager, "_local_tmp_parent", lambda: tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)

    manager.get_files(
        "/remote/results",
        tmp_path / "local",
        ("traces/a.h5", "images/manifest.json"),
        missing_ok=True,
    )

    argv, _ = calls[0]
    assert "--relative" in argv
    assert "--from0" in argv
    assert "--ignore-missing-args" in argv
    assert listed == [b"traces/a.h5", b"images/manifest.json"]
    assert argv[-2] == "user@login.example.edu:/remote/results/"


@pytest.mark.parametrize(
    ("exit_status", "stderr_text", "fails"),
    [(None, b"", False), (0, b"warning", False), (3, b"", True)],
)
def test_sftp_required_set_uses_only_explicit_tar_members(
    tmp_path,
    exit_status,
    stderr_text,
    fails,
):
    commands = []
    file_list = b""

    class FakeSFTP:
        def put(self, local, remote):
            nonlocal file_list
            file_list = Path(local).read_bytes()

        def get(self, remote, local):
            paths = [item.decode() for item in file_list.split(b"\0") if item]
            with tarfile.open(local, "w:gz") as archive:
                for path in paths:
                    payload = b"data"
                    info = tarfile.TarInfo(path)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

        def close(self):
            pass

    stdout = SimpleNamespace()
    if exit_status is not None:
        stdout.channel = SimpleNamespace(recv_exit_status=lambda: exit_status)
    site = SimpleNamespace(
        _login_client=SimpleNamespace(is_proxy=lambda: False),
        login_client=SimpleNamespace(open_sftp=lambda: FakeSFTP()),
        transfer_method="sftp",
        remote_tmp_dir=Path("/remote/tmp"),
        _site_config_path=None,
        run_login=lambda command: commands.append(command) or "",
        run_login_cmd=lambda command: commands.append(command)
        or (None, stdout, SimpleNamespace(read=lambda: stderr_text)),
    )
    manager = SlurmTransferManager(site)
    manager._local_tmp_parent = lambda: tmp_path / "scratch"
    (tmp_path / "scratch").mkdir()

    if fails:
        with pytest.raises(RuntimeError, match="status 3"):
            manager.get_files(
                "/remote/results",
                tmp_path / "local",
                ("traces/a.h5", "images/b.h5"),
            )
        return

    fetched = manager.get_files(
        "/remote/results",
        tmp_path / "local",
        ("traces/a.h5", "images/b.h5"),
    )

    assert [path.relative_to(tmp_path / "local").as_posix() for path in fetched] == [
        "traces/a.h5",
        "images/b.h5",
    ]
    tar_command = next(command for command in commands if command.startswith("tar "))
    assert "--null" in tar_command
    assert "--verbatim-files-from" in tar_command
    assert "/remote/results" in tar_command


@pytest.mark.parametrize("member_name", ["../escape.h5", "./traces/a.h5"])
def test_sftp_rejects_unexpected_or_noncanonical_tar_member(tmp_path, member_name):
    class FakeSFTP:
        def put(self, local, remote):
            pass

        def get(self, remote, local):
            with tarfile.open(local, "w:gz") as archive:
                payload = b"escape"
                info = tarfile.TarInfo(member_name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

        def close(self):
            pass

    site = SimpleNamespace(
        _login_client=SimpleNamespace(is_proxy=lambda: False),
        login_client=SimpleNamespace(open_sftp=lambda: FakeSFTP()),
        transfer_method="sftp",
        remote_tmp_dir=Path("/remote/tmp"),
        _site_config_path=None,
        run_login=lambda command: "",
        run_login_cmd=lambda command: (
            None,
            SimpleNamespace(),
            SimpleNamespace(read=lambda: b""),
        ),
    )
    manager = SlurmTransferManager(site)
    manager._local_tmp_parent = lambda: tmp_path / "scratch"
    (tmp_path / "scratch").mkdir()

    with pytest.raises(RuntimeError, match="unexpected"):
        manager.get_files(
            "/remote/results",
            tmp_path / "local",
            ("traces/a.h5",),
        )

    assert not (tmp_path / "escape.h5").exists()


def test_site_fetches_manifest_first_then_only_default_artifact_closure(tmp_path):
    result_path = tmp_path / "results"
    manifest = {
        "attributes": {"schema": "fs-sharded-array-1"},
        "storage": {
            "part_count": 1,
            "metadata": {"path": "metadata/layout.h5", "bytes": 6},
            "parts": [{"path": "parts/part-00000.h5", "bytes": 4}],
        },
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    payloads = {
        "traces/data.h5": b"trace",
        "images/run/manifest.json": manifest_bytes,
        "images/run/metadata/layout.h5": b"layout",
        "images/run/parts/part-00000.h5": b"part",
        "fields/forward.h5": b"wave",
    }
    records = [
        _record(
            result_path,
            id="traces",
            role="simulated_traces",
            path="traces/data.h5",
            bytes=5,
        ),
        _record(
            result_path,
            id="image",
            role="image",
            path="images/run/manifest.json",
            bytes=len(manifest_bytes),
            representation="collection_manifest",
            schema="fs-sharded-array-1",
        ),
        _record(
            result_path,
            id="forward-wavefield",
            role="wavefield",
            path="fields/forward.h5",
            bytes=4,
        ),
    ]
    calls = []

    class Transfer:
        def get_files(self, remote, local, paths, **kwargs):
            del remote, kwargs
            paths = tuple(paths)
            calls.append(paths)
            for relative in paths:
                target = Path(local) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payloads[relative])

    class Site:
        fetch_artifacts = SlurmSite.fetch_artifacts

        def __init__(self):
            self._transfer = Transfer()

        def fetch_artifact_catalog(self, job):
            calls.append(("catalog",))
            return _Catalog(records)

        def _remote_result_dir(self, job):
            return Path("/remote/results")

    job = SimpleNamespace(_result_path=result_path)
    fetched = Site().fetch_artifacts(job)

    assert calls == [
        ("catalog",),
        ("images/run/manifest.json",),
        (
            "traces/data.h5",
            "images/run/metadata/layout.h5",
            "images/run/parts/part-00000.h5",
        ),
    ]
    assert {path.relative_to(result_path).as_posix() for path in fetched} == {
        "traces/data.h5",
        "images/run/manifest.json",
        "images/run/metadata/layout.h5",
        "images/run/parts/part-00000.h5",
    }
    assert not (result_path / "fields/forward.h5").exists()


@pytest.mark.parametrize(
    ("metadata_payload", "error"),
    [(None, FileNotFoundError), (b"x", ArtifactContractError)],
)
def test_collection_metadata_missing_or_size_mismatch_is_rejected(
    tmp_path,
    metadata_payload,
    error,
):
    manifest = {
        "attributes": {"schema": "fs-sharded-array-1"},
        "storage": {
            "part_count": 1,
            "metadata": {"path": "metadata/layout.h5", "bytes": 6},
            "parts": [{"path": "parts/part.h5", "bytes": 4}],
        },
    }
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    record = _record(
        tmp_path,
        id="wavefield",
        role="wavefield",
        path="fields/manifest.json",
        bytes=len(encoded),
        representation="collection_manifest",
        schema="fs-sharded-array-1",
    )

    def fetch_files(paths):
        for relative in paths:
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith("manifest.json"):
                target.write_bytes(encoded)
            elif relative.endswith("layout.h5"):
                if metadata_payload is not None:
                    target.write_bytes(metadata_payload)
            else:
                target.write_bytes(b"part")

    with pytest.raises(error):
        fetch_artifact_payloads(
            _Catalog((record,)),
            fetch_files=fetch_files,
            requests=(ArtifactRequest(role="wavefield"),),
            include_defaults=False,
        )


def test_site_fetch_image_transfers_no_traces_or_wavefields(monkeypatch, tmp_path):
    result_path = tmp_path / "results"
    payloads = {
        "traces/data.h5": b"trace",
        "images/gradient.h5": b"image",
        "fields/forward.h5": b"wave",
    }
    records = [
        _record(
            result_path,
            id="traces",
            role="simulated_traces",
            path="traces/data.h5",
            bytes=5,
        ),
        _record(
            result_path,
            id="gradient",
            role="image",
            path="images/gradient.h5",
            bytes=5,
        ),
        _record(
            result_path,
            id="forward-wavefield",
            role="wavefield",
            path="fields/forward.h5",
            bytes=4,
        ),
    ]
    calls = []

    class Transfer:
        def get_files(self, remote, local, paths, **kwargs):
            del remote, kwargs
            calls.extend(paths)
            for relative in paths:
                target = Path(local) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payloads[relative])

    class Site:
        fetch_artifacts = SlurmSite.fetch_artifacts
        fetch_image = SlurmSite.fetch_image

        def __init__(self):
            self._transfer = Transfer()

        def fetch_artifact_catalog(self, job, *, operations=()):
            assert operations == ("smooth",)
            return _Catalog(records)

        def _remote_result_dir(self, job):
            return Path("/remote/results")

    job = SimpleNamespace(
        name="image-job",
        _result_path=result_path,
        load_images=lambda: "loaded-image",
    )
    monkeypatch.setattr(
        "frequensolve.orchestrator.sites.hpc.site._as_list",
        lambda value, expected: ([value], True),
    )

    image = Site().fetch_image(job)

    assert image == "loaded-image"
    assert calls == ["images/gradient.h5"]
    assert not (result_path / "traces/data.h5").exists()
    assert not (result_path / "fields/forward.h5").exists()


def test_site_catalog_fetch_uses_only_index_then_fixed_task_results(tmp_path):
    result_path = tmp_path / "results"
    fingerprint = "sha256:" + "a" * 64
    stale_index = _write_stale_empty_index(result_path, fingerprint)
    assert stale_index.is_file()
    task_result = json.dumps(
        {
            "schema": "fs-task-result-2",
            "partition": {
                "task": 1,
                "task_count": 1,
                "frequency": {"real": 2.0, "imag": 0.1},
            },
            "fingerprints": {
                "job": fingerprint,
                "simulation": fingerprint,
                "outputs": fingerprint,
            },
            "status": {"state": "success", "code": 0},
            "artifacts": [],
        }
    ).encode()
    calls = []

    class Transfer:
        def get_files(self, remote, local, paths, **kwargs):
            del remote
            paths = tuple(paths)
            calls.append((paths, kwargs))
            for relative in paths:
                if relative.endswith("result.json"):
                    target = Path(local) / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(task_result)

    class Site:
        fetch_artifact_catalog = SlurmSite.fetch_artifact_catalog
        _validate_remote_catalog = SlurmSite._validate_remote_catalog

        def __init__(self):
            self._transfer = Transfer()

        def _remote_result_dir(self, job):
            return Path("/remote/results")

    job = SimpleNamespace(
        n_tasks=1,
        f_list=(complex(2.0, 0.1),),
        _result_path=result_path,
        staged_artifact_fingerprints=lambda site: {
            "job": fingerprint,
            "simulation": fingerprint,
            "outputs": fingerprint,
        },
    )

    catalog = Site().fetch_artifact_catalog(job)

    assert isinstance(catalog, ArtifactCatalog)
    assert tuple(catalog.results) == (1,)
    assert calls == [
        (("_fs_run/tasks.h5",), {"missing_ok": True}),
        (
            ("_fs_run/tasks/task_000001/result.json",),
            {"missing_ok": True},
        ),
    ]
    assert (result_path / "_fs_run/tasks/task_000001/result.json").is_file()
    assert not stale_index.exists()
    assert not (result_path / "_fs_run/run_manifest.json").exists()


def test_site_catalog_fetches_only_explicit_fixed_operation_results(tmp_path):
    result_path = tmp_path / "results"
    fingerprint = "sha256:" + "a" * 64
    task_result = {
        "schema": "fs-task-result-2",
        "partition": {
            "task": 1,
            "frequency": {"real": 2.0, "imag": 0.1},
        },
        "fingerprints": {
            "job": fingerprint,
            "simulation": fingerprint,
            "outputs": fingerprint,
        },
        "status": {"state": "success", "code": 0},
        "artifacts": [],
    }
    operation_result = {
        "schema": "fs-operation-result-1",
        "operation": {"name": "smooth", "generation": "smooth-1"},
        "fingerprints": task_result["fingerprints"],
        "status": {"state": "success", "code": 0},
        "artifacts": [],
    }
    remote = {
        "_fs_run/tasks/task_000001/result.json": json.dumps(task_result).encode(),
        "_fs_run/operations/smooth/result.json": json.dumps(operation_result).encode(),
    }
    calls = []

    class Transfer:
        def get_files(self, remote_root, local, paths, **kwargs):
            del remote_root
            paths = tuple(paths)
            calls.append(paths)
            for relative in paths:
                payload = remote.get(relative)
                if payload is None:
                    continue
                target = Path(local) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)

    class Site:
        fetch_artifact_catalog = SlurmSite.fetch_artifact_catalog
        _validate_remote_catalog = SlurmSite._validate_remote_catalog
        _validate_remote_operation = SlurmSite._validate_remote_operation

        def __init__(self):
            self._transfer = Transfer()

        def _remote_result_dir(self, job):
            return Path("/remote/results")

    job = SimpleNamespace(
        n_tasks=1,
        f_list=(complex(2.0, 0.1),),
        _result_path=result_path,
        staged_task_fingerprints=lambda site: task_result["fingerprints"],
        staged_artifact_fingerprints=lambda site: task_result["fingerprints"],
    )

    catalog = Site().fetch_artifact_catalog(job, operations=("smooth",))

    assert set(catalog.operations) == {"smooth"}
    assert calls == [
        ("_fs_run/tasks.h5",),
        ("_fs_run/tasks/task_000001/result.json",),
        ("_fs_run/operations/smooth/result.json",),
    ]
    assert (
        result_path / "_fs_run/operations/smooth/result.json"
    ).read_bytes() == remote["_fs_run/operations/smooth/result.json"]
