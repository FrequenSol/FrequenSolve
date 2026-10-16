import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.execution_plan import PlanError, digest
from frequensolve.orchestrator.sites.aws.plan_results import ExecutionResults
from frequensolve.orchestrator.sites.aws.plans import PlanHandle


def fixture():
    project, run, plan = "fake-project", "fake-run", "fake-plan"
    prefix, bucket = f"{project}/results/runs/{run}/", "fake-result-bucket"
    payloads, records = {}, []
    identity = {
        "execution_id": run,
        "frequency_index": 0,
        "plan_digest": digest("fake-plan"),
    }
    for name, body in [
        ("ParaView/pv_00000.vtu", b"fake geometry"),
        ("traces/data.h5", b"fake traces"),
        ("empty.bin", b""),
    ]:
        checksum = "sha256:" + hashlib.sha256(body).hexdigest()
        key = prefix + "frequency-results/frequency-000000/objects/" + checksum[7:]
        version = "fake-object-version-" + str(len(records))
        payloads[key, version] = body
        records.append(
            {
                "path": name,
                "size": len(body),
                "digest": checksum,
                "kind": "fake-native",
                "schema": None,
                "object": {"key": key, "version_id": version},
            }
        )
    snapshot = digest("fake-snapshot")
    manifest = {
        "version": "frequency-output-archive-manifest.v1",
        "identity": identity,
        "snapshot_digest": snapshot,
        "total_bytes": sum(item["size"] for item in records),
        "files": records,
    }
    manifest["digest"] = digest(manifest)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    key = (
        prefix
        + "frequency-results/frequency-000000/manifests/"
        + snapshot[7:]
        + ".json"
    )
    ref = {
        "bucket": bucket,
        "key": key,
        "version_id": "fake-manifest-version",
        "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
    payloads[key, ref["version_id"]] = raw
    archive = {
        "version": "frequency-output-archive.v1",
        "identity": identity,
        "snapshot_digest": snapshot,
        "total_bytes": manifest["total_bytes"],
        "file_count": 3,
        "manifest": ref,
    }
    value = {
        "version": "frequency-execution-results.v1",
        "plan_id": plan,
        "plan_revision": 1,
        "execution_id": run,
        "plan_digest": identity["plan_digest"],
        "representation": "per-frequency-native-archives.v1",
        "combined_packed_product": None,
        "frequencies": [
            {"frequency_index": 0, "archive": archive},
            {"frequency_index": 1, "archive": None},
        ],
    }
    value["digest"] = digest(value)
    response = {
        "head": {
            "plan_id": plan,
            "revision": 1,
            "execution_id": run,
            "status": "executed",
            "owner": {"project_key": project},
        },
        "results": value,
        "result_location": {"bucket": bucket, "result_s3_prefix": prefix},
    }
    requests = []

    def get(**request):
        requests.append(request)
        assert request["Bucket"] == bucket
        raw = payloads[request["Key"], request["VersionId"]]
        return {
            "Body": io.BytesIO(raw),
            "ContentLength": len(raw),
            "VersionId": request["VersionId"],
        }

    site = SimpleNamespace(
        config=SimpleNamespace(s3_bucket=bucket),
        s3_client=SimpleNamespace(get_object=get),
    )

    def read():
        return ExecutionResults(
            site, response, plan_id=plan, execution_id=run, project=project
        )

    return SimpleNamespace(
        site=site,
        response=response,
        payloads=payloads,
        requests=requests,
        read=read,
        manifest=manifest,
    )


def test_exact_versioned_download_preserves_native_frequency_namespaces_and_empty_files(
    tmp_path,
):
    f = fixture()
    result = f.read()
    exposed = result.data
    exposed["frequencies"].clear()
    root = result.download(tmp_path)
    assert root.name == "fake-run"
    assert (
        root / "frequency-000000/files/ParaView/pv_00000.vtu"
    ).read_bytes() == b"fake geometry"
    assert (root / "frequency-000000/files/empty.bin").read_bytes() == b""
    assert not (root / "frequency-000001").exists()
    assert (
        len(json.loads((root / "execution-results.json").read_text())["frequencies"])
        == 2
    )
    assert all(call["VersionId"] for call in f.requests)
    with pytest.raises(FileExistsError):
        result.download(tmp_path)
    assert len(f.requests) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_id", "other-run"),
        ("plan_id", "other-plan"),
        ("plan_revision", 2),
        ("digest", digest("tampered")),
    ],
)
def test_mismatched_cloud_result_identity_is_rejected_before_storage(field, value):
    f = fixture()
    f.response["results"][field] = value
    with pytest.raises(PlanError):
        f.read()
    assert f.requests == []


@pytest.mark.parametrize(
    "kind", ["foreign", "traversal", "unversioned", "changed-bytes", "too-large"]
)
def test_changed_or_foreign_archive_never_publishes_a_local_result_tree(tmp_path, kind):
    f = fixture()
    manifest = deepcopy(f.manifest)
    if kind == "foreign":
        manifest["files"][0]["object"]["key"] = "other-user/secret"
    elif kind == "traversal":
        manifest["files"][0]["path"] = "../../secret"
    elif kind == "unversioned":
        manifest["files"][0]["object"]["version_id"] = "null"
    elif kind == "too-large":
        manifest["files"][0]["size"] = 6 * 1024**3
    else:
        record = manifest["files"][0]
        f.payloads[record["object"]["key"], record["object"]["version_id"]] = (
            b"wrong content"
        )
    manifest["digest"] = digest(
        {key: value for key, value in manifest.items() if key != "digest"}
    )
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ref = f.response["results"]["frequencies"][0]["archive"]["manifest"]
    ref["digest"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    f.payloads[ref["key"], ref["version_id"]] = raw
    f.response["results"]["digest"] = digest(
        {key: value for key, value in f.response["results"].items() if key != "digest"}
    )
    with pytest.raises(PlanError):
        f.read().download(tmp_path)
    assert not (tmp_path / "fake-run").exists()
    assert not list(tmp_path.glob(".fs-result-download-*"))


def test_result_inspection_never_changes_current_plan_or_authorizes_compute():
    f = fixture()
    requests = []

    class Client:
        def execute(self, _query, values):
            requests.append(json.loads(values["input"]))
            return {"requestMyExecutionPlan": f.response}

    f.site.graphql_client = Client()
    current = {
        "head": {
            "plan_id": "fake-plan",
            "revision": 2,
            "status": "executing",
            "execution_id": "new-run",
            "owner": {"project_key": "fake-project"},
        }
    }
    plan = PlanHandle(f.site, deepcopy(current))
    result = plan.results(execution_id="fake-run")
    assert result.data["execution_id"] == "fake-run"
    assert plan.data == current
    assert requests == [
        {
            "action": "results",
            "project_key": "fake-project",
            "plan_id": "fake-plan",
            "execution_id": "fake-run",
        }
    ]


def test_current_runtime_result_fixture_is_readable_without_launching_or_reading_storage():
    value = json.loads(
        (
            Path(__file__).parent / "fixtures/execution-plan/v1/execution-results.json"
        ).read_text()
    )
    command, index = value["command"], value["index"]
    site = SimpleNamespace(
        config=SimpleNamespace(s3_bucket=command["destination"]["bucket"])
    )
    response = {
        "head": {
            "plan_id": command["plan_id"],
            "revision": command["revision"],
            "status": "executed",
            "execution_id": command["execution_id"],
            "owner": {"project_key": "fake-project"},
        },
        "result_location": command["destination"],
        "results": index,
    }
    result = ExecutionResults(
        site,
        response,
        plan_id=command["plan_id"],
        execution_id=command["execution_id"],
        project="fake-project",
    )
    assert result.data == index
    assert [row["actual"]["nodes"] for row in result.data["frequencies"]] == [2, 2]
    assert all(
        row["actual"]["measured_memory"] is None for row in result.data["frequencies"]
    )


def scientific_fixture(tmp_path, *, payload_task=1, payload_frequency=10.0):
    import h5py
    import numpy as np

    f = fixture()
    source = tmp_path / "native-source"
    source.mkdir()
    metadata = source / "trace_metadata.h5"
    shard = source / "payload.h5"
    for path, metadata_only in ((metadata, True), (shard, False)):
        with h5py.File(path, "w") as data:
            values = data.create_dataset(
                "surface", data=np.array([[[1.0]], [[2.0]]], dtype=np.float32)
            )
            values.attrs["dims"] = ["receiver", "component", "shot"]
            values.attrs["receiver"] = [1, 2]
            values.attrs["component"] = np.array(["p"], dtype="S")
            values.attrs["shot"] = [1]
            if metadata_only:
                data.create_dataset(
                    "metadata/schema_version",
                    data=np.array(["fs_trace_metadata_v1"], dtype="S"),
                )
            else:
                data.create_dataset("frequency", data=payload_frequency)
                data.create_dataset("laplace", data=-0.25)
                data.create_dataset("task_id", data=payload_task)
    artifacts = []
    for path, relative, role, representation, schema in (
        (
            metadata,
            "traces/trace_metadata.h5",
            "traces_metadata",
            "file",
            "fs_trace_metadata_v1",
        ),
        (
            shard,
            "traces/shards/payload.h5",
            "simulated_traces",
            "hdf5_shard",
            "fs_trace_payload_shard_v1",
        ),
    ):
        artifacts.append(
            {
                "id": role,
                "role": role,
                "representation": representation,
                "schema": schema,
                "path": relative,
                "retention": "durable",
                "bytes": path.stat().st_size,
            }
        )
    task = source / "_fs_run/tasks/task_000001/result.json"
    task.parent.mkdir(parents=True)
    task.write_text(
        json.dumps(
            {
                "schema": "fs-task-result-2",
                "partition": {"task": 1, "frequency": {"real": 10.0, "imag": -0.25}},
                "fingerprints": {
                    "job": "0" * 64,
                    "simulation": "0" * 64,
                    "outputs": "0" * 64,
                },
                "status": {"state": "success", "code": 0},
                "artifacts": artifacts,
            }
        )
    )
    records = []
    prefix = (
        f.response["result_location"]["result_s3_prefix"]
        + "frequency-results/frequency-000000/"
    )
    for path, relative, kind, schema in (
        (
            metadata,
            "traces/trace_metadata.h5",
            "traces_metadata",
            "fs_trace_metadata_v1",
        ),
        (shard, "traces/shards/payload.h5", "hdf5", "fs_trace_payload_shard_v1"),
        (task, "_fs_run/tasks/task_000001/result.json", "native_metadata", None),
    ):
        raw = path.read_bytes()
        checksum = "sha256:" + hashlib.sha256(raw).hexdigest()
        key = prefix + "objects/" + checksum[7:]
        f.payloads[key, "fake-native-version"] = raw
        records.append(
            {
                "path": relative,
                "size": len(raw),
                "digest": checksum,
                "kind": kind,
                "schema": schema,
                "object": {"key": key, "version_id": "fake-native-version"},
            }
        )
    row = f.response["results"]["frequencies"][0]
    row.update(scientific={"converged": True}, frequency_value=["10", "-0.25"])
    row["archive"]["identity"]["native_result_digest"] = records[2]["digest"]
    f.manifest.update(files=records, total_bytes=sum(item["size"] for item in records))
    f.manifest["digest"] = digest(
        {key: value for key, value in f.manifest.items() if key != "digest"}
    )
    raw = json.dumps(f.manifest, sort_keys=True, separators=(",", ":")).encode()
    row["archive"]["total_bytes"] = f.manifest["total_bytes"]
    row["archive"]["manifest"]["digest"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    reference = row["archive"]["manifest"]
    f.payloads[reference["key"], reference["version_id"]] = raw
    f.response["results"]["digest"] = digest(
        {key: value for key, value in f.response["results"].items() if key != "digest"}
    )
    return f


def test_native_trace_loading_pairs_exact_metadata_and_payload_without_another_planner(
    tmp_path,
):
    f = scientific_fixture(tmp_path)
    result = f.read()
    root = result.download(tmp_path / "downloads")
    calls = len(f.requests)
    catalog = result.artifact_catalog(root, frequency_index=0)
    assert catalog.results[1].partition.frequency == 10.0 - 0.25j
    traces = result.trace_dataset(root, frequency_index=0)
    assert traces.groups == ["surface"]
    assert traces.manifest.frequencies == {1: 10.0}
    assert traces.manifest.laplace == {1: -0.25}
    assert traces.files[0].endswith("frequency-000000/files/traces/shards/payload.h5")
    assert len(f.requests) == calls


def test_local_native_tampering_or_failed_frequency_cannot_be_presented_as_verified_traces(
    tmp_path,
):
    f = scientific_fixture(tmp_path)
    result = f.read()
    root = result.download(tmp_path / "downloads")
    with pytest.raises(PlanError, match="no retained"):
        result.trace_dataset(root, frequency_index=1)
    payload = root / "frequency-000000/files/traces/shards/payload.h5"
    payload.write_bytes(b"changed local native data")
    with pytest.raises(PlanError, match="bytes changed|size changed"):
        result.trace_dataset(root, frequency_index=0)
    (root / "execution-results.json").write_text("{}")
    with pytest.raises(PlanError, match="different saved execution"):
        result.artifact_catalog(root, frequency_index=0)


def test_nonconverged_native_traces_remain_inspectable_raw_but_not_verified_gathers(
    tmp_path,
):
    f = scientific_fixture(tmp_path)
    f.response["results"]["frequencies"][0]["scientific"]["converged"] = False
    f.response["results"]["digest"] = digest(
        {key: value for key, value in f.response["results"].items() if key != "digest"}
    )
    result = f.read()
    root = result.download(tmp_path / "downloads")
    assert result.artifact_catalog(root, frequency_index=0).results
    with pytest.raises(PlanError, match="no verified converged traces"):
        result.trace_dataset(root, frequency_index=0)


@pytest.mark.parametrize("change", [{"payload_task": 2}, {"payload_frequency": 30.0}])
def test_native_payload_header_cannot_select_another_task_or_frequency(
    tmp_path, change
):
    f = scientific_fixture(tmp_path, **change)
    result = f.read()
    root = result.download(tmp_path / "downloads")
    with pytest.raises(PlanError, match="saved frequency/task/survey"):
        result.trace_dataset(root, frequency_index=0)
