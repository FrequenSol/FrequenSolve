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
