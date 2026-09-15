import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws.aws import AWSSite
from frequensolve.orchestrator.sites.base import JobStatus
from frequensolve.simulation.artifact_contract import ArtifactRecord, ArtifactRequest
from frequensolve.simulation.jobs import ImagingJob


class FakePaginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, Bucket, Prefix):
        self.client.paginate_calls.append({"Bucket": Bucket, "Prefix": Prefix})
        contents = [
            {"Key": key}
            for key in sorted(self.client.objects)
            if key.startswith(Prefix)
        ]
        return [{"Contents": contents}]


class FakeS3Client:
    def __init__(self, objects):
        self.objects = objects
        self.downloads = []
        self.head_calls = []
        self.paginate_calls = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self)

    def head_object(self, Bucket, Key):
        self.head_calls.append({"Bucket": Bucket, "Key": Key})
        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not found"}},
                "HeadObject",
            )
        return {}

    def download_file(self, bucket, key, filename):
        self.downloads.append({"Bucket": bucket, "Key": key, "Filename": filename})
        if key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not found"}},
                "GetObject",
            )
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text(self.objects[key])


def make_site(s3_client):
    site = AWSSite.__new__(AWSSite)
    site.s3_client = s3_client
    site.config = SimpleNamespace(s3_bucket="bucket")
    return site


def test_get_downloads_single_s3_object_without_forcing_prefix(tmp_path):
    s3_client = FakeS3Client({"path/to/file.json": "{}"})
    site = make_site(s3_client)

    site.get("s3://bucket/path/to/file.json", tmp_path / "downloads")

    assert (tmp_path / "downloads" / "file.json").read_text() == "{}"
    assert s3_client.downloads == [
        {
            "Bucket": "bucket",
            "Key": "path/to/file.json",
            "Filename": str(tmp_path / "downloads" / "file.json"),
        }
    ]
    assert s3_client.paginate_calls == []


def test_get_falls_back_to_prefix_download_when_exact_object_is_missing(tmp_path):
    s3_client = FakeS3Client(
        {
            "path/to/results/a.txt": "a",
            "path/to/results/nested/b.txt": "b",
        }
    )
    site = make_site(s3_client)

    site.get("s3://bucket/path/to/results", tmp_path / "downloads")

    assert (tmp_path / "downloads" / "a.txt").read_text() == "a"
    assert (tmp_path / "downloads" / "nested" / "b.txt").read_text() == "b"
    assert s3_client.paginate_calls == [
        {"Bucket": "bucket", "Prefix": "path/to/results/"}
    ]


def test_fetch_vtk_reraises_download_failures(tmp_path):
    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="bucket")

    def fail_fetch(*args, **kwargs):
        raise RuntimeError("download failed")

    site.fetch_artifacts = fail_fetch
    job = SimpleNamespace(
        project_path=tmp_path,
        name="job-a",
        simulation=SimpleNamespace(
            name="simulation-a",
            project_path=tmp_path / "project-a",
        ),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        site.fetch_vtk(job)


def test_fetch_image_downloads_only_the_aggregate_image(tmp_path):
    project = tmp_path / "project"
    image = project / "results/opaque-generation/image.h5"
    key = "project/results/opaque-generation/image.h5"
    client = FakeS3Client({key: "image payload", "project/results/unused.h5": "old"})
    site = make_site(client)
    job = object.__new__(ImagingJob)
    job.simulation = SimpleNamespace(project_path=project)
    job.save_path = image.parent
    job.load_images = lambda: "image reader"

    def fetch(job, **kwargs):
        assert kwargs["operations"] == ("smooth",)
        assert kwargs["requests"] == (ArtifactRequest(role="image"),)
        assert not kwargs["include_defaults"]
        return site._download_s3_files(
            "project/results", project / "results", ("opaque-generation/image.h5",)
        )

    site.fetch_artifacts = fetch
    assert site.fetch_image(job) == "image reader"
    assert image.read_text() == "image payload"
    assert [row["Key"] for row in client.downloads] == [key]
    assert not client.paginate_calls


def test_fetch_image_rejects_paths_outside_the_project(tmp_path):
    project_path = tmp_path / "imaging-project"
    site = make_site(FakeS3Client({}))
    site.config = SimpleNamespace(s3_bucket="bucket")

    job = object.__new__(ImagingJob)
    job.name = "rtm"
    job.simulation = SimpleNamespace(project_path=project_path, name="model")
    job.save_path = tmp_path / "other" / "imaging"

    with pytest.raises(ValueError, match="outside project root"):
        site.fetch_image(job)


def test_fetch_image_normalizes_missing_output_after_credential_refresh(tmp_path):
    project_path = tmp_path / "imaging-project"

    class ExpiredThenMissingS3Client:
        def __init__(self):
            self.attempts = 0

        def download_file(self, bucket, key, filename):
            self.attempts += 1
            code = "ExpiredToken" if self.attempts == 1 else "NoSuchKey"
            raise ClientError(
                {"Error": {"Code": code, "Message": code}},
                "GetObject",
            )

    s3_client = ExpiredThenMissingS3Client()
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site.cognito_auth = object()
    site._refresh_s3_credentials = lambda: None

    job = object.__new__(ImagingJob)
    job.name = "rtm"
    job.simulation = SimpleNamespace(project_path=project_path, name="model")
    job.save_path = project_path / "jobs" / "model" / "rtm" / "results" / "imaging"

    site.fetch_artifacts = lambda *_args, **_kwargs: site._download_s3_files(
        "project/results", tmp_path / "downloads", ("opaque/image.h5",)
    )
    with pytest.raises(FileNotFoundError, match="S3 artifact is missing"):
        site.fetch_image(job)

    assert s3_client.attempts == 2


def test_fetch_run_metadata_downloads_job_run_directory(tmp_path):
    site = make_site(FakeS3Client({}))
    job = SimpleNamespace(_result_path=tmp_path / "results")
    calls = []
    site.fetch_artifact_catalog = lambda item: calls.append(item)
    assert site.fetch_run_metadata(job) == job._result_path / "_fs_run"
    assert calls == [job]


def test_fetch_outputs_downloads_complete_configured_artifact_set():
    site = object.__new__(AWSSite)
    calls = []
    site.fetch_artifacts = lambda job, **kwargs: calls.append(kwargs)
    job = SimpleNamespace(
        outputs=SimpleNamespace(wavefields=[object()]),
        traces=SimpleNamespace(open=lambda: "trace-data"),
        wavefields=SimpleNamespace(open=lambda: "wave-data"),
    )
    assert site.fetch_outputs(job) == {
        "traces": "trace-data",
        "wavefields": "wave-data",
    }
    assert calls == [{"requests": (), "include_defaults": True, "operations": ()}]


def test_fetch_outputs_downloads_aggregate_image_for_imaging_job(monkeypatch):
    site = object.__new__(AWSSite)
    calls = []
    site.fetch_artifacts = lambda job, **kwargs: calls.append(kwargs)
    job = object.__new__(ImagingJob)
    job.outputs = SimpleNamespace(wavefields=[])
    job.load_images = lambda: calls.append("image")
    monkeypatch.setattr(
        ImagingJob,
        "traces",
        property(lambda self: SimpleNamespace(open=lambda: "trace-data")),
    )
    assert site.fetch_outputs(job) == "trace-data"
    assert calls == [
        {"requests": (), "include_defaults": True, "operations": ("smooth",)},
        "image",
    ]


def test_aws_run_handle_honors_submit_time_fetch_after_success():
    site = AWSSite.__new__(AWSSite)
    fetch_calls = []
    site._poll_run = lambda run: JobStatus(
        state="completed",
        return_code=0,
        job_id=str(run.id),
    )
    site.fetch_outputs = lambda job: fetch_calls.append(job)
    site._emit_status = lambda *args, **kwargs: None
    job = SimpleNamespace(
        name="job-a",
        trace_manifest=None,
        _stdout_path=None,
        run_metadata=None,
    )
    run = site._make_run_handle(
        job,
        "simulation-1",
        poll_interval=0.0,
        fetch=True,
        check=True,
    )

    result = run.wait()

    assert result.successful
    assert fetch_calls == [job]


def test_fetch_vtk_downloads_only_configured_output_paths(tmp_path):
    site = make_site(FakeS3Client({"project/results/pv/opaque.vtu": "mesh"}))
    calls = []

    def fetch(job, **kwargs):
        calls.append(kwargs)
        return site._download_s3_files("project/results", tmp_path, ("pv/opaque.vtu",))

    site.fetch_artifacts = fetch
    site.fetch_vtk(SimpleNamespace())
    assert (tmp_path / "pv/opaque.vtu").read_text() == "mesh"
    assert {request.role for request in calls[0]["requests"]} == {
        "visualization",
        "visualization_data",
    }
    assert not site.s3_client.paginate_calls


def test_exact_s3_transfer_downloads_direct_keys_without_listing(tmp_path):
    client = FakeS3Client(
        {
            "project/jobs/simulation/job/results/traces/data.h5": "trace",
            "project/jobs/simulation/job/results/fields/forward.h5": "wave",
        }
    )
    site = make_site(client)

    fetched = site._download_s3_files(
        "project/jobs/simulation/job/results",
        tmp_path,
        ("traces/data.h5",),
    )

    assert fetched == [tmp_path / "traces/data.h5"]
    assert [item["Key"] for item in client.downloads] == [
        "project/jobs/simulation/job/results/traces/data.h5"
    ]
    assert client.paginate_calls == []


def test_exact_s3_artifact_fetch_excludes_unrelated_wavefields(tmp_path):
    result_path = tmp_path / "project/jobs/simulation/job/results"
    records = (
        ArtifactRecord.from_fs(
            {
                "id": "traces",
                "role": "simulated_traces",
                "representation": "hdf5_shard",
                "schema": "trace-1",
                "path": "traces/data.h5",
                "retention": "durable",
                "bytes": 5,
            },
            result_path=result_path,
        ),
        ArtifactRecord.from_fs(
            {
                "id": "forward-wavefield",
                "role": "wavefield",
                "representation": "hdf5",
                "schema": "field-1",
                "path": "fields/forward.h5",
                "retention": "durable",
                "bytes": 4,
            },
            result_path=result_path,
        ),
    )

    class Catalog:
        def query(self, **filters):
            return [
                record
                for record in records
                if all(
                    value is None or getattr(record, name) == value
                    for name, value in filters.items()
                )
            ]

        def select(self, request, *, task=None):
            del task
            return self.query(
                id=request.id,
                role=request.role,
                retention=request.retention,
            )

    client = FakeS3Client(
        {
            "project/jobs/simulation/job/results/traces/data.h5": "trace",
            "project/jobs/simulation/job/results/fields/forward.h5": "wave",
        }
    )
    site = make_site(client)
    site.fetch_artifact_catalog = lambda job, project_path=None: Catalog()
    job = SimpleNamespace(
        project_path=tmp_path / "project",
        _result_path=result_path,
        name="job",
        simulation=SimpleNamespace(name="simulation"),
    )

    fetched = site.fetch_artifacts(job)

    assert fetched == [result_path / "traces/data.h5"]
    assert [item["Key"] for item in client.downloads] == [
        "project/jobs/simulation/job/results/traces/data.h5"
    ]
    assert client.paginate_calls == []


def test_s3_json_catalog_fallback_removes_stale_local_index(tmp_path):
    fingerprint = "sha256:" + "a" * 64
    result_path = tmp_path / "project/jobs/simulation/job/results"
    stale_index = result_path / "_fs_run/tasks.h5"
    stale_index.parent.mkdir(parents=True)
    stale_index.write_bytes(b"stale-index")
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
    )
    result_key = (
        "project/jobs/simulation/job/results/" "_fs_run/tasks/task_000001/result.json"
    )
    client = FakeS3Client({result_key: task_result})
    site = make_site(client)
    job = SimpleNamespace(
        n_tasks=1,
        f_list=(complex(2.0, 0.1),),
        project_path=tmp_path / "project",
        _result_path=result_path,
        name="job",
        simulation=SimpleNamespace(name="simulation"),
        staged_artifact_fingerprints=lambda site_name: {
            "job": fingerprint,
            "simulation": fingerprint,
            "outputs": fingerprint,
        },
    )

    catalog = site.fetch_artifact_catalog(job)

    assert tuple(catalog.results) == (1,)
    assert not stale_index.exists()
    assert client.paginate_calls == []


def test_s3_fetches_explicit_operation_catalog_and_payload_without_listing(tmp_path):
    fingerprint = "sha256:" + "a" * 64
    result_path = tmp_path / "project/jobs/simulation/job/results"
    fingerprints = {
        "job": fingerprint,
        "simulation": fingerprint,
        "outputs": fingerprint,
    }
    task_result = {
        "schema": "fs-task-result-2",
        "partition": {
            "task": 1,
            "frequency": {"real": 2.0, "imag": 0.1},
        },
        "fingerprints": fingerprints,
        "status": {"state": "success", "code": 0},
        "artifacts": [],
    }
    operation_result = {
        "schema": "fs-operation-result-1",
        "operation": {"name": "smooth", "generation": "smooth-1"},
        "fingerprints": fingerprints,
        "status": {"state": "success", "code": 0},
        "artifacts": [
            {
                "id": "smooth-image",
                "role": "image",
                "representation": "hdf5",
                "schema": "image-1",
                "path": "images/smooth.h5",
                "retention": "durable",
                "bytes": 5,
            }
        ],
    }
    prefix = "project/jobs/simulation/job/results"
    client = FakeS3Client(
        {
            f"{prefix}/_fs_run/tasks/task_000001/result.json": json.dumps(task_result),
            f"{prefix}/_fs_run/operations/smooth/result.json": json.dumps(
                operation_result
            ),
            f"{prefix}/images/smooth.h5": "image",
        }
    )
    site = make_site(client)
    job = SimpleNamespace(
        n_tasks=1,
        f_list=(complex(2.0, 0.1),),
        project_path=tmp_path / "project",
        _result_path=result_path,
        name="job",
        simulation=SimpleNamespace(name="simulation"),
        staged_artifact_fingerprints=lambda site_name: fingerprints,
    )

    fetched = site.fetch_artifacts(
        job,
        requests=(ArtifactRequest(role="image"),),
        include_defaults=False,
        operations=("smooth",),
    )

    assert fetched == [result_path / "images/smooth.h5"]
    assert result_path.joinpath("images/smooth.h5").read_text() == "image"
    assert client.paginate_calls == []
    assert [item["Key"] for item in client.downloads] == [
        f"{prefix}/_fs_run/tasks.h5",
        f"{prefix}/_fs_run/tasks/task_000001/result.json",
        f"{prefix}/_fs_run/operations/smooth/result.json",
        f"{prefix}/images/smooth.h5",
    ]
