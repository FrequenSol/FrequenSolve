from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws import aws as aws_module
from frequensolve.orchestrator.sites.aws.aws import AWSSite


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
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text(self.objects[key])


def make_site(s3_client):
    site = AWSSite.__new__(AWSSite)
    site.s3_client = s3_client
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


def test_fetch_paraview_reraises_download_failures(tmp_path):
    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="bucket")

    def fail_get(*args, **kwargs):
        raise RuntimeError("download failed")

    site.get = fail_get
    job = SimpleNamespace(
        project_path=tmp_path,
        name="job-a",
        simulation=SimpleNamespace(
            name="simulation-a",
            _remote_path=Path("project-a/simulation-a"),
        ),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        site.fetch_paraview(job)


def test_fetch_paraview_uses_nested_project_result_prefix_when_canonical_empty(
    tmp_path,
):
    key = (
        "ex_1_1/ex_1_1/jobs/simple_acoustic/freq_10hz/results/ParaView/"
        "pv_fine_00000.vtu"
    )
    s3_client = FakeS3Client({key: "vtu"})
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    project_path = tmp_path / "ex_1_1"
    job = SimpleNamespace(
        project_path=project_path,
        name="freq_10hz",
        simulation=SimpleNamespace(
            name="simple_acoustic",
            _remote_path=Path("ex_1_1/simulations/simple_acoustic"),
        ),
    )

    site.fetch_paraview(job)

    assert (
        project_path
        / "jobs"
        / "simple_acoustic"
        / "freq_10hz"
        / "results"
        / "ParaView"
        / "pv_fine_00000.vtu"
    ).read_text() == "vtu"
    assert s3_client.paginate_calls == [
        {
            "Bucket": "bucket",
            "Prefix": "ex_1_1/jobs/simple_acoustic/freq_10hz/results/ParaView/",
        },
        {
            "Bucket": "bucket",
            "Prefix": (
                "ex_1_1/ex_1_1/jobs/simple_acoustic/freq_10hz/" "results/ParaView/"
            ),
        },
    ]


def test_fetch_traces_uses_nested_project_result_prefix_when_canonical_empty(
    tmp_path, monkeypatch
):
    key = (
        "ex_1_1/ex_1_1/jobs/simple_acoustic/time/results/traces/"
        "shards/f_1.00000_hz.h5"
    )
    s3_client = FakeS3Client({key: "trace"})
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site._emit = lambda message: None
    monkeypatch.setattr(
        aws_module.TraceDataset,
        "from_job",
        lambda job, upscale, project_path: "dataset",
    )
    project_path = tmp_path / "ex_1_1"
    job = SimpleNamespace(
        project_path=project_path,
        name="time",
        simulation=SimpleNamespace(name="simple_acoustic"),
        trace_outputs=SimpleNamespace(
            path=project_path
            / "jobs"
            / "simple_acoustic"
            / "time"
            / "results"
            / "traces"
        ),
    )

    result = site.fetch_traces([job])

    assert result == "dataset"
    assert (
        project_path
        / "jobs"
        / "simple_acoustic"
        / "time"
        / "results"
        / "traces"
        / "shards"
        / "f_1.00000_hz.h5"
    ).read_text() == "trace"
    assert s3_client.paginate_calls == [
        {
            "Bucket": "bucket",
            "Prefix": "ex_1_1/jobs/simple_acoustic/time/results/traces/",
        },
        {
            "Bucket": "bucket",
            "Prefix": "ex_1_1/ex_1_1/jobs/simple_acoustic/time/results/traces/",
        },
    ]
