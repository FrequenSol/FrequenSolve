from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws.aws import AWSSite
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


def test_fetch_vtk_reraises_download_failures(tmp_path):
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
            project_path=tmp_path / "project-a",
        ),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        site.fetch_vtk(job)


def test_fetch_image_downloads_project_relative_image_directory(tmp_path):
    project_path = tmp_path / "imaging-project"
    image_path = project_path / "jobs" / "model" / "rtm" / "results" / "imaging"
    image_key = "imaging-project/jobs/model/rtm/results/imaging/image.h5"
    s3_client = FakeS3Client({image_key: "image payload"})
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")

    job = object.__new__(ImagingJob)
    job.name = "rtm"
    job.simulation = SimpleNamespace(project_path=project_path, name="model")
    job.save_path = image_path
    expected = object()
    job.load_images = lambda: expected

    images = site.fetch_image(job)

    assert images is expected
    assert (image_path / "image.h5").read_text() == "image payload"
    assert s3_client.paginate_calls == [
        {
            "Bucket": "bucket",
            "Prefix": "imaging-project/jobs/model/rtm/results/imaging/",
        }
    ]


def test_fetch_image_omits_dot_segment_for_project_root_output(tmp_path):
    project_path = tmp_path / "imaging-project"
    image_key = "imaging-project/image.h5"
    s3_client = FakeS3Client({image_key: "current image"})
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")

    job = object.__new__(ImagingJob)
    job.name = "rtm"
    job.simulation = SimpleNamespace(project_path=project_path, name="model")
    job.save_path = project_path
    expected = object()
    job.load_images = lambda: expected

    images = site.fetch_image(job)

    assert images is expected
    assert (project_path / "image.h5").read_text() == "current image"
    assert s3_client.paginate_calls == [
        {"Bucket": "bucket", "Prefix": "imaging-project/"}
    ]


def test_fetch_image_rejects_missing_remote_aggregate_before_using_local(tmp_path):
    project_path = tmp_path / "imaging-project"
    image_path = project_path / "jobs" / "model" / "rtm" / "results" / "imaging"
    image_path.mkdir(parents=True)
    (image_path / "image.h5").write_text("stale image")
    image_key = "imaging-project/jobs/model/rtm/results/imaging/image_1.h5"
    s3_client = FakeS3Client({image_key: "current shard"})
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")

    job = object.__new__(ImagingJob)
    job.name = "rtm"
    job.simulation = SimpleNamespace(project_path=project_path, name="model")
    job.save_path = image_path
    job.load_images = lambda: pytest.fail("stale local images must not be opened")

    with pytest.raises(FileNotFoundError, match="required aggregate image.h5"):
        site.fetch_image(job)

    assert (image_path / "image.h5").read_text() == "stale image"
