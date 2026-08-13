from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")
from botocore.exceptions import ClientError

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
    def __init__(self, objects, *, failures=None):
        self.objects = objects
        self.downloads = []
        self.head_calls = []
        self.paginate_calls = []
        self.failures = list(failures or [])

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self)

    def head_object(self, Bucket, Key):
        self.head_calls.append({"Bucket": Bucket, "Key": Key})
        if self.failures:
            raise self.failures.pop(0)
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


def test_get_reports_missing_object_or_prefix_without_private_key(tmp_path):
    s3_client = FakeS3Client({})
    site = make_site(s3_client)
    private_path = "s3://bucket/accounts/private-user/missing-result"

    with pytest.raises(FileNotFoundError) as exc_info:
        site.get(private_path, tmp_path / "downloads")

    assert "No S3 objects matched" in str(exc_info.value)
    assert "private-user" not in str(exc_info.value)


def test_get_refreshes_expired_credentials_once_then_downloads(tmp_path):
    expired = ClientError(
        {"Error": {"Code": "ExpiredToken", "Message": "private token"}},
        "HeadObject",
    )
    s3_client = FakeS3Client({"path/result.json": "{}"}, failures=[expired])
    site = make_site(s3_client)
    refreshes = []
    site.cognito_auth = object()
    site._refresh_s3_credentials = lambda: refreshes.append(True)

    site.get("s3://bucket/path/result.json", tmp_path / "downloads")

    assert refreshes == [True]
    assert (tmp_path / "downloads" / "result.json").read_text() == "{}"


def test_get_stops_after_one_refresh_and_sanitizes_provider_failure(tmp_path):
    failures = [
        ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "first secret"}},
            "HeadObject",
        ),
        ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "second secret"}},
            "HeadObject",
        ),
    ]
    site = make_site(FakeS3Client({}, failures=failures))
    site.cognito_auth = object()
    refreshes = []
    site._refresh_s3_credentials = lambda: refreshes.append(True)

    with pytest.raises(RuntimeError, match="AWS error code ExpiredToken") as exc_info:
        site.get(
            "s3://bucket/accounts/private-user/result.json",
            tmp_path / "downloads",
        )

    assert refreshes == [True]
    diagnostic = str(exc_info.value)
    for secret in ("first secret", "second secret", "private-user"):
        assert secret not in diagnostic


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


def test_fetch_vtk_downloads_only_configured_output_paths(tmp_path):
    key = "project-a/jobs/simulation-a/job-a/results/paraview/pv_00000.vtu"
    s3_client = FakeS3Client({key: "mesh"})
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    job = SimpleNamespace(
        project_path=tmp_path,
        name="job-a",
        outputs=SimpleNamespace(paraview=[SimpleNamespace(path="paraview")]),
        simulation=SimpleNamespace(
            name="simulation-a",
            project_path=tmp_path / "project-a",
        ),
    )

    site.fetch_vtk(job)

    assert (
        tmp_path / "jobs/simulation-a/job-a/results/paraview/pv_00000.vtu"
    ).read_text() == "mesh"
    assert {
        "Bucket": "bucket",
        "Prefix": "project-a/jobs/simulation-a/job-a/results/paraview/",
    } in s3_client.paginate_calls
    assert {
        "Bucket": "bucket",
        "Prefix": "project-a/jobs/simulation-a/job-a/results/ParaView/",
    } not in s3_client.paginate_calls


def test_fetch_output_files_downloads_paraview_outputs(tmp_path):
    site = AWSSite.__new__(AWSSite)
    calls = []

    def fetch_paraview(job):
        calls.append(job)

    site.fetch_paraview = fetch_paraview
    job = SimpleNamespace(
        _result_path=tmp_path / "results",
        outputs=SimpleNamespace(paraview=[object()]),
    )

    assert site.fetch_output_files(job) == job._result_path
    assert calls == [job]


@pytest.mark.parametrize(
    ("kind", "suffix"),
    [
        ("xmf", None),
        ("xdmf", None),
        (None, ".xmf"),
        (" XDMF ", ".XMF"),
        (None, (".h5", ".xmf")),
    ],
)
def test_fetch_output_files_downloads_xdmf_outputs(tmp_path, kind, suffix):
    site = AWSSite.__new__(AWSSite)
    calls = []
    site.fetch_paraview = calls.append
    job = SimpleNamespace(
        _result_path=tmp_path / "results",
        outputs=SimpleNamespace(paraview=[object()]),
    )

    assert site.fetch_output_files(job, kind=kind, suffix=suffix) == job._result_path
    assert calls == [job]


@pytest.mark.parametrize(
    ("kind", "suffix"),
    [
        ("hdf5", None),
        (None, ".h5"),
        ("vtk", ".h5"),
        ("xdmf", ".h5"),
    ],
)
def test_fetch_output_files_skips_unsupported_filters(tmp_path, kind, suffix):
    site = AWSSite.__new__(AWSSite)
    calls = []
    site.fetch_paraview = calls.append
    job = SimpleNamespace(
        _result_path=tmp_path / "results",
        outputs=SimpleNamespace(paraview=[object()]),
    )

    assert site.fetch_output_files(job, kind=kind, suffix=suffix) == job._result_path
    assert calls == []
