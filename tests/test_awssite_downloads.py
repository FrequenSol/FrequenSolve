from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws.aws import AWSSite
from frequensolve.orchestrator.sites.base import JobStatus
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


def test_fetch_image_downloads_only_the_aggregate_image(tmp_path):
    project_path = tmp_path / "imaging-project"
    image_path = project_path / "jobs" / "model" / "rtm" / "results" / "imaging"
    image_key = "imaging-project/jobs/model/rtm/results/imaging/image.h5"
    s3_client = FakeS3Client(
        {
            image_key: "image payload",
            f"{image_key.removesuffix('image.h5')}image_1.h5": "shard payload",
        }
    )
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site._emit = lambda message: None

    job = object.__new__(ImagingJob)
    job.name = "rtm"
    job.simulation = SimpleNamespace(project_path=project_path, name="model")
    job.save_path = image_path
    expected = object()
    job.load_images = lambda: expected

    assert site.fetch_image(job) is expected
    assert (image_path / "image.h5").read_text() == "image payload"
    assert s3_client.downloads == [
        {
            "Bucket": "bucket",
            "Key": image_key,
            "Filename": str(image_path / "image.h5"),
        }
    ]
    assert s3_client.paginate_calls == []


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


def test_fetch_run_metadata_downloads_job_run_directory(tmp_path):
    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="bucket")
    downloads = []
    messages = []
    site.get = lambda remote, local: downloads.append((remote, local))
    site._emit = messages.append
    manifest_path = (
        tmp_path / "project-a/jobs/simulation-a/job-a/results/_fs_run/run_manifest.json"
    )
    job = SimpleNamespace(
        project_path=tmp_path / "project-a",
        _result_path=tmp_path / "project-a/jobs/simulation-a/job-a/results",
        simulation=SimpleNamespace(name="simulation-a"),
        name="job-a",
        collect_task_run_manifests=lambda: manifest_path,
    )

    assert site.fetch_run_metadata(job) == manifest_path
    assert downloads == [
        (
            "s3://bucket/project-a/jobs/simulation-a/job-a/results/_fs_run",
            job._result_path / "_fs_run",
        )
    ]
    assert messages == [
        "Fetched AWS run metadata from "
        "s3://bucket/project-a/jobs/simulation-a/job-a/results/_fs_run"
    ]


def test_fetch_outputs_downloads_complete_configured_artifact_set():
    site = AWSSite.__new__(AWSSite)
    calls = []
    site.fetch_run_metadata = lambda job: calls.append("metadata")
    site.fetch_traces = lambda job: calls.append("traces") or "trace-data"
    site.fetch_wavefields = lambda job: calls.append("wavefields") or "wave-data"
    site.fetch_paraview = lambda job: calls.append("paraview")
    job = SimpleNamespace(
        outputs=SimpleNamespace(wavefields=[object()], paraview=[object()])
    )

    assert site.fetch_outputs(job) == {
        "traces": "trace-data",
        "wavefields": "wave-data",
    }
    assert calls == ["metadata", "traces", "wavefields", "paraview"]


def test_fetch_outputs_downloads_aggregate_image_for_imaging_job():
    site = AWSSite.__new__(AWSSite)
    calls = []
    site.fetch_run_metadata = lambda job: calls.append("metadata")
    site.fetch_traces = lambda job: calls.append("traces") or "trace-data"
    site.fetch_image = lambda job: calls.append("image")
    job = object.__new__(ImagingJob)
    job.outputs = SimpleNamespace(wavefields=[], paraview=[])

    assert site.fetch_outputs(job) == "trace-data"
    assert calls == ["metadata", "traces", "image"]


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
