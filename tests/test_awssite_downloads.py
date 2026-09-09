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


def test_get_refreshes_when_head_object_masks_expired_credentials(tmp_path):
    masked_expiry = ClientError(
        {"Error": {"Code": "400", "Message": "private provider detail"}},
        "HeadObject",
    )
    s3_client = FakeS3Client({"path/result.json": "{}"}, failures=[masked_expiry])
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


def test_get_rejects_object_key_that_escapes_destination(tmp_path):
    outside = tmp_path / "outside.txt"
    s3_client = FakeS3Client({"path/results/../../outside.txt": "private content"})
    site = make_site(s3_client)

    with pytest.raises(RuntimeError, match="S3 transfer failed"):
        site.get("s3://bucket/path/results/", tmp_path / "downloads")

    assert not outside.exists()
    assert s3_client.downloads == []


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


@pytest.mark.parametrize(
    ("credential_error_code", "operation_name"),
    [("ExpiredToken", "GetObject"), ("400", "HeadObject")],
)
def test_fetch_image_normalizes_missing_output_after_credential_refresh(
    tmp_path, credential_error_code, operation_name
):
    project_path = tmp_path / "imaging-project"

    class ExpiredThenMissingS3Client:
        def __init__(self):
            self.attempts = 0

        def download_file(self, bucket, key, filename):
            self.attempts += 1
            code = credential_error_code if self.attempts == 1 else "NoSuchKey"
            raise ClientError(
                {"Error": {"Code": code, "Message": code}},
                operation_name if self.attempts == 1 else "GetObject",
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

    with pytest.raises(FileNotFoundError, match="AWS imaging output .* is missing"):
        site.fetch_image(job)

    assert s3_client.attempts == 2


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


def test_fetch_logs_falls_back_to_authenticated_cloudwatch_events(tmp_path):
    class CloudLogs:
        def __init__(self):
            self.requested = []

        def list_simulation_frequency_jobs(self, simulation_id):
            assert simulation_id == "simulation-1"
            return [
                {"frequencyIndex": 0, "batchJobId": "batch-1"},
                {"frequencyIndex": 1, "batchJobId": "batch-2"},
            ]

        def get_job_logs(self, batch_job_id):
            self.requested.append(batch_job_id)
            return [
                {
                    "timestamp": "2026-09-06T14:00:00.000Z",
                    "message": f"{batch_job_id} complete\n",
                }
            ]

    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site.graphql_client = CloudLogs()
    site.get = lambda remote, local: (_ for _ in ()).throw(FileNotFoundError())
    job = SimpleNamespace(
        project_path=tmp_path / "project-a",
        _stdout_path=tmp_path / "project-a/logs",
        _job_id="simulation-1",
        f_list=[10.0, 20.0],
        name="job-a",
        simulation=SimpleNamespace(name="simulation-a"),
    )
    job._stdout_path.mkdir(parents=True)
    (job._stdout_path / "task_99.log").write_text("stale\n")

    assert site.fetch_logs(job) == job._stdout_path
    assert site.graphql_client.requested == ["batch-1", "batch-2"]
    assert not (job._stdout_path / "task_99.log").exists()
    assert (job._stdout_path / "task_1.log").read_text() == "batch-1 complete\n"
    assert (job._stdout_path / "task_2.log").read_text() == "batch-2 complete\n"


def test_fetch_logs_cloudwatch_fallback_honors_frequency_selector(tmp_path):
    class CloudLogs:
        def list_simulation_frequency_jobs(self, simulation_id):
            return [
                {"frequencyIndex": 0, "batchJobId": "batch-1"},
                {"frequencyIndex": 1, "batchJobId": "batch-2"},
            ]

        def get_job_logs(self, batch_job_id):
            assert batch_job_id == "batch-2"
            return [{"timestamp": "now", "message": "selected"}]

    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site.graphql_client = CloudLogs()
    site.get = lambda remote, local: (_ for _ in ()).throw(FileNotFoundError())
    job = SimpleNamespace(
        project_path=tmp_path / "project-a",
        _stdout_path=tmp_path / "project-a/logs",
        _job_id="simulation-1",
        f_list=[10.0, 20.0],
        name="job-a",
        simulation=SimpleNamespace(name="simulation-a"),
    )

    selected = site.fetch_logs(job, frequency=20.0)

    assert selected == job._stdout_path / "task_2.log"
    assert selected.read_text() == "selected\n"
    assert not (job._stdout_path / "task_1.log").exists()


def test_fetch_logs_cloudwatch_failure_preserves_existing_cache(tmp_path):
    class CloudLogs:
        def list_simulation_frequency_jobs(self, simulation_id):
            return [
                {"frequencyIndex": 0, "batchJobId": "batch-1"},
                {"frequencyIndex": 1, "batchJobId": "batch-2"},
            ]

        def get_job_logs(self, batch_job_id):
            if batch_job_id == "batch-2":
                raise RuntimeError("CloudWatch unavailable")
            return [{"timestamp": "now", "message": "replacement"}]

    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site.graphql_client = CloudLogs()
    site.get = lambda remote, local: (_ for _ in ()).throw(FileNotFoundError())
    job = SimpleNamespace(
        project_path=tmp_path / "project-a",
        _stdout_path=tmp_path / "project-a/logs",
        _job_id="simulation-1",
        f_list=[10.0, 20.0],
        name="job-a",
        simulation=SimpleNamespace(name="simulation-a"),
    )
    job._stdout_path.mkdir(parents=True)
    (job._stdout_path / "task_1.log").write_text("cached one\n")
    (job._stdout_path / "task_2.log").write_text("cached two\n")

    with pytest.raises(RuntimeError, match="CloudWatch unavailable"):
        site.fetch_logs(job)

    assert (job._stdout_path / "task_1.log").read_text() == "cached one\n"
    assert (job._stdout_path / "task_2.log").read_text() == "cached two\n"


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


@pytest.mark.parametrize("focus", [False, True])
def test_fetch_outputs_downloads_control_postprocess_products(tmp_path, focus):
    from frequensolve.simulation.jobs import (
        RTMControlSensitivityJob,
        TimeReversalFocusJob,
    )
    from frequensolve.simulation.simulation import SeismicSimulation

    simulation = SeismicSimulation(
        name="sim", physics="acoustic", dimension=2, project_path=tmp_path
    )
    job_type = TimeReversalFocusJob if focus else RTMControlSensitivityJob
    kwargs = (
        {"objective_file": tmp_path / "objective.json", "softening": 1.0}
        if focus
        else {}
    )
    job = job_type(
        "gradient",
        simulation,
        [3.0],
        observed=tmp_path / "observed.h5",
        gradient=tmp_path / "gradient.h5",
        **kwargs,
    )
    outputs = [tmp_path / "gradient_raw.h5", tmp_path / "gradient.h5"]
    if focus:
        outputs.append(tmp_path / "objective.json")
    assert job.postprocess_fetch_files() == outputs
    objects = {
        f"{tmp_path.name}/{path.relative_to(tmp_path).as_posix()}": path.name
        for path in outputs
    }
    client = FakeS3Client(objects)
    site = make_site(client)
    site.config = SimpleNamespace(s3_bucket="test-bucket")
    site.fetch_run_metadata = lambda job: None
    assert site.fetch_outputs(job) == outputs
    assert [path.read_text() for path in outputs] == [path.name for path in outputs]
    assert len(client.downloads) == len(outputs)


def test_fetch_postprocess_propagates_missing_required_product(tmp_path):
    site = make_site(FakeS3Client({}))
    site.config = SimpleNamespace(s3_bucket="test-bucket")
    job = SimpleNamespace(
        project_path=tmp_path,
        postprocess_fetch_files=lambda: [tmp_path / "gradient.h5"],
    )
    with pytest.raises(FileNotFoundError, match="No S3 objects"):
        site.fetch_postprocess(job)
