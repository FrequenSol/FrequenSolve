import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws.aws import AWSSite
from frequensolve.orchestrator.sites.base import JobStatus
from frequensolve.simulation.jobs import BaseJob, ImagingJob
from frequensolve.simulation.outputs import JobOutputs


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


def result_job(project_path, run_id="simulation-1"):
    job = object.__new__(BaseJob)
    job.name = "job-a"
    job.simulation = SimpleNamespace(
        name="simulation-a",
        project_path=project_path,
        acquisition=SimpleNamespace(receiver_groups=[], source_field_ids=lambda: []),
    )
    job.outputs = JobOutputs()
    job.f_list = [10.0]
    job._job_id = run_id
    return job


def result_api(site, job):
    prefix = f"{job.project_path.name}/{job._result_path.relative_to(job.project_path).as_posix()}"
    site.graphql_client = SimpleNamespace(
        get_simulation_status_details=lambda run_id: {
            "id": run_id,
            "status": "SUCCEEDED",
            "outputIdentity": f"s3://bucket/{prefix}/runs/{run_id}/",
        }
    )


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
    job = result_job(tmp_path / "project-a")
    result_api(site, job)

    with pytest.raises(RuntimeError, match="download failed"):
        site.fetch_vtk(job)


def test_fetch_image_downloads_only_the_aggregate_image(tmp_path):
    project_path = tmp_path / "imaging-project"
    image_path = project_path / "jobs" / "model" / "rtm" / "results" / "imaging"
    image_key = (
        "imaging-project/jobs/model/rtm/results/runs/simulation-1/imaging/image.h5"
    )
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
    job.outputs = JobOutputs()
    job.f_list = [10.0]
    job._job_id = "simulation-1"
    result_api(site, job)
    image_path = image_path.parent / "runs/simulation-1/imaging"
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
    job.outputs = JobOutputs()
    job.f_list = [10.0]
    job._job_id = "simulation-1"

    with pytest.raises(ValueError):
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
    job.outputs = JobOutputs()
    job.f_list = [10.0]
    job._job_id = "simulation-1"
    result_api(site, job)

    with pytest.raises(FileNotFoundError, match="AWS imaging output .* is missing"):
        site.fetch_image(job)

    assert s3_client.attempts == 2


def test_fetch_vtk_downloads_only_configured_output_paths(tmp_path):
    key = "project-a/jobs/simulation-a/job-a/results/runs/simulation-1/paraview/pv_00000.vtu"
    s3_client = FakeS3Client({key: "mesh"})
    site = make_site(s3_client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    job = result_job(tmp_path / "project-a")
    job.outputs.paraview = [SimpleNamespace(path="paraview")]
    result_api(site, job)

    site.fetch_vtk(job)

    assert (
        tmp_path
        / "project-a/jobs/simulation-a/job-a/results/runs/simulation-1/paraview/pv_00000.vtu"
    ).read_text() == "mesh"
    assert {
        "Bucket": "bucket",
        "Prefix": "project-a/jobs/simulation-a/job-a/results/runs/simulation-1/paraview/",
    } in s3_client.paginate_calls
    assert {
        "Bucket": "bucket",
        "Prefix": "project-a/jobs/simulation-a/job-a/results/runs/simulation-1/ParaView/",
    } not in s3_client.paginate_calls


def test_fetch_output_files_downloads_paraview_outputs(tmp_path):
    site = AWSSite.__new__(AWSSite)
    site._result_job = lambda job: job
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
    site._result_job = lambda job: job
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
    site._result_job = lambda job: job
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
    job = result_job(tmp_path / "project-a")
    result_api(site, job)
    manifest_path = job._result_path / "runs/simulation-1/_fs_run/run_manifest.json"
    job.collect_task_run_manifests = lambda: manifest_path

    assert site.fetch_run_metadata(job) == manifest_path
    assert downloads == [
        (
            "s3://bucket/project-a/jobs/simulation-a/job-a/results/runs/simulation-1/_fs_run",
            manifest_path.parent,
        )
    ]


def test_fetch_outputs_downloads_complete_configured_artifact_set():
    site = AWSSite.__new__(AWSSite)
    site._result_job = lambda job: job
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
    site._result_job = lambda job: job
    calls = []
    site.fetch_run_metadata = lambda job: calls.append("metadata")
    site.fetch_traces = lambda job: calls.append("traces") or "trace-data"
    site.fetch_image = lambda job: calls.append("image")
    job = object.__new__(ImagingJob)
    job.outputs = SimpleNamespace(wavefields=[], paraview=[])

    assert site.fetch_outputs(job) == "trace-data"
    assert calls == ["metadata", "traces", "image"]


def test_aws_run_handle_honors_submit_time_fetch_after_success(tmp_path):
    site = AWSSite.__new__(AWSSite)
    fetch_calls = []
    site._poll_run = lambda run: JobStatus(
        state="completed",
        return_code=0,
        job_id=str(run.id),
    )
    site.fetch_outputs = lambda job: fetch_calls.append(job)
    site._emit_status = lambda *args, **kwargs: None
    job = result_job(tmp_path / "project-a")
    run = site._make_run_handle(
        job,
        "simulation-1",
        poll_interval=0.0,
        fetch=True,
        check=True,
    )

    result = run.wait()

    assert result.successful
    assert fetch_calls == [run.job]
    assert run.job is not job
    assert run.job._result_path == job._result_path / "runs/simulation-1"


def log_job(tmp_path):
    return SimpleNamespace(
        project_path=tmp_path / "project-a",
        _stdout_path=tmp_path / "logs",
        _result_path=tmp_path / "project-a/jobs/simulation-a/job-a/results",
        outputs=JobOutputs(),
        _job_id="simulation-1",
        f_list=[10.0, 20.0],
        name="job-a",
        simulation=SimpleNamespace(name="simulation-a"),
    )


def test_slurm_logs_read_only_the_requested_simulation_and_replace_stale_tasks(
    tmp_path,
):
    prefix = "project-a/jobs/simulation-a/job-a/logs/"
    client = FakeS3Client(
        {
            prefix + "simulation-1/task_1.log": "current run",
            prefix + "simulation-1/task_2.log": "current second task",
            prefix + "simulation-2/task_1.log": "another run",
            prefix + "task_1.log": "old unscoped log",
        }
    )
    site = make_site(client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    job = log_job(tmp_path)
    cache = job._stdout_path / job._job_id
    cache.mkdir(parents=True)
    (cache / "task_99.log").write_text("obsolete task")
    assert site.fetch_logs(job) == cache
    assert sorted(p.name for p in cache.iterdir()) == ["task_1.log", "task_2.log"]
    assert (cache / "task_1.log").read_text() == "current run"
    assert len(client.downloads) == 2


def test_slurm_logs_download_only_the_selected_frequency(tmp_path):
    key = "project-a/jobs/simulation-a/job-a/logs/simulation-1/task_2.log"
    client = FakeS3Client({key: "selected frequency"})
    site = make_site(client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    job = log_job(tmp_path)
    selected = site.fetch_logs(job, frequency=20.0)
    assert selected == job._stdout_path / "simulation-1/task_2.log"
    assert selected.read_text() == "selected frequency"
    assert [row["Key"] for row in client.downloads] == [key]


def test_slurm_log_download_failure_preserves_cache(tmp_path):
    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="bucket")
    job = log_job(tmp_path)
    cache = job._stdout_path / job._job_id
    cache.mkdir(parents=True)
    (cache / "task_1.log").write_text("cached")

    def fail(remote, local):
        (local / "task_1.log").write_text("incomplete refresh")
        raise RuntimeError("S3 unavailable")

    site.get = fail
    with pytest.raises(RuntimeError, match="S3 unavailable"):
        site.fetch_logs(job)
    assert (cache / "task_1.log").read_text() == "cached"


@pytest.mark.parametrize("run_id", [None, "", "../other", "run/other", True])
def test_slurm_log_identity_is_required_before_storage_access(tmp_path, run_id):
    site = AWSSite.__new__(AWSSite)
    job = log_job(tmp_path)
    job._job_id = run_id
    with pytest.raises(ValueError, match="simulation id"):
        site.fetch_logs(job)


def test_run_and_result_logs_remain_bound_when_the_job_is_resubmitted(tmp_path):
    prefix = "project-a/jobs/simulation-a/job-a/logs/"
    site = make_site(FakeS3Client({prefix + "simulation-1/task_1.log": "original"}))
    site.config = SimpleNamespace(s3_bucket="bucket")
    job = log_job(tmp_path)
    run = site._make_run_handle(job, "simulation-1")
    result = run._make_result(JobStatus(state="completed", return_code=0))
    job._job_id = "simulation-2"
    for value in (run, result):
        selected = value.logs(task=1)
        assert selected == job._stdout_path / "simulation-1/task_1.log"
        assert selected.read_text() == "original"


def test_rerun_handles_and_results_keep_their_own_downloads(tmp_path, monkeypatch):
    prefix = "project-a/jobs/simulation-a/job-a/results/runs/"
    client = FakeS3Client(
        {
            prefix + "simulation-1/paraview/mesh.vtu": "first mesh",
            prefix + "simulation-2/paraview/mesh.vtu": "second mesh",
            prefix + "simulation-1/nested/traces/traces_1.h5": "first traces",
            prefix + "simulation-2/nested/traces/traces_1.h5": "second traces",
        }
    )
    site = make_site(client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site._emit = lambda *_: None
    job = result_job(tmp_path / "project-a")
    job.outputs.paraview = [SimpleNamespace(path="paraview")]
    job.outputs.traces.path = "nested/traces"
    result_api(site, job)
    first = site._make_run_handle(job, "simulation-1")
    job._job_id = "simulation-2"
    second = site._make_run_handle(job, "simulation-2")
    results = [
        run._make_result(JobStatus(state="completed", return_code=0))
        for run in (first, second)
    ]
    monkeypatch.setattr(
        "frequensolve.orchestrator.sites.aws.aws.TraceDataset.from_job",
        lambda bound_job, *args, **kwargs: bound_job.trace_outputs.path,
    )

    for result, content in zip(results, ("first", "second")):
        traces = result.traces()
        assert (traces / "traces_1.h5").read_text() == f"{content} traces"
        files = result.output_files(kind="vtu", existing=True)
        assert [path.read_text() for path in files] == [f"{content} mesh"]
        assert all(path.is_relative_to(result.job._result_path) for path in files)
    # Repeated explicit fetch of the old run still uses its own job snapshot.
    site.fetch_run_metadata = lambda *_: None
    first.fetch()
    first.fetch()
    assert (first.job._result_path / "paraview/mesh.vtu").read_text() == "first mesh"
    assert (second.job._result_path / "paraview/mesh.vtu").read_text() == "second mesh"
    assert first.job._job_id == "simulation-1"
    assert first.job._result_path != second.job._result_path
    assert not (job._result_path / "paraview/mesh.vtu").exists()
    assert {call["Prefix"] for call in client.paginate_calls} == {
        prefix + run_id + "/" + path + "/"
        for run_id in ("simulation-1", "simulation-2")
        for path in ("nested/traces", "paraview")
    }


@pytest.mark.parametrize(
    "details",
    [
        {
            "id": "simulation-2",
            "outputIdentity": "s3://bucket/project-a/jobs/simulation-a/job-a/results/runs/simulation-1/",
        },
        {
            "id": "simulation-1",
            "outputIdentity": "s3://other-bucket/project-a/jobs/simulation-a/job-a/results/runs/simulation-1/",
        },
        {
            "id": "simulation-1",
            "outputIdentity": "s3://bucket/project-a/jobs/simulation-a/job-a/results/runs/simulation-2/",
        },
        {
            "id": "simulation-1",
            "outputIdentity": "s3://bucket/project-a/jobs/simulation-a/job-a/results/",
        },
        {"id": "simulation-1", "outputIdentity": None},
    ],
)
def test_result_identity_must_match_owned_run_before_storage_access(tmp_path, details):
    client = FakeS3Client({})
    site = make_site(client)
    site.config = SimpleNamespace(s3_bucket="bucket")
    site.graphql_client = SimpleNamespace(
        get_simulation_status_details=lambda *_: details
    )
    with pytest.raises(RuntimeError, match="does not match"):
        site.fetch_vtk(result_job(tmp_path / "project-a"))
    assert client.head_calls == client.paginate_calls == client.downloads == []


@pytest.mark.parametrize("run_id", [None, "", "../other", "run/other", True])
def test_result_fetch_requires_submission_identity(tmp_path, run_id):
    site = make_site(FakeS3Client({}))
    with pytest.raises(ValueError, match="simulation id"):
        site.fetch_vtk(result_job(tmp_path / "project-a", run_id))


@pytest.mark.parametrize(
    "output_path",
    ["/absolute", "../outside", "nested/../../outside", "nested\\outside"],
)
def test_configured_outputs_cannot_escape_run_results(tmp_path, output_path):
    site = make_site(FakeS3Client({}))
    site.config = SimpleNamespace(s3_bucket="bucket")
    job = result_job(tmp_path / "project-a")
    job.outputs.paraview = [SimpleNamespace(path=output_path)]
    result_api(site, job)
    with pytest.raises(ValueError, match="inside the run"):
        site.fetch_vtk(job)
    assert site.s3_client.downloads == []


def test_wavefields_preserve_nested_output_path_in_the_selected_run(
    tmp_path, monkeypatch
):
    from frequensolve.simulation.outputs import WavefieldOutput

    prefix = "project-a/jobs/simulation-a/job-a/results/runs/simulation-1/"
    site = make_site(FakeS3Client({prefix + "nested/waves/traces_1.h5": "wavefield"}))
    site.config = SimpleNamespace(s3_bucket="bucket")
    site._emit = lambda *_: None
    job = result_job(tmp_path / "project-a")
    job.outputs.wavefields = [
        WavefieldOutput(
            name="pressure",
            field="pressure",
            path="nested/waves",
            dims=("z", "r"),
            coords={"z": [0.0, 1.0], "r": [0.0, 1.0]},
        )
    ]
    result_api(site, job)
    monkeypatch.setattr(
        "frequensolve.seismic.traces.TraceDataset.from_manifest",
        lambda manifest, **kwargs: manifest.output_path,
    )
    alternate = tmp_path / "downloaded-project"
    path = site.fetch_wavefields(job, path=alternate)
    assert (
        path
        == alternate / "jobs/simulation-a/job-a/results/runs/simulation-1/nested/waves"
    )
    assert (path / "traces_1.h5").read_text() == "wavefield"
    assert site.s3_client.paginate_calls == [
        {"Bucket": "bucket", "Prefix": prefix + "nested/waves/"}
    ]


def test_imaging_results_keep_each_run_separate(tmp_path, monkeypatch):
    prefix = "project-a/jobs/simulation-a/job-a/results/runs/"
    site = make_site(
        FakeS3Client(
            {
                prefix + "simulation-1/imaging/image.h5": "first image",
                prefix + "simulation-2/imaging/image.h5": "second image",
            }
        )
    )
    site.config = SimpleNamespace(s3_bucket="bucket")
    site._emit = lambda *_: None
    job = object.__new__(ImagingJob)
    job.__dict__.update(result_job(tmp_path / "project-a").__dict__)
    job.save_path = job._result_path / "imaging"
    result_api(site, job)
    monkeypatch.setattr(
        ImagingJob, "load_images", lambda selected: selected.image_file().read_text()
    )
    first = site._make_run_handle(job, "simulation-1")._make_result(
        JobStatus(state="completed", return_code=0)
    )
    job._job_id = "simulation-2"
    second = site._make_run_handle(job, "simulation-2")._make_result(
        JobStatus(state="completed", return_code=0)
    )
    assert second.images() == "second image"
    assert first.images() == "first image"
    assert second.job.image_file().read_text() == "second image"
    assert first.job.image_file().read_text() == "first image"
    assert not job.image_file().exists()


def test_old_run_keeps_nested_receiver_and_simulation_metadata(tmp_path):
    site = make_site(FakeS3Client({}))
    job = result_job(tmp_path / "project-a")
    group = SimpleNamespace(
        name="line-a",
        device=SimpleNamespace(components=[SimpleNamespace(name="pressure")]),
    )
    job.simulation.acquisition.receiver_groups = [group]
    job.simulation.units = {"pressure": "Pa"}
    job.simulation.extra = {"coordinates": [1.0, 2.0]}
    job.k_list = [1.0]
    job.k_weights = [0.5]
    first = site._make_run_handle(job, "simulation-1")
    first_result = first._make_result(JobStatus(state="completed", return_code=0))

    group.name = "changed-line"
    group.device.components[0].name = "velocity"
    job.simulation.acquisition.receiver_groups.append(
        SimpleNamespace(name="line-b", device=SimpleNamespace(components=[]))
    )
    job.simulation.units["pressure"] = "kPa"
    job.simulation.extra["coordinates"][0] = 99.0
    job.k_list[0] = 2.0
    job.k_weights[0] = 0.25
    second = site._make_run_handle(job, "simulation-2")

    for run in (first, first_result):
        assert run.job.trace_outputs.groups == ["line-a"]
        assert run.job.trace_outputs.components == ["line-a:pressure"]
        assert run.job.simulation.units == {"pressure": "Pa"}
        assert run.job.simulation.extra == {"coordinates": [1.0, 2.0]}
        assert run.job.k_list == [1.0]
        assert run.job.k_weights == [0.5]
    assert second.job.trace_outputs.groups == ["changed-line", "line-b"]
    assert second.job.trace_outputs.components == ["changed-line:velocity"]


def test_snapshot_copies_real_simulation_without_copying_its_project(tmp_path):
    from frequensolve.simulation.simulation import SeismicSimulation

    class ProjectReference:
        def __deepcopy__(self, memo):
            raise AssertionError("A run must not recursively copy its owning project")

    job = result_job(tmp_path / "project-a")
    job.simulation = SeismicSimulation(
        name="simulation-a",
        physics="acoustic",
        dimension=3,
        project_path=job.project_path,
    )
    project = ProjectReference()
    job.simulation._project = project
    job.simulation.extra = {"geometry": {"origin": [1.0, 2.0, 3.0]}}
    snapshot = AWSSite._snapshot_run_job(job, "simulation-1")
    job.simulation.extra["geometry"]["origin"][0] = 99.0

    assert snapshot.simulation is not job.simulation
    assert snapshot.simulation.acquisition is not job.simulation.acquisition
    assert snapshot.simulation.model is not job.simulation.model
    assert snapshot.simulation.mesh is not job.simulation.mesh
    assert snapshot.simulation._project is project
    assert snapshot.simulation.extra["geometry"]["origin"] == [1.0, 2.0, 3.0]


def test_real_imaging_run_preserves_grid_and_fwi_metadata(tmp_path):
    import h5py
    import numpy as np

    from frequensolve.geometry.grids import CartesianGrid
    from frequensolve.simulation.jobs.fwi import build_imaging_job
    from frequensolve.simulation.simulation import SeismicSimulation

    simulation = SeismicSimulation(
        name="model", physics="elastic", dimension=2, project_path=tmp_path
    )
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[2.0, 1.0])
    job = build_imaging_job(
        simulation,
        frequencies=[5.0],
        grid=grid,
        parameters=["Vp"],
        weights=[1.0],
        regularization={"type": "TV", "schedule": [0.1]},
        interpretation={"units": ["m/s"]},
    )
    site = make_site(FakeS3Client({}))
    first = site._make_run_handle(job, "simulation-1")
    first_result = first._make_result(JobStatus(state="completed", return_code=0))
    first_image = first.job.image_file()
    first_image.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(first_image, "w") as image:
        image.create_dataset("vp", data=np.arange(6).reshape(2, 3))

    job.grid.n[0] = 4
    job.grid.x1[0] = 3.0
    job.weights[0] = 2.0
    job.regularization["schedule"][0] = 0.9
    job.kwargs["interpretation"]["units"][0] = "km/s"
    job.images.clear()
    second = site._make_run_handle(job, "simulation-2")

    for run in (first, first_result):
        assert run.job.grid.shape == (2, 3)
        assert run.job.grid.x1 == [2.0, 1.0]
        assert run.job.weights == [1.0]
        assert run.job.regularization["schedule"] == [0.1]
        assert run.job.kwargs["interpretation"]["units"] == ["m/s"]
        assert run.job.images
        assert run.job.load_images().shape == (2, 3)
    assert second.job.grid.shape == (2, 4)
    assert second.job.regularization["schedule"] == [0.9]
    assert second.job.images == {}
    assert first.job.grid is not second.job.grid
    assert job.save_path == job._result_path / "imaging"


@pytest.mark.parametrize("method", ["sync_s3", "put"])
@pytest.mark.parametrize("error_kind", ["client", "managed", "unexpected"])
def test_upload_failure_sanitizes_provider_payload_and_traceback(
    tmp_path, caplog, method, error_kind
):
    private = "private-bucket/private-account/private-key?token=synthetic-secret"
    if error_kind == "client":
        error = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": private}}, "PutObject"
        )
    elif error_kind == "managed":
        error = S3UploadFailedError(private)
    else:
        error = OSError(private)
    calls = []

    def upload_file(*args):
        calls.append(args)
        raise error

    site = make_site(SimpleNamespace(upload_file=upload_file))
    site.config = SimpleNamespace(
        s3_bucket="private-bucket", s3_prefix="private-account"
    )
    source = tmp_path / "input.json"
    source.write_text("synthetic input")
    with caplog.at_level("DEBUG"):
        with pytest.raises(RuntimeError, match="Cloud file upload failed") as exc_info:
            getattr(site, method)(source, "private-key")
    rendered = "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert "Check your storage access and retry" in str(exc_info.value)
    assert "synthetic-secret" not in rendered
    assert "AccessDenied" not in rendered
    assert private not in rendered + caplog.text
    assert "s3://private-bucket" not in caplog.text
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["sync_s3", "put"])
def test_upload_stops_on_partial_failure_without_deleting_existing_objects(
    tmp_path, method
):
    uploaded = []

    def upload_file(path, bucket, key):
        uploaded.append(key)
        if len(uploaded) == 2:
            raise S3UploadFailedError("synthetic private provider failure")

    site = make_site(SimpleNamespace(upload_file=upload_file))
    site.config = SimpleNamespace(s3_bucket="bucket", s3_prefix="tenant-prefix")
    source = tmp_path / "inputs"
    source.mkdir()
    for name in ["a.json", "b.json", "c.json"]:
        (source / name).write_text("synthetic input")
    with pytest.raises(RuntimeError, match="Cloud file upload failed"):
        getattr(site, method)(source, "project/inputs")
    assert len(uploaded) == 2
    # The SDK has no ownership proof permitting deletion of pre-existing keys.
    # No delete API exists on this fake, so an invented rollback would fail.


@pytest.mark.parametrize("method", ["sync_s3", "put"])
def test_upload_preserves_file_and_nested_directory_destinations(tmp_path, method):
    calls = []
    site = make_site(SimpleNamespace(upload_file=lambda *args: calls.append(args)))
    site.config = SimpleNamespace(s3_bucket="bucket", s3_prefix="tenant-prefix")
    source = tmp_path / "inputs"
    (source / "nested").mkdir(parents=True)
    (source / "a.json").write_text("a")
    (source / "nested/b.json").write_text("b")
    prefix = "tenant-prefix/" if method == "put" else ""
    result = getattr(site, method)(source / "a.json", "single.json")
    assert calls == [(str(source / "a.json"), "bucket", prefix + "single.json")]
    assert result == ("single.json" if method == "sync_s3" else None)
    calls.clear()
    getattr(site, method)(source, "project/inputs")
    assert {call[2] for call in calls} == {
        prefix + "project/inputs/a.json",
        prefix + "project/inputs/nested/b.json",
    }


@pytest.mark.parametrize("method", ["sync_s3", "put"])
def test_upload_rejects_missing_local_path_before_any_provider_call(tmp_path, method):
    site = make_site(SimpleNamespace())
    site.config = SimpleNamespace(s3_bucket="bucket", s3_prefix="tenant-prefix")
    with pytest.raises(FileNotFoundError):
        getattr(site, method)(tmp_path / "missing", "remote-key")
