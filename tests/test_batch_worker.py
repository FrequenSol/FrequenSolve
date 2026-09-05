import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("boto3")
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws import batch_worker


def _client_error(code="AccessDenied", message="private/provider/detail"):
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "HermeticOperation",
    )


class FakeBatchClient:
    def __init__(self, *, fail_tasks=(), malformed_tasks=()):
        self.calls = []
        self.fail_tasks = set(fail_tasks)
        self.malformed_tasks = set(malformed_tasks)

    def submit_job(self, **kwargs):
        self.calls.append(kwargs)
        task_id = int(kwargs["parameters"]["task_id"])
        if task_id in self.fail_tasks:
            raise _client_error(message="secret-account-id/private-key")
        if task_id in self.malformed_tasks:
            return {"jobId": ""}
        return {"jobId": f"job-{task_id}"}


class FakeS3Client:
    def __init__(self, *, fail_upload_number=None, reported_size=None):
        self.uploads = []
        self.deleted = []
        self.objects = {}
        self.fail_upload_number = fail_upload_number
        self.reported_size = reported_size

    def upload_file(self, filename, bucket, key):
        upload_number = len(self.uploads) + 1
        if upload_number == self.fail_upload_number:
            raise _client_error(message=f"private/{key}")
        content = Path(filename).read_bytes()
        self.uploads.append((filename, bucket, key))
        self.objects[key] = content

    def head_object(self, Bucket, Key):
        size = (
            self.reported_size
            if self.reported_size is not None
            else len(self.objects[Key])
        )
        return {"ContentLength": size}

    def delete_object(self, Bucket, Key):
        self.deleted.append((Bucket, Key))
        self.objects.pop(Key, None)


def _worker(tmp_path, **kwargs):
    kwargs.setdefault("upload_id_factory", lambda: "a" * 32)
    return batch_worker.BatchWorker(
        "private-bucket",
        "us-test-2",
        local_base=tmp_path / "worker",
        connect_aws=False,
        owns_local_base=True,
        **kwargs,
    )


def test_worker_accepts_structured_clients_without_credential_discovery(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        batch_worker.boto3,
        "client",
        lambda *args, **kwargs: pytest.fail("credential discovery was attempted"),
    )
    s3_client = FakeS3Client()
    batch_client = FakeBatchClient()

    worker = _worker(tmp_path, s3_client=s3_client, batch_client=batch_client)

    assert worker.s3_client is s3_client
    assert worker.batch_client is batch_client
    assert worker.simulation_dir.is_dir()
    assert worker.job_dir.is_dir()


def test_download_uses_configured_region_without_logging_private_key(tmp_path, caplog):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    worker = _worker(tmp_path, process_runner=run)
    private_key = "accounts/private-user/simulation-input"

    with caplog.at_level(logging.INFO, logger=batch_worker.__name__):
        worker.download_from_s3(private_key, worker.simulation_dir)

    assert calls[0][0][-2:] == ["--region", "us-test-2"]
    assert calls[0][0][3] == f"s3://private-bucket/{private_key}"
    assert private_key not in caplog.text
    assert "private-bucket" not in caplog.text


def test_download_failure_is_actionable_without_echoing_private_command(
    tmp_path, caplog
):
    private_key = "accounts/account-id/private-object"

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(23, command, stderr="solver secret")

    worker = _worker(tmp_path, process_runner=fail)
    with caplog.at_level(logging.INFO, logger=batch_worker.__name__):
        with pytest.raises(RuntimeError, match="AWS CLI status 23") as exc_info:
            worker.download_from_s3(private_key, worker.simulation_dir)

    diagnostic = f"{exc_info.value}\n{caplog.text}"
    assert private_key not in diagnostic
    assert "private-bucket" not in diagnostic
    assert "solver secret" not in diagnostic
    assert exc_info.value.__cause__ is None


def test_download_explains_missing_aws_cli(tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("aws")

    worker = _worker(tmp_path, process_runner=missing)
    with pytest.raises(RuntimeError, match="AWS CLI is unavailable"):
        worker.download_from_s3("private/key", worker.simulation_dir)


def test_preliminary_analysis_reports_counts_not_private_filenames(tmp_path, caplog):
    worker = _worker(tmp_path)
    (worker.simulation_dir / "private-account.json").write_text("{}")
    (worker.simulation_dir / "customer-mesh.h5").write_bytes(b"mesh")

    with caplog.at_level(logging.INFO, logger=batch_worker.__name__):
        result = worker.run_preliminary_analysis()

    assert result["tasks_required"] == 10
    assert "Found 1 simulation configuration file(s)" in caplog.text
    assert "Found 1 simulation mesh file(s)" in caplog.text
    assert "private-account" not in caplog.text
    assert "customer-mesh" not in caplog.text


def test_submit_task_jobs_builds_exact_payloads(tmp_path):
    client = FakeBatchClient()
    worker = _worker(tmp_path, batch_client=client, clock=lambda: 1234.9)
    analysis = {"tasks_required": 2, "estimated_runtime": 60}

    assert worker.submit_task_jobs(analysis, "queue", "definition") == [
        "job-0",
        "job-1",
    ]

    first = client.calls[0]
    assert first["jobName"] == "frequensolve-task-0-1234"
    assert first["jobQueue"] == "queue"
    assert first["jobDefinition"] == "definition"
    assert first["parameters"] == {
        "task_id": "0",
        "simulation_dir": str(worker.simulation_dir),
        "job_dir": str(worker.job_dir),
        "analysis_results": json.dumps(analysis),
    }
    assert first["containerOverrides"]["environment"][-1] == {
        "name": "ANALYSIS_RESULTS",
        "value": json.dumps(analysis),
    }


@pytest.mark.parametrize("tasks", [-1, True, 10_001, 1.5, "2"])
def test_submit_task_jobs_rejects_unsafe_task_counts(tmp_path, tasks):
    worker = _worker(tmp_path, batch_client=FakeBatchClient())

    with pytest.raises(ValueError, match="tasks_required"):
        worker.submit_task_jobs({"tasks_required": tasks}, "queue", "definition")


@pytest.mark.parametrize("queue,definition", [("", "definition"), ("queue", "")])
def test_submit_task_jobs_requires_batch_routing(tmp_path, queue, definition):
    worker = _worker(tmp_path, batch_client=FakeBatchClient())

    with pytest.raises(ValueError, match="non-empty"):
        worker.submit_task_jobs({"tasks_required": 1}, queue, definition)


def test_submit_task_jobs_continues_after_sanitized_provider_failure(tmp_path, caplog):
    client = FakeBatchClient(fail_tasks={1})
    worker = _worker(tmp_path, batch_client=client)

    with caplog.at_level(logging.ERROR, logger=batch_worker.__name__):
        job_ids = worker.submit_task_jobs({"tasks_required": 3}, "queue", "definition")

    assert job_ids == ["job-0", "job-2"]
    assert "AccessDenied" in caplog.text
    assert "secret-account-id" not in caplog.text
    assert "private-key" not in caplog.text


def test_submit_task_jobs_rejects_malformed_provider_response(tmp_path):
    worker = _worker(
        tmp_path,
        batch_client=FakeBatchClient(malformed_tasks={0}),
    )

    with pytest.raises(RuntimeError, match="did not contain a job ID"):
        worker.submit_task_jobs({"tasks_required": 1}, "queue", "definition")


def test_simulation_task_uses_current_python_timeout_and_work_directory(tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    worker = _worker(tmp_path, process_runner=run)

    assert worker.run_simulation_task(7, {"estimated_runtime": 12.5}) is True
    assert calls[0][0][0] == sys.executable
    assert calls[0][1]["cwd"] == worker.simulation_dir
    assert calls[0][1]["timeout"] == 12.5


def test_simulation_task_maps_timeout_to_failure(tmp_path):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    worker = _worker(tmp_path, process_runner=timeout)
    assert worker.run_simulation_task(0, {"estimated_runtime": 1}) is False


def test_simulation_task_nonzero_exit_does_not_log_solver_details(tmp_path, caplog):
    def fail(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            17,
            "",
            "token=secret-token object=private/object account=account-id",
        )

    worker = _worker(tmp_path, process_runner=fail)
    with caplog.at_level(logging.ERROR, logger=batch_worker.__name__):
        assert worker.run_simulation_task(0, {}) is False

    assert "process status 17" in caplog.text
    for secret in ("secret-token", "private/object", "account-id"):
        assert secret not in caplog.text


@pytest.mark.parametrize(
    "task_id,analysis",
    [(-1, {}), (True, {}), (0, []), (0, {"estimated_runtime": 0})],
)
def test_simulation_task_rejects_invalid_inputs(tmp_path, task_id, analysis):
    worker = _worker(tmp_path)

    with pytest.raises(ValueError):
        worker.run_simulation_task(task_id, analysis)


def test_result_upload_verifies_size_and_preserves_nested_paths(tmp_path):
    s3 = FakeS3Client()
    worker = _worker(tmp_path, s3_client=s3)
    results = tmp_path / "results"
    (results / "nested").mkdir(parents=True)
    (results / "summary.json").write_text("{}")
    (results / "nested" / "trace.bin").write_bytes(b"trace")

    key = worker.upload_results_to_s3(results, "run-prefix")

    assert key == f"run-prefix/results/{'a' * 32}"
    assert set(s3.objects) == {
        f"run-prefix/results/{'a' * 32}/summary.json",
        f"run-prefix/results/{'a' * 32}/nested/trace.bin",
    }
    assert s3.deleted == []


def test_partial_result_upload_rolls_back_exact_uploaded_keys_without_leaking(
    tmp_path, caplog
):
    s3 = FakeS3Client(fail_upload_number=2)
    worker = _worker(tmp_path, s3_client=s3)
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.txt").write_text("a")
    (results / "b.txt").write_text("b")

    with caplog.at_level(logging.INFO, logger=batch_worker.__name__):
        with pytest.raises(RuntimeError, match="AWS error code AccessDenied") as exc:
            worker.upload_results_to_s3(results, "private/account-prefix")

    assert s3.objects == {}
    assert s3.deleted == [("private-bucket", s3.uploads[0][2])]
    diagnostic = f"{exc.value}\n{caplog.text}"
    assert "account-prefix" not in diagnostic
    assert "private-bucket" not in diagnostic
    assert exc.value.__cause__ is None


def test_managed_upload_failure_rolls_back_and_is_sanitized(tmp_path):
    s3 = FakeS3Client()
    worker = _worker(tmp_path, s3_client=s3)
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.txt").write_text("a")
    (results / "b.txt").write_text("b")
    original_upload = s3.upload_file

    def managed_upload(filename, bucket, key):
        if key.endswith("/b.txt"):
            raise S3UploadFailedError(f"private provider detail for {key}")
        original_upload(filename, bucket, key)

    s3.upload_file = managed_upload

    with pytest.raises(RuntimeError, match="result upload failed") as exc:
        worker.upload_results_to_s3(results, "private/account-prefix")

    assert s3.objects == {}
    assert s3.deleted == [("private-bucket", s3.uploads[0][2])]
    assert "account-prefix" not in str(exc.value)
    assert exc.value.__cause__ is None


def test_result_size_mismatch_removes_uploaded_object(tmp_path):
    s3 = FakeS3Client(reported_size=999)
    worker = _worker(tmp_path, s3_client=s3)
    results = tmp_path / "results"
    results.mkdir()
    (results / "result.bin").write_bytes(b"data")

    with pytest.raises(RuntimeError, match="result upload failed"):
        worker.upload_results_to_s3(results, "prefix")

    assert s3.objects == {}
    assert s3.deleted == [("private-bucket", f"prefix/results/{'a' * 32}/result.bin")]


@pytest.mark.parametrize("prefix", ["", "../escape", "/absolute", r"windows\escape"])
def test_result_upload_rejects_unsafe_prefixes(tmp_path, prefix):
    worker = _worker(tmp_path, s3_client=FakeS3Client())
    results = tmp_path / "results"
    results.mkdir()

    with pytest.raises(ValueError, match="relative prefix"):
        worker.upload_results_to_s3(results, prefix)


def test_result_upload_rejects_symlinks_before_upload(tmp_path):
    s3 = FakeS3Client()
    worker = _worker(tmp_path, s3_client=s3)
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "private-outside.txt"
    outside.write_text("private")
    (results / "linked.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic links"):
        worker.upload_results_to_s3(results, "prefix")

    assert s3.uploads == []


def test_result_upload_rejects_empty_result_directory(tmp_path):
    worker = _worker(tmp_path, s3_client=FakeS3Client())
    results = tmp_path / "results"
    results.mkdir()

    with pytest.raises(ValueError, match="contains no result files"):
        worker.upload_results_to_s3(results, "prefix")


def test_cleanup_is_idempotent(tmp_path):
    worker = _worker(tmp_path)

    worker.cleanup()
    worker.cleanup()

    assert not worker.local_base.exists()


def test_separate_workers_own_unique_temporary_roots():
    first = batch_worker.BatchWorker("", connect_aws=False)
    second = batch_worker.BatchWorker("", connect_aws=False)

    try:
        assert first.local_base != second.local_base
        assert first.local_base.is_dir()
        assert second.local_base.is_dir()
    finally:
        first.cleanup()
        second.cleanup()


def test_task_cleanup_removes_only_unique_owned_root(tmp_path):
    worker = batch_worker.BatchWorker("", connect_aws=False)
    owned_root = worker.local_base
    external_simulation = tmp_path / "simulation"
    external_job = tmp_path / "job"
    external_simulation.mkdir()
    external_job.mkdir()
    worker.simulation_dir = external_simulation
    worker.job_dir = external_job

    worker.cleanup()

    assert not owned_root.exists()
    assert external_simulation.is_dir()
    assert external_job.is_dir()


def test_repeated_uploads_use_distinct_result_prefixes(tmp_path):
    upload_ids = iter(["a" * 32, "b" * 32])
    s3 = FakeS3Client()
    worker = batch_worker.BatchWorker(
        "private-bucket",
        local_base=tmp_path / "worker",
        s3_client=s3,
        batch_client=FakeBatchClient(),
        connect_aws=False,
        upload_id_factory=lambda: next(upload_ids),
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "result.txt").write_text("result")

    first = worker.upload_results_to_s3(results, "prefix")
    second = worker.upload_results_to_s3(results, "prefix")

    assert first == f"prefix/results/{'a' * 32}"
    assert second == f"prefix/results/{'b' * 32}"
    assert len(s3.objects) == 2


class FakeCliWorker:
    def __init__(self, bucket, region, connect_aws):
        self.initialized = (bucket, region, connect_aws)
        self.local_base = Path("/unused/worker")
        self.simulation_dir = self.local_base / "simulation"
        self.job_dir = self.local_base / "job"
        self.task_result = True
        self.task_calls = []
        self.downloads = []
        self.cleaned = 0

    def download_from_s3(self, key, path):
        self.downloads.append((key, path))

    def run_preliminary_analysis(self):
        return {"tasks_required": 1}

    def submit_task_jobs(self, analysis, queue, definition):
        return ["job-1"]

    def run_simulation_task(self, task_id, analysis):
        self.task_calls.append((task_id, analysis))
        return self.task_result

    def cleanup(self):
        self.cleaned += 1


def test_task_cli_avoids_aws_clients_and_always_cleans(tmp_path):
    created = []

    def factory(*args, **kwargs):
        worker = FakeCliWorker(*args, **kwargs)
        created.append(worker)
        return worker

    exit_code = batch_worker.main(
        [
            "--mode",
            "task",
            "--task-id",
            "0",
            "--simulation-dir",
            str(tmp_path / "simulation"),
            "--job-dir",
            str(tmp_path / "job"),
        ],
        worker_factory=factory,
    )

    assert exit_code == 0
    assert created[0].initialized == ("", "us-east-1", False)
    assert created[0].task_calls == [(0, {})]
    assert created[0].cleaned == 1


def test_task_cli_maps_nonzero_worker_result_and_cleans(tmp_path):
    worker = FakeCliWorker("", "us-east-1", False)
    worker.task_result = False

    exit_code = batch_worker.main(
        [
            "--mode",
            "task",
            "--task-id",
            "2",
            "--simulation-dir",
            str(tmp_path / "simulation"),
            "--job-dir",
            str(tmp_path / "job"),
        ],
        worker_factory=lambda *args, **kwargs: worker,
    )

    assert exit_code == 1
    assert worker.cleaned == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "main", "--simulation-key", "sim", "--job-key", "job"],
        ["--mode", "main", "--bucket", "bucket"],
        ["--mode", "task", "--task-id", "0"],
    ],
)
def test_cli_rejects_incomplete_mode_inputs_before_constructing_worker(argv):
    with pytest.raises(SystemExit) as exc_info:
        batch_worker.main(
            argv,
            worker_factory=lambda *args, **kwargs: pytest.fail(
                "worker was constructed"
            ),
        )

    assert exc_info.value.code == 2


def test_main_cli_downloads_inputs_and_cleans():
    worker = FakeCliWorker("bucket", "us-west-1", True)

    exit_code = batch_worker.main(
        [
            "--mode",
            "main",
            "--bucket",
            "bucket",
            "--region",
            "us-west-1",
            "--simulation-key",
            "private/simulation",
            "--job-key",
            "private/job",
        ],
        worker_factory=lambda *args, **kwargs: worker,
    )

    assert exit_code == 0
    assert worker.downloads == [
        ("private/simulation", worker.simulation_dir),
        ("private/job", worker.job_dir),
    ]
    assert worker.cleaned == 1


def test_cli_failure_diagnostic_does_not_echo_private_exception(
    caplog,
):
    worker = FakeCliWorker("bucket", "us-east-1", True)

    def fail(*args, **kwargs):
        raise RuntimeError("token=secret account=private-user object=private/key")

    worker.download_from_s3 = fail
    with caplog.at_level(logging.ERROR, logger=batch_worker.__name__):
        exit_code = batch_worker.main(
            [
                "--mode",
                "main",
                "--bucket",
                "bucket",
                "--simulation-key",
                "simulation",
                "--job-key",
                "job",
            ],
            worker_factory=lambda *args, **kwargs: worker,
        )

    assert exit_code == 1
    assert worker.cleaned == 1
    assert "RuntimeError" in caplog.text
    for secret in ("secret", "private-user", "private/key"):
        assert secret not in caplog.text
