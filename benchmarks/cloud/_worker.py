from __future__ import annotations

import argparse
import os
import runpy
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from ._shared import CASE_SCHEMA, sanitize, utc_now, write_json

TERMINAL_SUCCESS = {"SUCCEEDED", "COMPLETED", "COMPLETE", "SUCCESS"}


def _iso_seconds(start: object, end: object) -> float | None:
    from datetime import datetime

    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        return (
            datetime.fromisoformat(end.replace("Z", "+00:00"))
            - datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds()
    except ValueError:
        return None


def _cloud_diagnostics(site: Any, simulation_id: str) -> dict[str, Any]:
    client = getattr(site, "graphql_client", None)
    if client is None:
        return {"instrumentationErrors": ["Selected site has no Cloud GraphQL client"]}
    diagnostics: dict[str, Any] = {"instrumentationErrors": []}
    try:
        diagnostics["status"] = client.get_simulation_status_details(simulation_id)
    except Exception as error:
        diagnostics["instrumentationErrors"].append(
            f"status details: {type(error).__name__}: {error}"
        )
    query = """
      query BenchmarkSimulation($id: ID!) {
        getSimulation(id: $id) {
          id status startTime planningStartedAt planningCompletedAt duration terminalAt
          totalFrequencies completedFrequencies failedFrequencies abortedFrequencies
          failurePhase failureCategory failureCode failureMessage
          creditSettlementMode creditSettlementStatus creditSettlementAmount
          executionBackend executionTarget providerAttemptId
          slurmPartition slurmNodes slurmRanksPerNode slurmWallTimeSeconds
          frequencyJobs(limit: 100) { items {
            frequencyIndex frequency frequencyImag status errorMessage startTime endTime duration
            batchJobId requestedComputeMode effectiveComputeMode attemptNumber autoEscalated
            failureCategory failureCode queueName queueWaitSeconds runSeconds
            creditAuthorizationWaitSeconds activeWorkSeconds stateDurations
          } nextToken }
          batchAttempts(limit: 100) { items {
            frequencyIndex attemptNumber status statusReason submittedAt startedAt stoppedAt
            queueWaitSeconds runSeconds stateDurations effectiveComputeMode
            runtimeQueueMilliseconds batchRunningUpperBoundMilliseconds
          } nextToken }
          executionAttempts(limit: 100) { items {
            executionBackend providerAttemptId frequencyIndex attemptNumber status statusReason
            nodes ranksPerNode submittedAt gatewayReceivedAt plannerSubmittedAt plannerCompletedAt
            creditAuthorizedAt queuedAt solverStartedAt solverStoppedAt terminalAt
            activeWorkMilliseconds
          } nextToken }
        }
      }
    """
    try:
        payload = client.execute(query, {"id": simulation_id}).get("getSimulation")
        if isinstance(payload, Mapping):
            diagnostics["provider"] = dict(payload)
            if isinstance(payload.get("frequencyJobs"), Mapping):
                diagnostics["frequencyJobs"] = payload["frequencyJobs"].get("items", [])
            if isinstance(payload.get("batchAttempts"), Mapping):
                diagnostics["batchAttempts"] = payload["batchAttempts"].get("items", [])
            if isinstance(payload.get("executionAttempts"), Mapping):
                diagnostics["executionAttempts"] = payload["executionAttempts"].get(
                    "items", []
                )
            for relation in ("frequencyJobs", "batchAttempts", "executionAttempts"):
                page = payload.get(relation)
                if isinstance(page, Mapping) and page.get("nextToken"):
                    diagnostics["instrumentationErrors"].append(
                        f"{relation} exceeded the 100-row benchmark snapshot"
                    )
    except Exception as error:
        diagnostics["instrumentationErrors"].append(
            f"provider details: {type(error).__name__}: {error}"
        )
    diagnostics["instrumentationErrors"] = diagnostics["instrumentationErrors"] or []
    return sanitize(diagnostics)


def _derived_metrics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    provider = diagnostics.get("provider")
    if not isinstance(provider, Mapping):
        return {}
    frequencies = diagnostics.get("frequencyJobs")
    attempts = diagnostics.get("executionAttempts")
    metrics: dict[str, Any] = {
        "planningSeconds": _iso_seconds(
            provider.get("planningStartedAt"), provider.get("planningCompletedAt")
        ),
        "serverTotalSeconds": _iso_seconds(
            provider.get("startTime"), provider.get("terminalAt")
        ),
    }
    if isinstance(frequencies, list):
        terminal_times = [
            item.get("endTime") for item in frequencies if isinstance(item, Mapping)
        ]
        terminal_times = [value for value in terminal_times if isinstance(value, str)]
        if terminal_times:
            metrics["packingAndProjectionSeconds"] = _iso_seconds(
                max(terminal_times), provider.get("terminalAt")
            )
        metrics["frequencyQueueSeconds"] = [
            item["queueWaitSeconds"]
            for item in frequencies
            if isinstance(item, Mapping)
            and isinstance(item.get("queueWaitSeconds"), (int, float))
        ]
        metrics["frequencyActiveWorkSeconds"] = [
            item["activeWorkSeconds"]
            for item in frequencies
            if isinstance(item, Mapping)
            and isinstance(item.get("activeWorkSeconds"), (int, float))
        ]
    if isinstance(attempts, list):
        metrics["gatewaySeconds"] = [
            value
            for item in attempts
            if isinstance(item, Mapping)
            for value in [
                _iso_seconds(item.get("submittedAt"), item.get("gatewayReceivedAt"))
            ]
            if value is not None
        ]
        metrics["schedulerQueueSeconds"] = [
            value
            for item in attempts
            if isinstance(item, Mapping)
            for value in [
                _iso_seconds(item.get("queuedAt"), item.get("solverStartedAt"))
            ]
            if value is not None
        ]
    return metrics


class Recorder:
    def __init__(
        self,
        profile: str,
        email: str | None,
        declared_backend: str,
        run_id: str,
        case_id: str,
    ):
        import frequensolve as fs

        self.fs = fs
        self.profile = profile
        self.email = email
        self.declared_backend = declared_backend
        self.run_id = run_id
        self.case_id = case_id
        self.original_site = fs.Site
        self.original_project_post_init = fs.Project.__post_init__
        self.submissions: list[dict[str, Any]] = []
        self.site_adaptations: list[dict[str, Any]] = []

    def site_factory(self, *args: Any, **kwargs: Any) -> Any:
        selected: dict[str, Any] = {"profile": self.profile}
        if self.email:
            selected["email"] = self.email
        if "interactive" in kwargs:
            selected["interactive"] = kwargs["interactive"]
        self.site_adaptations.append(
            {"requestedProfile": kwargs.get("profile"), "selectedProfile": self.profile}
        )
        site = self.original_site(**selected)
        original_submit = site.submit

        def submit(job: Any, *submit_args: Any, **submit_kwargs: Any) -> Any:
            for legacy_key in ("duration", "queue", "nodes", "ranks_per_node"):
                submit_kwargs.pop(legacy_key, None)
            submit_kwargs["force"] = True
            record: dict[str, Any] = {
                "index": len(self.submissions),
                "job": getattr(job, "name", None),
                "submittedAt": utc_now(),
                "status": "SUBMITTING",
            }
            started = time.monotonic()
            self.submissions.append(record)
            try:
                handle = original_submit(job, *submit_args, **submit_kwargs)
            except Exception as error:
                record.update(
                    {
                        "status": "SUBMISSION_FAILED",
                        "acceptSeconds": time.monotonic() - started,
                        "failure": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                raise
            record.update(
                {
                    "simulationId": str(handle.id),
                    "acceptedAt": utc_now(),
                    "acceptSeconds": time.monotonic() - started,
                    "backend": sanitize(dict(getattr(handle, "backend", {}) or {})),
                    "status": "ACCEPTED",
                }
            )
            original_wait = handle.wait

            def wait(*wait_args: Any, **wait_kwargs: Any) -> Any:
                record["waitStartedAt"] = utc_now()
                wait_started = time.monotonic()
                wait_seconds: float | None = None
                terminal_observed_at: str | None = None
                try:
                    result = original_wait(*wait_args, **wait_kwargs)
                    wait_seconds = time.monotonic() - wait_started
                    terminal_observed_at = utc_now()
                    result.raise_for_status()
                    instrumentation_started = time.monotonic()
                    diagnostics = _cloud_diagnostics(site, str(handle.id))
                    record.update(
                        {
                            "terminalObservedAt": terminal_observed_at,
                            "waitSeconds": wait_seconds,
                            "instrumentationSeconds": time.monotonic()
                            - instrumentation_started,
                            "backend": sanitize(
                                dict(getattr(handle, "backend", {}) or {})
                            ),
                            "diagnostics": diagnostics,
                            "providerMetrics": _derived_metrics(diagnostics),
                            "status": "PASSED",
                        }
                    )
                    return result
                except Exception as error:
                    if wait_seconds is None:
                        wait_seconds = time.monotonic() - wait_started
                    if terminal_observed_at is None:
                        terminal_observed_at = utc_now()
                    instrumentation_started = time.monotonic()
                    diagnostics = _cloud_diagnostics(site, str(handle.id))
                    attached = getattr(error, "result", None)
                    raw_status = getattr(getattr(attached, "status", None), "raw", None)
                    record.update(
                        {
                            "terminalObservedAt": terminal_observed_at,
                            "waitSeconds": wait_seconds,
                            "instrumentationSeconds": time.monotonic()
                            - instrumentation_started,
                            "backend": sanitize(
                                dict(getattr(handle, "backend", {}) or {})
                            ),
                            "diagnostics": diagnostics,
                            "providerMetrics": _derived_metrics(diagnostics),
                            "status": "FAILED",
                            "failure": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "rawStatus": sanitize(raw_status),
                            },
                        }
                    )
                    raise

            handle.wait = wait
            return handle

        site.submit = submit
        return site

    def install(self) -> None:
        self.fs.Site = self.site_factory

        original_project_post_init = self.original_project_post_init
        prefix = (
            "cloud-benchmark-"
            + "".join(
                character.lower()
                for character in f"{self.run_id}-{self.case_id}"
                if character.isalnum()
            )[-32:]
        )

        def benchmark_project_post_init(project: Any) -> None:
            from pathlib import Path

            path = Path(project.path)
            if path.name != prefix:
                project.path = path / prefix
            original_project_post_init(project)

        self.fs.Project.__post_init__ = benchmark_project_post_init

    def restore(self) -> None:
        self.fs.Site = self.original_site
        self.fs.Project.__post_init__ = self.original_project_post_init


def execute(args: argparse.Namespace) -> int:
    result_path = Path(args.result).resolve()
    workload = Path(args.workload).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    started_at = utc_now()
    started = time.monotonic()
    recorder = Recorder(
        args.profile, args.email, args.backend, args.run_id, args.case_id
    )
    result: dict[str, Any] = {
        "schema": CASE_SCHEMA,
        "caseId": args.case_id,
        "profile": args.profile,
        "declaredBackend": args.backend,
        "expectedSubmissions": args.expected_submissions,
        "startedAt": started_at,
        "status": "FAIL",
    }
    try:
        recorder.install()
        runpy.run_path(str(workload), run_name="__cloud_benchmark__")
        if len(recorder.submissions) != args.expected_submissions:
            raise RuntimeError(
                f"Expected {args.expected_submissions} submissions but observed "
                f"{len(recorder.submissions)}"
            )
        incomplete = [
            entry for entry in recorder.submissions if entry.get("status") != "PASSED"
        ]
        if incomplete:
            raise RuntimeError(
                f"{len(incomplete)} benchmark submissions did not reach a successful terminal result"
            )
        actual_backends = {
            str(entry.get("backend", {}).get("executionBackend", "")).lower()
            for entry in recorder.submissions
        }
        actual_backends.discard("")
        if actual_backends and actual_backends != {args.backend.lower()}:
            raise RuntimeError(
                f"Profile routed to {sorted(actual_backends)}, expected {args.backend}"
            )
        result["status"] = "PASS"
    except Exception as error:
        result["failure"] = {
            "phase": "setup" if not recorder.submissions else "execution",
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        recorder.restore()
    result.update(
        {
            "finishedAt": utc_now(),
            "totalSeconds": time.monotonic() - started,
            "siteAdaptations": recorder.site_adaptations,
            "submissions": recorder.submissions,
        }
    )
    measured_submission_seconds = sum(
        float(submission.get("acceptSeconds", 0))
        + float(submission.get("waitSeconds", 0))
        + float(submission.get("instrumentationSeconds", 0))
        for submission in recorder.submissions
    )
    result["caseMetrics"] = {
        "outsideMeasuredSubmissionSeconds": max(
            float(result["totalSeconds"]) - measured_submission_seconds, 0.0
        )
    }
    write_json(result_path, sanitize(result))
    return 0 if result["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--backend", choices=("batch", "slurm"), required=True)
    parser.add_argument("--expected-submissions", type=int, required=True)
    parser.add_argument("--email")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
