"""Wait for one exact private DockerImage workflow run in a bounded window."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

WORKFLOW_REPOSITORY = "FrequenSol/FrequenSolveDockerImage"
PENDING_STATUSES = {"in_progress", "pending", "queued", "requested", "waiting"}


def completion_state(run: dict[str, Any]) -> bool:
    """Return whether a run succeeded, rejecting failed or unknown states."""
    status = run.get("status")
    conclusion = run.get("conclusion")
    if status == "completed":
        if conclusion != "success":
            raise ValueError(
                "DockerImage run completed unsuccessfully; "
                f"got conclusion={conclusion!r}"
            )
        return True
    if status in PENDING_STATUSES and conclusion in (None, ""):
        return False
    raise ValueError(
        "DockerImage run returned an unexpected state; "
        f"got status={status!r}, conclusion={conclusion!r}"
    )


def wait_for_completion(
    fetch: Callable[[], dict[str, Any]],
    *,
    timeout_seconds: int,
    interval_seconds: int,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll until success or the bounded token window elapses."""
    deadline = monotonic() + timeout_seconds
    while True:
        if completion_state(fetch()):
            return True
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleep(min(interval_seconds, remaining))


def fetch_run(run_id: int) -> dict[str, Any]:
    """Fetch one workflow run with the GitHub CLI and the current token."""
    if not os.environ.get("GH_TOKEN"):
        raise ValueError("GH_TOKEN is required")
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{WORKFLOW_REPOSITORY}/actions/runs/{run_id}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("GitHub workflow run response must be an object")
    if payload.get("id") != run_id:
        raise ValueError("GitHub workflow run response returned the wrong run ID")
    repository = payload.get("repository") or {}
    if repository.get("full_name") != WORKFLOW_REPOSITORY:
        raise ValueError("GitHub workflow run response returned the wrong repository")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--interval-seconds", default=15, type=int)
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args()

    if args.run_id <= 0:
        parser.error("run ID must be positive")
    if not 1 <= args.timeout_seconds <= 3000:
        parser.error("timeout must be between 1 and 3000 seconds")
    if not 1 <= args.interval_seconds <= 60:
        parser.error("interval must be between 1 and 60 seconds")

    try:
        completed = wait_for_completion(
            lambda: fetch_run(args.run_id),
            timeout_seconds=args.timeout_seconds,
            interval_seconds=args.interval_seconds,
        )
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"completed={str(completed).lower()}\n")
    except (
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    if completed:
        print(f"DockerImage run {args.run_id} completed successfully.")
    else:
        print(f"DockerImage run {args.run_id} is still pending after this window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
