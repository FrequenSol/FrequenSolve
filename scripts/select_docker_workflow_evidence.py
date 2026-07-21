"""Select an exact DockerImage workflow run and its evidence artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKFLOW_REPOSITORY = "FrequenSol/FrequenSolveDockerImage"
WORKFLOW_PATH = ".github/workflows/cicd-workflow.yml"
WORKFLOW_ID = 163413973
EVIDENCE_ARTIFACT = "frequensolve-test-evidence"


def _timestamp(value: str) -> datetime:
    """Parse one GitHub timestamp as an aware UTC datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("GitHub timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def matching_runs(
    payload: dict[str, Any],
    *,
    request_id: str,
    actor: str,
    head_sha: str,
    created_after: str,
) -> list[dict[str, Any]]:
    """Return runs that exactly match one uniquely correlated dispatch."""
    threshold = _timestamp(created_after)
    expected_title = f"Runtime image {request_id}"
    matches = []
    for run in payload.get("workflow_runs", []):
        repository = run.get("repository") or {}
        run_actor = run.get("actor") or {}
        try:
            created_at = _timestamp(run.get("created_at", ""))
        except (TypeError, ValueError):
            continue
        if (
            run.get("display_title") == expected_title
            and run.get("event") == "workflow_dispatch"
            and run.get("path") == WORKFLOW_PATH
            and run.get("workflow_id") == WORKFLOW_ID
            and run.get("head_branch") == "main"
            and run.get("head_sha") == head_sha
            and run_actor.get("login") == actor
            and repository.get("full_name") == WORKFLOW_REPOSITORY
            and created_at >= threshold
        ):
            matches.append(run)
    return sorted(matches, key=lambda run: (run["created_at"], run["id"]))


def select_run(
    payload: dict[str, Any],
    *,
    request_id: str,
    actor: str,
    head_sha: str,
    created_after: str,
    require_success: bool = False,
) -> dict[str, Any] | None:
    """Select one exact run, rejecting ambiguity and unsuccessful completion."""
    matches = matching_runs(
        payload,
        request_id=request_id,
        actor=actor,
        head_sha=head_sha,
        created_after=created_after,
    )
    if len(matches) > 1:
        raise ValueError(f"dispatch matched {len(matches)} DockerImage runs")
    if not matches:
        return None
    run = matches[0]
    if require_success and (
        run.get("status") != "completed" or run.get("conclusion") != "success"
    ):
        raise ValueError(
            "DockerImage run must be completed successfully; "
            f"got status={run.get('status')!r}, conclusion={run.get('conclusion')!r}"
        )
    return run


def select_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the one live heavy-evidence artifact from the selected run."""
    artifacts = [
        artifact
        for artifact in payload.get("artifacts", [])
        if artifact.get("name") == EVIDENCE_ARTIFACT
        and artifact.get("expired") is False
    ]
    if len(artifacts) != 1:
        raise ValueError(
            f"expected one live {EVIDENCE_ARTIFACT} artifact, got {len(artifacts)}"
        )
    return artifacts[0]


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("payload", type=Path)
    run_parser.add_argument("--request-id", required=True)
    run_parser.add_argument("--actor", required=True)
    run_parser.add_argument("--head-sha", required=True)
    run_parser.add_argument("--created-after", required=True)
    run_parser.add_argument("--require-success", action="store_true")
    run_parser.add_argument("--allow-missing", action="store_true")

    artifact_parser = subparsers.add_parser("artifact")
    artifact_parser.add_argument("payload", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "run":
            run = select_run(
                _load(args.payload),
                request_id=args.request_id,
                actor=args.actor,
                head_sha=args.head_sha,
                created_after=args.created_after,
                require_success=args.require_success,
            )
            if run is None:
                return 3 if args.allow_missing else parser.error("no matching run")
            print(run["id"])
        else:
            print(select_artifact(_load(args.payload))["id"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
