"""Find or validate an exact-commit successful FrequenSolve CI run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from typing import Any
from urllib.parse import urlencode

WORKFLOW_PATH = ".github/workflows/cicd-workflow.yml"
REQUIRED_JOB = "Required CI"
EXACT_TREE_EVENTS = frozenset({"push", "workflow_dispatch"})
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def gh_api(endpoint: str) -> Any:
    """Read JSON from the GitHub REST API through the authenticated gh CLI."""
    result = subprocess.run(
        ["gh", "api", endpoint],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_matches(run: dict[str, Any], commit: str) -> bool:
    """Return whether a successful run tested the exact requested commit tree.

    GitHub records a pull-request run's ``head_sha`` as the contributor branch
    head even though ``actions/checkout`` tests the synthetic
    ``refs/pull/<number>/merge`` commit.  Those runs therefore cannot prove
    that the requested commit itself passed and must not authorize a release.
    """
    return (
        run.get("head_sha") == commit
        and run.get("conclusion") == "success"
        and run.get("path") == WORKFLOW_PATH
        and run.get("event") in EXACT_TREE_EVENTS
    )


def has_required_job(jobs: list[dict[str, Any]]) -> bool:
    """Return whether the stable aggregate job succeeded."""
    return any(
        job.get("name") == REQUIRED_JOB and job.get("conclusion") == "success"
        for job in jobs
    )


def validate_run(repository: str, commit: str, run: dict[str, Any]) -> dict[str, Any]:
    """Fail unless a run and its stable aggregate prove exact-SHA CI success."""
    if not run_matches(run, commit):
        raise ValueError(
            f"CI run {run.get('id')} is not a successful {WORKFLOW_PATH} run "
            f"for {commit} from an exact-tree event "
            f"({', '.join(sorted(EXACT_TREE_EVENTS))}); got {run.get('event')!r}"
        )

    run_id = int(run["id"])
    jobs_payload = gh_api(f"repos/{repository}/actions/runs/{run_id}/jobs?per_page=100")
    if not has_required_job(jobs_payload.get("jobs", [])):
        raise ValueError(f"CI run {run_id} lacks a successful {REQUIRED_JOB} job")

    return {
        "commit": commit,
        "runId": run_id,
        "runUrl": run["html_url"],
        "workflow": WORKFLOW_PATH,
        "requiredJob": REQUIRED_JOB,
    }


def find_run(repository: str, commit: str) -> dict[str, Any]:
    """Return the newest successful exact-SHA run with the aggregate gate."""
    query = urlencode({"head_sha": commit, "status": "success", "per_page": 100})
    payload = gh_api(
        f"repos/{repository}/actions/workflows/cicd-workflow.yml/runs?{query}"
    )
    candidates = sorted(
        payload.get("workflow_runs", []), key=lambda run: int(run["id"]), reverse=True
    )
    errors: list[str] = []
    for run in candidates:
        try:
            return validate_run(repository, commit, run)
        except ValueError as exc:
            errors.append(str(exc))
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise ValueError(f"no successful exact-SHA {REQUIRED_JOB} run for {commit}{detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", type=int)
    args = parser.parse_args()

    if not SHA_RE.fullmatch(args.commit):
        parser.error("--commit must be a lowercase 40-character Git SHA")

    try:
        if args.run_id is None:
            evidence = find_run(args.repository, args.commit)
        else:
            run = gh_api(f"repos/{args.repository}/actions/runs/{args.run_id}")
            evidence = validate_run(args.repository, args.commit, run)
    except (subprocess.CalledProcessError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
