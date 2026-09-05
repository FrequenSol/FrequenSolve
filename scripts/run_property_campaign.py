#!/usr/bin/env python3
"""Run the opt-in SDK property campaign with a retained seed and hard cap.

This runner owns only FrequenSolve's Python model, validation, serialization,
and filesystem-safety boundary. Native solver-parser fuzzing remains in
FrequenSol/Sauce#53.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_MAX_SECONDS = 900
TERMINATION_GRACE_SECONDS = 5


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _seed(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be zero or greater")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-seconds",
        type=_positive_int,
        default=_positive_int(
            os.environ.get(
                "FREQUENSOLVE_PROPERTY_CAMPAIGN_MAX_SECONDS",
                str(DEFAULT_MAX_SECONDS),
            )
        ),
        help="hard wall-clock cap for the entire pytest process group",
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=None,
        help="Hypothesis seed to replay; a random seed is recorded when omitted",
    )
    return parser


def _stop_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _write_metadata(root: Path, payload: dict) -> Path:
    output_dir = root / ".hypothesis" / "frequensolve-campaigns"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["started_at"].replace(":", "").replace("-", "")
    destination = output_dir / f"{stamp}-{payload['seed']}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-ra",
        "-o",
        "addopts=",
        "--strict-markers",
        "-m",
        "property_contract",
        f"--hypothesis-seed={seed}",
        "tests",
    ]
    environment = dict(os.environ)
    environment["FREQUENSOLVE_HYPOTHESIS_PROFILE"] = "campaign"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(root / "src"),
                environment.get("PYTHONPATH"),
            ),
        )
    )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    status = "failed"

    print(
        "FrequenSolve property campaign: "
        f"profile=campaign seed={seed} max_seconds={args.max_seconds}"
    )
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=args.max_seconds)
        status = "passed" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "timed_out"
        return_code = 124
        _stop_process_group(process)
    except KeyboardInterrupt:
        status = "interrupted"
        return_code = 130
        _stop_process_group(process)

    metadata = {
        "command": command,
        "corpus_directory": ".hypothesis",
        "duration_seconds": round(time.monotonic() - started, 3),
        "max_seconds": args.max_seconds,
        "profile": "campaign",
        "return_code": return_code,
        "seed": seed,
        "started_at": started_at,
        "status": status,
    }
    metadata_path = _write_metadata(root, metadata)
    print(f"Campaign metadata: {metadata_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(run())
