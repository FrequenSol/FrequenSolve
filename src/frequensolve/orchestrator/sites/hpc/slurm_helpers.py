"""Small helpers shared by SLURM-backed site implementations."""

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

SLURM_QUEUE_STATES = {
    "PD": "pending",
    "R": "running",
    "CG": "running",
    "CD": "complete",
    "F": "failed",
    "TO": "timeout",
    "CA": "cancelled",
}

SLURM_ACCOUNTING_STATES = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "complete",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "PREEMPTED": "failed",
    "TIMEOUT": "timeout",
    "CANCELLED": "cancelled",
}

SLURM_STATUS_COLORS = {
    "pending": "\033[38;5;27m",
    "running": "\033[38;5;28m",
    "complete": "\033[38;5;40m",
    "timeout": "\033[38;5;202m",
    "failed": "\033[38;5;160m",
    "cancelled": "\033[38;5;160m",
    "unknown": "\033[38;5;244m",
}


def hms_to_seconds(hms: str) -> int:
    """Convert HH:MM:SS or D-HH:MM:SS to seconds."""

    if "-" in hms:
        d, tmp = hms.split("-", 1)
        days = int(d)
    else:
        days = 0
        tmp = hms
    h, m, s = map(int, tmp.split(":"))
    return days * 86400 + h * 3600 + m * 60 + s


def seconds_to_hms(seconds: int) -> str:
    """Convert seconds to D-HH:MM:SS."""

    days = seconds // 86400
    seconds %= 86400
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def read_stream(stream) -> str:
    """Read a Paramiko/subprocess stream as stripped text."""

    output = stream.read()
    if isinstance(output, bytes):
        return output.decode().strip()
    return output.strip()


def normalize_slurm_state(state: str) -> str:
    """Map a SLURM queue/accounting state to the public PoolInfo status."""

    state = (state or "").strip()
    if not state:
        return "unknown"
    token = state.split("|")[-1].split()[0].upper()
    return SLURM_QUEUE_STATES.get(token, SLURM_ACCOUNTING_STATES.get(token, "unknown"))


def parse_sbatch_job_id(output: str) -> str:
    """Extract the job id from sbatch output."""

    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        raise ValueError(f"failed to get job ID from sbatch output: {output}")
    return match.group(1)


def as_list(value, item_type) -> Tuple[List[Any], bool]:
    """Return (items, was_single) for public APIs accepting one item or a list."""

    if isinstance(value, item_type):
        return [value], True
    return list(value), False


@contextmanager
def temporary_text_file(
    text: str,
    *,
    suffix: str,
    prefix: str,
    directory: Optional[Union[str, Path]] = None,
):
    """Create a temporary executable text file and remove it afterwards."""

    temporary_directory = None
    if directory is not None:
        temporary_directory = Path(directory).expanduser()
        temporary_directory.mkdir(parents=True, exist_ok=True)
    fd, script_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=prefix,
        dir=temporary_directory,
    )
    path = Path(script_path)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(path, 0o700)
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
