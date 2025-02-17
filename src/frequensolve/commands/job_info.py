import sys
from pathlib import Path

import click

from frequensolve.simulation.jobs import SimulationJob


def get_job_size(job_file: str) -> int:
    """Gets the number of frequencies in a job file."""
    job_file = Path(job_file)
    if not job_file.exists():
        raise FileNotFoundError(f"Job file {job_file} does not exist")
    job = SimulationJob.load(job_file)
    return len(job.f_list)


@click.command()
@click.option("-j", "--job_file", "job_file", required=True, help="Path to jobfile")
def main(job_file):
    """Command line tool to get the number of frequencies in a job file."""
    try:
        size = get_job_size(job_file)
        print(size)
    except Exception as e:
        print(f"Error reading job file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
