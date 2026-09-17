"""Top-level FrequenSolve command-line interface."""

import click

from frequensolve._platform import platform_support_guidance
from frequensolve.commands.site import site


@click.group()
@click.version_option(package_name="frequensolve")
def main() -> None:
    """Configure and inspect FrequenSolve on Linux or macOS.

    Native Windows is unsupported; run Python inside WSL2, a Linux container,
    or a remote Linux/macOS host.
    """
    if guidance := platform_support_guidance():
        click.echo(guidance, err=True)


main.add_command(site)


if __name__ == "__main__":  # pragma: no cover
    main()
