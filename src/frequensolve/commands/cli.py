"""Top-level FrequenSolve command-line interface."""

import click

from frequensolve._cli_support import SupportedHostGroup
from frequensolve.commands.site import site


@click.group(cls=SupportedHostGroup)
@click.version_option(package_name="frequensolve")
def main() -> None:
    """Configure and inspect FrequenSolve on Linux or macOS.

    Native Windows is unsupported; run Python inside WSL2, a Linux container,
    or a remote Linux/macOS host.
    """


main.add_command(site)


if __name__ == "__main__":  # pragma: no cover
    main()
