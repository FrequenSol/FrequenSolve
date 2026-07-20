"""Top-level FrequenSolve command-line interface."""

import click

from frequensolve.commands.site import site


@click.group()
@click.version_option(package_name="frequensolve")
def main() -> None:
    """Configure and inspect FrequenSolve from the command line."""


main.add_command(site)


if __name__ == "__main__":  # pragma: no cover
    main()
