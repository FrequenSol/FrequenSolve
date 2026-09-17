"""Top-level FrequenSolve command-line interface."""

import click

from frequensolve._platform import platform_support_guidance
from frequensolve.commands.site import site


class _SupportedHostGroup(click.Group):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not ctx.resilient_parsing and (guidance := platform_support_guidance()):
            click.echo(guidance, err=True)
        return super().parse_args(ctx, args)


@click.group(cls=_SupportedHostGroup)
@click.version_option(package_name="frequensolve")
def main() -> None:
    """Configure and inspect FrequenSolve on Linux or macOS.

    Native Windows is unsupported; run Python inside WSL2, a Linux container,
    or a remote Linux/macOS host.
    """


main.add_command(site)


if __name__ == "__main__":  # pragma: no cover
    main()
