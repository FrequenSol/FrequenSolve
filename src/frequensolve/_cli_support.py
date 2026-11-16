"""Shared CLI diagnostics without importing optional command implementations."""

import click

from frequensolve._platform import platform_support_guidance


class SupportedHostGroup(click.Group):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not ctx.resilient_parsing and (guidance := platform_support_guidance()):
            click.echo(guidance, err=True)
        return super().parse_args(ctx, args)
