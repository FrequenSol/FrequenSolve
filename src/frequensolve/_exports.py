"""Internal helpers for package export aggregation."""

from __future__ import annotations

from collections.abc import Iterable


def unique_exports(*groups: Iterable[str]) -> list[str]:
    """Return export names in declaration order without duplicates.

    Args:
        *groups: Export-name iterables to concatenate.

    Returns:
        Ordered list containing the first occurrence of each name.
    """

    seen = set()
    exports = []
    for group in groups:
        for name in group:
            if name not in seen:
                exports.append(name)
                seen.add(name)
    return exports
