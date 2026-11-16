"""Dependency-free signals for read-only run monitoring."""


class TransientStatusReadError(RuntimeError):
    """A status read failed transiently; the submitted run may still be active."""
