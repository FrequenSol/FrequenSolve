"""Supported-host guidance without importing optional runtime dependencies."""

import sys

WINDOWS_GUIDANCE = (
    "Native Windows is unsupported for this release. Run the FrequenSolve "
    "Python environment inside WSL2, a Linux container, or a remote Linux/macOS "
    "host. Selecting a remote execution site from native Windows does not "
    "make the local Python environment supported."
)


def platform_support_guidance() -> str:
    """Return actionable guidance only for a native Windows Python host."""
    return WINDOWS_GUIDANCE if sys.platform == "win32" else ""
