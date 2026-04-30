"""Public package entrypoint for the FrequenSolve Python SDK.

Importing :mod:`frequensolve` is intentionally lightweight. Optional execution
backends, plotting helpers, and cloud/HPC integrations are imported from their
submodules explicitly.
"""

from frequensolve._version import get_versions

__version__ = get_versions()["version"]

__all__ = ["__version__"]
