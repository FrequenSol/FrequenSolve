"""User configuration and cache storage paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

FREQUENSOLVE_HOME_ENV_VAR = "FREQUENSOLVE_HOME"
DEFAULT_FREQUENSOLVE_HOME_NAME = ".frequensolve"

__all__ = [
    "DEFAULT_FREQUENSOLVE_HOME_NAME",
    "FREQUENSOLVE_HOME_ENV_VAR",
    "frequensolve_home",
]


def frequensolve_home(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the root FrequenSolve user configuration directory.

    Args:
        path: Optional explicit directory. When omitted, the
            ``FREQUENSOLVE_HOME`` environment variable is honored before
            falling back to ``~/.frequensolve``.

    Returns:
        Expanded path to the directory used for SDK caches and user-level
        configuration. The directory is not created by this helper.
    """

    if path is not None:
        return Path(path).expanduser()
    env_path = os.getenv(FREQUENSOLVE_HOME_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / DEFAULT_FREQUENSOLVE_HOME_NAME
