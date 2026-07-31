"""Public package entrypoint for the FrequenSolve Python API.

The root namespace lazily re-exports the stable authoring API from the main
subpackages so users can continue to write ``import frequensolve as fs``.  A
submodule import such as ``frequensolve.mcp_server.server`` must not pay the
startup cost of unrelated plotting, orchestration, and IO modules.

Optional execution backends remain guarded in ``frequensolve.orchestrator.sites``:
their names are always available after the public facade is hydrated, while
missing extras fail with an install hint when used.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Any

from frequensolve._exports import unique_exports
from frequensolve._version import get_versions

_PUBLIC_MODULE_ORDER = (
    "_loading",
    "frequensolver",
    "geometry",
    "knowledge",
    "mesh",
    "model",
    "orchestrator",
    "plotting",
    "project",
    "seismic",
    "simulation",
    "units",
    "util",
    "validation",
)
_PUBLIC_EXPORT_ORDER = (
    "project",
    "units",
    "geometry",
    "knowledge",
    "frequensolver",
    "model",
    "mesh",
    "seismic",
    "plotting",
    "simulation",
    "orchestrator",
    "util",
    "validation",
    "_loading",
)
_COLORMAP_EXPORTS = ("get_colormap", "RdYlBu", "RdYlBu_r", "BuGrOr", "BuGrOr_r")
_PUBLIC_API_LOCK = threading.RLock()
_PUBLIC_API_LOADED = False

__version__ = get_versions()["version"]


def _load_public_api() -> None:
    """Hydrate the historical root exports exactly once on first use."""

    global _PUBLIC_API_LOADED, __all__
    if _PUBLIC_API_LOADED:
        return
    with _PUBLIC_API_LOCK:
        if _PUBLIC_API_LOADED:
            return
        modules: dict[str, ModuleType] = {}
        for module_name in _PUBLIC_MODULE_ORDER:
            module = importlib.import_module(f"frequensolve.{module_name}")
            modules[module_name] = module
            for export_name in module.__all__:
                globals()[export_name] = getattr(module, export_name)

        __all__ = [
            "__version__",
            *unique_exports(
                *(modules[name].__all__ for name in _PUBLIC_EXPORT_ORDER),
                _COLORMAP_EXPORTS,
            ),
        ]
        _PUBLIC_API_LOADED = True


def __getattr__(name: str) -> Any:
    if name in _COLORMAP_EXPORTS:
        colormaps = importlib.import_module("frequensolve.util.colormaps")

        return getattr(colormaps, name)
    _load_public_api()
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    _load_public_api()
    return sorted({*globals(), *__all__})
