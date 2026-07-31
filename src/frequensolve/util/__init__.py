"""Utility facade that leaves individual submodule imports lightweight."""

from __future__ import annotations

import importlib
import threading
from typing import Any

from frequensolve._exports import unique_exports

_PUBLIC_MODULE_ORDER = ("fft", "fields", "mixins", "named_list", "physics", "store")
_PUBLIC_API_LOCK = threading.RLock()
_PUBLIC_API_LOADED = False


def _load_public_api() -> None:
    global _PUBLIC_API_LOADED, __all__
    if _PUBLIC_API_LOADED:
        return
    with _PUBLIC_API_LOCK:
        if _PUBLIC_API_LOADED:
            return
        modules = {
            name: importlib.import_module(f"frequensolve.util.{name}")
            for name in _PUBLIC_MODULE_ORDER
        }
        for module in modules.values():
            for export_name in module.__all__:
                globals()[export_name] = getattr(module, export_name)

        setup_logger = importlib.import_module("frequensolve.util.setup_logger")
        globals()["configure_logging"] = setup_logger.configure_logging
        globals()["set_log_level"] = setup_logger.set_log_level
        __all__ = unique_exports(
            *(modules[name].__all__ for name in _PUBLIC_MODULE_ORDER),
            ["configure_logging", "set_log_level"],
        )
        _PUBLIC_API_LOADED = True


def __getattr__(name: str) -> Any:
    _load_public_api()
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    _load_public_api()
    return sorted({*globals(), *__all__})
