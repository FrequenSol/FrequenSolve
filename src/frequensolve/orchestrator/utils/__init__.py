"""Utility helpers for orchestration backends."""

from importlib import import_module

_EXPORT_MODULES = {
    "CloudCredentials": "frequensolve.orchestrator.utils.credentials",
    "Credentials": "frequensolve.orchestrator.utils.credentials",
    "PoolInfo": "frequensolve.orchestrator.utils.pool",
    "PoolStatus": "frequensolve.orchestrator.utils.pool",
    "RunMonitor": "frequensolve.orchestrator.utils.progress",
    "SSHClientClass": "frequensolve.orchestrator.utils.ssh",
    "SSHProxy": "frequensolve.orchestrator.utils.ssh",
    "status_table_html": "frequensolve.orchestrator.utils.progress",
    "status_text": "frequensolve.orchestrator.utils.progress",
    "wait": "frequensolve.orchestrator.utils.progress",
    "wait_all": "frequensolve.orchestrator.utils.progress",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    if name in _EXPORT_MODULES:
        value = getattr(import_module(_EXPORT_MODULES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
