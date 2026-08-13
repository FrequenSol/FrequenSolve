"""Utility helpers for orchestration backends."""

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from frequensolve.orchestrator.utils.credentials import (
        CloudCredentials as CloudCredentials,
    )
    from frequensolve.orchestrator.utils.credentials import Credentials as Credentials
    from frequensolve.orchestrator.utils.pool import PoolInfo as PoolInfo
    from frequensolve.orchestrator.utils.pool import PoolStatus as PoolStatus
    from frequensolve.orchestrator.utils.progress import RunMonitor as RunMonitor
    from frequensolve.orchestrator.utils.progress import (
        status_table_html as status_table_html,
    )
    from frequensolve.orchestrator.utils.progress import status_text as status_text
    from frequensolve.orchestrator.utils.progress import wait as wait
    from frequensolve.orchestrator.utils.progress import wait_all as wait_all
    from frequensolve.orchestrator.utils.ssh import SSHClientClass as SSHClientClass
    from frequensolve.orchestrator.utils.ssh import SSHProxy as SSHProxy

_EXPORT_MODULES: Final[dict[str, str]] = {
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

__all__: list[str] = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MODULES:
        value = getattr(import_module(_EXPORT_MODULES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
