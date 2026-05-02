"""Dask preload helpers for quiet FrequenSolve local runs."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Union

try:
    from dask import config as dask_config
except ModuleNotFoundError:  # pragma: no cover - only imported by Dask workers.
    dask_config = None


DEPENDENCY_LOGGERS = ("dask", "distributed", "tornado", "bokeh")
EXPLICIT_LOGGERS = (
    "distributed.client",
    "distributed.core",
    "distributed.nanny",
    "distributed.scheduler",
    "distributed.worker",
)


class _DistributedConnectionClosedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "distributed.core" or record.levelno >= logging.WARNING:
            return True
        message = record.getMessage()
        return not (
            message.startswith("Connection to ")
            and message.endswith(" has been closed.")
        )


_CONNECTION_CLOSED_FILTER = _DistributedConnectionClosedFilter()


def _normalize_level(level: Union[int, str, None], default: int = logging.ERROR) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    try:
        return int(level)
    except (TypeError, ValueError):
        pass
    name = str(level).upper()
    value = getattr(logging, name, default)
    return value if isinstance(value, int) else default


def _level_from_dask_config(default: int = logging.ERROR) -> int:
    if dask_config is None:
        return default
    try:
        logging_config = dask_config.config.get("distributed", {}).get("logging", {})
    except Exception:
        return default
    if not isinstance(logging_config, dict):
        return default
    return _normalize_level(
        logging_config.get("distributed.core")
        or logging_config.get("distributed")
        or logging_config.get("dask"),
        default=default,
    )


def _logger_names(prefixes: Iterable[str]) -> set[str]:
    names = set(prefixes)
    names.update(EXPLICIT_LOGGERS)
    for name, logger in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger, logging.Logger):
            continue
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            names.add(name)
    return names


def _add_filter_once(handler_or_logger: Any) -> None:
    if _CONNECTION_CLOSED_FILTER not in getattr(handler_or_logger, "filters", []):
        handler_or_logger.addFilter(_CONNECTION_CLOSED_FILTER)


def configure_dependency_logging(level: Optional[Union[int, str]] = None) -> int:
    """Apply FrequenSolve's dependency logging policy in a Dask process."""

    normalized = _normalize_level(level, default=_level_from_dask_config())
    for name in _logger_names(DEPENDENCY_LOGGERS):
        logger = logging.getLogger(name)
        logger.setLevel(normalized)
        _add_filter_once(logger)
        for handler in logger.handlers:
            handler.setLevel(normalized)
            _add_filter_once(handler)
    for handler in logging.getLogger().handlers:
        _add_filter_once(handler)
    return normalized


def dask_setup(_server: Any) -> None:
    configure_dependency_logging()
