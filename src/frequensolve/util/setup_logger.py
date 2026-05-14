"""
Module for setting up logging with a RotatingFileHandler.

This setup ensures that logs are saved to a file, and when the file reaches a specified size,
it automatically rotates the log file with backups.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional, Union

logging_level = logging.INFO
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
DEFAULT_DEPENDENCY_LOGGERS = ("dask", "distributed", "tornado", "bokeh")


def normalize_log_level(level: Union[int, str]) -> int:
    """Normalize integer or string log levels."""

    if isinstance(level, int):
        return level
    try:
        return int(level)
    except ValueError:
        pass
    normalized = str(level).upper()
    if not hasattr(logging, normalized):
        raise ValueError(f"Unknown logging level: {level}")
    value = getattr(logging, normalized)
    if not isinstance(value, int):
        raise ValueError(f"Unknown logging level: {level}")
    return value


def init_logger(
    name: str = "FrequenSolve",
    log_file: Optional[Union[str, Path]] = "frequensolve.log",
    level: Union[int, str] = logging_level,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
) -> logging.Logger:
    """Set up a logger that saves logs to a file.

    Args:
        name (str): The name of the logger.
        log_file (str): The file where logs will be saved.
        level (int): The logging level.
        max_bytes (int): The maximum size (in bytes) of the log file before rotating.
        backup_count (int): The number of backup files to keep.

    Returns:
        logging.Logger: Configured logger.
    """
    # Create a logger
    logger = logging.getLogger(name)
    level = normalize_log_level(level)
    logger.setLevel(level)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if not any(
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == log_file
            for handler in logger.handlers
        ):
            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
            handler.setLevel(level)
            logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    return logger


def configure_logging(
    *,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = False,
    dependency_level: Optional[Union[int, str]] = logging.WARNING,
    dependency_loggers: Iterable[str] = DEFAULT_DEPENDENCY_LOGGERS,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configure FrequenSolve package logging.

    Args:
        level: Logging level as an integer or name such as ``"INFO"``.
        log_file: Optional file path for rotating package logs.
        console: If True, also emit logs to stderr.
        dependency_level: Optional level applied to known noisy dependency loggers.
            Use ``None`` to leave dependency loggers unchanged.
        dependency_loggers: Logger-name prefixes controlled by ``dependency_level``.
        max_bytes: Maximum rotating log file size.
        backup_count: Number of rotated log files to keep.
    """

    level = normalize_log_level(level)
    set_log_level(level)
    package_logger = logging.getLogger("frequensolve")
    package_logger.setLevel(level)
    package_logger.propagate = not (console or log_file is not None)

    formatter = logging.Formatter(DEFAULT_FORMAT)

    for handler in package_logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == log_file
            for handler in package_logger.handlers
        ):
            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            handler.setFormatter(formatter)
            handler.setLevel(level)
            package_logger.addHandler(handler)

    if console and not any(
        getattr(handler, "_frequensolve_console", False)
        for handler in package_logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler._frequensolve_console = True
        handler.setFormatter(formatter)
        handler.setLevel(level)
        package_logger.addHandler(handler)

    if dependency_level is not None:
        dependency_level = normalize_log_level(dependency_level)
        for name in dependency_loggers:
            logger = logging.getLogger(name)
            logger.setLevel(dependency_level)
            logger._frequensolve_dependency_level = dependency_level


def set_log_level(level: Union[int, str]):
    global logging_level
    level = normalize_log_level(level)
    logging_level = level
    logging.getLogger("frequensolve").setLevel(level)
    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("frequensolve") and isinstance(logger, logging.Logger):
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


def in_jupyter_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter Notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False
    except NameError:
        return False


def disable_jupyter_logging():
    if in_jupyter_notebook():
        logging.disable(logging.CRITICAL)


if __name__ == "__main__":
    logger = init_logger()
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
