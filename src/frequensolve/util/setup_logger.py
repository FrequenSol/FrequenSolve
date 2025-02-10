"""
Module for setting up logging with a RotatingFileHandler.

This setup ensures that logs are saved to a file, and when the file reaches a specified size,
it automatically rotates the log file with backups.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

logging_level = logging.INFO


def init_logger(
    name: str = "FrequenSolve",
    log_file: str = "frequensolve.log",
    level: int = logging_level,
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
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a rotating file handler
    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)

    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def set_log_level(level: int):
    global logging_level
    logging_level = level


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
