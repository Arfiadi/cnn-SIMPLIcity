"""Logging setup and formatting."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "simplicity_cnn",
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure a unified logger with console and optional file handler.

    Args:
        name: Logger name.
        log_file: Optional file path to store log output.
        level: Console logging level (defaults to INFO).
        file_level: File logging level (defaults to DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(level, file_level))

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)

    return logger
