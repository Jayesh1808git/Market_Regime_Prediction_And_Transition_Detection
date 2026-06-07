"""
src/utils/logger.py

Centralized logging setup for the entire project.

WHY THIS PATTERN:
  Instead of using print() everywhere (which you can't control or filter),
  every module gets a named logger. You can set the log level once here
  and it applies everywhere. Logs go to both console and a file.

LESSON FOR ANY PROJECT:
  Copy this file as-is into any project. Change LOG_FILE name only.
  In every other module: `from src.utils.logger import get_logger`
                         `logger = get_logger(__name__)`
"""

import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE    = "logs/training.log"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a named logger. Creates log directory if needed.

    Args:
        name  : Logger name — always pass __name__ so you know
                which module produced each log line.
        level : Logging level (default: INFO)

    Returns:
        Configured logger with console + file handlers.

    Usage:
        logger = get_logger(__name__)
        logger.info("Loading data...")
        logger.warning("Missing VIX data for 3 dates")
        logger.error("Model training failed")
    """
    logger = logging.getLogger(name)

    # Don't add handlers twice if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Force UTF-8 stdout on Windows consoles to avoid UnicodeEncodeError
    # when log messages include arrows/symbols.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # If reconfigure is unsupported, continue with safe defaults.
        pass

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console)

    # File handler
    Path(LOG_FILE).parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger
