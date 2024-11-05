import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Set up a logger with console and file handlers.

    Args:
        name: Name of the logger
        log_file: Optional path to log file. If None, only console output is created

    Returns
    -------
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    simple_formatter = logging.Formatter("%(message)s")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is provided
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Prevent duplicate logging
    logger.propagate = False

    return logger
