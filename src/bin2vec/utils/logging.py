"""Logging configuration for bin2vec."""

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger for bin2vec."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("bin2vec")
    root.setLevel(level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a bin2vec sub-logger."""
    return logging.getLogger(f"bin2vec.{name}")
