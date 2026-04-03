from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "zonas_manejo",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or return a configured logger.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str | Path | None
        Optional file path for logging to disk.
    level : int
        Logging level, default logging.INFO.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # file handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger