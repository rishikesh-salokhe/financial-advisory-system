"""
Logging configuration using loguru.

Call ``configure_logging()`` once at application startup. After that, just::

    from loguru import logger
    logger.info("something happened")

The loguru sink also intercepts stdlib ``logging`` calls so libraries that use
the standard logging module (uvicorn, motor, transformers, etc.) flow through
the same pipeline.
"""
from __future__ import annotations

import logging
import sys

from loguru import logger

from backend.core.config import settings


class InterceptHandler(logging.Handler):
    """Route stdlib ``logging`` records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller frame so loguru reports the original location, not this shim.
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging() -> None:
    """Wire up loguru as the single logging backend for the process."""
    logger.remove()

    # Console sink — pretty in dev, structured-ish in prod.
    if settings.app_env == "development":
        fmt = (
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(sys.stdout, format=fmt, level=settings.log_level, colorize=True)
    else:
        logger.add(
            sys.stdout,
            level=settings.log_level,
            serialize=True,   # JSON output for log aggregators
        )

    # Replace stdlib logging handlers with our interceptor.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        std_logger = logging.getLogger(noisy)
        std_logger.handlers = [InterceptHandler()]
        std_logger.propagate = False

    logger.info(f"Logging configured (env={settings.app_env}, level={settings.log_level})")
