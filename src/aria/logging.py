"""Logging configuration for ARIA with structlog."""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import structlog


def setup_logging(
    level: str | None = "INFO",
    log_file: Path | None = None,
    show_timestamps: bool = True,
) -> None:
    """Configure structlog for console and optional file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR), or None to use INFO
        log_file: Optional file path to write logs
        show_timestamps: Include timestamps in console output
    """
    # Configure standard logging
    level = level or "INFO"  # Handle None case
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Setup handlers
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(message)s",
        force=True,
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]

    if show_timestamps:
        processors.append(structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False))

    # Add different formatters for console vs file
    if log_file:
        # File output: use JSON for parsing
        processors.extend(
            [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        # Console output: use colored, human-readable format
        processors.extend(
            [
                structlog.dev.set_exc_info,
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger for a module.

    Args:
        name: Module name (e.g., "aria.agent.core")

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class Timer:
    """Context manager for timing operations with logging.

    Usage:
        with Timer("operation_name", logger) as timer:
            # do work
            pass
        print(f"Took {timer.elapsed:.3f}s")
    """

    def __init__(self, name: str, logger: Any | None = None):
        """Initialize timer.

        Args:
            name: Operation name for logging
            logger: Logger instance (uses default if None)
        """
        self.name = name
        self.logger = logger or get_logger("timer")
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting: {self.name}")
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and log result."""
        self.elapsed = time.perf_counter() - self.start_time
        self.logger.debug(f"Completed: {self.name}", elapsed_s=f"{self.elapsed:.3f}")


class AsyncTimer:
    """Async context manager for timing operations.

    Usage:
        async with AsyncTimer("operation_name", logger) as timer:
            await async_operation()
        print(f"Took {timer.elapsed:.3f}s")
    """

    def __init__(self, name: str, logger: Any | None = None):
        """Initialize async timer.

        Args:
            name: Operation name for logging
            logger: Logger instance (uses default if None)
        """
        self.name = name
        self.logger = logger or get_logger("timer")
        self.start_time: float = 0
        self.elapsed: float = 0

    async def __aenter__(self) -> "AsyncTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting: {self.name}")
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop timing and log result."""
        self.elapsed = time.perf_counter() - self.start_time
        self.logger.debug(f"Completed: {self.name}", elapsed_s=f"{self.elapsed:.3f}")
