"""Structured logging configuration using structlog.

Provides JSON logging for production and console logging for development,
with correlation ID support for request tracing.
"""

import logging
import os
import sys

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.typing import Processor


def _get_log_level(level: str) -> int:
    """Convert string log level to logging constant."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level.upper(), logging.INFO)


def get_processors(json_mode: bool) -> list[Processor]:
    """Build the processor chain for structlog.

    Args:
        json_mode: If True, use JSON renderer; otherwise use console renderer.

    Returns:
        List of structlog processors.
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ExtraAdder(),
    ]

    if json_mode:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )

    return shared_processors


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with stdlib logging integration.

    Sets up structured logging with:
    - JSON output when LOG_FORMAT=json environment variable is set
    - Console output with colors by default
    - Correlation ID support via structlog.contextvars
    - Stdlib logging integration via ProcessorFormatter

    Args:
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to "INFO".
    """
    log_format = os.environ.get("LOG_FORMAT", "console").lower()
    json_mode = log_format == "json"
    log_level = _get_log_level(level)

    # Get processors for the output format (used internally)
    _processors = get_processors(json_mode)

    # For stdlib integration, we need to handle the renderer separately
    # ProcessorFormatter expects to do the final rendering
    formatter_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ExtraAdder(),
    ]

    # Configure stdlib logging to use structlog's ProcessorFormatter
    formatter: logging.Formatter
    if json_mode:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=formatter_processors,
            processors=[structlog.processors.JSONRenderer()],
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=formatter_processors,
            processors=[structlog.dev.ConsoleRenderer(colors=True)],
        )

    # Set up the root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Build the final processor list for structlog
    # structlog internally handles the rendering
    structlog_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ExtraAdder(),
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    if json_mode:
        structlog_processors.append(structlog.processors.JSONRenderer())
    else:
        structlog_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=structlog_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a configured structlog logger.

    Returns a logger bound with the specified name for easy identification
    in log output.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A structlog BoundLogger instance.
    """
    return structlog.get_logger(name)


# Expose contextvars functions for correlation ID support
__all__ = [
    "configure_logging",
    "get_logger",
    "bind_contextvars",
    "clear_contextvars",
]
