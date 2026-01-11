"""
Structured Logging Configuration

Provides JSON-formatted logs with correlation IDs for production monitoring.
P0-4: Production Monitoring (Week 7)

Updated: 2025-11-07
"""

import logging
import json
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar


# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs in JSON format with correlation IDs and additional context.
    Compatible with log aggregation systems (ELK, Splunk, Datadog).
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID for request tracking
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add process info
        log_data["process_id"] = os.getpid()
        log_data["thread_id"] = record.thread

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Add performance metrics if present
        if hasattr(record, 'duration_ms'):
            log_data["duration_ms"] = record.duration_ms

        if hasattr(record, 'scenario_id'):
            log_data["scenario_id"] = record.scenario_id

        return json.dumps(log_data, default=str)


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
    """
    # Create formatter
    formatter = StructuredFormatter()

    handlers = []

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File handler (JSON logs)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    # Log configuration
    root_logger.info("Structured logging configured", extra={
        "extra_fields": {
            "log_level": log_level,
            "log_file": log_file,
            "handlers": len(handlers)
        }
    })


def get_correlation_id() -> str:
    """
    Get current correlation ID or create new one.

    Returns:
        str: Correlation ID (UUID4)
    """
    correlation_id = correlation_id_var.get()
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Correlation ID to set
    """
    correlation_id_var.set(correlation_id)


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **extra_fields
) -> None:
    """
    Log with structured context.

    Args:
        logger: Logger instance
        level: Log level (info, warning, error, etc.)
        message: Log message
        **extra_fields: Additional fields to include in log

    Example:
        >>> log_with_context(
        ...     logger,
        ...     "error",
        ...     "scenario_execution_failed",
        ...     scenario_id=3,
        ...     disease="breast cancer",
        ...     phase="marker_discovery",
        ...     error="Connection timeout"
        ... )
    """
    log_func = getattr(logger, level.lower())
    log_func(message, extra={"extra_fields": extra_fields})


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with structured logging.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for creating scenario-specific loggers
def get_scenario_logger(scenario_id: int, scenario_name: str) -> logging.Logger:
    """
    Get a logger for a specific scenario with common fields.

    Args:
        scenario_id: Scenario ID
        scenario_name: Scenario name

    Returns:
        logging.Logger: Configured logger with scenario context
    """
    logger = logging.getLogger(f"scenario.{scenario_id}.{scenario_name}")
    logger.setScenarioId(scenario_id)
    logger.setScenarioName(scenario_name)
    return logger


# Performance logging decorator
def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger to use for performance logging

    Example:
        >>> @log_execution_time(logger)
        ... async def expensive_operation():
        ...     pass
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.info(
                    f"{func.__name__} completed",
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "duration_ms": duration,
                            "status": "success"
                        }
                    }
                )
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.error(
                    f"{func.__name__} failed",
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "duration_ms": duration,
                            "status": "error",
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    }
                )
                raise
        return async_wrapper
    return decorator
