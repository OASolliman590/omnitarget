"""
OmniTarget Retry Logic with Exponential Backoff

Provides retry decorators and utilities for handling transient failures.
Part of P0-2: Error Handling & Retry Logic critical fix.

Author: OmniTarget Team
Date: 2025-01-06
"""

import asyncio
import logging
from typing import Callable, Optional, Type, Tuple, Any
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    before_sleep_log,
    after_log,
    RetryCallState,
    AsyncRetrying,
    Retrying,
)

from .exceptions import (
    OmniTargetException,
    DatabaseConnectionError,
    DatabaseTimeoutError,
    DatabaseUnavailableError,
    MCPServerError,
    is_transient_error,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Retry Configuration
# =============================================================================

class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (default: 3)
        initial_wait: Initial wait time in seconds (default: 1)
        max_wait: Maximum wait time in seconds (default: 10)
        multiplier: Exponential backoff multiplier (default: 2)
        retry_on: Tuple of exception types to retry on
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 10.0,
        multiplier: float = 2.0,
        retry_on: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum retry attempts (1-10)
            initial_wait: Initial wait in seconds (0.1-5.0)
            max_wait: Maximum wait in seconds (1.0-60.0)
            multiplier: Backoff multiplier (1.0-5.0)
            retry_on: Exception types to retry (defaults to transient errors)
        """
        self.max_attempts = max(1, min(10, max_attempts))
        self.initial_wait = max(0.1, min(5.0, initial_wait))
        self.max_wait = max(1.0, min(60.0, max_wait))
        self.multiplier = max(1.0, min(5.0, multiplier))

        # Default to common transient errors
        self.retry_on = retry_on or (
            DatabaseConnectionError,
            DatabaseTimeoutError,
            DatabaseUnavailableError,
            ConnectionError,
            ConnectionResetError,  # CRITICAL FIX: Add ConnectionResetError for ECONNRESET
            TimeoutError,
        )


# Default configurations for different use cases
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_wait=1.0,
    max_wait=10.0,
    multiplier=2.0
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_wait=0.5,
    max_wait=30.0,
    multiplier=2.0
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    initial_wait=2.0,
    max_wait=10.0,
    multiplier=2.0
)

# CRITICAL FIX: Reactome-specific retry config for ECONNRESET errors
# Shorter backoff for connection resets (they're usually quick to recover)
REACTOME_RETRY_CONFIG = RetryConfig(
    max_attempts=3,  # 3 attempts total (1 initial + 2 retries)
    initial_wait=0.5,  # Start with 0.5s wait (shorter than default 1.0s)
    max_wait=5.0,  # Max 5s wait (shorter than default 10.0s)
    multiplier=2.0  # Exponential backoff: 0.5s, 1.0s, 2.0s
)


# =============================================================================
# Retry Condition Helpers
# =============================================================================

def should_retry_exception(exception: Exception) -> bool:
    """
    Determine if an exception should trigger a retry.

    Args:
        exception: The exception to check

    Returns:
        True if the exception is retryable

    Example:
        >>> try:
        ...     await mcp_client.call_tool(...)
        ... except Exception as e:
        ...     if should_retry_exception(e):
        ...         # Retry logic
    """
    # Use the helper from exceptions module
    if is_transient_error(exception):
        return True

    # Additional checks for MCP-specific errors
    if isinstance(exception, MCPServerError):
        return exception.is_retryable()

    return False


def log_retry_attempt(retry_state: RetryCallState) -> None:
    """
    Log retry attempts with context.

    Args:
        retry_state: Tenacity retry state object
    """
    if retry_state.outcome and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        attempt_number = retry_state.attempt_number

        logger.warning(
            f"Retry attempt {attempt_number} after error: {type(exception).__name__}: {exception}"
        )


# =============================================================================
# Async Retry Decorators
# =============================================================================

def async_retry_with_backoff(
    config: Optional[RetryConfig] = None,
    logger_name: Optional[str] = None
) -> Callable:
    """
    Decorator for async functions with exponential backoff retry.

    Args:
        config: Retry configuration (defaults to DEFAULT_RETRY_CONFIG)
        logger_name: Logger name for retry logs (defaults to function's module)

    Returns:
        Decorated async function with retry logic

    Example:
        >>> @async_retry_with_backoff()
        ... async def fetch_data(gene: str):
        ...     return await mcp_client.call_tool("get_gene_info", {"gene": gene})

        >>> # Custom config
        >>> @async_retry_with_backoff(config=AGGRESSIVE_RETRY_CONFIG)
        ... async def critical_operation():
        ...     ...
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    log = logging.getLogger(logger_name) if logger_name else logger

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(cfg.max_attempts),
                wait=wait_exponential(
                    multiplier=cfg.multiplier,
                    min=cfg.initial_wait,
                    max=cfg.max_wait
                ),
                retry=retry_if_exception(should_retry_exception),
                before_sleep=before_sleep_log(log, logging.WARNING),
                reraise=True
            ):
                with attempt:
                    result = await func(*args, **kwargs)

                    # Log success after retry
                    if attempt.retry_state.attempt_number > 1:
                        log.info(
                            f"{func.__name__} succeeded after "
                            f"{attempt.retry_state.attempt_number} attempts"
                        )

                    return result

        return wrapper
    return decorator


def sync_retry_with_backoff(
    config: Optional[RetryConfig] = None,
    logger_name: Optional[str] = None
) -> Callable:
    """
    Decorator for sync functions with exponential backoff retry.

    Args:
        config: Retry configuration (defaults to DEFAULT_RETRY_CONFIG)
        logger_name: Logger name for retry logs

    Returns:
        Decorated sync function with retry logic

    Example:
        >>> @sync_retry_with_backoff()
        ... def fetch_data(gene: str):
        ...     return requests.get(f"https://api.example.com/gene/{gene}")
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    log = logging.getLogger(logger_name) if logger_name else logger

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in Retrying(
                stop=stop_after_attempt(cfg.max_attempts),
                wait=wait_exponential(
                    multiplier=cfg.multiplier,
                    min=cfg.initial_wait,
                    max=cfg.max_wait
                ),
                retry=retry_if_exception(should_retry_exception),
                before_sleep=before_sleep_log(log, logging.WARNING),
                reraise=True
            ):
                with attempt:
                    result = func(*args, **kwargs)

                    # Log success after retry
                    if attempt.retry_state.attempt_number > 1:
                        log.info(
                            f"{func.__name__} succeeded after "
                            f"{attempt.retry_state.attempt_number} attempts"
                        )

                    return result

        return wrapper
    return decorator


# =============================================================================
# Context-Based Retry Functions
# =============================================================================

async def retry_async_operation(
    operation: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    operation_name: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Retry an async operation with exponential backoff.

    Useful when you can't use a decorator (e.g., dynamic operations).

    Args:
        operation: Async callable to retry
        *args: Positional arguments for operation
        config: Retry configuration
        operation_name: Name for logging (defaults to operation.__name__)
        **kwargs: Keyword arguments for operation

    Returns:
        Result of the operation

    Raises:
        Exception: If all retry attempts fail

    Example:
        >>> result = await retry_async_operation(
        ...     mcp_client.call_tool,
        ...     "get_gene_info",
        ...     {"gene": "AXL"},
        ...     config=AGGRESSIVE_RETRY_CONFIG,
        ...     operation_name="fetch_AXL_info"
        ... )
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    op_name = operation_name or getattr(operation, '__name__', 'async_operation')

    last_exception = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            result = await operation(*args, **kwargs)

            # Log success after retry
            if attempt > 1:
                logger.info(f"{op_name} succeeded after {attempt} attempts")

            return result

        except Exception as e:
            last_exception = e

            # Check if we should retry
            if not should_retry_exception(e):
                # Special handling for HPA "Invalid gene arguments" - not really an error
                if (hasattr(e, 'error_code') and e.error_code == -32602 and
                    'get_pathology_data' in op_name and 'Invalid gene arguments' in str(e)):
                    logger.info(
                        f"{op_name}: No pathology data available in HPA database (expected for some genes)"
                    )
                else:
                    logger.error(
                        f"{op_name} failed with non-retryable error: "
                        f"{type(e).__name__}: {e}"
                    )
                raise

            # Log retry attempt
            logger.warning(
                f"{op_name} attempt {attempt}/{cfg.max_attempts} failed: "
                f"{type(e).__name__}: {e}"
            )

            # Don't sleep after the last attempt
            if attempt < cfg.max_attempts:
                # Calculate exponential backoff
                wait_time = min(
                    cfg.initial_wait * (cfg.multiplier ** (attempt - 1)),
                    cfg.max_wait
                )
                logger.info(f"Waiting {wait_time:.2f}s before retry...")
                await asyncio.sleep(wait_time)

    # All attempts failed
    logger.error(
        f"{op_name} failed after {cfg.max_attempts} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}"
    )
    raise last_exception


def retry_sync_operation(
    operation: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    operation_name: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Retry a sync operation with exponential backoff.

    Useful when you can't use a decorator (e.g., dynamic operations).

    Args:
        operation: Callable to retry
        *args: Positional arguments for operation
        config: Retry configuration
        operation_name: Name for logging (defaults to operation.__name__)
        **kwargs: Keyword arguments for operation

    Returns:
        Result of the operation

    Raises:
        Exception: If all retry attempts fail

    Example:
        >>> result = retry_sync_operation(
        ...     requests.get,
        ...     "https://api.example.com/gene/AXL",
        ...     config=CONSERVATIVE_RETRY_CONFIG,
        ...     operation_name="fetch_AXL_from_API"
        ... )
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    op_name = operation_name or getattr(operation, '__name__', 'sync_operation')

    last_exception = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            result = operation(*args, **kwargs)

            # Log success after retry
            if attempt > 1:
                logger.info(f"{op_name} succeeded after {attempt} attempts")

            return result

        except Exception as e:
            last_exception = e

            # Check if we should retry
            if not should_retry_exception(e):
                # Special handling for HPA "Invalid gene arguments" - not really an error
                if (hasattr(e, 'error_code') and e.error_code == -32602 and
                    'get_pathology_data' in op_name and 'Invalid gene arguments' in str(e)):
                    logger.info(
                        f"{op_name}: No pathology data available in HPA database (expected for some genes)"
                    )
                else:
                    logger.error(
                        f"{op_name} failed with non-retryable error: "
                        f"{type(e).__name__}: {e}"
                    )
                raise

            # Log retry attempt
            logger.warning(
                f"{op_name} attempt {attempt}/{cfg.max_attempts} failed: "
                f"{type(e).__name__}: {e}"
            )

            # Don't sleep after the last attempt
            if attempt < cfg.max_attempts:
                # Calculate exponential backoff
                wait_time = min(
                    cfg.initial_wait * (cfg.multiplier ** (attempt - 1)),
                    cfg.max_wait
                )
                logger.info(f"Waiting {wait_time:.2f}s before retry...")
                import time
                time.sleep(wait_time)

    # All attempts failed
    logger.error(
        f"{op_name} failed after {cfg.max_attempts} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}"
    )
    raise last_exception


# =============================================================================
# Convenience Aliases
# =============================================================================

# Short aliases for common use
retry_async = async_retry_with_backoff
retry_sync = sync_retry_with_backoff

# Default decorated retry (most common case)
retry = async_retry_with_backoff()
