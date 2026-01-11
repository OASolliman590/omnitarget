"""
Unit tests for retry logic with exponential backoff (P0-2).

Tests retry decorators, configurations, and retry strategies.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, call
from typing import List

from src.core.retry import (
    RetryConfig,
    DEFAULT_RETRY_CONFIG,
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
    should_retry_exception,
    async_retry_with_backoff,
    sync_retry_with_backoff,
    retry_async_operation,
    retry_sync_operation,
)
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseTimeoutError,
    DatabaseUnavailableError,
    MCPServerError,
    DataValidationError,
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_retry_config(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_wait == 1.0
        assert config.max_wait == 10.0
        assert config.multiplier == 2.0
        assert DatabaseConnectionError in config.retry_on

    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_wait=0.5,
            max_wait=30.0,
            multiplier=3.0
        )
        assert config.max_attempts == 5
        assert config.initial_wait == 0.5
        assert config.max_wait == 30.0
        assert config.multiplier == 3.0

    def test_retry_config_bounds(self):
        """Test retry configuration enforces bounds."""
        # Test max_attempts bounds
        config = RetryConfig(max_attempts=100)
        assert config.max_attempts <= 10

        config = RetryConfig(max_attempts=0)
        assert config.max_attempts >= 1

        # Test initial_wait bounds
        config = RetryConfig(initial_wait=10.0)
        assert config.initial_wait <= 5.0

        # Test max_wait bounds
        config = RetryConfig(max_wait=100.0)
        assert config.max_wait <= 60.0

    def test_predefined_configs(self):
        """Test predefined retry configurations."""
        # Default
        assert DEFAULT_RETRY_CONFIG.max_attempts == 3
        assert DEFAULT_RETRY_CONFIG.initial_wait == 1.0

        # Aggressive
        assert AGGRESSIVE_RETRY_CONFIG.max_attempts == 5
        assert AGGRESSIVE_RETRY_CONFIG.initial_wait == 0.5

        # Conservative
        assert CONSERVATIVE_RETRY_CONFIG.max_attempts == 2
        assert CONSERVATIVE_RETRY_CONFIG.initial_wait == 2.0


class TestShouldRetryException:
    """Test should_retry_exception helper."""

    def test_should_retry_transient_errors(self):
        """Test transient errors should be retried."""
        transient_errors = [
            DatabaseConnectionError("Test", "Connection failed"),
            DatabaseTimeoutError("Test", 30.0, "query"),
            DatabaseUnavailableError("Test", "Down"),
            ConnectionError("Lost connection"),
            TimeoutError("Timed out"),
        ]

        for error in transient_errors:
            assert should_retry_exception(error), \
                f"{type(error).__name__} should be retried"

    def test_should_not_retry_non_transient_errors(self):
        """Test non-transient errors should not be retried."""
        non_transient_errors = [
            DataValidationError("Bad data"),
            ValueError("Bad value"),
            TypeError("Bad type"),
        ]

        for error in non_transient_errors:
            assert not should_retry_exception(error), \
                f"{type(error).__name__} should not be retried"

    def test_should_retry_mcp_server_error_based_on_code(self):
        """Test MCPServerError retry based on error code."""
        # Retryable (server error)
        retryable = MCPServerError("Test", -32000, "Server error")
        assert should_retry_exception(retryable)

        # Non-retryable (client error)
        non_retryable = MCPServerError("Test", -32602, "Invalid params")
        assert not should_retry_exception(non_retryable)


@pytest.mark.asyncio
class TestAsyncRetryDecorator:
    """Test async retry decorator."""

    async def test_async_retry_success_first_attempt(self):
        """Test successful operation on first attempt."""
        call_count = 0

        @async_retry_with_backoff()
        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_operation()
        assert result == "success"
        assert call_count == 1

    async def test_async_retry_success_after_retries(self):
        """Test successful operation after retries."""
        call_count = 0

        @async_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DatabaseConnectionError("Test", "Connection failed")
            return "success"

        result = await flaky_operation()
        assert result == "success"
        assert call_count == 3

    async def test_async_retry_failure_after_max_attempts(self):
        """Test failure after max retry attempts."""
        call_count = 0

        @async_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise DatabaseTimeoutError("Test", 30.0, "query")

        with pytest.raises(DatabaseTimeoutError):
            await always_fails()

        assert call_count == 3

    async def test_async_retry_no_retry_on_non_transient_error(self):
        """Test no retry on non-transient errors."""
        call_count = 0

        @async_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
        async def validation_error():
            nonlocal call_count
            call_count += 1
            raise DataValidationError("Bad data")

        with pytest.raises(DataValidationError):
            await validation_error()

        # Should fail immediately, no retries
        assert call_count == 1

    async def test_async_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times: List[float] = []

        @async_retry_with_backoff(config=RetryConfig(
            max_attempts=3,
            initial_wait=0.1,
            multiplier=1.0  # Using multiplier=1.0 to get predictable short waits
        ))
        async def timed_failure():
            call_times.append(time.time())
            raise DatabaseConnectionError("Test", "Connection failed")

        with pytest.raises(DatabaseConnectionError):
            await timed_failure()

        # With multiplier=1.0, tenacity uses: wait = max(min, multiplier * 2^(n-1))
        # Attempt 1 fails: wait = max(0.1, 1.0 * 2^0) = max(0.1, 1.0) = 1.0s
        # Attempt 2 fails: wait = max(0.1, 1.0 * 2^1) = max(0.1, 2.0) = 2.0s
        assert len(call_times) == 3

        # Allow tolerance for timing variations
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        assert 0.9 <= delay1 <= 1.2  # ~1.0s
        assert 1.8 <= delay2 <= 2.3  # ~2.0s


class TestSyncRetryDecorator:
    """Test sync retry decorator."""

    def test_sync_retry_success_first_attempt(self):
        """Test successful operation on first attempt."""
        call_count = 0

        @sync_retry_with_backoff()
        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_operation()
        assert result == "success"
        assert call_count == 1

    def test_sync_retry_success_after_retries(self):
        """Test successful operation after retries."""
        call_count = 0

        @sync_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DatabaseConnectionError("Test", "Connection failed")
            return "success"

        result = flaky_operation()
        assert result == "success"
        assert call_count == 3

    def test_sync_retry_failure_after_max_attempts(self):
        """Test failure after max retry attempts."""
        call_count = 0

        @sync_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise DatabaseTimeoutError("Test", 30.0, "query")

        with pytest.raises(DatabaseTimeoutError):
            always_fails()

        assert call_count == 3


@pytest.mark.asyncio
class TestRetryAsyncOperation:
    """Test retry_async_operation function."""

    async def test_retry_async_operation_success(self):
        """Test retry_async_operation with successful operation."""
        async def successful_op(value):
            return value * 2

        result = await retry_async_operation(
            successful_op,
            21,
            config=RetryConfig(max_attempts=3, initial_wait=0.1)
        )
        assert result == 42

    async def test_retry_async_operation_with_retries(self):
        """Test retry_async_operation with transient failures."""
        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DatabaseConnectionError("Test", "Connection failed")
            return "success"

        result = await retry_async_operation(
            flaky_op,
            config=RetryConfig(max_attempts=3, initial_wait=0.1),
            operation_name="test_operation"
        )
        assert result == "success"
        assert call_count == 3

    async def test_retry_async_operation_max_attempts_exceeded(self):
        """Test retry_async_operation fails after max attempts."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise DatabaseTimeoutError("Test", 30.0, "query")

        with pytest.raises(DatabaseTimeoutError):
            await retry_async_operation(
                always_fails,
                config=RetryConfig(max_attempts=2, initial_wait=0.1)
            )

        assert call_count == 2

    async def test_retry_async_operation_with_args_kwargs(self):
        """Test retry_async_operation passes args and kwargs correctly."""
        async def operation_with_params(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await retry_async_operation(
            operation_with_params,
            "arg1",
            "arg2",
            c="kwarg1",
            config=RetryConfig(max_attempts=1)
        )
        assert result == "arg1-arg2-kwarg1"


class TestRetrySyncOperation:
    """Test retry_sync_operation function."""

    def test_retry_sync_operation_success(self):
        """Test retry_sync_operation with successful operation."""
        def successful_op(value):
            return value * 2

        result = retry_sync_operation(
            successful_op,
            21,
            config=RetryConfig(max_attempts=3, initial_wait=0.1)
        )
        assert result == 42

    def test_retry_sync_operation_with_retries(self):
        """Test retry_sync_operation with transient failures."""
        call_count = 0

        def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise DatabaseConnectionError("Test", "Connection failed")
            return "success"

        result = retry_sync_operation(
            flaky_op,
            config=RetryConfig(max_attempts=3, initial_wait=0.1)
        )
        assert result == "success"
        assert call_count == 2


@pytest.mark.asyncio
class TestRetryIntegration:
    """Integration tests for retry functionality."""

    async def test_retry_with_different_exception_types(self):
        """Test retry behavior with different exception types."""
        # Transient errors should be retried
        transient_errors = [
            DatabaseConnectionError("Test", "Error"),
            DatabaseTimeoutError("Test", 30.0, "query"),
            ConnectionError("Error"),
        ]

        for error in transient_errors:
            call_count = 0

            @async_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
            async def operation():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise error
                return "success"

            result = await operation()
            assert result == "success"
            assert call_count == 3, f"Expected 3 calls for {type(error).__name__}"

    async def test_retry_respects_mcp_error_retryability(self):
        """Test retry respects MCPServerError.is_retryable()."""
        # Non-retryable MCP error
        call_count = 0

        @async_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
        async def non_retryable_mcp_error():
            nonlocal call_count
            call_count += 1
            raise MCPServerError("Test", -32602, "Invalid params")

        with pytest.raises(MCPServerError):
            await non_retryable_mcp_error()

        # Should not retry non-retryable errors
        assert call_count == 1

        # Retryable MCP error
        call_count = 0

        @async_retry_with_backoff(config=RetryConfig(max_attempts=3, initial_wait=0.1))
        async def retryable_mcp_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise MCPServerError("Test", -32000, "Server error")
            return "success"

        result = await retryable_mcp_error()
        assert result == "success"
        assert call_count == 2

    async def test_retry_with_aggressive_config(self):
        """Test retry with aggressive configuration."""
        call_count = 0

        @async_retry_with_backoff(config=AGGRESSIVE_RETRY_CONFIG)
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise DatabaseConnectionError("Test", "Error")
            return "success"

        result = await operation()
        assert result == "success"
        assert call_count == 5  # Should allow 5 attempts

    async def test_retry_with_conservative_config(self):
        """Test retry with conservative configuration."""
        call_count = 0

        @async_retry_with_backoff(config=CONSERVATIVE_RETRY_CONFIG)
        async def operation():
            nonlocal call_count
            call_count += 1
            raise DatabaseConnectionError("Test", "Error")

        with pytest.raises(DatabaseConnectionError):
            await operation()

        assert call_count == 2  # Conservative allows only 2 attempts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
