"""
OmniTarget Circuit Breaker Pattern

Prevents cascading failures by failing fast when a service is unavailable.
Part of P0-2: Error Handling & Retry Logic critical fix.

Author: OmniTarget Team
Date: 2025-01-06
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from functools import wraps

from .exceptions import (
    DatabaseUnavailableError,
    is_transient_error,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker States
# =============================================================================

class CircuitState(Enum):
    """
    Circuit breaker states.

    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, fail fast without trying
    - HALF_OPEN: Testing if service has recovered
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# Circuit Breaker Configuration
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit (default: 5)
        success_threshold: Number of successes in HALF_OPEN to close circuit (default: 2)
        timeout: Seconds to wait in OPEN before trying HALF_OPEN (default: 60)
        half_open_max_calls: Max concurrent calls allowed in HALF_OPEN (default: 1)
        expected_exception: Exception type that counts as failure (default: Exception)
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_max_calls: int = 1
    expected_exception: type = Exception

    def __post_init__(self):
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.timeout < 0:
            raise ValueError("timeout must be >= 0")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    The circuit breaker tracks failures and successes, transitioning between
    states to protect the system from repeated calls to a failing service.

    State Transitions:
        CLOSED --[failure_threshold failures]--> OPEN
        OPEN --[timeout expires]--> HALF_OPEN
        HALF_OPEN --[success_threshold successes]--> CLOSED
        HALF_OPEN --[any failure]--> OPEN

    Example:
        >>> breaker = CircuitBreaker(
        ...     name="kegg",
        ...     config=CircuitBreakerConfig(failure_threshold=5, timeout=60)
        ... )
        >>>
        >>> @breaker.call
        ... async def fetch_data():
        ...     return await mcp_client.call_tool("get_data", {})
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            name: Human-readable name for logging (e.g., "KEGG", "Reactome")
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        # Thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._state_changes: Dict[str, int] = {
            "closed_to_open": 0,
            "open_to_half_open": 0,
            "half_open_to_closed": 0,
            "half_open_to_open": 0,
        }

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def get_stats(self) -> Dict[str, Any]:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with statistics including calls, failures, successes, state changes
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "state_changes": self._state_changes.copy(),
            "last_failure_time": self._last_failure_time,
        }

    async def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        if self._state != CircuitState.OPEN:
            old_state = self._state.value
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            self._state_changes["closed_to_open" if old_state == "closed" else "half_open_to_open"] += 1

            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self._failure_count} failures. "
                f"Will retry in {self.config.timeout}s."
            )

    async def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        if self._state == CircuitState.OPEN:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            self._failure_count = 0
            self._half_open_calls = 0
            self._state_changes["open_to_half_open"] += 1

            logger.info(
                f"Circuit breaker '{self.name}' half-open. "
                f"Testing service recovery (need {self.config.success_threshold} successes)."
            )

    async def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._state_changes["half_open_to_closed"] += 1

            logger.info(
                f"Circuit breaker '{self.name}' closed. "
                f"Service recovered, normal operation resumed."
            )

    async def _should_allow_request(self) -> bool:
        """
        Determine if request should be allowed based on circuit state.

        Returns:
            True if request should proceed, False if should fail fast
        """
        async with self._lock:
            # CLOSED: always allow
            if self._state == CircuitState.CLOSED:
                return True

            # OPEN: check if timeout expired
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is None:
                    # Should not happen, but handle gracefully
                    await self._transition_to_half_open()
                    return True

                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout:
                    # Timeout expired, try recovery
                    await self._transition_to_half_open()
                    return True

                # Still in timeout period, fail fast
                return False

            # HALF_OPEN: limit concurrent calls
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                # Too many concurrent calls in HALF_OPEN, fail fast
                return False

            return False

    async def _on_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._total_calls += 1
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                self._success_count += 1

                if self._success_count >= self.config.success_threshold:
                    # Enough successes, close circuit
                    await self._transition_to_closed()
                else:
                    logger.debug(
                        f"Circuit breaker '{self.name}' HALF_OPEN: "
                        f"{self._success_count}/{self.config.success_threshold} successes"
                    )

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in CLOSED state
                self._failure_count = 0

    async def _on_failure(self, exception: Exception) -> None:
        """
        Record failed call.

        Args:
            exception: The exception that occurred
        """
        async with self._lock:
            self._total_calls += 1
            self._total_failures += 1

            # Only count transient errors as failures
            # Programming errors (TypeError, AttributeError) should not open circuit
            if not is_transient_error(exception):
                logger.debug(
                    f"Circuit breaker '{self.name}': Non-transient error ignored "
                    f"({type(exception).__name__})"
                )
                return

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN reopens circuit
                self._half_open_calls = max(0, self._half_open_calls - 1)
                logger.warning(
                    f"Circuit breaker '{self.name}' failure in HALF_OPEN state. "
                    f"Reopening circuit."
                )
                await self._transition_to_open()

            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1

                if self._failure_count >= self.config.failure_threshold:
                    # Too many failures, open circuit
                    await self._transition_to_open()
                else:
                    logger.debug(
                        f"Circuit breaker '{self.name}' CLOSED: "
                        f"{self._failure_count}/{self.config.failure_threshold} failures"
                    )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function call

        Raises:
            DatabaseUnavailableError: If circuit is open
            Exception: Original exception from function if circuit allows call
        """
        # Check if request should be allowed
        allowed = await self._should_allow_request()

        if not allowed:
            # Calculate retry_after based on remaining timeout
            retry_after = None
            if self._state == CircuitState.OPEN and self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                retry_after = max(1, int(self.config.timeout - elapsed))

            raise DatabaseUnavailableError(
                server_name=self.name,
                reason="Circuit breaker is OPEN",
                retry_after=retry_after
            )

        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap async function with circuit breaker.

        Args:
            func: Async function to wrap

        Returns:
            Wrapped function

        Example:
            >>> breaker = CircuitBreaker("my_service")
            >>>
            >>> @breaker
            ... async def my_function():
            ...     return await risky_operation()
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper


# =============================================================================
# Circuit Breaker Manager
# =============================================================================

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers.

    Provides centralized management of circuit breakers for different services.

    Example:
        >>> manager = CircuitBreakerManager()
        >>> manager.add_breaker("kegg", CircuitBreakerConfig(failure_threshold=5))
        >>> manager.add_breaker("reactome", CircuitBreakerConfig(failure_threshold=3))
        >>>
        >>> # Use specific breaker
        >>> result = await manager.call("kegg", fetch_kegg_data)
        >>>
        >>> # Get all statistics
        >>> stats = manager.get_all_stats()
    """

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def add_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Add circuit breaker for a service.

        Args:
            name: Service name (e.g., "KEGG", "Reactome")
            config: Circuit breaker configuration

        Returns:
            The created circuit breaker

        Raises:
            ValueError: If breaker with name already exists
        """
        if name in self._breakers:
            raise ValueError(f"Circuit breaker '{name}' already exists")

        breaker = CircuitBreaker(name, config)
        self._breakers[name] = breaker
        logger.info(f"Added circuit breaker for '{name}'")
        return breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get circuit breaker by name.

        Args:
            name: Service name

        Returns:
            Circuit breaker or None if not found
        """
        return self._breakers.get(name)

    async def call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through named circuit breaker.

        Args:
            name: Service name
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function call

        Raises:
            ValueError: If circuit breaker not found
            DatabaseUnavailableError: If circuit is open
            Exception: Original exception from function
        """
        breaker = self.get_breaker(name)
        if not breaker:
            raise ValueError(f"Circuit breaker '{name}' not found")

        return await breaker.call(func, *args, **kwargs)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all circuit breakers.

        Returns:
            Dictionary mapping service names to their statistics
        """
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker._state = CircuitState.CLOSED
            breaker._failure_count = 0
            breaker._success_count = 0
            breaker._last_failure_time = None
        logger.info("Reset all circuit breakers")


# =============================================================================
# Global Circuit Breaker Manager
# =============================================================================

# Global manager instance for convenience
_global_manager = CircuitBreakerManager()


def get_global_manager() -> CircuitBreakerManager:
    """
    Get global circuit breaker manager.

    Returns:
        Global CircuitBreakerManager instance
    """
    return _global_manager
