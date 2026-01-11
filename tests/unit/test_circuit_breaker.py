"""
Unit tests for Circuit Breaker Pattern.
"""

import asyncio
import pytest
from src.core.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerManager
)


@pytest.mark.unit
class TestCircuitBreakerConfig:
    def test_default_config(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0


@pytest.mark.unit
class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        return CircuitBreaker("test_service")

    def test_initial_state(self, breaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker):
        async def mock_func():
            return "success"
        result = await breaker.call(mock_func)
        assert result == "success"


@pytest.mark.unit
class TestCircuitBreakerManager:
    @pytest.fixture
    def manager(self):
        return CircuitBreakerManager()

    def test_add_breaker(self, manager):
        breaker = manager.add_breaker("test_service")
        assert breaker.name == "test_service"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'unit'])
