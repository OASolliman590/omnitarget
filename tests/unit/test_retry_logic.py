"""
Test retry logic with exponential backoff.

Validates that the retry decorator properly handles failures and retries.
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock


@pytest.mark.unit
class TestRetryLogic:
    """Test retry decorator functionality."""

    def test_retry_decorator_exists(self):
        """Test that retry decorator is defined."""
        from src.scenarios.scenario_4_mra_simulation import retry_with_exponential_backoff

        assert retry_with_exponential_backoff is not None
        assert callable(retry_with_exponential_backoff)

    def test_retry_successful_first_attempt(self):
        """Test that successful calls don't retry."""
        from src.scenarios.scenario_4_mra_simulation import retry_with_exponential_backoff

        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, base_delay=0.1)
        async def test_success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = asyncio.run(test_success())

        assert result == "success"
        assert call_count == 1  # Only called once

    async def test_retry_successful_second_attempt(self):
        """Test that failures are retried and eventually succeed."""
        from src.scenarios.scenario_4_mra_simulation import retry_with_exponential_backoff

        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, base_delay=0.1, max_delay=1.0)
        async def test_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await test_retry()

        assert result == "success"
        assert call_count == 2  # Called twice before success

    async def test_retry_max_attempts_exceeded(self):
        """Test that max attempts limit is enforced."""
        from src.scenarios.scenario_4_mra_simulation import retry_with_exponential_backoff

        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, base_delay=0.1)
        async def test_failure():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            await test_failure()

        assert call_count == 3  # Tried all 3 attempts

    def test_exponential_backoff_delay_calculation(self):
        """Test that exponential backoff delays are calculated correctly."""
        from src.scenarios.scenario_4_mra_simulation import retry_with_exponential_backoff

        # The decorator should calculate delays: 1s, 2s, 4s (with max of 3s)
        # For max_attempts=3, base_delay=1.0, max_delay=3.0, exponential_base=2.0

        call_count = 0
        delays = []

        @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0, max_delay=3.0, jitter=False)
        async def test_delays():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Fail")
            return "success"

        # Mock sleep to capture delays
        original_sleep = asyncio.sleep
        captured_delays = []

        async def mock_sleep(delay):
            captured_delays.append(delay)
            return original_sleep(0)  # Don't actually sleep

        asyncio.sleep = mock_sleep

        try:
            result = asyncio.run(test_delays())
            assert result == "success"
            assert call_count == 3

            # Should have 2 delays (between attempts 1-2 and 2-3)
            assert len(captured_delays) == 2
            # First delay: 1.0 * 2^0 = 1.0
            assert captured_delays[0] == 1.0
            # Second delay: 1.0 * 2^1 = 2.0
            assert captured_delays[1] == 2.0
        finally:
            asyncio.sleep = original_sleep

    async def test_retry_with_different_exception_types(self):
        """Test that retry works with different exception types."""
        from src.scenarios.scenario_4_mra_simulation import retry_with_exponential_backoff

        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, base_delay=0.1)
        async def test_timeout():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError("Timeout")
            return "success"

        result = await test_timeout()

        assert result == "success"
        assert call_count == 2

    def test_retry_applied_to_resolve_single_target(self):
        """Test that retry is applied to _resolve_single_target method."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        # Check that the method has the retry decorator
        assert hasattr(scenario._resolve_single_target, '__wrapped__')

    def test_retry_applied_to_step2_network_context(self):
        """Test that retry is applied to _step2_network_context method."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        # Check that the method has the retry decorator
        assert hasattr(scenario._step2_network_context, '__wrapped__')

    def test_retry_applied_to_extract_reactome_pathway_genes(self):
        """Test that retry is applied to _extract_reactome_pathway_genes method."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        # Check that the method has the retry decorator
        assert hasattr(scenario._extract_reactome_pathway_genes, '__wrapped__')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'unit'])
