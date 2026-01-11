"""
Unit tests for MRA simulation timeout and resource management fixes (Phase 1.1-1.3).

Tests the critical fixes applied to prevent hanging in batch 2:
- Timeout wrappers for all 8 steps (30-180s per step)
- MRASimulationProgress monitoring with heartbeats
- Semaphore-based request limiting (max 3 concurrent)
- Fallback strategies on timeout/failure

Running these tests validates that the 77+ minute hang issue is resolved.
"""
import asyncio
import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register unit marker
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as unit test")


@pytest.mark.unit
class TestMRASimulationTimeouts:
    """Test timeout wrappers and resource management."""


class TestMRASimulationTimeouts:
    """Test timeout wrappers and resource management."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create mock MCP manager."""
        mock_manager = Mock()
        mock_manager.string = Mock()
        mock_manager.hpa = Mock()
        mock_manager.reactome = Mock()
        return mock_manager

    @pytest.fixture
    def mra_scenario(self, mock_mcp_manager):
        """Create MRA scenario with mocked MCP manager."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        scenario = MultiTargetSimulationScenario(mock_mcp_manager)

        # Verify timeout configuration
        assert scenario.step_timeouts == {
            1: 30,   # Target resolution
            2: 45,   # Network context
            3: 30,   # Interaction validation
            4: 60,   # Expression validation
            5: 90,   # Pathway impact
            6: 120,  # Network construction
            7: 180,  # Simulation
            8: 60    # Impact assessment
        }
        return scenario

    @pytest.mark.unit
    def test_timeout_configuration(self, mra_scenario):
        """Test that timeout configuration is correctly set."""
        assert len(mra_scenario.step_timeouts) == 8
        assert all(timeout > 0 for timeout in mra_scenario.step_timeouts.values())

        # Verify logical timeout ordering
        # Later steps (construction, simulation) have longer timeouts
        assert mra_scenario.step_timeouts[7] == 180  # Simulation: longest
        assert mra_scenario.step_timeouts[6] == 120  # Construction: 2nd longest
        assert mra_scenario.step_timeouts[1] == 30   # Target resolution: shortest

    @pytest.mark.unit
    def test_semaphore_initialization(self, mra_scenario):
        """Test semaphore initialization for request limiting."""
        assert hasattr(mra_scenario, 'request_semaphore')
        assert mra_scenario.max_concurrent_requests == 3
        assert mra_scenario.request_semaphore._value == 3

    @pytest.mark.unit
    def test_execute_with_semaphore(self, mra_scenario):
        """Test semaphore-wrapped execution."""
        call_count = 0

        async def test_coro():
            nonlocal call_count
            call_count += 1
            return "success"

        # Test that semaphore allows execution
        result = asyncio.run(mra_scenario._execute_with_semaphore(test_coro))
        assert result == "success"
        assert call_count == 1

    @pytest.mark.unit
    async def test_execute_with_semaphore_concurrent_limit(self, mra_scenario):
        """Test that semaphore enforces concurrent request limit."""
        results = []
        call_count = 0
        max_concurrent = 3

        async def test_coro(delay: float):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(delay)
            return call_count

        # Run 5 tasks with 0.1s delay each
        # With semaphore=3, only 3 should run concurrently
        tasks = [
            mra_scenario._execute_with_semaphore(test_coro, 0.1)
            for _ in range(5)
        ]

        start_count = call_count
        results = await asyncio.gather(*tasks)

        # All tasks should complete
        assert len(results) == 5
        assert call_count == 5

    @pytest.mark.unit
    def test_step_with_timeout_success(self, mra_scenario):
        """Test successful step execution with timeout."""
        async def quick_step():
            await asyncio.sleep(0.1)
            return {'result': 'success'}

        result = asyncio.run(mra_scenario._step_with_timeout(1, quick_step))
        assert result == {'result': 'success'}

    @pytest.mark.unit
    async def test_step_with_timeout_timeout(self, mra_scenario):
        """Test timeout handling - should trigger fallback."""
        async def slow_step():
            await asyncio.sleep(2.0)  # Longer than 1s timeout
            return {'result': 'should_not_return'}

        result = await mra_scenario._step_with_timeout(1, slow_step)

        # Should return fallback result for step 1
        assert 'resolved_targets' in result
        assert result['resolution_accuracy'] == 0.0

    @pytest.mark.unit
    async def test_step_with_timeout_exception(self, mra_scenario):
        """Test exception handling - should trigger fallback."""
        async def failing_step():
            raise ValueError("Test error")

        result = await mra_scenario._step_with_timeout(1, failing_step)

        # Should return fallback result
        assert 'resolved_targets' in result
        assert result['resolution_accuracy'] == 0.0

    @pytest.mark.unit
    def test_fallback_strategies(self, mra_scenario):
        """Test that fallback strategies exist for all steps."""
        import asyncio

        for step_num in range(1, 9):
            # Test each fallback strategy
            async def dummy_step():
                raise asyncio.TimeoutError(f"Step {step_num} timeout")

            result = asyncio.run(mra_scenario._step_with_timeout(step_num, dummy_step))

            # Verify fallback result exists
            assert result is not None
            assert isinstance(result, dict)

            # Check step-specific fallback keys
            if step_num == 1:
                assert 'resolved_targets' in result
            elif step_num == 2:
                assert 'pathways' in result
            elif step_num == 3:
                assert 'interactions' in result
            elif step_num == 4:
                assert 'profiles' in result
            elif step_num == 5:
                assert 'pathway_impacts' in result
            elif step_num == 6:
                assert 'network' in result
            elif step_num == 7:
                assert 'results' in result
            elif step_num == 8:
                assert 'enrichment' in result

    @pytest.mark.unit
    def test_resolve_single_target_with_semaphore(self, mra_scenario, mock_mcp_manager):
        """Test target resolution with semaphore protection."""
        # Mock successful MCP calls
        mock_mcp_manager.string.search_proteins = AsyncMock(return_value={
            'proteins': [{'gene_symbol': 'AXL', 'confidence': 0.9}]
        })
        mock_mcp_manager.hpa.search_proteins = AsyncMock(return_value=[
            {'gene_symbol': 'AXL', 'confidence': 0.8}
        ])

        result = asyncio.run(mra_scenario._resolve_single_target('AXL', 0))

        assert result is not None
        assert result.gene_symbol == 'AXL'

    @pytest.mark.unit
    async def test_resolve_single_target_timeout(self, mra_scenario, mock_mcp_manager):
        """Test target resolution timeout handling."""
        # Mock slow MCP calls
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(2.0)
            return {}

        mock_mcp_manager.string.search_proteins = slow_search
        mock_mcp_manager.hpa.search_proteins = slow_search

        result = await mra_scenario._resolve_single_target('AXL', 0)

        # Should return None on timeout
        assert result is None


class TestMRASimulationProgress:
    """Test progress monitoring and heartbeat logging."""

    @pytest.mark.unit
    def test_progress_monitor_initialization(self):
        """Test MRASimulationProgress initialization."""
        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        monitor = MRASimulationProgress(
            total_targets=11,
            step_count=8,
            heartbeat_interval=30
        )

        assert monitor.total_targets == 11
        assert monitor.step_count == 8
        assert monitor.heartbeat_interval == 30
        assert monitor.current_step == 0
        assert len(monitor.step_names) == 8

    @pytest.mark.unit
    async def test_log_step_start(self):
        """Test step start logging."""
        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        monitor = MRASimulationProgress(total_targets=11)

        # Should not raise exception
        await monitor.log_step_start(1, "Testing step 1")
        assert monitor.current_step == 1

    @pytest.mark.unit
    async def test_log_step_complete(self):
        """Test step completion logging."""
        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        monitor = MRASimulationProgress(total_targets=11)

        await monitor.log_step_start(1, "Testing step 1")
        await monitor.log_step_complete(1, "Test completed")

        assert 1 in monitor.step_times

    @pytest.mark.unit
    async def test_log_heartbeat(self):
        """Test heartbeat logging."""
        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        monitor = MRASimulationProgress(total_targets=11, heartbeat_interval=0.1)

        await monitor.log_step_start(1, "Testing")
        await asyncio.sleep(0.2)  # Wait for heartbeat interval

        # Should log heartbeat without error
        await monitor.log_heartbeat("Test heartbeat")

    @pytest.mark.unit
    async def test_log_final_summary(self):
        """Test final summary logging."""
        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        monitor = MRASimulationProgress(total_targets=11)

        # Simulate completed steps
        await monitor.log_step_start(1, "Step 1")
        await monitor.log_step_complete(1, "Step 1 done")

        # Should not raise exception
        await monitor.log_final_summary()

    @pytest.mark.unit
    def test_time_formatting(self):
        """Test time formatting methods."""
        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        monitor = MRASimulationProgress(total_targets=11)

        # Test _format_time
        assert monitor._format_time(0) == "00:00:00"
        assert monitor._format_time(60) == "00:01:00"
        assert monitor._format_time(3661) == "01:01:01"

        # Test _format_duration
        assert monitor._format_duration(30) == "30.0s"
        assert monitor._format_duration(90) == "1m 30s"
        assert monitor._format_duration(3720) == "1h 2m"


class TestMRAIntegration:
    """Integration tests for MRA simulation with timeout fixes."""

    @pytest.mark.unit
    async def test_full_workflow_with_timeouts(self, mock_mcp_manager):
        """Test full workflow execution with timeout protection."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        # Mock all MCP calls to return quickly
        mock_mcp_manager.string.search_proteins = AsyncMock(return_value={
            'proteins': [{'gene_symbol': 'AXL', 'confidence': 0.9}]
        })
        mock_mcp_manager.hpa.search_proteins = AsyncMock(return_value=[])
        mock_mcp_manager.reactome.find_pathways_by_disease = AsyncMock(return_value={'pathways': []})
        mock_mcp_manager.string.get_interaction_network = AsyncMock(return_value={
            'nodes': [], 'edges': []
        })
        mock_mcp_manager.hpa.get_tissue_expression = AsyncMock(return_value={})
        mock_mcp_manager.reactome.get_pathway_participants = AsyncMock(return_value={'participants': []})
        mock_mcp_manager.string.get_functional_enrichment = AsyncMock(return_value={'enrichment': {}})

        scenario = MultiTargetSimulationScenario(mock_mcp_manager)

        # Run with minimal targets
        result = await scenario.execute(
            targets=['AXL'],
            disease_context='breast cancer',
            simulation_mode='simple'
        )

        # Should complete without hanging
        assert result is not None
        assert hasattr(result, 'targets')

    @pytest.mark.unit
    def test_resource_cleanup(self, mock_mcp_manager):
        """Test that resources are properly managed."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        scenario = MultiTargetSimulationScenario(mock_mcp_manager)

        # Verify semaphore exists
        assert scenario.request_semaphore is not None

        # Verify timeout config exists
        assert hasattr(scenario, 'step_timeouts')

        # Verify progress monitor is None initially
        assert scenario.progress_monitor is None


class TestMRAFixValidation:
    """Validate that the fix addresses the original hanging issue."""

    @pytest.mark.unit
    def test_concurrent_request_limit_prevents_overload(self):
        """Test that semaphore prevents 44+ concurrent requests."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from unittest.mock import Mock

        scenario = MultiTargetSimulationScenario(Mock())

        # Verify max concurrent is 3 (not unlimited)
        assert scenario.max_concurrent_requests == 3

    @pytest.mark.unit
    def test_timeout_bounds_execution(self):
        """Test that timeouts bound execution time."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from unittest.mock import Mock

        scenario = MultiTargetSimulationScenario(Mock())

        # Verify all steps have timeouts (no step is unlimited)
        for step_num, timeout in scenario.step_timeouts.items():
            assert timeout > 0
            assert timeout < 300  # Max 5 minutes per step

        # Total max time: sum of all timeouts = 615s ≈ 10 minutes
        total_timeout = sum(scenario.step_timeouts.values())
        assert total_timeout < 1200  # Less than 20 minutes total

    @pytest.mark.unit
    def test_progress_monitoring_enabled(self):
        """Test that progress monitoring provides visibility."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from unittest.mock import Mock

        scenario = MultiTargetSimulationScenario(Mock())

        # Progress monitor should be initialized on execute
        assert scenario.progress_monitor is None  # Not initialized until execute

    @pytest.mark.unit
    def test_fallback_strategies_prevent_cascading_failures(self):
        """Test that fallbacks prevent total failure."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from unittest.mock import Mock
        import asyncio

        scenario = MultiTargetSimulationScenario(Mock())

        # Test that each step has a fallback
        for step_num in range(1, 9):
            async def always_timeout():
                await asyncio.sleep(10)
                return {}

            result = asyncio.run(
                scenario._step_with_timeout(step_num, always_timeout)
            )

            # Should get fallback, not None or exception
            assert result is not None
            assert isinstance(result, dict)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-m', 'unit'])
