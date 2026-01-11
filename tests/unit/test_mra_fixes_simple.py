"""
Simple unit tests for MRA timeout and resource management fixes.

Tests validate that the hanging issue (77+ minutes) is resolved.
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock


@pytest.mark.unit
class TestMRAFixes:
    """Test core MRA fixes for hanging issue."""

    def test_timeout_configuration(self):
        """Test that timeout configuration is set correctly."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        # Verify all 8 steps have timeouts
        assert len(scenario.step_timeouts) == 8
        assert all(timeout > 0 for timeout in scenario.step_timeouts.values())

        # Verify longest timeouts are for complex steps
        assert scenario.step_timeouts[7] == 180  # Simulation: 3 minutes
        assert scenario.step_timeouts[6] == 120  # Construction: 2 minutes
        assert scenario.step_timeouts[5] == 90   # Pathway impact: 1.5 minutes

    def test_semaphore_lazy_initialization(self):
        """Test semaphore is lazy-initialized to avoid event loop issues."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        # Should have _request_semaphore attribute (not initialized yet)
        assert hasattr(scenario, '_request_semaphore')
        assert scenario._request_semaphore is None
        assert scenario.max_concurrent_requests == 3

    def test_semaphore_initialization_on_use(self):
        """Test semaphore is created when first used."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        # Get semaphore (should trigger lazy initialization)
        semaphore = scenario._get_request_semaphore()

        # Now semaphore should be created
        assert scenario._request_semaphore is not None
        assert scenario._request_semaphore._value == 3

    async def test_semaphore_execution(self):
        """Test semaphore-wrapped execution works."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        call_count = 0

        async def test_coro():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await scenario._execute_with_semaphore(test_coro)

        assert result == "success"
        assert call_count == 1

    async def test_step_with_timeout_fast(self):
        """Test fast step completes successfully."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        async def quick_step():
            return {'result': 'success'}

        result = await scenario._step_with_timeout(1, quick_step)

        assert result == {'result': 'success'}

    async def test_step_with_timeout_slow_fallback(self):
        """Test slow step triggers fallback."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        async def slow_step():
            await asyncio.sleep(2.0)
            return {'result': 'should_not_return'}

        result = await scenario._step_with_timeout(1, slow_step)

        # Should get fallback for step 1
        assert 'resolved_targets' in result
        assert result['resolution_accuracy'] == 0.0

    async def test_fallback_for_all_steps(self):
        """Test each step has a fallback strategy."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        scenario = MultiTargetSimulationScenario(mock_manager)

        for step_num in range(1, 9):
            async def always_timeout():
                await asyncio.sleep(10)
                return {}

            result = await scenario._step_with_timeout(step_num, always_timeout)

            # Should get valid fallback for every step
            assert result is not None
            assert isinstance(result, dict)

    def test_progress_monitor_exists(self):
        """Test progress monitor class exists."""
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

    def test_time_formatting(self):
        """Test time formatting utilities."""
        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        monitor = MRASimulationProgress(total_targets=11)

        # Test _format_time
        assert monitor._format_time(0) == "00:00:00"
        assert monitor._format_time(60) == "00:01:00"
        assert monitor._format_time(3661) == "01:01:01"

        # Test _format_duration
        assert "30.0s" in monitor._format_duration(30)
        assert "1m" in monitor._format_duration(90)
        assert "1h" in monitor._format_duration(3720)

    async def test_full_workflow_mocked(self):
        """Test full workflow doesn't hang with mocked MCP."""
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        mock_manager = Mock()
        mock_manager.string.search_proteins = AsyncMock(return_value={
            'proteins': [{'gene_symbol': 'AXL', 'confidence': 0.9}]
        })
        mock_manager.hpa.search_proteins = AsyncMock(return_value=[])
        mock_manager.reactome.find_pathways_by_disease = AsyncMock(return_value={'pathways': []})
        mock_manager.string.get_interaction_network = AsyncMock(return_value={'nodes': [], 'edges': []})
        mock_manager.hpa.get_tissue_expression = AsyncMock(return_value={})
        mock_manager.reactome.get_pathway_participants = AsyncMock(return_value={'participants': []})
        mock_manager.string.get_functional_enrichment = AsyncMock(return_value={'enrichment': {}})

        scenario = MultiTargetSimulationScenario(mock_manager)

        # Run with 1 target - should complete quickly
        result = await scenario.execute(
            targets=['AXL'],
            disease_context='breast cancer',
            simulation_mode='simple'
        )

        # Should not hang and return valid result
        assert result is not None
        assert hasattr(result, 'targets')


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-m', 'unit'])
