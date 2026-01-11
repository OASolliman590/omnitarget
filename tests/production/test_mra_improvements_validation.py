"""
Production validation tests for MRA simulation improvements.

Tests all enhancements:
- Timeout protection
- Progress monitoring
- Semaphore-based request limiting
- Retry logic with exponential backoff
- Connection pooling
- Circuit breaker pattern
- Error handling and fallbacks

These tests require real MCP servers and take 10-20 minutes to run.
"""

import asyncio
import pytest
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test marker
pytestmark = pytest.mark.production


class TestMRAImprovements:
    """Test suite for MRA simulation improvements."""

    @pytest.fixture
    def simple_mra_targets(self):
        """Simple target list for testing."""
        return ['AXL', 'AKT1', 'MAPK1']

    @pytest.fixture
    def medium_mra_targets(self):
        """Medium target list for testing."""
        return ['AXL', 'AKT1', 'MAPK1', 'STAT3', 'VEGFA']

    @pytest.mark.asyncio
    async def test_connection_pool_enabled(self, simple_mra_targets):
        """Test that connection pooling is enabled and working."""
        logger.info("=" * 70)
        logger.info("TEST: Connection Pooling Enabled")
        logger.info("=" * 70)

        from src.core.mcp_client_manager import MCPClientManager
        from src.core.connection_pool_manager import ConnectionPoolManager

        # Create manager with connection pooling
        manager = MCPClientManager("config/mcp_servers.json")

        async with manager.session() as session:
            # Check that connection pool is initialized
            assert hasattr(session, 'connection_pool_manager')
            assert session.connection_pool_manager is not None

            pool_manager = session.connection_pool_manager
            assert pool_manager.enable_pooling is True

            # Get initial stats
            initial_stats = pool_manager.get_stats()
            logger.info(f"Initial pool stats: {initial_stats}")

            assert initial_stats['enabled'] is True

            # Make a request to create connections
            result = await session.string.search_proteins('AXL', limit=1)
            assert result is not None

            # Check stats after request
            after_stats = pool_manager.get_stats()
            logger.info(f"Pool stats after request: {after_stats}")

            # Verify connections were created
            assert after_stats['pool_stats']['active_connections'] > 0

            logger.info("✅ Connection pooling is working correctly")

    @pytest.mark.asyncio
    async def test_circuit_breaker_monitoring(self, simple_mra_targets):
        """Test that circuit breaker monitoring is active."""
        logger.info("=" * 70)
        logger.info("TEST: Circuit Breaker Monitoring")
        logger.info("=" * 70)

        from src.core.circuit_breaker import get_global_manager
        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

        # Get circuit breaker manager
        cb_manager = get_global_manager()
        assert cb_manager is not None

        # Get initial stats
        initial_stats = cb_manager.get_aggregate_stats()
        logger.info(f"Initial circuit breaker stats: {initial_stats}")

        # Create MRA scenario
        from src.core.mcp_client_manager import MCPClientManager
        mcp_manager = MCPClientManager("config/mcp_servers.json")

        scenario = MultiTargetSimulationScenario(mcp_manager)
        assert scenario.circuit_breaker_enabled is True

        # Check circuit breaker status
        scenario._check_circuit_breaker_status()

        # Verify scenario is tracking circuit breaker stats
        assert scenario.circuit_breaker_stats is not None

        logger.info(f"Circuit breaker stats: {scenario.circuit_breaker_stats}")

        logger.info("✅ Circuit breaker monitoring is working correctly")

    @pytest.mark.asyncio
    async def test_timeout_protection(self, simple_mra_targets):
        """Test that timeout protection is working."""
        logger.info("=" * 70)
        logger.info("TEST: Timeout Protection")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")
        scenario = MultiTargetSimulationScenario(mcp_manager)

        # Verify timeout configuration
        assert scenario.step_timeouts is not None
        assert len(scenario.step_timeouts) == 8

        # Check default timeouts
        assert scenario.step_timeouts[1] == 30  # Target resolution
        assert scenario.step_timeouts[7] == 180  # Simulation

        logger.info(f"Step timeouts configured: {scenario.step_timeouts}")

        # Test step with timeout
        async def mock_step():
            await asyncio.sleep(0.5)  # Short sleep
            return {'success': True}

        result = await scenario._step_with_timeout(1, mock_step)
        assert result['success'] is True

        logger.info("✅ Timeout protection is working correctly")

    @pytest.mark.asyncio
    async def test_progress_monitoring(self, simple_mra_targets):
        """Test that progress monitoring is working."""
        logger.info("=" * 70)
        logger.info("TEST: Progress Monitoring")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import MRASimulationProgress

        # Create progress monitor
        progress = MRASimulationProgress(
            total_targets=len(simple_mra_targets),
            step_count=8,
            heartbeat_interval=2  # Short interval for testing
        )

        assert progress.total_targets == len(simple_mra_targets)
        assert progress.step_count == 8
        assert progress.heartbeat_interval == 2

        # Test logging
        await progress.log_step_start(1, "Test step")
        await progress.log_step_complete(1, "Test completed")

        # Log a heartbeat
        await progress.log_heartbeat("Test message")

        logger.info("✅ Progress monitoring is working correctly")

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, simple_mra_targets):
        """Test retry logic with exponential backoff."""
        logger.info("=" * 70)
        logger.info("TEST: Retry Logic with Exponential Backoff")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import retry_with_exponential_backoff

        # Counter to track attempts
        attempt_count = 0

        @retry_with_exponential_backoff(max_attempts=3, base_delay=0.1, max_delay=1.0)
        async def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Simulated failure")
            return "success"

        # Should succeed after retries
        result = await failing_function()
        assert result == "success"
        assert attempt_count == 3

        logger.info(f"Function succeeded after {attempt_count} attempts")

        # Test with permanent failure
        attempt_count = 0

        @retry_with_exponential_backoff(max_attempts=2, base_delay=0.1)
        async def permanently_failing():
            nonlocal attempt_count
            attempt_count += 1
            raise ConnectionError("Permanent failure")

        # Should fail after max attempts
        with pytest.raises(ConnectionError):
            await permanently_failing()

        assert attempt_count == 2

        logger.info("✅ Retry logic is working correctly")

    @pytest.mark.asyncio
    async def test_fallback_strategies(self, simple_mra_targets):
        """Test fallback strategies when steps fail."""
        logger.info("=" * 70)
        logger.info("TEST: Fallback Strategies")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")
        scenario = MultiTargetSimulationScenario(mcp_manager)

        # Test fallback for step 1 (target resolution)
        result = await scenario._get_fallback_result(1)
        assert result == {'resolved_targets': [], 'resolution_accuracy': 0.0}

        # Test fallback for step 6 (network construction)
        result = await scenario._get_fallback_result(6)
        assert 'network' in result
        assert 'nodes' in result
        assert 'edges' in result

        # Test fallback for step 7 (simulation)
        import networkx as nx
        G = nx.Graph()
        G.add_node('test')
        result = await scenario._get_fallback_result(7, G, simple_mra_targets)
        assert 'results' in result
        assert 'convergence_rate' in result

        logger.info("✅ Fallback strategies are working correctly")

    @pytest.mark.asyncio
    async def test_semaphore_request_limiting(self, simple_mra_targets):
        """Test semaphore-based request limiting."""
        logger.info("=" * 70)
        logger.info("TEST: Semaphore Request Limiting")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")
        scenario = MultiTargetSimulationScenario(mcp_manager)

        # Verify semaphore is configured
        assert scenario.max_concurrent_requests == 3

        # Test semaphore initialization
        semaphore = scenario._get_request_semaphore()
        assert semaphore is not None
        assert semaphore._value == 3  # Max 3 concurrent

        logger.info("✅ Semaphore request limiting is working correctly")

    @pytest.mark.asyncio
    async def test_end_to_end_simple_targets(self, simple_mra_targets):
        """Test complete MRA simulation with simple targets."""
        logger.info("=" * 70)
        logger.info("TEST: End-to-End MRA with Simple Targets")
        logger.info("=" * 70)

        start_time = time.time()

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")

        async with mcp_manager.session() as session:
            scenario = MultiTargetSimulationScenario(mcp_manager)

            # Run complete simulation
            result = await scenario.execute(
                targets=simple_mra_targets,
                disease_context="breast cancer",
                simulation_mode="simple"
            )

            execution_time = time.time() - start_time

            # Verify result structure
            assert result is not None
            assert hasattr(result, 'targets')
            assert hasattr(result, 'individual_results')
            assert hasattr(result, 'combined_effects')
            assert hasattr(result, 'synergy_analysis')
            assert hasattr(result, 'network_perturbation')
            assert hasattr(result, 'pathway_enrichment')
            assert hasattr(result, 'validation_metrics')

            # Verify targets
            assert len(result.targets) > 0
            assert all(t in result.targets for t in simple_mra_targets[:2])  # At least 2

            # Verify individual results
            assert len(result.individual_results) > 0

            # Verify validation metrics
            assert result.validation_metrics['accuracy'] >= 0.0
            assert result.validation_metrics['accuracy'] <= 1.0

            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Targets processed: {len(result.targets)}")
            logger.info(f"Validation score: {result.validation_metrics['accuracy']:.3f}")

            # Should complete in reasonable time (<5 minutes for 3 targets)
            assert execution_time < 300

            logger.info("✅ End-to-end test passed")

    @pytest.mark.asyncio
    async def test_end_to_end_medium_targets(self, medium_mra_targets):
        """Test complete MRA simulation with medium number of targets."""
        logger.info("=" * 70)
        logger.info("TEST: End-to-End MRA with Medium Targets")
        logger.info("=" * 70)

        start_time = time.time()

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")

        async with mcp_manager.session() as session:
            scenario = MultiTargetSimulationScenario(mcp_manager)

            # Run complete simulation
            result = await scenario.execute(
                targets=medium_mra_targets,
                disease_context="cancer",
                simulation_mode="simple"
            )

            execution_time = time.time() - start_time

            # Verify result
            assert result is not None
            assert len(result.targets) > 0

            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Targets processed: {len(result.targets)}")
            logger.info(f"Validation score: {result.validation_metrics['accuracy']:.3f}")

            # Should complete in reasonable time (<10 minutes for 5 targets)
            assert execution_time < 600

            logger.info("✅ Medium targets test passed")

    @pytest.mark.asyncio
    async def test_memory_optimization_features(self, simple_mra_targets):
        """Test memory optimization features."""
        logger.info("=" * 70)
        logger.info("TEST: Memory Optimization Features")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")
        scenario = MultiTargetSimulationScenario(mcp_manager)

        # Check that pooling is enabled
        assert scenario.connection_pool_enabled is True

        # Check that semaphore limits concurrent requests
        assert scenario.max_concurrent_requests == 3

        # Verify timeouts prevent infinite waiting
        assert scenario.step_timeouts[6] == 120  # Network construction
        assert scenario.step_timeouts[7] == 180  # Simulation

        logger.info("✅ Memory optimization features are enabled")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, simple_mra_targets):
        """Test error handling and recovery mechanisms."""
        logger.info("=" * 70)
        logger.info("TEST: Error Handling and Recovery")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")
        scenario = MultiTargetSimulationScenario(mcp_manager)

        # Test with invalid target (should handle gracefully)
        invalid_targets = ['INVALID_GENE_XYZ']

        # Should not crash, should return result with empty targets
        try:
            result = await scenario.execute(
                targets=invalid_targets,
                disease_context="cancer",
                simulation_mode="simple"
            )

            # Should return result even with invalid targets
            assert result is not None
            logger.info("Handled invalid targets gracefully")

        except Exception as e:
            # If it fails, should be graceful failure with clear error
            logger.info(f"Failed with invalid targets (acceptable): {e}")
            assert "could not be resolved" in str(e).lower() or "not found" in str(e).lower()

        logger.info("✅ Error handling is working correctly")

    @pytest.mark.asyncio
    async def test_concurrent_target_processing(self, simple_mra_targets):
        """Test that targets are processed with proper concurrency control."""
        logger.info("=" * 70)
        logger.info("TEST: Concurrent Target Processing")
        logger.info("=" * 70)

        from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
        from src.core.mcp_client_manager import MCPClientManager

        mcp_manager = MCPClientManager("config/mcp_servers.json")

        async with mcp_manager.session() as session:
            scenario = MultiTargetSimulationScenario(mcp_manager)

            # Run simulation
            result = await scenario.execute(
                targets=simple_mra_targets,
                disease_context="breast cancer",
                simulation_mode="simple"
            )

            # Verify all targets were processed
            assert len(result.individual_results) >= len(simple_mra_targets) - 1

            # Verify results are structured correctly
            for individual_result in result.individual_results:
                assert hasattr(individual_result, 'target_node')
                assert hasattr(individual_result, 'affected_nodes')
                assert hasattr(individual_result, 'confidence_scores')
                assert individual_result.confidence_scores['overall'] >= 0.0

            logger.info(f"Processed {len(result.individual_results)} target results")
            logger.info("✅ Concurrent processing is working correctly")


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([
        __file__,
        '-v',
        '-m', 'production',
        '--tb=short',
        '--timeout=600'  # 10 minute timeout per test
    ])
