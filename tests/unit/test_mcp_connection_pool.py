"""
Unit tests for MCP Connection Pool.

Tests the connection pooling and resource management features
for the MRA simulation improvement.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path

from src.core.mcp_connection_pool import MCPConnectionPool, PooledConnection
from src.core.connection_pool_manager import ConnectionPoolManager, PooledMCPClientWrapper


@pytest.mark.unit
class TestPooledConnection:
    """Test PooledConnection class."""

    def test_pooled_connection_creation(self):
        """Test PooledConnection is created correctly."""
        process = Mock()
        process.stdin.is_closing.return_value = False
        process.stdout.is_closing.return_value = False
        process.stderr.is_closing.return_value = False

        conn = PooledConnection(
            server_name="test_server",
            process=process,
            created_at=datetime.now(),
            last_used=datetime.now()
        )

        assert conn.server_name == "test_server"
        assert conn.process == process
        assert conn.request_count == 0
        assert conn.error_count == 0
        assert conn.is_healthy is True
        assert conn.created_at is not None
        assert conn.last_used is not None

    def test_mark_used(self):
        """Test mark_used updates request count and timestamp."""
        process = Mock()
        process.stdin.is_closing.return_value = False
        process.stdout.is_closing.return_value = False
        process.stderr.is_closing.return_value = False

        conn = PooledConnection(
            server_name="test_server",
            process=process,
            created_at=datetime.now(),
            last_used=datetime.now()
        )

        initial_request_count = conn.request_count
        initial_last_used = conn.last_used

        time.sleep(0.01)  # Small delay to ensure time difference
        conn.mark_used()

        assert conn.request_count == initial_request_count + 1
        assert conn.last_used > initial_last_used

    def test_mark_error(self):
        """Test mark_error increments error count."""
        process = Mock()
        process.stdin.is_closing.return_value = False
        process.stdout.is_closing.return_value = False
        process.stderr.is_closing.return_value = False

        conn = PooledConnection(
            server_name="test_server",
            process=process,
            created_at=datetime.now(),
            last_used=datetime.now()
        )

        assert conn.error_count == 0
        assert conn.is_healthy is True

        conn.mark_error()
        assert conn.error_count == 1
        assert conn.is_healthy is True  # Still healthy at 1 error

        # Mark unhealthy after 5 errors
        for _ in range(4):
            conn.mark_error()
        assert conn.error_count == 5
        assert conn.is_healthy is False

    def test_reset_errors(self):
        """Test reset_errors clears error count."""
        process = Mock()
        process.stdin.is_closing.return_value = False
        process.stdout.is_closing.return_value = False
        process.stderr.is_closing.return_value = False

        conn = PooledConnection(
            server_name="test_server",
            process=process,
            created_at=datetime.now(),
            last_used=datetime.now()
        )

        # Simulate errors
        conn.mark_error()
        conn.mark_error()
        assert conn.error_count == 2

        # Reset errors
        conn.reset_errors()
        assert conn.error_count == 0

    def test_age_seconds(self):
        """Test age_seconds returns correct age."""
        process = Mock()
        process.stdin.is_closing.return_value = False
        process.stdout.is_closing.return_value = False
        process.stderr.is_closing.return_value = False

        now = datetime.now()
        conn = PooledConnection(
            server_name="test_server",
            process=process,
            created_at=now,
            last_used=now
        )

        time.sleep(0.1)
        age = conn.age_seconds()
        assert age >= 0.1
        assert age < 0.2

    def test_idle_seconds(self):
        """Test idle_seconds returns correct idle time."""
        process = Mock()
        process.stdin.is_closing.return_value = False
        process.stdout.is_closing.return_value = False
        process.stderr.is_closing.return_value = False

        now = datetime.now()
        conn = PooledConnection(
            server_name="test_server",
            process=process,
            created_at=now,
            last_used=now
        )

        time.sleep(0.1)
        idle = conn.idle_seconds()
        assert idle >= 0.1
        assert idle < 0.2


@pytest.mark.unit
class TestMCPConnectionPool:
    """Test MCPConnectionPool class."""

    @pytest.fixture
    def server_configs(self):
        """Create test server configurations."""
        return {
            'server1': {'path': '/path/to/server1'},
            'server2': {'path': '/path/to/server2'}
        }

    @pytest.fixture
    def connection_pool(self, server_configs):
        """Create a test connection pool."""
        return MCPConnectionPool(
            server_configs=server_configs,
            max_connections_per_server=3,
            idle_timeout_seconds=300,
            max_lifetime_seconds=3600,
            health_check_interval=60
        )

    def test_pool_initialization(self, connection_pool, server_configs):
        """Test pool initializes correctly."""
        assert connection_pool.server_configs == server_configs
        assert connection_pool.max_connections_per_server == 3
        assert connection_pool.idle_timeout_seconds == 300
        assert connection_pool.max_lifetime_seconds == 3600
        assert connection_pool.health_check_interval == 60
        assert connection_pool._connections == {}
        assert connection_pool._total_connections_created == 0
        assert connection_pool._total_connections_closed == 0

    def test_get_stats(self, connection_pool):
        """Test get_stats returns correct statistics."""
        stats = connection_pool.get_stats()

        assert 'total_connections_created' in stats
        assert 'total_connections_closed' in stats
        assert 'active_connections' in stats
        assert 'servers' in stats

        assert stats['total_connections_created'] == 0
        assert stats['total_connections_closed'] == 0
        assert stats['active_connections'] == 0
        assert stats['servers'] == {}

    @pytest.mark.asyncio
    async def test_get_connection_unknown_server(self, connection_pool):
        """Test get_connection with unknown server returns None."""
        result = await connection_pool.get_connection('unknown_server')
        assert result is None

    @pytest.mark.asyncio
    async def test_get_connection_no_pooling(self, connection_pool):
        """Test get_connection when no connections available."""
        # Server has no connections yet
        result = await connection_pool.get_connection('server1')
        assert result is None  # No connections available and can't create without actual process

    @pytest.mark.asyncio
    async def test_start_stop(self, connection_pool):
        """Test starting and stopping the pool."""
        assert connection_pool._health_check_task is None
        assert connection_pool._shutdown is False

        await connection_pool.start()

        assert connection_pool._health_check_task is not None
        assert connection_pool._shutdown is False

        # Stop the pool
        await connection_pool.stop()

        assert connection_pool._shutdown is True
        # Health check task should be cancelled
        if connection_pool._health_check_task:
            assert connection_pool._health_check_task.done()

    @pytest.mark.asyncio
    async def test_get_connection_count(self, connection_pool):
        """Test get_connection_count returns correct count."""
        # Initially no connections
        count = connection_pool.get_connection_count('server1')
        assert count == 0

    @pytest.mark.asyncio
    async def test_is_server_available(self, connection_pool):
        """Test is_server_available works correctly."""
        # Initially no connections, so not available
        assert connection_pool.is_server_available('server1') is False


@pytest.mark.unit
class TestConnectionPoolManager:
    """Test ConnectionPoolManager class."""

    @pytest.fixture
    def server_configs(self):
        """Create test server configurations."""
        return {
            'server1': {'path': '/path/to/server1'},
            'server2': {'path': '/path/to/server2'}
        }

    @pytest.fixture
    def pool_manager(self, server_configs):
        """Create a test pool manager."""
        return ConnectionPoolManager(
            server_configs=server_configs,
            enable_pooling=True,
            max_connections_per_server=3,
            idle_timeout_seconds=300,
            max_lifetime_seconds=3600,
            health_check_interval=60
        )

    def test_pool_manager_initialization(self, pool_manager, server_configs):
        """Test pool manager initializes correctly."""
        assert pool_manager.server_configs == server_configs
        assert pool_manager.enable_pooling is True
        assert pool_manager.connection_pool is not None
        assert pool_manager._started is False

    def test_pool_manager_disabled(self, server_configs):
        """Test pool manager with pooling disabled."""
        manager = ConnectionPoolManager(
            server_configs=server_configs,
            enable_pooling=False
        )

        assert manager.enable_pooling is False
        assert manager.connection_pool is None

    @pytest.mark.asyncio
    async def test_get_connection_not_started(self, pool_manager):
        """Test get_connection when manager not started."""
        result = await pool_manager.get_connection('server1')
        # Should return None when not started
        assert result is None

    @pytest.mark.asyncio
    async def test_start_stop(self, pool_manager):
        """Test starting and stopping the pool manager."""
        assert pool_manager._started is False

        await pool_manager.start()
        assert pool_manager._started is True

        await pool_manager.stop()
        assert pool_manager._started is False

    @pytest.mark.asyncio
    async def test_is_server_available(self, pool_manager):
        """Test is_server_available method."""
        # Initially no connections
        assert pool_manager.is_server_available('server1') is False

    def test_get_connection_count(self, pool_manager):
        """Test get_connection_count method."""
        count = pool_manager.get_connection_count('server1')
        assert count == 0

    def test_get_stats(self, pool_manager):
        """Test get_stats method."""
        stats = pool_manager.get_stats()

        assert 'enabled' in stats
        assert stats['enabled'] is True
        assert 'pool_stats' in stats

    def test_get_pool_status_summary(self, pool_manager):
        """Test get_pool_status_summary method."""
        summary = pool_manager.get_pool_status_summary()
        assert 'ENABLED' in summary

    def test_get_pool_status_summary_disabled(self, server_configs):
        """Test get_pool_status_summary when pooling disabled."""
        manager = ConnectionPoolManager(
            server_configs=server_configs,
            enable_pooling=False
        )

        summary = manager.get_pool_status_summary()
        assert 'DISABLED' in summary

    @pytest.mark.asyncio
    async def test_log_stats(self, pool_manager):
        """Test log_stats method (should not raise)."""
        # Start the pool first
        await pool_manager.start()

        # Should not raise any exceptions
        pool_manager.log_stats()

        await pool_manager.stop()


@pytest.mark.unit
class TestPooledMCPClientWrapper:
    """Test PooledMCPClientWrapper class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock MCP client."""
        return AsyncMock(return_value={'result': 'success'})

    @pytest.fixture
    def pool_manager(self):
        """Create a test pool manager."""
        return Mock(spec=ConnectionPoolManager)

    @pytest.fixture
    def wrapped_client(self, mock_client, pool_manager):
        """Create a wrapped client."""
        return PooledMCPClientWrapper(
            client_name='test_server',
            original_client=mock_client,
            pool_manager=pool_manager
        )

    @pytest.mark.asyncio
    async def test_wrapper_execution_direct(self, wrapped_client, pool_manager):
        """Test wrapper execution without pool connection."""
        # Pool returns no connection
        pool_manager.get_connection = AsyncMock(return_value=None)

        result = await wrapped_client('arg1', 'arg2', kwarg1='value1')

        assert result == {'result': 'success'}
        assert wrapped_client.request_count == 1
        assert wrapped_client.error_count == 0

    @pytest.mark.asyncio
    async def test_wrapper_execution_with_pool_connection(self, wrapped_client, pool_manager):
        """Test wrapper execution with pool connection."""
        # Mock pooled connection
        mock_connection = Mock()
        mock_connection.mark_used = Mock()
        mock_connection.reset_errors = Mock()
        pool_manager.get_connection = AsyncMock(return_value=mock_connection)

        result = await wrapped_client('arg1', 'arg2', kwarg1='value1')

        assert result == {'result': 'success'}
        assert wrapped_client.request_count == 1
        assert wrapped_client.error_count == 0
        mock_connection.mark_used.assert_called_once()
        mock_connection.reset_errors.assert_called_once()

    @pytest.mark.asyncio
    async def test_wrapper_error_handling(self, wrapped_client, pool_manager):
        """Test wrapper error handling."""
        # Pool returns connection but call fails
        mock_connection = Mock()
        mock_connection.mark_used = Mock()
        mock_connection.mark_error = Mock()
        pool_manager.get_connection = AsyncMock(return_value=mock_connection)

        # Make client raise an error
        wrapped_client._execute_with_client = AsyncMock(
            side_effect=Exception("Test error")
        )

        with pytest.raises(Exception, match="Test error"):
            await wrapped_client('arg1', 'arg2', kwarg1='value1')

        assert wrapped_client.request_count == 1
        assert wrapped_client.error_count == 1
        mock_connection.mark_error.assert_called_once()

    def test_get_metrics(self, wrapped_client):
        """Test get_metrics method."""
        metrics = wrapped_client.get_metrics()

        assert 'client_name' in metrics
        assert 'total_requests' in metrics
        assert 'total_errors' in metrics
        assert 'error_rate' in metrics
        assert 'pool_enabled' in metrics
        assert 'active_connections' in metrics

        assert metrics['client_name'] == 'test_server'
        assert metrics['total_requests'] == 0
        assert metrics['total_errors'] == 0
        assert metrics['error_rate'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'unit'])
