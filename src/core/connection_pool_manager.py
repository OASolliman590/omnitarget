"""
Connection Pool Manager for MCP Clients

Manages connection pools for MCP servers and integrates with the existing
MCPClientManager to provide connection reuse and resource management.

This module enhances the MRA simulation by providing:
- Connection reuse across multiple requests
- Automatic connection health monitoring
- Resource limit enforcement
- Performance metrics and statistics
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from .mcp_connection_pool import MCPConnectionPool, PooledConnection

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """
    Manages connection pools for all MCP clients.

    This manager:
    - Creates and manages connection pools for each MCP server
    - Integrates with existing MCPClientManager
    - Provides connection reuse and health monitoring
    - Tracks performance metrics
    - Supports graceful shutdown
    """

    def __init__(
        self,
        server_configs: Dict[str, Dict[str, Any]],
        enable_pooling: bool = True,
        max_connections_per_server: int = 3,
        idle_timeout_seconds: int = 300,
        max_lifetime_seconds: int = 3600,
        health_check_interval: int = 60
    ):
        """
        Initialize connection pool manager.

        Args:
            server_configs: MCP server configurations
            enable_pooling: Whether to enable connection pooling
            max_connections_per_server: Max connections per server
            idle_timeout_seconds: Close idle connections after this many seconds
            max_lifetime_seconds: Close connections older than this
            health_check_interval: Health check frequency in seconds
        """
        self.server_configs = server_configs
        self.enable_pooling = enable_pooling and max_connections_per_server > 0

        # Initialize connection pool
        if self.enable_pooling:
            self.connection_pool = MCPConnectionPool(
                server_configs=server_configs,
                max_connections_per_server=max_connections_per_server,
                idle_timeout_seconds=idle_timeout_seconds,
                max_lifetime_seconds=max_lifetime_seconds,
                health_check_interval=health_check_interval
            )
            logger.info(
                f"Connection pooling enabled: {max_connections_per_server} max connections per server, "
                f"idle_timeout={idle_timeout_seconds}s"
            )
        else:
            self.connection_pool = None
            logger.info("Connection pooling disabled")

        # Track whether pool is started
        self._started = False

    async def start(self):
        """Start the connection pool manager."""
        if self.enable_pooling and not self._started:
            await self.connection_pool.start()
            self._started = True
            logger.info("ConnectionPoolManager started")

    async def stop(self):
        """Stop the connection pool manager and cleanup connections."""
        if self.enable_pooling and self._started:
            await self.connection_pool.stop()
            self._started = False
            logger.info("ConnectionPoolManager stopped")

    async def get_connection(self, server_name: str) -> Optional[PooledConnection]:
        """
        Get a connection from the pool for the specified server.

        Args:
            server_name: Name of the MCP server (e.g., 'string', 'reactome', 'hpa')

        Returns:
            PooledConnection if available, None otherwise
        """
        if not self.enable_pooling:
            # Pooling disabled, return None (caller should create direct connection)
            return None

        if not self._started:
            logger.warning("ConnectionPoolManager not started, starting now")
            await self.start()

        return await self.connection_pool.get_connection(server_name)

    def is_server_available(self, server_name: str) -> bool:
        """
        Check if a server has any healthy connections.

        Args:
            server_name: Name of the MCP server

        Returns:
            True if server has healthy connections, False otherwise
        """
        if not self.enable_pooling:
            return True  # No pool, assume available

        return self.connection_pool.is_server_available(server_name)

    def get_connection_count(self, server_name: str) -> int:
        """
        Get number of active connections for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Number of active connections
        """
        if not self.enable_pooling:
            return 0

        return self.connection_pool.get_connection_count(server_name)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the connection pool.

        Returns:
            Dictionary with pool statistics
        """
        if not self.enable_pooling:
            return {
                'enabled': False,
                'message': 'Connection pooling is disabled'
            }

        return {
            'enabled': True,
            'pool_stats': self.connection_pool.get_stats()
        }

    def log_stats(self):
        """Log connection pool statistics."""
        if not self.enable_pooling:
            logger.info("Connection pooling is disabled")
            return

        stats = self.connection_pool.get_stats()

        logger.info("=" * 60)
        logger.info("CONNECTION POOL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total connections created: {stats['total_connections_created']}")
        logger.info(f"Total connections closed: {stats['total_connections_closed']}")
        logger.info(f"Active connections: {stats['active_connections']}")
        logger.info("-" * 60)

        for server_name, server_stats in stats['servers'].items():
            logger.info(f"Server: {server_name}")
            logger.info(f"  Total connections: {server_stats['total']}")
            logger.info(f"  Healthy: {server_stats['healthy']}")
            logger.info(f"  Unhealthy: {server_stats['unhealthy']}")
            logger.info(f"  Total requests: {server_stats['total_requests']}")
            logger.info(f"  Total errors: {server_stats['total_errors']}")
            logger.info(f"  Average age: {server_stats['avg_age_seconds']:.1f}s")
            logger.info("")

        logger.info("=" * 60)

    def get_pool_status_summary(self) -> str:
        """
        Get a human-readable summary of pool status.

        Returns:
            Formatted string summary
        """
        if not self.enable_pooling:
            return "Connection pooling: DISABLED"

        stats = self.connection_pool.get_stats()
        active = stats['active_connections']
        created = stats['total_connections_created']

        return (
            f"Connection pooling: ENABLED | "
            f"Active: {active} | "
            f"Total created: {created}"
        )


class PooledMCPClientWrapper:
    """
    Wrapper for MCP client that uses connection pooling.

    This wrapper:
    - Intercepts MCP client calls
    - Uses pooled connections when available
    - Falls back to direct connections when pool unavailable
    - Tracks connection usage and health
    """

    def __init__(
        self,
        client_name: str,
        original_client,
        pool_manager: ConnectionPoolManager
    ):
        """
        Initialize pooled MCP client wrapper.

        Args:
            client_name: Name of the client (e.g., 'string', 'reactome')
            original_client: Original MCP client instance
            pool_manager: ConnectionPoolManager instance
        """
        self.client_name = client_name
        self.original_client = original_client
        self.pool_manager = pool_manager
        self.request_count = 0
        self.error_count = 0

    async def __call__(self, *args, **kwargs):
        """Intercept client calls and use pooled connection if available."""
        self.request_count += 1

        # Try to get a pooled connection
        connection = await self.pool_manager.get_connection(self.client_name)

        if connection:
            # Use pooled connection
            try:
                # Mark connection as used
                connection.mark_used()

                # Execute request
                result = await self._execute_with_client(*args, **kwargs)

                # Reset error count on success
                connection.reset_errors()

                logger.debug(
                    f"✅ Pooled request via {self.client_name}: "
                    f"connection_age={connection.age_seconds():.1f}s, "
                    f"total_requests={connection.request_count}"
                )

                return result

            except Exception as e:
                # Mark error on connection
                connection.mark_error()
                self.error_count += 1
                logger.error(
                    f"❌ Pooled request failed via {self.client_name}: {e}"
                )
                # Fall through to try direct connection
                logger.debug(f"Falling back to direct connection for {self.client_name}")
            else:
                # Success case already handled above
                pass

        # Fall back to direct client call
        try:
            result = await self._execute_with_client(*args, **kwargs)
            logger.debug(f"✅ Direct request via {self.client_name} (no pool available)")
            return result
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Direct request failed via {self.client_name}: {e}")
            raise

    async def _execute_with_client(self, *args, **kwargs):
        """Execute the actual client call."""
        # Call the original client method
        # The original client should be a callable that handles MCP communication
        if asyncio.iscoroutinefunction(self.original_client):
            return await self.original_client(*args, **kwargs)
        else:
            return self.original_client(*args, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this client wrapper.

        Returns:
            Dictionary with client metrics
        """
        return {
            'client_name': self.client_name,
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count),
            'pool_enabled': self.pool_manager.enable_pooling,
            'active_connections': self.pool_manager.get_connection_count(self.client_name)
        }
