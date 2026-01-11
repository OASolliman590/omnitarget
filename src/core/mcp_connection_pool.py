"""
MCP Connection Pool for Resource Management

Provides connection pooling and reuse for MCP (Model Context Protocol) clients
to improve performance and reduce resource overhead.

Based on MRA simulation improvements for handling multiple concurrent requests
while preventing resource exhaustion.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import subprocess
import time

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """Represents a pooled MCP server connection."""
    server_name: str
    process: subprocess.Popen
    created_at: datetime
    last_used: datetime
    request_count: int = 0
    error_count: int = 0
    is_healthy: bool = True

    def mark_used(self):
        """Update last used timestamp and increment request count."""
        self.last_used = datetime.now()
        self.request_count += 1

    def mark_error(self):
        """Increment error count and mark as unhealthy if too many errors."""
        self.error_count += 1
        if self.error_count >= 5:
            self.is_healthy = False
            logger.warning(
                f"Connection {self.server_name} marked unhealthy after {self.error_count} errors"
            )

    def reset_errors(self):
        """Reset error count (called on successful request)."""
        self.error_count = 0
        if self.error_count < 5:
            self.is_healthy = True

    def age_seconds(self) -> float:
        """Get connection age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return (datetime.now() - self.last_used).total_seconds()


class MCPConnectionPool:
    """
    Connection pool for MCP servers with connection reuse and lifecycle management.

    Features:
    - Connection reuse to reduce overhead
    - Health monitoring and automatic cleanup
    - Max connections per server
    - Idle connection timeout
    - Request tracking and statistics
    """

    def __init__(
        self,
        server_configs: Dict[str, Dict[str, Any]],
        max_connections_per_server: int = 3,
        idle_timeout_seconds: int = 300,  # 5 minutes
        max_lifetime_seconds: int = 3600,  # 1 hour
        health_check_interval: int = 60  # 1 minute
    ):
        """
        Initialize MCP connection pool.

        Args:
            server_configs: Dictionary of server_name -> config (path, etc.)
            max_connections_per_server: Maximum connections to maintain per server
            idle_timeout_seconds: Close connections idle longer than this
            max_lifetime_seconds: Close connections older than this
            health_check_interval: Run health checks every N seconds
        """
        self.server_configs = server_configs
        self.max_connections_per_server = max_connections_per_server
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_lifetime_seconds = max_lifetime_seconds
        self.health_check_interval = health_check_interval

        # Connection storage: server_name -> list of PooledConnection
        self._connections: Dict[str, List[PooledConnection]] = {}

        # Tracking
        self._total_connections_created = 0
        self._total_connections_closed = 0
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info(
            f"MCPConnectionPool initialized: {len(server_configs)} servers, "
            f"max_connections={max_connections_per_server}, "
            f"idle_timeout={idle_timeout_seconds}s, "
            f"max_lifetime={max_lifetime_seconds}s"
        )

    async def start(self):
        """Start the connection pool and health check task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._shutdown = False
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("MCPConnectionPool started with health monitoring")

    async def stop(self):
        """Stop the connection pool and close all connections."""
        self._shutdown = True

        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        await self._close_all_connections()

        logger.info(
            f"MCPConnectionPool stopped. "
            f"Total created: {self._total_connections_created}, "
            f"Total closed: {self._total_connections_closed}"
        )

    async def get_connection(self, server_name: str) -> Optional[PooledConnection]:
        """
        Get a healthy connection for the specified server.

        Args:
            server_name: Name of the MCP server

        Returns:
            PooledConnection or None if unavailable
        """
        if server_name not in self.server_configs:
            logger.error(f"Unknown server: {server_name}")
            return None

        # Clean up unhealthy connections first
        await self._cleanup_connections(server_name)

        connections = self._connections.get(server_name, [])

        # Find a healthy, available connection
        healthy_connections = [
            conn for conn in connections
            if conn.is_healthy and conn.idle_seconds() < self.idle_timeout_seconds
        ]

        if healthy_connections:
            # Return the least recently used connection
            connection = min(healthy_connections, key=lambda c: c.last_used)
            connection.mark_used()
            logger.debug(
                f"Reusing connection for {server_name}: "
                f"age={connection.age_seconds():.1f}s, "
                f"requests={connection.request_count}, "
                f"idle={connection.idle_seconds():.1f}s"
            )
            return connection

        # No healthy connection available, create new one if under limit
        if len(connections) < self.max_connections_per_server:
            connection = await self._create_connection(server_name)
            if connection:
                if server_name not in self._connections:
                    self._connections[server_name] = []
                self._connections[server_name].append(connection)
                logger.info(
                    f"Created new connection for {server_name}: "
                    f"total connections={len(self._connections[server_name])}"
                )
                return connection

        # At capacity, return None (caller should wait or use fallback)
        logger.debug(
            f"No connection available for {server_name}: "
            f"active={len(connections)}, "
            f"healthy={len(healthy_connections)}, "
            f"max={self.max_connections_per_server}"
        )
        return None

    async def _create_connection(self, server_name: str) -> Optional[PooledConnection]:
        """Create a new MCP server connection."""
        config = self.server_configs[server_name]
        server_path = config.get('path')

        if not server_path:
            logger.error(f"No path configured for server: {server_name}")
            return None

        try:
            logger.debug(f"Starting MCP server process: {server_name} at {server_path}")

            # Start the server process
            process = await asyncio.create_subprocess_exec(
                server_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Create pooled connection
            connection = PooledConnection(
                server_name=server_name,
                process=process,
                created_at=datetime.now(),
                last_used=datetime.now()
            )

            self._total_connections_created += 1

            # Give the server a moment to start
            await asyncio.sleep(0.1)

            # Test the connection
            if await self._test_connection(connection):
                logger.info(f"✅ Connection created and verified: {server_name}")
                return connection
            else:
                logger.error(f"❌ Connection test failed: {server_name}")
                await self._terminate_connection(connection)
                return None

        except Exception as e:
            logger.error(f"❌ Failed to create connection for {server_name}: {e}")
            return None

    async def _test_connection(self, connection: PooledConnection) -> bool:
        """Test if a connection is healthy by sending a ping request."""
        try:
            # Send a simple JSON-RPC ping request
            ping_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
                "params": {}
            }

            # Try to send and receive
            process = connection.process

            # Write request
            if process.stdin.is_closing():
                return False

            request_str = json.dumps(ping_request) + "\n"
            process.stdin.write(request_str.encode())
            await process.stdin.drain()

            # Try to read response with short timeout
            try:
                await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=2.0
                )
                return True
            except asyncio.TimeoutError:
                logger.warning(
                    f"Connection test timeout: {connection.server_name} "
                    f"(may still be starting up)"
                )
                # Don't mark as unhealthy, just return False for test
                return False

        except Exception as e:
            logger.debug(f"Connection test failed for {connection.server_name}: {e}")
            connection.mark_error()
            return False

    async def _cleanup_connections(self, server_name: str):
        """Clean up unhealthy, idle, or expired connections for a server."""
        if server_name not in self._connections:
            return

        connections = self._connections[server_name]
        to_remove = []

        for connection in connections:
            should_remove = False

            # Check if unhealthy
            if not connection.is_healthy:
                logger.info(f"Removing unhealthy connection: {server_name}")
                should_remove = True

            # Check if idle too long
            elif connection.idle_seconds() > self.idle_timeout_seconds:
                logger.debug(
                    f"Removing idle connection: {server_name} "
                    f"(idle={connection.idle_seconds():.1f}s)"
                )
                should_remove = True

            # Check if too old
            elif connection.age_seconds() > self.max_lifetime_seconds:
                logger.debug(
                    f"Removing old connection: {server_name} "
                    f"(age={connection.age_seconds():.1f}s)"
                )
                should_remove = True

            # Check if process has died
            elif connection.process.stdin.is_closing() or connection.process.stderr.is_closing():
                logger.warning(
                    f"Removing connection with dead process: {server_name}"
                )
                should_remove = True

            if should_remove:
                to_remove.append(connection)

        # Remove connections and terminate processes
        for connection in to_remove:
            await self._terminate_connection(connection)
            if connection in self._connections[server_name]:
                self._connections[server_name].remove(connection)

        if to_remove:
            logger.info(
                f"Cleaned up {len(to_remove)} connections for {server_name}, "
                f"remaining: {len(self._connections.get(server_name, []))}"
            )

    async def _terminate_connection(self, connection: PooledConnection):
        """Terminate a connection and its process."""
        try:
            if not connection.process.stdin.is_closing():
                connection.process.stdin.close()
                await connection.process.stdin.wait_closed()

            if not connection.process.stdout.is_closing():
                connection.process.stdout.close()
                await connection.process.stdout.wait_closed()

            if not connection.process.stderr.is_closing():
                connection.process.stderr.close()
                await connection.process.stderr.wait_closed()

            # Terminate the process
            if connection.process.returncode is None:
                connection.process.terminate()
                try:
                    await asyncio.wait_for(
                        connection.process.wait(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Process {connection.server_name} did not terminate, "
                        f"forcing kill"
                    )
                    connection.process.kill()
                    await connection.process.wait()

        except Exception as e:
            logger.error(
                f"Error terminating connection {connection.server_name}: {e}"
            )
        finally:
            self._total_connections_closed += 1

    async def _close_all_connections(self):
        """Close all connections in the pool."""
        total_closed = 0
        for server_name, connections in self._connections.items():
            for connection in connections:
                await self._terminate_connection(connection)
                total_closed += 1

        self._connections.clear()
        logger.info(f"Closed {total_closed} connections during shutdown")

    async def _health_check_loop(self):
        """Periodic health check and cleanup task."""
        while not self._shutdown:
            try:
                start_time = time.time()

                # Check all servers
                for server_name in list(self.server_configs.keys()):
                    await self._cleanup_connections(server_name)

                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(1.0, self.health_check_interval - elapsed)

                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        stats = {
            'total_connections_created': self._total_connections_created,
            'total_connections_closed': self._total_connections_closed,
            'active_connections': sum(
                len(conns) for conns in self._connections.values()
            ),
            'servers': {}
        }

        for server_name, connections in self._connections.items():
            server_stats = {
                'total': len(connections),
                'healthy': sum(1 for c in connections if c.is_healthy),
                'unhealthy': sum(1 for c in connections if not c.is_healthy),
                'total_requests': sum(c.request_count for c in connections),
                'total_errors': sum(c.error_count for c in connections),
                'avg_age_seconds': (
                    sum(c.age_seconds() for c in connections) / len(connections)
                    if connections else 0
                )
            }
            stats['servers'][server_name] = server_stats

        return stats

    def get_connection_count(self, server_name: str) -> int:
        """Get number of active connections for a server."""
        return len(self._connections.get(server_name, []))

    def is_server_available(self, server_name: str) -> bool:
        """Check if a server has any healthy connections."""
        connections = self._connections.get(server_name, [])
        return any(conn.is_healthy for conn in connections)
