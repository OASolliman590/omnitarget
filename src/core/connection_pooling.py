"""
OmniTarget Pipeline Connection Pooling

High-performance connection pooling for MCP servers with load balancing,
health monitoring, and automatic failover capabilities.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import weakref
from collections import deque
import threading
import json

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class ConnectionInfo:
    """Information about a connection."""
    id: str
    state: ConnectionState
    created_at: float
    last_used: float
    use_count: int
    error_count: int
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    # Pool settings
    min_connections: int = 2
    max_connections: int = 10
    initial_connections: int = 3
    
    # Connection lifecycle
    connection_timeout: int = 30  # seconds
    idle_timeout: int = 300  # 5 minutes
    max_lifetime: int = 3600  # 1 hour
    
    # Health monitoring
    health_check_interval: int = 60  # seconds
    health_check_timeout: int = 10  # seconds
    max_consecutive_errors: int = 3
    
    # Load balancing
    load_balancing_strategy: str = "round_robin"  # round_robin, least_connections, weighted
    connection_weights: Dict[str, float] = None
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Performance
    queue_timeout: int = 30  # seconds
    enable_connection_reuse: bool = True
    enable_connection_warming: bool = True


class MCPConnection:
    """Represents a connection to an MCP server."""
    
    def __init__(self, connection_id: str, server_config: Dict[str, Any]):
        self.connection_id = connection_id
        self.server_config = server_config
        self.process = None
        self.state = ConnectionState.IDLE
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.error_count = 0
        self.last_error = None
        self.metadata = {}
        self._lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """Establish connection to MCP server."""
        try:
            async with self._lock:
                # Start MCP server process
                self.process = await asyncio.create_subprocess_exec(
                    self.server_config['command'],
                    *self.server_config['args'],
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                self.state = ConnectionState.IDLE
                logger.debug(f"Connected to MCP server: {self.connection_id}")
                return True
                
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to connect to MCP server {self.connection_id}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        async with self._lock:
            if self.process:
                try:
                    self.process.terminate()
                    await self.process.wait()
                except Exception as e:
                    logger.warning(f"Error disconnecting {self.connection_id}: {e}")
                finally:
                    self.process = None
                    self.state = ConnectionState.CLOSED
    
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a request on the MCP server."""
        if not self.process or self.state != ConnectionState.IDLE:
            raise RuntimeError(f"Connection {self.connection_id} is not available")
        
        try:
            async with self._lock:
                self.state = ConnectionState.BUSY
                self.last_used = time.time()
                self.use_count += 1
                
                # Send request
                request_json = json.dumps(request) + '\n'
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()
                
                # Read response
                response_line = await self.process.stdout.readline()
                response = json.loads(response_line.decode().strip())
                
                self.state = ConnectionState.IDLE
                return response
                
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Request failed on {self.connection_id}: {e}")
            raise
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return (self.state == ConnectionState.IDLE and 
               self.error_count < 3 and
               time.time() - self.created_at < 3600)  # 1 hour max lifetime
    
    def get_info(self) -> ConnectionInfo:
        """Get connection information."""
        return ConnectionInfo(
            id=self.connection_id,
            state=self.state,
            created_at=self.created_at,
            last_used=self.last_used,
            use_count=self.use_count,
            error_count=self.error_count,
            last_error=self.last_error,
            metadata=self.metadata.copy()
        )


class ConnectionPool:
    """Connection pool for MCP servers."""
    
    def __init__(self, server_name: str, server_config: Dict[str, Any], config: PoolConfig):
        self.server_name = server_name
        self.server_config = server_config
        self.config = config
        self.connections: Dict[str, MCPConnection] = {}
        self.available_connections: deque = deque()
        self.busy_connections: Dict[str, MCPConnection] = {}
        self.connection_counter = 0
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        async with self._lock:
            # Create initial connections
            for _ in range(self.config.initial_connections):
                await self._create_connection()
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Connection pool initialized for {self.server_name}: "
                       f"{len(self.connections)} connections")
    
    async def shutdown(self) -> None:
        """Shutdown the connection pool."""
        self._shutdown = True
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for connection in self.connections.values():
                await connection.disconnect()
            
            self.connections.clear()
            self.available_connections.clear()
            self.busy_connections.clear()
        
        logger.info(f"Connection pool shutdown for {self.server_name}")
    
    async def get_connection(self, timeout: Optional[int] = None) -> MCPConnection:
        """Get an available connection from the pool."""
        timeout = timeout or self.config.queue_timeout
        start_time = time.time()
        
        while True:
            async with self._lock:
                # Try to get an available connection
                if self.available_connections:
                    connection_id = self.available_connections.popleft()
                    connection = self.connections[connection_id]
                    
                    if connection.is_healthy():
                        self.busy_connections[connection_id] = connection
                        return connection
                    else:
                        # Remove unhealthy connection
                        await self._remove_connection(connection_id)
                
                # Create new connection if under limit
                if len(self.connections) < self.config.max_connections:
                    connection = await self._create_connection()
                    if connection:
                        self.busy_connections[connection.connection_id] = connection
                        return connection
            
            # Wait for connection to become available
            if time.time() - start_time > timeout:
                raise TimeoutError(f"No connection available for {self.server_name} after {timeout}s")
            
            await asyncio.sleep(0.1)
    
    async def return_connection(self, connection: MCPConnection) -> None:
        """Return a connection to the pool."""
        async with self._lock:
            connection_id = connection.connection_id
            
            if connection_id in self.busy_connections:
                del self.busy_connections[connection_id]
            
            if connection.is_healthy():
                self.available_connections.append(connection_id)
            else:
                await self._remove_connection(connection_id)
    
    async def _create_connection(self) -> Optional[MCPConnection]:
        """Create a new connection."""
        self.connection_counter += 1
        connection_id = f"{self.server_name}_{self.connection_counter}"
        
        connection = MCPConnection(connection_id, self.server_config)
        
        if await connection.connect():
            self.connections[connection_id] = connection
            self.available_connections.append(connection_id)
            return connection
        else:
            return None
    
    async def _remove_connection(self, connection_id: str) -> None:
        """Remove a connection from the pool."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            await connection.disconnect()
            
            del self.connections[connection_id]
            
            # Remove from available connections
            if connection_id in self.available_connections:
                self.available_connections.remove(connection_id)
    
    async def _health_check_loop(self) -> None:
        """Health check loop for connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {self.server_name}: {e}")
    
    async def _perform_health_check(self) -> None:
        """Perform health check on all connections."""
        async with self._lock:
            connections_to_remove = []
            
            for connection_id, connection in self.connections.items():
                if not connection.is_healthy():
                    connections_to_remove.append(connection_id)
                elif connection.state == ConnectionState.ERROR:
                    # Try to reconnect
                    if await connection.connect():
                        logger.info(f"Reconnected {connection_id}")
                    else:
                        connections_to_remove.append(connection_id)
            
            # Remove unhealthy connections
            for connection_id in connections_to_remove:
                await self._remove_connection(connection_id)
            
            # Ensure minimum connections
            while len(self.connections) < self.config.min_connections:
                await self._create_connection()
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        async with self._lock:
            return {
                'server_name': self.server_name,
                'total_connections': len(self.connections),
                'available_connections': len(self.available_connections),
                'busy_connections': len(self.busy_connections),
                'min_connections': self.config.min_connections,
                'max_connections': self.config.max_connections,
                'connection_details': [
                    conn.get_info() for conn in self.connections.values()
                ]
            }


class ConnectionPoolManager:
    """Manage multiple connection pools for different MCP servers."""
    
    def __init__(self, config: PoolConfig):
        self.config = config
        self.pools: Dict[str, ConnectionPool] = {}
        self._shutdown = False
    
    async def initialize(self, server_configs: Dict[str, Dict[str, Any]]) -> None:
        """Initialize connection pools for all servers."""
        for server_name, server_config in server_configs.items():
            pool = ConnectionPool(server_name, server_config, self.config)
            await pool.initialize()
            self.pools[server_name] = pool
        
        logger.info(f"Connection pool manager initialized with {len(self.pools)} pools")
    
    async def shutdown(self) -> None:
        """Shutdown all connection pools."""
        self._shutdown = True
        
        for pool in self.pools.values():
            await pool.shutdown()
        
        self.pools.clear()
        logger.info("Connection pool manager shutdown complete")
    
    async def get_connection(self, server_name: str, timeout: Optional[int] = None) -> MCPConnection:
        """Get a connection for a specific server."""
        if server_name not in self.pools:
            raise ValueError(f"No connection pool for server: {server_name}")
        
        return await self.pools[server_name].get_connection(timeout)
    
    async def return_connection(self, connection: MCPConnection) -> None:
        """Return a connection to its pool."""
        server_name = connection.server_config.get('name', 'unknown')
        if server_name in self.pools:
            await self.pools[server_name].return_connection(connection)
    
    async def execute_request(self, server_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a request using connection pooling."""
        connection = None
        try:
            connection = await self.get_connection(server_name)
            return await connection.execute_request(request)
        finally:
            if connection:
                await self.return_connection(connection)
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all connection pools."""
        stats = {}
        for server_name, pool in self.pools.items():
            stats[server_name] = await pool.get_pool_stats()
        return stats


# Global connection pool manager
_global_pool_manager: Optional[ConnectionPoolManager] = None


async def get_global_pool_manager() -> ConnectionPoolManager:
    """Get or create global connection pool manager."""
    global _global_pool_manager
    if _global_pool_manager is None:
        config = PoolConfig()
        _global_pool_manager = ConnectionPoolManager(config)
    return _global_pool_manager


async def shutdown_global_pool_manager() -> None:
    """Shutdown global connection pool manager."""
    global _global_pool_manager
    if _global_pool_manager:
        await _global_pool_manager.shutdown()
        _global_pool_manager = None
