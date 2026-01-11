"""
Unit Tests for Optimization Components

Test caching, parallel processing, memory optimization, and connection pooling.
"""

import pytest
import asyncio
import time
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch, AsyncMock

from src.core.caching import (
    CacheConfig, CacheKey, MemoryCache, RedisCache, OmniTargetCache,
    cache_mcp_response, get_cached_mcp_response
)
from src.core.parallel_processing import (
    ParallelConfig, Task, TaskPriority, ParallelExecutor, MCPParallelProcessor,
    parallel_mcp_calls, parallel_scenario_execution
)
from src.core.memory_optimization import (
    MemoryConfig, MemoryMonitor, SparseMatrixManager, ChunkedProcessor,
    LazyLoader, MemoryOptimizer, get_global_optimizer, optimize_network_memory
)
from src.core.connection_pooling import (
    PoolConfig, ConnectionState, ConnectionInfo, MCPConnection, ConnectionPool,
    ConnectionPoolManager, get_global_pool_manager
)

pytestmark = pytest.mark.unit


class TestCachingSystem:
    """Test caching system functionality."""
    
    def test_cache_config_creation(self):
        """Test cache configuration creation."""
        config = CacheConfig()
        assert config.use_memory_cache == True
        assert config.max_memory_size == 1000
        assert config.memory_ttl == 3600
        assert config.cache_mcp_responses == True
    
    def test_cache_key_generation(self):
        """Test cache key generation for different data types."""
        # MCP response key
        mcp_key = CacheKey.mcp_response("kegg", "search_diseases", {"query": "cancer"})
        assert mcp_key.startswith("omnitarget:")
        assert len(mcp_key) > 20
        
        # Simulation result key
        sim_key = CacheKey.simulation_result("TP53", "inhibit", "network_hash_123")
        assert sim_key.startswith("omnitarget:")
        
        # Network data key
        network_key = CacheKey.network_data("network_123", "1.0")
        assert network_key.startswith("omnitarget:")
    
    @pytest.mark.asyncio
    async def test_memory_cache_basic_operations(self):
        """Test basic memory cache operations."""
        cache = MemoryCache(max_size=10, default_ttl=60)
        
        # Test set and get
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Test expiration
        await cache.set("expired_key", "expired_value", ttl=0.1)
        await asyncio.sleep(0.2)
        value = await cache.get("expired_key")
        assert value is None
        
        # Test deletion
        await cache.set("delete_key", "delete_value")
        deleted = await cache.delete("delete_key")
        assert deleted == True
        value = await cache.get("delete_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_memory_cache_lru_eviction(self):
        """Test LRU eviction in memory cache."""
        cache = MemoryCache(max_size=3, default_ttl=60)
        
        # Fill cache to capacity
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        await cache.get("key1")
        
        # Add new key - should evict key2 (least recently used)
        await cache.set("key4", "value4")
        
        # Check that key2 is evicted but key1 remains
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"
    
    @pytest.mark.asyncio
    async def test_memory_cache_stats(self):
        """Test memory cache statistics."""
        cache = MemoryCache(max_size=5, default_ttl=60)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        stats = await cache.stats()
        assert stats['size'] == 2
        assert stats['max_size'] == 5
        assert stats['expired_entries'] == 0
    
    @pytest.mark.asyncio
    async def test_omnitarget_cache_integration(self):
        """Test OmniTarget cache integration."""
        config = CacheConfig(use_redis_cache=False)  # Disable Redis for testing
        cache = OmniTargetCache(config)
        await cache.initialize()
        
        # Test basic operations
        await cache.set("test_key", {"data": "test_value"})
        value = await cache.get("test_key")
        assert value == {"data": "test_value"}
        
        # Test get_or_set
        def factory_func():
            return {"computed": "value"}
        
        value = await cache.get_or_set("computed_key", factory_func)
        assert value == {"computed": "value"}
        
        # Test cache hit
        value = await cache.get("computed_key")
        assert value == {"computed": "value"}
    
    @pytest.mark.asyncio
    async def test_mcp_response_caching(self):
        """Test MCP response caching utilities."""
        # Mock the global cache
        with patch('src.core.caching.get_global_cache') as mock_get_cache:
            mock_cache = AsyncMock()
            mock_get_cache.return_value = mock_cache
            
            # Test caching MCP response
            response = {"diseases": [{"id": "hsa05224", "name": "Breast cancer"}]}
            await cache_mcp_response("kegg", "search_diseases", {"query": "cancer"}, response)
            
            mock_cache.set.assert_called_once()
            
            # Test getting cached response
            await get_cached_mcp_response("kegg", "search_diseases", {"query": "cancer"})
            mock_cache.get.assert_called_once()


class TestParallelProcessing:
    """Test parallel processing functionality."""
    
    def test_parallel_config_creation(self):
        """Test parallel configuration creation."""
        config = ParallelConfig()
        assert config.max_concurrent_tasks == 10
        assert config.max_workers == 4
        assert config.task_timeout == 300
        assert config.max_retries == 3
    
    def test_task_creation(self):
        """Test task creation and properties."""
        def test_func(x, y):
            return x + y
        
        task = Task(
            id="test_task",
            func=test_func,
            args=(1, 2),
            kwargs={"multiply": 2},
            priority=TaskPriority.HIGH,
            timeout=60
        )
        
        assert task.id == "test_task"
        assert task.func == test_func
        assert task.args == (1, 2)
        assert task.kwargs == {"multiply": 2}
        assert task.priority == TaskPriority.HIGH
        assert task.timeout == 60
        assert task.created_at is not None
    
    @pytest.mark.asyncio
    async def test_parallel_executor_basic_operations(self):
        """Test basic parallel executor operations."""
        config = ParallelConfig(max_concurrent_tasks=2)
        executor = ParallelExecutor(config)
        await executor.initialize()
        
        # Test task submission
        def simple_task(x):
            return x * 2
        
        task = Task(id="test1", func=simple_task, args=(5,))
        task_id = await executor.submit_task(task)
        assert task_id == "test1"
        
        # Wait for completion
        results = await executor.wait_for_completion(["test1"])
        assert "test1" in results
        assert results["test1"] == 10
        
        await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_parallel_executor_multiple_tasks(self):
        """Test parallel executor with multiple tasks."""
        config = ParallelConfig(max_concurrent_tasks=3)
        executor = ParallelExecutor(config)
        await executor.initialize()
        
        # Submit multiple tasks
        tasks = []
        for i in range(5):
            def task_func(x):
                return x * x
            
            task = Task(id=f"task_{i}", func=task_func, args=(i,))
            tasks.append(task)
            await executor.submit_task(task)
        
        # Wait for all tasks to complete
        task_ids = [f"task_{i}" for i in range(5)]
        results = await executor.wait_for_completion(task_ids)
        
        assert len(results) == 5
        for i in range(5):
            assert results[f"task_{i}"] == i * i
        
        await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_parallel_executor_error_handling(self):
        """Test parallel executor error handling."""
        config = ParallelConfig(max_retries=2)
        executor = ParallelExecutor(config)
        await executor.initialize()
        
        # Task that will fail
        def failing_task():
            raise ValueError("Test error")
        
        task = Task(id="failing_task", func=failing_task)
        await executor.submit_task(task)
        
        # Wait for completion
        results = await executor.wait_for_completion(["failing_task"])
        assert "failing_task" in results
        assert "error" in results["failing_task"]
        
        await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_mcp_parallel_processor(self):
        """Test MCP parallel processor."""
        config = ParallelConfig()
        processor = MCPParallelProcessor(config)
        await processor.initialize()
        
        # Register MCP operation
        def mock_mcp_operation(query):
            return {"result": f"processed_{query}"}
        
        processor.register_mcp_operation("search", mock_mcp_operation)
        
        # Execute MCP calls
        calls = [
            {"operation": "search", "args": ("cancer",)},
            {"operation": "search", "args": ("diabetes",)}
        ]
        
        results = await processor.execute_mcp_calls(calls)
        assert len(results) == 2
        assert "mcp_call_0" in results
        assert "mcp_call_1" in results
        
        await processor.shutdown()


class TestMemoryOptimization:
    """Test memory optimization functionality."""
    
    def test_memory_config_creation(self):
        """Test memory configuration creation."""
        config = MemoryConfig()
        assert config.max_memory_mb == 2048
        assert config.warning_threshold == 0.8
        assert config.enable_sparse_matrices == True
        assert config.chunk_size == 1000
    
    def test_memory_monitor_basic(self):
        """Test memory monitor basic functionality."""
        config = MemoryConfig()
        monitor = MemoryMonitor(config)
        
        # Test memory usage retrieval
        usage = monitor.get_memory_usage()
        assert 'rss_mb' in usage
        assert 'vms_mb' in usage
        assert 'percent' in usage
        assert usage['rss_mb'] > 0
        
        # Test memory limits check
        status = monitor.check_memory_limits()
        assert 'usage_mb' in status
        assert 'max_mb' in status
        assert 'status' in status
    
    def test_sparse_matrix_manager(self):
        """Test sparse matrix manager."""
        config = MemoryConfig()
        manager = SparseMatrixManager(config)
        
        # Test sparse matrix creation
        matrix = manager.create_sparse_matrix("test_matrix", (100, 100))
        assert matrix.shape == (100, 100)
        assert "test_matrix" in manager.matrices
        
        # Test matrix retrieval
        retrieved = manager.get_matrix("test_matrix")
        assert retrieved is not None
        assert retrieved.shape == (100, 100)
        
        # Test sparse conversion decision
        dense_matrix = np.zeros((10, 10))
        dense_matrix[0, 0] = 1.0  # Only one non-zero element (density = 1%)
        
        should_sparse = manager.should_use_sparse(dense_matrix)
        assert should_sparse == True
        
        # Test sparse conversion
        sparse_matrix = manager.convert_to_sparse(dense_matrix, "sparse_test")
        assert sparse_matrix.nnz < 100  # Should be sparse
        assert "sparse_test" in manager.matrices
    
    def test_chunked_processor(self):
        """Test chunked processor."""
        config = MemoryConfig(chunk_size=3)
        processor = ChunkedProcessor(config)
        
        # Test chunked iterator
        data = list(range(10))
        chunks = list(processor.create_chunked_iterator(data))
        
        assert len(chunks) == 4  # 10 items / 3 chunk_size = 4 chunks
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]
    
    def test_lazy_loader(self):
        """Test lazy loader."""
        config = MemoryConfig()
        loader = LazyLoader(config)
        
        # Register loader function
        def expensive_operation():
            return {"data": "expensive_computation"}
        
        loader.register_loader("expensive_data", expensive_operation)
        
        # Test lazy loading
        data = loader.get_data("expensive_data")
        assert data == {"data": "expensive_computation"}
        assert "expensive_data" in loader.loaded_data
        
        # Test unloading
        unloaded = loader.unload_data("expensive_data")
        assert unloaded == True
        assert "expensive_data" not in loader.loaded_data
    
    def test_memory_optimizer_integration(self):
        """Test memory optimizer integration."""
        config = MemoryConfig()
        optimizer = MemoryOptimizer(config)
        
        # Test optimization
        results = optimizer.optimize_memory_usage()
        assert 'initial_status' in results
        assert 'final_status' in results
        
        # Test statistics
        stats = optimizer.get_optimization_stats()
        assert 'gc_runs' in stats
        assert 'current_memory' in stats


class TestConnectionPooling:
    """Test connection pooling functionality."""
    
    def test_pool_config_creation(self):
        """Test pool configuration creation."""
        config = PoolConfig()
        assert config.min_connections == 2
        assert config.max_connections == 10
        assert config.connection_timeout == 30
        assert config.max_retries == 3
    
    def test_connection_info_creation(self):
        """Test connection info creation."""
        info = ConnectionInfo(
            id="test_connection",
            state=ConnectionState.IDLE,
            created_at=time.time(),
            last_used=time.time(),
            use_count=5,
            error_count=0
        )
        
        assert info.id == "test_connection"
        assert info.state == ConnectionState.IDLE
        assert info.use_count == 5
        assert info.error_count == 0
    
    @pytest.mark.asyncio
    async def test_connection_pool_basic_operations(self):
        """Test basic connection pool operations."""
        server_config = {
            "command": "echo",
            "args": ["test"]
        }
        config = PoolConfig(min_connections=1, max_connections=2)
        pool = ConnectionPool("test_server", server_config, config)
        
        # Mock the subprocess creation
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.stdin = AsyncMock()
            mock_process.stdout = AsyncMock()
            mock_process.stderr = AsyncMock()
            mock_subprocess.return_value = mock_process
            
            await pool.initialize()
            
            # Test getting connection
            connection = await pool.get_connection()
            assert connection is not None
            assert connection.connection_id.startswith("test_server_")
            
            # Test returning connection
            await pool.return_connection(connection)
            
            await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_connection_pool_manager(self):
        """Test connection pool manager."""
        config = PoolConfig()
        manager = ConnectionPoolManager(config)
        
        server_configs = {
            "kegg": {"command": "echo", "args": ["kegg"]},
            "reactome": {"command": "echo", "args": ["reactome"]}
        }
        
        # Mock subprocess creation
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.stdin = AsyncMock()
            mock_process.stdout = AsyncMock()
            mock_process.stderr = AsyncMock()
            mock_subprocess.return_value = mock_process
            
            await manager.initialize(server_configs)
            
            # Test getting connection
            connection = await manager.get_connection("kegg")
            assert connection is not None
            
            # Test returning connection
            await manager.return_connection(connection)
            
            # Test statistics
            stats = await manager.get_all_stats()
            assert "kegg" in stats
            assert "reactome" in stats
            
            await manager.shutdown()


class TestOptimizationIntegration:
    """Test integration of optimization components."""
    
    @pytest.mark.asyncio
    async def test_caching_with_parallel_processing(self):
        """Test caching integration with parallel processing."""
        # This would test the integration of caching with parallel processing
        # For now, we'll test that both systems can work together
        
        # Test cache configuration
        cache_config = CacheConfig(use_redis_cache=False)
        cache = OmniTargetCache(cache_config)
        await cache.initialize()
        
        # Test parallel processing configuration
        parallel_config = ParallelConfig(max_concurrent_tasks=2)
        executor = ParallelExecutor(parallel_config)
        await executor.initialize()
        
        # Test that both systems are functional
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"
        
        def simple_task():
            return "task_result"
        
        task = Task(id="test_task", func=simple_task)
        await executor.submit_task(task)
        results = await executor.wait_for_completion(["test_task"])
        assert results["test_task"] == "task_result"
        
        await executor.shutdown()
    
    def test_memory_optimization_with_networks(self):
        """Test memory optimization with network data."""
        # Create a test network
        network = nx.Graph()
        network.add_nodes_from(range(100))
        network.add_edges_from([(i, i+1) for i in range(99)])
        
        # Test memory optimization
        optimizer = get_global_optimizer()
        optimized_network = optimize_network_memory(network)
        
        # Network should be returned (optimization is internal)
        assert optimized_network is not None
        assert optimized_network.number_of_nodes() == 100
        assert optimized_network.number_of_edges() == 99
    
    def test_optimization_performance_metrics(self):
        """Test optimization performance metrics."""
        # Test memory optimization stats
        optimizer = get_global_optimizer()
        stats = optimizer.get_optimization_stats()
        
        assert 'gc_runs' in stats
        assert 'current_memory' in stats
        assert 'sparse_matrices' in stats
        
        # Test that stats are reasonable
        assert stats['gc_runs'] >= 0
        assert stats['current_memory']['rss_mb'] > 0
