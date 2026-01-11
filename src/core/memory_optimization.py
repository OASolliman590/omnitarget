"""
OmniTarget Pipeline Memory Optimization

Advanced memory management for large biological networks, sparse matrices, and simulation data.
Implements memory-efficient data structures and garbage collection strategies.
"""

import gc
import psutil
import logging
import weakref
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum
import numpy as np
import networkx as nx
from scipy import sparse
import threading
import time

logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    SPARSE_MATRICES = "sparse_matrices"
    CHUNKED_PROCESSING = "chunked_processing"
    LAZY_LOADING = "lazy_loading"
    COMPRESSION = "compression"
    GARBAGE_COLLECTION = "garbage_collection"


@dataclass
class MemoryConfig:
    """Memory optimization configuration."""
    # Memory limits
    max_memory_mb: int = 2048  # 2GB
    warning_threshold: float = 0.8  # 80% of max memory
    critical_threshold: float = 0.9  # 90% of max memory
    
    # Optimization strategies
    enable_sparse_matrices: bool = True
    enable_chunked_processing: bool = True
    enable_lazy_loading: bool = True
    enable_compression: bool = True
    enable_garbage_collection: bool = True
    
    # Chunked processing
    chunk_size: int = 1000
    max_chunks_in_memory: int = 10
    
    # Sparse matrix settings
    sparse_threshold: float = 0.1  # Use sparse if density < 10%
    sparse_format: str = "csr"  # Compressed Sparse Row
    
    # Compression settings
    compression_level: int = 6
    compression_algorithm: str = "lz4"
    
    # Garbage collection
    gc_frequency: int = 100  # Run GC every N operations
    gc_threshold: float = 0.7  # Run GC when memory usage > 70%


class MemoryMonitor:
    """Monitor and manage memory usage."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.process = psutil.Process()
        self.operation_count = 0
        self.memory_history: List[float] = []
        self.lock = threading.Lock()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': system_memory.available / 1024 / 1024,
            'total_mb': system_memory.total / 1024 / 1024
        }
    
    def check_memory_limits(self) -> Dict[str, Any]:
        """Check if memory usage is within limits."""
        usage = self.get_memory_usage()
        rss_mb = usage['rss_mb']
        max_mb = self.config.max_memory_mb
        
        status = {
            'usage_mb': rss_mb,
            'max_mb': max_mb,
            'usage_percent': rss_mb / max_mb,
            'status': 'normal',
            'warning': False,
            'critical': False
        }
        
        if rss_mb > max_mb * self.config.critical_threshold:
            status['status'] = 'critical'
            status['critical'] = True
        elif rss_mb > max_mb * self.config.warning_threshold:
            status['status'] = 'warning'
            status['warning'] = True
        
        return status
    
    def should_run_gc(self) -> bool:
        """Determine if garbage collection should be run."""
        self.operation_count += 1
        
        # Run GC based on frequency
        if self.operation_count % self.config.gc_frequency == 0:
            return True
        
        # Run GC based on memory usage
        usage = self.get_memory_usage()
        if usage['rss_mb'] > self.config.max_memory_mb * self.config.gc_threshold:
            return True
        
        return False
    
    def run_garbage_collection(self) -> Dict[str, Any]:
        """Run garbage collection and return statistics."""
        before = self.get_memory_usage()
        
        # Run garbage collection
        collected = gc.collect()
        
        after = self.get_memory_usage()
        
        return {
            'objects_collected': collected,
            'memory_freed_mb': before['rss_mb'] - after['rss_mb'],
            'memory_before_mb': before['rss_mb'],
            'memory_after_mb': after['rss_mb']
        }
    
    def log_memory_usage(self, operation: str = "") -> None:
        """Log current memory usage."""
        usage = self.get_memory_usage()
        status = self.check_memory_limits()
        
        logger.info(f"Memory usage {operation}: {usage['rss_mb']:.1f}MB "
                   f"({status['usage_percent']:.1%}) - {status['status']}")


class SparseMatrixManager:
    """Manage sparse matrices for memory efficiency."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.matrices: Dict[str, sparse.spmatrix] = {}
        self.matrix_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_sparse_matrix(self, name: str, shape: Tuple[int, int], 
                           dtype: np.dtype = np.float64) -> sparse.spmatrix:
        """Create a sparse matrix with metadata."""
        if self.config.sparse_format == "csr":
            matrix = sparse.csr_matrix(shape, dtype=dtype)
        elif self.config.sparse_format == "csc":
            matrix = sparse.csc_matrix(shape, dtype=dtype)
        elif self.config.sparse_format == "coo":
            matrix = sparse.coo_matrix(shape, dtype=dtype)
        else:
            raise ValueError(f"Unsupported sparse format: {self.config.sparse_format}")
        
        self.matrices[name] = matrix
        self.matrix_metadata[name] = {
            'shape': shape,
            'dtype': dtype,
            'format': self.config.sparse_format,
            'created_at': time.time(),
            'access_count': 0
        }
        
        return matrix
    
    def get_matrix(self, name: str) -> Optional[sparse.spmatrix]:
        """Get a sparse matrix by name."""
        if name in self.matrices:
            self.matrix_metadata[name]['access_count'] += 1
            return self.matrices[name]
        return None
    
    def should_use_sparse(self, matrix: np.ndarray) -> bool:
        """Determine if a matrix should be converted to sparse format."""
        if not self.config.enable_sparse_matrices:
            return False
        
        # Calculate density
        total_elements = matrix.size
        non_zero_elements = np.count_nonzero(matrix)
        density = non_zero_elements / total_elements
        
        return density < self.config.sparse_threshold
    
    def convert_to_sparse(self, matrix: np.ndarray, name: str) -> sparse.spmatrix:
        """Convert dense matrix to sparse format."""
        if self.config.sparse_format == "csr":
            sparse_matrix = sparse.csr_matrix(matrix)
        elif self.config.sparse_format == "csc":
            sparse_matrix = sparse.csc_matrix(matrix)
        else:
            sparse_matrix = sparse.coo_matrix(matrix)
        
        self.matrices[name] = sparse_matrix
        self.matrix_metadata[name] = {
            'shape': matrix.shape,
            'dtype': matrix.dtype,
            'format': self.config.sparse_format,
            'created_at': time.time(),
            'access_count': 0,
            'original_dense': True
        }
        
        return sparse_matrix
    
    def get_matrix_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a sparse matrix."""
        if name not in self.matrices:
            return {}
        
        matrix = self.matrices[name]
        metadata = self.matrix_metadata[name]
        
        return {
            'name': name,
            'shape': matrix.shape,
            'nnz': matrix.nnz,  # Number of non-zero elements
            'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
            'memory_usage_mb': matrix.data.nbytes / 1024 / 1024,
            'format': metadata['format'],
            'access_count': metadata['access_count']
        }
    
    def cleanup_unused_matrices(self, max_age_hours: int = 24) -> int:
        """Clean up unused matrices older than specified age."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        matrices_to_remove = []
        for name, metadata in self.matrix_metadata.items():
            age = current_time - metadata['created_at']
            if age > max_age_seconds and metadata['access_count'] == 0:
                matrices_to_remove.append(name)
        
        for name in matrices_to_remove:
            if name in self.matrices:
                del self.matrices[name]
            if name in self.matrix_metadata:
                del self.matrix_metadata[name]
            cleaned_count += 1
        
        return cleaned_count


class ChunkedProcessor:
    """Process large datasets in chunks to manage memory usage."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.active_chunks: Dict[str, Any] = {}
        self.chunk_cache: Dict[str, List[Any]] = {}
    
    def create_chunked_iterator(self, data: List[Any], chunk_size: Optional[int] = None) -> Iterator[List[Any]]:
        """Create an iterator that yields data in chunks."""
        chunk_size = chunk_size or self.config.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            yield chunk
    
    async def process_chunked_data(self, data: List[Any], processor_func: callable,
                                 chunk_size: Optional[int] = None) -> List[Any]:
        """Process data in chunks with memory management."""
        chunk_size = chunk_size or self.config.chunk_size
        results = []
        
        for chunk in self.create_chunked_iterator(data, chunk_size):
            # Process chunk
            chunk_result = await processor_func(chunk)
            results.extend(chunk_result)
            
            # Check memory usage
            monitor = MemoryMonitor(self.config)
            if monitor.should_run_gc():
                gc_stats = monitor.run_garbage_collection()
                logger.debug(f"GC during chunked processing: {gc_stats}")
        
        return results
    
    def cache_chunk(self, chunk_id: str, data: Any) -> None:
        """Cache a chunk of data."""
        if len(self.chunk_cache) >= self.config.max_chunks_in_memory:
            # Remove oldest chunk
            oldest_key = min(self.chunk_cache.keys())
            del self.chunk_cache[oldest_key]
        
        self.chunk_cache[chunk_id] = data
    
    def get_cached_chunk(self, chunk_id: str) -> Optional[Any]:
        """Get a cached chunk."""
        return self.chunk_cache.get(chunk_id)
    
    def clear_chunk_cache(self) -> None:
        """Clear all cached chunks."""
        self.chunk_cache.clear()


class LazyLoader:
    """Lazy loading for large datasets to reduce memory usage."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.loaded_data: Dict[str, Any] = {}
        self.loader_functions: Dict[str, callable] = {}
        self.access_times: Dict[str, float] = {}
    
    def register_loader(self, name: str, loader_func: callable) -> None:
        """Register a loader function for lazy loading."""
        self.loader_functions[name] = loader_func
        logger.debug(f"Registered lazy loader: {name}")
    
    def get_data(self, name: str) -> Any:
        """Get data, loading it if necessary."""
        if name not in self.loaded_data:
            if name not in self.loader_functions:
                raise ValueError(f"No loader registered for: {name}")
            
            # Load data
            self.loaded_data[name] = self.loader_functions[name]()
            logger.debug(f"Lazy loaded data: {name}")
        
        # Update access time
        self.access_times[name] = time.time()
        return self.loaded_data[name]
    
    def unload_data(self, name: str) -> bool:
        """Unload data to free memory."""
        if name in self.loaded_data:
            del self.loaded_data[name]
            if name in self.access_times:
                del self.access_times[name]
            logger.debug(f"Unloaded data: {name}")
            return True
        return False
    
    def unload_unused_data(self, max_age_hours: int = 1) -> int:
        """Unload data that hasn't been accessed recently."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        unloaded_count = 0
        
        data_to_unload = []
        for name, access_time in self.access_times.items():
            age = current_time - access_time
            if age > max_age_seconds:
                data_to_unload.append(name)
        
        for name in data_to_unload:
            if self.unload_data(name):
                unloaded_count += 1
        
        return unloaded_count


class MemoryOptimizer:
    """Main memory optimization manager."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.monitor = MemoryMonitor(config)
        self.sparse_manager = SparseMatrixManager(config)
        self.chunked_processor = ChunkedProcessor(config)
        self.lazy_loader = LazyLoader(config)
        self.optimization_stats: Dict[str, Any] = {
            'gc_runs': 0,
            'memory_freed_mb': 0,
            'sparse_conversions': 0,
            'chunks_processed': 0,
            'data_unloaded': 0
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Run comprehensive memory optimization."""
        optimization_results = {}
        
        # Check current memory status
        memory_status = self.monitor.check_memory_limits()
        optimization_results['initial_status'] = memory_status
        
        # Run garbage collection if needed
        if self.monitor.should_run_gc():
            gc_stats = self.monitor.run_garbage_collection()
            optimization_results['garbage_collection'] = gc_stats
            self.optimization_stats['gc_runs'] += 1
            self.optimization_stats['memory_freed_mb'] += gc_stats['memory_freed_mb']
        
        # Clean up unused sparse matrices
        if self.config.enable_sparse_matrices:
            cleaned = self.sparse_manager.cleanup_unused_matrices()
            optimization_results['sparse_cleanup'] = {'matrices_removed': cleaned}
        
        # Unload unused lazy data
        if self.config.enable_lazy_loading:
            unloaded = self.lazy_loader.unload_unused_data()
            optimization_results['lazy_unload'] = {'data_unloaded': unloaded}
            self.optimization_stats['data_unloaded'] += unloaded
        
        # Final memory status
        final_status = self.monitor.check_memory_limits()
        optimization_results['final_status'] = final_status
        
        return optimization_results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            **self.optimization_stats,
            'current_memory': self.monitor.get_memory_usage(),
            'sparse_matrices': len(self.sparse_manager.matrices),
            'cached_chunks': len(self.chunked_processor.chunk_cache),
            'loaded_data': len(self.lazy_loader.loaded_data)
        }
    
    def log_memory_status(self, operation: str = "") -> None:
        """Log current memory status."""
        self.monitor.log_memory_usage(operation)
        
        # Log optimization stats
        stats = self.get_optimization_stats()
        logger.info(f"Memory optimization stats: {stats}")


# Global memory optimizer instance
_global_optimizer: Optional[MemoryOptimizer] = None


def get_global_optimizer() -> MemoryOptimizer:
    """Get or create global memory optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        config = MemoryConfig()
        _global_optimizer = MemoryOptimizer(config)
    return _global_optimizer


def optimize_network_memory(network: nx.Graph) -> nx.Graph:
    """Optimize network memory usage."""
    optimizer = get_global_optimizer()
    
    # Convert to sparse representation if beneficial
    if optimizer.config.enable_sparse_matrices:
        # This would require converting the network to a matrix representation
        # For now, we'll just log the network size
        logger.debug(f"Network optimization: {network.number_of_nodes()} nodes, "
                    f"{network.number_of_edges()} edges")
    
    return network


def optimize_simulation_memory(simulation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize simulation data memory usage."""
    optimizer = get_global_optimizer()
    
    # Run memory optimization
    optimization_results = optimizer.optimize_memory_usage()
    logger.debug(f"Simulation memory optimization: {optimization_results}")
    
    return simulation_data
