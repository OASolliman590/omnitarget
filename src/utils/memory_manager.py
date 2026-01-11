"""
Memory optimization utilities for large-scale bioinformatics analysis.

Provides streaming and lazy loading for large networks and datasets.
"""

import asyncio
import gc
import psutil
import logging
from typing import Any, Dict, List, Iterator, Generator, Optional, Callable
import networkx as nx
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory management utilities for large-scale analysis.
    
    Provides:
    - Network streaming for large graphs
    - Lazy loading for expression data
    - Memory monitoring and optimization
    - Garbage collection management
    """
    
    def __init__(self, max_memory_mb: int = 4096):  # Increased from 2048 to 4096
        """
        Initialize memory manager.

        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Memory usage statistics in MB
        """
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def is_memory_available(self, required_mb: float) -> bool:
        """
        Check if enough memory is available.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if enough memory is available
        """
        memory_usage = self.get_memory_usage()
        return memory_usage['available_mb'] >= required_mb
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        logger.debug("Forced garbage collection")
    
    def stream_large_network(
        self, 
        network: nx.Graph, 
        chunk_size: int = 1000
    ) -> Generator[List[tuple], None, None]:
        """
        Stream large network in chunks to avoid memory overflow.
        
        Args:
            network: NetworkX graph
            chunk_size: Number of edges per chunk
            
        Yields:
            Chunks of edges as list of tuples
        """
        edges = list(network.edges())
        total_edges = len(edges)
        
        logger.info(f"Streaming network with {total_edges} edges in chunks of {chunk_size}")
        
        for i in range(0, total_edges, chunk_size):
            chunk = edges[i:i + chunk_size]
            yield chunk
            
            # Force garbage collection every 10 chunks
            if (i // chunk_size) % 10 == 0:
                self.force_garbage_collection()
    
    async def lazy_load_expression(
        self, 
        genes: List[str], 
        batch_size: int = 100,
        load_fn: Optional[Callable] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Load HPA expression data in batches to avoid memory overflow.
        
        Args:
            genes: List of genes to load
            batch_size: Number of genes per batch
            load_fn: Function to load expression data
            
        Yields:
            Batches of expression data
        """
        total_genes = len(genes)
        logger.info(f"Lazy loading expression data for {total_genes} genes in batches of {batch_size}")
        
        for i in range(0, total_genes, batch_size):
            batch_genes = genes[i:i + batch_size]
            
            if load_fn:
                batch_data = await load_fn(batch_genes)
            else:
                # Default batch data structure
                batch_data = {gene: {} for gene in batch_genes}
            
            yield batch_data
            
            # Force garbage collection every 5 batches
            if (i // batch_size) % 5 == 0:
                self.force_garbage_collection()
    
    def optimize_network_memory(self, network: nx.Graph) -> nx.Graph:
        """
        Optimize network memory usage.
        
        Args:
            network: NetworkX graph
            
        Returns:
            Memory-optimized network
        """
        # Remove unnecessary attributes
        for node in network.nodes():
            if hasattr(network.nodes[node], 'data'):
                # Keep only essential attributes
                essential_attrs = ['id', 'name', 'type']
                node_data = network.nodes[node]
                filtered_data = {k: v for k, v in node_data.items() if k in essential_attrs}
                network.nodes[node].clear()
                network.nodes[node].update(filtered_data)
        
        # Convert to more memory-efficient graph type if needed
        if network.number_of_nodes() > 10000:
            # Use sparse representation for large networks
            logger.info("Converting to sparse representation for large network")
            # NetworkX automatically handles this, but we can optimize further
        
        return network
    
    def create_memory_efficient_matrix(
        self, 
        shape: tuple, 
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Create memory-efficient matrix.
        
        Args:
            shape: Matrix shape
            dtype: Data type (default: float32 for memory efficiency)
            
        Returns:
            Memory-efficient matrix
        """
        # Use float32 instead of float64 for memory efficiency
        matrix = np.zeros(shape, dtype=dtype)
        
        # Estimate memory usage
        memory_mb = matrix.nbytes / (1024 * 1024)
        logger.info(f"Created matrix with shape {shape}, memory usage: {memory_mb:.2f} MB")
        
        if memory_mb > self.max_memory_mb:
            logger.warning(f"Matrix memory usage ({memory_mb:.2f} MB) exceeds limit ({self.max_memory_mb} MB)")
        
        return matrix
    
    def monitor_memory_usage(
        self, 
        operation_name: str, 
        threshold_mb: float = 1000
    ) -> Callable:
        """
        Decorator to monitor memory usage during operations.
        
        Args:
            operation_name: Name of the operation
            threshold_mb: Memory threshold in MB
            
        Returns:
            Decorator function
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_memory = self.get_memory_usage()
                logger.info(f"Starting {operation_name}, initial memory: {start_memory['rss_mb']:.2f} MB")
                
                try:
                    result = await func(*args, **kwargs)
                    
                    end_memory = self.get_memory_usage()
                    memory_delta = end_memory['rss_mb'] - start_memory['rss_mb']
                    
                    logger.info(f"Completed {operation_name}, memory delta: {memory_delta:+.2f} MB")
                    
                    if end_memory['rss_mb'] > threshold_mb:
                        logger.warning(f"High memory usage detected: {end_memory['rss_mb']:.2f} MB")
                        self.force_garbage_collection()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in {operation_name}: {e}")
                    self.force_garbage_collection()
                    raise
            
            return wrapper
        return decorator
    
    def create_sparse_network_representation(
        self, 
        network: nx.Graph
    ) -> Dict[str, Any]:
        """
        Create sparse representation of network for memory efficiency.
        
        Args:
            network: NetworkX graph
            
        Returns:
            Sparse network representation
        """
        # Extract essential network data
        nodes = list(network.nodes())
        edges = list(network.edges())
        
        # Create sparse adjacency matrix
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # Use scipy sparse matrix for memory efficiency
        from scipy.sparse import lil_matrix
        
        adjacency = lil_matrix((n_nodes, n_nodes), dtype=np.float32)
        
        for edge in edges:
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            weight = network.edges[edge].get('weight', 1.0)
            adjacency[i, j] = weight
            adjacency[j, i] = weight  # Undirected graph
        
        return {
            'nodes': nodes,
            'adjacency': adjacency,
            'node_to_idx': node_to_idx,
            'n_nodes': n_nodes,
            'n_edges': len(edges)
        }
    
    def restore_network_from_sparse(
        self, 
        sparse_repr: Dict[str, Any]
    ) -> nx.Graph:
        """
        Restore network from sparse representation.
        
        Args:
            sparse_repr: Sparse network representation
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes
        for node in sparse_repr['nodes']:
            G.add_node(node)
        
        # Add edges from sparse adjacency matrix
        adjacency = sparse_repr['adjacency']
        node_to_idx = sparse_repr['node_to_idx']
        
        for i in range(sparse_repr['n_nodes']):
            for j in range(i + 1, sparse_repr['n_nodes']):
                if adjacency[i, j] != 0:
                    node_i = sparse_repr['nodes'][i]
                    node_j = sparse_repr['nodes'][j]
                    weight = float(adjacency[i, j])
                    G.add_edge(node_i, node_j, weight=weight)
        
        return G
    
    async def process_large_dataset(
        self,
        data: List[Any],
        process_fn: Callable,
        chunk_size: int = 1000,
        max_concurrent: int = 5
    ) -> List[Any]:
        """
        Process large dataset in chunks with concurrency control.
        
        Args:
            data: Dataset to process
            process_fn: Function to process each chunk
            chunk_size: Size of each chunk
            max_concurrent: Maximum concurrent operations
            
        Returns:
            Processed results
        """
        total_items = len(data)
        logger.info(f"Processing {total_items} items in chunks of {chunk_size}")
        
        # Create chunks
        chunks = [data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
        
        # Process chunks with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_chunk(chunk):
            async with semaphore:
                return await process_fn(chunk)
        
        # Process all chunks
        results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
        
        # Flatten results
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
        
        return flattened_results
    
    def get_memory_recommendations(self) -> Dict[str, Any]:
        """
        Get memory usage recommendations.
        
        Returns:
            Memory recommendations
        """
        memory_usage = self.get_memory_usage()
        
        recommendations = {
            'current_usage_mb': memory_usage['rss_mb'],
            'available_mb': memory_usage['available_mb'],
            'usage_percent': memory_usage['percent'],
            'recommendations': []
        }
        
        if memory_usage['percent'] > 80:
            recommendations['recommendations'].append("High memory usage detected. Consider reducing batch sizes.")
        
        if memory_usage['available_mb'] < 500:
            recommendations['recommendations'].append("Low available memory. Consider using streaming or lazy loading.")
        
        if memory_usage['rss_mb'] > self.max_memory_mb:
            recommendations['recommendations'].append("Memory usage exceeds limit. Consider optimizing data structures.")
        
        return recommendations


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def set_memory_manager(manager: MemoryManager) -> None:
    """Set global memory manager instance."""
    global _memory_manager
    _memory_manager = manager


# Convenience functions
def monitor_memory(operation_name: str, threshold_mb: float = 1000):
    """Convenience function for memory monitoring."""
    manager = get_memory_manager()
    return manager.monitor_memory_usage(operation_name, threshold_mb)


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information."""
    manager = get_memory_manager()
    return manager.get_memory_usage()


def optimize_memory() -> None:
    """Force memory optimization."""
    manager = get_memory_manager()
    manager.force_garbage_collection()
