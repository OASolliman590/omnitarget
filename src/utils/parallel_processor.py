"""
Parallel processing utilities for MCP calls and data processing.

Provides optimized parallel execution for MCP queries and data processing.
"""

import asyncio
import logging
from typing import Any, Dict, List, Callable, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_concurrent: int = 10
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    use_threading: bool = True
    use_multiprocessing: bool = False


class ParallelProcessor:
    """
    Parallel processing utilities for MCP calls and data processing.
    
    Provides:
    - Concurrent MCP calls
    - Batch processing with concurrency control
    - Retry logic with exponential backoff
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel processor.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, self.config.max_concurrent))
    
    async def parallel_mcp_calls(
        self,
        calls: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Execute multiple MCP calls in parallel.
        
        Args:
            calls: List of MCP call specifications
            timeout: Timeout for each call
            
        Returns:
            List of results
        """
        timeout = timeout or self.config.timeout
        
        async def execute_call(call_spec):
            async with self.semaphore:
                try:
                    # Extract call parameters
                    server = call_spec['server']
                    tool = call_spec['tool']
                    params = call_spec.get('params', {})
                    mcp_client = call_spec['client']
                    
                    # Execute MCP call with timeout
                    result = await asyncio.wait_for(
                        mcp_client.call_tool(tool, params),
                        timeout=timeout
                    )
                    
                    return {
                        'success': True,
                        'result': result,
                        'server': server,
                        'tool': tool
                    }
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for {call_spec['server']}.{call_spec['tool']}")
                    return {
                        'success': False,
                        'error': 'timeout',
                        'server': call_spec['server'],
                        'tool': call_spec['tool']
                    }
                except Exception as e:
                    logger.warning(f"Error in {call_spec['server']}.{call_spec['tool']}: {e}")
                    return {
                        'success': False,
                        'error': str(e),
                        'server': call_spec['server'],
                        'tool': call_spec['tool']
                    }
        
        # Execute all calls in parallel
        results = await asyncio.gather(*[execute_call(call) for call in calls])
        
        # Log results
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        logger.info(f"Parallel MCP calls: {successful}/{total} successful")
        
        return results
    
    async def parallel_data_processing(
        self,
        data: List[Any],
        process_fn: Callable,
        batch_size: int = 100,
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """
        Process data in parallel batches.
        
        Args:
            data: Data to process
            process_fn: Function to process each batch
            batch_size: Size of each batch
            max_concurrent: Maximum concurrent batches
            
        Returns:
            Processed results
        """
        max_concurrent = max_concurrent or self.config.max_concurrent
        
        # Create batches
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                return await process_fn(batch)
        
        # Process all batches
        results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten results
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
        
        return flattened_results
    
    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_attempts: Optional[int] = None,
        base_delay: Optional[float] = None
    ) -> Any:
        """
        Execute function with exponential backoff retry.
        
        Args:
            func: Function to execute
            *args: Function arguments
            max_attempts: Maximum retry attempts
            base_delay: Base delay between retries
            
        Returns:
            Function result
        """
        max_attempts = max_attempts or self.config.retry_attempts
        base_delay = base_delay or self.config.retry_delay
        
        for attempt in range(max_attempts):
            try:
                return await func(*args)
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} attempts failed: {e}")
                    raise
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
    
    async def parallel_with_retry(
        self,
        tasks: List[Callable],
        max_attempts: Optional[int] = None
    ) -> List[Any]:
        """
        Execute tasks in parallel with retry logic.
        
        Args:
            tasks: List of async functions to execute
            max_attempts: Maximum retry attempts per task
            
        Returns:
            List of results
        """
        max_attempts = max_attempts or self.config.retry_attempts
        
        async def execute_with_retry(task):
            return await self.retry_with_backoff(task, max_attempts=max_attempts)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*[execute_with_retry(task) for task in tasks])
        return results
    
    def optimize_mcp_parallel_calls(
        self,
        mcp_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize MCP calls for parallel execution.
        
        Args:
            mcp_calls: List of MCP call specifications
            
        Returns:
            Optimized call specifications
        """
        # Group calls by server to optimize connection reuse
        server_groups = {}
        for call in mcp_calls:
            server = call['server']
            if server not in server_groups:
                server_groups[server] = []
            server_groups[server].append(call)
        
        # Reorder calls to minimize server switching
        optimized_calls = []
        for server, calls in server_groups.items():
            optimized_calls.extend(calls)
        
        logger.info(f"Optimized {len(mcp_calls)} MCP calls across {len(server_groups)} servers")
        return optimized_calls
    
    async def batch_mcp_calls(
        self,
        calls: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[Any]:
        """
        Execute MCP calls in batches to avoid overwhelming servers.
        
        Args:
            calls: List of MCP call specifications
            batch_size: Size of each batch
            
        Returns:
            List of results
        """
        total_calls = len(calls)
        results = []
        
        for i in range(0, total_calls, batch_size):
            batch = calls[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_calls + batch_size - 1)//batch_size}")
            
            batch_results = await self.parallel_mcp_calls(batch)
            results.extend(batch_results)
            
            # Small delay between batches to avoid overwhelming servers
            if i + batch_size < total_calls:
                await asyncio.sleep(0.1)
        
        return results
    
    async def monitor_performance(
        self,
        operation_name: str,
        func: Callable,
        *args
    ) -> Dict[str, Any]:
        """
        Monitor performance of an operation.
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = await func(*args)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            performance = {
                'operation': operation_name,
                'duration_seconds': end_time - start_time,
                'memory_delta_mb': end_memory - start_memory,
                'success': True
            }
            
            logger.info(f"Performance: {operation_name} took {performance['duration_seconds']:.2f}s")
            return performance
            
        except Exception as e:
            end_time = time.time()
            performance = {
                'operation': operation_name,
                'duration_seconds': end_time - start_time,
                'error': str(e),
                'success': False
            }
            
            logger.error(f"Performance: {operation_name} failed after {performance['duration_seconds']:.2f}s: {e}")
            return performance
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    async def close(self):
        """Close thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("Parallel processor closed")


# Global parallel processor instance
_parallel_processor: Optional[ParallelProcessor] = None


def get_parallel_processor() -> ParallelProcessor:
    """Get global parallel processor instance."""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelProcessor()
    return _parallel_processor


def set_parallel_processor(processor: ParallelProcessor) -> None:
    """Set global parallel processor instance."""
    global _parallel_processor
    _parallel_processor = processor


# Convenience functions
async def parallel_mcp_calls(calls: List[Dict[str, Any]]) -> List[Any]:
    """Convenience function for parallel MCP calls."""
    processor = get_parallel_processor()
    return await processor.parallel_mcp_calls(calls)


async def parallel_data_processing(
    data: List[Any], 
    process_fn: Callable, 
    batch_size: int = 100
) -> List[Any]:
    """Convenience function for parallel data processing."""
    processor = get_parallel_processor()
    return await processor.parallel_data_processing(data, process_fn, batch_size)


async def retry_with_backoff(func: Callable, *args, max_attempts: int = 3) -> Any:
    """Convenience function for retry with backoff."""
    processor = get_parallel_processor()
    return await processor.retry_with_backoff(func, *args, max_attempts=max_attempts)
