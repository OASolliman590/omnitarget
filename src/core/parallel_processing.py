"""
OmniTarget Pipeline Parallel Processing

High-performance parallel execution for MCP calls, scenario processing, and simulation.
Supports concurrent operations with intelligent load balancing and resource management.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for parallel processing."""
    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    # Execution settings - OPTIMIZED for faster execution
    max_concurrent_tasks: int = 20  # Increased from 10 for better parallelism
    max_workers: int = 8  # Increased from 4 for CPU-intensive tasks
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    
    # Timeout settings
    task_timeout: int = 300  # 5 minutes
    batch_timeout: int = 600  # 10 minutes
    
    # Resource management
    memory_limit_mb: int = 2048  # 2GB
    cpu_limit_percent: float = 80.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Load balancing
    enable_load_balancing: bool = True
    task_priority_weight: float = 1.0
    resource_usage_weight: float = 0.5


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a parallel task."""
    id: str
    func: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[int] = None
    retries: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.kwargs is None:
            self.kwargs = {}


class ParallelExecutor:
    """High-performance parallel executor for OmniTarget pipeline."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        self._shutdown = False
        
    async def initialize(self) -> None:
        """Initialize the parallel executor."""
        if self.config.execution_mode == ExecutionMode.THREAD:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        elif self.config.execution_mode == ExecutionMode.PROCESS:
            self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info(f"Parallel executor initialized: {self.config.execution_mode.value}")
    
    async def shutdown(self) -> None:
        """Shutdown the parallel executor."""
        self._shutdown = True
        
        # Cancel active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task: {task_id}")
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Parallel executor shutdown complete")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task for parallel execution."""
        if self._shutdown:
            raise RuntimeError("Executor is shutting down")
        
        # Add to queue
        self.task_queue.append(task)
        
        # Start execution if under limit
        if len(self.active_tasks) < self.config.max_concurrent_tasks:
            await self._start_task(task)
        
        return task.id
    
    async def _start_task(self, task: Task) -> None:
        """Start executing a task."""
        try:
            if self.config.execution_mode == ExecutionMode.ASYNC:
                # Async execution
                coro = self._execute_async_task(task)
                asyncio_task = asyncio.create_task(coro)
            else:
                # Thread/Process execution
                coro = self._execute_sync_task(task)
                asyncio_task = asyncio.create_task(coro)
            
            self.active_tasks[task.id] = asyncio_task
            
            # Set up completion handler
            asyncio_task.add_done_callback(
                lambda t: asyncio.create_task(self._handle_task_completion(task.id, t))
            )
            
        except Exception as e:
            logger.error(f"Failed to start task {task.id}: {e}")
            self.failed_tasks[task.id] = e
    
    async def _execute_async_task(self, task: Task) -> Any:
        """Execute async task."""
        timeout = task.timeout or self.config.task_timeout
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=timeout
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, task.func, *task.args, **task.kwargs
                )
            
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task.id} timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            raise
    
    async def _execute_sync_task(self, task: Task) -> Any:
        """Execute sync task in thread/process pool."""
        if not self.executor:
            raise RuntimeError("Executor not initialized")
        
        timeout = task.timeout or self.config.task_timeout
        
        try:
            # Submit to executor
            future = self.executor.submit(task.func, *task.args, **task.kwargs)
            
            # Wait for completion with timeout
            result = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=timeout
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            raise
    
    async def _handle_task_completion(self, task_id: str, task: asyncio.Task) -> None:
        """Handle task completion."""
        try:
            if task.cancelled():
                logger.info(f"Task {task_id} was cancelled")
                return
            
            if task.exception():
                # Handle failure with retry logic
                await self._handle_task_failure(task_id, task.exception())
            else:
                # Task completed successfully
                result = task.result()
                self.completed_tasks[task_id] = result
                logger.debug(f"Task {task_id} completed successfully")
        
        finally:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Start next task in queue
            await self._start_next_task()
    
    async def _handle_task_failure(self, task_id: str, exception: Exception) -> None:
        """Handle task failure with retry logic."""
        # Find the original task
        original_task = None
        for task in self.task_queue:
            if task.id == task_id:
                original_task = task
                break
        
        if not original_task:
            self.failed_tasks[task_id] = exception
            return
        
        # Check retry logic
        if original_task.retries < self.config.max_retries:
            original_task.retries += 1
            
            # Calculate retry delay
            delay = self.config.retry_delay
            if self.config.exponential_backoff:
                delay *= (2 ** (original_task.retries - 1))
            
            logger.info(f"Retrying task {task_id} in {delay}s (attempt {original_task.retries})")
            
            # Wait and retry
            await asyncio.sleep(delay)
            await self._start_task(original_task)
        else:
            logger.error(f"Task {task_id} failed after {self.config.max_retries} retries: {exception}")
            self.failed_tasks[task_id] = exception
    
    async def _start_next_task(self) -> None:
        """Start the next task in queue if under limit."""
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            return
        
        # Find next task to execute
        next_task = None
        for task in self.task_queue:
            if task.id not in self.active_tasks and task.id not in self.completed_tasks:
                next_task = task
                break
        
        if next_task:
            await self._start_task(next_task)
    
    async def wait_for_completion(self, task_ids: Optional[List[str]] = None, 
                                timeout: Optional[int] = None) -> Dict[str, Any]:
        """Wait for task completion and return results."""
        timeout = timeout or self.config.batch_timeout
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch execution timed out after {timeout}s")
            
            # Check if all requested tasks are complete
            if task_ids:
                completed = all(
                    task_id in self.completed_tasks or task_id in self.failed_tasks
                    for task_id in task_ids
                )
            else:
                completed = len(self.active_tasks) == 0 and len(self.task_queue) == 0
            
            if completed:
                break
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        # Return results
        results = {}
        for task_id in (task_ids or list(self.completed_tasks.keys())):
            if task_id in self.completed_tasks:
                results[task_id] = self.completed_tasks[task_id]
            elif task_id in self.failed_tasks:
                results[task_id] = {'error': str(self.failed_tasks[task_id])}
        
        return results
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        if task_id in self.completed_tasks:
            return {'status': 'completed', 'result': self.completed_tasks[task_id]}
        elif task_id in self.failed_tasks:
            return {'status': 'failed', 'error': str(self.failed_tasks[task_id])}
        elif task_id in self.active_tasks:
            return {'status': 'running'}
        else:
            return {'status': 'pending'}
    
    async def get_executor_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'executor_type': self.config.execution_mode.value,
            'max_concurrent': self.config.max_concurrent_tasks
        }


class MCPParallelProcessor:
    """Specialized parallel processor for MCP operations."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.executor = ParallelExecutor(config)
        self.mcp_operations: Dict[str, Callable] = {}
    
    async def initialize(self) -> None:
        """Initialize the MCP parallel processor."""
        await self.executor.initialize()
        logger.info("MCP parallel processor initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the MCP parallel processor."""
        await self.executor.shutdown()
        logger.info("MCP parallel processor shutdown")
    
    def register_mcp_operation(self, name: str, operation: Callable) -> None:
        """Register an MCP operation for parallel execution."""
        self.mcp_operations[name] = operation
        logger.debug(f"Registered MCP operation: {name}")
    
    async def execute_mcp_calls(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple MCP calls in parallel."""
        tasks = []
        
        for i, call in enumerate(calls):
            operation_name = call.get('operation')
            if operation_name not in self.mcp_operations:
                raise ValueError(f"Unknown MCP operation: {operation_name}")
            
            task = Task(
                id=f"mcp_call_{i}",
                func=self.mcp_operations[operation_name],
                args=call.get('args', ()),
                kwargs=call.get('kwargs', {}),
                priority=TaskPriority.HIGH if call.get('priority') == 'high' else TaskPriority.NORMAL,
                timeout=call.get('timeout', self.config.task_timeout)
            )
            tasks.append(task)
        
        # Submit all tasks
        task_ids = []
        for task in tasks:
            task_id = await self.executor.submit_task(task)
            task_ids.append(task_id)
        
        # Wait for completion
        results = await self.executor.wait_for_completion(task_ids)
        return results
    
    async def execute_scenario_batch(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple scenarios in parallel."""
        tasks = []
        
        for i, scenario in enumerate(scenarios):
            scenario_func = scenario.get('func')
            if not scenario_func:
                raise ValueError("Scenario function not provided")
            
            task = Task(
                id=f"scenario_{i}",
                func=scenario_func,
                args=scenario.get('args', ()),
                kwargs=scenario.get('kwargs', {}),
                priority=TaskPriority.NORMAL,
                timeout=scenario.get('timeout', self.config.task_timeout)
            )
            tasks.append(task)
        
        # Submit all tasks
        task_ids = []
        for task in tasks:
            task_id = await self.executor.submit_task(task)
            task_ids.append(task_id)
        
        # Wait for completion
        results = await self.executor.wait_for_completion(task_ids)
        return results


# Utility functions for common parallel operations
async def parallel_mcp_calls(mcp_calls: List[Dict[str, Any]], 
                           config: Optional[ParallelConfig] = None) -> Dict[str, Any]:
    """Execute multiple MCP calls in parallel."""
    if config is None:
        config = ParallelConfig()
    
    processor = MCPParallelProcessor(config)
    await processor.initialize()
    
    try:
        results = await processor.execute_mcp_calls(mcp_calls)
        return results
    finally:
        await processor.shutdown()


async def parallel_scenario_execution(scenarios: List[Dict[str, Any]], 
                                    config: Optional[ParallelConfig] = None) -> Dict[str, Any]:
    """Execute multiple scenarios in parallel."""
    if config is None:
        config = ParallelConfig()
    
    processor = MCPParallelProcessor(config)
    await processor.initialize()
    
    try:
        results = await processor.execute_scenario_batch(scenarios)
        return results
    finally:
        await processor.shutdown()


# Global parallel processor instance
_global_processor: Optional[MCPParallelProcessor] = None


async def get_global_processor() -> MCPParallelProcessor:
    """Get or create global parallel processor."""
    global _global_processor
    if _global_processor is None:
        config = ParallelConfig()
        _global_processor = MCPParallelProcessor(config)
        await _global_processor.initialize()
    return _global_processor


async def shutdown_global_processor() -> None:
    """Shutdown global parallel processor."""
    global _global_processor
    if _global_processor:
        await _global_processor.shutdown()
        _global_processor = None
