"""
Batch Query Utilities for Parallel Execution

Enables parallel database queries across multiple MCP servers for 10-20x performance improvement.
Part of P0-3: Connection Pooling critical optimization.
P0-4: Production Monitoring with structured logging and metrics

Key improvements:
- batch_query(): Execute multiple queries in parallel batches
- parallel_query(): Execute different queries in parallel
- parallel_server_query(): Query multiple servers simultaneously

Author: OmniTarget Team
Date: 2025-01-06
Updated: 2025-11-07 (P0-4 monitoring)
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from collections import defaultdict

from ..core.exceptions import is_transient_error, format_error_for_logging
from ..core.logging_config import log_with_context
from ..core.metrics import record_batch_query

logger = logging.getLogger(__name__)


async def batch_query(
    query_func: Callable,
    items: List[Any],
    batch_size: int = 10,
    max_retries: int = 3,
    return_exceptions: bool = True
) -> List[Any]:
    """
    Execute queries in parallel batches.

    This is the key to 10-20x performance improvement!
    Instead of querying one item at a time, we query multiple items in parallel.

    P0-4: Production Monitoring - Tracks metrics and uses structured logging.

    Args:
        query_func: Async function to call for each item
            Example: lambda gene: await mcp_manager.kegg.get_pathway_info(gene)
        items: List of items to query
            Example: ["hsa05200", "hsa05210", ...]  # pathway IDs
        batch_size: Number of concurrent queries per batch (default: 10)
            Higher = faster but more resource intensive
        max_retries: Retry attempts for failed queries (default: 3)
        return_exceptions: If True, return exceptions as results instead of raising

    Returns:
        List of results (None for failed queries if return_exceptions=True)

    Example - Before (Serial):
        # Time: 100 pathways × 1 second = 100 seconds
        for pathway_id in pathway_ids:
            result = await kegg.get_pathway_info(pathway_id)
            results.append(result)

    Example - After (Parallel):
        # Time: 100 pathways / 10 parallel = 10 seconds (10x faster!)
        results = await batch_query(
            lambda pid: kegg.get_pathway_info(pid),
            pathway_ids,
            batch_size=10
        )
    """
    if not items:
        return []

    # Track start time for metrics (P0-4)
    start_time = time.time()

    results = []
    total_items = len(items)
    operation = query_func.__name__ if hasattr(query_func, '__name__') else 'unknown'

    # Structured log: batch query started (P0-4)
    log_with_context(
        logger,
        "debug",
        "batch_query_started",
        operation=operation,
        total_items=total_items,
        batch_size=batch_size,
        max_retries=max_retries
    )

    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_items + batch_size - 1) // batch_size

        # Structured logging for each batch (P0-4)
        log_with_context(
            logger,
            "debug",
            "processing_batch",
            operation=operation,
            batch_num=batch_num,
            total_batches=total_batches,
            batch_size=len(batch),
            completed_items=len(results),
            total_items=total_items
        )

        # Execute batch in parallel
        tasks = [_query_with_retry(query_func, item, max_retries) for item in batch]

        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        except Exception as e:
            # Structured logging for batch failure (P0-4)
            log_with_context(
                logger,
                "error",
                "batch_processing_error",
                operation=operation,
                batch_num=batch_num,
                error=str(e),
                error_type=type(e).__name__
            )
            # Fill batch with exceptions if return_exceptions=False
            if not return_exceptions:
                batch_results = [e] * len(batch)

        # Handle results
        for item, result in zip(batch, batch_results):
            if isinstance(result, Exception) and return_exceptions:
                # Structured logging for query failure (P0-4)
                log_with_context(
                    logger,
                    "warning",
                    "query_failed",
                    operation=operation,
                    item=str(item),
                    error=str(result),
                    error_type=type(result).__name__
                )
                results.append(None)
            else:
                results.append(result)

    # Calculate duration and success rate
    duration = time.time() - start_time
    success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    success_rate = success_count / total_items if total_items > 0 else 0.0

    # Determine status for metrics
    if success_rate == 1.0:
        status = "success"
    elif success_rate > 0.0:
        status = "partial"
    else:
        status = "error"

    # Record batch query metrics (P0-4)
    record_batch_query(
        operation=operation,
        batch_size=batch_size,
        status=status,
        duration=duration
    )

    # Log summary with structured logging (P0-4)
    log_with_context(
        logger,
        "info",
        "batch_query_completed",
        operation=operation,
        total_items=total_items,
        successful_items=success_count,
        success_rate=success_rate,
        duration=duration,
        status=status
    )

    return results


async def _query_with_retry(query_func: Callable, item: Any, max_retries: int) -> Any:
    """Execute single query with retry."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await query_func(item)
        except Exception as e:
            last_error = e

            # Check if error is retryable
            if not is_transient_error(e):
                # Non-retryable error, don't retry
                raise

            if attempt < max_retries:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.debug(
                    f"Query failed for {item}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(wait_time)

    # All retries exhausted
    raise last_error


async def parallel_query(
    queries: Dict[str, Callable],
    return_exceptions: bool = True,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Execute multiple different queries in parallel.

    Perfect for querying different servers simultaneously!

    Args:
        queries: Dict mapping names to async callables
            Example:
            {
                "kegg_pathways": lambda: kegg.search_pathways("cancer"),
                "reactome_pathways": lambda: reactome.search_pathways("cancer"),
                "string_network": lambda: string.get_network(["AXL", "BRCA1"])
            }
        return_exceptions: If True, return None for failed queries instead of raising
        timeout: Optional timeout for all queries

    Returns:
        Dict mapping names to results

    Example - Before (Sequential):
        # Time: 3 seconds (1 sec per server)
        kegg_result = await kegg.search_pathways("cancer")
        reactome_result = await reactome.search_pathways("cancer")
        string_result = await string.get_network(["AXL", "BRCA1"])

    Example - After (Parallel):
        # Time: 1 second (all 3 servers queried simultaneously!)
        results = await parallel_query({
            "kegg": lambda: kegg.search_pathways("cancer"),
            "reactome": lambda: reactome.search_pathways("cancer"),
            "string": lambda: string.get_network(["AXL", "BRCA1"])
        })
    """
    if not queries:
        return {}

    # Wrap queries in a way that preserves the name
    async def execute_query(name_func_pair):
        name, query_func = name_func_pair
        try:
            result = await query_func()
            return name, result
        except Exception as e:
            if return_exceptions:
                logger.warning(f"Query '{name}' failed: {e}")
                return name, None
            else:
                raise

    # Execute all queries in parallel
    tasks = [execute_query(pair) for pair in queries.items()]

    if timeout:
        results_list = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=return_exceptions),
            timeout=timeout
        )
    else:
        results_list = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    # Convert back to dict
    results = {}
    for name, result in results_list:
        if isinstance(result, Exception) and return_exceptions:
            results[name] = None
        else:
            results[name] = result

    # Log summary
    success_count = sum(1 for r in results.values() if r is not None)
    logger.info(
        f"Parallel query complete: {success_count}/{len(queries)} successful "
        f"({success_count/len(queries):.1%} success rate)"
    )

    return results


async def parallel_server_query(
    mcp_manager,
    server_queries: Dict[str, Callable],
    return_exceptions: bool = True
) -> Dict[str, Any]:
    """
    Execute queries on different MCP servers in parallel.

    This is specifically designed for the MCPClientManager pattern.
    It queries multiple servers (KEGG, Reactome, STRING, etc.) simultaneously.

    Args:
        mcp_manager: MCPClientManager instance
        server_queries: Dict mapping server names to query callables
            Example:
            {
                "kegg": lambda: mcp_manager.kegg.search_diseases("breast cancer"),
                "reactome": lambda: mcp_manager.reactome.find_pathways_by_disease("breast cancer"),
                "hpa": lambda: mcp_manager.hpa.search_cancer_markers("breast cancer")
            }
        return_exceptions: If True, return None for failed servers instead of raising

    Returns:
        Dict mapping server names to results

    Example:
        results = await parallel_server_query(
            mcp_manager,
            {
                "kegg": lambda: mcp_manager.kegg.search_diseases(query),
                "reactome": lambda: mcp_manager.reactome.find_pathways_by_disease(query),
                "string": lambda: mcp_manager.string.get_network(genes)
            }
        )
        kegg_result = results["kegg"]
        reactome_result = results["reactome"]
        string_result = results["string"]
    """
    # Transform to parallel_query format
    named_queries = {
        f"{server}_{i}": query_func
        for i, (server, query_func) in enumerate(server_queries.items())
    }

    results = await parallel_query(named_queries, return_exceptions=return_exceptions)

    # Convert back to server-based dict
    server_results = {}
    for i, (server, _) in enumerate(server_queries.items()):
        key = f"{server}_{i}"
        server_results[server] = results.get(key)

    return server_results


# Convenience functions for common patterns

async def query_multiple_pathways(kegg_client, pathway_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Convenience function to query multiple pathways in parallel.

    Example:
        pathway_ids = ["hsa05200", "hsa05210", "hsa05211", ...]
        results = await query_multiple_pathways(kegg, pathway_ids)
    """
    return await batch_query(
        lambda pid: kegg_client.get_pathway_info(pid),
        pathway_ids,
        batch_size=10
    )


async def query_multiple_genes(kegg_client, gene_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Convenience function to query multiple genes in parallel.

    Example:
        gene_ids = ["hsa:123", "hsa:456", "hsa:789", ...]
        results = await query_multiple_genes(kegg, gene_ids)
    """
    return await batch_query(
        lambda gid: kegg_client.get_gene_info(gid),
        gene_ids,
        batch_size=10
    )


async def multi_server_disease_search(
    mcp_manager,
    disease_query: str,
    servers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to search for a disease across multiple servers in parallel.

    Example:
        results = await multi_server_disease_search(
            mcp_manager,
            "breast cancer",
            servers=["kegg", "reactome", "hpa"]
        )
        # Returns: {"kegg": [...], "reactome": [...], "hpa": [...]}
    """
    if servers is None:
        servers = ["kegg", "reactome", "hpa"]

    server_queries = {}
    for server in servers:
        if hasattr(mcp_manager, server):
            client = getattr(mcp_manager, server)
            if server == "kegg":
                server_queries[server] = lambda: client.search_diseases(disease_query)
            elif server == "reactome":
                server_queries[server] = lambda: client.find_pathways_by_disease(disease_query)
            elif server == "hpa":
                server_queries[server] = lambda: client.search_cancer_markers(disease_query)

    return await parallel_server_query(mcp_manager, server_queries)


# Performance monitoring

class BatchQueryStats:
    """Track performance statistics for batch queries."""

    def __init__(self):
        self.total_queries = 0
        self.total_time = 0.0
        self.success_count = 0
        self.error_count = 0
        self.by_server = defaultdict(lambda: {"count": 0, "time": 0.0, "success": 0})

    def record_query(self, server_name: str, duration: float, success: bool):
        """Record a query execution."""
        self.total_queries += 1
        self.total_time += duration
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        self.by_server[server_name]["count"] += 1
        self.by_server[server_name]["time"] += duration
        if success:
            self.by_server[server_name]["success"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        avg_time = self.total_time / self.total_queries if self.total_queries > 0 else 0
        success_rate = self.success_count / self.total_queries if self.total_queries > 0 else 0

        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_time": avg_time,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "by_server": dict(self.by_server)
        }

    def print_summary(self):
        """Print formatted performance summary."""
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("Batch Query Performance Summary")
        print("=" * 80)
        print(f"Total Queries: {summary['total_queries']}")
        print(f"Total Time: {summary['total_time']:.2f}s")
        print(f"Average Time: {summary['average_time']:.3f}s/query")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print("\nBy Server:")
        for server, stats in summary['by_server'].items():
            server_rate = stats['success'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {server}: {stats['count']} queries, "
                  f"{stats['time']:.2f}s, {server_rate:.1%} success")
        print("=" * 80 + "\n")


# Global stats instance
_batch_stats = BatchQueryStats()


def get_batch_stats() -> BatchQueryStats:
    """Get the global batch query statistics instance."""
    return _batch_stats


def print_batch_stats():
    """Print batch query performance statistics."""
    _batch_stats.print_summary()
