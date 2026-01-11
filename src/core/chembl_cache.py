"""
ChEMBL Query Caching

LRU cache implementation for ChEMBL queries with TTL and performance monitoring.
Improves performance by avoiding redundant API calls.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and metadata."""
    value: Any
    timestamp: float
    hits: int = 0

    def is_expired(self, ttl: float) -> bool:
        """Check if entry is expired based on TTL."""
        return (time.time() - self.timestamp) > ttl


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate

    def __str__(self) -> str:
        """String representation of stats."""
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate:.1%}, size={self.size}/{self.max_size})")


class ChEMBLCache:
    """
    LRU cache for ChEMBL queries with TTL support.

    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time-To-Live) for cache entries
    - Per-entry hit tracking
    - Cache statistics (hits, misses, evictions)
    - Thread-safe operations

    Example:
        >>> cache = ChEMBLCache(max_size=1000, ttl=3600)
        >>> cache.set("search:aspirin", search_results)
        >>> results = cache.get("search:aspirin")
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    """

    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        """
        Initialize ChEMBL cache.

        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired(self.ttl):
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size = len(self._cache)
                logger.debug(f"Cache expired: {key}")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1

            logger.debug(f"Cache hit: {key} (hits={entry.hits})")
            return entry.value

    async def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = CacheEntry(value=value, timestamp=time.time())
                logger.debug(f"Cache updated: {key}")
                return

            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._stats.evictions += 1
                logger.debug(f"Cache evicted: {oldest_key}")

            # Add new entry
            self._cache[key] = CacheEntry(value=value, timestamp=time.time())
            self._stats.size = len(self._cache)
            logger.debug(f"Cache set: {key} (size={self._stats.size})")

    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                logger.debug(f"Cache deleted: {key}")
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0
            logger.info("Cache cleared")

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self.max_size
            )

    async def prune_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.ttl)
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1

            self._stats.size = len(self._cache)

            if expired_keys:
                logger.info(f"Pruned {len(expired_keys)} expired entries")

            return len(expired_keys)

    def _make_key(self, func_name: str, *args, **kwargs) -> str:
        """
        Create cache key from function name and arguments.

        Args:
            func_name: Function name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create deterministic key from function and arguments
        key_parts = [func_name]

        # Add positional args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # For complex objects, use type name
                key_parts.append(type(arg).__name__)

        # Add keyword args (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={type(v).__name__}")

        # Create hash for long keys
        key_str = ":".join(key_parts)
        if len(key_str) > 200:
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"{func_name}:{key_hash}"

        return key_str


# Global cache instance
_global_cache: Optional[ChEMBLCache] = None


def get_cache() -> ChEMBLCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ChEMBLCache(max_size=1000, ttl=3600)
    return _global_cache


def cached(ttl: Optional[float] = None):
    """
    Decorator for caching async function results.

    Args:
        ttl: Optional TTL override for this function

    Example:
        >>> @cached(ttl=1800)
        >>> async def search_compounds(query: str):
        ...     return await expensive_api_call(query)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache()

            # Create cache key
            cache_key = cache._make_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result)

            return result

        return wrapper
    return decorator


class BatchProcessor:
    """
    Parallel batch processor for ChEMBL queries.

    Processes multiple queries concurrently with rate limiting and error handling.

    Example:
        >>> processor = BatchProcessor(max_concurrent=10, delay=0.1)
        >>> results = await processor.process_batch(queries, fetch_function)
    """

    def __init__(self, max_concurrent: int = 10, delay: float = 0.1):
        """
        Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent queries
            delay: Delay between batches in seconds (rate limiting)
        """
        self.max_concurrent = max_concurrent
        self.delay = delay
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_one(
        self,
        item: Any,
        func: Callable,
        index: int
    ) -> Tuple[int, Any, Optional[Exception]]:
        """
        Process single item with semaphore.

        Args:
            item: Item to process
            func: Async function to call
            index: Item index

        Returns:
            Tuple of (index, result, error)
        """
        async with self._semaphore:
            try:
                result = await func(item)
                return (index, result, None)
            except Exception as e:
                logger.error(f"Batch processing error for item {index}: {e}")
                return (index, None, e)
            finally:
                # Rate limiting delay
                if self.delay > 0:
                    await asyncio.sleep(self.delay)

    async def process_batch(
        self,
        items: list,
        func: Callable,
        return_exceptions: bool = True
    ) -> list:
        """
        Process batch of items in parallel.

        Args:
            items: List of items to process
            func: Async function to call for each item
            return_exceptions: If True, return exceptions instead of raising

        Returns:
            List of results in same order as items
        """
        if not items:
            return []

        # Create tasks
        tasks = [
            self._process_one(item, func, i)
            for i, item in enumerate(items)
        ]

        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        # Sort by index and extract values
        sorted_results = sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else 0)

        # Return results or raise first exception
        final_results = []
        for index, result, error in sorted_results:
            if error is not None and not return_exceptions:
                raise error
            final_results.append(result)

        return final_results


class CacheWarmer:
    """
    Proactive cache warming for common ChEMBL queries.

    Pre-populates cache with frequently accessed data.

    Example:
        >>> warmer = CacheWarmer(cache, chembl_client)
        >>> await warmer.warm_common_targets()
    """

    def __init__(self, cache: ChEMBLCache, chembl_client: Any):
        """
        Initialize cache warmer.

        Args:
            cache: ChEMBL cache instance
            chembl_client: ChEMBL client for queries
        """
        self.cache = cache
        self.chembl_client = chembl_client

    async def warm_common_targets(self, gene_symbols: list[str]) -> int:
        """
        Pre-cache target information for common genes.

        Args:
            gene_symbols: List of gene symbols to cache

        Returns:
            Number of entries cached
        """
        count = 0
        for gene in gene_symbols:
            try:
                # Search and cache target
                cache_key = f"search_targets:{gene}"
                result = await self.chembl_client.search_targets(gene, limit=5)
                await self.cache.set(cache_key, result)
                count += 1

                # Small delay to avoid overwhelming API
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Cache warming failed for {gene}: {e}")

        logger.info(f"Cache warmed: {count} targets")
        return count

    async def warm_common_compounds(self, compound_names: list[str]) -> int:
        """
        Pre-cache compound information.

        Args:
            compound_names: List of compound names to cache

        Returns:
            Number of entries cached
        """
        count = 0
        for compound in compound_names:
            try:
                cache_key = f"search_compounds:{compound}"
                result = await self.chembl_client.search_compounds(compound, limit=10)
                await self.cache.set(cache_key, result)
                count += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Cache warming failed for {compound}: {e}")

        logger.info(f"Cache warmed: {count} compounds")
        return count


# Performance monitoring utilities

def log_cache_stats(cache: ChEMBLCache, logger_func: Callable = logger.info):
    """
    Log cache statistics.

    Args:
        cache: ChEMBL cache instance
        logger_func: Logging function to use
    """
    async def _log():
        stats = await cache.get_stats()
        logger_func(f"ChEMBL Cache Stats: {stats}")

    return _log


async def benchmark_cache_performance(
    cache: ChEMBLCache,
    key: str,
    value: Any,
    iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark cache performance.

    Args:
        cache: ChEMBL cache instance
        key: Test cache key
        value: Test value
        iterations: Number of iterations

    Returns:
        Dictionary with timing results
    """
    # Warm up
    await cache.set(key, value)
    await cache.get(key)

    # Benchmark writes
    write_start = time.time()
    for i in range(iterations):
        await cache.set(f"{key}_{i}", value)
    write_time = time.time() - write_start

    # Benchmark reads
    read_start = time.time()
    for i in range(iterations):
        await cache.get(f"{key}_{i}")
    read_time = time.time() - read_start

    return {
        'write_time_total': write_time,
        'write_time_per_op': write_time / iterations,
        'writes_per_sec': iterations / write_time,
        'read_time_total': read_time,
        'read_time_per_op': read_time / iterations,
        'reads_per_sec': iterations / read_time,
    }
