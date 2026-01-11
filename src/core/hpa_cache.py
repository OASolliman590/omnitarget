"""
HPA Query Caching

LRU cache implementation for HPA queries with TTL and performance monitoring.
Improves performance by avoiding redundant API calls to HPA server.

HPA data changes infrequently, so longer TTL values are appropriate.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Dict, Optional, Callable, Tuple, List
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


class HPACache:
    """
    LRU cache for HPA queries with TTL support.

    HPA data (tissue expression, subcellular location, etc.) changes infrequently,
    so longer TTL values are appropriate (default: 24 hours).

    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time-To-Live) for cache entries
    - Per-entry hit tracking
    - Cache statistics (hits, misses, evictions)
    - Thread-safe operations
    - Optimized for HPA data patterns

    Example:
        >>> cache = HPACache(max_size=2000, ttl=86400)  # 24 hours
        >>> await cache.set("expression:AXL", expression_data)
        >>> data = await cache.get("expression:AXL")
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.1%}")
    """

    def __init__(self, max_size: int = 2000, ttl: float = 86400):
        """
        Initialize HPA cache.

        Args:
            max_size: Maximum number of entries (default: 2000)
                      HPA has fewer unique queries than ChEMBL
            ttl: Time-to-live in seconds (default: 86400 = 24 hours)
                 HPA data changes infrequently, so longer TTL is appropriate
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
                # Remove expired entry
                del self._cache[key]
                self._stats.size -= 1
                self._stats.misses += 1
                logger.debug(f"Cache entry expired: {key}")
                return None

            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1

            logger.debug(f"Cache hit: {key} (hits: {entry.hits})")
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
                self._cache[key].value = value
                self._cache[key].timestamp = time.time()
                self._cache.move_to_end(key)
                logger.debug(f"Cache updated: {key}")
                return

            # Check if cache is full
            if len(self._cache) >= self.max_size:
                # Remove least recently used item
                lru_key, lru_entry = self._cache.popitem(last=False)
                self._stats.evictions += 1
                self._stats.size -= 1
                logger.debug(f"Cache evicted: {lru_key} (hits: {lru_entry.hits})")

            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                hits=0
            )
            self._stats.size += 1
            logger.debug(f"Cache set: {key}")

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
                self._stats.size -= 1
                logger.debug(f"Cache deleted: {key}")
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0
            logger.info("HPA cache cleared")

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats object with current statistics
        """
        return self._stats

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information.

        Returns:
            Dictionary with cache information
        """
        return {
            'max_size': self.max_size,
            'ttl_seconds': self.ttl,
            'current_size': len(self._cache),
            'stats': {
                'hits': self._stats.hits,
                'misses': self._stats.misses,
                'evictions': self._stats.evictions,
                'hit_rate': self._stats.hit_rate,
                'miss_rate': self._stats.miss_rate
            },
            'most_accessed': self._get_most_accessed(5)
        }

    def _get_most_accessed(self, n: int) -> List[Dict[str, Any]]:
        """Get most accessed entries."""
        entries = []
        for key, entry in self._cache.items():
            entries.append({
                'key': key,
                'hits': entry.hits,
                'age_seconds': time.time() - entry.timestamp
            })

        # Sort by hits (descending)
        entries.sort(key=lambda x: x['hits'], reverse=True)
        return entries[:n]

    async def warm_cache(self, queries: List[Dict[str, Any]]) -> None:
        """
        Warm cache with predefined queries.

        Args:
            queries: List of query dictionaries with 'key' and 'value' keys

        Example:
            >>> queries = [
            ...     {'key': 'expression:AXL', 'value': axl_data},
            ...     {'key': 'protein:EGFR', 'value': egfr_data}
            ... ]
            >>> await cache.warm_cache(queries)
        """
        logger.info(f"Warming HPA cache with {len(queries)} entries")
        for query in queries:
            await self.set(query['key'], query['value'])
        logger.info("HPA cache warming complete")


# Global HPA cache instance
_hpa_cache: Optional[HPACache] = None


def get_hpa_cache() -> HPACache:
    """
    Get global HPA cache instance.

    Returns:
        HPACache instance
    """
    global _hpa_cache
    if _hpa_cache is None:
        _hpa_cache = HPACache(max_size=2000, ttl=86400)
    return _hpa_cache


def cached_hpa(ttl: Optional[float] = None):
    """
    Decorator to cache HPA query results.

    Args:
        ttl: Optional TTL override (uses cache default if not specified)

    Example:
        @cached_hpa(ttl=3600)  # 1 hour
        async def get_tissue_expression(gene: str):
            return await hpa_client.get_tissue_expression(gene)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            key = _generate_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            cache = get_hpa_cache()
            cached_value = await cache.get(key)

            if cached_value is not None:
                logger.debug(f"HPA cache hit for {func.__name__}")
                return cached_value

            # Call function
            logger.debug(f"HPA cache miss for {func.__name__}, querying HPA...")
            result = await func(*args, **kwargs)

            # Cache result
            await cache.set(key, result)

            return result

        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Generate cache key from function name and arguments.

    Args:
        func_name: Name of function
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Create a hash from function name and arguments
    key_data = {
        'function': func_name,
        'args': args,
        'kwargs': sorted(kwargs.items())
    }

    # Convert to JSON string
    key_str = json.dumps(key_data, sort_keys=True, default=str)

    # Generate hash (shortened for readability)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:16]

    return f"{func_name}:{key_hash}"


# Convenience functions for common HPA queries
async def get_cached_tissue_expression(gene: str) -> Optional[Dict[str, Any]]:
    """
    Get cached tissue expression data for a gene.

    Args:
        gene: Gene symbol

    Returns:
        Cached expression data or None
    """
    cache = get_hpa_cache()
    key = f"tissue_expression:{gene}"
    return await cache.get(key)


async def set_cached_tissue_expression(gene: str, data: Dict[str, Any]) -> None:
    """
    Cache tissue expression data for a gene.

    Args:
        gene: Gene symbol
        data: Expression data
    """
    cache = get_hpa_cache()
    key = f"tissue_expression:{gene}"
    await cache.set(key, data)


async def get_cached_protein_info(gene: str) -> Optional[Dict[str, Any]]:
    """
    Get cached protein info for a gene.

    Args:
        gene: Gene symbol

    Returns:
        Cached protein info or None
    """
    cache = get_hpa_cache()
    key = f"protein_info:{gene}"
    return await cache.get(key)


async def set_cached_protein_info(gene: str, data: Dict[str, Any]) -> None:
    """
    Cache protein info for a gene.

    Args:
        gene: Gene symbol
        data: Protein info data
    """
    cache = get_hpa_cache()
    key = f"protein_info:{gene}"
    await cache.set(key, data)


async def get_cached_subcellular_location(gene: str) -> Optional[Dict[str, Any]]:
    """
    Get cached subcellular location for a gene.

    Args:
        gene: Gene symbol

    Returns:
        Cached location data or None
    """
    cache = get_hpa_cache()
    key = f"subcellular_location:{gene}"
    return await cache.get(key)


async def set_cached_subcellular_location(gene: str, data: Dict[str, Any]) -> None:
    """
    Cache subcellular location for a gene.

    Args:
        gene: Gene symbol
        data: Location data
    """
    cache = get_hpa_cache()
    key = f"subcellular_location:{gene}"
    await cache.set(key, data)


async def clear_hpa_cache() -> None:
    """Clear the global HPA cache."""
    cache = get_hpa_cache()
    await cache.clear()


async def get_hpa_cache_stats() -> Dict[str, Any]:
    """
    Get HPA cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    cache = get_hpa_cache()
    return cache.get_cache_info()
