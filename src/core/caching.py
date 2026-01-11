"""
OmniTarget Pipeline Caching System

High-performance caching for MCP responses, simulation results, and network data.
Supports both in-memory and Redis caching with intelligent cache invalidation.
"""

import json
import hashlib
import time
import asyncio
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    # Cache types
    use_memory_cache: bool = True
    use_redis_cache: bool = False
    
    # Memory cache settings - OPTIMIZED for faster execution
    max_memory_size: int = 5000  # Increased from 1000 for more cache hits
    memory_ttl: int = 7200  # Increased from 3600 (2 hours) for better reuse
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ttl: int = 7200  # Time to live in seconds (2 hours)
    
    # Cache policies
    enable_compression: bool = True
    enable_serialization: bool = True
    cache_mcp_responses: bool = True
    cache_simulation_results: bool = True
    cache_network_data: bool = True
    
    # Performance settings
    max_concurrent_operations: int = 10
    cache_hit_threshold: float = 0.8  # Minimum hit rate for cache effectiveness


class CacheKey:
    """Generate consistent cache keys for different data types."""
    
    @staticmethod
    def mcp_response(server: str, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key for MCP responses."""
        param_str = json.dumps(params, sort_keys=True)
        content = f"mcp:{server}:{method}:{param_str}"
        return f"omnitarget:{hashlib.md5(content.encode()).hexdigest()}"
    
    @staticmethod
    def simulation_result(target: str, mode: str, network_hash: str) -> str:
        """Generate cache key for simulation results."""
        content = f"sim:{target}:{mode}:{network_hash}"
        return f"omnitarget:{hashlib.md5(content.encode()).hexdigest()}"
    
    @staticmethod
    def network_data(network_id: str, version: str = "1.0") -> str:
        """Generate cache key for network data."""
        content = f"network:{network_id}:{version}"
        return f"omnitarget:{hashlib.md5(content.encode()).hexdigest()}"
    
    @staticmethod
    def pathway_data(pathway_id: str, source_db: str) -> str:
        """Generate cache key for pathway data."""
        content = f"pathway:{pathway_id}:{source_db}"
        return f"omnitarget:{hashlib.md5(content.encode()).hexdigest()}"
    
    @staticmethod
    def expression_data(gene: str, tissue: str) -> str:
        """Generate cache key for expression data."""
        content = f"expression:{gene}:{tissue}"
        return f"omnitarget:{hashlib.md5(content.encode()).hexdigest()}"


class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            entry = self.cache[key]
            if time.time() > entry['expires_at']:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
            
            # Update access time for LRU
            self.access_times[key] = time.time()
            return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        async with self.lock:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            
            # Evict if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self.access_times[key] = time.time()
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        if lru_key in self.cache:
            del self.cache[lru_key]
        if lru_key in self.access_times:
            del self.access_times[lru_key]
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self.lock:
            current_time = time.time()
            expired_count = sum(
                1 for entry in self.cache.values() 
                if current_time > entry['expires_at']
            )
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'expired_entries': expired_count,
                'hit_rate': getattr(self, '_hit_rate', 0.0)
            }


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: Optional[str] = None, 
                 default_ttl: int = 7200):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=True
        )
        self.default_ttl = default_ttl
        self._hit_count = 0
        self._miss_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = self.redis_client.get(key)
            if value is None:
                self._miss_count += 1
                return None
            
            self._hit_count += 1
            return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self._miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries with omnitarget prefix."""
        try:
            keys = self.redis_client.keys("omnitarget:*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
    
    async def stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            info = self.redis_client.info()
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0
            
            return {
                'redis_info': info,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'hit_rate': hit_rate,
                'memory_usage': info.get('used_memory_human', 'unknown')
            }
        except Exception as e:
            logger.warning(f"Redis stats error: {e}")
            return {'error': str(e)}


class OmniTargetCache:
    """Unified caching system for OmniTarget pipeline."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache: Optional[MemoryCache] = None
        self.redis_cache: Optional[RedisCache] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize cache systems."""
        if self.config.use_memory_cache:
            self.memory_cache = MemoryCache(
                max_size=self.config.max_memory_size,
                default_ttl=self.config.memory_ttl
            )
        
        if self.config.use_redis_cache and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    default_ttl=self.config.redis_ttl
                )
                # Test Redis connection
                await self.redis_cache.get("test")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
                self.redis_cache = None
        
        self._initialized = True
        logger.info(f"Cache initialized - Memory: {self.memory_cache is not None}, Redis: {self.redis_cache is not None}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (tries memory first, then Redis)."""
        if not self._initialized:
            await self.initialize()
        
        # Try memory cache first
        if self.memory_cache:
            value = await self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Try Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                if self.memory_cache:
                    await self.memory_cache.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (stores in both memory and Redis)."""
        if not self._initialized:
            await self.initialize()
        
        # Store in memory cache
        if self.memory_cache:
            await self.memory_cache.set(key, value, ttl)
        
        # Store in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._initialized:
            await self.initialize()
        
        deleted = False
        
        if self.memory_cache:
            deleted = await self.memory_cache.delete(key) or deleted
        
        if self.redis_cache:
            deleted = await self.redis_cache.delete(key) or deleted
        
        return deleted
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        if not self._initialized:
            await self.initialize()
        
        if self.memory_cache:
            await self.memory_cache.clear()
        
        if self.redis_cache:
            await self.redis_cache.clear()
    
    async def get_or_set(self, key: str, factory_func, ttl: Optional[int] = None) -> Any:
        """Get value from cache or compute and store using factory function."""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Compute value using factory function
        if asyncio.iscoroutinefunction(factory_func):
            value = await factory_func()
        else:
            value = factory_func()
        
        # Store in cache
        await self.set(key, value, ttl)
        return value
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        if not self.redis_cache:
            return 0
        
        try:
            keys = self.redis_cache.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_cache.redis_client.delete(*keys)
                return deleted
        except Exception as e:
            logger.warning(f"Pattern invalidation error: {e}")
        
        return 0
    
    async def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self._initialized:
            await self.initialize()
        
        stats = {
            'memory_cache': None,
            'redis_cache': None,
            'total_entries': 0
        }
        
        if self.memory_cache:
            stats['memory_cache'] = await self.memory_cache.stats()
            stats['total_entries'] += stats['memory_cache']['size']
        
        if self.redis_cache:
            stats['redis_cache'] = await self.redis_cache.stats()
        
        return stats


# Cache decorators for easy integration
def cache_result(ttl: int = 3600, key_func: Optional[callable] = None):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = f"omnitarget:{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Try to get from cache
            # Note: This requires access to the global cache instance
            # In practice, you'd pass the cache instance to the decorator
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global cache instance
_global_cache: Optional[OmniTargetCache] = None


async def get_global_cache() -> OmniTargetCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        config = CacheConfig()
        _global_cache = OmniTargetCache(config)
        await _global_cache.initialize()
    return _global_cache


async def cache_mcp_response(server: str, method: str, params: Dict[str, Any], 
                           response: Any, ttl: int = 3600) -> None:
    """Cache MCP response."""
    cache = await get_global_cache()
    key = CacheKey.mcp_response(server, method, params)
    await cache.set(key, response, ttl)


async def get_cached_mcp_response(server: str, method: str, 
                                params: Dict[str, Any]) -> Optional[Any]:
    """Get cached MCP response."""
    cache = await get_global_cache()
    key = CacheKey.mcp_response(server, method, params)
    return await cache.get(key)


async def cache_simulation_result(target: str, mode: str, network_hash: str,
                                result: Any, ttl: int = 7200) -> None:
    """Cache simulation result."""
    cache = await get_global_cache()
    key = CacheKey.simulation_result(target, mode, network_hash)
    await cache.set(key, result, ttl)


async def get_cached_simulation_result(target: str, mode: str, 
                                    network_hash: str) -> Optional[Any]:
    """Get cached simulation result."""
    cache = await get_global_cache()
    key = CacheKey.simulation_result(target, mode, network_hash)
    return await cache.get(key)
