"""
Intelligent caching for MCP calls.

Provides file-based caching with TTL expiration to avoid redundant MCP calls.
"""

import asyncio
import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union
import aiofiles
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MCPCache:
    """
    Intelligent caching for MCP calls with TTL expiration.
    
    Caches MCP responses to avoid redundant calls:
    - Disease/pathway lookups: 24hr TTL
    - Network data: 1hr TTL
    - Expression data: 12hr TTL
    """
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        """
        Initialize MCP cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(exist_ok=True)
        
        # TTL settings for different data types
        self.ttl_settings = {
            'disease': 24 * 3600,  # 24 hours
            'pathway': 24 * 3600,  # 24 hours
            'network': 1 * 3600,   # 1 hour
            'expression': 12 * 3600,  # 12 hours
            'interaction': 6 * 3600,  # 6 hours
            'default': default_ttl
        }
    
    def _generate_cache_key(self, server: str, tool: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key from server, tool, and parameters.
        
        Args:
            server: MCP server name
            tool: Tool name
            params: Tool parameters
            
        Returns:
            Cache key string
        """
        # Create deterministic key from parameters
        key_data = {
            'server': server,
            'tool': tool,
            'params': sorted(params.items()) if params else {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _get_data_type_from_tool(self, tool: str) -> str:
        """
        Determine data type from tool name for TTL selection.
        
        Args:
            tool: Tool name
            
        Returns:
            Data type for TTL lookup
        """
        if any(keyword in tool.lower() for keyword in ['disease', 'pathway']):
            return 'disease'
        elif any(keyword in tool.lower() for keyword in ['network', 'interaction']):
            return 'network'
        elif any(keyword in tool.lower() for keyword in ['expression', 'tissue']):
            return 'expression'
        elif any(keyword in tool.lower() for keyword in ['interaction', 'binding']):
            return 'interaction'
        else:
            return 'default'
    
    async def get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached data if it exists and is not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            async with aiofiles.open(cache_file, 'r') as f:
                cache_data = json.loads(await f.read())
            
            # Check if expired
            if 'expires_at' in cache_data:
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                if datetime.now() > expires_at:
                    # Remove expired cache file
                    cache_file.unlink()
                    return None
            
            # Return cached data
            return cache_data.get('data')
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            # Remove corrupted cache file
            if cache_file.exists():
                cache_file.unlink()
            return None
    
    async def set_cached_data(
        self, 
        cache_key: str, 
        data: Any, 
        ttl_override: Optional[int] = None
    ) -> None:
        """
        Cache data with TTL.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            ttl_override: Override TTL in seconds
        """
        cache_file = self._get_cache_file_path(cache_key)
        
        # Determine TTL
        ttl = ttl_override if ttl_override is not None else self.default_ttl
        
        # Calculate expiration time
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        # Create cache data
        cache_data = {
            'data': data,
            'cached_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat(),
            'ttl': ttl
        }
        
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))
            
            logger.debug(f"Cached data for key {cache_key} (expires at {expires_at})")
            
        except Exception as e:
            logger.error(f"Failed to write cache file {cache_file}: {e}")
    
    async def get_or_fetch(
        self,
        server: str,
        tool: str,
        params: Dict[str, Any],
        fetch_fn: Callable,
        ttl_override: Optional[int] = None
    ) -> Any:
        """
        Get data from cache or fetch if not cached/expired.
        
        Args:
            server: MCP server name
            tool: Tool name
            params: Tool parameters
            fetch_fn: Function to fetch data if not cached
            ttl_override: Override TTL in seconds
            
        Returns:
            Cached or fetched data
        """
        # Generate cache key
        cache_key = self._generate_cache_key(server, tool, params)
        
        # Try to get from cache
        cached_data = await self.get_cached_data(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {server}.{tool}")
            return cached_data
        
        # Fetch data
        logger.debug(f"Cache miss for {server}.{tool}, fetching...")
        try:
            data = await fetch_fn()
            
            # Determine TTL
            if ttl_override is None:
                data_type = self._get_data_type_from_tool(tool)
                ttl_override = self.ttl_settings.get(data_type, self.default_ttl)
            
            # Cache the data
            await self.set_cached_data(cache_key, data, ttl_override)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {server}.{tool}: {e}")
            raise
    
    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match cache keys
            
        Returns:
            Number of cache files removed
        """
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if pattern and pattern not in cache_file.name:
                    continue
                
                cache_file.unlink()
                removed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Removed {removed_count} cache files")
        return removed_count
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        total_files = 0
        total_size = 0
        expired_files = 0
        data_types = {}
        
        for cache_file in self.cache_dir.glob("*.json"):
            total_files += 1
            total_size += cache_file.stat().st_size
            
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    cache_data = json.loads(await f.read())
                
                # Check if expired
                if 'expires_at' in cache_data:
                    expires_at = datetime.fromisoformat(cache_data['expires_at'])
                    if datetime.now() > expires_at:
                        expired_files += 1
                
                # Count by data type
                ttl = cache_data.get('ttl', self.default_ttl)
                data_type = 'default'
                for dt, dt_ttl in self.ttl_settings.items():
                    if ttl == dt_ttl:
                        data_type = dt
                        break
                
                data_types[data_type] = data_types.get(data_type, 0) + 1
                
            except Exception:
                # Count as expired if we can't read it
                expired_files += 1
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'expired_files': expired_files,
            'data_types': data_types
        }
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache files.
        
        Returns:
            Number of expired files removed
        """
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    cache_data = json.loads(await f.read())
                
                # Check if expired
                if 'expires_at' in cache_data:
                    expires_at = datetime.fromisoformat(cache_data['expires_at'])
                    if datetime.now() > expires_at:
                        cache_file.unlink()
                        removed_count += 1
                
            except Exception:
                # Remove corrupted files
                cache_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache files")
        
        return removed_count


# Global cache instance
_cache_instance: Optional[MCPCache] = None


def get_cache() -> MCPCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MCPCache()
    return _cache_instance


def set_cache(cache: MCPCache) -> None:
    """Set global cache instance."""
    global _cache_instance
    _cache_instance = cache


# Convenience functions
async def cached_mcp_call(
    server: str,
    tool: str,
    params: Dict[str, Any],
    fetch_fn: Callable,
    ttl_override: Optional[int] = None
) -> Any:
    """
    Convenience function for cached MCP calls.
    
    Args:
        server: MCP server name
        tool: Tool name
        params: Tool parameters
        fetch_fn: Function to fetch data if not cached
        ttl_override: Override TTL in seconds
        
    Returns:
        Cached or fetched data
    """
    cache = get_cache()
    return await cache.get_or_fetch(server, tool, params, fetch_fn, ttl_override)


async def clear_cache(pattern: Optional[str] = None) -> int:
    """
    Clear cache entries.
    
    Args:
        pattern: Optional pattern to match cache keys
        
    Returns:
        Number of cache files removed
    """
    cache = get_cache()
    return await cache.invalidate_cache(pattern)


async def get_cache_info() -> Dict[str, Any]:
    """
    Get cache information.
    
    Returns:
        Cache statistics
    """
    cache = get_cache()
    return await cache.get_cache_stats()
