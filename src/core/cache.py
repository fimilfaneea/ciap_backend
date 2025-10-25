"""
Cache System for CIAP
SQLite-based caching with optional in-memory layer for improved performance
"""

import asyncio
import hashlib
import json
import zlib
import re
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
from cachetools import TTLCache
from sqlalchemy import select, delete, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import db_manager
from src.core.models import Cache as CacheModel


class CacheManager:
    """SQLite-based cache with optional in-memory layer"""

    def __init__(self, enable_memory_cache: bool = True):
        """
        Initialize cache manager

        Args:
            enable_memory_cache: Enable in-memory cache layer for faster access
        """
        self.enable_memory_cache = enable_memory_cache

        # In-memory cache (LRU with TTL)
        if enable_memory_cache:
            self.memory_cache = TTLCache(
                maxsize=1000,  # Maximum 1000 items in memory
                ttl=60  # 60 seconds memory cache TTL
            )
        else:
            self.memory_cache = None

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "memory_hits": 0,
            "db_hits": 0
        }

        # Background cleanup task
        self.cleanup_task = None

    async def initialize(self):
        """Initialize cache and start cleanup task"""
        # Start background cleanup
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())

        # Initial cleanup
        await self.cleanup_expired()

    async def get(
        self,
        key: str,
        default: Any = None,
        deserialize: bool = True
    ) -> Optional[Any]:
        """
        Get cached value by key

        Args:
            key: Cache key
            default: Default value if not found
            deserialize: Deserialize JSON value

        Returns:
            Cached value or default
        """
        # Check memory cache first
        if self.memory_cache is not None:
            if key in self.memory_cache:
                self.stats["memory_hits"] += 1
                self.stats["hits"] += 1
                value = self.memory_cache[key]
                return json.loads(value) if deserialize else value

        # Check database cache
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(CacheModel).where(
                    and_(
                        CacheModel.key == key,
                        CacheModel.expires_at > datetime.utcnow()
                    )
                )
            )
            cache_entry = result.scalar_one_or_none()

            if cache_entry:
                self.stats["db_hits"] += 1
                self.stats["hits"] += 1

                # Store in memory cache
                if self.memory_cache is not None:
                    self.memory_cache[key] = cache_entry.value

                value = cache_entry.value
                return json.loads(value) if deserialize else value
            else:
                self.stats["misses"] += 1
                return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set cache value with TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default from config)
            serialize: Serialize value to JSON

        Returns:
            Success status
        """
        try:
            # Use default TTL if not specified
            if ttl is None:
                ttl = settings.CACHE_TTL_SECONDS

            # Serialize value
            cached_value = json.dumps(value) if serialize else value
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            # Store in memory cache
            if self.memory_cache is not None:
                self.memory_cache[key] = cached_value

            # Store in database
            async with db_manager.get_session() as session:
                # Delete existing entry
                await session.execute(
                    delete(CacheModel).where(CacheModel.key == key)
                )

                # Insert new entry
                cache_entry = CacheModel(
                    key=key,
                    value=cached_value,
                    expires_at=expires_at
                )
                session.add(cache_entry)
                await session.commit()

            self.stats["sets"] += 1
            return True

        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete cache entry

        Args:
            key: Cache key

        Returns:
            Success status
        """
        try:
            # Remove from memory cache
            if self.memory_cache is not None and key in self.memory_cache:
                del self.memory_cache[key]

            # Remove from database
            async with db_manager.get_session() as session:
                result = await session.execute(
                    delete(CacheModel).where(CacheModel.key == key)
                )
                await session.commit()

            self.stats["deletes"] += 1
            return result.rowcount > 0

        except Exception as e:
            print(f"Cache delete error: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete cache entries matching pattern

        Args:
            pattern: SQL LIKE pattern (e.g., 'user_%')

        Returns:
            Number of deleted entries
        """
        try:
            # Clear memory cache (simple approach)
            if self.memory_cache is not None:
                keys_to_delete = [
                    k for k in list(self.memory_cache.keys())
                    if self._match_pattern(k, pattern)
                ]
                for key in keys_to_delete:
                    del self.memory_cache[key]

            # Delete from database
            async with db_manager.get_session() as session:
                result = await session.execute(
                    delete(CacheModel).where(CacheModel.key.like(pattern))
                )
                await session.commit()
                return result.rowcount

        except Exception as e:
            print(f"Cache pattern delete error: {e}")
            return 0

    async def clear(self) -> int:
        """
        Clear all cache entries

        Returns:
            Number of deleted entries
        """
        try:
            # Clear memory cache
            if self.memory_cache is not None:
                self.memory_cache.clear()

            # Clear database cache
            async with db_manager.get_session() as session:
                result = await session.execute(delete(CacheModel))
                await session.commit()
                return result.rowcount

        except Exception as e:
            print(f"Cache clear error: {e}")
            return 0

    async def cleanup_expired(self) -> int:
        """
        Remove expired cache entries

        Returns:
            Number of cleaned entries
        """
        try:
            async with db_manager.get_session() as session:
                result = await session.execute(
                    delete(CacheModel).where(
                        CacheModel.expires_at < datetime.utcnow()
                    )
                )
                await session.commit()
                return result.rowcount

        except Exception as e:
            print(f"Cache cleanup error: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Cache statistics dictionary
        """
        # Get database cache size
        async with db_manager.get_session() as session:
            total_result = await session.execute(
                select(func.count()).select_from(CacheModel)
            )
            total_entries = total_result.scalar()

            expired_result = await session.execute(
                select(func.count()).select_from(CacheModel).where(
                    CacheModel.expires_at < datetime.utcnow()
                )
            )
            expired_entries = expired_result.scalar()

        # Calculate hit rate
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            (self.stats["hits"] / total_requests * 100)
            if total_requests > 0 else 0
        )

        return {
            **self.stats,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "hit_rate": f"{hit_rate:.2f}%",
            "memory_cache_size": len(self.memory_cache) if self.memory_cache else 0
        }

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        value = await self.get(key, deserialize=False)
        return value is not None

    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for key

        Args:
            key: Cache key

        Returns:
            Remaining seconds or None if not found
        """
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(CacheModel.expires_at).where(CacheModel.key == key)
            )
            expires_at = result.scalar_one_or_none()

            if expires_at:
                remaining = (expires_at - datetime.utcnow()).total_seconds()
                return max(0, int(remaining))
            return None

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(key_parts)

        # Hash long keys
        if len(key_string) > 200:
            return hashlib.md5(key_string.encode()).hexdigest()
        return key_string

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple cache entries efficiently (batch operation)

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values
        """
        results = {}

        # Check memory cache first
        db_keys = []
        for key in keys:
            if self.memory_cache is not None and key in self.memory_cache:
                results[key] = json.loads(self.memory_cache[key])
                self.stats["memory_hits"] += 1
                self.stats["hits"] += 1
            else:
                db_keys.append(key)

        # Batch database query for remaining keys
        if db_keys:
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(CacheModel).where(
                        and_(
                            CacheModel.key.in_(db_keys),
                            CacheModel.expires_at > datetime.utcnow()
                        )
                    )
                )
                entries = result.scalars().all()

                for entry in entries:
                    value = json.loads(entry.value)
                    results[entry.key] = value
                    # Update memory cache
                    if self.memory_cache is not None:
                        self.memory_cache[entry.key] = entry.value
                    self.stats["db_hits"] += 1
                    self.stats["hits"] += 1

        # Count misses
        found_keys = set(results.keys())
        misses = len(keys) - len(found_keys)
        self.stats["misses"] += misses

        return results

    async def set_compressed(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        threshold: int = 1000
    ) -> bool:
        """
        Set large values with compression

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            threshold: Size threshold in bytes for compression

        Returns:
            Success status
        """
        json_str = json.dumps(value)

        if len(json_str) > threshold:
            # Compress if over threshold
            compressed = zlib.compress(json_str.encode())
            return await self.set(f"compressed:{key}", compressed, ttl, serialize=False)
        else:
            # Store normally
            return await self.set(key, value, ttl)

    async def get_compressed(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get compressed cache value

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Decompressed value or default
        """
        # Try compressed key first
        compressed_value = await self.get(f"compressed:{key}", deserialize=False)

        if compressed_value is not None:
            try:
                decompressed = zlib.decompress(compressed_value)
                return json.loads(decompressed.decode())
            except Exception as e:
                print(f"Decompression error: {e}")
                return default

        # Try normal key
        return await self.get(key, default)

    async def _cleanup_worker(self):
        """Background task to cleanup expired entries"""
        while True:
            try:
                await asyncio.sleep(settings.CACHE_CLEANUP_INTERVAL)
                deleted = await self.cleanup_expired()
                if deleted > 0:
                    print(f"[Cache] Cleaned up {deleted} expired entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Cache] Cleanup worker error: {e}")

    @staticmethod
    def _match_pattern(key: str, pattern: str) -> bool:
        """
        Simple pattern matching for memory cache

        Args:
            key: Cache key
            pattern: SQL LIKE pattern (% and _)

        Returns:
            True if key matches pattern
        """
        # Convert SQL LIKE pattern to regex
        regex_pattern = pattern.replace("%", ".*").replace("_", ".")
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.match(regex_pattern, key))

    async def close(self):
        """Cleanup resources"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass


# Utility functions for cache warming
async def warm_cache(operations: List[Dict[str, Any]]) -> int:
    """
    Pre-load common data into cache (cache warming)

    Args:
        operations: List of cache operations, each dict containing:
            - 'key': Cache key
            - 'value': Value to cache
            - 'ttl': Optional TTL in seconds

    Returns:
        Number of successfully cached items

    Example:
        operations = [
            {'key': 'config:settings', 'value': settings_dict, 'ttl': 3600},
            {'key': 'common:queries', 'value': query_list, 'ttl': 1800}
        ]
        cached_count = await warm_cache(operations)
    """
    success_count = 0

    for op in operations:
        try:
            key = op.get('key')
            value = op.get('value')
            ttl = op.get('ttl', None)

            if key and value is not None:
                await cache.set(key, value, ttl=ttl)
                success_count += 1
        except Exception as e:
            print(f"[Cache] Warm cache error for key {op.get('key')}: {e}")

    return success_count


# Global cache instance
cache = CacheManager()
