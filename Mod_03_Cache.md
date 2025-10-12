# Module 3: Cache System

## Overview
**Purpose:** SQLite-based caching system with TTL support for reducing redundant API calls and improving performance.

**Responsibilities:**
- Cache storage with expiration
- Get/Set/Delete operations
- Automatic expiration handling
- Cache statistics tracking
- Background cleanup of expired entries
- Optional in-memory cache layer

**Development Time:** 2 days (Week 1, Day 3-4)

---

## Interface Specification

### Input
```python
# Cache operations
key: str  # Cache key
value: Any  # Data to cache (JSON-serializable)
ttl: int  # Time-to-live in seconds
```

### Output
```python
# Cached data or None if not found/expired
cached_value: Optional[Any]
cache_stats: Dict[str, int]  # Hit rate, size, etc.
```

---

## Dependencies

### External
```txt
aiosqlite==0.19.0
redis==5.0.1  # Optional for Redis comparison
cachetools==5.3.2  # For in-memory cache
```

### Internal
- Module 1: Database Infrastructure
- Module 2: Configuration

---

## Implementation Guide

### Step 1: Cache Manager (`src/core/cache.py`)

```python
import asyncio
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Union, List
from functools import lru_cache
from cachetools import TTLCache
import aiosqlite
from sqlalchemy import select, delete, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import db_manager, get_db
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
                    k for k in self.memory_cache.keys()
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

    async def _cleanup_worker(self):
        """Background task to cleanup expired entries"""
        while True:
            try:
                await asyncio.sleep(settings.CACHE_CLEANUP_INTERVAL)
                deleted = await self.cleanup_expired()
                if deleted > 0:
                    print(f"Cleaned up {deleted} expired cache entries")
            except Exception as e:
                print(f"Cleanup worker error: {e}")

    @staticmethod
    def _match_pattern(key: str, pattern: str) -> bool:
        """Simple pattern matching for memory cache"""
        import re
        # Convert SQL LIKE pattern to regex
        regex_pattern = pattern.replace("%", ".*").replace("_", ".")
        return bool(re.match(regex_pattern, key))

    async def close(self):
        """Cleanup resources"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

# Global cache instance
cache = CacheManager()
```

### Step 2: Cache Decorators (`src/core/cache_decorators.py`)

```python
import functools
import hashlib
import json
from typing import Optional, Callable, Any

from src.core.cache import cache

def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_builder: Optional[Callable] = None
):
    """
    Decorator for caching function results

    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache keys
        key_builder: Custom key builder function

    Example:
        @cached(ttl=3600, key_prefix="search")
        async def search_google(query: str):
            # Expensive operation
            return results
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key building
                key_parts = [key_prefix] if key_prefix else []
                key_parts.append(func.__name__)
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache.set(cache_key, result, ttl=ttl)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions (convert to async)
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_result(ttl: int = 3600):
    """
    Simple cache decorator with default TTL

    Example:
        @cache_result(ttl=600)
        async def expensive_operation():
            return data
    """
    return cached(ttl=ttl)


def invalidate_cache(pattern: str):
    """
    Decorator to invalidate cache after function execution

    Args:
        pattern: Cache key pattern to invalidate

    Example:
        @invalidate_cache("search:*")
        async def update_search_index():
            # Update operation
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            await cache.delete_pattern(pattern)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            import asyncio
            asyncio.run(cache.delete_pattern(pattern))
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
```

### Step 3: Specialized Cache Types (`src/core/cache_types.py`)

```python
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from src.core.cache import cache

class SearchCache:
    """Cache for search results"""

    @staticmethod
    async def get_search_results(
        query: str,
        source: str
    ) -> Optional[List[Dict]]:
        """Get cached search results"""
        key = cache.make_key("search", query=query, source=source)
        return await cache.get(key)

    @staticmethod
    async def set_search_results(
        query: str,
        source: str,
        results: List[Dict],
        ttl: int = 3600
    ):
        """Cache search results"""
        key = cache.make_key("search", query=query, source=source)
        await cache.set(key, results, ttl=ttl)

    @staticmethod
    async def invalidate_search(query: str):
        """Invalidate all cached results for a query"""
        pattern = f"search:{query}:%"
        await cache.delete_pattern(pattern)


class LLMCache:
    """Cache for LLM responses"""

    @staticmethod
    async def get_analysis(
        text_hash: str,
        analysis_type: str
    ) -> Optional[Dict]:
        """Get cached LLM analysis"""
        key = cache.make_key("llm", hash=text_hash, type=analysis_type)
        return await cache.get(key)

    @staticmethod
    async def set_analysis(
        text_hash: str,
        analysis_type: str,
        result: Dict,
        ttl: int = 7200  # 2 hours default
    ):
        """Cache LLM analysis result"""
        key = cache.make_key("llm", hash=text_hash, type=analysis_type)
        await cache.set(key, result, ttl=ttl)


class RateLimitCache:
    """Cache for rate limiting"""

    @staticmethod
    async def check_rate_limit(
        identifier: str,
        limit: int,
        window: int = 60
    ) -> bool:
        """
        Check if rate limit exceeded

        Args:
            identifier: User/IP identifier
            limit: Maximum requests
            window: Time window in seconds

        Returns:
            True if within limit, False if exceeded
        """
        key = f"rate_limit:{identifier}"

        # Get current count
        count = await cache.get(key, default=0, deserialize=False)
        count = int(count) if count else 0

        if count >= limit:
            return False

        # Increment counter
        await cache.set(key, count + 1, ttl=window, serialize=False)
        return True

    @staticmethod
    async def get_remaining(
        identifier: str,
        limit: int
    ) -> int:
        """Get remaining requests in current window"""
        key = f"rate_limit:{identifier}"
        count = await cache.get(key, default=0, deserialize=False)
        count = int(count) if count else 0
        return max(0, limit - count)


class SessionCache:
    """Cache for user sessions"""

    @staticmethod
    async def create_session(
        user_id: str,
        data: Dict,
        ttl: int = 3600
    ) -> str:
        """Create new session"""
        import uuid
        session_id = str(uuid.uuid4())
        key = f"session:{session_id}"

        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "data": data
        }

        await cache.set(key, session_data, ttl=ttl)
        return session_id

    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict]:
        """Get session data"""
        key = f"session:{session_id}"
        return await cache.get(key)

    @staticmethod
    async def update_session(
        session_id: str,
        data: Dict,
        extend_ttl: bool = True
    ):
        """Update session data"""
        key = f"session:{session_id}"
        session = await cache.get(key)

        if session:
            session["data"].update(data)
            session["updated_at"] = datetime.utcnow().isoformat()

            ttl = 3600 if extend_ttl else await cache.get_ttl(key)
            await cache.set(key, session, ttl=ttl)

    @staticmethod
    async def delete_session(session_id: str):
        """Delete session"""
        key = f"session:{session_id}"
        await cache.delete(key)
```

---

## Testing Guide

### Unit Tests (`tests/test_cache.py`)

```python
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.core.cache import CacheManager, cache
from src.core.cache_decorators import cached, cache_result
from src.core.cache_types import SearchCache, LLMCache, RateLimitCache

@pytest.fixture
async def test_cache():
    """Create test cache instance"""
    test_cache = CacheManager(enable_memory_cache=True)
    await test_cache.initialize()
    yield test_cache
    await test_cache.close()

@pytest.mark.asyncio
async def test_cache_basic_operations(test_cache):
    """Test basic cache operations"""
    # Set value
    assert await test_cache.set("test_key", {"value": "test"}, ttl=60)

    # Get value
    result = await test_cache.get("test_key")
    assert result == {"value": "test"}

    # Delete value
    assert await test_cache.delete("test_key")

    # Get deleted value
    result = await test_cache.get("test_key")
    assert result is None

@pytest.mark.asyncio
async def test_cache_expiration(test_cache):
    """Test cache TTL expiration"""
    # Set with short TTL
    await test_cache.set("expire_key", "value", ttl=1)

    # Value should exist
    assert await test_cache.exists("expire_key")

    # Wait for expiration
    await asyncio.sleep(2)

    # Value should be expired
    assert not await test_cache.exists("expire_key")
    assert await test_cache.get("expire_key") is None

@pytest.mark.asyncio
async def test_cache_pattern_delete(test_cache):
    """Test pattern-based deletion"""
    # Set multiple keys
    await test_cache.set("user:1:profile", "profile1")
    await test_cache.set("user:1:settings", "settings1")
    await test_cache.set("user:2:profile", "profile2")
    await test_cache.set("product:1", "product1")

    # Delete user:1:* pattern
    deleted = await test_cache.delete_pattern("user:1:%")
    assert deleted == 2

    # Check remaining keys
    assert await test_cache.get("user:2:profile") == "profile2"
    assert await test_cache.get("product:1") == "product1"
    assert await test_cache.get("user:1:profile") is None

@pytest.mark.asyncio
async def test_cache_statistics(test_cache):
    """Test cache statistics tracking"""
    # Reset stats
    test_cache.stats = {
        "hits": 0, "misses": 0, "sets": 0,
        "deletes": 0, "memory_hits": 0, "db_hits": 0
    }

    # Generate some activity
    await test_cache.set("key1", "value1")
    await test_cache.get("key1")  # Hit
    await test_cache.get("key2")  # Miss
    await test_cache.delete("key1")

    stats = await test_cache.get_stats()
    assert stats["sets"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["deletes"] == 1
    assert "50.00%" in stats["hit_rate"]

@pytest.mark.asyncio
async def test_cache_decorator():
    """Test cache decorator"""
    call_count = 0

    @cached(ttl=60, key_prefix="test")
    async def expensive_function(param: str):
        nonlocal call_count
        call_count += 1
        return f"result_{param}"

    # First call - should execute function
    result1 = await expensive_function("value")
    assert result1 == "result_value"
    assert call_count == 1

    # Second call - should use cache
    result2 = await expensive_function("value")
    assert result2 == "result_value"
    assert call_count == 1  # Function not called again

    # Different parameter - should execute function
    result3 = await expensive_function("other")
    assert result3 == "result_other"
    assert call_count == 2

@pytest.mark.asyncio
async def test_search_cache():
    """Test search-specific cache"""
    results = [{"title": "Result 1"}, {"title": "Result 2"}]

    # Cache search results
    await SearchCache.set_search_results(
        "python tutorial",
        "google",
        results
    )

    # Retrieve cached results
    cached = await SearchCache.get_search_results(
        "python tutorial",
        "google"
    )
    assert cached == results

    # Different source should be separate
    assert await SearchCache.get_search_results(
        "python tutorial",
        "bing"
    ) is None

@pytest.mark.asyncio
async def test_rate_limit_cache():
    """Test rate limiting cache"""
    identifier = "user_123"
    limit = 3

    # Should allow first requests
    for i in range(limit):
        assert await RateLimitCache.check_rate_limit(
            identifier, limit, window=60
        )

    # Should block after limit
    assert not await RateLimitCache.check_rate_limit(
        identifier, limit, window=60
    )

    # Check remaining
    remaining = await RateLimitCache.get_remaining(identifier, limit)
    assert remaining == 0

@pytest.mark.asyncio
async def test_memory_cache_layer(test_cache):
    """Test memory cache layer performance"""
    # Set value (goes to both memory and DB)
    await test_cache.set("mem_test", "value")

    # Reset DB hit counter
    test_cache.stats["db_hits"] = 0
    test_cache.stats["memory_hits"] = 0

    # Get should hit memory cache
    result = await test_cache.get("mem_test")
    assert result == "value"
    assert test_cache.stats["memory_hits"] == 1
    assert test_cache.stats["db_hits"] == 0

@pytest.mark.asyncio
async def test_cache_cleanup():
    """Test expired entry cleanup"""
    test_cache = CacheManager()
    await test_cache.initialize()

    # Add entries with different TTLs
    await test_cache.set("keep", "value", ttl=3600)
    await test_cache.set("expire", "value", ttl=1)

    # Wait for expiration
    await asyncio.sleep(2)

    # Run cleanup
    cleaned = await test_cache.cleanup_expired()
    assert cleaned == 1

    # Check remaining
    assert await test_cache.get("keep") == "value"
    assert await test_cache.get("expire") is None

    await test_cache.close()

@pytest.mark.asyncio
async def test_cache_key_generation():
    """Test cache key generation"""
    # Simple key
    key1 = CacheManager.make_key("prefix", "arg1", param="value")
    assert key1 == "prefix:arg1:param=value"

    # Long key should be hashed
    long_args = ["x" * 50 for _ in range(10)]
    key2 = CacheManager.make_key(*long_args)
    assert len(key2) == 32  # MD5 hash length
```

---

## Integration Points

### With Database Module
```python
from src.core.database import db_manager
from src.core.models import Cache as CacheModel

# Cache uses database for persistence
async with db_manager.get_session() as session:
    # Cache operations use database session
    pass
```

### With Config Module
```python
from src.core.config import settings

# Use configuration for defaults
ttl = settings.CACHE_TTL_SECONDS
cleanup_interval = settings.CACHE_CLEANUP_INTERVAL
```

### With Scraper Module
```python
from src.core.cache_decorators import cached

@cached(ttl=3600, key_prefix="scrape")
async def scrape_google(query: str):
    # Expensive scraping operation
    return results
```

### With API Module
```python
from src.core.cache_types import RateLimitCache

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    identifier = request.client.host

    if not await RateLimitCache.check_rate_limit(
        identifier,
        settings.API_RATE_LIMIT_REQUESTS
    ):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )

    return await call_next(request)
```

---

## Common Issues & Solutions

### Issue 1: Cache Inconsistency
**Problem:** Memory cache out of sync with database
**Solution:** Clear memory cache on updates
```python
# Clear both layers
if self.memory_cache:
    self.memory_cache.clear()
await self.clear()  # Database cache
```

### Issue 2: Cache Key Collisions
**Problem:** Different data with same key
**Solution:** Use namespaced keys
```python
key = cache.make_key("namespace", "type", unique_id=id)
```

### Issue 3: Large Cached Values
**Problem:** SQLite TEXT column limit
**Solution:** Use compression or external storage
```python
import zlib
compressed = zlib.compress(json.dumps(large_data).encode())
await cache.set(key, compressed, serialize=False)
```

### Issue 4: Cleanup Performance
**Problem:** Cleanup blocking operations
**Solution:** Use index on expires_at
```sql
CREATE INDEX idx_cache_expires ON cache(expires_at);
```

### Issue 5: Memory Cache Size
**Problem:** Memory cache growing too large
**Solution:** Adjust TTLCache parameters
```python
self.memory_cache = TTLCache(
    maxsize=500,  # Reduce size
    ttl=30  # Shorter TTL
)
```

---

## Performance Optimization

### 1. Batch Operations
```python
async def get_many(keys: List[str]) -> Dict[str, Any]:
    """Get multiple cache entries efficiently"""
    results = {}

    # Check memory cache first
    memory_hits = []
    db_keys = []

    for key in keys:
        if key in cache.memory_cache:
            results[key] = cache.memory_cache[key]
            memory_hits.append(key)
        else:
            db_keys.append(key)

    # Batch database query
    if db_keys:
        async with db_manager.get_session() as session:
            entries = await session.execute(
                select(CacheModel).where(
                    CacheModel.key.in_(db_keys)
                )
            )
            for entry in entries:
                results[entry.key] = entry.value

    return results
```

### 2. Compression for Large Values
```python
import zlib

async def set_compressed(key: str, value: Any, ttl: int = 3600):
    """Set large values with compression"""
    json_str = json.dumps(value)

    if len(json_str) > 1000:  # Compress if > 1KB
        compressed = zlib.compress(json_str.encode())
        await cache.set(f"compressed:{key}", compressed, ttl, serialize=False)
    else:
        await cache.set(key, value, ttl)
```

### 3. Warm Cache on Startup
```python
async def warm_cache():
    """Pre-load frequently accessed data"""
    # Load common search queries
    common_queries = ["AI", "machine learning", "python"]

    for query in common_queries:
        # Trigger cache population
        await search_service.search(query)
```

---

## Module Checklist

- [ ] Cache manager implemented
- [ ] Get/Set/Delete operations working
- [ ] TTL expiration tested
- [ ] Pattern deletion working
- [ ] Memory cache layer functional
- [ ] Cache decorators created
- [ ] Specialized cache types implemented
- [ ] Statistics tracking
- [ ] Cleanup task running
- [ ] Unit tests passing
- [ ] Integration documented

---

## Next Steps
After completing this module:
1. **Module 4: Task Queue** - Uses cache for job state
2. **Module 5: Scraper** - Cache search results
3. **Module 7: Analyzer** - Cache LLM responses