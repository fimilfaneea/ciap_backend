"""
Comprehensive tests for Cache System
Tests CacheManager, decorators, and specialized cache types
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.cache import CacheManager, cache, warm_cache
from src.cache import cached, cache_result, invalidate_cache
from src.cache import SearchCache, LLMCache, RateLimitCache, SessionCache


@pytest.fixture
async def test_cache():
    """Create test cache instance"""
    test_cache = CacheManager(enable_memory_cache=True)
    await test_cache.initialize()
    yield test_cache
    await test_cache.close()


@pytest.mark.asyncio
async def test_cache_basic_operations(test_cache):
    """Test basic cache operations: get/set/delete/exists"""
    # Set value
    assert await test_cache.set("test_key", {"value": "test"}, ttl=60)

    # Get value
    result = await test_cache.get("test_key")
    assert result == {"value": "test"}

    # Exists check
    assert await test_cache.exists("test_key")

    # Delete value
    assert await test_cache.delete("test_key")

    # Get deleted value
    result = await test_cache.get("test_key")
    assert result is None

    # Exists check after delete
    assert not await test_cache.exists("test_key")


@pytest.mark.asyncio
async def test_cache_expiration(test_cache):
    """Test cache TTL expiration"""
    # Set with short TTL
    await test_cache.set("expire_key", "value", ttl=1)

    # Value should exist
    assert await test_cache.exists("expire_key")

    # Check TTL
    ttl = await test_cache.get_ttl("expire_key")
    assert ttl is not None
    assert 0 <= ttl <= 1

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
    assert await test_cache.get("user:1:settings") is None


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
    """Test @cached decorator"""
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
    """Test SearchCache operations"""
    results = [{"title": "Result 1"}, {"title": "Result 2"}]

    # Cache search results
    await SearchCache.set_search_results(
        "python tutorial",
        "google",
        results,
        ttl=60
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

    # Test invalidation
    deleted = await SearchCache.invalidate_search("python tutorial")
    assert deleted == 1

    # Should be gone now
    assert await SearchCache.get_search_results(
        "python tutorial",
        "google"
    ) is None


@pytest.mark.asyncio
async def test_llm_cache():
    """Test LLMCache operations"""
    import hashlib

    text = "This is test text for sentiment analysis"
    text_hash = hashlib.md5(text.encode()).hexdigest()
    analysis_result = {
        "sentiment": "positive",
        "confidence": 0.95
    }

    # Cache analysis
    await LLMCache.set_analysis(
        text_hash,
        "sentiment",
        analysis_result,
        ttl=60
    )

    # Retrieve cached analysis
    cached = await LLMCache.get_analysis(text_hash, "sentiment")
    assert cached == analysis_result

    # Different analysis type should be separate
    assert await LLMCache.get_analysis(text_hash, "summary") is None


@pytest.mark.asyncio
async def test_rate_limit_cache():
    """Test rate limiting logic"""
    identifier = "user_123"
    limit = 3

    # Should allow first requests
    for i in range(limit):
        allowed = await RateLimitCache.check_rate_limit(
            identifier, limit, window=60
        )
        assert allowed is True

    # Should block after limit
    blocked = await RateLimitCache.check_rate_limit(
        identifier, limit, window=60
    )
    assert blocked is False

    # Check remaining
    remaining = await RateLimitCache.get_remaining(identifier, limit)
    assert remaining == 0

    # Reset rate limit
    reset = await RateLimitCache.reset_rate_limit(identifier)
    assert reset is True

    # Should allow again after reset
    allowed = await RateLimitCache.check_rate_limit(
        identifier, limit, window=60
    )
    assert allowed is True


@pytest.mark.asyncio
async def test_session_cache():
    """Test SessionCache CRUD operations"""
    user_id = "user_456"
    session_data = {"username": "testuser", "role": "admin"}

    # Create session
    session_id = await SessionCache.create_session(
        user_id,
        session_data,
        ttl=60
    )
    assert session_id is not None
    assert len(session_id) == 36  # UUID format

    # Get session
    retrieved = await SessionCache.get_session(session_id)
    assert retrieved is not None
    assert retrieved["user_id"] == user_id
    assert retrieved["data"] == session_data

    # Update session
    updated = await SessionCache.update_session(
        session_id,
        {"last_login": "2024-01-01"},
        extend_ttl=True
    )
    assert updated is True

    # Verify update
    retrieved = await SessionCache.get_session(session_id)
    assert "last_login" in retrieved["data"]

    # Session exists check
    exists = await SessionCache.session_exists(session_id)
    assert exists is True

    # Delete session
    deleted = await SessionCache.delete_session(session_id)
    assert deleted is True

    # Should not exist after delete
    exists = await SessionCache.session_exists(session_id)
    assert exists is False


@pytest.mark.asyncio
async def test_memory_cache_layer(test_cache):
    """Test memory cache layer performance"""
    # Set value (goes to both memory and DB)
    await test_cache.set("mem_test", "value")

    # Reset counters
    test_cache.stats["db_hits"] = 0
    test_cache.stats["memory_hits"] = 0

    # Get should hit memory cache
    result = await test_cache.get("mem_test")
    assert result == "value"
    assert test_cache.stats["memory_hits"] == 1
    assert test_cache.stats["db_hits"] == 0

    # Clear memory cache only
    if test_cache.memory_cache:
        test_cache.memory_cache.clear()

    # Reset counters
    test_cache.stats["db_hits"] = 0
    test_cache.stats["memory_hits"] = 0

    # Get should hit database cache
    result = await test_cache.get("mem_test")
    assert result == "value"
    assert test_cache.stats["db_hits"] == 1
    assert test_cache.stats["memory_hits"] == 0


@pytest.mark.asyncio
async def test_cache_cleanup():
    """Test expired entry cleanup"""
    test_cache = CacheManager(enable_memory_cache=False)
    await test_cache.initialize()

    # Add entries with different TTLs
    await test_cache.set("keep", "value", ttl=3600)
    await test_cache.set("expire1", "value", ttl=1)
    await test_cache.set("expire2", "value", ttl=1)

    # Wait for expiration
    await asyncio.sleep(2)

    # Run cleanup
    cleaned = await test_cache.cleanup_expired()
    assert cleaned == 2

    # Check remaining
    assert await test_cache.get("keep") == "value"
    assert await test_cache.get("expire1") is None
    assert await test_cache.get("expire2") is None

    await test_cache.close()


@pytest.mark.asyncio
async def test_batch_operations(test_cache):
    """Test get_many batch retrieval"""
    # Set multiple values
    await test_cache.set("batch1", "value1")
    await test_cache.set("batch2", "value2")
    await test_cache.set("batch3", "value3")

    # Batch get
    results = await test_cache.get_many(["batch1", "batch2", "batch3", "nonexistent"])

    assert results["batch1"] == "value1"
    assert results["batch2"] == "value2"
    assert results["batch3"] == "value3"
    assert "nonexistent" not in results


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


@pytest.mark.asyncio
async def test_compressed_cache(test_cache):
    """Test compression for large values"""
    # Small value - should not compress
    small_data = {"key": "value"}
    await test_cache.set_compressed("small", small_data, ttl=60, threshold=1000)

    # Large value - should compress
    large_data = {"data": "x" * 5000}
    await test_cache.set_compressed("large", large_data, ttl=60, threshold=1000)

    # Retrieve small value normally
    small_result = await test_cache.get("small")
    assert small_result == small_data

    # Retrieve large value with decompression
    large_result = await test_cache.get_compressed("large")
    assert large_result == large_data


@pytest.mark.asyncio
async def test_warm_cache_utility():
    """Test cache warming utility"""
    operations = [
        {'key': 'config:setting1', 'value': 'value1', 'ttl': 60},
        {'key': 'config:setting2', 'value': 'value2', 'ttl': 60},
        {'key': 'config:setting3', 'value': 'value3', 'ttl': 60}
    ]

    # Warm cache
    count = await warm_cache(operations)
    assert count == 3

    # Verify cached values
    assert await cache.get('config:setting1') == 'value1'
    assert await cache.get('config:setting2') == 'value2'
    assert await cache.get('config:setting3') == 'value3'


@pytest.mark.asyncio
async def test_invalidate_cache_decorator():
    """Test @invalidate_cache decorator"""
    # Set some cached data
    await cache.set("invalidate:test1", "value1")
    await cache.set("invalidate:test2", "value2")
    await cache.set("other:test", "value3")

    # Verify cached
    assert await cache.exists("invalidate:test1")
    assert await cache.exists("invalidate:test2")

    # Function with invalidation decorator
    @invalidate_cache("invalidate:*")
    async def update_data():
        return "updated"

    result = await update_data()
    assert result == "updated"

    # Should be invalidated
    assert not await cache.exists("invalidate:test1")
    assert not await cache.exists("invalidate:test2")

    # Other keys should remain
    assert await cache.exists("other:test")


@pytest.mark.asyncio
async def test_clear_all_cache(test_cache):
    """Test clearing all cache entries"""
    # Add multiple entries
    await test_cache.set("clear1", "value1")
    await test_cache.set("clear2", "value2")
    await test_cache.set("clear3", "value3")

    # Clear all
    cleared = await test_cache.clear()
    assert cleared == 3

    # All should be gone
    assert await test_cache.get("clear1") is None
    assert await test_cache.get("clear2") is None
    assert await test_cache.get("clear3") is None
