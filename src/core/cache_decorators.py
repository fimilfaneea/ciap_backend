"""
Cache Decorators for CIAP
Provides convenient decorators for function-level caching
"""

import asyncio
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
        ttl: Cache TTL in seconds (uses default if None)
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

    Args:
        ttl: Time-to-live in seconds (default: 3600)

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
        pattern: Cache key pattern to invalidate (SQL LIKE syntax)

    Example:
        @invalidate_cache("search:*")
        async def update_search_index():
            # Update operation
            pass

    Note:
        SQL LIKE patterns: % = any characters, _ = single character
        Use "search:%" to invalidate all keys starting with "search:"
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            # Convert * to % for SQL LIKE syntax
            sql_pattern = pattern.replace("*", "%")
            await cache.delete_pattern(sql_pattern)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Convert * to % for SQL LIKE syntax
            sql_pattern = pattern.replace("*", "%")
            asyncio.run(cache.delete_pattern(sql_pattern))
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def conditional_cache(
    condition: Callable[..., bool],
    ttl: Optional[int] = None,
    key_prefix: str = ""
):
    """
    Cache decorator that only caches when condition is met

    Args:
        condition: Function that takes same args as decorated function
                  and returns True if result should be cached
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache keys

    Example:
        @conditional_cache(
            condition=lambda query: len(query) > 3,
            ttl=1800,
            key_prefix="search"
        )
        async def search(query: str):
            # Only cache if query length > 3
            return results
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check condition
            should_cache = condition(*args, **kwargs)

            if not should_cache:
                # Don't use cache, just execute
                return await func(*args, **kwargs)

            # Use caching
            key_parts = [key_prefix] if key_prefix else []
            key_parts.append(func.__name__)
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute and cache
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_aside(
    key_func: Callable[..., str],
    ttl: Optional[int] = None,
    update_on_miss: bool = True
):
    """
    Cache-aside pattern decorator with custom key function

    Args:
        key_func: Function to generate cache key from args
        ttl: Cache TTL in seconds
        update_on_miss: Whether to cache result on cache miss

    Example:
        @cache_aside(
            key_func=lambda user_id: f"user:{user_id}:profile",
            ttl=3600
        )
        async def get_user_profile(user_id: int):
            # Database query
            return profile
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_func(*args, **kwargs)

            # Try cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Update cache if enabled
            if update_on_miss and result is not None:
                await cache.set(cache_key, result, ttl=ttl)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
