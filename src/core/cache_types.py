"""
Specialized Cache Types for CIAP
Provides domain-specific caching utilities for common use cases
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.cache import cache


class SearchCache:
    """Cache for search results"""

    @staticmethod
    async def get_search_results(
        query: str,
        source: str
    ) -> Optional[List[Dict]]:
        """
        Get cached search results

        Args:
            query: Search query string
            source: Search source (e.g., 'google', 'bing')

        Returns:
            List of search results or None if not cached
        """
        key = cache.make_key("search", query=query, source=source)
        return await cache.get(key)

    @staticmethod
    async def set_search_results(
        query: str,
        source: str,
        results: List[Dict],
        ttl: int = 3600
    ):
        """
        Cache search results

        Args:
            query: Search query string
            source: Search source (e.g., 'google', 'bing')
            results: List of search result dictionaries
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        """
        key = cache.make_key("search", query=query, source=source)
        await cache.set(key, results, ttl=ttl)

    @staticmethod
    async def invalidate_search(query: str):
        """
        Invalidate all cached results for a query (all sources)

        Args:
            query: Search query string to invalidate

        Returns:
            Number of entries deleted
        """
        # Use SQL LIKE pattern to match all sources for this query
        # Key format is: search:query={query}:source={source}
        pattern = f"search:query={query}:%"
        return await cache.delete_pattern(pattern)

    @staticmethod
    async def get_or_search(
        query: str,
        source: str,
        search_func: callable,
        ttl: int = 3600
    ) -> List[Dict]:
        """
        Get cached results or execute search function

        Args:
            query: Search query string
            source: Search source
            search_func: Async function to execute if cache miss
            ttl: Cache TTL in seconds

        Returns:
            List of search results
        """
        # Try cache first
        results = await SearchCache.get_search_results(query, source)

        if results is not None:
            return results

        # Cache miss - execute search
        results = await search_func(query)

        # Cache the results
        await SearchCache.set_search_results(query, source, results, ttl)

        return results


class LLMCache:
    """Cache for LLM analysis responses"""

    @staticmethod
    async def get_analysis(
        text_hash: str,
        analysis_type: str
    ) -> Optional[Dict]:
        """
        Get cached LLM analysis

        Args:
            text_hash: Hash of the text being analyzed (e.g., MD5)
            analysis_type: Type of analysis (e.g., 'sentiment', 'competitor', 'summary')

        Returns:
            Analysis result dictionary or None if not cached
        """
        key = cache.make_key("llm", hash=text_hash, type=analysis_type)
        return await cache.get(key)

    @staticmethod
    async def set_analysis(
        text_hash: str,
        analysis_type: str,
        result: Dict,
        ttl: int = 7200  # 2 hours default
    ):
        """
        Cache LLM analysis result

        Args:
            text_hash: Hash of the text being analyzed
            analysis_type: Type of analysis
            result: Analysis result dictionary
            ttl: Time-to-live in seconds (default: 7200 = 2 hours)
        """
        key = cache.make_key("llm", hash=text_hash, type=analysis_type)
        await cache.set(key, result, ttl=ttl)

    @staticmethod
    async def get_or_analyze(
        text: str,
        analysis_type: str,
        analysis_func: callable,
        ttl: int = 7200
    ) -> Dict:
        """
        Get cached analysis or execute LLM analysis

        Args:
            text: Text to analyze
            analysis_type: Type of analysis
            analysis_func: Async function to execute if cache miss
            ttl: Cache TTL in seconds

        Returns:
            Analysis result dictionary
        """
        import hashlib

        # Generate text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Try cache first
        result = await LLMCache.get_analysis(text_hash, analysis_type)

        if result is not None:
            return result

        # Cache miss - execute analysis
        result = await analysis_func(text)

        # Cache the result
        await LLMCache.set_analysis(text_hash, analysis_type, result, ttl)

        return result

    @staticmethod
    async def invalidate_analysis_type(analysis_type: str) -> int:
        """
        Invalidate all cached analyses of a specific type

        Args:
            analysis_type: Type of analysis to invalidate

        Returns:
            Number of entries deleted
        """
        pattern = f"llm:%:type={analysis_type}"
        return await cache.delete_pattern(pattern)


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
            identifier: User/IP/API key identifier
            limit: Maximum requests allowed
            window: Time window in seconds (default: 60)

        Returns:
            True if within limit (request allowed), False if exceeded
        """
        key = f"rate_limit:{identifier}"

        # Get current count
        count = await cache.get(key, default=0, deserialize=False)
        count = int(count) if count else 0

        if count >= limit:
            return False

        # Increment counter
        await cache.set(key, str(count + 1), ttl=window, serialize=False)
        return True

    @staticmethod
    async def get_remaining(
        identifier: str,
        limit: int
    ) -> int:
        """
        Get remaining requests in current window

        Args:
            identifier: User/IP/API key identifier
            limit: Maximum requests allowed

        Returns:
            Number of remaining requests
        """
        key = f"rate_limit:{identifier}"
        count = await cache.get(key, default=0, deserialize=False)
        count = int(count) if count else 0
        return max(0, limit - count)

    @staticmethod
    async def reset_rate_limit(identifier: str) -> bool:
        """
        Reset rate limit for an identifier

        Args:
            identifier: User/IP/API key identifier

        Returns:
            True if reset successful
        """
        key = f"rate_limit:{identifier}"
        return await cache.delete(key)

    @staticmethod
    async def get_ttl(identifier: str) -> Optional[int]:
        """
        Get time until rate limit window resets

        Args:
            identifier: User/IP/API key identifier

        Returns:
            Seconds until reset or None if no limit active
        """
        key = f"rate_limit:{identifier}"
        return await cache.get_ttl(key)

    @staticmethod
    async def increment(
        identifier: str,
        amount: int = 1,
        window: int = 60
    ) -> int:
        """
        Increment rate limit counter

        Args:
            identifier: User/IP/API key identifier
            amount: Amount to increment by (default: 1)
            window: Time window in seconds

        Returns:
            New count value
        """
        key = f"rate_limit:{identifier}"

        # Get current count
        count = await cache.get(key, default=0, deserialize=False)
        count = int(count) if count else 0

        # Increment
        new_count = count + amount

        # Update cache
        await cache.set(key, str(new_count), ttl=window, serialize=False)

        return new_count


class SessionCache:
    """Cache for user sessions"""

    @staticmethod
    async def create_session(
        user_id: str,
        data: Dict,
        ttl: int = 3600
    ) -> str:
        """
        Create new session

        Args:
            user_id: User identifier
            data: Session data dictionary
            ttl: Session TTL in seconds (default: 3600 = 1 hour)

        Returns:
            Session ID (UUID)
        """
        # Generate unique session ID
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
        """
        Get session data

        Args:
            session_id: Session ID

        Returns:
            Session data dictionary or None if not found
        """
        key = f"session:{session_id}"
        return await cache.get(key)

    @staticmethod
    async def update_session(
        session_id: str,
        data: Dict,
        extend_ttl: bool = True,
        ttl: int = 3600
    ) -> bool:
        """
        Update session data

        Args:
            session_id: Session ID
            data: New/updated data to merge into session
            extend_ttl: Whether to extend the session TTL (default: True)
            ttl: New TTL if extending (default: 3600)

        Returns:
            True if session updated, False if session not found
        """
        key = f"session:{session_id}"
        session = await cache.get(key)

        if session is None:
            return False

        # Update session data
        session["data"].update(data)
        session["updated_at"] = datetime.utcnow().isoformat()

        # Determine TTL
        if extend_ttl:
            new_ttl = ttl
        else:
            # Keep existing TTL
            new_ttl = await cache.get_ttl(key)
            if new_ttl is None:
                new_ttl = ttl  # Fallback if TTL can't be determined

        await cache.set(key, session, ttl=new_ttl)
        return True

    @staticmethod
    async def delete_session(session_id: str) -> bool:
        """
        Delete session

        Args:
            session_id: Session ID

        Returns:
            True if session deleted, False if not found
        """
        key = f"session:{session_id}"
        return await cache.delete(key)

    @staticmethod
    async def get_user_sessions(user_id: str) -> List[str]:
        """
        Get all session IDs for a user

        Note: This is inefficient for large numbers of sessions.
        Consider using a dedicated user-to-sessions mapping in production.

        Args:
            user_id: User identifier

        Returns:
            List of session IDs
        """
        # This would require scanning all sessions
        # For production use, maintain a separate user->sessions mapping
        # For now, return empty list as this requires database query
        return []

    @staticmethod
    async def extend_session(
        session_id: str,
        additional_seconds: int = 3600
    ) -> bool:
        """
        Extend session TTL

        Args:
            session_id: Session ID
            additional_seconds: Seconds to add to current TTL

        Returns:
            True if extended, False if session not found
        """
        key = f"session:{session_id}"
        session = await cache.get(key)

        if session is None:
            return False

        # Get current TTL
        current_ttl = await cache.get_ttl(key)
        if current_ttl is None:
            current_ttl = 0

        # Extend TTL
        new_ttl = current_ttl + additional_seconds
        await cache.set(key, session, ttl=new_ttl)

        return True

    @staticmethod
    async def session_exists(session_id: str) -> bool:
        """
        Check if session exists and is valid

        Args:
            session_id: Session ID

        Returns:
            True if session exists and not expired
        """
        key = f"session:{session_id}"
        return await cache.exists(key)

    @staticmethod
    async def get_session_ttl(session_id: str) -> Optional[int]:
        """
        Get remaining session TTL

        Args:
            session_id: Session ID

        Returns:
            Remaining seconds or None if session not found
        """
        key = f"session:{session_id}"
        return await cache.get_ttl(key)
