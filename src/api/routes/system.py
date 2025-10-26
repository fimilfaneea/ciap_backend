"""
System Routes for CIAP API
Health checks, statistics, and system information endpoints
"""

from fastapi import APIRouter
from typing import Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint

    Checks status of:
    - Database connection
    - Ollama LLM service
    - Task queue
    - Cache

    Returns:
        Health status of all components
    """
    from ...database import db_manager
    from ...task_queue.manager import task_queue
    from ...cache.manager import cache

    # Check database
    db_healthy = await db_manager.health_check()

    # Check Ollama
    ollama_healthy = False
    try:
        from ...analyzers.ollama_client import ollama_client
        ollama_healthy = await ollama_client.check_health()
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")

    # Check task queue
    queue_healthy = task_queue.running

    # Check cache
    cache_healthy = True
    try:
        # Test cache operation
        await cache.set("health_check", "ok", ttl=60)
        test_value = await cache.get("health_check")
        cache_healthy = (test_value == "ok")
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        cache_healthy = False

    # Aggregate checks
    checks = {
        "database": db_healthy,
        "ollama": ollama_healthy,
        "task_queue": queue_healthy,
        "cache": cache_healthy
    }

    # Overall health (all components must be healthy)
    overall_healthy = all(checks.values())

    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "checks": checks,
        "timestamp": time.time()
    }


@router.get("/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """
    Get system-wide statistics

    Returns statistics for:
    - Task queue (pending, processing, completed, failed tasks)
    - Cache (hits, misses, size)
    - Database (table counts, size)

    Returns:
        Dictionary with statistics from all subsystems
    """
    from ...database import db_manager
    from ...task_queue.manager import task_queue
    from ...cache.manager import cache

    stats = {}

    # Task queue statistics
    try:
        queue_stats = await task_queue.get_queue_stats()
        stats["task_queue"] = queue_stats
    except Exception as e:
        logger.error(f"Failed to get task queue stats: {e}")
        stats["task_queue"] = {"error": str(e)}

    # Cache statistics
    try:
        cache_stats = await cache.get_stats()
        stats["cache"] = cache_stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        stats["cache"] = {"error": str(e)}

    # Database statistics
    try:
        db_stats = await db_manager.get_stats()
        stats["database"] = db_stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        stats["database"] = {"error": str(e)}

    # Add timestamp
    stats["timestamp"] = time.time()

    return stats


@router.get("/version")
async def get_version():
    """
    Get API version and metadata

    Returns:
        API version, environment, and configuration info
    """
    from ...config.settings import settings

    return {
        "name": "CIAP API",
        "version": "1.0.0",
        "description": "Competitive Intelligence Automation Platform",
        "environment": settings.ENVIRONMENT.value,
        "api_prefix": settings.API_PREFIX,
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "openapi": "/api/openapi.json"
    }
