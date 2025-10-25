"""
FastAPI Main Application for CIAP
RESTful API endpoints for competitive intelligence platform
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List
import asyncio
import json
import time
import logging

from ..config.settings import settings
from ..database import db_manager
from ..task_queue.manager import task_queue
from ..task_queue.handlers import register_default_handlers
from ..cache.manager import cache
from ..cache.types import RateLimitCache

logger = logging.getLogger(__name__)


# ============================================================
# WebSocket Connection Manager
# ============================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast WebSocket message: {e}")


# Create global connection manager
ws_manager = ConnectionManager()


# ============================================================
# Application Lifespan Management
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle

    Startup:
        - Initialize database
        - Start task queue with handlers
        - Initialize cache

    Shutdown:
        - Stop task queue
        - Close database connections
        - Close cache
    """
    # Startup
    logger.info("Starting CIAP API...")

    try:
        # Initialize database
        logger.info("Initializing database...")
        await db_manager.initialize()
        logger.info("Database initialized successfully")

        # Register task handlers
        logger.info("Registering task handlers...")
        register_default_handlers()
        logger.info("Task handlers registered")

        # Start task queue
        logger.info("Starting task queue...")
        await task_queue.start()
        logger.info(f"Task queue started with {task_queue.worker_count} workers")

        # Initialize cache
        logger.info("Initializing cache...")
        await cache.initialize()
        logger.info("Cache initialized successfully")

        logger.info("CIAP API started successfully")

    except Exception as e:
        logger.error(f"Failed to start CIAP API: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down CIAP API...")

    try:
        # Stop task queue
        logger.info("Stopping task queue...")
        await task_queue.stop()
        logger.info("Task queue stopped")

        # Close database
        logger.info("Closing database connections...")
        await db_manager.close()
        logger.info("Database closed")

        # Close cache
        logger.info("Closing cache...")
        await cache.close()
        logger.info("Cache closed")

        logger.info("CIAP API shut down successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="CIAP - Competitive Intelligence Automation Platform",
    description="Open-source competitive intelligence platform for SMEs with automated data collection and LLM analysis",
    version="0.8.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# ============================================================
# CORS Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Middleware: Rate Limiting
# ============================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware using RateLimitCache

    Limits requests per IP address based on settings.API_RATE_LIMIT_REQUESTS
    Window: 60 seconds (1 minute)
    """
    client_ip = request.client.host

    # Check rate limit
    allowed = await RateLimitCache.check_rate_limit(
        identifier=client_ip,
        limit=settings.API_RATE_LIMIT_REQUESTS,
        window=60
    )

    if not allowed:
        # Get remaining count (should be 0)
        remaining = await RateLimitCache.get_remaining(
            identifier=client_ip,
            limit=settings.API_RATE_LIMIT_REQUESTS
        )

        logger.warning(f"Rate limit exceeded for {client_ip}")

        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum {settings.API_RATE_LIMIT_REQUESTS} requests per minute",
                "retry_after": 60,
                "remaining": remaining
            }
        )

    response = await call_next(request)
    return response


# ============================================================
# Middleware: Request Logging
# ============================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Log all requests with timing information
    Adds X-Process-Time header to response
    """
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} - {process_time:.3f}s"
    )

    # Add timing header
    response.headers["X-Process-Time"] = f"{process_time:.3f}"

    return response


# ============================================================
# Global Exception Handler
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle unhandled exceptions globally

    In development: Return full error message
    In production: Return generic error message
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Determine error details based on environment
    error_detail = None
    if settings.ENVIRONMENT.value == "development":
        error_detail = str(exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": error_detail,
            "path": request.url.path
        }
    )


# ============================================================
# Root Endpoints
# ============================================================

@app.get("/")
async def root():
    """
    Root endpoint - API information

    Returns:
        API metadata and links to documentation
    """
    return {
        "name": "CIAP API",
        "version": "0.8.0",
        "description": "Competitive Intelligence Automation Platform",
        "status": "running",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "openapi": "/api/openapi.json"
    }


@app.get("/health")
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
    # Check database
    db_healthy = await db_manager.health_check()

    # Check Ollama
    ollama_healthy = False
    try:
        from ..analyzers.ollama_client import ollama_client
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


# ============================================================
# WebSocket Endpoint
# ============================================================

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates

    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier

    Message format:
        Incoming:
            {
                "action": "subscribe",
                "search_id": 123
            }

        Outgoing:
            {
                "type": "status_update",
                "search_id": 123,
                "status": "completed",
                "timestamp": 1234567890.123
            }
    """
    await ws_manager.connect(websocket)

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            action = message.get("action")

            # Handle subscription to search updates
            if action == "subscribe":
                search_id = message.get("search_id")

                if not search_id:
                    await ws_manager.send_personal_message(
                        {"error": "search_id required for subscribe action"},
                        websocket
                    )
                    continue

                logger.info(f"Client {client_id} subscribed to search {search_id}")

                # Send initial status
                from ..database.operations import DatabaseOperations
                async with db_manager.get_session() as session:
                    search = await DatabaseOperations.get_search(session, search_id)

                    if not search:
                        await ws_manager.send_personal_message(
                            {"error": f"Search {search_id} not found"},
                            websocket
                        )
                        continue

                    # Send status updates every 2 seconds until completed
                    while True:
                        async with db_manager.get_session() as session:
                            search = await DatabaseOperations.get_search(session, search_id)

                            await ws_manager.send_personal_message(
                                {
                                    "type": "status_update",
                                    "search_id": search_id,
                                    "status": search.status,
                                    "timestamp": time.time()
                                },
                                websocket
                            )

                            # Stop polling if search is completed or failed
                            if search.status in ["completed", "failed"]:
                                logger.info(f"Search {search_id} finished with status: {search.status}")
                                break

                        await asyncio.sleep(2)

            # Handle ping (keep-alive)
            elif action == "ping":
                await ws_manager.send_personal_message(
                    {"type": "pong", "timestamp": time.time()},
                    websocket
                )

            else:
                await ws_manager.send_personal_message(
                    {"error": f"Unknown action: {action}"},
                    websocket
                )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        ws_manager.disconnect(websocket)


# ============================================================
# Router Registration
# ============================================================

# Import routers
from .routes import search, tasks, analysis, export

# Register routers
app.include_router(
    search.router,
    prefix=f"{settings.API_PREFIX}/search",
    tags=["search"]
)

app.include_router(
    tasks.router,
    prefix=f"{settings.API_PREFIX}/tasks",
    tags=["tasks"]
)

app.include_router(
    analysis.router,
    prefix=f"{settings.API_PREFIX}/analysis",
    tags=["analysis"]
)

app.include_router(
    export.router,
    prefix=f"{settings.API_PREFIX}/export",
    tags=["export"]
)
