# Module 8: FastAPI REST API

## Overview
**Purpose:** RESTful API endpoints for all CIAP functionality with FastAPI.

**Responsibilities:**
- Search endpoints (CRUD)
- Task management endpoints
- Analysis endpoints
- Export endpoints
- WebSocket for real-time updates
- Authentication & rate limiting
- API documentation (OpenAPI/Swagger)

**Development Time:** 2 days (Week 9, Day 32-36)

---

## Implementation Guide

### Main API Application (`src/api/main.py`)

```python
from fastapi import FastAPI, HTTPException, Depends, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import logging

from src.core.config import settings
from src.core.database import db_manager
from src.core.queue import task_queue
from src.core.task_handlers import register_default_handlers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting CIAP API...")

    # Initialize database
    await db_manager.initialize()

    # Start task queue
    register_default_handlers()
    await task_queue.start()

    # Initialize cache
    from src.core.cache import cache
    await cache.initialize()

    logger.info("CIAP API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down CIAP API...")

    # Stop task queue
    await task_queue.stop()

    # Close database
    await db_manager.close()

    # Close cache
    await cache.close()

    logger.info("CIAP API shut down")


# Create FastAPI app
app = FastAPI(
    title="CIAP - Competitive Intelligence Platform",
    description="Open-source CI platform for SMEs",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from src.api.routes import search, tasks, analysis, export, websocket

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

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "CIAP API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs"
    }

# Health check
@app.get("/health")
async def health_check():
    # Check components
    checks = {
        "database": await db_manager.health_check(),
        "ollama": False,
        "queue": task_queue.running
    }

    # Check Ollama
    from src.analyzers.ollama_client import ollama_client
    checks["ollama"] = await ollama_client.check_health()

    # Overall health
    healthy = all(checks.values())

    return {
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks
    }
```

### Search Routes (`src/api/routes/search.py`)

```python
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from src.core.database import get_db
from src.core.db_ops import DatabaseOperations
from src.scrapers.manager import scraper_manager

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    sources: List[str] = Field(default=["google", "bing"])
    max_results: int = Field(default=50, ge=10, le=200)
    analyze: bool = Field(default=True)


class SearchResponse(BaseModel):
    search_id: int
    query: str
    status: str
    created_at: datetime
    task_id: Optional[int] = None


@router.post("/", response_model=SearchResponse)
async def create_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create new search"""
    try:
        # Create search record
        search = await DatabaseOperations.create_search(
            db,
            query=request.query,
            sources=request.sources
        )
        await db.commit()

        # Schedule scraping
        task_id = await scraper_manager.schedule_scraping(
            query=request.query,
            sources=request.sources
        )

        return SearchResponse(
            search_id=search.id,
            query=search.query,
            status=search.status,
            created_at=search.created_at,
            task_id=task_id
        )

    except Exception as e:
        logger.error(f"Search creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{search_id}")
async def get_search(
    search_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get search details and results"""
    search = await DatabaseOperations.get_search(db, search_id)

    if not search:
        raise HTTPException(status_code=404, detail="Search not found")

    # Get results
    results = await DatabaseOperations.get_search_results(
        db, search_id
    )

    return {
        "search": {
            "id": search.id,
            "query": search.query,
            "status": search.status,
            "sources": search.sources,
            "created_at": search.created_at,
            "completed_at": search.completed_at
        },
        "results": [
            {
                "title": r.title,
                "snippet": r.snippet,
                "url": r.url,
                "source": r.source,
                "position": r.position
            }
            for r in results
        ]
    }


@router.get("/")
async def list_searches(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List all searches"""
    result = await db.execute(
        select(Search)
        .order_by(Search.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    searches = result.scalars().all()

    return {
        "searches": [
            {
                "id": s.id,
                "query": s.query,
                "status": s.status,
                "created_at": s.created_at
            }
            for s in searches
        ],
        "total": await db.scalar(select(func.count()).select_from(Search))
    }
```

### Analysis Routes (`src/api/routes/analysis.py`)

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.analyzers.sentiment import SentimentAnalyzer
from src.analyzers.ollama_client import ollama_client

router = APIRouter()


class AnalyzeRequest(BaseModel):
    text: str
    analysis_type: str = "summary"


@router.post("/text")
async def analyze_text(request: AnalyzeRequest):
    """Analyze arbitrary text"""
    try:
        result = await ollama_client.analyze(
            text=request.text,
            analysis_type=request.analysis_type
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/{search_id}")
async def analyze_sentiment(search_id: int):
    """Analyze sentiment for search results"""
    analyzer = SentimentAnalyzer()

    try:
        result = await analyzer.analyze_search_results(search_id)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitors/{search_id}")
async def analyze_competitors(search_id: int):
    """Analyze competitor mentions"""
    from src.analyzers.sentiment import CompetitorAnalyzer

    analyzer = CompetitorAnalyzer()

    try:
        result = await analyzer.analyze_competitors(search_id)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{search_id}")
async def analyze_trends(search_id: int):
    """Analyze trends in search results"""
    from src.analyzers.sentiment import TrendAnalyzer

    analyzer = TrendAnalyzer()

    try:
        result = await analyzer.analyze_trends(search_id)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### WebSocket Support (`src/api/routes/websocket.py`)

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)

    try:
        while True:
            # Wait for messages
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle subscription to search updates
            if message.get("action") == "subscribe":
                search_id = message.get("search_id")

                # Send updates
                while True:
                    status = await get_search_status(search_id)
                    await websocket.send_json({
                        "type": "status_update",
                        "search_id": search_id,
                        "status": status
                    })

                    if status in ["completed", "failed"]:
                        break

                    await asyncio.sleep(2)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Middleware (`src/api/middleware.py`)

```python
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time
import logging

from src.core.cache_types import RateLimitCache

logger = logging.getLogger(__name__)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = request.client.host

    # Check rate limit
    if not await RateLimitCache.check_rate_limit(
        client_ip,
        settings.API_RATE_LIMIT_REQUESTS,
        window=60
    ):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": 60
            }
        )

    response = await call_next(request)
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} - {process_time:.3f}s"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.ENVIRONMENT == "development" else None
        }
    )
```

---

## Testing

```python
from fastapi.testclient import TestClient

def test_create_search():
    client = TestClient(app)

    response = client.post(
        "/api/v1/search",
        json={
            "query": "artificial intelligence",
            "sources": ["google"]
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "search_id" in data


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
```

---

## Module Checklist

- [ ] FastAPI app configured
- [ ] Search endpoints working
- [ ] Task management endpoints
- [ ] Analysis endpoints functional
- [ ] WebSocket support added
- [ ] Rate limiting implemented
- [ ] CORS configured
- [ ] Error handling complete
- [ ] API documentation generated
- [ ] Unit tests passing

---

## Next Steps
- Module 9: Export - Export functionality
- Module 10: Scheduler - Job scheduling