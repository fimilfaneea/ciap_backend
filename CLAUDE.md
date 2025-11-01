# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CIAP (Competitive Intelligence Automation Platform) is a Python-based competitive intelligence solution for SMEs that automates data collection and analysis using web scrapers and LLMs. The platform uses FastAPI, SQLite with async support (SQLAlchemy + aiosqlite), and integrates with OpenAI, Anthropic, and local LLM providers (Ollama).

**Core Value Proposition**: 70-90% cost reduction vs commercial CI tools while providing enterprise-level intelligence capabilities.

## Essential Commands

### Environment Setup
```bash
# Activate virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (ONE-TIME SETUP, required for Scrapy-Playwright)
python setup_playwright.py
# OR manually: playwright install chromium

# Create .env file from template
cp .env.example .env         # Then edit with actual API keys

# Optional: Install Ollama for local LLM support
# Download from https://ollama.ai and run:
ollama pull llama3.1:8b      # Or other models
```

### Database Operations
```bash
# Initialize database (creates tables, indexes, FTS5)
python -m src.database.manager

# Run database tests
pytest tests/test_database.py -v

# Run all tests
pytest tests/ -v
```

### API Server

**CRITICAL: Always use `run.py` or `run_dev.py` to start the server, NOT `uvicorn` directly.**

These startup scripts install the AsyncioSelectorReactor before FastAPI initializes, which is required for Scrapy-Playwright integration. Starting with `uvicorn` directly will cause reactor mismatch errors.

```bash
# Development mode (recommended for development)
python run_dev.py
# - Auto-reload on code changes
# - Debug logging enabled
# - Runs on 127.0.0.1:8000

# Production mode
python run.py
# - No auto-reload (better performance)
# - Info-level logging
# - Runs on 0.0.0.0:8000

# Access interactive API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Development Workflow
```bash
# Format code
black src/ tests/

# Run specific module tests
pytest tests/test_scrapers.py -v      # Web scraping tests
pytest tests/test_analyzers.py -v     # LLM analyzer tests
pytest tests/test_cache.py -v         # Cache system tests
pytest tests/test_task_queue.py -v    # Task queue tests
pytest tests/test_api.py -v           # API endpoint tests
pytest tests/test_export_service.py -v # Export system tests
pytest tests/test_scheduler.py -v     # Scheduler tests

# Run integration tests
pytest tests/test_module6_integration.py -v  # Processor integration
pytest tests/test_module7_integration.py -v  # Analyzer integration
pytest tests/test_module8_integration.py -v  # API integration

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## Architecture Overview

### Modular Monolith Design

CIAP follows a modular monolith architecture with 10 self-contained modules:

```
src/
├── database/       # Module 1: Database Layer
│   ├── models.py          # 23 SQLAlchemy ORM models
│   ├── manager.py         # DatabaseManager with async connections
│   ├── operations.py      # 73 CRUD operations
│   └── fts.py            # FTS5 full-text search setup
│
├── config/         # Module 2: Configuration
│   ├── settings.py        # 33 Pydantic v2 validated settings
│   └── utils.py          # Config utilities & validation
│
├── cache/          # Module 3: Cache System
│   ├── manager.py         # Two-tier cache (memory + SQLite)
│   ├── decorators.py      # @cached, @invalidate_cache decorators
│   └── types.py          # Specialized caches (Search, LLM, RateLimit)
│
├── task_queue/     # Module 4: Task Queue
│   ├── manager.py         # TaskQueueManager with workers
│   ├── handlers.py        # Task handlers (scrape, analyze, export)
│   └── utils.py          # TaskChain, TaskGroup, wait_for_task
│
├── scrapers/       # Module 5: Web Scraper (Scrapy Framework)
│   ├── base.py           # BaseScraper abstract class (legacy)
│   ├── google.py         # Google search scraper (legacy)
│   ├── bing.py           # Bing search scraper (legacy)
│   ├── manager.py        # ScraperManager orchestrator
│   ├── scrapy_runner.py  # Async Scrapy runner with crochet
│   ├── scrapy_settings.py # Scrapy configuration
│   ├── items.py          # Scrapy item definitions
│   ├── pipelines.py      # Scrapy pipelines (validation, cleaning, deduplication)
│   └── spiders/          # Scrapy spiders
│       ├── google_spider.py  # Google search spider
│       └── bing_spider.py    # Bing search spider
│
├── processors/     # Module 6: Data Processor
│   ├── cleaner.py        # Data cleaning & deduplication
│   └── batch_processor.py # Bulk processing with progress
│
├── analyzers/      # Module 7: LLM Analyzer
│   ├── ollama_client.py  # Ollama integration client
│   └── sentiment.py      # Sentiment analysis handler
│
├── api/            # Module 8: REST API
│   ├── main.py           # FastAPI app with lifespan, WebSocket
│   ├── schemas/          # Pydantic schemas for request/response
│   │   ├── common.py     # Shared schemas (pagination, errors)
│   │   └── __init__.py
│   └── routes/           # API endpoints (all prefixed with /api/v1)
│       ├── search.py     # Search operations (POST /search)
│       ├── tasks.py      # Task queue management
│       ├── analysis.py   # LLM analysis endpoints
│       ├── export.py     # Export to CSV/Excel/JSON
│       ├── scheduler.py  # Scheduled job management
│       └── system.py     # System health, stats, diagnostics
│
├── services/       # Module 9 & 10: Business Services
│   ├── export_service.py # Export to CSV/Excel/JSON
│   └── scheduler.py      # APScheduler integration
│
└── __init__.py     # Version 1.0.0
```

### Database-Centric Foundation

The project is built around a comprehensive SQLite database with **23 models** organized into 6 categories:

1. **Search Operations** (3 models): `Search`, `SearchResult`, `Analysis`
2. **Product Intelligence** (4 models): `Product`, `PriceData`, `Offer`, `ProductReview`
3. **Competitor Tracking** (3 models): `Competitor`, `CompetitorProducts`, `CompetitorTracking`
4. **Market Intelligence** (4 models): `MarketTrend`, `SERPData`, `SocialSentiment`, `NewsContent`
5. **Feature Analysis** (2 models): `FeatureComparison`, `Insights`
6. **Infrastructure** (5 models): `Cache`, `TaskQueue`, `ScrapingJob`, `RateLimit`, `PriceHistory`

### Key Architectural Patterns

**Async-First Design**: All database operations use `async/await` with `aiosqlite`
```python
async with db_manager.get_session() as session:
    search = await DatabaseOperations.create_search(session, query, sources)
```

**Session Management**: Context manager pattern for automatic commit/rollback
```python
@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    # Auto-commits on success, auto-rolls back on exception
```

**Bulk Operations with Chunking**: For large datasets to prevent memory issues
```python
await DatabaseOperations.bulk_insert_results_chunked(
    session, search_id, results, chunk_size=100, progress_callback=None
)
```

**Full-Text Search Integration**: SQLite FTS5 with Porter stemming for 4 tables:
- `products_fts`, `competitors_fts`, `news_content_fts`, `product_reviews_fts`
- Automatically synchronized via triggers on INSERT/UPDATE/DELETE

**Task Queue Pattern**: Priority-based queue with retry logic and concurrency control
```python
task = await DatabaseOperations.dequeue_task(session)  # Uses skip_locked=True
```

**Scrapy Integration**: Scrapy (Twisted) + FastAPI (asyncio) bridged via crochet
- **Reactor requirement**: AsyncioSelectorReactor must be installed before FastAPI starts
- **Startup scripts**: `run.py` and `run_dev.py` handle reactor installation
- **Never start directly**: Do NOT use `uvicorn src.api.main:app` - will fail with reactor errors
- **Scrapy runner**: `scrapy_runner.py` uses crochet to run spiders in Twisted thread
- **Playwright support**: Requires AsyncioSelectorReactor + WindowsSelectorEventLoopPolicy (Windows)

## Database Performance Features

### Optimizations Applied Automatically

1. **WAL Mode**: Write-Ahead Logging for concurrent reads during writes
2. **64MB Cache**: `PRAGMA cache_size=-64000`
3. **21 Indexes**: Foreign keys + frequently queried columns
4. **Memory Temp Store**: `PRAGMA temp_store=MEMORY`
5. **Connection Pre-ping**: Validates connections before use

### Important Index Strategy

Composite indexes for common query patterns:
- `idx_products_search` on `(product_name, brand_name, company_name)`
- `idx_price_latest` on `(product_id, scraped_at DESC)`
- `idx_task_queue_status` on `(status, priority)`

### Database Statistics

Access via:
```python
stats = await db_manager.get_stats()
# Returns: table counts, cache stats, task queue breakdown, DB size
```

## Configuration System

Centralized configuration management using Pydantic v2 Settings for type-safe, validated configuration loading from `.env` files.

**33 configuration settings** organized into 11 categories:
1. Environment, 2. Database, 3. Ollama LLM, 4. Web Scraping, 5. Google/Bing Scrapers, 6. Cache, 7. Task Queue, 8. API Server, 9. Security, 10. Logging, 11. Directories

**Usage**:
```python
from src.config.settings import settings

# Access configuration values
database_url = settings.get_database_url_async()
model = settings.OLLAMA_MODEL
api_port = settings.API_PORT
```

**Key Features**:
- Type validation and constraints via Pydantic v2
- Automatic directory creation (`data/`, `data/logs/`, `data/exports/`, `config/prompts/`)
- Secret filtering for exports
- Prompt template loading from `config/prompts/`

See `.env.example` for complete configuration template and `Backend_Modules/Mod_02_Config.md` for detailed documentation.

## Google Custom Search API Integration

**Recommended search method** - avoids CAPTCHA challenges and bot detection issues with web scraping.

### Why Use Google API Instead of Web Scraping?

**Web Scraping Issues**:
- CAPTCHA challenges from Google
- Bot detection and IP bans
- Requires proxies, CAPTCHA solvers, anti-detection tools
- Legal gray area
- Unreliable results

**Google Custom Search API Benefits**:
- Official Google service - no CAPTCHA
- 100 free queries per day
- Clean, structured JSON responses
- Reliable 99.9% uptime
- Legal and authorized
- $5 per 1000 queries beyond free tier

### Setup

1. **Get API Key**: https://console.cloud.google.com/apis/credentials
2. **Create Search Engine**: https://programmablesearchengine.google.com/
3. **Configure .env**:
```bash
GOOGLE_API_KEY=your_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
GOOGLE_API_ENABLED=true
```

See `docs/GOOGLE_API_SETUP.md` for detailed setup instructions.

### Usage

**Direct API usage**:
```python
from src.scrapers.google_api import search_google_api

results = await search_google_api(
    query="mango pickle",
    api_key=settings.GOOGLE_API_KEY,
    search_engine_id=settings.GOOGLE_SEARCH_ENGINE_ID,
    max_results=10,
    lang="en",
    region="us"
)
```

**Via ScraperManager** (automatically uses API when enabled):
```python
from src.scrapers.manager import ScraperManager

manager = ScraperManager()
results = await manager.scrape(
    query="mango pickle",
    sources=["google"],  # Uses API if GOOGLE_API_ENABLED=true
    max_results_per_source=10
)
```

**Response format**:
```python
{
    "title": "Page title",
    "url": "https://example.com/page",
    "snippet": "Brief description...",
    "position": 1,
    "source": "google_api",
    "scraped_at": "2025-01-15T10:30:00"
}
```

### Pricing & Quotas

- **Free tier**: 100 queries/day
- **Paid tier**: $5 per 1000 queries (max 10k/day)
- **Alternative**: Switch to SerpAPI or ScraperAPI for higher volume

## Cache System

SQLite-based caching system with optional in-memory layer. Supports TTL expiration, pattern-based deletion, and specialized cache types.

**Key Features**: Two-tier caching (memory + database), TTL expiration, statistics tracking, compression for large values, specialized cache types (Search, LLM, RateLimit, Session)

**Basic Usage**:
```python
from src.cache import cache

# Set/get with TTL
await cache.set("key", value, ttl=3600)
data = await cache.get("key")

# Pattern-based deletion
await cache.delete_pattern("user:123:%")

# Statistics
stats = await cache.get_stats()
```

**Cache Decorators**:
```python
from src.cache.decorators import cached

@cached(ttl=3600, key_prefix="search")
async def search_products(query: str):
    results = await db_search(query)
    return results

# First call executes, second call uses cache
```

**Specialized Cache Types** available: `SearchCache`, `LLMCache`, `RateLimitCache`, `SessionCache` - see `Backend_Modules/Mod_03_Cache.md` for detailed usage.

## Task Queue System

SQLite-based asynchronous task queue with priority scheduling, retry logic, and worker pool management.

**Key Features**: 5 priority levels, configurable workers, retry with exponential backoff, 6 task states (pending, processing, completed, failed, dead, cancelled), task chaining/grouping, workflows

**Basic Usage**:
```python
from src.task_queue import task_queue, TaskStatus, TaskPriority

# Register handler
async def my_handler(payload):
    return {"result": "success"}

task_queue.register_handler("my_task", my_handler)

# Start workers and enqueue task
await task_queue.start()
task_id = await task_queue.enqueue("my_task", {"data": "value"}, priority=TaskPriority.HIGH)

# Check status
status = await task_queue.get_task_status(task_id)
```

**Priority Levels**: `CRITICAL` (1), `HIGH` (3), `NORMAL` (5), `LOW` (7), `BACKGROUND` (10)

**Task States**: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`, `DEAD`, `CANCELLED`

**Advanced Features** available: `TaskChain` (sequential), `TaskGroup` (parallel), batch enqueue, recurring tasks, workflows - see `Backend_Modules/Mod_04_Queue.md`

## Critical Implementation Details

### DatabaseOperations Static Methods

**73 operations** organized into 19 categories. Key patterns:

**Upsert Pattern** (create_or_update):
```python
product = await DatabaseOperations.create_or_update_product(session, product_data)
# Finds by SKU or name+company, updates if exists, creates if not
```

**Streaming for Large Datasets**:
```python
async for chunk in DatabaseOperations.stream_products(session, chunk_size=50):
    process_batch(chunk)  # Memory-efficient processing
```

**Pagination with Full Metadata**:
```python
result = await DatabaseOperations.get_paginated_products(session, page=1, per_page=20)
# Returns: items, total, page, per_page, total_pages, has_prev, has_next
```

**Rate Limiting**:
```python
can_proceed = await DatabaseOperations.check_rate_limit(session, "google_scraper", requests_per_minute=10)
```

**Concurrency-Safe Task Dequeue**:
```python
# Uses with_for_update(skip_locked=True) to prevent race conditions
task = await DatabaseOperations.dequeue_task(session)
```

### FTS5 Query Syntax

Supports advanced searches:
```python
# Phrase search
results = await DatabaseOperations.search_products_fts(session, '"wireless headphones"')

# Boolean operators
results = await DatabaseOperations.search_products_fts(session, 'sony OR bose')

# Prefix search
results = await DatabaseOperations.search_products_fts(session, 'head*')

# Column-specific
results = await DatabaseOperations.search_products_fts(session, 'brand_name:sony')
```

### Scrapy Web Scraping

**Running Scrapy spiders from asyncio context**:

```python
from src.scrapers.scrapy_runner import run_google_spider, run_bing_spider

# Google search
results = await run_google_spider(
    query="AI news",
    max_results=100,
    lang="en",
    region="us",
    date_range="d",  # d=day, w=week, m=month, y=year
    timeout=180
)

# Bing search
results = await run_bing_spider(
    query="machine learning",
    max_results=50,
    lang="en",
    region="us",
    timeout=180
)
```

**Scrapy initialization** (happens automatically in FastAPI lifespan):
```python
from src.scrapers.scrapy_runner import initialize_scrapy, shutdown_scrapy

# Call once during app startup
initialize_scrapy()

# Call during app shutdown
shutdown_scrapy()
```

**Scrapy pipelines** (automatic processing):
1. `ValidationPipeline` (priority 100) - Validates required fields
2. `CleaningPipeline` (priority 200) - Cleans/normalizes text
3. `DeduplicationPipeline` (priority 300) - Removes duplicates by URL
4. `CollectorPipeline` (priority 900) - Collects results for return

**Scrapy settings** (`src/scrapers/scrapy_settings.py`):
- Playwright enabled: JavaScript rendering for dynamic pages
- Auto-throttle: Adaptive rate limiting based on response times
- Retry logic: 3 retries with exponential backoff
- Safety limits: 300s timeout, 500 items max, 100 pages max

## Testing Strategy

### Test Organization (11 test files)

**Module Tests**:
- `test_cache.py`: Cache system, TTL expiration, specialized cache types
- `test_task_queue.py`: Task queue, priority scheduling, retry logic
- `test_scrapers.py`: Scrapy spiders, scraping functionality
- `test_processors.py`: Data cleaning, deduplication, batch processing
- `test_analyzers.py`: LLM analyzer, Ollama integration
- `test_api.py`: API endpoints, request/response validation
- `test_export_service.py`: Export to CSV/Excel/JSON
- `test_scheduler.py`: APScheduler integration, recurring jobs

**Integration Tests**:
- `test_module6_integration.py`: Processor integration workflows
- `test_module7_integration.py`: Analyzer integration workflows
- `test_module8_integration.py`: API integration workflows

### Test Database Pattern

Always use in-memory database for tests:
```python
@pytest.fixture
async def test_db():
    manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await manager.initialize()
    yield manager
    await manager.close()
```

## Module Documentation

10 comprehensive module guides (`Mod_01_Database.md` through `Mod_10_Scheduler.md`) provide:
- Step-by-step implementation guides
- Interface specifications
- Integration points
- Common issues & solutions

The `src/core/Documentation.md` (2487 lines) is the complete reference for the database layer.

## Dependencies

**Core Stack**:
- FastAPI 0.104.1 + Uvicorn 0.24.0 (API framework)
- SQLAlchemy 2.0.23 + aiosqlite 0.19.0 (async ORM)
- Pydantic 2.5.0 (validation)
- python-dotenv 1.0.0 (config)

**Scraping**:
- Scrapy 2.11.0+ (primary scraping framework)
- scrapy-playwright 0.0.34+ (JavaScript rendering)
- playwright 1.40.0+ (browser automation)
- crochet 2.1.1+ (Twisted/asyncio bridge)
- requests 2.31.0, beautifulsoup4 4.12.2, lxml 4.9.3 (legacy, backward compatibility)

**LLM Integration**:
- Ollama (local LLMs via HTTP API)
- openai 1.3.5, anthropic 0.7.1 (optional cloud providers)

**Testing & Dev**:
- pytest 7.4.3, black 23.11.0

## Common Gotchas

### Reactor/Event Loop Issues

**CRITICAL: Server startup must use run.py or run_dev.py**

The most common issue is starting the server incorrectly:

```bash
# WRONG - Will fail with reactor errors
uvicorn src.api.main:app --reload

# CORRECT - Installs reactor before FastAPI
python run_dev.py  # Development
python run.py      # Production
```

**Why this matters**:
- Scrapy requires Twisted reactor (AsyncioSelectorReactor)
- FastAPI uses asyncio event loop
- Reactor must be installed BEFORE asyncio loop starts
- On Windows: Must use WindowsSelectorEventLoopPolicy, not ProactorEventLoop
- Once installed, reactor cannot be changed (process-wide singleton)

**Symptoms of wrong reactor**:
- `ReactorAlreadyInstalledError`
- `NotImplementedError` from Playwright
- "Event loop is closed" errors
- Scrapy spiders hang indefinitely

**Solution**: Always use startup scripts, never `uvicorn` directly.

### SQLite Limitations

**One writer at a time**: WAL mode allows concurrent reads but only one write operation. The task queue uses `skip_locked=True` to handle this gracefully.

**No ALTER TABLE for constraints**: Migrations require recreate-and-copy pattern. Use Alembic if needed (listed in dependencies).

### Session Handling

**Never reuse sessions across requests**: Each operation should get a fresh session via `get_session()`.

**Flush vs Commit**:
- `await session.flush()` - Makes ID available without committing
- Commit happens automatically via context manager

### Async Patterns

**Always await database operations**:
```python
# Correct
search = await DatabaseOperations.create_search(session, query, sources)

# Wrong - will not work
search = DatabaseOperations.create_search(session, query, sources)
```

**Context manager is async**:
```python
async with db_manager.get_session() as session:
    # Do work
```

## Performance Recommendations

### For Large Datasets

1. **Use chunked operations**: `bulk_insert_results_chunked()` instead of `bulk_insert_results()`
2. **Stream instead of load all**: Use `stream_products()` for processing millions of records
3. **Paginate API responses**: Use `get_paginated_*()` methods
4. **Use FTS5 for text search**: 50-100x faster than LIKE queries

### Database Maintenance

Run periodically (e.g., weekly):
```python
await db_manager.optimize()
# Runs ANALYZE, incremental VACUUM, expires cache, removes old tasks
```

### Query Optimization

Check if indexes are used:
```python
result = await session.execute(text("EXPLAIN QUERY PLAN SELECT ..."))
# Should show "USING INDEX" for optimal queries
```

## Configuration Pattern

The project uses Pydantic v2 Settings for centralized configuration management:

```python
# Import global settings instance
from src.config.settings import settings

# Access configuration values
database_url = settings.get_database_url_async()  # Async SQLite URL
model = settings.OLLAMA_MODEL
api_port = settings.API_PORT

# Settings are validated at startup
# All environment variables loaded from .env file
```

**Configuration Loading**:
- Settings load from `.env` file (see `.env.example` for template)
- 33 settings across 11 categories (Database, LLM, API, Cache, etc.)
- Pydantic v2 validators ensure type safety and constraints
- Directories auto-created on startup (`data/`, `data/logs/`, etc.)

## Project Status

**Version 1.0.0 - Production Ready** ✅

All 10 backend modules completed and tested:

1. ✅ **Database Layer** (Module 1) - 23 SQLAlchemy models, 73 operations, FTS5 search
2. ✅ **Configuration System** (Module 2) - 33 Pydantic v2 settings, validation, auto-directory creation
3. ✅ **Cache System** (Module 3) - Two-tier caching, TTL expiration, specialized cache types
4. ✅ **Task Queue** (Module 4) - Priority scheduling, retry logic, worker pool management
5. ✅ **Web Scraper** (Module 5) - Google & Bing scrapers with rate limiting
6. ✅ **Data Processor** (Module 6) - Data cleaning, deduplication, batch processing
7. ✅ **LLM Analyzer** (Module 7) - Ollama integration for sentiment/competitor analysis
8. ✅ **REST API** (Module 8) - FastAPI with 40+ endpoints, WebSocket support, CORS
9. ✅ **Export System** (Module 9) - CSV, Excel, JSON exports with formatting
10. ✅ **Scheduler** (Module 10) - APScheduler for recurring jobs, cron support

**Code Metrics**:
- ~17,000 lines of production code
- 150+ comprehensive tests
- 85%+ test coverage
- 11 test files covering all modules


## Key Files for Context

When working on backend features, reference:
- `src/database/models.py` - All 23 data models
- `src/database/operations.py` - 73 CRUD operations
- `src/config/settings.py` - 33 configuration settings
- `src/api/main.py` - FastAPI application entry point
- `Backend_Modules/` - 10 module implementation guides (Mod_01 through Mod_10)
- `tests/` - Usage examples and expected behaviors
- `.env.example` - Complete configuration template


## Working with This Codebase

### Adding a New API Endpoint

1. Create route function in appropriate `src/api/routes/*.py` file
2. Add Pydantic schemas for request/response in `src/api/schemas/`
3. Register router in `src/api/main.py` if new route module
4. Use dependency injection for database sessions
5. Write tests in `tests/test_api.py`

Example:
```python
# In src/api/routes/search.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..dependencies import get_db_session

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def create_search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db_session)
):
    # Implementation
    pass
```

### Adding a New Scrapy Spider

1. Create spider class in `src/scrapers/spiders/`
2. Inherit from `scrapy.Spider` or use existing base
3. Define `name`, `allowed_domains`, `start_requests()`
4. Implement `parse()` method to yield items
5. Items automatically processed by pipelines
6. Add convenience function to `scrapy_runner.py`
7. Write tests in `tests/test_scrapers.py`

### Adding a New Database Model

1. Define in `src/database/models.py` with relationships
2. Add CRUD operations to `src/database/operations.py`
3. Consider FTS5 if text-searchable (update `src/database/fts.py`)
4. Create indexes in model definition or migration
5. Write tests in appropriate test file

### Adding a New Task Handler

1. Define async handler function in `src/task_queue/handlers.py`
2. Register handler with task queue in FastAPI lifespan
3. Handler receives payload dict, returns result dict
4. Use `@retry` decorator for automatic retries
5. Write tests in `tests/test_task_queue.py`

### Debugging Issues

**Database**:
```python
# Enable SQL echo
db_manager.engine = create_async_engine(url, echo=True)

# Check health
is_healthy = await db_manager.health_check()

# Get stats
stats = await db_manager.get_stats()
```

**Scrapy**:
```python
# Check which reactor is installed
from twisted.internet import reactor
print(f"Reactor: {reactor.__class__.__module__}.{reactor.__class__.__name__}")

# Enable Scrapy debug logging in .env
SCRAPY_LOG_LEVEL=DEBUG
```

**Task Queue**:
```python
# Check task status
status = await task_queue.get_task_status(task_id)

# Get queue stats
stats = await task_queue.get_stats()
```
- add latest to memory