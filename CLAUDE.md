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

# Create .env file from template
cp .env.example .env         # Then edit with actual API keys
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
```bash
# Start FastAPI development server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Access interactive API documentation
# Swagger UI: http://localhost:8000/api/docs
# ReDoc: http://localhost:8000/api/redoc

# Production server (without reload)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
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
├── scrapers/       # Module 5: Web Scraper
│   ├── base.py           # BaseScraper abstract class
│   ├── google.py         # Google search scraper
│   ├── bing.py           # Bing search scraper
│   └── manager.py        # ScraperManager orchestrator
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
│   └── routes/           # 5 route modules (search, tasks, analysis, export, scheduler)
│       ├── search.py
│       ├── tasks.py
│       ├── analysis.py
│       ├── export.py
│       └── scheduler.py
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

## Testing Strategy

### Test Organization (7 test files, 244 KB total)

- `test_database.py`: Core CRUD operations, session management
- `test_models.py`: Model validation, relationships, constraints
- `test_integration.py`: End-to-end workflows
- `test_concurrency.py`: Concurrent access, race conditions
- `test_performance.py`: Bulk operations, query optimization
- `test_transactions.py`: Transaction handling, rollback scenarios
- `test_utils.py`: Helper functions, utilities

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
- requests 2.31.0, beautifulsoup4 4.12.2, lxml 4.9.3

**LLM Integration**:
- openai 1.3.5, anthropic 0.7.1

**Testing & Dev**:
- pytest 7.4.3, black 23.11.0

## Common Gotchas

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

### Adding a New Model

1. Define in `src/core/models.py` with relationships
2. Add indexes in `database.py` `_create_indexes()`
3. Add CRUD operations to `db_ops.py`
4. Consider FTS5 if text-searchable (update `fts_setup.py`)
5. Write tests in `tests/`

### Adding Database Operations

1. Add static method to `DatabaseOperations` class in `db_ops.py`
2. Use existing patterns (pagination, streaming, chunking)
3. Handle errors gracefully (operations auto-rollback)
4. Write unit tests in `tests/test_database.py`
5. Write integration tests if multiple operations involved

### Debugging Database Issues

Enable SQL echo:
```python
db_manager.engine = create_async_engine(url, echo=True)  # Logs all SQL
```

Check health:
```python
is_healthy = await db_manager.health_check()
```

Verify indexes:
```python
await session.execute(text("PRAGMA index_list('table_name')"))
```
