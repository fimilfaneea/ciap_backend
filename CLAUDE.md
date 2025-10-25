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
python -m src.core.database

# Run database tests
pytest tests/test_database.py -v

# Run all tests
pytest tests/ -v
```

### Development Workflow
```bash
# Format code
black src/ tests/

# Run specific test file
pytest tests/test_integration.py -v

# Run tests with coverage (if configured)
pytest tests/ --cov=src
```

## Architecture Overview

### Database-Centric Design

The project is built around a comprehensive SQLite database with **23 models** organized into 6 categories:

1. **Search Operations** (3 models): `Search`, `SearchResult`, `Analysis`
2. **Product Intelligence** (4 models): `Product`, `PriceData`, `Offer`, `ProductReview`
3. **Competitor Tracking** (3 models): `Competitor`, `CompetitorProducts`, `CompetitorTracking`
4. **Market Intelligence** (4 models): `MarketTrend`, `SERPData`, `SocialSentiment`, `NewsContent`
5. **Feature Analysis** (2 models): `FeatureComparison`, `Insights`
6. **Infrastructure** (5 models): `Cache`, `TaskQueue`, `ScrapingJob`, `RateLimit`, `PriceHistory`

### Core Module: `src/core/`

The heart of the system with 6 critical files:

- **`models.py`** (435 lines): All 23 SQLAlchemy ORM models
- **`database.py`** (356 lines): `DatabaseManager` class with async connection management, WAL mode, FTS5 setup
- **`db_ops.py`** (1817 lines): `DatabaseOperations` class with 73 static methods for CRUD operations
- **`fts_setup.py`** (338 lines): SQLite FTS5 full-text search configuration with automatic triggers
- **`config.py`** (335 lines): Type-safe configuration management with Pydantic v2 validation
- **`config_utils.py`** (276 lines): Configuration utilities, validation, and export functionality

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

### Overview

Centralized configuration management using Pydantic v2 Settings for type-safe, validated configuration loading.

**Key Features**:
- Environment variable loading from `.env` files
- Type validation and constraints
- Automatic directory creation
- Secret filtering for exports
- Runtime configuration inspection

### Core Configuration (`src/core/config.py`)

**33 configuration settings** organized into 11 categories:

1. **Environment**: `ENVIRONMENT` (development/testing/production), `LOG_LEVEL`
2. **Database**: `DATABASE_URL`, `DATABASE_POOL_SIZE`, `DATABASE_ECHO`
3. **Ollama LLM**: `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`
4. **Web Scraping**: `SCRAPER_TIMEOUT`, `SCRAPER_RETRY_COUNT`, `SCRAPER_RATE_LIMIT_DELAY`, `SCRAPER_USER_AGENTS`
5. **Google/Bing Scrapers**: URLs and max results per platform
6. **Cache**: `CACHE_TTL_SECONDS`, `CACHE_CLEANUP_INTERVAL`
7. **Task Queue**: `TASK_QUEUE_MAX_WORKERS`, `TASK_QUEUE_POLL_INTERVAL`, `TASK_MAX_RETRIES`
8. **API Server**: `API_HOST`, `API_PORT`, `API_PREFIX`, `API_CORS_ORIGINS`, `API_RATE_LIMIT_REQUESTS`
9. **Security**: `SECRET_KEY`, `API_KEY` (filtered in exports)
10. **Logging**: `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`
11. **Directories**: `DATA_DIR`, `EXPORT_DIR`, `PROMPTS_DIR`

### Usage Patterns

**Basic Access**:
```python
from src.core.config import settings

# Access configuration values
database_url = settings.get_database_url_async()  # Converts to async format
model = settings.OLLAMA_MODEL
user_agent = settings.get_user_agent()  # Random user agent from list
```

**Database Integration**:
```python
# database.py automatically uses settings
db_manager = DatabaseManager()  # Uses settings.get_database_url_async()
# Or override for testing:
db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
```

**Configuration Export**:
```python
# Export configuration (secrets excluded by default)
config_dict = settings.to_dict(exclude_secrets=True)
```

### Configuration Utilities (`src/core/config_utils.py`)

**ConfigManager** provides 6 static methods:

1. **`load_prompts()`**: Load all LLM prompt templates from `config/prompts/`
```python
prompts = ConfigManager.load_prompts()
sentiment_prompt = prompts["sentiment"]  # Returns prompt content
```

2. **`validate_environment()`**: Check directories and external services
```python
issues = ConfigManager.validate_environment()
# Returns list of issues: ["Cannot connect to Ollama - is it running?"]
```

3. **`export_config(output_file)`**: Export configuration to JSON/YAML
```python
path = ConfigManager.export_config("config_export.json")
# Secrets automatically filtered
```

4. **`get_runtime_info()`**: Gather system and runtime information
```python
info = ConfigManager.get_runtime_info()
# Returns: python_version, platform, environment, database, etc. (20 fields)
```

5. **`get_prompt_path(prompt_name)`**: Get path to specific prompt file
```python
path = settings.get_prompt_path("sentiment")
# Returns: Path("config/prompts/sentiment.txt")
```

6. **`print_config_summary()`**: Display configuration overview

### Environment Variables

Configuration loads from `.env` file (see `.env.example` for template). Key variables:

```bash
# Environment
ENVIRONMENT=development  # development, testing, or production

# Database
DATABASE_URL=sqlite:///data/ciap.db
DATABASE_POOL_SIZE=5

# Ollama LLM
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# API
API_HOST=0.0.0.0
API_PORT=8000

# Security (REQUIRED in production)
SECRET_KEY=your-secret-key-at-least-32-characters-long
```

### Validation Features

**Pydantic v2 Field Validators**:
- Database URLs must start with `sqlite:///`
- Pool size: 1-20 connections
- API port: 1000-65535
- Timeouts: positive integers with max limits
- Ollama timeout: 10-300 seconds
- **Production check**: Warns if using default SECRET_KEY in production

**Auto-Directory Creation**:
- `data/`, `data/logs/`, `data/exports/`, `config/prompts/`
- Created automatically when Settings is instantiated
- Validators ensure directories exist before use

### Prompt Template System

Prompts stored in `config/prompts/` as `.txt` files:
- `sentiment.txt`: Sentiment analysis prompt
- `competitor.txt`: Competitive intelligence prompt
- `summary.txt`: Text summarization prompt
- `keywords.txt`: Keyword extraction prompt

Access programmatically:
```python
prompts = ConfigManager.load_prompts()
# Returns dict: {"sentiment": "...", "competitor": "...", ...}
```

### Configuration Testing

Test suite in `tests/test_config.py` with 11 test functions:
- Default value verification
- Environment variable overrides
- Validation error handling
- Production security checks
- Directory auto-creation
- URL conversion (sync → async)
- User agent rotation
- Secret filtering
- Prompt loading
- Environment validation
- Config export (JSON/YAML)

### Important Notes

**Pydantic v2 Patterns**:
- Use `pydantic_settings.BaseSettings` (not `pydantic.BaseSettings`)
- Use `@field_validator` decorator (not `@validator`)
- Validators must be `@classmethod` methods
- Use `model_dump()` instead of `dict()`

**Singleton Pattern**:
```python
# Single global settings instance
from src.core.config import settings

# Always use this instance (don't create new Settings())
# Exception: Testing with custom environments
```

**Testing Override**:
```python
# For tests, use monkeypatch to override environment variables
monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
test_settings = Settings()
```

## Cache System

### Overview

SQLite-based caching system with optional in-memory layer for improved performance. Supports TTL expiration, pattern-based deletion, statistics tracking, and specialized cache types.

**Key Features**:
- Two-tier caching (memory + database)
- TTL-based expiration
- Pattern-based deletion
- Statistics tracking (hits, misses, hit rate)
- Background cleanup task
- Compression for large values
- Batch operations
- Specialized cache types (Search, LLM, RateLimit, Session)

### Core CacheManager (`src/core/cache.py`)

**Basic Usage**:
```python
from src.core.cache import cache

# Set value with TTL
await cache.set("user:123:profile", user_data, ttl=3600)

# Get value
profile = await cache.get("user:123:profile")

# Check existence
if await cache.exists("user:123:profile"):
    # ...

# Delete value
await cache.delete("user:123:profile")

# Pattern-based deletion (SQL LIKE syntax)
deleted_count = await cache.delete_pattern("user:123:%")

# Get statistics
stats = await cache.get_stats()
# Returns: hits, misses, sets, deletes, hit_rate, total_entries, etc.
```

**Advanced Operations**:
```python
# Batch retrieval
keys = ["key1", "key2", "key3"]
results = await cache.get_many(keys)  # Dict[str, Any]

# Compression for large values (>1KB)
await cache.set_compressed("large_data", huge_object, ttl=3600)
data = await cache.get_compressed("large_data")

# Cache warming on startup
operations = [
    {'key': 'config:settings', 'value': settings_dict, 'ttl': 7200},
    {'key': 'common:queries', 'value': query_list, 'ttl': 3600}
]
from src.core.cache import warm_cache
count = await warm_cache(operations)
```

**Key Generation**:
```python
# Automatic key generation with namespacing
from src.core.cache import CacheManager

key = CacheManager.make_key("user", "profile", user_id=123, version="v2")
# Result: "user:profile:user_id=123:version=v2"

# Long keys (>200 chars) are automatically hashed (MD5)
```

### Cache Decorators (`src/core/cache_decorators.py`)

**@cached Decorator**:
```python
from src.core.cache_decorators import cached

@cached(ttl=3600, key_prefix="search")
async def search_products(query: str, limit: int = 10):
    # Expensive database operation
    results = await db_search(query, limit)
    return results

# First call executes function
results = await search_products("laptop")  # Cache miss, executes

# Second call uses cache
results = await search_products("laptop")  # Cache hit, instant
```

**@cache_result Decorator** (Simple):
```python
@cache_result(ttl=600)
async def get_trending_products():
    # Expensive computation
    return trending_list
```

**@invalidate_cache Decorator**:
```python
@invalidate_cache("product:*")
async def update_product_catalog():
    # Update operation
    await db.update_products()
    # Cache automatically invalidated after execution
```

**@conditional_cache Decorator**:
```python
@conditional_cache(
    condition=lambda query: len(query) > 3,
    ttl=1800,
    key_prefix="search"
)
async def search(query: str):
    # Only cache if query length > 3
    return results
```

### Specialized Cache Types (`src/core/cache_types.py`)

**SearchCache** - For search results:
```python
from src.core.cache_types import SearchCache

# Cache search results
results = [{"title": "...", "url": "..."}]
await SearchCache.set_search_results("python tutorial", "google", results, ttl=3600)

# Retrieve cached results
cached = await SearchCache.get_search_results("python tutorial", "google")

# Invalidate all results for a query (all sources)
await SearchCache.invalidate_search("python tutorial")

# Get or execute pattern
results = await SearchCache.get_or_search(
    "python tutorial",
    "google",
    search_func=scraper.search_google,
    ttl=3600
)
```

**LLMCache** - For LLM analysis (longer TTL):
```python
from src.core.cache_types import LLMCache
import hashlib

# Cache LLM analysis (2 hour default TTL)
text = "Product review text..."
text_hash = hashlib.md5(text.encode()).hexdigest()
analysis = {"sentiment": "positive", "score": 0.9}

await LLMCache.set_analysis(text_hash, "sentiment", analysis, ttl=7200)

# Retrieve cached analysis
cached = await LLMCache.get_analysis(text_hash, "sentiment")

# Get or execute pattern
analysis = await LLMCache.get_or_analyze(
    text,
    "sentiment",
    analysis_func=llm.analyze_sentiment,
    ttl=7200
)
```

**RateLimitCache** - For rate limiting:
```python
from src.core.cache_types import RateLimitCache

# Check rate limit (e.g., 100 requests per minute)
identifier = request.client.host  # IP address
allowed = await RateLimitCache.check_rate_limit(identifier, limit=100, window=60)

if not allowed:
    raise HTTPException(status_code=429, detail="Rate limit exceeded")

# Get remaining requests
remaining = await RateLimitCache.get_remaining(identifier, limit=100)

# Manual reset
await RateLimitCache.reset_rate_limit(identifier)
```

**SessionCache** - For user sessions:
```python
from src.core.cache_types import SessionCache

# Create session (returns UUID)
session_id = await SessionCache.create_session(
    user_id="user_123",
    data={"username": "john", "role": "admin"},
    ttl=3600
)

# Get session
session = await SessionCache.get_session(session_id)

# Update session
await SessionCache.update_session(
    session_id,
    {"last_activity": datetime.utcnow().isoformat()},
    extend_ttl=True
)

# Delete session
await SessionCache.delete_session(session_id)
```

### Performance Optimization Tips

**1. Use Appropriate TTLs**:
- Short-lived data (5-15 minutes): Search results, API responses
- Medium-lived data (1-2 hours): LLM analysis, processed data
- Long-lived data (6-24 hours): Configuration, reference data

**2. Enable Memory Cache Layer**:
```python
# Memory cache provides 10-100x speedup for hot data
cache = CacheManager(enable_memory_cache=True)  # Default
```

**3. Use Batch Operations**:
```python
# Instead of multiple get() calls
keys = [f"product:{id}" for id in product_ids]
products = await cache.get_many(keys)  # Single DB query
```

**4. Compress Large Values**:
```python
# Automatically compress values >1KB
await cache.set_compressed("large_json", big_data, threshold=1000)
```

**5. Pattern-Based Invalidation**:
```python
# Invalidate related keys efficiently
await cache.delete_pattern("user:123:%")  # All user data
await cache.delete_pattern("product:%")   # All products
```

### Integration Points

**With Database Module**:
```python
from src.core.database import db_manager
from src.core.cache_decorators import cached

@cached(ttl=300, key_prefix="db")
async def get_product_by_id(product_id: int):
    async with db_manager.get_session() as session:
        return await DatabaseOperations.get_product_by_id(session, product_id)
```

**With Config Module**:
```python
from src.core.config import settings

# Configuration values are automatically used
default_ttl = settings.CACHE_TTL_SECONDS  # Default: 3600
cleanup_interval = settings.CACHE_CLEANUP_INTERVAL  # Default: 3600
```

**With API Module** (Rate Limiting):
```python
from fastapi import Request, HTTPException
from src.core.cache_types import RateLimitCache

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    identifier = request.client.host

    if not await RateLimitCache.check_rate_limit(identifier, limit=100, window=60):
        raise HTTPException(status_code=429, detail="Too many requests")

    return await call_next(request)
```

### Cache Statistics & Monitoring

```python
# Get comprehensive statistics
stats = await cache.get_stats()

# Returns:
# {
#     'hits': 150,
#     'misses': 50,
#     'sets': 200,
#     'deletes': 10,
#     'memory_hits': 120,
#     'db_hits': 30,
#     'total_entries': 190,
#     'expired_entries': 5,
#     'active_entries': 185,
#     'hit_rate': '75.00%',
#     'memory_cache_size': 45
# }
```

### Background Cleanup Task

The cache automatically runs a background task to clean up expired entries:

```python
# Automatic cleanup every CACHE_CLEANUP_INTERVAL seconds
# Set in .env: CACHE_CLEANUP_INTERVAL=3600

# Manual cleanup
deleted_count = await cache.cleanup_expired()
print(f"Cleaned up {deleted_count} expired entries")
```

### Testing Cache

```python
import pytest
from src.core.cache import CacheManager

@pytest.fixture
async def test_cache():
    """Create isolated test cache instance"""
    cache = CacheManager(enable_memory_cache=True)
    await cache.initialize()
    yield cache
    await cache.close()

@pytest.mark.asyncio
async def test_cache_operations(test_cache):
    await test_cache.set("key", "value", ttl=60)
    assert await test_cache.get("key") == "value"
```

## Task Queue System

### Overview

SQLite-based asynchronous task queue with priority scheduling, retry logic, and worker pool management. Provides background job processing without external dependencies like Celery or Redis.

**Key Features**:
- Priority-based task scheduling (5 priority levels)
- Worker pool with configurable size
- Retry logic with exponential backoff
- Task status tracking (6 states)
- Task result caching
- Task chaining and grouping
- Recurring task scheduling
- Workflow management

### Core TaskQueueManager (`src/task_queue/manager.py`)

**Basic Usage**:
```python
from src.task_queue import task_queue, TaskStatus, TaskPriority

# Register handler
async def my_handler(payload):
    # Process task
    return {"result": "success"}

task_queue.register_handler("my_task", my_handler)

# Start workers
await task_queue.start()

# Enqueue task
task_id = await task_queue.enqueue(
    "my_task",
    {"data": "value"},
    priority=TaskPriority.HIGH
)

# Check status
status = await task_queue.get_task_status(task_id)
print(f"Status: {status['status']}, Result: {status['result']}")

# Stop workers
await task_queue.stop()
```

**Priority Levels**:
```python
from src.task_queue import TaskPriority

TaskPriority.CRITICAL    # 1 - Highest priority
TaskPriority.HIGH        # 3
TaskPriority.NORMAL      # 5 - Default
TaskPriority.LOW         # 7
TaskPriority.BACKGROUND  # 10 - Lowest priority
```

**Task States**:
```python
from src.task_queue import TaskStatus

TaskStatus.PENDING      # Waiting to be processed
TaskStatus.PROCESSING   # Currently being executed
TaskStatus.COMPLETED    # Successfully finished
TaskStatus.FAILED       # Failed but will retry
TaskStatus.DEAD         # Failed after max retries
TaskStatus.CANCELLED    # Manually cancelled
```

### Task Handlers (`src/task_queue/handlers.py`)

**Default Handlers** (Placeholders for future modules):
```python
from src.task_queue import register_default_handlers

# Register all default handlers at startup
register_default_handlers()

# Available handlers:
# - scrape_handler (Module 5 - Web Scraper)
# - analyze_handler (Module 7 - LLM Analyzer)
# - export_handler (Module 9 - Export)
# - batch_handler (functional - processes multiple tasks)
```

**Custom Handler Pattern**:
```python
async def custom_handler(payload: Dict[str, Any]) -> Dict:
    """
    Custom task handler

    Args:
        payload: Task data dictionary

    Returns:
        Result dictionary (will be cached)
    """
    # Extract data
    data = payload.get("data")

    # Process data
    result = process_data(data)

    # Return result
    return {"status": "success", "result": result}

# Register handler
task_queue.register_handler("custom", custom_handler)
```

### Task Utilities

**Sequential Execution (TaskChain)**:
```python
from src.task_queue import TaskChain

chain = TaskChain()
chain.add("scrape", {"query": "AI news"})
chain.add("analyze", {"type": "sentiment"})
chain.add("export", {"format": "csv"})

# Execute tasks one after another
task_ids = await chain.execute(timeout_per_task=300)
```

**Parallel Execution (TaskGroup)**:
```python
from src.task_queue import TaskGroup

group = TaskGroup()
group.add("scrape", {"query": "AI", "sources": ["google"]})
group.add("scrape", {"query": "ML", "sources": ["bing"]})
group.add("scrape", {"query": "Data", "sources": ["google", "bing"]})

# Enqueue all tasks simultaneously
task_ids = await group.execute()

# Wait for all to complete
results = await group.wait_all(task_ids, timeout=300)
```

**Wait for Task**:
```python
from src.task_queue import wait_for_task

task_id = await task_queue.enqueue("process", {"data": "value"})

# Poll until completion or timeout
result = await wait_for_task(task_id, timeout=60, poll_interval=1.0)

if result and result["status"] == TaskStatus.COMPLETED:
    print(f"Result: {result['result']}")
```

**Recurring Tasks**:
```python
from src.task_queue import schedule_recurring_task, TaskPriority

# Run every hour
recurring_task = await schedule_recurring_task(
    "scrape",
    {"query": "AI news"},
    interval_seconds=3600,
    max_runs=24,  # Run 24 times (24 hours)
    priority=TaskPriority.BACKGROUND
)

# Cancel later
recurring_task.cancel()
```

**Workflows**:
```python
from src.task_queue import create_workflow, get_workflow_status

# Create multi-step workflow
workflow = await create_workflow(
    "competitor_analysis",
    [
        {"type": "scrape", "payload": {"query": "competitor"}},
        {"type": "analyze", "payload": {"type": "competitor"}},
        {"type": "export", "payload": {"format": "csv"}}
    ]
)

# Check workflow status
status = await get_workflow_status(workflow["workflow_key"])
```

### Advanced Queue Operations

**Batch Enqueue**:
```python
tasks = [
    {"type": "scrape", "payload": {"query": f"query{i}"}}
    for i in range(100)
]

task_ids = await task_queue.enqueue_batch(tasks, priority=TaskPriority.NORMAL)
# More efficient than 100 individual enqueue() calls
```

**Cancel Task**:
```python
task_id = await task_queue.enqueue("long_task", {})

# Cancel if still pending
cancelled = await task_queue.cancel_task(task_id)
```

**Requeue Failed Tasks**:
```python
# Requeue all failed/stuck tasks for retry
requeued_count = await task_queue.requeue_failed()
print(f"Requeued {requeued_count} tasks")
```

**Queue Statistics**:
```python
stats = await task_queue.get_queue_stats()

# Returns:
# {
#     'status_counts': {'pending': 10, 'processing': 2, 'completed': 50, ...},
#     'type_counts': {'scrape': 30, 'analyze': 20, 'export': 12},
#     'avg_wait_time_seconds': 5.2,
#     'worker_count': 3,
#     'active_workers': 3,
#     'tasks_processed': 50,
#     'tasks_failed': 2,
#     'tasks_retried': 3,
#     'avg_processing_time': 1.5
# }
```

**Cleanup Old Tasks**:
```python
# Delete completed tasks older than 7 days
deleted = await task_queue.cleanup_old_tasks(days=7)
```

### Integration Points

**With Database Module**:
```python
from src.task_queue import task_queue
from src.database import db_manager, DatabaseOperations

async def database_task_handler(payload):
    search_id = payload.get("search_id")

    async with db_manager.get_session() as session:
        search = await DatabaseOperations.get_search_by_id(session, search_id)
        # Process search
        return {"search": search.query}

task_queue.register_handler("db_task", database_task_handler)
```

**With Cache Module**:
```python
from src.task_queue import task_queue
from src.cache import cache, SearchCache

async def cached_scrape_handler(payload):
    query = payload.get("query")

    # Check cache first
    cached_results = await SearchCache.get_search_results(query, "google")
    if cached_results:
        return cached_results

    # Perform scrape
    results = perform_scrape(query)

    # Cache results
    await SearchCache.set_search_results(query, "google", results, ttl=3600)

    return results
```

**With Config Module**:
```python
from src.task_queue import task_queue
from src.config import settings

# Configuration automatically used:
# - settings.TASK_QUEUE_MAX_WORKERS (default: 3)
# - settings.TASK_QUEUE_POLL_INTERVAL (default: 1.0)
# - settings.TASK_MAX_RETRIES (default: 3)

print(f"Worker count: {task_queue.worker_count}")
print(f"Max retries: {task_queue.max_retries}")
```

### Testing Patterns

**Test Fixture**:
```python
import pytest
from src.task_queue import TaskQueueManager

@pytest.fixture
async def test_queue():
    """Create isolated test queue"""
    queue = TaskQueueManager()

    # Register test handler
    async def test_handler(payload):
        await asyncio.sleep(0.1)
        return {"result": f"processed_{payload.get('data')}"}

    queue.register_handler("test", test_handler)
    await queue.start()

    yield queue

    await queue.stop()
    await queue.clear_queue()

@pytest.mark.asyncio
async def test_task_processing(test_queue):
    task_id = await test_queue.enqueue("test", {"data": "value"})
    await asyncio.sleep(0.3)

    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.COMPLETED
    assert status["result"]["result"] == "processed_value"
```

### Common Issues & Solutions

**Issue 1: Tasks Not Being Processed**
```python
# Problem: Workers not started
await task_queue.start()  # Must start workers

# Problem: No handler registered
task_queue.register_handler("my_task", my_handler)
```

**Issue 2: Memory Leak with Long-Running Queue**
```python
# Solution: Periodically clean old tasks
async def periodic_cleanup():
    while True:
        await asyncio.sleep(86400)  # Daily
        await task_queue.cleanup_old_tasks(days=7)

asyncio.create_task(periodic_cleanup())
```

**Issue 3: Task Timeout**
```python
# Wrap handler with timeout
async def handler_with_timeout(payload):
    try:
        return await asyncio.wait_for(
            actual_handler(payload),
            timeout=60
        )
    except asyncio.TimeoutError:
        raise Exception("Task timeout after 60s")
```

**Issue 4: Handling Large Results**
```python
# Problem: Result too large for cache
async def handler_with_large_result(payload):
    result = generate_large_result()

    if len(json.dumps(result)) > 1_000_000:  # 1MB
        # Save to file instead
        file_path = f"data/results/{payload['task_id']}.json"
        with open(file_path, "w") as f:
            json.dump(result, f)
        return {"file": file_path}

    return result
```

**Issue 5: Worker Crashes**
```python
# Workers auto-restart on error
# Check worker health:
stats = await task_queue.get_queue_stats()
if stats["active_workers"] < stats["worker_count"]:
    logger.warning("Some workers may have crashed")
    # Restart queue
    await task_queue.stop()
    await task_queue.start()
```

### Performance Optimization

**1. Batch Enqueue for Efficiency**:
```python
# Slow: 100 individual enqueue() calls
for i in range(100):
    await task_queue.enqueue("task", {"n": i})

# Fast: Single batch enqueue
tasks = [{"type": "task", "payload": {"n": i}} for i in range(100)]
await task_queue.enqueue_batch(tasks)
```

**2. Tune Worker Count**:
```python
# In .env file
TASK_QUEUE_MAX_WORKERS=5  # Increase for high throughput

# For I/O-bound tasks: 5-10 workers
# For CPU-bound tasks: 2-4 workers
```

**3. Use Priority Wisely**:
```python
# High priority for user-facing tasks
await task_queue.enqueue("user_request", payload, priority=TaskPriority.HIGH)

# Low priority for background maintenance
await task_queue.enqueue("cleanup", payload, priority=TaskPriority.BACKGROUND)
```

**4. Monitor Queue Depth**:
```python
stats = await task_queue.get_queue_stats()
pending_count = stats["status_counts"]["pending"]

if pending_count > 1000:
    logger.warning(f"Queue backlog: {pending_count} pending tasks")
    # Consider scaling up workers
```

### Application Startup Pattern

```python
from src.task_queue import task_queue, register_default_handlers
from src.database import init_db

async def startup():
    """Application startup"""
    # Initialize database
    await init_db()

    # Register task handlers
    register_default_handlers()

    # Register custom handlers
    task_queue.register_handler("custom", custom_handler)

    # Start task queue workers
    await task_queue.start()

    print(f"Task queue started with {task_queue.worker_count} workers")

async def shutdown():
    """Application shutdown"""
    # Stop task queue gracefully
    await task_queue.stop()
    print("Task queue stopped")

# Usage with FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    await startup()

@app.on_event("shutdown")
async def on_shutdown():
    await shutdown()
```

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

The project uses pydantic-settings for config (see `progress.md` for detailed guide):
```python
# Expected pattern (not yet implemented in src/):
from config import settings

DATABASE_URL = settings.DATABASE_URL
OLLAMA_URL = settings.OLLAMA_URL
```

Currently using direct environment variables via `python-dotenv`.

## Project Status

**Completed**:
- ✅ Complete database layer (23 models, 73 operations)
- ✅ Full-text search (FTS5) with 4 virtual tables
- ✅ Configuration management system (33 settings, Pydantic v2 validation)
- ✅ Comprehensive test suite (8 test files including config tests)
- ✅ Documentation (10 module guides + core reference)

**In Development** (per `MODULE_STATUS.md`):
- Scraper implementations (Module 3)
- Cache system (Module 4)
- Task queue (Module 5)
- LLM analyzer integration (Module 7)
- API endpoints (Module 8)
- Export functionality (Module 9)
- Scheduler (Module 10)

## Key Files for Context

When working on features, reference:
- `src/core/models.py` - All data structures
- `src/core/db_ops.py` - Available database operations
- `src/core/Documentation.md` - Complete API reference
- Module guides (`Mod_*.md`) - Implementation patterns
- `tests/` - Usage examples and expected behaviors

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
