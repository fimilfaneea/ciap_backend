# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Project Context

**CIAP (Competitive Intelligence Automation Platform)** is an active M.Tech IT project for building an open-source competitive intelligence solution for SMEs. This is a 12-week implementation project with specific academic requirements.

**Student:** Fimil Faneea (24102371), M.Tech IT
**Timeline:** 12 weeks total
**Goal:** Create a cost-effective alternative to expensive CI tools (70-90% cost reduction) using SQLite-only architecture

**IMPORTANT:** Unless specified otherwise, assume all questions relate to implementing this system within the 12-week timeline. This is an academic project focusing on practical implementation over perfect architecture.

## 📍 Current Implementation Status

### Completed Components
- ✅ **Database Models** (src/core/models.py - 435 lines):
  - All 23 models fully defined: Search, Product, Competitor, PriceData, Offer, ProductReview, MarketTrend, SERPData, SocialSentiment, NewsContent, FeatureComparison, Insights, etc.
  - SQLite with async support via aiosqlite
  - Comprehensive relationships, foreign keys, and cascades

- ✅ **Database Manager** (src/core/database.py - 346 lines):
  - Async database operations with connection pooling
  - WAL mode enabled for concurrent access
  - SQLite optimizations (cache_size=10000, temp_store=MEMORY)
  - Health check, stats, and graceful shutdown

- ✅ **FTS5 Setup** (src/core/fts_setup.py - 338 lines):
  - Full-text search configuration for all content models
  - Virtual tables for optimized text searching

- ✅ **Database Operations** (src/core/db_ops.py - 1817 lines):
  - Complete CRUD operations for all 23 models
  - Bulk operations with chunk processing
  - Cache management with TTL support
  - Task queue operations (enqueue, dequeue, retry)
  - Price history and competitor tracking

- ✅ **Database Initialization** (scripts/init_database.py - 732 lines):
  - Automated database setup and table creation
  - Index creation for performance
  - Schema verification and integrity checks
  - Sample data insertion for testing

- ✅ **Comprehensive Tests** (tests/test_database.py - 732 lines):
  - Unit tests for all database operations
  - Integration tests for complex workflows
  - Performance benchmarks included
  - Custom test runner (no pytest dependency)

### Next Priority Components
- 🔴 **Main FastAPI application** (main.py) - Core API server
- 🔴 **Configuration module** (config.py) - Environment and settings management
- 🔴 **Google Scraper** (scrapers/google_scraper.py) - Primary data source
- 🟡 **LLM Integration** (analyzers/llm_analyzer.py) - Ollama setup for analysis
- 🟡 **Task Queue Processor** - Background job execution
- 🟡 **Basic API Routes** - Search and analysis endpoints

## 🛠 Common Development Commands

### Environment Setup
```bash
# Activate virtual environment (Windows)
F:\Project\CIAP\venv\Scripts\activate

# Install/update dependencies
pip install -r requirements.txt

# Install Ollama (Windows)
# Download from: https://ollama.com/download/windows
ollama pull llama3.1:8b
```

### Database Operations
```bash
# Initialize database with all tables and indexes
python scripts/init_database.py

# Check database status without initialization
python scripts/init_database.py --check-only

# Test database operations
python tests/test_database.py

# Quick database connection test
python -c "from src.core.database import DatabaseManager; import asyncio; db=DatabaseManager(); asyncio.run(db.initialize()); print('Database connected successfully')"
```

### Running the Application
```bash
# Start FastAPI server (when main.py is implemented)
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run database tests (custom test runner - no pytest needed)
python tests/test_database.py

# Run specific test groups
python tests/test_database.py TestDatabaseInitialization
python tests/test_database.py TestDatabaseOperations

# For pytest users (if pytest is installed)
pytest tests/ -v
pytest --cov=src --cov-report=html
```

## 🏗 Architecture Overview

### Project Structure
```
F:\Project\CIAP\
├── src/
│   └── core/
│       ├── models.py         # ✅ 23 database models
│       ├── database.py       # ✅ Async database manager
│       ├── db_ops.py         # ✅ CRUD operations
│       └── fts_setup.py      # ✅ Full-text search
├── scripts/
│   └── init_database.py     # ✅ Database initialization
├── tests/
│   └── test_database.py     # ✅ Test suite
├── data/                     # SQLite database storage
├── venv/                     # Python virtual environment
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables
└── ciap.db                   # SQLite database file
```

### SQLite-Only Design
The project uses SQLite for ALL data storage needs:
- **Main Database**: Application data (ciap.db)
- **Cache Storage**: TTL-based caching in cache table
- **Task Queue**: Background jobs in task_queue table
- **Session Management**: User sessions (when implemented)
- **Rate Limiting**: API rate limits in rate_limits table

### Key Design Decisions
1. **No External Dependencies**: No Redis, MongoDB, or PostgreSQL
2. **Async Everything**: Using aiosqlite for async SQLite operations
3. **WAL Mode**: Enabled for better concurrency
4. **Local Development**: Designed to run on a single laptop
5. **Modular Structure**: Each component in separate modules

### Database Schema Highlights
- **23 Total Models**: Comprehensive competitive intelligence tracking
- **FTS5 Integration**: Full-text search capabilities
- **JSON Fields**: Flexible data storage for varied content
- **Optimized Indexes**: Performance-tuned for common queries
- **Relationship Mapping**: Proper foreign keys and cascades

## 🚀 Next Implementation Steps

Based on current progress (database layer complete):

### Immediate Priority: Core Application (Next 2-3 days)
```python
# 1. Create main.py - FastAPI application
# Key endpoints needed:
# - POST /api/search - Initiate competitive intelligence search
# - GET /api/search/{id} - Get search results
# - POST /api/analyze - Trigger LLM analysis
# - GET /api/insights - Get generated insights

# 2. Create config.py - Configuration management
# Required settings:
# - Database path and settings
# - API keys (Ollama, scraper proxies if needed)
# - Rate limiting configuration
# - Scraping delays and limits

# 3. Implement scrapers/google_scraper.py
# - Use requests + BeautifulSoup
# - Respect rate limits
# - Store results in SearchResult model
```

### Implementation Order (Tested Workflow)
1. **config.py** → Environment and settings (30 min)
2. **main.py** → FastAPI with basic routes (2 hrs)
3. **scrapers/google_scraper.py** → Primary data source (2 hrs)
4. **api/search_service.py** → Search orchestration (1 hr)
5. **Test end-to-end** → Search → Scrape → Store (1 hr)

## 📝 Important Implementation Notes

### When Implementing New Features
1. **Always use async/await** for database operations
2. **Use the existing db_ops functions** rather than raw SQL
3. **Follow the established pattern** in existing modules
4. **Add proper error handling** with try/except blocks
5. **Include logging** for debugging and monitoring
6. **Write tests** alongside new functionality

### Database Best Practices
- Use `DatabaseManager.get_session()` context manager for all DB operations
- Always handle transactions properly (commit/rollback)
- Use bulk operations for multiple inserts/updates
- Leverage existing indexes, don't create duplicates
- Clean up expired cache entries regularly

### Common Patterns in This Codebase
```python
# Initialize database manager
from src.core.database import DatabaseManager
from src.core.db_ops import DatabaseOperations

db_manager = DatabaseManager()
db_ops = DatabaseOperations()

# Async database operation pattern
async with db_manager.get_session() as session:
    # Create a new search
    search = await db_ops.create_search(session, {
        "query": "competitor analysis",
        "search_type": "competitor",
        "sources": ["google", "bing"]
    })

    # Bulk insert search results
    await db_ops.bulk_insert_search_results(
        session, search.id, results, chunk_size=100
    )

# Cache usage pattern
async with db_manager.get_session() as session:
    cached = await db_ops.get_cache(session, "my_key")
    if not cached or cached.get("expired"):
        result = await expensive_operation()
        await db_ops.set_cache(session, "my_key", result, ttl=3600)
    else:
        result = cached["value"]

# Task queue pattern
async with db_manager.get_session() as session:
    # Enqueue task
    task = await db_ops.enqueue_task(session, {
        "task_type": "scrape",
        "payload": {"url": "..."},
        "priority": 5
    })

    # Process tasks
    pending = await db_ops.get_pending_tasks(session, limit=10)
    for task in pending:
        # Process task...
        await db_ops.complete_task(session, task.id)
```

## ⚠️ Known Issues & Workarounds

1. **Windows Path Issues**: Use forward slashes or raw strings for paths
2. **Async Testing**: Use pytest-asyncio fixtures
3. **SQLite Locking**: Ensure WAL mode is enabled (already done)
4. **Import Errors**: Ensure src is in PYTHONPATH or use relative imports

## 🔍 Debugging Tips

1. **Database Queries**: Set `echo=True` in database.py for SQL logging
2. **Check Database**: Use DB Browser for SQLite to inspect ciap.db
3. **Test Connections**: Run `python src/core/database.py` to verify DB
4. **API Testing**: Use /docs endpoint for Swagger UI (when implemented)

## 📚 Key Files Reference

### Core Database Layer (Completed)
- `src/core/models.py`: All 23 database models defined
- `src/core/database.py`: Async database manager with connection pooling
- `src/core/db_ops.py`: Complete CRUD operations for all models
- `src/core/fts_setup.py`: Full-text search configuration
- `scripts/init_database.py`: Database initialization and setup
- `tests/test_database.py`: Comprehensive test suite

### Module Implementation Guides
- `Mod_01_Database.md`: Database infrastructure (COMPLETED)
- `Mod_02_Config.md`: Configuration management guide
- `Mod_03_Cache.md`: Caching system design
- `Mod_04_Queue.md`: Task queue implementation
- `Mod_05_Scraper.md`: Web scraping modules
- `Mod_06_Processor.md`: Data processing pipeline
- `Mod_07_Analyzer.md`: LLM analysis integration
- `Mod_08_API.md`: FastAPI endpoints
- `Mod_09_Export.md`: Export service design
- `Mod_10_Scheduler.md`: Task scheduling

### Project Documentation
- `README.md`: Project overview and quick start
- `ProjectDetail.md`: Full academic project specification
- `progress.md`: Current implementation status
- `.env.example`: Environment variable template

## 🎓 Academic Requirements

This is an academic project with specific requirements:
- Must demonstrate working MVP within 12 weeks
- Focus on practical implementation over perfect architecture
- Document all design decisions and trade-offs
- Provide measurable performance metrics
- Include comprehensive testing suite

Remember: The goal is a working system that SMEs can actually use, not a perfect enterprise solution.

## 🚦 Quick Task Reference

### Starting a New Component
1. Check the relevant `Mod_*.md` file for detailed implementation guide
2. Create the directory structure if needed
3. Follow the async patterns established in existing code
4. Add tests alongside implementation
5. Update this CLAUDE.md when component is complete

### Common Tasks
```bash
# Check what's implemented
ls -la src/core/
python scripts/init_database.py --check-only

# Start implementing main.py
# 1. Copy structure from Mod_08_API.md
# 2. Use existing db_ops functions
# 3. Test with: python main.py

# Add a new scraper
# 1. Create scrapers/ directory
# 2. Follow pattern in Mod_05_Scraper.md
# 3. Use db_ops.create_search_result()

# Debug database issues
python -c "from src.core.database import DatabaseManager; import asyncio; db=DatabaseManager(); asyncio.run(db.health_check())"
```

### Windows-Specific Commands
```bash
# Activate virtual environment
F:\Project\CIAP\venv\Scripts\activate

# Run tests with output
python tests/test_database.py 2>&1 | more

# Check Python path issues
python -c "import sys; print('\n'.join(sys.path))"
```