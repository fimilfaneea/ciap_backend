# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Project Context

**CIAP (Competitive Intelligence Automation Platform)** is an active M.Tech IT project for building an open-source competitive intelligence solution for SMEs. This is a 12-week implementation project with specific academic requirements.

**Student:** Fimil Faneea (24102371), M.Tech IT
**Timeline:** 12 weeks total, currently in Week 1
**Goal:** Create a cost-effective alternative to expensive CI tools using SQLite-only architecture

**IMPORTANT:** Unless specified otherwise, assume all questions relate to implementing this system within the 12-week timeline.

## 📍 Current Implementation Status

### Completed Components
- ✅ **Database Models** (src/core/models.py):
  - All 23 models defined including Search, Product, Competitor, etc.
  - SQLite with async support via aiosqlite
  - Comprehensive relationships and indexes

- ✅ **Database Manager** (src/core/database.py):
  - Async database operations with connection pooling
  - WAL mode enabled for concurrency
  - Performance optimizations implemented
  - Health check and stats functionality

- ✅ **FTS5 Setup** (src/core/fts_setup.py):
  - Full-text search configuration
  - Virtual tables for search optimization

- ✅ **Database Operations** (src/core/db_ops.py):
  - Comprehensive CRUD operations for all models
  - Bulk operations and batch processing
  - Cache management and task queue operations
  - Price history tracking

### In Progress
- 🔄 **Scrapers**: Need implementation (Google scraper mentioned but not found)
- 🔄 **LLM Integration**: Ollama setup pending
- 🔄 **API Layer**: FastAPI endpoints need implementation
- 🔄 **Task Queue**: SQLite-based queue system needs completion

### Not Started
- ❌ **Main FastAPI application** (main.py)
- ❌ **Configuration module** (config.py)
- ❌ **Scraper implementations** (scrapers/)
- ❌ **Analysis modules** (analyzers/)
- ❌ **Export service** (services/export_service.py)
- ❌ **Scheduler implementation** (services/scheduler.py)

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
# Initialize database (run from project root)
python -c "from src.core.database import db_manager; import asyncio; asyncio.run(db_manager.initialize())"

# Test database connection
python src/core/database.py

# Run database operations tests
python src/core/db_ops.py
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
# Run all tests
pytest

# Run specific test file
pytest tests/test_database.py

# Run with coverage
pytest --cov=src --cov-report=html
```

## 🏗 Architecture Overview

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

Based on current progress and the 12-week timeline:

### Immediate Priorities (Week 1-2)
1. **Create main.py**: FastAPI application with basic endpoints
2. **Implement config.py**: Configuration management with .env support
3. **Build Google scraper**: Basic search result scraping
4. **Setup Ollama integration**: Local LLM for analysis

### Week 3-4 Focus
1. **Implement task queue processor**: Background job handling
2. **Add Bing scraper**: Additional data source
3. **Create API routes**: Search, analysis, and export endpoints
4. **Basic caching layer**: Implement cache operations

## 📝 Important Implementation Notes

### When Implementing New Features
1. **Always use async/await** for database operations
2. **Use the existing db_ops functions** rather than raw SQL
3. **Follow the established pattern** in existing modules
4. **Add proper error handling** with try/except blocks
5. **Include logging** for debugging and monitoring
6. **Write tests** alongside new functionality

### Database Best Practices
- Use `db_manager.get_session()` context manager for all DB operations
- Always handle transactions properly (commit/rollback)
- Use bulk operations for multiple inserts/updates
- Leverage existing indexes, don't create duplicates
- Clean up expired cache entries regularly

### Common Patterns in This Codebase
```python
# Async database operation pattern
async with db_manager.get_session() as session:
    result = await db_ops.create_search(session, search_data)
    # Session auto-commits on success, rollbacks on error

# Bulk insert pattern
await db_ops.bulk_insert_results(session, results, chunk_size=100)

# Cache usage pattern
cached = await db_ops.get_cache(session, key)
if not cached:
    result = await expensive_operation()
    await db_ops.set_cache(session, key, result, ttl=3600)
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

### Core Database Layer
- `src/core/models.py`: All database models (435 lines)
- `src/core/database.py`: Database manager with async support (346 lines)
- `src/core/db_ops.py`: CRUD and business operations (926 lines)
- `src/core/fts_setup.py`: Full-text search configuration (80 lines)

### Configuration & Setup
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (API keys, settings)
- `progress.md`: Detailed implementation progress tracker

### Documentation
- `README.md`: Project overview and setup instructions
- `Mod_*.md`: Detailed module implementation plans
- `ProjectDetail.md`: Full project specification

## 🎓 Academic Requirements

This is an academic project with specific requirements:
- Must demonstrate working MVP within 12 weeks
- Focus on practical implementation over perfect architecture
- Document all design decisions and trade-offs
- Provide measurable performance metrics
- Include comprehensive testing suite

Remember: The goal is a working system that SMEs can actually use, not a perfect enterprise solution.