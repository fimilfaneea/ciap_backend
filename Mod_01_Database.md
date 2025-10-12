# Module 1: Database Infrastructure

## Overview
**Purpose:** Central SQLite database management with async support for all CIAP operations.

**Responsibilities:**
- Database connection management
- Model definitions (ORM)
- Transaction handling
- Connection pooling
- WAL mode for concurrent access
- Database initialization

**Development Time:** 2 days (Week 1, Day 1-2)

---

## Interface Specification

### Input
```python
# Configuration parameters
DATABASE_URL = "sqlite:///data/ciap.db"
POOL_SIZE = 5
ENABLE_WAL = True
```

### Output
```python
# Database session objects
async def get_db() -> AsyncSession
# Model classes for ORM
class Search, SearchResult, Cache, TaskQueue, etc.
```

---

## Dependencies

### External
```txt
sqlalchemy==2.0.23
aiosqlite==0.19.0
alembic==1.12.1  # For migrations (optional)
```

### Internal
- None (foundational module)

---

## Implementation Guide

### Step 1: Database Models (`src/core/models.py`)

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Search(Base):
    """Main search queries and metadata"""
    __tablename__ = "searches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String(500), nullable=False)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    sources = Column(JSON)  # ["google", "bing"]
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    user_id = Column(String(100), nullable=True)  # For multi-user support later

class SearchResult(Base):
    """Scraped search results"""
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("searches.id"))
    source = Column(String(50))  # google, bing, etc.
    title = Column(String(500))
    snippet = Column(Text)
    url = Column(String(1000))
    position = Column(Integer)  # Ranking position
    scraped_at = Column(DateTime, default=func.now())

    # Analysis results
    sentiment_score = Column(Float, nullable=True)
    competitor_mentioned = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)

class Cache(Base):
    """SQLite-based cache with TTL"""
    __tablename__ = "cache"

    key = Column(String(255), primary_key=True)
    value = Column(Text)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())

class TaskQueue(Base):
    """Background task queue"""
    __tablename__ = "task_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(100))  # scrape, analyze, export
    payload = Column(JSON)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    priority = Column(Integer, default=5)  # 1=highest, 10=lowest
    scheduled_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

class ScrapingJob(Base):
    """Track individual scraping jobs"""
    __tablename__ = "scraping_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("searches.id"))
    scraper_name = Column(String(50))
    status = Column(String(50), default="pending")
    results_count = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_log = Column(Text, nullable=True)

class RateLimit(Base):
    """Track rate limits for scrapers"""
    __tablename__ = "rate_limits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scraper_name = Column(String(50))
    last_request_at = Column(DateTime)
    request_count = Column(Integer, default=0)
    reset_at = Column(DateTime)
```

### Step 2: Database Connection (`src/core/database.py`)

```python
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text, event
from contextlib import asynccontextmanager
import aiosqlite
from .models import Base

class DatabaseManager:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///data/ciap.db"):
        self.database_url = database_url
        self.engine = None
        self.async_session = None

    async def initialize(self):
        """Initialize database connection and create tables"""
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections are alive
        )

        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Enable WAL mode for better concurrency
        async with self.engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA cache_size=10000"))
            await conn.execute(text("PRAGMA temp_store=MEMORY"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))

        # Create all tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create indexes for better performance
        await self._create_indexes()

    async def _create_indexes(self):
        """Create database indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_searches_status ON searches(status)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_search_id ON search_results(search_id)",
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status, priority)",
            "CREATE INDEX IF NOT EXISTS idx_rate_limits_scraper ON rate_limits(scraper_name)",
        ]

        async with self.engine.begin() as conn:
            for index_sql in indexes:
                await conn.execute(text(index_sql))

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with context manager"""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()

    async def health_check(self) -> bool:
        """Check if database is accessible"""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception:
            return False

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with db_manager.get_session() as session:
        yield session
```

### Step 3: Database Operations (`src/core/db_ops.py`)

```python
from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from .models import Search, SearchResult, TaskQueue, Cache, ScrapingJob
from .database import db_manager

class DatabaseOperations:
    """Common database operations"""

    # Search Operations
    @staticmethod
    async def create_search(session: AsyncSession, query: str, sources: List[str], user_id: Optional[str] = None) -> Search:
        """Create new search record"""
        search = Search(
            query=query,
            sources=sources,
            user_id=user_id,
            status="pending"
        )
        session.add(search)
        await session.flush()  # Get the ID without committing
        return search

    @staticmethod
    async def get_search(session: AsyncSession, search_id: int) -> Optional[Search]:
        """Get search by ID"""
        result = await session.execute(
            select(Search).where(Search.id == search_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_search_status(session: AsyncSession, search_id: int, status: str, error: Optional[str] = None):
        """Update search status"""
        stmt = update(Search).where(Search.id == search_id).values(
            status=status,
            updated_at=datetime.utcnow(),
            completed_at=datetime.utcnow() if status in ["completed", "failed"] else None,
            error_message=error
        )
        await session.execute(stmt)

    # Search Results Operations
    @staticmethod
    async def bulk_insert_results(session: AsyncSession, results: List[Dict[str, Any]]):
        """Bulk insert search results"""
        # Convert dicts to SearchResult objects
        result_objects = [SearchResult(**result) for result in results]
        session.add_all(result_objects)
        await session.flush()

    @staticmethod
    async def get_search_results(session: AsyncSession, search_id: int, limit: int = 100) -> List[SearchResult]:
        """Get results for a search"""
        result = await session.execute(
            select(SearchResult)
            .where(SearchResult.search_id == search_id)
            .order_by(SearchResult.position)
            .limit(limit)
        )
        return result.scalars().all()

    # Task Queue Operations
    @staticmethod
    async def enqueue_task(session: AsyncSession, task_type: str, payload: Dict, priority: int = 5) -> TaskQueue:
        """Add task to queue"""
        task = TaskQueue(
            task_type=task_type,
            payload=payload,
            priority=priority,
            status="pending"
        )
        session.add(task)
        await session.flush()
        return task

    @staticmethod
    async def dequeue_task(session: AsyncSession) -> Optional[TaskQueue]:
        """Get next pending task by priority"""
        result = await session.execute(
            select(TaskQueue)
            .where(TaskQueue.status == "pending")
            .order_by(TaskQueue.priority, TaskQueue.scheduled_at)
            .limit(1)
            .with_for_update(skip_locked=True)  # Skip locked rows
        )
        task = result.scalar_one_or_none()

        if task:
            task.status = "processing"
            task.started_at = datetime.utcnow()
            await session.flush()

        return task

    @staticmethod
    async def complete_task(session: AsyncSession, task_id: int, status: str = "completed", error: Optional[str] = None):
        """Mark task as completed or failed"""
        stmt = update(TaskQueue).where(TaskQueue.id == task_id).values(
            status=status,
            completed_at=datetime.utcnow(),
            error_message=error
        )
        await session.execute(stmt)

    # Cache Operations
    @staticmethod
    async def get_cache(session: AsyncSession, key: str) -> Optional[str]:
        """Get cached value if not expired"""
        result = await session.execute(
            select(Cache)
            .where(and_(
                Cache.key == key,
                Cache.expires_at > datetime.utcnow()
            ))
        )
        cache_entry = result.scalar_one_or_none()
        return cache_entry.value if cache_entry else None

    @staticmethod
    async def set_cache(session: AsyncSession, key: str, value: str, expires_at: datetime):
        """Set cache value with expiration"""
        # Use INSERT OR REPLACE semantics
        await session.execute(
            delete(Cache).where(Cache.key == key)
        )

        cache_entry = Cache(
            key=key,
            value=value,
            expires_at=expires_at
        )
        session.add(cache_entry)
        await session.flush()

    @staticmethod
    async def cleanup_expired_cache(session: AsyncSession) -> int:
        """Remove expired cache entries"""
        result = await session.execute(
            delete(Cache).where(Cache.expires_at < datetime.utcnow())
        )
        return result.rowcount

    # Statistics
    @staticmethod
    async def get_database_stats(session: AsyncSession) -> Dict[str, int]:
        """Get database statistics"""
        stats = {}

        # Count records in each table
        tables = [Search, SearchResult, TaskQueue, Cache, ScrapingJob]
        for table in tables:
            result = await session.execute(select(func.count()).select_from(table))
            stats[table.__tablename__] = result.scalar()

        return stats
```

---

## Testing Guide

### Unit Tests (`tests/test_database.py`)

```python
import pytest
import asyncio
from datetime import datetime, timedelta
from src.core.database import DatabaseManager, db_manager
from src.core.db_ops import DatabaseOperations
from src.core.models import Base

@pytest.fixture
async def test_db():
    """Create test database"""
    manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await manager.initialize()
    yield manager
    await manager.close()

@pytest.mark.asyncio
async def test_database_initialization(test_db):
    """Test database initializes correctly"""
    assert await test_db.health_check()

@pytest.mark.asyncio
async def test_search_operations(test_db):
    """Test search CRUD operations"""
    async with test_db.get_session() as session:
        # Create search
        search = await DatabaseOperations.create_search(
            session,
            "test query",
            ["google", "bing"]
        )
        assert search.id is not None
        assert search.status == "pending"

        # Get search
        retrieved = await DatabaseOperations.get_search(session, search.id)
        assert retrieved.query == "test query"

        # Update status
        await DatabaseOperations.update_search_status(
            session,
            search.id,
            "completed"
        )
        await session.commit()

        # Verify update
        updated = await DatabaseOperations.get_search(session, search.id)
        assert updated.status == "completed"

@pytest.mark.asyncio
async def test_task_queue(test_db):
    """Test task queue operations"""
    async with test_db.get_session() as session:
        # Enqueue tasks with different priorities
        task1 = await DatabaseOperations.enqueue_task(
            session, "scrape", {"query": "test1"}, priority=5
        )
        task2 = await DatabaseOperations.enqueue_task(
            session, "scrape", {"query": "test2"}, priority=1
        )
        await session.commit()

        # Dequeue should get highest priority first
        next_task = await DatabaseOperations.dequeue_task(session)
        assert next_task.id == task2.id
        assert next_task.status == "processing"

        # Complete task
        await DatabaseOperations.complete_task(session, next_task.id)
        await session.commit()

        # Verify completion
        async with test_db.get_session() as new_session:
            completed = await new_session.get(TaskQueue, next_task.id)
            assert completed.status == "completed"

@pytest.mark.asyncio
async def test_cache_operations(test_db):
    """Test cache with TTL"""
    async with test_db.get_session() as session:
        # Set cache
        expires = datetime.utcnow() + timedelta(hours=1)
        await DatabaseOperations.set_cache(
            session, "test_key", "test_value", expires
        )
        await session.commit()

        # Get cache
        value = await DatabaseOperations.get_cache(session, "test_key")
        assert value == "test_value"

        # Test expired cache
        expired = datetime.utcnow() - timedelta(hours=1)
        await DatabaseOperations.set_cache(
            session, "expired_key", "expired_value", expired
        )
        await session.commit()

        # Should return None for expired
        value = await DatabaseOperations.get_cache(session, "expired_key")
        assert value is None

        # Cleanup expired
        deleted = await DatabaseOperations.cleanup_expired_cache(session)
        assert deleted == 1

@pytest.mark.asyncio
async def test_concurrent_access(test_db):
    """Test concurrent database access"""
    async def worker(worker_id: int):
        async with test_db.get_session() as session:
            search = await DatabaseOperations.create_search(
                session, f"query_{worker_id}", ["google"]
            )
            await session.commit()
            return search.id

    # Run 10 concurrent workers
    tasks = [worker(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # All should have unique IDs
    assert len(set(results)) == 10
```

---

## Integration Points

### With Config Module
```python
from src.core.config import settings
db_manager = DatabaseManager(settings.DATABASE_URL)
```

### With Task Queue Module
```python
from src.core.database import get_db
from src.core.db_ops import DatabaseOperations

async def process_queue():
    async for db in get_db():
        task = await DatabaseOperations.dequeue_task(db)
        if task:
            # Process task
            pass
```

### With API Module
```python
from fastapi import Depends
from src.core.database import get_db

@app.post("/api/search")
async def create_search(query: str, db: AsyncSession = Depends(get_db)):
    search = await DatabaseOperations.create_search(db, query, ["google"])
    await db.commit()
    return {"search_id": search.id}
```

---

## Common Issues & Solutions

### Issue 1: Database Locked Error
**Problem:** SQLite returns "database is locked" error
**Solution:** Enable WAL mode (already in initialization)
```python
await conn.execute(text("PRAGMA journal_mode=WAL"))
```

### Issue 2: Slow Queries
**Problem:** Queries taking too long
**Solution:** Add indexes (already in `_create_indexes`)
```sql
CREATE INDEX idx_search_results_search_id ON search_results(search_id);
```

### Issue 3: Connection Pool Exhaustion
**Problem:** Too many concurrent connections
**Solution:** Adjust pool size or use connection limiting
```python
engine = create_async_engine(url, pool_size=10, max_overflow=20)
```

### Issue 4: Memory Usage with Large Results
**Problem:** Loading too many results into memory
**Solution:** Use pagination and streaming
```python
# Use limit and offset
results = await session.execute(
    select(SearchResult).limit(100).offset(page * 100)
)
```

### Issue 5: Test Database Cleanup
**Problem:** Test data persisting between tests
**Solution:** Use in-memory database for tests
```python
test_db_url = "sqlite+aiosqlite:///:memory:"
```

---

## Performance Optimization

### 1. Enable WAL Mode
```sql
PRAGMA journal_mode=WAL;  -- Better concurrent access
```

### 2. Optimize Cache Size
```sql
PRAGMA cache_size=10000;  -- 10MB cache
PRAGMA temp_store=MEMORY;  -- Use RAM for temp tables
```

### 3. Use Bulk Operations
```python
# Instead of individual inserts
session.add_all(objects)  # Bulk insert

# Bulk update
await session.execute(
    update(Table).where(Table.id.in_(ids)).values(status="done")
)
```

### 4. Connection Pooling
```python
# Reuse connections
async_session = async_sessionmaker(engine, expire_on_commit=False)
```

### 5. Index Strategy
```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_searches_status ON searches(status);
CREATE INDEX idx_task_queue_status_priority ON task_queue(status, priority);
```

---

## Module Checklist

- [ ] Database models defined
- [ ] Connection manager implemented
- [ ] WAL mode enabled
- [ ] Indexes created
- [ ] CRUD operations working
- [ ] Task queue operations tested
- [ ] Cache with TTL working
- [ ] Unit tests passing
- [ ] Concurrent access tested
- [ ] Documentation complete

---

## Next Steps
After completing this module, move to:
1. **Module 2: Configuration** - Load database URL and settings
2. **Module 3: Cache** - Build on top of cache table
3. **Module 4: Task Queue** - Use queue table for background tasks