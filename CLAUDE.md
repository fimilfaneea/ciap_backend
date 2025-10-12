# CIAP Project - Claude Reference Guide

> **Claude: This is my active M.Tech IT project that I'm currently implementing. Unless I say otherwise, assume all my questions and requests relate to building this system. I need practical, implementation-focused guidance that fits within my 12-week timeline.**

## 🎯 Quick Context

**Who I Am:** Fimil Faneea (24102371), M.Tech IT student
**What I'm Building:** Open-source Competitive Intelligence Automation Platform for SMEs
**Timeline:** 12 weeks (Weeks 1-12)
**Goal:** Create a cost-effective alternative to expensive CI tools like SEMrush/SimilarWeb
**Deployment:** Local laptop only (no Docker, no external services for MVP)

---

## 📍 Current Status

**Current Phase:** Week 1 - Foundation & Setup
**Last Updated:** October 12, 2025

### Completed Tasks
- [x] Requirements definition
- [x] Basic architecture design
- [x] Initial FastAPI setup with SQLite
- [x] Basic Google scraper implementation
- [x] LLM analyzer framework (OpenAI/Anthropic)
- [x] Database models created
- [x] Basic API endpoints working
- [x] Detailed implementation plan created

### In Progress
- [ ] Setting up Ollama for local LLM
- [ ] Creating SQLite-based task queue
- [ ] Enhancing project structure

### Next Immediate Steps
1. Install and configure Ollama with llama3.1:8b model
2. Create SQLite-based caching and task queue
3. Build modular scraper system
4. Implement Bing scraper

---

## 🛠️ Tech Stack (Finalized)

### Backend
- **Language:** Python 3.10+
- **Framework:** FastAPI ✅
- **Task Queue:** Custom SQLite-based queue (no Celery/Redis)
- **Scheduler:** APScheduler with SQLite backend

### Data Processing
- **Scrapers:** Google (done), Bing, Crawlee
- **LLM:** Ollama with llama3.1:8b (local, free) → OpenAI later if needed
- **ML Libraries:** scikit-learn for deduplication

### Infrastructure (All Local - Single Laptop)
- **Database:** SQLite for everything:
  - Main data storage
  - Cache tables
  - Task queue
  - Session management
- **Deployment:** Local laptop only
- **No External Services:** No Docker, No Redis, No cloud

### Frontend/Dashboards
- **Framework:** Skip initially (use API directly)
- **BI Tools:** Power BI integration via exports

---

## 💡 Simplified Architecture

### Why SQLite for Everything?
- **Single dependency** - No Redis/PostgreSQL/MongoDB needed
- **Zero configuration** - Works out of the box
- **Portable** - Single file database
- **Sufficient for MVP** - Handles thousands of records easily
- **Built-in with Python** - No installation required

### SQLite Usage
1. **Main Database** - All application data
2. **Cache Storage** - Temporary results with TTL
3. **Task Queue** - Background job management
4. **Session Store** - API session management
5. **File Storage** - BLOB storage for scraped content

---

## 📂 Simplified Project Structure

```
F:\Project\CIAP\
├── venv\                    # Python virtual environment
├── data\                    # Local data storage
│   ├── ciap.db             # Main SQLite database
│   ├── exports\            # Export files for Power BI
│   └── logs\               # Application logs
├── config\                  # Configuration files
│   ├── prompts\            # LLM prompt templates
│   └── user_agents.txt     # Rotating user agents
├── src\
│   ├── core\               # Core utilities
│   │   ├── config.py       # Configuration
│   │   ├── database.py     # SQLite models & connection
│   │   ├── cache.py        # SQLite caching
│   │   ├── queue.py        # SQLite task queue
│   │   └── logging.py      # Logging setup
│   ├── scrapers\           # Data collection
│   │   ├── base.py         # Base scraper interface
│   │   ├── google.py       # Google scraper
│   │   ├── bing.py         # Bing scraper
│   │   ├── crawlee_wrapper.py  # Deep crawler
│   │   └── manager.py      # Scraper orchestration
│   ├── processors\         # Data processing
│   │   ├── cleaner.py      # Text cleaning
│   │   ├── normalizer.py   # Data standardization
│   │   └── deduplicator.py # Duplicate detection
│   ├── analyzers\          # Analysis engines
│   │   ├── ollama_client.py # Local LLM
│   │   ├── sentiment.py    # Sentiment analysis
│   │   ├── competitor.py   # Competitor analysis
│   │   └── trends.py       # Trend detection
│   ├── services\           # Business logic
│   │   ├── search_service.py   # Search orchestration
│   │   ├── export_service.py   # Data export
│   │   └── scheduler.py        # Job scheduling
│   └── api\                # API layer
│       ├── routes.py       # All endpoints
│       └── middleware.py   # Auth & rate limiting
├── tests\                  # Test suite
├── scripts\                # Utility scripts
│   ├── setup.py           # One-click setup
│   └── demo.py            # Demo runner
├── main.py                # FastAPI app
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
└── CLAUDE.md             # This file
```

---

## 📅 Revised Implementation Plan (SQLite-Only)

### Week 1-2: Foundation & Setup ✅
**Goal:** Simplified local setup with SQLite

#### Day 1-2: Environment Setup (Simplified!)
```bash
# Install Python dependencies only
cd F:\Project\CIAP
pip install fastapi uvicorn
pip install apscheduler  # For task scheduling
pip install playwright crawlee beautifulsoup4
pip install pandas scikit-learn
pip install python-docx openpyxl  # For exports
pip install aiofiles aiosqlite  # Async SQLite

# Install Ollama (only external tool needed)
# Download from: https://ollama.com/download/windows
ollama pull llama3.1:8b  # Balanced performance and capability for local development
```

#### Day 3-4: Enhanced SQLite Schema
```python
# New tables for extended functionality
class Cache(Base):
    __tablename__ = "cache"
    key = Column(String, primary_key=True)
    value = Column(Text)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class TaskQueue(Base):
    __tablename__ = "task_queue"
    id = Column(Integer, primary_key=True)
    task_type = Column(String)
    payload = Column(JSON)
    status = Column(String)  # pending, processing, completed, failed
    priority = Column(Integer, default=5)
    scheduled_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)

class ScrapingJob(Base):
    __tablename__ = "scraping_jobs"
    id = Column(Integer, primary_key=True)
    search_id = Column(Integer, ForeignKey("searches.id"))
    scraper_name = Column(String)
    status = Column(String)
    results_count = Column(Integer)
    error_log = Column(Text)
```

#### Day 5-6: SQLite-Based Queue System
```python
# src/core/queue.py - Simple task queue using SQLite
class SQLiteQueue:
    def __init__(self, db_path="data/ciap.db"):
        self.db_path = db_path

    async def enqueue(self, task_type: str, payload: dict, priority=5):
        """Add task to queue"""
        # Insert into task_queue table

    async def dequeue(self):
        """Get next task from queue"""
        # SELECT with ORDER BY priority, scheduled_at

    async def process_tasks(self):
        """Background task processor"""
        while True:
            task = await self.dequeue()
            if task:
                await self.execute_task(task)
            await asyncio.sleep(1)
```

#### Day 7: Setup Script & Testing
```python
# scripts/setup.py - One-click setup
def setup_environment():
    """Complete setup in one command"""
    # 1. Check Python version
    # 2. Install requirements
    # 3. Initialize SQLite database
    # 4. Create necessary tables
    # 5. Test Ollama connection
    # 6. Create directories
    # 7. Run initial tests
    print("✅ Setup complete! No external services needed!")
```

### Week 3-4: Scraping System 🔍
**Goal:** Multi-source scraping without external dependencies

#### Day 8-10: Scraper Manager with SQLite Queue
```python
# No Celery needed - use async tasks with SQLite
class ScraperManager:
    def __init__(self):
        self.queue = SQLiteQueue()
        self.scrapers = {
            'google': GoogleScraper(),
            'bing': BingScraper(),
        }

    async def schedule_scraping(self, query: str):
        """Add scraping job to SQLite queue"""
        await self.queue.enqueue(
            task_type="scrape",
            payload={"query": query, "sources": ["google", "bing"]}
        )
```

#### Day 11-14: Scrapers with Built-in Rate Limiting
```python
# src/scrapers/base.py
class BaseScraper:
    def __init__(self):
        self.rate_limit_table = "scraper_rate_limits"

    async def check_rate_limit(self):
        """Use SQLite to track request times"""
        # Query last request time from SQLite
        # Enforce delays based on stored timestamps
```

### Week 5: Data Processing 🔄
**Goal:** Process data using SQLite for everything

#### Day 15-17: Processing Pipeline
```python
# All processing with SQLite storage
class DataProcessor:
    def __init__(self, db_path="data/ciap.db"):
        self.db = db_path

    async def process_batch(self, data: List[Dict]):
        """Process and store in SQLite"""
        # Clean data
        # Normalize
        # Store in SQLite
        # Update processing status
```

#### Day 18-19: SQLite-Based Deduplication
```python
# Use SQLite for similarity matching
class Deduplicator:
    async def find_duplicates(self):
        """Use SQLite FTS5 for text similarity"""
        # CREATE VIRTUAL TABLE USING fts5
        # Use SQLite's full-text search for deduplication
```

### Week 6-7: LLM Analysis 🤖
**Goal:** Ollama + SQLite caching

#### Day 20-22: Ollama with SQLite Cache
```python
# src/analyzers/ollama_client.py
class OllamaClient:
    def __init__(self):
        self.cache = SQLiteCache()

    async def analyze(self, text: str, analysis_type: str):
        # Check SQLite cache first
        cached = await self.cache.get(text_hash)
        if cached:
            return cached

        # Call Ollama
        result = await self.call_ollama(text)

        # Store in SQLite cache
        await self.cache.set(text_hash, result, ttl=3600)
        return result
```

#### Day 23-28: Analysis Modules
- All analysis results stored in SQLite
- Batch processing to optimize Ollama calls
- Results cached to avoid re-analysis

### Week 8: Services & Scheduling 💼
**Goal:** Background tasks without Celery/Redis

#### Day 29-31: APScheduler with SQLite
```python
# src/services/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

class TaskScheduler:
    def __init__(self):
        jobstores = {
            'default': SQLAlchemyJobStore(url='sqlite:///data/ciap.db')
        }
        self.scheduler = AsyncIOScheduler(jobstores=jobstores)

    def schedule_scraping(self, search_id: int):
        """Schedule scraping job"""
        self.scheduler.add_job(
            func=scrape_task,
            trigger="interval",
            minutes=30,
            id=f"scrape_{search_id}"
        )
```

### Week 9: API Enhancement 🔌
**Goal:** Full-featured API with SQLite

#### Day 32-36: Enhanced Endpoints
```python
# All session management in SQLite
@app.post("/api/search")
async def create_search(request: SearchRequest):
    # Store in SQLite
    # Queue processing tasks in SQLite
    # Return search ID

@app.get("/api/search/{id}/progress")
async def get_progress(id: int):
    # Query SQLite for task status
    # Return real-time progress
```

#### Day 37-38: WebSocket with SQLite
```python
# Use SQLite for WebSocket session management
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # Store connection info in SQLite
    # Push updates from SQLite changes
```

### Week 10: Export & Visualization 📊
**Goal:** Power BI integration

#### Day 39-42: Export Service
```python
# Direct SQLite to Power BI formats
class ExportService:
    def export_to_powerbi(self, search_id: int):
        """Export SQLite data to Power BI format"""
        # Query SQLite
        # Format for Power BI
        # Save to exports folder
```

### Week 11: Testing & Optimization ⚡
**Goal:** Optimize SQLite performance

#### Day 43-45: SQLite Optimization
```sql
-- Add indexes for performance
CREATE INDEX idx_searches_status ON searches(status);
CREATE INDEX idx_search_results_search_id ON search_results(search_id);
CREATE INDEX idx_cache_expires ON cache(expires_at);

-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;

-- Optimize SQLite settings
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;
```

#### Day 46-49: Testing & Security
- Test SQLite under load
- Implement SQL injection prevention
- Add API authentication
- Rate limiting using SQLite

### Week 12: Documentation & Demo 📚
**Goal:** Complete project submission

#### Day 50-56: Same as before
- Documentation
- Demo preparation
- Final testing
- Submission package

---

## 🚀 Simplified Quick Start

```bash
# 1. Setup (one time)
cd F:\Project\CIAP
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Initialize database
python scripts/setup.py

# 3. Start Ollama (only external service)
ollama serve

# 4. Run application
python main.py

# That's it! No Redis, no Docker, no PostgreSQL needed!
```

---

## 📊 SQLite Performance Targets

### What SQLite Can Handle (More than enough for MVP)
- **Searches:** 10,000+ records
- **Search Results:** 100,000+ records
- **Cache Entries:** 50,000+ records
- **Concurrent Reads:** Unlimited
- **Concurrent Writes:** Sequential (but fast enough for single user)
- **Database Size:** Up to 281 TB (way more than needed)

### Performance Optimizations
```python
# Connection pool for SQLite
DATABASE_URL = "sqlite:///data/ciap.db?check_same_thread=False"

# Enable WAL mode for better concurrency
conn.execute("PRAGMA journal_mode=WAL")

# In-memory cache for hot data
conn.execute("PRAGMA cache_size=10000")
```

---

## 🔧 SQLite-Specific Solutions

### Task Queue without Celery
```python
# Simple but effective
async def background_worker():
    """Runs in background, processes tasks from SQLite"""
    while True:
        task = await get_next_task_from_sqlite()
        if task:
            await process_task(task)
        await asyncio.sleep(1)

# Start with FastAPI
@app.on_event("startup")
async def startup():
    asyncio.create_task(background_worker())
```

### Caching without Redis
```python
class SQLiteCache:
    async def get(self, key: str):
        """Get from cache if not expired"""
        row = await db.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, datetime.now())
        )
        return row[0] if row else None

    async def set(self, key: str, value: str, ttl: int = 3600):
        """Set cache with expiration"""
        expires_at = datetime.now() + timedelta(seconds=ttl)
        await db.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
            (key, value, expires_at)
        )
```

### Session Management without Redis
```python
# Store sessions in SQLite
class SessionStore:
    async def create_session(self, user_id: str):
        session_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (session_id, user_id, datetime.now())
        )
        return session_id
```

---

## 💡 Benefits of SQLite-Only Approach

### Development Benefits
1. **Zero configuration** - No services to install/manage
2. **Instant setup** - Run immediately on any Windows laptop
3. **Easy backup** - Just copy one .db file
4. **Simple deployment** - Copy folder and run
5. **No dependency conflicts** - SQLite is built into Python

### Technical Benefits
1. **ACID compliant** - Data integrity guaranteed
2. **Fast for reads** - Perfect for caching
3. **Full SQL support** - Complex queries possible
4. **Full-text search** - Built-in FTS5
5. **JSON support** - Store complex data easily

### Project Benefits
1. **Meets all requirements** - Sufficient for MVP
2. **Easier to debug** - Everything in one place
3. **Lower complexity** - Less moving parts
4. **Free forever** - No service costs
5. **Portable** - Run on any laptop

---

## 📈 Weekly Progress Tracking

### Week 1 (Oct 12-18, 2025)
- [ ] Environment setup (Python packages only)
- [ ] Ollama installed and working with llama3.1:8b model
- [ ] SQLite schema with all tables
- [ ] Basic task queue implemented
- [ ] Cache system working

### Week 2 (Oct 19-25, 2025)
- [ ] Scraper interface complete
- [ ] SQLite-based configuration
- [ ] Background task processor
- [ ] Setup script created

---

## 📝 Updated Decisions

### Architecture Decisions
- ✅ **SQLite for everything** - Database, cache, queue, sessions
- ✅ **No external services** - Only Ollama for LLM
- ✅ **Async SQLite** - Using aiosqlite for performance
- ✅ **APScheduler** - For scheduled tasks (SQLite backend)
- ✅ **Single process** - No need for multiple workers

### This Simplifies
- **No Redis installation/configuration**
- **No Celery complexity**
- **No Docker needed**
- **No service management**
- **Single database file to backup**

---

**Version:** 3.0 (SQLite-Only Architecture)
**Last Updated:** October 12, 2025
**Next Review:** End of Week 1