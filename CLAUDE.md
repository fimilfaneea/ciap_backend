# CIAP Project - Claude Reference Guide

> **Claude: This is my active M.Tech IT project that I'm currently implementing. Unless I say otherwise, assume all my questions and requests relate to building this system. I need practical, implementation-focused guidance that fits within my 12-week timeline.**

## 🎯 Quick Context

**Who I Am:** Fimil Faneea (24102371), M.Tech IT student
**What I'm Building:** Open-source Competitive Intelligence Automation Platform for SMEs
**Timeline:** 12 weeks (Weeks 1-12)
**Goal:** Create a cost-effective alternative to expensive CI tools like SEMrush/SimilarWeb
**Deployment:** Local laptop only (no Docker, no cloud for MVP)

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

### In Progress
- [ ] Setting up Ollama for local LLM
- [ ] Installing Redis for Windows
- [ ] Enhancing project structure

### Next Immediate Steps
1. Install and configure Ollama with llama2/mistral model
2. Set up Redis for caching and Celery
3. Create modular scraper system
4. Build Bing scraper

---

## 🛠️ Tech Stack (Finalized)

### Backend
- **Language:** Python 3.10+
- **Framework:** FastAPI ✅
- **Task Queue:** Celery (with Redis)
- **Message Broker:** Redis (simpler than RabbitMQ)

### Data Processing
- **Scrapers:** Google (done), Bing, Crawlee
- **LLM:** Ollama (local, free) → OpenAI later
- **ML Libraries:** scikit-learn for deduplication

### Infrastructure (All Local)
- **Database:** SQLite (keep it simple for MVP)
- **Cache:** Redis (Windows version)
- **Deployment:** Local laptop only
- **No Docker:** Direct installation

### Frontend/Dashboards
- **Framework:** Skip initially (use API directly)
- **BI Tools:** Power BI integration via exports

---

## 💡 Preferences & Constraints

### Budget
- **Cloud Costs:** None - local only
- **API Costs:** Free tier only (Ollama locally)
- **Target:** $0 for MVP

### Development Environment
- **OS:** Windows
- **RAM:** Available on laptop
- **Storage:** Local SSD
- **No Docker/Kubernetes**

### Development Approach
- **Priority:** Working MVP > Perfect code
- **Testing:** Manual testing + basic unit tests
- **Documentation:** As-needed for project submission

---

## 📂 Project Structure

```
F:\Project\CIAP\
├── venv\                    # Python virtual environment
├── data\                    # Local data storage
│   ├── scraped\            # Raw scraped data
│   ├── processed\          # Cleaned data
│   └── exports\            # Export files for Power BI
├── logs\                    # Application logs
├── config\                  # Configuration files
│   ├── prompts\            # LLM prompt templates
│   └── scrapers\           # Scraper configurations
├── src\
│   ├── core\               # Core utilities
│   ├── scrapers\           # Data collection modules
│   ├── processors\         # Data processing pipeline
│   ├── analyzers\          # LLM and analysis
│   ├── services\           # Business logic
│   ├── api\                # API endpoints
│   └── tasks\              # Background tasks
├── tests\                  # Test suite
├── scripts\                # Utility scripts
├── static\                 # Static files
├── templates\              # HTML templates
├── main.py                 # FastAPI app entry
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
└── CLAUDE.md              # This file
```

---

## 📅 Detailed Implementation Plan

### Week 1-2: Foundation & Setup ✅
**Goal:** Get everything running locally

#### Day 1-2: Environment Setup
```bash
# Install core dependencies
cd F:\Project\CIAP
pip install fastapi uvicorn celery redis
pip install playwright crawlee beautifulsoup4
pip install pandas scikit-learn
pip install python-docx openpyxl  # For exports

# Install Redis for Windows
# Download from: https://github.com/microsoftarchive/redis/releases

# Install Ollama
# Download from: https://ollama.com/download/windows
ollama pull llama2  # or mistral for lighter model
```

#### Day 3-4: Database Schema Enhancement
- Add new models: ScrapingJob, DataSource, Alert
- Create migration scripts
- Set up database utilities

#### Day 5-6: Core Module Structure
- Create base scraper interface
- Set up configuration management with Pydantic
- Implement logging system
- Create exception handling

#### Day 7: Setup Script & Testing
- Create one-click setup script
- Test all components
- Document setup process

### Week 3-4: Enhanced Scraping System 🔍
**Goal:** Multi-source scraping with fallbacks

#### Day 8-10: Scraper Interface & Manager
- Abstract base scraper class
- Scraper manager for orchestration
- Rate limiting implementation
- Error handling and retries

#### Day 11-12: Bing Scraper
- Implement Bing search scraping
- Test with various queries
- Add to scraper manager

#### Day 13-14: Crawlee Integration
- Deep website crawling
- JavaScript rendering support
- Content extraction

### Week 5: Data Processing Pipeline 🔄
**Goal:** Clean, normalize, and enrich data

#### Day 15-17: Data Processors
- HTML cleaning and text extraction
- Data normalization (URLs, dates, entities)
- Metadata extraction

#### Day 18-19: Deduplication System
- TF-IDF similarity detection
- Fuzzy matching for near-duplicates
- Efficient batch processing

#### Day 20-21: Content Enrichment
- Extract contact information
- Identify pricing data
- Company/product recognition

### Week 6-7: LLM Analysis Engine 🤖
**Goal:** Ollama integration for local AI

#### Day 22-24: Ollama Integration
- Set up Ollama client
- Test with different models
- Implement batch processing
- Create caching layer

#### Day 25-27: Analysis Modules
- Sentiment analysis
- Competitor identification
- SWOT analysis generation
- Trend detection

#### Day 28: Prompt Management
- Create prompt templates
- A/B testing framework
- Version control for prompts

### Week 8: Service Layer & Business Logic 💼
**Goal:** Orchestrate all components

#### Day 29-31: Core Services
- Enhanced search service
- Analysis pipeline
- Competitor profiling
- Report generation

#### Day 32-33: Alert System
- Condition monitoring
- Notification system
- Scheduled checks

### Week 9: Enhanced API Layer 🔌
**Goal:** Production-ready API

#### Day 34-36: API Routes Enhancement
- Advanced search endpoints
- Real-time status updates
- Bulk operations
- Export endpoints

#### Day 37-38: WebSocket Support
- Real-time progress updates
- Live analysis streaming
- Dashboard data push

### Week 10: Visualization & Export 📊
**Goal:** Power BI integration

#### Day 39-41: Power BI Connector
- Data formatting for Power BI
- Export templates
- Automated report generation

#### Day 42: Multiple Export Formats
- Excel with formatting
- CSV for analysis
- JSON for APIs
- Word documents for reports

### Week 11: Testing & Optimization ⚡
**Goal:** Production readiness

#### Day 43-45: Comprehensive Testing
- Unit tests for all modules
- Integration testing
- Performance testing
- Load testing

#### Day 46-47: Performance Optimization
- Query optimization
- Caching strategy
- Memory management
- Async improvements

#### Day 48-49: Security Implementation
- API key authentication
- Rate limiting per IP
- Input validation
- SQL injection prevention

### Week 12: Documentation & Demo 📚
**Goal:** Project completion

#### Day 50-52: Documentation
- User manual (5 pages)
- Technical documentation (10 pages)
- API reference
- Project report (20-30 pages)

#### Day 53-54: Demo Preparation
- Create demo script
- Record demo video
- Prepare presentation slides

#### Day 55-56: Final Testing & Submission
- End-to-end testing
- Performance benchmarks
- Code cleanup
- Final submission package

---

## 🚀 Quick Start Commands

```bash
# 1. Activate environment
cd F:\Project\CIAP
venv\Scripts\activate

# 2. Start Ollama (in separate terminal)
ollama serve

# 3. Start Redis (if not running as service)
redis-server

# 4. Run development server
python main.py
# OR
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# 5. Run tests
pytest tests/ -v

# 6. Access API
# Browser: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## 📊 Success Metrics & Validation

### Week 1-2 Checkpoints
- [ ] Ollama responds to prompts
- [ ] Redis is running and accessible
- [ ] Enhanced database schema created
- [ ] Base scraper interface works

### Week 3-4 Checkpoints
- [ ] Can scrape from Google and Bing
- [ ] Crawlee can deep-scrape websites
- [ ] Rate limiting prevents bans
- [ ] Scraper manager coordinates multiple sources

### Week 5 Checkpoints
- [ ] Duplicate detection works (>90% accuracy)
- [ ] Text extraction clean and readable
- [ ] Entity extraction identifies companies
- [ ] Processing pipeline handles 100 URLs < 2 min

### Week 6-7 Checkpoints
- [ ] Ollama generates analysis offline
- [ ] Sentiment analysis accuracy > 75%
- [ ] Competitor identification works
- [ ] Analysis cached in Redis

### Week 8-9 Checkpoints
- [ ] Services orchestrate full workflow
- [ ] API handles concurrent requests
- [ ] WebSocket provides real-time updates
- [ ] Alerts trigger on conditions

### Week 10-11 Checkpoints
- [ ] Power BI can import exported data
- [ ] All export formats work
- [ ] Performance meets targets
- [ ] Security measures in place

### Final Validation
- [ ] Complete search → analysis → export workflow
- [ ] Demo runs without errors
- [ ] Documentation complete
- [ ] All code committed to git

---

## 🎯 Daily Development Checklist

```markdown
## Today's Focus: [Current Day from Plan]

### Morning (2 hours)
- [ ] Review today's plan tasks
- [ ] Update current status in CLAUDE.md
- [ ] Code primary feature

### Afternoon (2 hours)
- [ ] Test morning's work
- [ ] Fix any bugs found
- [ ] Code secondary feature

### Evening (1 hour)
- [ ] Test complete integration
- [ ] Commit working code
- [ ] Update todo list
- [ ] Note any blockers

### Before Sleep
- [ ] Plan tomorrow's tasks
- [ ] Prepare any questions for Claude
```

---

## 🔧 Common Issues & Quick Fixes

### Ollama Not Responding
```bash
# Check if running
ollama list
# Restart
ollama serve
# Try lighter model
ollama pull mistral
```

### Redis Connection Failed
```bash
# Check if running
redis-cli ping
# Start manually
redis-server
# Check Windows service
services.msc → Redis → Start
```

### Scraping Blocked
```python
# Increase delay
SCRAPE_DELAY = 5  # seconds
# Rotate user agents
# Use Crawlee with browser
```

### Out of Memory
```python
# Process in smaller batches
BATCH_SIZE = 10
# Clear cache periodically
redis_client.flushdb()
# Use generator functions
```

---

## 💡 Time-Saving Shortcuts

1. **Use existing Google scraper code** - Just enhance it
2. **Copy patterns between scrapers** - DRY principle
3. **Test with mock data first** - Don't wait for real scrapes
4. **Use Postman for API testing** - Faster than building UI
5. **Commit after each working feature** - Easy rollback

---

## 📈 Weekly Progress Tracking

### Week 1 (Oct 12-18, 2025)
- [ ] Environment fully set up
- [ ] Ollama working with local model
- [ ] Redis installed and running
- [ ] Enhanced project structure created
- [ ] Database schema updated

### Week 2 (Oct 19-25, 2025)
- [ ] Base scraper interface complete
- [ ] Configuration management working
- [ ] Logging system operational
- [ ] Setup script created

[Continue updating weekly...]

---

## 📝 Notes & Learnings

### What's Working Well
- FastAPI is fast and easy to work with
- SQLite is sufficient for MVP
- Existing code provides good foundation

### Current Challenges
- [Document any issues here]

### Decisions Made
- Use Ollama for free local LLM
- Keep SQLite instead of PostgreSQL
- Focus on Power BI export over custom dashboard
- No Docker - direct installation only

---

**Version:** 2.0
**Last Updated:** October 12, 2025
**Next Review:** End of Week 1