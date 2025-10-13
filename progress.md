https://markdownlivepreview.com/

# Complete Configuration Management Guide for Beginners

## Table of Contents
1. [What is Configuration Management?](#what-is-configuration-management)
2. [Why Do We Need It?](#why-do-we-need-it)
3. [Project Structure Setup](#project-structure-setup)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing Your Configuration](#testing-your-configuration)
6. [Common Pitfalls](#common-pitfalls)

---

## What is Configuration Management?

Configuration management is how your application stores and accesses **settings** that control its behavior. Think of it like a control panel for your app.

### Real-World Analogy
Imagine your app is a car:
- **Configuration** = Dashboard settings (radio volume, seat position, AC temperature)
- **Environment Variables** = Driver preferences that change per person
- **Validation** = Safety checks (can't set speed to 1000 mph)

### What Goes in Configuration?
- Database connection strings
- API endpoints (like Ollama URL)
- Timeouts and retry counts
- File paths
- Security keys
- Feature flags

---

## Why Do We Need It?

### Problem Without Config Management
```python
# Bad: Hardcoded values scattered everywhere
def scrape_website():
    timeout = 30  # What if we need to change this?
    retry = 3
    url = "http://localhost:11434"  # What if server moves?
```

**Issues:**
- Change one value → edit 20 files
- Different settings for dev/production → maintain 2 codebases
- Secrets in code → security risk
- No validation → runtime errors

### Solution With Config Management
```python
# Good: Centralized, validated settings
from src.core.config import settings

def scrape_website():
    timeout = settings.SCRAPER_TIMEOUT  # Change once, affects everywhere
    retry = settings.SCRAPER_RETRY_COUNT
    url = settings.OLLAMA_URL
```

**Benefits:**
- Single source of truth
- Environment-specific configs (.env files)
- Type validation (catches errors early)
- Secure secret management

---

## Project Structure Setup

### Step 1: Create Directory Structure

```bash
ciap_project/
├── src/
│   └── core/
│       ├── __init__.py          # Empty file (makes it a package)
│       ├── config.py            # Main settings class
│       └── config_utils.py      # Helper utilities
├── config/
│   └── prompts/                 # LLM prompt templates
│       ├── sentiment.txt
│       └── competitor.txt
├── data/
│   ├── logs/                    # Log files
│   └── exports/                 # Export outputs
├── tests/
│   └── test_config.py           # Configuration tests
├── .env                         # Your actual settings (NEVER commit)
├── .env.example                 # Template (safe to commit)
└── requirements.txt             # Python dependencies
```

**Why This Structure?**
- `src/core/` = Foundational modules used by everything
- `config/` = Static configuration files
- `data/` = Runtime-generated files
- `.env` = Secret settings (gitignored)
- `.env.example` = Public template

### Step 2: Create `__init__.py` Files

```python
# src/__init__.py
# Empty file - just tells Python this is a package

# src/core/__init__.py
from .config import settings

__all__ = ["settings"]
```

**Why?** This allows you to import like:
```python
from src.core import settings  # Instead of from src.core.config import settings
```

---

## Step-by-Step Implementation

### Step 1: Install Dependencies

Create `requirements.txt`:
```txt
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
```

Install them:
```bash
pip install -r requirements.txt
```

**What Each Does:**
- `pydantic` = Data validation library (ensures types are correct)
- `pydantic-settings` = Loads settings from environment variables
- `python-dotenv` = Reads `.env` files

---

### Step 2: Create `.env.example` Template

This is a **template** that shows what settings are available. It's safe to commit to git.

```bash
# .env.example

# === Environment ===
ENVIRONMENT=development

# === Database ===
DATABASE_URL=sqlite:///data/ciap.db
DATABASE_POOL_SIZE=5

# === Ollama LLM ===
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=60

# === Scraping ===
SCRAPER_TIMEOUT=30
SCRAPER_RETRY_COUNT=3

# === API ===
API_HOST=0.0.0.0
API_PORT=8000

# === Security ===
SECRET_KEY=CHANGE-THIS-IN-PRODUCTION

# === Logging ===
LOG_LEVEL=INFO
LOG_FILE=data/logs/ciap.log
```

---

### Step 3: Create Your Actual `.env` File

Copy the template and fill in **real values**:

```bash
cp .env.example .env
```

Edit `.env` with your actual settings:
```bash
# .env (NEVER COMMIT THIS FILE)

ENVIRONMENT=development
DATABASE_URL=sqlite:///data/ciap.db
OLLAMA_URL=http://localhost:11434
SECRET_KEY=my-super-secret-key-that-is-32-chars-long
```

**Important:** Add `.env` to `.gitignore`:
```
# .gitignore
.env
*.log
data/
```

---

### Step 4: Create `config.py` - The Main Settings Class

Let's build this step-by-step with explanations:

```python
# src/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from enum import Enum
from pathlib import Path
from typing import List, Optional
```

**What's Imported:**
- `BaseSettings` = Base class that loads environment variables
- `Field` = Adds validation rules and descriptions
- `validator` = Custom validation functions
- `Enum` = Creates constants (like DEVELOPMENT, PRODUCTION)

#### Part A: Define Enums (Constants)

```python
class Environment(str, Enum):
    """Valid application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Valid logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
```

**Why Enums?**
- Prevents typos (can't accidentally use "PROD" instead of "production")
- IDE autocomplete
- Type checking

#### Part B: Create Settings Class

```python
class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Environment
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
```

**Breaking This Down:**
- `ENVIRONMENT: Environment` = Variable name and type
- `Field(...)` = Adds validation and metadata
- `default=...` = Value if not in .env
- `description=...` = Documentation

**More Examples:**

```python
    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///data/ciap.db",
        description="SQLite database URL"
    )
    
    DATABASE_POOL_SIZE: int = Field(
        default=5,
        ge=1,      # Greater than or equal to 1
        le=20,     # Less than or equal to 20
        description="Database connection pool size"
    )
```

**What `ge` and `le` Do:**
```python
DATABASE_POOL_SIZE=25  # ❌ Error: Must be <= 20
DATABASE_POOL_SIZE=0   # ❌ Error: Must be >= 1
DATABASE_POOL_SIZE=10  # ✅ Valid
```

#### Part C: Configuration for Pydantic

```python
    model_config = SettingsConfigDict(
        env_file=".env",              # Read from .env file
        env_file_encoding="utf-8",    # File encoding
        case_sensitive=True,          # ENVIRONMENT != environment
        extra="ignore"                # Ignore unknown env vars
    )
```

**Why This Matters:**
- `env_file=".env"` = Automatically loads .env
- `case_sensitive=True` = `API_PORT` ≠ `api_port` (prevents confusion)
- `extra="ignore"` = Don't error on extra env vars (useful if system has other vars)

#### Part D: Custom Validators

```python
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Ensure database URL is valid SQLite"""
        if not v.startswith("sqlite:///"):
            raise ValueError("DATABASE_URL must start with 'sqlite:///'")
        return v
```

**How This Works:**
1. User sets `DATABASE_URL=postgresql://localhost/db` in .env
2. Pydantic loads it
3. This validator runs automatically
4. Sees it doesn't start with `sqlite:///`
5. **Raises error immediately** (before your app even starts)

**Another Example - Production Safety:**

```python
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v, values):
        """Warn about default secret key in production"""
        env = values.get("ENVIRONMENT")
        if env == Environment.PRODUCTION:
            if v == "CHANGE-THIS-IN-PRODUCTION":
                raise ValueError("Must change SECRET_KEY in production!")
        return v
```

**What This Prevents:**
```python
# .env
ENVIRONMENT=production
SECRET_KEY=CHANGE-THIS-IN-PRODUCTION  # ❌ App won't start - forces you to change it
```

#### Part E: Helper Methods

```python
    def get_database_url_async(self) -> str:
        """Convert sync database URL to async"""
        return self.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
```

**Why?** SQLAlchemy needs different URLs for sync vs async:
- Sync: `sqlite:///data/ciap.db`
- Async: `sqlite+aiosqlite:///data/ciap.db`

```python
    def get_user_agent(self) -> str:
        """Get random user agent for scraping"""
        import random
        return random.choice(self.SCRAPER_USER_AGENTS)
```

**Usage:**
```python
headers = {"User-Agent": settings.get_user_agent()}  # Different each time
```

#### Part F: Create Global Instance

```python
# At the end of config.py
settings = Settings()
```

**This is the magic line!** Now everywhere in your app:
```python
from src.core.config import settings

print(settings.DATABASE_URL)  # Works immediately
```

---

### Step 5: Create `config_utils.py` - Helper Functions

```python
# src/core/config_utils.py

from pathlib import Path
from typing import Dict, List
from .config import settings
import requests

class ConfigManager:
    """Utilities for managing configuration"""
    
    @staticmethod
    def validate_environment():
        """Check if environment is set up correctly"""
        issues = []
        
        # Check if directories exist
        directories = [
            Path(settings.DATA_DIR),
            Path(settings.LOG_FILE).parent,
            Path(settings.EXPORT_DIR),
        ]
        
        for directory in directories:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    print(f"✅ Created directory: {directory}")
                except Exception as e:
                    issues.append(f"❌ Cannot create {directory}: {e}")
        
        # Check Ollama connection
        try:
            response = requests.get(
                f"{settings.OLLAMA_URL}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                print(f"✅ Ollama is running at {settings.OLLAMA_URL}")
            else:
                issues.append(f"⚠️  Ollama returned status {response.status_code}")
        except Exception as e:
            issues.append(f"❌ Cannot connect to Ollama: {e}")
        
        return issues
```

**What This Does:**
1. Creates missing directories automatically
2. Checks if Ollama is running
3. Returns list of problems (if any)

**Usage:**
```python
from src.core.config_utils import ConfigManager

issues = ConfigManager.validate_environment()
if issues:
    print("Problems found:")
    for issue in issues:
        print(issue)
    exit(1)  # Stop the app
```

---

## Testing Your Configuration

### Manual Testing

Create `test_config_manual.py`:

```python
# test_config_manual.py

from src.core.config import settings
from src.core.config_utils import ConfigManager

def test_basic_settings():
    """Test if settings load correctly"""
    print("=== Basic Settings ===")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"Ollama URL: {settings.OLLAMA_URL}")
    print(f"API Port: {settings.API_PORT}")
    print(f"Log Level: {settings.LOG_LEVEL}")

def test_validation():
    """Test environment validation"""
    print("\n=== Environment Validation ===")
    issues = ConfigManager.validate_environment()
    
    if not issues:
        print("✅ All checks passed!")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  {issue}")

def test_helper_methods():
    """Test helper methods"""
    print("\n=== Helper Methods ===")
    print(f"Async DB URL: {settings.get_database_url_async()}")
    print(f"Random User Agent: {settings.get_user_agent()}")

if __name__ == "__main__":
    test_basic_settings()
    test_validation()
    test_helper_methods()
```

Run it:
```bash
python test_config_manual.py
```

**Expected Output:**
```
=== Basic Settings ===
Environment: development
Database URL: sqlite:///data/ciap.db
Ollama URL: http://localhost:11434
API Port: 8000
Log Level: INFO

=== Environment Validation ===
✅ Created directory: data
✅ Created directory: data/logs
✅ Ollama is running at http://localhost:11434
✅ All checks passed!

=== Helper Methods ===
Async DB URL: sqlite+aiosqlite:///data/ciap.db
Random User Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)...
```

### Automated Unit Tests

Create `tests/test_config.py`:

```python
# tests/test_config.py

import pytest
from unittest.mock import patch
import os
from src.core.config import Settings, Environment

def test_default_values():
    """Test that default values work"""
    settings = Settings()
    assert settings.ENVIRONMENT == Environment.DEVELOPMENT
    assert settings.API_PORT == 8000

def test_env_override():
    """Test that .env overrides defaults"""
    with patch.dict(os.environ, {"API_PORT": "9000"}):
        settings = Settings()
        assert settings.API_PORT == 9000

def test_validation_catches_errors():
    """Test that validation works"""
    # Invalid database URL
    with pytest.raises(ValueError):
        Settings(DATABASE_URL="postgres://localhost/db")
    
    # Port out of range
    with pytest.raises(ValueError):
        Settings(API_PORT=99999)
```

Run tests:
```bash
pytest tests/test_config.py -v
```

---

## Common Pitfalls

### Pitfall 1: Forgetting to Create .env

**Error:**
```
❌ Settings not loading, using all defaults
```

**Solution:**
```bash
cp .env.example .env
```

### Pitfall 2: Wrong Data Types in .env

**Error:**
```
pydantic.ValidationError: API_PORT must be an integer
```

**Problem in .env:**
```bash
API_PORT="8000"  # ❌ Strings in quotes don't convert to int
```

**Fix:**
```bash
API_PORT=8000  # ✅ No quotes for numbers
```

### Pitfall 3: Case Sensitivity

**Problem:**
```bash
# .env
api_port=8000  # ❌ Wrong case
```

**Fix:**
```bash
# .env
API_PORT=8000  # ✅ Must match exactly
```

### Pitfall 4: Committing .env to Git

**Prevention:**
```bash
# .gitignore
.env
*.log
data/
__pycache__/
```

Verify:
```bash
git status  # .env should NOT appear
```

### Pitfall 5: Not Validating Early

**Bad:**
```python
# app.py
from src.core.config import settings

# App runs for 10 minutes...
# Then tries to connect to database
db.connect(settings.DATABASE_URL)  # ❌ Fails here - wasted time
```

**Good:**
```python
# app.py
from src.core.config import settings
from src.core.config_utils import ConfigManager

# Validate immediately on startup
issues = ConfigManager.validate_environment()
if issues:
    print("Configuration errors:")
    for issue in issues:
        print(issue)
    exit(1)

# Now we know config is good
db.connect(settings.DATABASE_URL)  # ✅ Will work
```

---

## Next Steps

1. **Implement the Settings class** from Step 4
2. **Create .env file** with your actual values
3. **Run manual tests** to verify everything works
4. **Add validation** to your app's startup sequence
5. **Integrate with other modules** (database, logging, etc.)

### Integration Example

```python
# main.py - Your app's entry point

from src.core.config import settings
from src.core.config_utils import ConfigManager
import logging

def setup():
    """Set up application"""
    # Validate environment
    issues = ConfigManager.validate_environment()
    if issues:
        for issue in issues:
            print(issue)
        exit(1)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        filename=settings.LOG_FILE,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Application started in {settings.ENVIRONMENT} mode")

def main():
    setup()
    # Your app logic here
    print(f"API running on {settings.API_HOST}:{settings.API_PORT}")

if __name__ == "__main__":
    main()
```

---

## Quick Reference

### Accessing Settings
```python
from src.core.config import settings

# Simple access
url = settings.DATABASE_URL
timeout = settings.SCRAPER_TIMEOUT

# Helper methods
async_url = settings.get_database_url_async()
user_agent = settings.get_user_agent()
```

### Overriding for Tests
```python
# In tests
test_settings = Settings(
    DATABASE_URL="sqlite:///:memory:",
    ENVIRONMENT=Environment.TESTING
)
```

### Validation
```python
from src.core.config_utils import ConfigManager

issues = ConfigManager.validate_environment()
if issues:
    # Handle problems
    pass
```

### Environment-Specific Logic
```python
if settings.ENVIRONMENT == Environment.PRODUCTION:
    # Production-only code
    enable_monitoring()
elif settings.ENVIRONMENT == Environment.DEVELOPMENT:
    # Dev-only code
    enable_debug_toolbar()
```

---

## Summary

Configuration management gives you:
- ✅ Single source of truth for settings
- ✅ Type safety and validation
- ✅ Environment-specific configs
- ✅ Secure secret management
- ✅ Early error detection
- ✅ Easy testing

By implementing this module first, you create a solid foundation for all other modules in your project.