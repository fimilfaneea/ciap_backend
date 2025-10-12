# Module 2: Configuration Management

## Overview
**Purpose:** Centralized configuration management with environment variables and validation.

**Responsibilities:**
- Load environment variables
- Validate configuration
- Provide type-safe settings
- Manage secrets securely
- Environment-specific configs

**Development Time:** 1 day (Week 1, Day 1)

---

## Interface Specification

### Input
```python
# .env file with configuration
DATABASE_URL=sqlite:///data/ciap.db
OLLAMA_URL=http://localhost:11434
LOG_LEVEL=INFO
```

### Output
```python
# Settings object with validated configuration
settings.DATABASE_URL  # Type-safe access
settings.OLLAMA_URL
settings.LOG_LEVEL
```

---

## Dependencies

### External
```txt
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
```

### Internal
- None (foundational module)

---

## Implementation Guide

### Step 1: Settings Class (`src/core/config.py`)

```python
from pydantic import BaseSettings, Field, validator
from pydantic_settings import SettingsConfigDict
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
from enum import Enum

class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Settings(BaseSettings):
    """Application settings with validation"""

    # Environment
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///data/ciap.db",
        description="SQLite database URL"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Database connection pool size"
    )
    DATABASE_ECHO: bool = Field(
        default=False,
        description="Echo SQL queries (debug mode)"
    )

    # Ollama LLM
    OLLAMA_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint"
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.1:8b",
        description="Ollama model to use"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Ollama request timeout in seconds"
    )

    # Scraping
    SCRAPER_USER_AGENTS: List[str] = Field(
        default=[
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ],
        description="User agents for web scraping"
    )
    SCRAPER_TIMEOUT: int = Field(
        default=30,
        ge=5,
        le=60,
        description="Scraper request timeout"
    )
    SCRAPER_RETRY_COUNT: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of scraping retries"
    )
    SCRAPER_RATE_LIMIT_DELAY: float = Field(
        default=1.0,
        ge=0.5,
        le=10.0,
        description="Delay between scraping requests in seconds"
    )

    # Google Scraper
    GOOGLE_SEARCH_URL: str = Field(
        default="https://www.google.com/search",
        description="Google search URL"
    )
    GOOGLE_MAX_RESULTS: int = Field(
        default=100,
        ge=10,
        le=200,
        description="Maximum Google results per search"
    )

    # Bing Scraper
    BING_SEARCH_URL: str = Field(
        default="https://www.bing.com/search",
        description="Bing search URL"
    )
    BING_MAX_RESULTS: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Maximum Bing results per search"
    )

    # Cache
    CACHE_TTL_SECONDS: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default cache TTL in seconds"
    )
    CACHE_CLEANUP_INTERVAL: int = Field(
        default=3600,
        ge=600,
        le=7200,
        description="Cache cleanup interval in seconds"
    )

    # Task Queue
    TASK_QUEUE_MAX_WORKERS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent task workers"
    )
    TASK_QUEUE_POLL_INTERVAL: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Task queue polling interval"
    )
    TASK_MAX_RETRIES: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum task retry attempts"
    )

    # API
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    API_PORT: int = Field(
        default=8000,
        ge=1000,
        le=65535,
        description="API server port"
    )
    API_PREFIX: str = Field(
        default="/api/v1",
        description="API route prefix"
    )
    API_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    API_RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Rate limit requests per minute"
    )

    # Security
    SECRET_KEY: str = Field(
        default="CHANGE-THIS-SECRET-KEY-IN-PRODUCTION",
        min_length=32,
        description="Secret key for sessions/tokens"
    )
    API_KEY: Optional[str] = Field(
        default=None,
        description="Optional API key for authentication"
    )

    # Logging
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    LOG_FILE: str = Field(
        default="data/logs/ciap.log",
        description="Log file path"
    )
    LOG_MAX_BYTES: int = Field(
        default=10_485_760,  # 10MB
        description="Maximum log file size in bytes"
    )
    LOG_BACKUP_COUNT: int = Field(
        default=5,
        description="Number of log backup files to keep"
    )

    # Export
    EXPORT_DIR: str = Field(
        default="data/exports",
        description="Directory for export files"
    )
    EXPORT_MAX_ROWS: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum rows in export files"
    )

    # Paths
    DATA_DIR: str = Field(
        default="data",
        description="Data directory path"
    )
    PROMPTS_DIR: str = Field(
        default="config/prompts",
        description="LLM prompts directory"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra environment variables
    )

    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Ensure database URL is valid SQLite"""
        if not v.startswith("sqlite:///"):
            raise ValueError("DATABASE_URL must start with 'sqlite:///'")
        return v

    @validator("SECRET_KEY")
    def validate_secret_key(cls, v, values):
        """Warn about default secret key in production"""
        if values.get("ENVIRONMENT") == Environment.PRODUCTION:
            if v == "CHANGE-THIS-SECRET-KEY-IN-PRODUCTION":
                raise ValueError("Must change SECRET_KEY in production!")
        return v

    @validator("LOG_FILE", "EXPORT_DIR", "DATA_DIR", "PROMPTS_DIR")
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        path = Path(v)
        if v.endswith(".log"):
            path = path.parent
        path.mkdir(parents=True, exist_ok=True)
        return v

    def get_database_url_async(self) -> str:
        """Get async database URL for SQLAlchemy"""
        return self.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")

    def get_user_agent(self) -> str:
        """Get random user agent for scraping"""
        import random
        return random.choice(self.SCRAPER_USER_AGENTS)

    def get_prompt_path(self, prompt_name: str) -> Path:
        """Get path to prompt file"""
        return Path(self.PROMPTS_DIR) / f"{prompt_name}.txt"

    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        data = self.model_dump()
        if exclude_secrets:
            data.pop("SECRET_KEY", None)
            data.pop("API_KEY", None)
        return data

# Create global settings instance
settings = Settings()
```

### Step 2: Environment File Template (`.env.example`)

```bash
# Environment
ENVIRONMENT=development

# Database
DATABASE_URL=sqlite:///data/ciap.db
DATABASE_POOL_SIZE=5
DATABASE_ECHO=false

# Ollama LLM
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=60

# Scraping
SCRAPER_TIMEOUT=30
SCRAPER_RETRY_COUNT=3
SCRAPER_RATE_LIMIT_DELAY=1.0

# Google Scraper
GOOGLE_MAX_RESULTS=100

# Bing Scraper
BING_MAX_RESULTS=50

# Cache
CACHE_TTL_SECONDS=3600
CACHE_CLEANUP_INTERVAL=3600

# Task Queue
TASK_QUEUE_MAX_WORKERS=3
TASK_QUEUE_POLL_INTERVAL=1.0
TASK_MAX_RETRIES=3

# API
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1
API_CORS_ORIGINS=["http://localhost:3000"]
API_RATE_LIMIT_REQUESTS=100

# Security
SECRET_KEY=your-secret-key-at-least-32-characters-long
# API_KEY=optional-api-key-for-authentication

# Logging
LOG_LEVEL=INFO
LOG_FILE=data/logs/ciap.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5

# Export
EXPORT_DIR=data/exports
EXPORT_MAX_ROWS=10000

# Paths
DATA_DIR=data
PROMPTS_DIR=config/prompts
```

### Step 3: Config Utilities (`src/core/config_utils.py`)

```python
from pathlib import Path
from typing import Dict, Any
import json
import yaml
from .config import settings

class ConfigManager:
    """Additional configuration management utilities"""

    @staticmethod
    def load_prompts() -> Dict[str, str]:
        """Load all prompt templates"""
        prompts = {}
        prompts_dir = Path(settings.PROMPTS_DIR)

        if prompts_dir.exists():
            for prompt_file in prompts_dir.glob("*.txt"):
                prompt_name = prompt_file.stem
                prompts[prompt_name] = prompt_file.read_text()

        return prompts

    @staticmethod
    def load_user_agents(file_path: str = "config/user_agents.txt") -> List[str]:
        """Load user agents from file"""
        path = Path(file_path)
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return settings.SCRAPER_USER_AGENTS

    @staticmethod
    def validate_environment():
        """Validate environment setup"""
        checks = {
            "database_dir": Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent,
            "log_dir": Path(settings.LOG_FILE).parent,
            "export_dir": Path(settings.EXPORT_DIR),
            "data_dir": Path(settings.DATA_DIR),
            "prompts_dir": Path(settings.PROMPTS_DIR)
        }

        issues = []
        for name, path in checks.items():
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"✅ Created {name}: {path}")
                except Exception as e:
                    issues.append(f"❌ Failed to create {name}: {e}")

        # Check Ollama connection
        import requests
        try:
            response = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if not any(m["name"] == settings.OLLAMA_MODEL for m in models):
                    issues.append(f"⚠️  Ollama model {settings.OLLAMA_MODEL} not found")
                else:
                    print(f"✅ Ollama connected with model {settings.OLLAMA_MODEL}")
        except Exception as e:
            issues.append(f"❌ Cannot connect to Ollama: {e}")

        return issues

    @staticmethod
    def export_config(output_file: str = "config_export.json"):
        """Export current configuration (without secrets)"""
        config_dict = settings.to_dict(exclude_secrets=True)

        output_path = Path(output_file)
        if output_file.endswith(".yaml"):
            output_path.write_text(yaml.dump(config_dict, default_flow_style=False))
        else:
            output_path.write_text(json.dumps(config_dict, indent=2))

        return output_path

    @staticmethod
    def get_runtime_info() -> Dict[str, Any]:
        """Get runtime configuration info"""
        import platform
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "environment": settings.ENVIRONMENT,
            "database": settings.DATABASE_URL,
            "ollama": settings.OLLAMA_URL,
            "api_port": settings.API_PORT,
            "log_level": settings.LOG_LEVEL,
            "workers": settings.TASK_QUEUE_MAX_WORKERS
        }
```

---

## Testing Guide

### Unit Tests (`tests/test_config.py`)

```python
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.core.config import Settings, Environment, LogLevel
from src.core.config_utils import ConfigManager

def test_default_settings():
    """Test default configuration values"""
    settings = Settings()

    assert settings.ENVIRONMENT == Environment.DEVELOPMENT
    assert settings.DATABASE_URL == "sqlite:///data/ciap.db"
    assert settings.OLLAMA_URL == "http://localhost:11434"
    assert settings.LOG_LEVEL == LogLevel.INFO
    assert settings.API_PORT == 8000

def test_env_override():
    """Test environment variable override"""
    with patch.dict(os.environ, {
        "DATABASE_URL": "sqlite:///test.db",
        "API_PORT": "9000",
        "LOG_LEVEL": "DEBUG"
    }):
        settings = Settings()
        assert settings.DATABASE_URL == "sqlite:///test.db"
        assert settings.API_PORT == 9000
        assert settings.LOG_LEVEL == LogLevel.DEBUG

def test_validation_errors():
    """Test configuration validation"""
    # Invalid database URL
    with pytest.raises(ValueError, match="must start with 'sqlite:///'"):
        Settings(DATABASE_URL="postgresql://localhost/db")

    # Invalid port range
    with pytest.raises(ValueError):
        Settings(API_PORT=99999)

    # Invalid log level
    with pytest.raises(ValueError):
        Settings(LOG_LEVEL="INVALID")

def test_production_secret_key():
    """Test secret key validation in production"""
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        # Should raise with default secret
        with pytest.raises(ValueError, match="Must change SECRET_KEY"):
            Settings()

        # Should work with custom secret
        with patch.dict(os.environ, {"SECRET_KEY": "a" * 32}):
            settings = Settings()
            assert settings.SECRET_KEY == "a" * 32

def test_directory_creation(tmp_path):
    """Test automatic directory creation"""
    with patch.dict(os.environ, {
        "DATA_DIR": str(tmp_path / "data"),
        "LOG_FILE": str(tmp_path / "logs/test.log"),
        "EXPORT_DIR": str(tmp_path / "exports")
    }):
        settings = Settings()

        assert (tmp_path / "data").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "exports").exists()

def test_async_database_url():
    """Test async database URL generation"""
    settings = Settings(DATABASE_URL="sqlite:///test.db")
    async_url = settings.get_database_url_async()
    assert async_url == "sqlite+aiosqlite:///test.db"

def test_random_user_agent():
    """Test random user agent selection"""
    settings = Settings()
    agents = set()

    # Get multiple agents
    for _ in range(10):
        agents.add(settings.get_user_agent())

    # Should have some variety
    assert len(agents) > 1
    assert all(agent in settings.SCRAPER_USER_AGENTS for agent in agents)

def test_to_dict_excludes_secrets():
    """Test configuration export excludes secrets"""
    settings = Settings(
        SECRET_KEY="secret123",
        API_KEY="apikey123"
    )

    config_dict = settings.to_dict(exclude_secrets=True)
    assert "SECRET_KEY" not in config_dict
    assert "API_KEY" not in config_dict

    config_dict = settings.to_dict(exclude_secrets=False)
    assert config_dict["SECRET_KEY"] == "secret123"
    assert config_dict["API_KEY"] == "apikey123"

def test_load_prompts(tmp_path):
    """Test prompt loading"""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create test prompts
    (prompts_dir / "sentiment.txt").write_text("Analyze sentiment: {text}")
    (prompts_dir / "competitor.txt").write_text("Find competitors: {text}")

    with patch.dict(os.environ, {"PROMPTS_DIR": str(prompts_dir)}):
        settings = Settings()
        with patch("src.core.config.settings", settings):
            prompts = ConfigManager.load_prompts()

    assert len(prompts) == 2
    assert prompts["sentiment"] == "Analyze sentiment: {text}"
    assert prompts["competitor"] == "Find competitors: {text}"

def test_validate_environment(tmp_path):
    """Test environment validation"""
    with patch.dict(os.environ, {
        "DATA_DIR": str(tmp_path / "data"),
        "OLLAMA_URL": "http://localhost:11434"
    }):
        settings = Settings()

        # Mock Ollama API
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [{"name": "llama3.1:8b"}]
            }
            mock_get.return_value = mock_response

            with patch("src.core.config.settings", settings):
                issues = ConfigManager.validate_environment()

    assert len(issues) == 0  # No issues

def test_export_config(tmp_path):
    """Test configuration export"""
    settings = Settings()

    with patch("src.core.config.settings", settings):
        # Export as JSON
        json_path = tmp_path / "config.json"
        ConfigManager.export_config(str(json_path))
        assert json_path.exists()

        import json
        config = json.loads(json_path.read_text())
        assert config["ENVIRONMENT"] == "development"
        assert "SECRET_KEY" not in config  # Should be excluded
```

---

## Integration Points

### With Database Module
```python
from src.core.config import settings
from src.core.database import DatabaseManager

db_manager = DatabaseManager(settings.get_database_url_async())
```

### With Logging Module
```python
from src.core.config import settings
import logging

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    filename=settings.LOG_FILE
)
```

### With API Module
```python
from src.core.config import settings
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
```

### With Scraper Module
```python
from src.core.config import settings

class BaseScraper:
    def __init__(self):
        self.timeout = settings.SCRAPER_TIMEOUT
        self.retries = settings.SCRAPER_RETRY_COUNT
        self.user_agent = settings.get_user_agent()
```

---

## Common Issues & Solutions

### Issue 1: Missing .env File
**Problem:** Settings not loading from .env
**Solution:** Create .env from template
```bash
cp .env.example .env
```

### Issue 2: Invalid Environment Variables
**Problem:** Pydantic validation errors
**Solution:** Check data types in .env
```bash
# Wrong
API_PORT="eight thousand"

# Correct
API_PORT=8000
```

### Issue 3: Directory Permissions
**Problem:** Cannot create directories
**Solution:** Check permissions
```python
import os
os.makedirs("data", mode=0o755, exist_ok=True)
```

### Issue 4: Ollama Connection
**Problem:** Cannot connect to Ollama
**Solution:** Verify Ollama is running
```bash
# Start Ollama
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Issue 5: Secret Key in Production
**Problem:** Using default secret key
**Solution:** Generate secure key
```python
import secrets
print(secrets.token_urlsafe(32))
```

---

## Best Practices

### 1. Environment-Specific Settings
```python
if settings.ENVIRONMENT == Environment.DEVELOPMENT:
    settings.DATABASE_ECHO = True  # Enable SQL logging
    settings.LOG_LEVEL = LogLevel.DEBUG
```

### 2. Validate Early
```python
# In main.py
from src.core.config_utils import ConfigManager

issues = ConfigManager.validate_environment()
if issues:
    for issue in issues:
        print(issue)
    sys.exit(1)
```

### 3. Use Type Hints
```python
def process_data(timeout: int = settings.SCRAPER_TIMEOUT):
    # Type safety with settings
    pass
```

### 4. Override for Testing
```python
# In tests
test_settings = Settings(
    DATABASE_URL="sqlite:///:memory:",
    OLLAMA_URL="http://mock-ollama:11434"
)
```

### 5. Document Settings
```python
CACHE_TTL_SECONDS: int = Field(
    default=3600,
    ge=60,  # Minimum 1 minute
    le=86400,  # Maximum 24 hours
    description="Cache time-to-live in seconds"
)
```

---

## Module Checklist

- [ ] Settings class created
- [ ] Environment variables loading
- [ ] Validation rules implemented
- [ ] .env.example template created
- [ ] Directory auto-creation working
- [ ] Secret key validation
- [ ] Ollama connection check
- [ ] Unit tests passing
- [ ] Integration documented
- [ ] Common issues documented

---

## Next Steps
After completing this module:
1. **Module 3: Cache** - Uses cache settings
2. **Module 1: Database** - Uses database settings
3. **Module 5: Scraper** - Uses scraper settings