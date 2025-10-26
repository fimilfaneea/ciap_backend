"""
Configuration Management for CIAP
Centralized settings with environment variable loading and validation
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Dict, Any
from pathlib import Path
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

    # Scheduler
    SCHEDULER_TIMEZONE: str = Field(
        default="UTC",
        description="Scheduler timezone"
    )
    SCHEDULER_MAX_INSTANCES: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max concurrent instances per job"
    )
    SCHEDULER_MISFIRE_GRACE_TIME: int = Field(
        default=30,
        ge=0,
        le=300,
        description="Grace time for misfired jobs in seconds"
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

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL is valid SQLite"""
        if not v.startswith("sqlite:///"):
            raise ValueError("DATABASE_URL must start with 'sqlite:///'")
        return v

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Warn about default secret key in production"""
        # Note: In Pydantic v2, we can't access other fields in field_validator
        # This validation will be done in model_validator if needed
        if v == "CHANGE-THIS-SECRET-KEY-IN-PRODUCTION":
            import os
            if os.getenv("ENVIRONMENT", "development") == "production":
                raise ValueError("Must change SECRET_KEY in production!")
        return v

    @field_validator("LOG_FILE", "EXPORT_DIR", "DATA_DIR", "PROMPTS_DIR")
    @classmethod
    def create_directories(cls, v: str) -> str:
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
