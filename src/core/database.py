"""
Async Database Connection and Management for CIAP
Uses SQLAlchemy with aiosqlite for async SQLite operations
"""

import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text, event
from contextlib import asynccontextmanager
import aiosqlite
from pathlib import Path
import logging
from .models import Base
from .fts_setup import setup_fts5
from .config import settings

# Setup logging with configuration
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages async database connections with SQLite
    Implements connection pooling, WAL mode, and performance optimizations
    """

    def __init__(self, database_url: str = None):
        """
        Initialize database manager

        Args:
            database_url: SQLite database URL. If None, uses settings.get_database_url_async()
                         Allows override for testing with in-memory databases.
        """
        if database_url is None:
            # Use configuration-based database URL
            database_url = settings.get_database_url_async()

        self.database_url = database_url
        self.engine = None
        self.async_session = None
        self._initialized = False

    async def initialize(self):
        """
        Initialize database connection and create tables
        Configures SQLite for optimal performance
        """
        if self._initialized:
            logger.info("Database already initialized")
            return

        logger.info(f"Initializing database: {self.database_url}")

        # Create async engine
        # Note: SQLite doesn't support connection pooling (uses NullPool)
        self.engine = create_async_engine(
            self.database_url,
            echo=settings.DATABASE_ECHO,  # Use configuration for SQL logging
            pool_pre_ping=True,  # Verify connections are alive
        )

        # Set PRAGMAs on every new connection to ensure they persist
        # This is necessary because SQLite PRAGMAs are connection-specific
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            """Apply SQLite PRAGMAs to each connection"""
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA temp_store = MEMORY")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.close()

        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Configure SQLite for better performance
        await self._configure_sqlite()

        # Create all tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

        # Create indexes for better performance
        await self._create_indexes()

        # Setup FTS5 full-text search
        await setup_fts5(self.engine)
        logger.info("FTS5 full-text search configured")

        self._initialized = True
        logger.info("Database initialization complete")

    async def _configure_sqlite(self):
        """
        Configure SQLite for optimal performance
        Enables WAL mode, memory caching, and other optimizations
        """
        async with self.engine.begin() as conn:
            # Enable WAL mode for better concurrency
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            logger.info("WAL mode enabled")

            # Increase cache size (negative value = KB, positive = pages)
            await conn.execute(text("PRAGMA cache_size=-64000"))  # 64MB cache

            # Store temporary tables and indices in memory
            await conn.execute(text("PRAGMA temp_store=MEMORY"))

            # Synchronous mode NORMAL for better performance
            await conn.execute(text("PRAGMA synchronous=NORMAL"))

            # Enable foreign key constraints
            await conn.execute(text("PRAGMA foreign_keys=ON"))

            # Optimize for faster writes
            await conn.execute(text("PRAGMA page_size=4096"))

            # Auto-vacuum to prevent database bloat
            await conn.execute(text("PRAGMA auto_vacuum=INCREMENTAL"))

            logger.info("SQLite performance optimizations applied")

    async def _create_indexes(self):
        """
        Create database indexes for better query performance
        Indexes all foreign keys and frequently queried columns
        """
        indexes = [
            # Search indexes
            "CREATE INDEX IF NOT EXISTS idx_searches_status ON searches(status)",
            "CREATE INDEX IF NOT EXISTS idx_searches_created ON searches(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_searches_user ON searches(user_id)",

            # Search results indexes
            "CREATE INDEX IF NOT EXISTS idx_search_results_search_id ON search_results(search_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_source ON search_results(source)",

            # Product indexes
            "CREATE INDEX IF NOT EXISTS idx_products_company ON products(company_name)",
            "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)",
            "CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand_name)",

            # Price data indexes
            "CREATE INDEX IF NOT EXISTS idx_price_data_product ON price_data(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_price_data_scraped ON price_data(scraped_at)",

            # Offers indexes
            "CREATE INDEX IF NOT EXISTS idx_offers_product ON offers(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_offers_active ON offers(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_offers_dates ON offers(start_date, end_date)",

            # Reviews indexes
            "CREATE INDEX IF NOT EXISTS idx_reviews_product ON product_reviews(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_reviews_rating ON product_reviews(rating)",
            "CREATE INDEX IF NOT EXISTS idx_reviews_source ON product_reviews(review_source)",

            # SERP data indexes
            "CREATE INDEX IF NOT EXISTS idx_serp_search ON serp_data(search_id)",
            "CREATE INDEX IF NOT EXISTS idx_serp_query ON serp_data(search_query)",

            # Social sentiment indexes
            "CREATE INDEX IF NOT EXISTS idx_social_platform ON social_sentiment(platform)",
            "CREATE INDEX IF NOT EXISTS idx_social_sentiment ON social_sentiment(sentiment)",

            # News content indexes
            "CREATE INDEX IF NOT EXISTS idx_news_date ON news_content(publication_date)",
            "CREATE INDEX IF NOT EXISTS idx_news_source ON news_content(publication_source)",

            # Cache indexes
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)",

            # Task queue indexes
            "CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status, priority)",
            "CREATE INDEX IF NOT EXISTS idx_task_queue_scheduled ON task_queue(scheduled_at)",

            # Rate limits indexes
            "CREATE INDEX IF NOT EXISTS idx_rate_limits_scraper ON rate_limits(scraper_name)",

            # Composite indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_products_search ON products(product_name, brand_name, company_name)",
            "CREATE INDEX IF NOT EXISTS idx_price_latest ON price_data(product_id, scraped_at DESC)",
        ]

        async with self.engine.begin() as conn:
            for index_sql in indexes:
                await conn.execute(text(index_sql))
            logger.info(f"Created {len(indexes)} database indexes")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with context manager
        Handles commit/rollback automatically

        Yields:
            AsyncSession: Database session
        """
        if not self._initialized:
            await self.initialize()

        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()

    async def close(self):
        """Close database connection and cleanup resources"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")
            self._initialized = False

    async def health_check(self) -> bool:
        """
        Check if database is accessible and healthy

        Returns:
            bool: True if database is healthy
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def get_stats(self) -> dict:
        """
        Get database statistics

        Returns:
            dict: Database statistics including table counts
        """
        stats = {}

        async with self.get_session() as session:
            # Get database file size
            result = await session.execute(text("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"))
            stats['database_size_bytes'] = result.scalar()

            # Get table counts
            tables = [
                'searches', 'search_results', 'products', 'price_data',
                'offers', 'product_reviews', 'competitors', 'market_trends',
                'serp_data', 'social_sentiment', 'news_content', 'feature_comparisons',
                'cache', 'task_queue', 'scraping_jobs', 'rate_limits'
            ]

            for table in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[f'{table}_count'] = result.scalar()
                except:
                    stats[f'{table}_count'] = 0

            # Get cache stats
            result = await session.execute(text("SELECT COUNT(*) FROM cache WHERE expires_at > datetime('now')"))
            stats['active_cache_entries'] = result.scalar()

            # Get task queue stats
            result = await session.execute(text("SELECT status, COUNT(*) as count FROM task_queue GROUP BY status"))
            stats['task_queue_by_status'] = {row[0]: row[1] for row in result}

        return stats

    async def optimize(self):
        """
        Run database optimization tasks
        Includes VACUUM, ANALYZE, and cache cleanup
        """
        logger.info("Running database optimization...")

        async with self.engine.begin() as conn:
            # Run ANALYZE to update query planner statistics
            await conn.execute(text("ANALYZE"))

            # Incremental vacuum to reclaim space
            await conn.execute(text("PRAGMA incremental_vacuum"))

            # Clean expired cache entries
            await conn.execute(text("DELETE FROM cache WHERE expires_at < datetime('now')"))

            # Clean old completed tasks (older than 7 days)
            await conn.execute(text("""
                DELETE FROM task_queue
                WHERE status = 'completed'
                AND completed_at < datetime('now', '-7 days')
            """))

        logger.info("Database optimization complete")


# Global database manager instance
db_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session

    Yields:
        AsyncSession: Database session
    """
    async with db_manager.get_session() as session:
        yield session


# Utility functions
async def init_db():
    """Initialize database (for scripts and testing)"""
    await db_manager.initialize()


async def close_db():
    """Close database connection"""
    await db_manager.close()


if __name__ == "__main__":
    # Test database initialization when run directly
    async def test_db():
        await init_db()

        # Test health check
        is_healthy = await db_manager.health_check()
        print(f"Database health check: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")

        # Get stats
        stats = await db_manager.get_stats()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            if key == 'database_size_bytes':
                print(f"  {key}: {value / 1024:.2f} KB")
            else:
                print(f"  {key}: {value}")

        await close_db()

    asyncio.run(test_db())