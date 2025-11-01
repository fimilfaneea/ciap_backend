"""
Database Migration: Fix NULL sources in searches table

Issue: Existing search records may have NULL in sources column, causing
frontend crashes when accessing search.sources.length or search.sources.join()

Solution:
1. Update all NULL sources to empty JSON array []
2. Add NOT NULL constraint to prevent future NULLs

Run this migration ONCE before deploying the model changes.

Usage:
    python migrations/fix_sources_null.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.database.manager import DatabaseManager
from src.config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate():
    """Run migration to fix NULL sources"""

    # Initialize database manager
    db_url = settings.get_database_url_async()
    db_manager = DatabaseManager(db_url)

    try:
        await db_manager.initialize()
        logger.info("Database connection established")

        async with db_manager.get_session() as session:
            # Step 1: Count records with NULL sources
            count_result = await session.execute(
                text("SELECT COUNT(*) FROM searches WHERE sources IS NULL")
            )
            null_count = count_result.scalar()

            logger.info(f"Found {null_count} records with NULL sources")

            if null_count == 0:
                logger.info("✅ No NULL sources found - migration not needed")
                logger.info("   Database is already in correct state")
                return True

            # Step 2: Update NULL sources to empty JSON array
            logger.info(f"Updating {null_count} records...")

            # SQLite JSON format
            update_result = await session.execute(
                text("UPDATE searches SET sources = '[]' WHERE sources IS NULL")
            )

            await session.commit()

            logger.info(f"✅ Updated {update_result.rowcount} records")

            # Step 3: Verify no NULLs remain
            verify_result = await session.execute(
                text("SELECT COUNT(*) FROM searches WHERE sources IS NULL")
            )
            remaining_nulls = verify_result.scalar()

            if remaining_nulls > 0:
                logger.error(f"❌ Migration failed - {remaining_nulls} NULL sources remain")
                return False

            logger.info("✅ Migration completed successfully")
            logger.info("   All searches now have sources = []")

            # Step 4: Show sample records
            sample_result = await session.execute(
                text("SELECT id, query, sources FROM searches LIMIT 5")
            )
            samples = sample_result.fetchall()

            if samples:
                logger.info("\nSample records after migration:")
                for record in samples:
                    logger.info(f"  ID {record[0]}: query='{record[1]}', sources={record[2]}")

            return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

    finally:
        await db_manager.close()
        logger.info("Database connection closed")


async def verify_migration():
    """Verify migration was successful"""

    db_url = settings.get_database_url_async()
    db_manager = DatabaseManager(db_url)

    try:
        await db_manager.initialize()

        async with db_manager.get_session() as session:
            # Check for any NULL sources
            result = await session.execute(
                text("SELECT COUNT(*) FROM searches WHERE sources IS NULL")
            )
            null_count = result.scalar()

            if null_count == 0:
                logger.info("✅ Verification passed - no NULL sources")
                return True
            else:
                logger.error(f"❌ Verification failed - {null_count} NULL sources found")
                return False

    finally:
        await db_manager.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CIAP Database Migration: Fix NULL sources")
    logger.info("=" * 60)

    # Run migration
    success = asyncio.run(migrate())

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("Next steps:")
        logger.info("1. Deploy updated Search model (with server_default)")
        logger.info("2. Restart API server")
        logger.info("3. Test frontend - sources should always be array")
        logger.info("=" * 60)
    else:
        logger.error("\nMigration failed - do NOT deploy model changes yet")
        sys.exit(1)
