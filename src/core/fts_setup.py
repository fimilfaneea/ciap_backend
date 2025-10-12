"""
FTS5 Full-Text Search Setup for CIAP
Implements SQLite's FTS5 virtual tables for fast text searching
"""

from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


async def setup_fts5(engine):
    """
    Setup FTS5 full-text search for products and other searchable content

    Args:
        engine: SQLAlchemy async engine
    """
    logger.info("Setting up FTS5 full-text search...")

    async with engine.begin() as conn:
        # Create FTS5 virtual table for products
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
                product_name,
                brand_name,
                description,
                category,
                content='products',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """))
        logger.info("Created products_fts virtual table")

        # Create triggers to keep FTS in sync with products table
        # Trigger for INSERT
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS products_fts_insert
            AFTER INSERT ON products BEGIN
                INSERT INTO products_fts(rowid, product_name, brand_name, description, category)
                VALUES (new.id, new.product_name, new.brand_name, new.description, new.category);
            END
        """))

        # Trigger for DELETE
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS products_fts_delete
            AFTER DELETE ON products BEGIN
                DELETE FROM products_fts WHERE rowid = old.id;
            END
        """))

        # Trigger for UPDATE
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS products_fts_update
            AFTER UPDATE ON products BEGIN
                UPDATE products_fts
                SET product_name = new.product_name,
                    brand_name = new.brand_name,
                    description = new.description,
                    category = new.category
                WHERE rowid = new.id;
            END
        """))
        logger.info("Created FTS synchronization triggers for products")

        # Create FTS5 virtual table for competitors
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS competitors_fts USING fts5(
                company_name,
                description,
                target_audience,
                content='competitors',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """))
        logger.info("Created competitors_fts virtual table")

        # Triggers for competitors
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS competitors_fts_insert
            AFTER INSERT ON competitors BEGIN
                INSERT INTO competitors_fts(rowid, company_name, description, target_audience)
                VALUES (new.id, new.company_name, new.description, new.target_audience);
            END
        """))

        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS competitors_fts_delete
            AFTER DELETE ON competitors BEGIN
                DELETE FROM competitors_fts WHERE rowid = old.id;
            END
        """))

        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS competitors_fts_update
            AFTER UPDATE ON competitors BEGIN
                UPDATE competitors_fts
                SET company_name = new.company_name,
                    description = new.description,
                    target_audience = new.target_audience
                WHERE rowid = new.id;
            END
        """))
        logger.info("Created FTS synchronization triggers for competitors")

        # Create FTS5 virtual table for news content
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS news_content_fts USING fts5(
                article_title,
                article_summary,
                full_content,
                content='news_content',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """))
        logger.info("Created news_content_fts virtual table")

        # Triggers for news content
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS news_content_fts_insert
            AFTER INSERT ON news_content BEGIN
                INSERT INTO news_content_fts(rowid, article_title, article_summary, full_content)
                VALUES (new.id, new.article_title, new.article_summary, new.full_content);
            END
        """))

        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS news_content_fts_delete
            AFTER DELETE ON news_content BEGIN
                DELETE FROM news_content_fts WHERE rowid = old.id;
            END
        """))

        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS news_content_fts_update
            AFTER UPDATE ON news_content BEGIN
                UPDATE news_content_fts
                SET article_title = new.article_title,
                    article_summary = new.article_summary,
                    full_content = new.full_content
                WHERE rowid = new.id;
            END
        """))
        logger.info("Created FTS synchronization triggers for news content")

        # Create FTS5 virtual table for reviews
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS product_reviews_fts USING fts5(
                review_title,
                review_text,
                content='product_reviews',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """))
        logger.info("Created product_reviews_fts virtual table")

        # Triggers for product reviews
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS product_reviews_fts_insert
            AFTER INSERT ON product_reviews BEGIN
                INSERT INTO product_reviews_fts(rowid, review_title, review_text)
                VALUES (new.id, new.review_title, new.review_text);
            END
        """))

        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS product_reviews_fts_delete
            AFTER DELETE ON product_reviews BEGIN
                DELETE FROM product_reviews_fts WHERE rowid = old.id;
            END
        """))

        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS product_reviews_fts_update
            AFTER UPDATE ON product_reviews BEGIN
                UPDATE product_reviews_fts
                SET review_title = new.review_title,
                    review_text = new.review_text
                WHERE rowid = new.id;
            END
        """))
        logger.info("Created FTS synchronization triggers for product reviews")

        # Rebuild FTS indexes for existing data
        await rebuild_fts_indexes(conn)

    logger.info("FTS5 setup completed successfully")


async def rebuild_fts_indexes(conn):
    """
    Rebuild FTS indexes for existing data

    Args:
        conn: Database connection
    """
    try:
        # Rebuild products FTS
        await conn.execute(text("INSERT INTO products_fts(products_fts) VALUES('rebuild')"))
        logger.info("Rebuilt products_fts index")
    except Exception as e:
        logger.debug(f"No existing products to index: {e}")

    try:
        # Rebuild competitors FTS
        await conn.execute(text("INSERT INTO competitors_fts(competitors_fts) VALUES('rebuild')"))
        logger.info("Rebuilt competitors_fts index")
    except Exception as e:
        logger.debug(f"No existing competitors to index: {e}")

    try:
        # Rebuild news content FTS
        await conn.execute(text("INSERT INTO news_content_fts(news_content_fts) VALUES('rebuild')"))
        logger.info("Rebuilt news_content_fts index")
    except Exception as e:
        logger.debug(f"No existing news content to index: {e}")

    try:
        # Rebuild product reviews FTS
        await conn.execute(text("INSERT INTO product_reviews_fts(product_reviews_fts) VALUES('rebuild')"))
        logger.info("Rebuilt product_reviews_fts index")
    except Exception as e:
        logger.debug(f"No existing product reviews to index: {e}")


async def search_products_fts(conn, search_query: str, limit: int = 20):
    """
    Search products using FTS5

    Args:
        conn: Database connection
        search_query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching product IDs with relevance scores
    """
    # Use FTS5 MATCH for searching
    query = text("""
        SELECT rowid, rank
        FROM products_fts
        WHERE products_fts MATCH :query
        ORDER BY rank
        LIMIT :limit
    """)

    result = await conn.execute(query, {"query": search_query, "limit": limit})
    return result.fetchall()


async def search_competitors_fts(conn, search_query: str, limit: int = 20):
    """
    Search competitors using FTS5

    Args:
        conn: Database connection
        search_query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching competitor IDs with relevance scores
    """
    query = text("""
        SELECT rowid, rank
        FROM competitors_fts
        WHERE competitors_fts MATCH :query
        ORDER BY rank
        LIMIT :limit
    """)

    result = await conn.execute(query, {"query": search_query, "limit": limit})
    return result.fetchall()


async def search_news_fts(conn, search_query: str, limit: int = 20):
    """
    Search news content using FTS5

    Args:
        conn: Database connection
        search_query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching news IDs with relevance scores
    """
    query = text("""
        SELECT rowid, rank
        FROM news_content_fts
        WHERE news_content_fts MATCH :query
        ORDER BY rank
        LIMIT :limit
    """)

    result = await conn.execute(query, {"query": search_query, "limit": limit})
    return result.fetchall()


async def search_reviews_fts(conn, search_query: str, limit: int = 20):
    """
    Search product reviews using FTS5

    Args:
        conn: Database connection
        search_query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching review IDs with relevance scores
    """
    query = text("""
        SELECT rowid, rank
        FROM product_reviews_fts
        WHERE product_reviews_fts MATCH :query
        ORDER BY rank
        LIMIT :limit
    """)

    result = await conn.execute(query, {"query": search_query, "limit": limit})
    return result.fetchall()