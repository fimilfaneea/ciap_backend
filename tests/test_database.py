"""
Complete Database Initialization Script for CIAP
Initializes all database models including FTS5 and verifies setup
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import db_manager
from src.core.db_ops import DatabaseOperations
from src.core.models import *
from sqlalchemy import select
from datetime import datetime


async def verify_tables(session):
    """Verify all expected tables are created"""
    from sqlalchemy import text

    expected_tables = [
        'products', 'price_data', 'offers', 'product_reviews',
        'competitors', 'competitor_products', 'market_trends',
        'serp_data', 'social_sentiment', 'news_content',
        'feature_comparisons', 'insights', 'cache', 'task_queue',
        'scraping_jobs', 'rate_limits', 'price_history',
        'competitor_tracking', 'searches', 'search_results',
        'analyses'
    ]

    # Check for FTS5 virtual tables
    fts_tables = [
        'products_fts', 'competitors_fts', 'news_content_fts', 'product_reviews_fts'
    ]

    print("\n[OK] Verifying database tables...")

    # Get all tables from database
    result = await session.execute(text("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """))
    existing_tables = [row[0] for row in result]

    # Check regular tables
    missing_tables = []
    for table in expected_tables:
        if table in existing_tables:
            print(f"  [+] {table}")
        else:
            print(f"  [-] {table} - MISSING!")
            missing_tables.append(table)

    # Check FTS5 tables
    print("\n[OK] Verifying FTS5 virtual tables...")
    for table in fts_tables:
        if table in existing_tables:
            print(f"  [+] {table}")
        else:
            print(f"  [-] {table} - MISSING!")
            missing_tables.append(table)

    return len(missing_tables) == 0, existing_tables


async def test_basic_operations():
    """Test basic CRUD operations on new models"""
    print("\n[OK] Testing basic database operations...")

    async with db_manager.get_session() as session:
        try:
            # Test Product creation
            product = await DatabaseOperations.create_or_update_product(
                session,
                {
                    'product_name': 'Test Product Complete',
                    'brand_name': 'Test Brand',
                    'company_name': 'Test Company',
                    'category': 'Test Category',
                    'description': 'Complete test product for database validation',
                    'sku': 'TEST-COMPLETE-001'
                }
            )
            print(f"  [+] Created product: {product.product_name}")

            # Test Competitor creation
            competitor = await DatabaseOperations.create_or_update_competitor(
                session,
                {
                    'company_name': 'Test Competitor Complete Inc.',
                    'domain': 'testcompetitorcomplete.com',
                    'description': 'A test competitor for complete validation',
                    'products_services': ['Service X', 'Service Y'],
                    'market_share': 15.5
                }
            )
            print(f"  [+] Created competitor: {competitor.company_name}")

            # Test CompetitorProducts relationship
            relationship = await DatabaseOperations.link_competitor_product(
                session,
                competitor.id,
                product.id,
                "direct_competitor"
            )
            print(f"  [+] Linked competitor {competitor.id} with product {product.id}")

            # Test Insights creation
            insight = await DatabaseOperations.create_insight(
                session,
                insight_type="price_trend",
                title="Price Drop Detected",
                description="Significant price reduction observed",
                severity="high",
                confidence_score=0.85,
                product_id=product.id,
                insight_data={"old_price": 100, "new_price": 75},
                action_items=["Monitor competitor response", "Consider price adjustment"]
            )
            print(f"  [+] Created insight: {insight.title}")

            # Test FTS5 search
            print("\n[OK] Testing FTS5 full-text search...")

            # Search for product
            products = await DatabaseOperations.search_products_fts(
                session,
                "Test Product",
                limit=5
            )
            print(f"  [+] FTS5 product search returned {len(products)} results")

            # Commit all changes
            await session.commit()
            print("\n  [+] All test operations completed successfully")

            return True

        except Exception as e:
            print(f"\n  [-] Test operations failed: {e}")
            await session.rollback()
            return False


async def test_new_operations():
    """Test newly implemented database operations"""
    print("\n[OK] Testing newly implemented operations...")

    async with db_manager.get_session() as session:
        try:
            # Create a test search for SearchResult operations
            search = await DatabaseOperations.create_search(
                session,
                query="test search for new operations",
                sources=["google", "bing"],
                search_type="test"
            )
            print(f"  [+] Created test search with ID: {search.id}")

            # Test SearchResult operations
            print("\n  Testing SearchResult operations:")
            test_results = [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "snippet": "Test snippet 1",
                    "source": "google",
                    "rank": 1,
                    "metadata": {"score": 0.95}
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "snippet": "Test snippet 2",
                    "source": "bing",
                    "rank": 1,
                    "metadata": {"score": 0.90}
                }
            ]

            results = await DatabaseOperations.bulk_insert_results(
                session, search.id, test_results
            )
            print(f"    [+] Bulk inserted {len(results)} search results")

            all_results = await DatabaseOperations.get_search_results(session, search.id)
            print(f"    [+] Retrieved {len(all_results)} search results")

            if results:
                await DatabaseOperations.update_result_analysis(
                    session, results[0].id, {"sentiment_score": 0.8, "keywords": ["test", "example"]}
                )
                print(f"    [+] Updated analysis for result ID: {results[0].id}")

            count = await DatabaseOperations.count_results(session, search.id)
            print(f"    [+] Counted {count} results for search")

            # Test Task Queue operations
            print("\n  Testing Task Queue operations:")
            task = await DatabaseOperations.enqueue_task(
                session, "test_task", {"data": "test"}, priority=10
            )
            print(f"    [+] Enqueued task with ID: {task.id}")

            pending_tasks = await DatabaseOperations.get_pending_tasks(session, limit=5)
            print(f"    [+] Retrieved {len(pending_tasks)} pending tasks")

            # Test Cache operations
            print("\n  Testing Cache operations:")
            await DatabaseOperations.set_cache(
                session, "test_key", "test_value", ttl_seconds=3600
            )
            print("    [+] Set cache entry")

            cached_value = await DatabaseOperations.get_cache(session, "test_key")
            print(f"    [+] Retrieved cached value: {cached_value}")

            deleted = await DatabaseOperations.delete_cache(session, "test_key")
            print(f"    [+] Deleted cache entry: {deleted}")

            # Test Rate Limit operations
            print("\n  Testing Rate Limit operations:")
            await DatabaseOperations.update_rate_limit(session, "test_scraper")
            print("    [+] Updated rate limit")

            can_proceed = await DatabaseOperations.check_rate_limit(
                session, "test_scraper", requests_per_minute=10
            )
            print(f"    [+] Rate limit check: {can_proceed}")

            reset_count = await DatabaseOperations.reset_rate_limits(session, "test_scraper")
            print(f"    [+] Reset rate limits: {reset_count} entries")

            await session.commit()
            print("\n  [+] All new operations tested successfully")

            return True

        except Exception as e:
            print(f"\n  [-] New operations test failed: {e}")
            await session.rollback()
            return False


async def test_bulk_operations():
    """Test all bulk operations including chunking, batching, streaming, and pagination"""
    print("\n[OK] Testing bulk operations...")

    async with db_manager.get_session() as session:
        try:
            # 1. Test chunked bulk insert
            print("\n  Testing chunked bulk insert:")
            search = await DatabaseOperations.create_search(
                session, "bulk test", ["google"], "test"
            )

            # Create large dataset
            large_results = [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "snippet": f"Test snippet {i}",
                    "source": "google",
                    "rank": i
                }
                for i in range(1000)  # 1000 results to test chunking
            ]

            # Progress tracking
            progress_updates = []
            async def track_progress(current, total, processed):
                progress_updates.append((current, total, processed))
                print(f"    Chunk {current}/{total}: {processed} items processed")

            results = await DatabaseOperations.bulk_insert_results_chunked(
                session, search.id, large_results,
                chunk_size=100, progress_callback=track_progress
            )
            print(f"    [+] Inserted {len(results)} results in chunks")
            assert len(progress_updates) == 10  # 1000/100 = 10 chunks
            assert len(results) == 1000

            # 2. Test batch updates
            print("\n  Testing batch updates:")
            # Prepare update data for first 50 results
            update_data = []
            for i, r in enumerate(results[:50]):
                update_data.append({
                    "id": r.id,
                    "sentiment_score": 0.5 + (i * 0.01)  # Varying scores
                })

            updated_count = await DatabaseOperations.batch_update_search_results(
                session, update_data
            )
            print(f"    [+] Batch updated {updated_count} results")
            assert updated_count == 50

            # Verify updates
            updated_result = await session.execute(
                select(SearchResult).where(SearchResult.id == results[0].id)
            )
            first_result = updated_result.scalar_one()
            assert first_result.sentiment_score == 0.5

            # 3. Test streaming queries
            print("\n  Testing streaming queries:")
            total_streamed = 0
            chunk_count = 0
            async for chunk in DatabaseOperations.stream_search_results(
                session, search.id, chunk_size=100
            ):
                total_streamed += len(chunk)
                chunk_count += 1
                print(f"    Received chunk {chunk_count} with {len(chunk)} items")

            print(f"    [+] Streamed {total_streamed} total results in {chunk_count} chunks")
            assert total_streamed == 1000
            assert chunk_count == 10

            # 4. Test pagination
            print("\n  Testing pagination:")
            page_result = await DatabaseOperations.get_paginated_searches(
                session, page=1, per_page=10
            )

            print(f"    [+] Page 1/{page_result.total_pages}")
            print(f"    Items: {len(page_result.items)}")
            print(f"    Total: {page_result.total}")
            print(f"    Has next: {page_result.has_next}")
            print(f"    Has prev: {page_result.has_prev}")

            assert page_result.page == 1
            assert page_result.per_page == 10
            assert not page_result.has_prev  # First page
            assert len(page_result.items) <= 10

            # Test page navigation
            if page_result.has_next:
                page2 = await DatabaseOperations.get_paginated_searches(
                    session, page=2, per_page=10
                )
                print(f"    [+] Page 2 retrieved with {len(page2.items)} items")
                assert page2.has_prev  # Should have previous page
                assert page2.page == 2

            # 5. Test performance utilities
            print("\n  Testing performance utilities:")

            # Test count with filters
            count = await DatabaseOperations.count_with_filters(
                session, SearchResult, {"search_id": search.id}
            )
            print(f"    [+] Count with filters: {count} results")
            assert count == 1000

            # Test table statistics
            stats = await DatabaseOperations.get_table_statistics(
                session, SearchResult
            )
            print(f"    [+] Table statistics:")
            print(f"       Total records: {stats['total_records']}")
            print(f"       Created today: {stats['created_today']}")
            assert stats['total_records'] >= 1000

            # 6. Test batch task updates
            print("\n  Testing batch task updates:")
            # Create some tasks
            task_ids = []
            for i in range(5):
                task = await DatabaseOperations.enqueue_task(
                    session, f"test_task_{i}", {"data": f"test_{i}"}, priority=i
                )
                task_ids.append(task.id)

            # Batch update their status
            updated = await DatabaseOperations.batch_update_task_status(
                session, task_ids, "processing"
            )
            print(f"    [+] Batch updated {updated} tasks to processing")
            assert updated == 5

            # 7. Test chunked reviews insert
            print("\n  Testing chunked reviews insert:")
            # Create a test product first
            product = await DatabaseOperations.create_or_update_product(
                session,
                {
                    'product_name': 'Bulk Test Product',
                    'brand_name': 'Test Brand',
                    'company_name': 'Test Company',
                    'category': 'Test Category',
                    'sku': 'BULK-TEST-001'
                }
            )

            # Create large review dataset
            reviews = [
                {
                    "review_title": f"Review {i}",
                    "review_text": f"This is review text {i}",
                    "rating": 3.0 + (i % 3),  # Ratings between 3-5
                    "reviewer_name": f"Reviewer {i}",
                    "review_date": datetime.utcnow(),
                    "verified_purchase": i % 2 == 0  # Every other is verified
                }
                for i in range(500)
            ]

            reviews_inserted = await DatabaseOperations.bulk_add_reviews_chunked(
                session, product.id, reviews, chunk_size=50
            )
            print(f"    [+] Inserted {reviews_inserted} reviews in chunks")
            assert reviews_inserted == 500

            # Test paginated reviews
            review_page = await DatabaseOperations.get_paginated_reviews(
                session, product.id, page=1, per_page=20, min_rating=4.0
            )
            print(f"    [+] Paginated reviews: {len(review_page.items)} items (filtered by rating >= 4.0)")

            # 8. Test streaming reviews
            print("\n  Testing streaming reviews:")
            review_chunks = 0
            review_total = 0
            async for chunk in DatabaseOperations.stream_reviews(
                session, product.id, chunk_size=50
            ):
                review_chunks += 1
                review_total += len(chunk)

            print(f"    [+] Streamed {review_total} reviews in {review_chunks} chunks")
            assert review_total == 500

            await session.commit()
            print("\n  [+] All bulk operations tested successfully")
            return True

        except Exception as e:
            print(f"\n  [-] Bulk operations test failed: {e}")
            import traceback
            traceback.print_exc()
            await session.rollback()
            return False


async def init_complete_database():
    """Initialize complete database with all models and FTS5"""
    print("=" * 60)
    print("CIAP Complete Database Initialization")
    print("=" * 60)

    try:
        # Initialize database
        print("\n1. Initializing database with all models...")
        await db_manager.initialize()
        print("[+] Database initialized successfully")

        # Verify health
        is_healthy = await db_manager.health_check()
        print(f"[+] Health check: {'Healthy' if is_healthy else 'Unhealthy'}")

        # Verify all tables
        async with db_manager.get_session() as session:
            success, tables = await verify_tables(session)

        if not success:
            print("\n[!] Warning: Some tables are missing!")
            return False

        # Get statistics
        stats = await db_manager.get_stats()

        print("\n>> Database Statistics:")
        print(f"  - Database size: {stats.get('database_size_bytes', 0) / 1024:.2f} KB")
        print(f"  - Total tables: {len(tables)}")

        # Count records in key tables
        for key, value in stats.items():
            if key.endswith('_count'):
                table_name = key.replace('_count', '')
                print(f"  - {table_name}: {value} records")

        # Test basic operations
        test_success = await test_basic_operations()

        # Test newly implemented operations
        if test_success:
            test_success = await test_new_operations()

        # Test bulk operations
        if test_success:
            test_success = await test_bulk_operations()

        if test_success:
            print("\n" + "=" * 60)
            print("[SUCCESS] DATABASE INITIALIZATION COMPLETE!")
            print("=" * 60)

            print("\n>> Summary:")
            print(f"  - Regular tables: 21")
            print(f"  - FTS5 virtual tables: 4")
            print(f"  - Total tables: {len(tables)}")
            print("  - All models working: [+]")
            print("  - FTS5 search enabled: [+]")
            print("  - Relationships configured: [+]")
            print("  - Insights system ready: [+]")
            print("  - SearchResult operations: [+]")
            print("  - Task Queue operations: [+]")
            print("  - Cache operations: [+]")
            print("  - Rate Limiting operations: [+]")
            print("  - Bulk operations: [+]")
            print("  - Chunked inserts: [+]")
            print("  - Batch updates: [+]")
            print("  - Streaming queries: [+]")
            print("  - Pagination: [+]")
            print("  - Performance utilities: [+]")

            print("\n>> Your CIAP database is ready for use!")
            print("  - Location: data/ciap.db")
            print("  - All competitive intelligence models active")
            print("  - Full-text search operational")
            print("  - Junction tables configured")

            return True
        else:
            print("\n[!] Some tests failed. Please check the errors above.")
            return False

    except Exception as e:
        print(f"\n[ERROR] INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await db_manager.close()
        print("\n[+] Database connection closed")


if __name__ == "__main__":
    # Ensure scripts directory exists
    Path("scripts").mkdir(exist_ok=True)

    # Run initialization
    success = asyncio.run(init_complete_database())

    # Exit with appropriate code
    sys.exit(0 if success else 1)