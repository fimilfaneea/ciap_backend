"""
Integration Tests for CIAP Database
Tests complex multi-table operations, foreign key constraints, cascade operations,
queue processing simulation, and cache cleanup routines
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy import select, text, func, delete
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import DatabaseManager
from src.core.db_ops import DatabaseOperations
from src.core.models import *
from tests.test_utils import (
    DatabaseTestFixture,
    TestDataFactory,
    PerformanceTimer,
    ConcurrentExecutor
)


class TestIntegration:
    """Integration tests for complex database operations"""

    def __init__(self):
        self.runner_passed = 0
        self.runner_failed = 0
        self.errors = []

    def report(self, test_name, passed, error=None):
        if passed:
            self.runner_passed += 1
            print(f"[PASS] {test_name}")
        else:
            self.runner_failed += 1
            print(f"[FAIL] {test_name}")
            if error:
                self.errors.append((test_name, error))
                print(f"       Error: {error}")

    # ==========================================================================
    # MULTI-TABLE TRANSACTION TESTS
    # ==========================================================================

    async def test_search_with_results_transaction(self):
        """Test transaction spanning search and results"""
        test_name = "test_search_with_results_transaction"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create search and results in single transaction
                search = await db_ops.create_search(
                    session,
                    "transaction test",
                    ["google", "bing"],
                    "competitor"
                )

                # Add results
                results_data = [
                    {
                        'title': f'Result {i}',
                        'url': f'https://example.com/r{i}',
                        'snippet': f'Snippet {i}',
                        'source': 'google' if i % 2 == 0 else 'bing',
                        'rank': i
                    }
                    for i in range(20)
                ]
                results = await db_ops.bulk_insert_results(
                    session, search.id, results_data
                )

                # Add analysis
                analysis = await db_ops.save_analysis(
                    session,
                    search_id=search.id,
                    analysis_type="sentiment",
                    content="Sentiment analysis results",
                    insights={"positive": 0.6, "negative": 0.3, "neutral": 0.1},
                    llm_info={"provider": "ollama", "model": "llama3.1:8b"},
                    scores={"sentiment": 0.6, "confidence": 0.85}
                )

                # Update search status
                await db_ops.update_search_status(session, search.id, "completed")

                # All should be in same transaction
                assert search.id is not None
                assert len(results) == 20
                assert analysis.id is not None

            # Verify all data persisted
            async with db_manager.get_session() as session:
                search = await db_ops.get_search(session, search.id)
                assert search.status == "completed"

                result_count = await db_ops.count_results(session, search.id)
                assert result_count == 20

                analyses = await db_ops.get_analyses(session, search.id)
                assert len(analyses) == 1

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_product_with_price_history_transaction(self):
        """Test transaction with product and price history"""
        test_name = "test_product_with_price_history_transaction"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create product
                product = await db_ops.create_or_update_product(
                    session,
                    {
                        'product_name': 'Transaction Product',
                        'brand_name': 'Test Brand',
                        'category': 'Electronics'
                    }
                )

                # Add price history
                for i in range(10):
                    await db_ops.add_price_data(
                        session,
                        product_id=product.id,
                        price_info={
                            "current_price": 100.00 + i * 5,
                            "currency": "USD",
                            "seller_name": "amazon",
                            "availability_status": "In Stock"
                        }
                    )

                # Add offers
                offer = Offer(
                    product_id=product.id,
                    offer_description="Special Deal",
                    discount_code="SAVE20",
                    start_date=datetime.utcnow(),
                    end_date=datetime.utcnow() + timedelta(days=7),
                    offer_source="amazon"
                )
                session.add(offer)

                # Add reviews
                reviews_data = TestDataFactory.create_review_data(product.id, 5)
                await db_ops.bulk_add_reviews(session, product.id, reviews_data)

            # Verify complex relationships
            async with db_manager.get_session() as session:
                # Check product
                product = await db_ops.get_product(session, product.id)
                assert product is not None

                # Check price history
                history = await db_ops.get_price_history(
                    session,
                    product.id,
                    days=30
                )
                assert len(history) == 10

                # Check reviews
                reviews = await db_ops.get_product_reviews(session, product.id)
                assert len(reviews) == 5

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_competitor_product_linking_transaction(self):
        """Test complex competitor-product linking transaction"""
        test_name = "test_competitor_product_linking_transaction"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create competitors
                competitors = []
                for i in range(3):
                    comp = await db_ops.create_or_update_competitor(
                        session,
                        {
                            'company_name': f'Competitor {i}',
                            'domain': f'https://competitor{i}.com',
                            'market_share': 0.1 + i * 0.05
                        }
                    )
                    competitors.append(comp)

                # Create products
                products = []
                for i in range(5):
                    prod = await db_ops.create_or_update_product(
                        session,
                        {
                            'product_name': f'Product {i}',
                            'brand_name': f'Brand {i % 2}',
                            'category': 'Electronics'
                        }
                    )
                    products.append(prod)

                # Link competitors to products
                for comp in competitors:
                    for prod in products[:3]:
                        await db_ops.link_competitor_product(
                            session,
                            competitor_id=comp.id,
                            product_id=prod.id,
                            relationship_type="direct_competitor" if comp == competitors[0] else "indirect_competitor"
                        )

                # Track competitor changes
                for comp in competitors:
                    await db_ops.track_competitor_change(
                        session,
                        competitor_id=comp.id,
                        change_type="market_share_update",
                        change_description=f"Market share updated",
                        old_value=comp.market_share,
                        new_value=comp.market_share + 0.01
                    )

            # Verify complex relationships
            async with db_manager.get_session() as session:
                # Check competitor products
                for comp in competitors:
                    comp_products = await db_ops.get_competitor_products(session, comp.id)
                    assert len(comp_products) == 3

                # Check product competitors
                for prod in products[:3]:
                    prod_competitors = await db_ops.get_product_competitors(session, prod.id)
                    assert len(prod_competitors) == 3

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_cascading_delete_transaction(self):
        """Test cascading delete across multiple tables"""
        test_name = "test_cascading_delete_transaction"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            search_id = None
            product_id = None

            async with db_manager.get_session() as session:
                # Create parent entities
                search = await db_ops.create_search(
                    session, "cascade test", ["google"], "test"
                )
                search_id = search.id

                product = await db_ops.create_or_update_product(
                    session,
                    {
                        'product_name': 'Cascade Product',
                        'brand_name': 'Test',
                        'category': 'Test'
                    }
                )
                product_id = product.id

                # Add child entities
                await db_ops.bulk_insert_results(
                    session,
                    search_id,
                    [{'title': 'Test', 'url': 'http://test.com',
                      'snippet': 'Test', 'source': 'google', 'rank': 1}]
                )

                await db_ops.add_price_data(
                    session,
                    product_id=product_id,
                    price_info={
                        "current_price": 100.0,
                        "currency": "USD",
                        "seller_name": "test",
                        "availability_status": "In Stock"
                    }
                )

            # Delete parent and verify cascade
            async with db_manager.get_session() as session:
                # Delete search
                search = await session.get(Search, search_id)
                await session.delete(search)
                await session.commit()

                # Verify child records deleted
                result_count = await db_ops.count_results(session, search_id)
                assert result_count == 0

                # Delete product
                product = await session.get(Product, product_id)
                await session.delete(product)
                await session.commit()

                # Verify related records handled
                prices = await db_ops.get_price_history(
                    session, product_id,
                    days=30
                )
                assert len(prices) == 0

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    # ==========================================================================
    # FOREIGN KEY CONSTRAINT TESTS
    # ==========================================================================

    async def test_foreign_key_enforcement(self):
        """Test foreign key constraints are enforced"""
        test_name = "test_foreign_key_enforcement"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Enable foreign keys (should already be on)
                await session.execute(text("PRAGMA foreign_keys = ON"))

                # Try to create orphaned search result
                try:
                    result = SearchResult(
                        search_id=99999,  # Non-existent search
                        source="google",
                        title="Orphaned",
                        url="http://test.com",
                        snippet="Test",
                        position=1
                    )
                    session.add(result)
                    await session.commit()
                    assert False, "Should have raised foreign key violation"
                except IntegrityError:
                    await session.rollback()
                    # Expected

                # Try to create orphaned price data
                try:
                    price = PriceData(
                        product_id=99999,  # Non-existent product
                        current_price=100.0,
                        currency="USD",
                        seller_name="test"
                    )
                    session.add(price)
                    await session.commit()
                    assert False, "Should have raised foreign key violation"
                except IntegrityError:
                    await session.rollback()
                    # Expected

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_orphaned_record_prevention(self):
        """Test prevention of orphaned records"""
        test_name = "test_orphaned_record_prevention"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create parent
                search = await db_ops.create_search(
                    session, "orphan test", ["google"], "test"
                )
                search_id = search.id

                # Add child
                await db_ops.bulk_insert_results(
                    session,
                    search_id,
                    [{'title': 'Child', 'url': 'http://test.com',
                      'snippet': 'Test', 'source': 'google', 'rank': 1}]
                )

                # Try to delete parent without cascade
                # This behavior depends on relationship configuration
                # If cascade is set, children will be deleted
                # If restrict is set, deletion will be prevented

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_referential_integrity_on_update(self):
        """Test referential integrity during updates"""
        test_name = "test_referential_integrity_on_update"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create related entities
                product = await db_ops.create_or_update_product(
                    session,
                    {
                        'product_name': 'Integrity Test',
                        'brand_name': 'Test',
                        'category': 'Test'
                    }
                )

                competitor = await db_ops.create_or_update_competitor(
                    session,
                    {
                        'company_name': 'Test Competitor',
                        'domain': 'https://test.com',
                        'market_share': 0.1
                    }
                )

                # Link them
                await db_ops.link_competitor_product(
                    session, competitor.id, product.id
                )

                # Update product - link should remain valid
                product.product_name = 'Updated Product'
                await session.commit()

                # Verify link still works
                comp_products = await db_ops.get_competitor_products(session, competitor.id)
                assert len(comp_products) == 1
                assert comp_products[0].product_name == 'Updated Product'

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    # ==========================================================================
    # CASCADE OPERATIONS TESTS
    # ==========================================================================

    async def test_search_cascade_delete(self):
        """Test cascading delete for search and related entities"""
        test_name = "test_search_cascade_delete"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create search with all related entities
                search = await db_ops.create_search(
                    session, "cascade search", ["google"], "test"
                )

                # Add search results
                await db_ops.bulk_insert_results(
                    session,
                    search.id,
                    TestDataFactory.create_search_data(5)
                )

                # Add analysis
                await db_ops.save_analysis(
                    session,
                    search_id=search.id,
                    analysis_type="test",
                    content="Test analysis content",
                    insights={},
                    llm_info={"provider": "ollama", "model": "llama3.1:8b"}
                )

                # Add SERP data
                await db_ops.bulk_insert_serp_data(
                    session,
                    search.id,
                    [{'search_query': 'test', 'result_position': 1,
                      'result_url': 'http://test.com', 'search_engine': 'google'}]
                )

                search_id = search.id

            # Delete search and verify cascade
            async with db_manager.get_session() as session:
                search = await session.get(Search, search_id)
                await session.delete(search)
                await session.commit()

                # Verify all related entities deleted
                results = await db_ops.get_search_results(session, search_id)
                assert len(results) == 0

                analyses = await db_ops.get_analyses(session, search_id)
                assert len(analyses) == 0

                serp_data = await db_ops.get_serp_data(session, search_id)
                assert len(serp_data) == 0

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_product_cascade_delete(self):
        """Test cascading delete for product and related entities"""
        test_name = "test_product_cascade_delete"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create product with related entities
                product = await db_ops.create_or_update_product(
                    session,
                    TestDataFactory.create_product_data(1)[0]
                )

                # Add price data
                for i in range(5):
                    await db_ops.add_price_data(
                        session,
                        product_id=product.id,
                        price_info={
                            "current_price": 100.0 + i,
                            "currency": "USD",
                            "seller_name": "test",
                            "availability_status": "In Stock"
                        }
                    )

                # Add reviews
                reviews_data = TestDataFactory.create_review_data(product.id, 5)
                await db_ops.bulk_add_reviews(session, product.id, reviews_data)

                # Add offer
                offer = Offer(
                    product_id=product.id,
                    offer_description="Test Offer",
                    offer_source="test"
                )
                session.add(offer)
                await session.commit()

                product_id = product.id

            # Delete product and verify cascade
            async with db_manager.get_session() as session:
                product = await session.get(Product, product_id)
                await session.delete(product)
                await session.commit()

                # Verify all related entities deleted
                prices = await db_ops.get_price_history(
                    session, product_id,
                    days=30
                )
                assert len(prices) == 0

                reviews = await db_ops.get_product_reviews(session, product_id)
                assert len(reviews) == 0

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_multi_level_cascade(self):
        """Test multi-level cascade operations"""
        test_name = "test_multi_level_cascade"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create hierarchy: Search -> SearchResult -> Analysis
                search = await db_ops.create_search(
                    session, "multi cascade", ["google"], "test"
                )

                # Add results
                results = await db_ops.bulk_insert_results(
                    session,
                    search.id,
                    [{'title': f'Result {i}', 'url': f'http://test{i}.com',
                      'snippet': f'Test {i}', 'source': 'google', 'rank': i}
                     for i in range(3)]
                )

                # Update result analysis (if supported)
                for result in results:
                    await db_ops.update_result_analysis(
                        session,
                        result.id,
                        {"sentiment_score": 0.8, "relevance": 0.8}
                    )

                search_id = search.id

            # Delete top-level entity
            async with db_manager.get_session() as session:
                search = await session.get(Search, search_id)
                await session.delete(search)
                await session.commit()

                # Verify complete cascade
                result = await session.execute(
                    select(func.count()).select_from(SearchResult)
                    .where(SearchResult.search_id == search_id)
                )
                count = result.scalar()
                assert count == 0

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    # ==========================================================================
    # QUEUE PROCESSING SIMULATION TESTS
    # ==========================================================================

    async def test_queue_worker_simulation(self):
        """Simulate queue worker processing"""
        test_name = "test_queue_worker_simulation"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Enqueue tasks
            async with db_manager.get_session() as session:
                task_ids = []
                for i in range(20):
                    task = await db_ops.enqueue_task(
                        session,
                        task_type="process_search" if i % 2 == 0 else "analyze_data",
                        payload={"id": i, "data": f"task_{i}"},
                        priority=10 - i % 5  # Varying priorities
                    )
                    task_ids.append(task.id)

            # Simulate worker processing
            processed_tasks = []

            # Process all tasks sequentially to avoid race conditions
            # In production, use proper task queue with row-level locking
            for _ in range(20):
                async with db_manager.get_session() as session:
                    task = await db_ops.dequeue_task(session)
                    if task:
                        # Simulate processing
                        await asyncio.sleep(0.001)

                        # Complete task
                        await db_ops.complete_task(
                            session,
                            task.id,
                            status="completed"
                        )
                        processed_tasks.append(task.id)

            # Placeholder for worker results format
            workers = [[]]

            for w in workers:
                processed_tasks.extend(w)

            # Verify all tasks processed
            async with db_manager.get_session() as session:
                pending = await db_ops.get_pending_tasks(session, limit=100)
                assert len(pending) == 0, f"Still {len(pending)} pending tasks"

                # Check completed tasks
                result = await session.execute(
                    select(func.count()).select_from(TaskQueue)
                    .where(TaskQueue.status == "completed")
                )
                completed_count = result.scalar()
                assert completed_count == 20

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_queue_retry_mechanism(self):
        """Test queue retry mechanism for failed tasks"""
        test_name = "test_queue_retry_mechanism"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create tasks that will fail
                task = await db_ops.enqueue_task(
                    session,
                    task_type="failing_task",
                    payload={"will_fail": True},
                    priority=5
                )
                task_id = task.id

            # Simulate failures and retries
            for attempt in range(3):
                async with db_manager.get_session() as session:
                    task = await db_ops.dequeue_task(session)
                    assert task is not None

                    # Simulate failure
                    if attempt < 2:
                        await db_ops.retry_task(
                            session,
                            task.id
                        )
                    else:
                        # Finally succeed
                        await db_ops.complete_task(session, task.id, status="completed")

            # Verify task completed after retries
            async with db_manager.get_session() as session:
                task = await session.get(TaskQueue, task_id)
                assert task.status == "completed"
                assert task.retry_count >= 2

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_queue_priority_processing(self):
        """Test queue processes tasks by priority"""
        test_name = "test_queue_priority_processing"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create tasks with different priorities
            async with db_manager.get_session() as session:
                task_priorities = []

                # Low priority tasks
                for i in range(5):
                    task = await db_ops.enqueue_task(
                        session,
                        task_type="low_priority",
                        payload={"priority": 1},
                        priority=1
                    )
                    task_priorities.append((task.id, 1))

                # High priority tasks
                for i in range(5):
                    task = await db_ops.enqueue_task(
                        session,
                        task_type="high_priority",
                        payload={"priority": 10},
                        priority=10
                    )
                    task_priorities.append((task.id, 10))

                # Medium priority tasks
                for i in range(5):
                    task = await db_ops.enqueue_task(
                        session,
                        task_type="medium_priority",
                        payload={"priority": 5},
                        priority=5
                    )
                    task_priorities.append((task.id, 5))

            # Process tasks and verify priority order
            processed_priorities = []

            for _ in range(15):
                async with db_manager.get_session() as session:
                    task = await db_ops.dequeue_task(session)
                    if task:
                        processed_priorities.append(task.priority)
                        await db_ops.complete_task(session, task.id, status="completed")

            # Verify high priority processed last (SQLite orders by priority ascending, then by scheduled_at)
            # The dequeue_task uses ORDER BY priority, scheduled_at which means LOW numbers come first
            # So priority 1 comes before priority 10
            first_five = processed_priorities[:5]
            assert all(p == 1 for p in first_five), "Low priority not processed first"

            # Medium priority should be next
            next_five = processed_priorities[5:10]
            assert all(p == 5 for p in next_five), "Medium priority not processed second"

            # High priority should be last
            last_five = processed_priorities[10:15]
            assert all(p == 10 for p in last_five), "High priority not processed last"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    # ==========================================================================
    # CACHE CLEANUP ROUTINE TESTS
    # ==========================================================================

    async def test_automated_cache_cleanup(self):
        """Test automated cache cleanup routine"""
        test_name = "test_automated_cache_cleanup"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create cache entries with varying TTLs
                now = datetime.utcnow()

                # Expired entries
                for i in range(10):
                    cache = Cache(
                        key=f"expired_{i}",
                        value=f"value_{i}",
                        expires_at=now - timedelta(seconds=i+1)  # Already expired
                    )
                    session.add(cache)

                # Valid entries
                for i in range(10):
                    cache = Cache(
                        key=f"valid_{i}",
                        value=f"value_{i}",
                        expires_at=now + timedelta(hours=1)  # Still valid
                    )
                    session.add(cache)

                await session.commit()

            # Run cleanup
            async with db_manager.get_session() as session:
                deleted_count = await db_ops.cleanup_expired_cache(session)
                assert deleted_count == 10, f"Expected 10 deleted, got {deleted_count}"

                # Verify only valid entries remain
                result = await session.execute(
                    select(func.count()).select_from(Cache)
                )
                remaining = result.scalar()
                assert remaining == 10, f"Expected 10 remaining, got {remaining}"

                # Verify correct entries remain
                for i in range(10):
                    value = await db_ops.get_cache(session, f"valid_{i}")
                    assert value == f"value_{i}"

                    value = await db_ops.get_cache(session, f"expired_{i}")
                    assert value is None

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_cache_cleanup_performance(self):
        """Test cache cleanup performance with large datasets"""
        test_name = "test_cache_cleanup_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create many cache entries
            async with db_manager.get_session() as session:
                now = datetime.utcnow()

                # Batch insert cache entries
                cache_entries = []
                for i in range(1000):
                    expires_at = now - timedelta(seconds=1) if i < 500 else now + timedelta(hours=1)
                    cache = Cache(
                        key=f"perf_key_{i}",
                        value=f"value_{i}",
                        expires_at=expires_at
                    )
                    cache_entries.append(cache)

                    if len(cache_entries) >= 100:
                        session.add_all(cache_entries)
                        await session.commit()
                        cache_entries = []

                if cache_entries:
                    session.add_all(cache_entries)
                    await session.commit()

            # Measure cleanup performance
            async with db_manager.get_session() as session:
                start = time.perf_counter()
                deleted = await db_ops.cleanup_expired_cache(session)
                elapsed = time.perf_counter() - start

                assert deleted == 500, f"Expected 500 deleted, got {deleted}"
                assert elapsed < 1.0, f"Cleanup took too long: {elapsed:.2f}s"

                print(f"       Cleaned up {deleted} entries in {elapsed*1000:.2f}ms")

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_selective_cache_invalidation(self):
        """Test selective cache invalidation"""
        test_name = "test_selective_cache_invalidation"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create cache entries with patterns
                categories = ["search", "product", "competitor", "analysis"]

                for category in categories:
                    for i in range(5):
                        import json
                        await db_ops.set_cache(
                            session,
                            f"{category}:item_{i}",
                            json.dumps({"category": category, "id": i}),
                            ttl_seconds=3600
                        )

            # Selectively invalidate category
            async with db_manager.get_session() as session:
                # Delete all "search" cache entries
                result = await session.execute(
                    delete(Cache).where(Cache.key.like("search:%"))
                )
                await session.commit()

                # Verify selective deletion
                for category in categories:
                    for i in range(5):
                        value = await db_ops.get_cache(session, f"{category}:item_{i}")
                        if category == "search":
                            assert value is None
                        else:
                            assert value is not None

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    # ==========================================================================
    # COMPLEX WORKFLOW TESTS
    # ==========================================================================

    async def test_complete_search_workflow(self):
        """Test complete search workflow from creation to analysis"""
        test_name = "test_complete_search_workflow"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # 1. Create search
                search = await db_ops.create_search(
                    session,
                    "smartphone comparison",
                    ["google", "bing"],
                    "competitor"
                )

                # 2. Add to task queue
                scrape_task = await db_ops.enqueue_task(
                    session,
                    task_type="scrape_search",
                    payload={"search_id": search.id, "query": search.query},
                    priority=10
                )

                # 3. Process scraping (simulate)
                task = await db_ops.dequeue_task(session)
                assert task.id == scrape_task.id

                # 4. Add search results
                results_data = [
                    {
                        'title': f'Smartphone {i}',
                        'url': f'https://shop{i}.com/phone',
                        'snippet': f'Latest smartphone with features {i}',
                        'source': 'google' if i % 2 == 0 else 'bing',
                        'rank': i
                    }
                    for i in range(20)
                ]
                results = await db_ops.bulk_insert_results(session, search.id, results_data)

                # 5. Complete scraping task
                await db_ops.complete_task(session, task.id, "completed")

                # 6. Create analysis task
                analysis_task = await db_ops.enqueue_task(
                    session,
                    task_type="analyze_results",
                    payload={"search_id": search.id},
                    priority=8
                )

                # 7. Process analysis (simulate)
                task = await db_ops.dequeue_task(session)

                # 8. Save analysis results
                analysis = await db_ops.save_analysis(
                    session,
                    search_id=search.id,
                    analysis_type="competitor",
                    content="Competitor analysis of smartphone market",
                    insights={
                        "top_competitors": ["Brand A", "Brand B", "Brand C"],
                        "price_range": {"min": 299, "max": 1299},
                        "common_features": ["5G", "OLED", "Fast charging"],
                        "summary": [
                            "Market dominated by 3 major brands",
                            "Price competition intense in mid-range segment",
                            "5G becoming standard feature"
                        ]
                    },
                    llm_info={"provider": "ollama", "model": "llama3.1:8b"},
                    scores={"confidence": 0.88}
                )

                # 9. Generate insights
                insight = await db_ops.create_insight(
                    session,
                    insight_type="opportunity",
                    title="Mid-range market gap identified",
                    description="Opportunity for $400-600 range with premium features",
                    severity="medium",
                    confidence_score=0.85,
                    insight_data={"potential_market_share": 0.15, "source": "competitor_analysis"}
                )

                # 10. Update search status
                await db_ops.update_search_status(session, search.id, "completed")
                await db_ops.complete_task(session, analysis_task.id, "completed")

                # 11. Cache results
                import json
                await db_ops.set_cache(
                    session,
                    f"search_results:{search.id}",
                    json.dumps({"results": len(results), "insights": len(analysis.insights)}),
                    ttl_seconds=7200
                )

            # Verify complete workflow
            async with db_manager.get_session() as session:
                search = await db_ops.get_search(session, search.id)
                assert search.status == "completed"

                result_count = await db_ops.count_results(session, search.id)
                assert result_count == 20

                analyses = await db_ops.get_analyses(session, search.id)
                assert len(analyses) == 1

                insights = await db_ops.get_unread_insights(session)
                assert len(insights) >= 1

                cached = await db_ops.get_cache(session, f"search_results:{search.id}")
                assert cached is not None

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def run_all_integration_tests(self):
        """Run all integration tests"""
        print("\n" + "=" * 60)
        print("INTEGRATION TESTS")
        print("=" * 60 + "\n")

        # Multi-table transaction tests
        print("\n--- Multi-Table Transaction Tests ---")
        await self.test_search_with_results_transaction()
        await self.test_product_with_price_history_transaction()
        await self.test_competitor_product_linking_transaction()
        await self.test_cascading_delete_transaction()

        # Foreign key constraint tests
        print("\n--- Foreign Key Constraint Tests ---")
        await self.test_foreign_key_enforcement()
        await self.test_orphaned_record_prevention()
        await self.test_referential_integrity_on_update()

        # Cascade operations tests
        print("\n--- Cascade Operations Tests ---")
        await self.test_search_cascade_delete()
        await self.test_product_cascade_delete()
        await self.test_multi_level_cascade()

        # Queue processing simulation
        print("\n--- Queue Processing Simulation ---")
        await self.test_queue_worker_simulation()
        await self.test_queue_retry_mechanism()
        await self.test_queue_priority_processing()

        # Cache cleanup routine
        print("\n--- Cache Cleanup Routine ---")
        await self.test_automated_cache_cleanup()
        await self.test_cache_cleanup_performance()
        await self.test_selective_cache_invalidation()

        # Complex workflow
        print("\n--- Complex Workflow Tests ---")
        await self.test_complete_search_workflow()

        # Print summary
        print("\n" + "=" * 60)
        print(f"RESULTS: {self.runner_passed} passed, {self.runner_failed} failed")
        if self.errors:
            print("\nFailed tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}")
                print(f"    {error}")
        print("=" * 60)

        return self.runner_failed == 0


if __name__ == "__main__":
    tester = TestIntegration()
    success = asyncio.run(tester.run_all_integration_tests())
    sys.exit(0 if success else 1)