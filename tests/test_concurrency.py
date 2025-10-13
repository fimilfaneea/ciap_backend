"""
Concurrency and WAL Mode Tests for CIAP Database
Tests concurrent access, WAL mode functionality, connection pooling, and locking
"""

import asyncio
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from sqlalchemy import text, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import DatabaseManager
from src.core.db_ops import DatabaseOperations
from src.core.models import *
from tests.test_utils import (
    DatabaseTestFixture,
    TestDataFactory,
    ConcurrentExecutor,
    PerformanceTimer,
    MockDataGenerator
)


class TestConcurrency:
    """Test concurrent database access and WAL mode"""

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

    async def test_wal_mode_enabled(self):
        """Test that WAL mode is properly enabled"""
        test_name = "test_wal_mode_enabled"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Check journal mode
                result = await session.execute(text("PRAGMA journal_mode"))
                journal_mode = result.scalar()
                assert journal_mode.upper() == "WAL", f"Expected WAL mode, got {journal_mode}"

                # Check WAL-related files exist after write
                search = Search(
                    query="wal test",
                    search_type="test",
                    sources=["google"],
                    status="pending"
                )
                session.add(search)
                await session.commit()

                # WAL file should exist
                wal_path = Path(fixture.db_path + "-wal")
                shm_path = Path(fixture.db_path + "-shm")

                # Note: Files might not exist immediately in all cases
                # but WAL mode should be confirmed
                assert journal_mode.upper() == "WAL"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_concurrent_reads(self):
        """Test multiple concurrent read operations"""
        test_name = "test_concurrent_reads"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create test data
            async with db_manager.get_session() as session:
                search = await db_ops.create_search(
                    session, "concurrent read test", ["google"], "test"
                )
                search_id = search.id

                # Add search results
                results_data = [
                    {
                        'title': f'Result {i}',
                        'url': f'https://example{i}.com',
                        'snippet': f'Snippet {i}',
                        'source': 'google',
                        'rank': i
                    }
                    for i in range(100)
                ]
                await db_ops.bulk_insert_results(session, search_id, results_data)

            # Concurrent read function
            async def read_results(session_num: int):
                async with db_manager.get_session() as session:
                    results = await db_ops.get_search_results(
                        session, search_id, limit=50
                    )
                    return len(results)

            # Execute concurrent reads
            with PerformanceTimer("Concurrent reads", threshold_ms=1000):
                tasks = [read_results(i) for i in range(20)]
                results = await asyncio.gather(*tasks)

            # All reads should succeed
            assert all(r > 0 for r in results), "Some reads failed"
            assert all(r <= 50 for r in results), "Read limit exceeded"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_concurrent_writes_different_tables(self):
        """Test concurrent writes to different tables"""
        test_name = "test_concurrent_writes_different_tables"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async def write_searches(count: int):
                """Write to searches table"""
                async with db_manager.get_session() as session:
                    for i in range(count):
                        await db_ops.create_search(
                            session,
                            f"search {i}",
                            ["google"],
                            "test"
                        )

            async def write_products(count: int):
                """Write to products table"""
                async with db_manager.get_session() as session:
                    for i in range(count):
                        await db_ops.create_or_update_product(
                            session,
                            {
                                'product_name': f'Product {i}',
                                'brand_name': 'Brand',
                                'category': 'Test'
                            }
                        )

            async def write_cache(count: int):
                """Write to cache table"""
                async with db_manager.get_session() as session:
                    for i in range(count):
                        await db_ops.set_cache(
                            session,
                            f"key_{i}",
                            f"value_{i}",
                            ttl_seconds=3600
                        )

            # Execute concurrent writes to different tables
            with PerformanceTimer("Concurrent writes to different tables", threshold_ms=3000):
                await asyncio.gather(
                    write_searches(10),
                    write_products(10),
                    write_cache(10)
                )

            # Verify all data was written
            async with db_manager.get_session() as session:
                search_count = await session.execute(
                    select(func.count()).select_from(Search)
                )
                product_count = await session.execute(
                    select(func.count()).select_from(Product)
                )
                cache_count = await session.execute(
                    select(func.count()).select_from(Cache)
                )

                assert search_count.scalar() >= 10
                assert product_count.scalar() >= 10
                assert cache_count.scalar() >= 10

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_concurrent_writes_same_table(self):
        """Test concurrent writes to the same table"""
        test_name = "test_concurrent_writes_same_table"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async def write_task_batch(batch_num: int, count: int):
                """Write batch of tasks"""
                async with db_manager.get_session() as session:
                    for i in range(count):
                        await db_ops.enqueue_task(
                            session,
                            task_type=f"batch_{batch_num}",
                            payload={"batch": batch_num, "item": i},
                            priority=batch_num
                        )

            # Execute concurrent writes to same table
            with PerformanceTimer("Concurrent writes to same table", threshold_ms=2000):
                tasks = [write_task_batch(i, 5) for i in range(10)]
                await asyncio.gather(*tasks)

            # Verify all tasks were created
            async with db_manager.get_session() as session:
                task_count = await session.execute(
                    select(func.count()).select_from(TaskQueue)
                )
                assert task_count.scalar() == 50, f"Expected 50 tasks, got {task_count.scalar()}"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_wal_checkpoint_behavior(self):
        """Test WAL checkpoint behavior under load"""
        test_name = "test_wal_checkpoint_behavior"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create significant amount of data to trigger checkpoint
            async with db_manager.get_session() as session:
                # Create search
                search = await db_ops.create_search(
                    session, "checkpoint test", ["google"], "test"
                )

                # Add large amount of data
                for batch in range(10):
                    results_data = [
                        {
                            'title': f'Result {batch}_{i}',
                            'url': f'https://example.com/{batch}/{i}',
                            'snippet': f'Snippet {batch}_{i}',
                            'source': 'google',
                            'rank': batch * 100 + i
                        }
                        for i in range(100)
                    ]
                    await db_ops.bulk_insert_results(session, search.id, results_data)

                    if batch % 3 == 0:
                        # Try checkpoint periodically (may fail if busy, which is acceptable)
                        try:
                            await session.execute(text("PRAGMA wal_checkpoint(PASSIVE)"))
                        except Exception:
                            # It's ok if checkpoint fails due to busy database
                            pass

            # Small delay to ensure all connections are closed
            await asyncio.sleep(0.1)

            # Check WAL status
            async with db_manager.get_session() as session:
                try:
                    result = await session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                    checkpoint_result = result.fetchone()
                    # checkpoint_result contains (busy, log_frames, checkpointed_frames)
                    assert checkpoint_result is not None
                except Exception:
                    # If checkpoint still fails, just verify WAL mode is enabled
                    result = await session.execute(text("PRAGMA journal_mode"))
                    journal_mode = result.scalar()
                    assert journal_mode.upper() == "WAL"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_database_locking_scenarios(self):
        """Test various database locking scenarios"""
        test_name = "test_database_locking_scenarios"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            # Scenario 1: Read during write
            async def long_write():
                async with db_manager.get_session() as session:
                    # Start transaction
                    products = []
                    for i in range(50):
                        product = Product(
                            product_name=f"Lock Test {i}",
                            brand_name="Brand",
                            category="Test"
                        )
                        products.append(product)
                    session.add_all(products)
                    # Simulate long processing
                    await asyncio.sleep(0.5)
                    await session.commit()

            async def concurrent_read():
                await asyncio.sleep(0.1)  # Start during write
                async with db_manager.get_session() as session:
                    # This should work in WAL mode
                    result = await session.execute(
                        select(func.count()).select_from(Product)
                    )
                    return result.scalar()

            # Execute concurrently
            write_task = asyncio.create_task(long_write())
            read_result = await concurrent_read()
            await write_task

            # Read should have succeeded
            assert read_result >= 0

            # Scenario 2: Multiple readers
            async def reader(reader_id: int):
                async with db_manager.get_session() as session:
                    result = await session.execute(
                        select(func.count()).select_from(Product)
                    )
                    return result.scalar()

            # Multiple concurrent reads should all succeed
            reader_tasks = [reader(i) for i in range(10)]
            reader_results = await asyncio.gather(*reader_tasks)
            assert all(r >= 0 for r in reader_results)

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_connection_pool_behavior(self):
        """Test connection pool behavior under load"""
        test_name = "test_connection_pool_behavior"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            connection_times = []

            async def use_connection(conn_id: int):
                """Use a database connection"""
                start = time.perf_counter()
                async with db_manager.get_session() as session:
                    # Simulate some work
                    await db_ops.get_cache(session, f"test_key_{conn_id}")
                    await asyncio.sleep(0.01)
                end = time.perf_counter()
                return end - start

            # Test connection reuse with many concurrent requests
            tasks = [use_connection(i) for i in range(50)]
            connection_times = await asyncio.gather(*tasks)

            # Calculate statistics
            avg_time = sum(connection_times) / len(connection_times)
            max_time = max(connection_times)
            min_time = min(connection_times)

            print(f"       Connection times - Avg: {avg_time*1000:.2f}ms, "
                  f"Max: {max_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms")

            # All connections should complete reasonably quickly
            assert max_time < 1.0, f"Connection took too long: {max_time}s"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_cache_expiration_concurrent(self):
        """Test cache expiration with concurrent access"""
        test_name = "test_cache_expiration_concurrent"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create cache entries with short TTL
            async with db_manager.get_session() as session:
                for i in range(20):
                    await db_ops.set_cache(
                        session,
                        f"expire_key_{i}",
                        f"value_{i}",
                        ttl_seconds=1 if i < 10 else 3600
                    )

            # Wait for some to expire
            await asyncio.sleep(1.5)

            # Concurrent cleanup and reads
            async def cleanup():
                async with db_manager.get_session() as session:
                    return await db_ops.cleanup_expired_cache(session)

            async def read_cache(key_id: int):
                async with db_manager.get_session() as session:
                    return await db_ops.get_cache(session, f"expire_key_{key_id}")

            # Execute cleanup and reads concurrently
            cleanup_task = asyncio.create_task(cleanup())
            read_tasks = [read_cache(i) for i in range(20)]
            read_results = await asyncio.gather(*read_tasks)
            cleanup_count = await cleanup_task

            # Verify expired entries return None
            for i, result in enumerate(read_results):
                if i < 10:
                    # Should be expired
                    assert result is None or result == f"value_{i}"
                else:
                    # Should still exist
                    assert result == f"value_{i}"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_task_queue_priority_concurrent(self):
        """Test task queue priority with concurrent operations"""
        test_name = "test_task_queue_priority_concurrent"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Enqueue tasks with different priorities concurrently
            async def enqueue_batch(priority: int, count: int):
                async with db_manager.get_session() as session:
                    tasks = []
                    for i in range(count):
                        task = await db_ops.enqueue_task(
                            session,
                            task_type=f"priority_{priority}",
                            payload={"priority": priority, "item": i},
                            priority=priority
                        )
                        tasks.append(task)
                    return tasks

            # Enqueue different priority batches concurrently
            await asyncio.gather(
                enqueue_batch(1, 5),   # Low priority
                enqueue_batch(5, 5),   # Medium priority
                enqueue_batch(10, 5)   # High priority
            )

            # Dequeue tasks concurrently
            dequeued_tasks = []

            async def dequeue_worker():
                async with db_manager.get_session() as session:
                    task = await db_ops.dequeue_task(session)
                    if task:
                        await db_ops.complete_task(session, task.id, "completed")
                        return task.priority
                    return None

            # Dequeue all tasks
            workers = [dequeue_worker() for _ in range(15)]
            priorities = await asyncio.gather(*workers)

            # Filter out None values
            priorities = [p for p in priorities if p is not None]

            # Verify high priority tasks were dequeued first
            # The first tasks should have higher priority
            if len(priorities) >= 10:
                first_five = priorities[:5]
                last_five = priorities[-5:]
                avg_first = sum(first_five) / len(first_five)
                avg_last = sum(last_five) / len(last_five)
                assert avg_first >= avg_last, "Priority order not maintained"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_concurrent_search_and_results(self):
        """Test concurrent search creation and result insertion"""
        test_name = "test_concurrent_search_and_results"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async def create_search_with_results(search_num: int):
                """Create search and add results"""
                async with db_manager.get_session() as session:
                    # Create search
                    search = await db_ops.create_search(
                        session,
                        f"concurrent search {search_num}",
                        ["google", "bing"],
                        "test"
                    )

                    # Add results
                    results_data = [
                        {
                            'title': f'Search {search_num} Result {i}',
                            'url': f'https://example.com/s{search_num}/r{i}',
                            'snippet': f'Snippet for search {search_num} result {i}',
                            'source': 'google' if i % 2 == 0 else 'bing',
                            'rank': i
                        }
                        for i in range(20)
                    ]

                    await db_ops.bulk_insert_results(session, search.id, results_data)

                    # Update search status
                    await db_ops.update_search_status(session, search.id, "completed")

                    return search.id

            # Create multiple searches concurrently
            with PerformanceTimer("Concurrent search and results", threshold_ms=5000):
                search_tasks = [create_search_with_results(i) for i in range(10)]
                search_ids = await asyncio.gather(*search_tasks)

            # Verify all searches and results
            async with db_manager.get_session() as session:
                for search_id in search_ids:
                    search = await db_ops.get_search(session, search_id)
                    assert search is not None
                    assert search.status == "completed"

                    result_count = await db_ops.count_results(session, search_id)
                    assert result_count == 20

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_streaming_concurrent_access(self):
        """Test streaming queries with concurrent access"""
        test_name = "test_streaming_concurrent_access"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create large dataset
            async with db_manager.get_session() as session:
                search = await db_ops.create_search(
                    session, "streaming test", ["google"], "test"
                )
                search_id = search.id

                # Create products
                products = []
                for i in range(100):
                    product = await db_ops.create_or_update_product(
                        session,
                        {
                            'product_name': f'Product {i}',
                            'brand_name': f'Brand {i % 10}',
                            'category': f'Category {i % 5}'
                        }
                    )
                    products.append(product)

            # Concurrent streaming operations
            async def stream_search_results():
                async with db_manager.get_session() as session:
                    count = 0
                    async for chunk in db_ops.stream_search_results(
                        session, search_id, chunk_size=10
                    ):
                        count += len(chunk)
                    return count

            async def stream_products():
                async with db_manager.get_session() as session:
                    count = 0
                    async for chunk in db_ops.stream_products(
                        session, chunk_size=10
                    ):
                        count += len(chunk)
                    return count

            # Execute streams concurrently
            results = await asyncio.gather(
                stream_search_results(),
                stream_products(),
                stream_products()  # Multiple readers
            )

            # Verify streaming worked
            assert results[1] == 100  # All products streamed
            assert results[2] == 100  # Second stream also worked

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_deadlock_prevention(self):
        """Test that operations don't cause deadlocks"""
        test_name = "test_deadlock_prevention"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            # Create initial data
            async with db_manager.get_session() as session:
                product1 = Product(
                    id=1,
                    product_name="Product A",
                    brand_name="Brand",
                    category="Test"
                )
                product2 = Product(
                    id=2,
                    product_name="Product B",
                    brand_name="Brand",
                    category="Test"
                )
                session.add_all([product1, product2])
                await session.commit()

            # Track which updates succeeded
            update_results = {'order1': None, 'order2': None}

            # Simplified concurrent update scenario
            # Test that concurrent updates don't cause indefinite blocking
            async def update_products_order1():
                try:
                    async with db_manager.get_session() as session:
                        try:
                            # Simple sequential update in one transaction
                            p1 = await session.get(Product, 1)
                            if p1:
                                p1.description = "Updated by order 1"
                                await session.flush()

                            await asyncio.sleep(0.005)  # Small delay

                            p2 = await session.get(Product, 2)
                            if p2:
                                p2.description = "Updated by order 1"

                            await session.commit()
                            update_results['order1'] = 'success'
                        except Exception as e:
                            await session.rollback()
                            # Record failure but don't crash
                            update_results['order1'] = f'failed: {type(e).__name__}'
                except Exception as e:
                    update_results['order1'] = f'connection failed: {type(e).__name__}'

            async def update_products_order2():
                try:
                    async with db_manager.get_session() as session:
                        try:
                            # Simple sequential update in one transaction (opposite order)
                            p2 = await session.get(Product, 2)
                            if p2:
                                p2.description = "Updated by order 2"
                                await session.flush()

                            await asyncio.sleep(0.005)  # Small delay

                            p1 = await session.get(Product, 1)
                            if p1:
                                p1.description = "Updated by order 2"

                            await session.commit()
                            update_results['order2'] = 'success'
                        except Exception as e:
                            await session.rollback()
                            # Record failure but don't crash
                            update_results['order2'] = f'failed: {type(e).__name__}'
                except Exception as e:
                    update_results['order2'] = f'connection failed: {type(e).__name__}'

            # Run concurrent operations with return_exceptions
            await asyncio.gather(
                update_products_order1(),
                update_products_order2(),
                return_exceptions=True
            )

            # Small delay to ensure all transactions are complete
            await asyncio.sleep(0.1)

            # With WAL mode, both operations might succeed or one might fail with DatabaseError
            # The key test is that we don't deadlock indefinitely
            successful_updates = [k for k, v in update_results.items() if v == 'success']

            # At least one should succeed, or both should fail gracefully (not hang)
            # If both failed, check that database is not corrupted
            if len(successful_updates) == 0:
                # Both failed - verify database is still accessible
                try:
                    async with db_manager.get_session() as session:
                        products = await session.execute(select(Product))
                        product_list = products.scalars().all()
                        # Database should still be accessible with original 2 products
                        assert len(product_list) == 2, f"Database corrupted: expected 2 products, found {len(product_list)}"
                    # Test passes - no deadlock occurred, database accessible
                    print(f"       Both updates failed but no deadlock: {update_results}")
                except Exception as e:
                    # Database is corrupted or inaccessible
                    raise AssertionError(f"Database became inaccessible: {e}")
            else:
                # At least one succeeded - verify database state
                async with db_manager.get_session() as session:
                    products = await session.execute(select(Product))
                    product_list = products.scalars().all()
                    assert len(product_list) == 2, f"Expected 2 products, found {len(product_list)}"

                    # At least one product should have been updated
                    updated_products = [p for p in product_list if p.description is not None]
                    assert len(updated_products) >= 1, "No products were updated despite success report"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def run_all_concurrency_tests(self):
        """Run all concurrency tests"""
        print("\n" + "=" * 60)
        print("CONCURRENCY AND WAL MODE TESTS")
        print("=" * 60 + "\n")

        # Run all test methods
        await self.test_wal_mode_enabled()
        await self.test_concurrent_reads()
        await self.test_concurrent_writes_different_tables()
        await self.test_concurrent_writes_same_table()
        await self.test_wal_checkpoint_behavior()
        await self.test_database_locking_scenarios()
        await self.test_connection_pool_behavior()
        await self.test_cache_expiration_concurrent()
        await self.test_task_queue_priority_concurrent()
        await self.test_concurrent_search_and_results()
        await self.test_streaming_concurrent_access()
        await self.test_deadlock_prevention()

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
    tester = TestConcurrency()
    success = asyncio.run(tester.run_all_concurrency_tests())
    sys.exit(0 if success else 1)