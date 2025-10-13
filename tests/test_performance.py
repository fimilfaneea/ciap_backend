"""
Performance Benchmark Tests for CIAP Database
Tests query performance, bulk operations, indexing effectiveness, and throughput
"""

import asyncio
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from sqlalchemy import text, select, func

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import DatabaseManager
from src.core.db_ops import DatabaseOperations
from src.core.models import *
from tests.test_utils import (
    DatabaseTestFixture,
    TestDataFactory,
    PerformanceTimer,
    MockDataGenerator,
    ConcurrentExecutor,
    TEST_DATA_SIZE_SMALL,
    TEST_DATA_SIZE_MEDIUM,
    TEST_DATA_SIZE_LARGE
)


class PerformanceBenchmarks:
    """Performance benchmarks for database operations"""

    def __init__(self):
        self.runner_passed = 0
        self.runner_failed = 0
        self.errors = []
        self.metrics = {}

    def report(self, test_name, passed, error=None, metrics=None):
        if passed:
            self.runner_passed += 1
            print(f"[PASS] {test_name}")
        else:
            self.runner_failed += 1
            print(f"[FAIL] {test_name}")
            if error:
                self.errors.append((test_name, error))
                print(f"       Error: {error}")

        if metrics:
            self.metrics[test_name] = metrics
            print(f"       Metrics: {metrics}")

    async def test_bulk_insert_performance(self):
        """Benchmark bulk insert operations"""
        test_name = "test_bulk_insert_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            metrics = {}

            async with db_manager.get_session() as session:
                # Create search
                search = await db_ops.create_search(
                    session, "bulk insert test", ["google"], "test"
                )
                search_id = search.id

                # Test different batch sizes
                batch_sizes = [10, 50, 100, 500, 1000]

                for batch_size in batch_sizes:
                    results_data = [
                        {
                            'title': f'Result {i}',
                            'url': f'https://example.com/r{i}',
                            'snippet': f'Snippet text for result {i}',
                            'source': 'google',
                            'rank': i
                        }
                        for i in range(batch_size)
                    ]

                    start_time = time.perf_counter()
                    await db_ops.bulk_insert_results(session, search_id, results_data)
                    elapsed = (time.perf_counter() - start_time) * 1000

                    rate = batch_size / (elapsed / 1000)  # Records per second
                    metrics[f"batch_{batch_size}"] = {
                        "time_ms": round(elapsed, 2),
                        "rate_per_sec": round(rate, 2)
                    }

                    # Performance threshold: Should insert at least 100 records/second
                    assert rate > 100, f"Insert rate too slow: {rate:.2f} records/sec"

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_query_performance_with_indexes(self):
        """Benchmark query performance with indexes"""
        test_name = "test_query_performance_with_indexes"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create test data
            async with db_manager.get_session() as session:
                # Create multiple searches
                search_ids = []
                for i in range(10):
                    search = await db_ops.create_search(
                        session,
                        f"query test {i}",
                        ["google", "bing"],
                        "competitor" if i % 2 == 0 else "market"
                    )
                    search_ids.append(search.id)

                    # Add results for each search
                    results_data = [
                        {
                            'title': f'Search {i} Result {j}',
                            'url': f'https://example.com/s{i}/r{j}',
                            'snippet': f'Snippet for search {i} result {j}',
                            'source': 'google' if j % 2 == 0 else 'bing',
                            'rank': j
                        }
                        for j in range(100)
                    ]
                    await db_ops.bulk_insert_results(session, search.id, results_data)

            metrics = {}

            # Test indexed queries
            async with db_manager.get_session() as session:
                # Query by search_id (indexed)
                start = time.perf_counter()
                results = await db_ops.get_search_results(
                    session, search_ids[0], limit=50
                )
                indexed_time = (time.perf_counter() - start) * 1000
                metrics["indexed_query_ms"] = round(indexed_time, 2)

                # Query by status (indexed)
                start = time.perf_counter()
                result = await session.execute(
                    select(Search).where(Search.status == "pending").limit(5)
                )
                searches = result.scalars().all()
                status_query_time = (time.perf_counter() - start) * 1000
                metrics["status_query_ms"] = round(status_query_time, 2)

                # Count query
                start = time.perf_counter()
                count = await db_ops.count_results(session, search_ids[0])
                count_time = (time.perf_counter() - start) * 1000
                metrics["count_query_ms"] = round(count_time, 2)

            # Performance thresholds
            assert indexed_time < 50, f"Indexed query too slow: {indexed_time:.2f}ms"
            assert status_query_time < 50, f"Status query too slow: {status_query_time:.2f}ms"
            assert count_time < 20, f"Count query too slow: {count_time:.2f}ms"

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_fts5_search_performance(self):
        """Benchmark FTS5 full-text search performance"""
        test_name = "test_fts5_search_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create products with searchable text
            async with db_manager.get_session() as session:
                products = []
                for i in range(1000):
                    product = await db_ops.create_or_update_product(
                        session,
                        {
                            'product_name': f'Product {i} {TestDataFactory.random_string(10)}',
                            'brand_name': f'Brand {i % 50}',
                            'category': f'Category {i % 10}',
                            'description': f'This is a detailed description for product {i}. '
                                         f'It contains various keywords like quality, premium, '
                                         f'affordable, innovative, and reliable. '
                                         f'{TestDataFactory.random_string(50)}'
                        }
                    )
                    products.append(product)

                    if i % 100 == 0:
                        await session.commit()

                await session.commit()

            metrics = {}

            # Test FTS5 search performance
            async with db_manager.get_session() as session:
                search_terms = ["quality", "premium", "innovative", "product"]

                for term in search_terms:
                    start = time.perf_counter()

                    # Simulate FTS5 search
                    result = await session.execute(
                        select(Product).where(
                            Product.description.contains(term)
                        ).limit(20)
                    )
                    products = result.scalars().all()

                    elapsed = (time.perf_counter() - start) * 1000
                    metrics[f"fts_{term}_ms"] = round(elapsed, 2)

                    # FTS5 searches should be fast
                    assert elapsed < 100, f"FTS search for '{term}' too slow: {elapsed:.2f}ms"

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_pagination_performance(self):
        """Benchmark pagination performance"""
        test_name = "test_pagination_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create large dataset
            async with db_manager.get_session() as session:
                for i in range(100):
                    await db_ops.create_search(
                        session,
                        f"pagination test {i}",
                        ["google"],
                        "test"
                    )
                    if i % 20 == 0:
                        await session.commit()
                await session.commit()

            metrics = {}

            # Test pagination at different pages
            async with db_manager.get_session() as session:
                page_tests = [1, 5, 10, 20]

                for page in page_tests:
                    start = time.perf_counter()
                    result = await db_ops.get_paginated_searches(
                        session, page=page, per_page=10
                    )
                    elapsed = (time.perf_counter() - start) * 1000

                    metrics[f"page_{page}_ms"] = round(elapsed, 2)

                    # Pagination should be consistent
                    assert elapsed < 50, f"Page {page} query too slow: {elapsed:.2f}ms"
                    assert len(result.items) <= 10
                    assert result.page == page

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_concurrent_operations_throughput(self):
        """Benchmark throughput with concurrent operations"""
        test_name = "test_concurrent_operations_throughput"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            metrics = {}

            # Mixed workload function
            async def mixed_workload(worker_id: int):
                operations = []

                async with db_manager.get_session() as session:
                    # Read operation
                    start = time.perf_counter()
                    searches = await db_ops.get_recent_searches(session, limit=5)
                    operations.append(("read", time.perf_counter() - start))

                    # Write operation
                    start = time.perf_counter()
                    search = await db_ops.create_search(
                        session,
                        f"worker {worker_id} search",
                        ["google"],
                        "test"
                    )
                    operations.append(("write", time.perf_counter() - start))

                    # Cache operation
                    start = time.perf_counter()
                    await db_ops.set_cache(
                        session,
                        f"worker_{worker_id}_key",
                        f"value_{worker_id}",
                        ttl_seconds=3600
                    )
                    operations.append(("cache", time.perf_counter() - start))

                return operations

            # Run concurrent workload
            start_time = time.perf_counter()
            workers = 20
            tasks = [mixed_workload(i) for i in range(workers)]
            all_operations = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time

            # Calculate throughput
            total_ops = sum(len(ops) for ops in all_operations)
            throughput = total_ops / total_time
            metrics["throughput_ops_per_sec"] = round(throughput, 2)

            # Calculate average latencies
            all_times = []
            for ops in all_operations:
                all_times.extend([t for _, t in ops])

            metrics["avg_latency_ms"] = round(statistics.mean(all_times) * 1000, 2)
            metrics["p95_latency_ms"] = round(
                sorted(all_times)[int(len(all_times) * 0.95)] * 1000, 2
            )
            metrics["total_operations"] = total_ops
            metrics["duration_seconds"] = round(total_time, 2)

            # Performance threshold: At least 50 ops/second
            assert throughput > 50, f"Throughput too low: {throughput:.2f} ops/sec"

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_cache_performance(self):
        """Benchmark cache operations performance"""
        test_name = "test_cache_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            metrics = {}

            async with db_manager.get_session() as session:
                # Test cache write performance
                cache_keys = []
                start = time.perf_counter()
                for i in range(100):
                    key = f"perf_test_key_{i}"
                    cache_keys.append(key)
                    await db_ops.set_cache(
                        session,
                        key,
                        {"data": f"value_{i}", "timestamp": datetime.utcnow().isoformat()},
                        ttl_seconds=3600
                    )
                write_time = time.perf_counter() - start
                metrics["cache_writes_100"] = {
                    "time_ms": round(write_time * 1000, 2),
                    "rate_per_sec": round(100 / write_time, 2)
                }

                # Test cache read performance
                start = time.perf_counter()
                for key in cache_keys:
                    value = await db_ops.get_cache(session, key)
                    assert value is not None
                read_time = time.perf_counter() - start
                metrics["cache_reads_100"] = {
                    "time_ms": round(read_time * 1000, 2),
                    "rate_per_sec": round(100 / read_time, 2)
                }

                # Test cache cleanup performance
                start = time.perf_counter()
                deleted = await db_ops.cleanup_expired_cache(session)
                cleanup_time = time.perf_counter() - start
                metrics["cache_cleanup_ms"] = round(cleanup_time * 1000, 2)

            # Performance thresholds
            assert metrics["cache_writes_100"]["rate_per_sec"] > 100
            assert metrics["cache_reads_100"]["rate_per_sec"] > 200
            assert metrics["cache_cleanup_ms"] < 100

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_task_queue_performance(self):
        """Benchmark task queue operations"""
        test_name = "test_task_queue_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            metrics = {}

            # Test enqueue performance
            async with db_manager.get_session() as session:
                start = time.perf_counter()
                task_ids = []
                for i in range(100):
                    task = await db_ops.enqueue_task(
                        session,
                        task_type="performance_test",
                        payload={"id": i, "data": f"test_{i}"},
                        priority=i % 10
                    )
                    task_ids.append(task.id)
                enqueue_time = time.perf_counter() - start
                metrics["enqueue_100_tasks"] = {
                    "time_ms": round(enqueue_time * 1000, 2),
                    "rate_per_sec": round(100 / enqueue_time, 2)
                }

            # Test dequeue performance
            async with db_manager.get_session() as session:
                dequeued = []
                start = time.perf_counter()
                for _ in range(50):
                    task = await db_ops.dequeue_task(session)
                    if task:
                        dequeued.append(task)
                        await db_ops.complete_task(session, task.id, "completed")
                dequeue_time = time.perf_counter() - start
                metrics["dequeue_50_tasks"] = {
                    "time_ms": round(dequeue_time * 1000, 2),
                    "rate_per_sec": round(50 / dequeue_time, 2)
                }

            # Performance thresholds
            assert metrics["enqueue_100_tasks"]["rate_per_sec"] > 100
            assert metrics["dequeue_50_tasks"]["rate_per_sec"] > 50

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_streaming_performance(self):
        """Benchmark streaming query performance"""
        test_name = "test_streaming_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create large dataset
            async with db_manager.get_session() as session:
                search = await db_ops.create_search(
                    session, "streaming test", ["google"], "test"
                )

                # Add many results
                for batch in range(10):
                    results_data = [
                        {
                            'title': f'Result {batch * 100 + i}',
                            'url': f'https://example.com/r{batch * 100 + i}',
                            'snippet': f'Snippet {batch * 100 + i}',
                            'source': 'google',
                            'rank': batch * 100 + i
                        }
                        for i in range(100)
                    ]
                    await db_ops.bulk_insert_results(session, search.id, results_data)

            metrics = {}

            # Test streaming performance
            async with db_manager.get_session() as session:
                start = time.perf_counter()
                total_streamed = 0
                chunk_count = 0

                async for chunk in db_ops.stream_search_results(
                    session, search.id, chunk_size=50
                ):
                    total_streamed += len(chunk)
                    chunk_count += 1

                stream_time = time.perf_counter() - start
                metrics["stream_1000_results"] = {
                    "time_ms": round(stream_time * 1000, 2),
                    "rate_per_sec": round(total_streamed / stream_time, 2),
                    "chunks": chunk_count,
                    "total_records": total_streamed
                }

            # Performance threshold: Stream at least 500 records/second
            assert metrics["stream_1000_results"]["rate_per_sec"] > 500

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_batch_update_performance(self):
        """Benchmark batch update operations"""
        test_name = "test_batch_update_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create test data
            async with db_manager.get_session() as session:
                search_ids = []
                for i in range(20):
                    search = await db_ops.create_search(
                        session,
                        f"batch update test {i}",
                        ["google"],
                        "test"
                    )
                    search_ids.append(search.id)

                task_ids = []
                for i in range(50):
                    task = await db_ops.enqueue_task(
                        session,
                        task_type="batch_test",
                        payload={"id": i},
                        priority=5
                    )
                    task_ids.append(task.id)

            metrics = {}

            # Test batch updates
            async with db_manager.get_session() as session:
                # Batch update search status
                start = time.perf_counter()
                await db_ops.batch_update_search_status(
                    session, search_ids, "processing"
                )
                search_update_time = time.perf_counter() - start
                metrics["batch_update_20_searches_ms"] = round(search_update_time * 1000, 2)

                # Batch update task status
                start = time.perf_counter()
                await db_ops.batch_update_task_status(
                    session, task_ids[:25], "processing"
                )
                task_update_time = time.perf_counter() - start
                metrics["batch_update_25_tasks_ms"] = round(task_update_time * 1000, 2)

            # Performance thresholds
            assert search_update_time < 0.5  # Under 500ms
            assert task_update_time < 0.5    # Under 500ms

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_complex_query_performance(self):
        """Benchmark complex query with joins and aggregations"""
        test_name = "test_complex_query_performance"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            # Create related data
            async with db_manager.get_session() as session:
                # Create competitors and products
                competitors = []
                for i in range(5):
                    comp = await db_ops.create_or_update_competitor(
                        session,
                        TestDataFactory.create_competitor_data(1)[0]
                    )
                    competitors.append(comp)

                products = []
                for i in range(20):
                    prod = await db_ops.create_or_update_product(
                        session,
                        TestDataFactory.create_product_data(1)[0]
                    )
                    products.append(prod)

                    # Add price data
                    for j in range(5):
                        await db_ops.add_price_data(
                            session,
                            product_id=prod.id,
                            price=100.0 + j * 10,
                            currency="USD",
                            source="test"
                        )

                # Link competitors to products
                for comp in competitors:
                    for prod in products[:10]:
                        await db_ops.link_competitor_product(
                            session, comp.id, prod.id
                        )

            metrics = {}

            # Test complex queries
            async with db_manager.get_session() as session:
                # Query with joins
                start = time.perf_counter()
                result = await session.execute(
                    select(Product)
                    .join(CompetitorProducts)
                    .join(Competitor)
                    .where(Competitor.market_share > 0.1)
                    .distinct()
                    .limit(10)
                )
                products = result.scalars().all()
                join_time = time.perf_counter() - start
                metrics["join_query_ms"] = round(join_time * 1000, 2)

                # Aggregation query
                start = time.perf_counter()
                result = await session.execute(
                    select(
                        PriceData.product_id,
                        func.avg(PriceData.price).label("avg_price"),
                        func.count(PriceData.id).label("price_count")
                    )
                    .group_by(PriceData.product_id)
                    .having(func.count(PriceData.id) > 2)
                )
                aggregations = result.all()
                agg_time = time.perf_counter() - start
                metrics["aggregation_query_ms"] = round(agg_time * 1000, 2)

            # Performance thresholds
            assert join_time < 0.1  # Under 100ms
            assert agg_time < 0.1   # Under 100ms

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_database_size_impact(self):
        """Test performance impact with growing database size"""
        test_name = "test_database_size_impact"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            metrics = {}

            async with db_manager.get_session() as session:
                # Baseline performance with small dataset
                search = await db_ops.create_search(
                    session, "size test", ["google"], "test"
                )

                # Test at different database sizes
                sizes = [100, 500, 1000, 2000]

                for size in sizes:
                    # Add data
                    results_data = [
                        {
                            'title': f'Size test result {i}',
                            'url': f'https://example.com/size/{i}',
                            'snippet': f'Snippet for size test {i}',
                            'source': 'google',
                            'rank': i
                        }
                        for i in range(size, size + 100)
                    ]
                    await db_ops.bulk_insert_results(session, search.id, results_data)

                    # Measure query time at this size
                    start = time.perf_counter()
                    results = await db_ops.get_search_results(
                        session, search.id, limit=20
                    )
                    query_time = (time.perf_counter() - start) * 1000

                    # Get database stats
                    stats = await db_ops.get_database_stats(session)

                    metrics[f"size_{size + 100}"] = {
                        "query_ms": round(query_time, 2),
                        "total_records": stats.get("total_search_results", 0),
                        "db_size_mb": round(stats.get("database_size_mb", 0), 2)
                    }

                    # Performance should not degrade significantly
                    assert query_time < 100, f"Query too slow at size {size}: {query_time:.2f}ms"

            await fixture.teardown()
            self.report(test_name, True, metrics=metrics)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def run_all_performance_tests(self):
        """Run all performance benchmark tests"""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK TESTS")
        print("=" * 60 + "\n")

        # Run all test methods
        await self.test_bulk_insert_performance()
        await self.test_query_performance_with_indexes()
        await self.test_fts5_search_performance()
        await self.test_pagination_performance()
        await self.test_concurrent_operations_throughput()
        await self.test_cache_performance()
        await self.test_task_queue_performance()
        await self.test_streaming_performance()
        await self.test_batch_update_performance()
        await self.test_complex_query_performance()
        await self.test_database_size_impact()

        # Print performance summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        for test_name, metrics in self.metrics.items():
            print(f"\n{test_name}:")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {metrics}")

        # Print final results
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
    benchmarks = PerformanceBenchmarks()
    success = asyncio.run(benchmarks.run_all_performance_tests())
    sys.exit(0 if success else 1)