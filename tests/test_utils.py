"""
Test Utilities for CIAP Database Testing
Provides helper functions, fixtures, and utilities for comprehensive testing
"""

import asyncio
import tempfile
import time
import random
import string
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, TypeVar, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import DatabaseManager
from src.core.db_ops import DatabaseOperations
from src.core.models import *
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

T = TypeVar('T')

# Test configuration
TEST_CHUNK_SIZE = 50
TEST_CONCURRENCY_LEVEL = 10
PERFORMANCE_THRESHOLD_MS = 100
TEST_DATA_SIZE_SMALL = 10
TEST_DATA_SIZE_MEDIUM = 100
TEST_DATA_SIZE_LARGE = 1000


class TestDataFactory:
    """Factory for generating test data"""

    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate random string"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def create_search_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate search test data"""
        return [
            {
                'query': f"test query {i} {TestDataFactory.random_string(5)}",
                'search_type': random.choice(['competitor', 'market', 'product']),
                'sources': random.choice([['google'], ['bing'], ['google', 'bing']]),
                'status': 'pending',
                'user_id': f"user_{i}"
            }
            for i in range(count)
        ]

    @staticmethod
    def create_product_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate product test data"""
        return [
            {
                'product_name': f"Product {i} {TestDataFactory.random_string(5)}",
                'brand_name': f"Brand {i}",
                'company_name': f"Company {i}",
                'category': random.choice(['Electronics', 'Clothing', 'Food', 'Books']),
                'description': f"Description for product {i}",
                'product_url': f"https://example.com/product/{i}",
                'industry': random.choice(['Tech', 'Fashion', 'FMCG', 'Media'])
            }
            for i in range(count)
        ]

    @staticmethod
    def create_competitor_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate competitor test data"""
        return [
            {
                'company_name': f"Competitor {i}",
                'website': f"https://competitor{i}.com",
                'description': f"Competitor {i} description",
                'market_share': round(random.uniform(0.01, 0.5), 3),
                'employee_count': random.randint(10, 10000),
                'revenue': random.randint(100000, 10000000),
                'founded_year': random.randint(1990, 2023),
                'headquarters': f"City {i}, Country",
                'strengths': ['strength1', 'strength2'],
                'weaknesses': ['weakness1', 'weakness2']
            }
            for i in range(count)
        ]

    @staticmethod
    def create_review_data(product_id: int, count: int = 1) -> List[Dict[str, Any]]:
        """Generate review test data"""
        return [
            {
                'product_id': product_id,
                'reviewer_name': f"Reviewer {i}",
                'rating': random.randint(1, 5),
                'review_text': f"Review text {i} - {TestDataFactory.random_string(50)}",
                'review_date': datetime.utcnow() - timedelta(days=random.randint(0, 365)),
                'verified_purchase': random.choice([True, False]),
                'helpful_votes': random.randint(0, 100),
                'review_source': random.choice(['amazon', 'ebay', 'website'])
            }
            for i in range(count)
        ]

    @staticmethod
    def create_price_data(product_id: int, count: int = 1) -> List[Dict[str, Any]]:
        """Generate price data"""
        base_price = random.uniform(10.0, 1000.0)
        return [
            {
                'product_id': product_id,
                'price': round(base_price + random.uniform(-50, 50), 2),
                'currency': 'USD',
                'source': random.choice(['amazon', 'ebay', 'walmart', 'direct']),
                'recorded_at': datetime.utcnow() - timedelta(days=count-i),
                'availability': random.choice(['in_stock', 'out_of_stock', 'limited']),
                'discount_percentage': random.choice([0, 5, 10, 15, 20, 25, 30])
            }
            for i in range(count)
        ]

    @staticmethod
    def create_task_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate task queue data"""
        return [
            {
                'task_type': random.choice(['scrape', 'analyze', 'process', 'export']),
                'payload': {
                    'url': f"https://example.com/page{i}",
                    'params': {'id': i, 'action': 'test'}
                },
                'priority': random.randint(1, 10),
                'scheduled_at': datetime.utcnow() + timedelta(minutes=random.randint(0, 60)),
                'retry_count': 0,
                'max_retries': 3
            }
            for i in range(count)
        ]

    @staticmethod
    def create_cache_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate cache data"""
        return [
            {
                'key': f"cache_key_{i}_{TestDataFactory.random_string(5)}",
                'value': {
                    'data': f"cached_data_{i}",
                    'timestamp': datetime.utcnow().isoformat(),
                    'metadata': {'type': 'test', 'id': i}
                },
                'ttl': random.choice([60, 300, 600, 1800, 3600])
            }
            for i in range(count)
        ]


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, name: str = "Operation", threshold_ms: float = PERFORMANCE_THRESHOLD_MS):
        self.name = name
        self.threshold_ms = threshold_ms
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

        if self.elapsed_ms > self.threshold_ms:
            print(f"[PERF WARNING] {self.name} took {self.elapsed_ms:.2f}ms (threshold: {self.threshold_ms}ms)")
        else:
            print(f"[PERF OK] {self.name} completed in {self.elapsed_ms:.2f}ms")

        return False

    def assert_under_threshold(self):
        """Assert that operation completed under threshold"""
        if self.elapsed_ms and self.elapsed_ms > self.threshold_ms:
            raise AssertionError(
                f"{self.name} exceeded threshold: {self.elapsed_ms:.2f}ms > {self.threshold_ms}ms"
            )


class ConcurrentExecutor:
    """Helper for running concurrent tests"""

    @staticmethod
    async def run_concurrent_async(
        func: Callable[..., Coroutine[Any, Any, T]],
        args_list: List[tuple],
        max_concurrent: int = TEST_CONCURRENCY_LEVEL
    ) -> List[T]:
        """Run async functions concurrently with limit"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(func, args):
            async with semaphore:
                return await func(*args)

        tasks = [run_with_semaphore(func, args) for args in args_list]
        return await asyncio.gather(*tasks)

    @staticmethod
    def run_concurrent_threads(
        func: Callable[..., T],
        args_list: List[tuple],
        max_workers: int = TEST_CONCURRENCY_LEVEL
    ) -> List[T]:
        """Run functions in thread pool"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args) for args in args_list]
            return [future.result() for future in futures]

    @staticmethod
    def run_concurrent_processes(
        func: Callable[..., T],
        args_list: List[tuple],
        max_workers: int = 4
    ) -> List[T]:
        """Run functions in process pool"""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args) for args in args_list]
            return [future.result() for future in futures]


class DatabaseTestFixture:
    """Database fixture for testing"""

    def __init__(self):
        self.temp_dir = None
        self.db_path = None
        self.db_manager = None
        self.db_ops = DatabaseOperations()

    async def setup(self, seed_data: bool = False) -> DatabaseManager:
        """Setup test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"{self.temp_dir}/test.db"
        self.db_manager = DatabaseManager(f"sqlite+aiosqlite:///{self.db_path}")

        await self.db_manager.initialize()

        if seed_data:
            await self._seed_test_data()

        return self.db_manager

    async def _seed_test_data(self):
        """Seed database with test data"""
        async with self.db_manager.get_session() as session:
            # Add searches
            searches = TestDataFactory.create_search_data(5)
            for search_data in searches:
                await self.db_ops.create_search(session, **search_data)

            # Add products
            products = TestDataFactory.create_product_data(10)
            for product_data in products:
                await self.db_ops.create_or_update_product(session, product_data)

            # Add tasks
            tasks = TestDataFactory.create_task_data(10)
            for task_data in tasks:
                await self.db_ops.enqueue_task(session, **task_data)

            await session.commit()

    async def teardown(self):
        """Cleanup test database"""
        if self.db_manager:
            await self.db_manager.close()
            # Additional delay for Windows to release file handles
            await asyncio.sleep(0.2)

        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            import platform

            # Windows-specific cleanup with retry logic
            if platform.system() == 'Windows':
                max_retries = 3
                retry_delay = 0.5

                for attempt in range(max_retries):
                    try:
                        # Try to remove the directory
                        shutil.rmtree(self.temp_dir, ignore_errors=False)
                        break  # Success, exit the loop
                    except PermissionError as e:
                        if attempt < max_retries - 1:
                            # Wait a bit for file handles to be released
                            await asyncio.sleep(retry_delay)
                            # Try to forcefully close any remaining file handles
                            try:
                                # Attempt to delete individual files first
                                for file_path in Path(self.temp_dir).glob('*'):
                                    try:
                                        file_path.unlink()
                                    except:
                                        pass
                            except:
                                pass
                        else:
                            # Last attempt failed, use ignore_errors
                            try:
                                shutil.rmtree(self.temp_dir, ignore_errors=True)
                            except:
                                # If all else fails, just print a warning
                                print(f"Warning: Could not clean up temp directory: {self.temp_dir}")
            else:
                # Non-Windows systems
                shutil.rmtree(self.temp_dir)

    async def reset(self):
        """Reset database to clean state"""
        async with self.db_manager.get_session() as session:
            # Clear all tables
            tables = [
                'search_results', 'analyses', 'serp_data',
                'searches', 'price_data', 'offers', 'product_reviews',
                'products', 'competitor_tracking', 'competitor_products',
                'competitors', 'market_trends', 'social_sentiments',
                'news_contents', 'feature_comparisons', 'insights',
                'cache', 'task_queue', 'scraping_jobs', 'rate_limits',
                'price_history'
            ]

            for table in tables:
                try:
                    await session.execute(text(f"DELETE FROM {table}"))
                except:
                    pass  # Table might not exist

            await session.commit()


class AsyncTestRunner:
    """Helper for running async tests"""

    @staticmethod
    def run(test_func: Callable[..., Coroutine], *args, **kwargs):
        """Run an async test function"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(test_func(*args, **kwargs))
        finally:
            loop.close()

    @staticmethod
    async def run_with_timeout(
        test_func: Callable[..., Coroutine],
        timeout: float,
        *args,
        **kwargs
    ):
        """Run async test with timeout"""
        try:
            return await asyncio.wait_for(
                test_func(*args, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise AssertionError(f"Test timed out after {timeout} seconds")


class TestAssertions:
    """Custom assertions for database testing"""

    @staticmethod
    async def assert_record_exists(
        session: AsyncSession,
        model: type,
        **filters
    ) -> Any:
        """Assert that a record exists with given filters"""
        from sqlalchemy import select

        query = select(model)
        for key, value in filters.items():
            query = query.where(getattr(model, key) == value)

        result = await session.execute(query)
        record = result.scalar_one_or_none()

        if not record:
            raise AssertionError(
                f"No {model.__name__} found with filters: {filters}"
            )

        return record

    @staticmethod
    async def assert_record_not_exists(
        session: AsyncSession,
        model: type,
        **filters
    ):
        """Assert that no record exists with given filters"""
        from sqlalchemy import select

        query = select(model)
        for key, value in filters.items():
            query = query.where(getattr(model, key) == value)

        result = await session.execute(query)
        record = result.scalar_one_or_none()

        if record:
            raise AssertionError(
                f"Unexpected {model.__name__} found with filters: {filters}"
            )

    @staticmethod
    async def assert_count(
        session: AsyncSession,
        model: type,
        expected_count: int,
        **filters
    ):
        """Assert record count matches expected"""
        from sqlalchemy import select, func

        query = select(func.count()).select_from(model)
        for key, value in filters.items():
            query = query.where(getattr(model, key) == value)

        result = await session.execute(query)
        actual_count = result.scalar()

        if actual_count != expected_count:
            raise AssertionError(
                f"Expected {expected_count} {model.__name__} records, "
                f"found {actual_count} with filters: {filters}"
            )

    @staticmethod
    def assert_performance(elapsed_ms: float, threshold_ms: float, operation: str = "Operation"):
        """Assert performance is within threshold"""
        if elapsed_ms > threshold_ms:
            raise AssertionError(
                f"{operation} too slow: {elapsed_ms:.2f}ms > {threshold_ms}ms threshold"
            )


class MockDataGenerator:
    """Generate large amounts of mock data for stress testing"""

    @staticmethod
    async def generate_bulk_search_results(
        session: AsyncSession,
        search_id: int,
        count: int = 1000
    ) -> List[SearchResult]:
        """Generate bulk search results"""
        results_data = [
            {
                'search_id': search_id,
                'source': random.choice(['google', 'bing', 'duckduckgo']),
                'title': f"Result {i} - {TestDataFactory.random_string(20)}",
                'snippet': f"Snippet for result {i} - {TestDataFactory.random_string(100)}",
                'url': f"https://example{i % 100}.com/page{i}",
                'position': i,
                'relevance_score': round(random.uniform(0.1, 1.0), 3)
            }
            for i in range(count)
        ]

        return await DatabaseOperations.bulk_insert_results_chunked(
            session,
            search_id,
            results_data,
            chunk_size=100
        )

    @staticmethod
    async def generate_bulk_products(
        session: AsyncSession,
        count: int = 1000
    ) -> List[Product]:
        """Generate bulk products"""
        products = []
        product_data_list = TestDataFactory.create_product_data(count)

        for product_data in product_data_list:
            product = await DatabaseOperations.create_or_update_product(
                session,
                product_data
            )
            products.append(product)

            if len(products) % 100 == 0:
                await session.commit()

        await session.commit()
        return products


# Export commonly used components
__all__ = [
    'TestDataFactory',
    'PerformanceTimer',
    'ConcurrentExecutor',
    'DatabaseTestFixture',
    'AsyncTestRunner',
    'TestAssertions',
    'MockDataGenerator',
    'TEST_CHUNK_SIZE',
    'TEST_CONCURRENCY_LEVEL',
    'PERFORMANCE_THRESHOLD_MS',
    'TEST_DATA_SIZE_SMALL',
    'TEST_DATA_SIZE_MEDIUM',
    'TEST_DATA_SIZE_LARGE'
]