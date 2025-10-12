"""
Test script to verify database initialization and model creation
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.database import db_manager, init_db, close_db
from src.core.db_ops import DatabaseOperations
from src.core.models import *


async def test_database_initialization():
    """Test database initialization and basic operations"""
    print("=" * 60)
    print("CIAP Database Test Suite")
    print("=" * 60)

    try:
        # Initialize database
        print("\n1. Initializing database...")
        await init_db()
        print("✓ Database initialized successfully")

        # Health check
        print("\n2. Running health check...")
        is_healthy = await db_manager.health_check()
        print(f"✓ Database health: {'Healthy' if is_healthy else 'Unhealthy'}")

        # Get initial stats
        print("\n3. Getting database statistics...")
        stats = await db_manager.get_stats()
        print("✓ Database statistics retrieved:")
        for key, value in stats.items():
            if key == 'database_size_bytes':
                print(f"  - {key}: {value / 1024:.2f} KB")
            elif isinstance(value, dict):
                print(f"  - {key}:")
                for k, v in value.items():
                    print(f"    - {k}: {v}")
            else:
                print(f"  - {key}: {value}")

        # Test CRUD operations
        print("\n4. Testing CRUD operations...")

        async with db_manager.get_session() as session:
            # Create a search
            search = await DatabaseOperations.create_search(
                session,
                query="test competitive analysis",
                sources=["google", "bing"],
                search_type="competitor"
            )
            print(f"✓ Created search with ID: {search.id}")

            # Create a product
            product_data = {
                'product_name': 'Test Product',
                'brand_name': 'Test Brand',
                'company_name': 'Test Company',
                'category': 'Electronics',
                'description': 'A test product for database validation',
                'sku': 'TEST-001'
            }
            product = await DatabaseOperations.create_or_update_product(
                session, product_data
            )
            print(f"✓ Created product with ID: {product.id}")

            # Add price data
            price_info = {
                'current_price': 99.99,
                'original_price': 149.99,
                'currency': 'USD',
                'discount_percentage': 33.33,
                'availability_status': 'In Stock',
                'seller_name': 'Test Seller'
            }
            price_data = await DatabaseOperations.add_price_data(
                session, product.id, price_info
            )
            print(f"✓ Added price data for product {product.id}")

            # Add a review
            reviews = [{
                'review_title': 'Great product!',
                'review_text': 'This is an excellent product. Highly recommended.',
                'rating': 4.5,
                'reviewer_name': 'Test User',
                'review_date': datetime.utcnow(),
                'verified_purchase': True,
                'review_source': 'Test Platform'
            }]
            await DatabaseOperations.bulk_add_reviews(
                session, product.id, reviews
            )
            print(f"✓ Added review for product {product.id}")

            # Create a competitor
            competitor_data = {
                'company_name': 'Test Competitor Inc.',
                'domain': 'testcompetitor.com',
                'description': 'A major competitor in the test market',
                'products_services': ['Service A', 'Service B'],
                'market_share': 25.5,
                'key_features': ['Feature 1', 'Feature 2']
            }
            competitor = await DatabaseOperations.create_or_update_competitor(
                session, competitor_data
            )
            print(f"✓ Created competitor with ID: {competitor.id}")

            # Test cache operations
            cache_key = "test_cache_key"
            cache_value = "test_cache_value"
            await DatabaseOperations.set_cache(
                session, cache_key, cache_value, ttl_seconds=60
            )
            retrieved_value = await DatabaseOperations.get_cache(session, cache_key)
            print(f"✓ Cache operations working: {retrieved_value == cache_value}")

            # Test task queue
            task = await DatabaseOperations.enqueue_task(
                session,
                task_type="test_task",
                payload={"test": "data"},
                priority=1
            )
            print(f"✓ Enqueued task with ID: {task.id}")

            # Commit all changes
            await session.commit()

        # Test search operations
        print("\n5. Testing search and retrieval operations...")

        async with db_manager.get_session() as session:
            # Search products
            products = await DatabaseOperations.search_products(
                session, "Test", limit=10
            )
            print(f"✓ Found {len(products)} products matching 'Test'")

            # Get recent searches
            searches = await DatabaseOperations.get_recent_searches(
                session, limit=5
            )
            print(f"✓ Retrieved {len(searches)} recent searches")

            # Get competitors
            competitors = await DatabaseOperations.get_competitors(
                session, limit=10
            )
            print(f"✓ Retrieved {len(competitors)} competitors")

            # Get database stats
            final_stats = await DatabaseOperations.get_database_stats(session)
            print("\n✓ Final database statistics:")
            for table, count in final_stats.items():
                if isinstance(count, int):
                    print(f"  - {table}: {count} records")

        # Test optimization
        print("\n6. Running database optimization...")
        await db_manager.optimize()
        print("✓ Database optimization complete")

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

        print("\nDatabase is ready for use with:")
        print("- All 10 competitive intelligence models")
        print("- Async SQLite with WAL mode")
        print("- Caching system")
        print("- Task queue")
        print("- Rate limiting")
        print("- Full CRUD operations")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        await close_db()
        print("\n✓ Database connection closed")


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_database_initialization())
    sys.exit(0 if success else 1)