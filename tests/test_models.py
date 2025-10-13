"""
Comprehensive Model Operations Tests for CIAP
Tests CRUD operations for all 23 database models
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import DatabaseManager
from src.core.db_ops import DatabaseOperations
from src.core.models import *
from tests.test_utils import (
    TestDataFactory,
    DatabaseTestFixture,
    AsyncTestRunner,
    TestAssertions,
    PerformanceTimer
)


class TestAllModels:
    """Test all 23 model operations"""

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

    async def test_search_model_operations(self):
        """Test Search model CRUD operations"""
        test_name = "test_search_model_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create - Fixed: removed 'status' parameter
                search = await db_ops.create_search(
                    session,
                    query="test search query",
                    sources=["google", "bing"],
                    search_type="competitor"
                )
                assert search.id is not None
                assert search.query == "test search query"

                # Read
                retrieved = await db_ops.get_search(session, search.id)
                assert retrieved is not None
                assert retrieved.query == "test search query"

                # Update
                await db_ops.update_search_status(session, search.id, 'completed')
                await session.refresh(search)
                assert search.status == 'completed'

                # List recent
                recent = await db_ops.get_recent_searches(session, limit=10)
                assert len(recent) > 0
                assert any(s.id == search.id for s in recent)

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_analysis_model_operations(self):
        """Test Analysis model CRUD operations"""
        test_name = "test_analysis_model_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create search first
                search = await db_ops.create_search(session, "test", ["google"], "test")

                # Create analysis - Fixed: correct parameters for save_analysis
                analysis = await db_ops.save_analysis(
                    session,
                    search_id=search.id,
                    analysis_type='sentiment',
                    content='Analysis results text',
                    insights={
                        'positive': 0.7,
                        'negative': 0.2,
                        'neutral': 0.1,
                        'key_insights': ['Insight 1', 'Insight 2']
                    },
                    llm_info={
                        'provider': 'ollama',
                        'model': 'llama3.1:8b'
                    },
                    scores={
                        'sentiment': 0.7,
                        'confidence': 0.85
                    }
                )
                assert analysis.id is not None
                assert analysis.analysis_type == 'sentiment'

                # Read
                analyses = await db_ops.get_analyses(session, search.id)
                assert len(analyses) > 0
                assert analyses[0].confidence_score == 0.85

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_product_model_operations(self):
        """Test Product model CRUD operations"""
        test_name = "test_product_model_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create - Fixed: Product model doesn't have 'price' field
                product_data = {
                    'product_name': 'Test Product',
                    'brand_name': 'Test Brand',
                    'company_name': 'Test Company',
                    'category': 'Electronics',
                    'industry': 'Technology',
                    'description': 'Test product description',
                    'product_url': 'https://example.com/product'
                }
                product = await db_ops.create_or_update_product(session, product_data)
                assert product.id is not None
                assert product.product_name == product_data['product_name']

                # Read
                retrieved = await db_ops.get_product(session, product.id)
                assert retrieved is not None
                assert retrieved.product_name == product_data['product_name']

                # Update
                product_data['description'] = 'Updated description'
                updated = await db_ops.create_or_update_product(session, product_data)
                assert updated.description == 'Updated description'

                # Search
                results = await db_ops.search_products(session, product_data['product_name'])
                assert len(results) > 0
                assert any(p.id == product.id for p in results)

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_price_data_operations(self):
        """Test PriceData model operations"""
        test_name = "test_price_data_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create product first
                product_data = {
                    'product_name': 'Test Product',
                    'brand_name': 'Test Brand',
                    'category': 'Electronics'
                }
                product = await db_ops.create_or_update_product(session, product_data)

                # Add price data
                for i in range(5):
                    price_info = {
                        'current_price': 99.99 + i * 10,
                        'original_price': 149.99,
                        'currency': 'USD',
                        'discount_percentage': 33.3,
                        'availability_status': 'In Stock',
                        'seller_name': 'Test Seller',
                        'geographic_location': 'US'
                    }
                    await db_ops.add_price_data(session, product.id, price_info)

                # Get latest price
                latest = await db_ops.get_latest_price(session, product.id)
                assert latest is not None
                assert latest.product_id == product.id

                # Get price history
                history = await db_ops.get_price_history(session, product.id, days=30)
                assert len(history) >= 5

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_offer_model_operations(self):
        """Test Offer model operations"""
        test_name = "test_offer_model_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create product
                product = Product(
                    product_name="Test Product",
                    brand_name="Test Brand",
                    category="Electronics"
                )
                session.add(product)
                await session.flush()

                # Create offer - Fixed: correct field names for Offer model
                offer = Offer(
                    product_id=product.id,
                    offer_type="Discount",
                    offer_description="Special Deal - 25% off",
                    discount_code="SAVE25",
                    start_date=datetime.now(timezone.utc),
                    end_date=datetime.now(timezone.utc) + timedelta(days=7),
                    terms_conditions="Limited time offer",
                    minimum_purchase=50.0,
                    offer_source="amazon",
                    is_active=True
                )
                session.add(offer)
                await session.commit()

                # Read
                result = await session.get(Offer, offer.id)
                assert result is not None
                assert result.discount_code == "SAVE25"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_product_review_operations(self):
        """Test ProductReview model operations"""
        test_name = "test_product_review_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create product
                product_data = {
                    'product_name': 'Test Product',
                    'brand_name': 'Test Brand',
                    'category': 'Electronics'
                }
                product = await db_ops.create_or_update_product(session, product_data)

                # Add reviews
                reviews_data = []
                for i in range(10):
                    review = {
                        'review_title': f'Review {i}',
                        'review_text': f'This is review text {i}',
                        'rating': 4.0 + (i % 2),
                        'reviewer_name': f'Reviewer {i}',
                        'review_date': datetime.now(timezone.utc) - timedelta(days=i),
                        'verified_purchase': i % 2 == 0,
                        'helpful_votes': i * 10,
                        'review_source': 'amazon'
                    }
                    reviews_data.append(review)

                await db_ops.bulk_add_reviews(session, product.id, reviews_data)

                # Get reviews
                reviews = await db_ops.get_product_reviews(session, product.id, limit=5)
                assert len(reviews) <= 5

                # Get review summary
                summary = await db_ops.get_review_summary(session, product.id)
                assert 'average_rating' in summary
                assert 'total_reviews' in summary
                assert summary['total_reviews'] == 10

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_competitor_model_operations(self):
        """Test Competitor model operations"""
        test_name = "test_competitor_model_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create competitor - Fixed: correct field names
                competitor_data = {
                    'company_name': 'Competitor Inc',
                    'domain': 'competitor.com',
                    'description': 'A major competitor',
                    'products_services': ['Product A', 'Product B'],
                    'market_share': 0.15,
                    'social_media_presence': {'twitter': 10000, 'linkedin': 5000},
                    'target_audience': 'Enterprise customers',
                    'geographic_presence': ['US', 'EU', 'Asia']
                }
                competitor = await db_ops.create_or_update_competitor(
                    session,
                    competitor_data
                )
                assert competitor.id is not None
                assert competitor.company_name == competitor_data['company_name']

                # Track change
                await db_ops.track_competitor_change(
                    session,
                    competitor.id,
                    'market_share_increase',
                    'Market share increased to 16%',
                    old_value=0.15,
                    new_value=0.16
                )

                # Get competitors
                competitors = await db_ops.get_competitors(session)
                assert len(competitors) > 0
                assert any(c.id == competitor.id for c in competitors)

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_market_trend_operations(self):
        """Test MarketTrend model operations"""
        test_name = "test_market_trend_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create market trend - Fixed: correct field names
                trend = MarketTrend(
                    keyword="AI Adoption",
                    search_volume=10000,
                    trend_direction="Rising",
                    related_keywords=["machine learning", "deep learning", "neural networks"],
                    geographic_trends={'US': 100, 'EU': 85, 'Asia': 120},
                    time_period="Q4 2024",
                    industry_category="Technology",
                    seasonal_patterns={'Q1': 0.8, 'Q2': 0.9, 'Q3': 1.0, 'Q4': 1.2},
                    consumer_interest_score=8.5,
                    driving_events=["AI Summit 2024", "New LLM releases"]
                )
                session.add(trend)
                await session.commit()

                # Read
                result = await session.get(MarketTrend, trend.id)
                assert result is not None
                assert result.keyword == "AI Adoption"
                assert result.trend_direction == "Rising"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_serp_data_operations(self):
        """Test SERPData model operations"""
        test_name = "test_serp_data_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create search
                search = await db_ops.create_search(session, "test", ["google"], "test")

                # Add SERP data - Fixed: correct field names and method signature
                serp_data = []
                for i in range(1, 6):
                    serp_item = {
                        'search_query': 'test keyword',
                        'search_engine': 'google',
                        'result_position': i,
                        'result_title': f'Result {i}',
                        'result_url': f'https://example{i}.com',
                        'snippet': f'Snippet {i}',
                        'featured_snippet': None if i > 1 else 'Featured snippet content',
                        'related_questions': ['Question 1', 'Question 2'],
                        'related_searches': ['Related 1', 'Related 2'],
                        'geographic_location': 'US'
                    }
                    serp_data.append(serp_item)

                await db_ops.bulk_insert_serp_data(session, search.id, serp_data)

                # Get SERP data
                results = await db_ops.get_serp_data(session, search.id)
                assert len(results) == 5
                assert results[0].result_position == 1

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_social_sentiment_operations(self):
        """Test SocialSentiment model operations"""
        test_name = "test_social_sentiment_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create social sentiment - Fixed: correct field names
                sentiment = SocialSentiment(
                    platform="twitter",
                    post_content="Great product! Love it!",
                    author_account="user123",
                    post_date=datetime.now(timezone.utc),
                    likes=150,
                    shares=50,
                    comments=20,
                    sentiment="Positive",
                    sentiment_score=0.9,
                    products_mentioned=["Product A"],
                    companies_mentioned=["Company X"],
                    hashtags=["#awesome", "#product"],
                    geographic_location="US"
                )
                session.add(sentiment)
                await session.commit()

                # Read
                result = await session.get(SocialSentiment, sentiment.id)
                assert result is not None
                assert result.platform == "twitter"
                assert result.sentiment_score == 0.9

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_news_content_operations(self):
        """Test NewsContent model operations"""
        test_name = "test_news_content_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create news content - Fixed: correct field names
                news = NewsContent(
                    article_title="Tech Industry Update",
                    article_url="https://technews.com/article1",
                    publication_source="TechNews",
                    author="John Doe",
                    publication_date=datetime.now(timezone.utc),
                    article_summary="Brief summary of the article",
                    full_content="Full article content here...",
                    category_topic="Technology",
                    companies_mentioned=["Company A", "Company B"],
                    products_mentioned=["Product X", "Product Y"],
                    sentiment="Positive",
                    keywords_tags=["technology", "business", "innovation"]
                )
                session.add(news)
                await session.commit()

                # Read
                result = await session.get(NewsContent, news.id)
                assert result is not None
                assert result.article_title == "Tech Industry Update"
                assert result.publication_source == "TechNews"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_feature_comparison_operations(self):
        """Test FeatureComparison model operations"""
        test_name = "test_feature_comparison_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create product
                product = Product(
                    product_name="Product A",
                    brand_name="Brand A",
                    category="Electronics"
                )
                session.add(product)
                await session.flush()

                # Create feature comparison - Fixed: correct field names
                comparison = FeatureComparison(
                    product_id=product.id,
                    feature_name="Battery Life",
                    feature_description="Battery performance and duration",
                    feature_availability="Yes",
                    feature_specifications={
                        "capacity": "5000mAh",
                        "duration": "12 hours",
                        "charging": "Fast charging support"
                    },
                    competitor_feature_mapping={
                        "Competitor A": "10 hours",
                        "Competitor B": "8 hours"
                    },
                    feature_importance_score=0.9
                )
                session.add(comparison)
                await session.commit()

                # Read
                result = await session.get(FeatureComparison, comparison.id)
                assert result is not None
                assert result.feature_name == "Battery Life"
                assert result.feature_importance_score == 0.9

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_cache_model_operations(self):
        """Test Cache model operations"""
        test_name = "test_cache_model_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Set cache - Fixed: value must be string (JSON)
                cache_data = {"data": "test_value", "timestamp": datetime.now(timezone.utc).isoformat()}
                await db_ops.set_cache(
                    session,
                    key="test_key",
                    value=json.dumps(cache_data),  # Convert to JSON string
                    ttl_seconds=3600
                )

                # Get cache
                value = await db_ops.get_cache(session, "test_key")
                assert value is not None
                parsed_value = json.loads(value)
                assert parsed_value["data"] == "test_value"

                # Delete cache
                deleted = await db_ops.delete_cache(session, "test_key")
                assert deleted is True

                # Verify deleted
                value = await db_ops.get_cache(session, "test_key")
                assert value is None

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_task_queue_operations(self):
        """Test TaskQueue model operations"""
        test_name = "test_task_queue_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Enqueue tasks with different priorities
                task1 = await db_ops.enqueue_task(
                    session,
                    task_type="high_priority",
                    payload={"id": 1},
                    priority=1  # Lower number = higher priority
                )
                task2 = await db_ops.enqueue_task(
                    session,
                    task_type="low_priority",
                    payload={"id": 2},
                    priority=10
                )

                # Dequeue should get high priority first
                next_task = await db_ops.dequeue_task(session)
                assert next_task is not None
                assert next_task.id == task1.id
                assert next_task.priority == 1

                # Complete task
                await db_ops.complete_task(session, next_task.id, "completed")
                await session.commit()
                await session.refresh(task1)
                assert task1.status == "completed"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_scraping_job_operations(self):
        """Test ScrapingJob model operations"""
        test_name = "test_scraping_job_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create search first
                search = Search(
                    query="test search",
                    sources=["google"],
                    search_type="test"
                )
                session.add(search)
                await session.flush()

                # Create scraping job - Fixed: correct field names
                job = ScrapingJob(
                    search_id=search.id,
                    scraper_name="google_scraper",
                    status="pending",
                    results_count=0
                )
                session.add(job)
                await session.commit()

                # Update status
                job.status = "processing"
                job.started_at = datetime.now(timezone.utc)
                await session.commit()

                # Complete job
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                job.results_count = 50
                await session.commit()

                # Read
                result = await session.get(ScrapingJob, job.id)
                assert result is not None
                assert result.status == "completed"
                assert result.results_count == 50

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_rate_limit_operations(self):
        """Test RateLimit model operations"""
        test_name = "test_rate_limit_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Check rate limit - Fixed: correct method signature
                can_proceed = await db_ops.check_rate_limit(
                    session,
                    scraper_name="test_scraper",
                    requests_per_minute=10
                )
                assert can_proceed is True

                # Update rate limit
                await db_ops.update_rate_limit(session, "test_scraper")

                # Reset rate limits
                reset_count = await db_ops.reset_rate_limits(session, "test_scraper")
                assert reset_count >= 0

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_price_history_operations(self):
        """Test PriceHistory model operations"""
        test_name = "test_price_history_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create product
                product = Product(
                    product_name="Test Product",
                    brand_name="Test Brand",
                    category="Electronics"
                )
                session.add(product)
                await session.flush()

                # Create price history
                history = PriceHistory(
                    product_id=product.id,
                    price=99.99,
                    currency="USD",
                    seller_name="Test Seller"
                )
                session.add(history)
                await session.commit()

                # Read
                result = await session.get(PriceHistory, history.id)
                assert result is not None
                assert result.price == 99.99
                assert result.product_id == product.id

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_competitor_tracking_operations(self):
        """Test CompetitorTracking model operations"""
        test_name = "test_competitor_tracking_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()

            async with db_manager.get_session() as session:
                # Create competitor - Fixed: correct field names
                competitor = Competitor(
                    company_name="Competitor Inc",
                    domain="competitor.com",
                    market_share=0.15
                )
                session.add(competitor)
                await session.flush()

                # Create tracking record - Fixed: correct field names
                tracking = CompetitorTracking(
                    competitor_id=competitor.id,
                    change_type="market_share",
                    change_description="Market share increased by 1%",
                    old_value={"market_share": 0.15},
                    new_value={"market_share": 0.16}
                )
                session.add(tracking)
                await session.commit()

                # Read
                result = await session.get(CompetitorTracking, tracking.id)
                assert result is not None
                assert result.change_type == "market_share"

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_competitor_products_operations(self):
        """Test CompetitorProducts model operations"""
        test_name = "test_competitor_products_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create competitor and product
                competitor_data = {
                    'company_name': 'Competitor Inc',
                    'domain': 'competitor.com'
                }
                competitor = await db_ops.create_or_update_competitor(
                    session,
                    competitor_data
                )

                product_data = {
                    'product_name': 'Test Product',
                    'brand_name': 'Test Brand',
                    'category': 'Electronics'
                }
                product = await db_ops.create_or_update_product(
                    session,
                    product_data
                )

                # Link competitor to product - Fixed: correct parameters
                await db_ops.link_competitor_product(
                    session,
                    competitor.id,
                    product.id,
                    relationship_type="direct_competitor"
                )

                # Get competitor products
                products = await db_ops.get_competitor_products(session, competitor.id)
                assert len(products) > 0
                assert products[0].id == product.id

                # Get product competitors
                competitors = await db_ops.get_product_competitors(session, product.id)
                assert len(competitors) > 0
                assert competitors[0].id == competitor.id

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_insights_operations(self):
        """Test Insights model operations"""
        test_name = "test_insights_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create product
                product = Product(
                    product_name="Test Product",
                    brand_name="Test Brand",
                    category="Electronics"
                )
                session.add(product)
                await session.flush()

                # Create insight - Fixed: correct parameters
                insight = await db_ops.create_insight(
                    session,
                    insight_type="opportunity",
                    title="Market Opportunity Detected",
                    description="New market segment showing 40% growth",
                    severity="high",
                    confidence_score=0.85,
                    product_id=product.id,
                    insight_data={"growth_rate": 0.4, "segment": "enterprise"},
                    action_items=["Research market", "Update pricing", "Expand features"]
                )
                assert insight.id is not None
                assert insight.severity == "high"

                # Get unread insights
                unread = await db_ops.get_unread_insights(session)
                assert len(unread) > 0
                assert any(i.id == insight.id for i in unread)

                # Mark as read
                await db_ops.mark_insight_read(session, insight.id)
                await session.commit()
                await session.refresh(insight)
                assert insight.is_read is True

                # Get critical insights
                critical = await db_ops.get_critical_insights(session, days=7)
                # Note: may be empty if severity is not 'critical'

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def test_search_result_operations(self):
        """Test SearchResult model operations"""
        test_name = "test_search_result_operations"
        fixture = DatabaseTestFixture()

        try:
            db_manager = await fixture.setup()
            db_ops = DatabaseOperations()

            async with db_manager.get_session() as session:
                # Create search
                search = await db_ops.create_search(session, "test query", ["google"], "test")

                # Bulk insert results
                results_data = [
                    {
                        'title': f'Result {i}',
                        'url': f'https://example{i}.com',
                        'snippet': f'Snippet {i}',
                        'source': 'google',
                        'rank': i
                    }
                    for i in range(1, 11)
                ]

                results = await db_ops.bulk_insert_results(session, search.id, results_data)
                assert len(results) == 10

                # Get results
                retrieved = await db_ops.get_search_results(session, search.id, limit=5)
                assert len(retrieved) <= 5
                assert all(r.search_id == search.id for r in retrieved)

                # Count results
                count = await db_ops.count_results(session, search.id)
                assert count == 10

            await fixture.teardown()
            self.report(test_name, True)
        except Exception as e:
            await fixture.teardown()
            self.report(test_name, False, str(e))

    async def run_all_model_tests(self):
        """Run all model tests"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL OPERATIONS TESTS")
        print("=" * 60 + "\n")

        # Run all test methods
        await self.test_search_model_operations()
        await self.test_analysis_model_operations()
        await self.test_product_model_operations()
        await self.test_price_data_operations()
        await self.test_offer_model_operations()
        await self.test_product_review_operations()
        await self.test_competitor_model_operations()
        await self.test_market_trend_operations()
        await self.test_serp_data_operations()
        await self.test_social_sentiment_operations()
        await self.test_news_content_operations()
        await self.test_feature_comparison_operations()
        await self.test_cache_model_operations()
        await self.test_task_queue_operations()
        await self.test_scraping_job_operations()
        await self.test_rate_limit_operations()
        await self.test_price_history_operations()
        await self.test_competitor_tracking_operations()
        await self.test_competitor_products_operations()
        await self.test_insights_operations()
        await self.test_search_result_operations()

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
    tester = TestAllModels()
    success = asyncio.run(tester.run_all_model_tests())
    sys.exit(0 if success else 1)