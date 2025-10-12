"""
Database Operations for CIAP
Provides CRUD operations and business logic for all models
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select, update, delete, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import json
import logging
from .models import (
    Search, SearchResult, Analysis, Product, PriceData, Offer,
    ProductReview, Competitor, MarketTrend, SERPData, SocialSentiment,
    NewsContent, FeatureComparison, Cache, TaskQueue, ScrapingJob,
    RateLimit, PriceHistory, CompetitorTracking
)
from .database import db_manager

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """Common database operations for all models"""

    # =========================
    # SEARCH OPERATIONS
    # =========================

    @staticmethod
    async def create_search(
        session: AsyncSession,
        query: str,
        sources: List[str],
        search_type: str = "competitor",
        user_id: Optional[str] = None
    ) -> Search:
        """Create new search record"""
        search = Search(
            query=query,
            sources=sources,
            search_type=search_type,
            user_id=user_id,
            status="pending"
        )
        session.add(search)
        await session.flush()
        logger.info(f"Created search {search.id} for query: {query}")
        return search

    @staticmethod
    async def get_search(session: AsyncSession, search_id: int) -> Optional[Search]:
        """Get search by ID with relationships"""
        result = await session.execute(
            select(Search).where(Search.id == search_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_search_status(
        session: AsyncSession,
        search_id: int,
        status: str,
        error: Optional[str] = None
    ):
        """Update search status"""
        stmt = update(Search).where(Search.id == search_id).values(
            status=status,
            updated_at=datetime.utcnow(),
            completed_at=datetime.utcnow() if status in ["completed", "failed"] else None,
            error_message=error
        )
        await session.execute(stmt)
        logger.info(f"Updated search {search_id} status to {status}")

    @staticmethod
    async def get_recent_searches(
        session: AsyncSession,
        limit: int = 10,
        user_id: Optional[str] = None
    ) -> List[Search]:
        """Get recent searches"""
        query = select(Search).order_by(desc(Search.created_at)).limit(limit)
        if user_id:
            query = query.where(Search.user_id == user_id)
        result = await session.execute(query)
        return result.scalars().all()

    # =========================
    # PRODUCT OPERATIONS
    # =========================

    @staticmethod
    async def create_or_update_product(
        session: AsyncSession,
        product_data: Dict[str, Any]
    ) -> Product:
        """Create or update product based on SKU or name"""
        sku = product_data.get('sku')

        # Try to find existing product
        if sku:
            result = await session.execute(
                select(Product).where(Product.sku == sku)
            )
            product = result.scalar_one_or_none()
        else:
            # Try to match by name and company
            result = await session.execute(
                select(Product).where(
                    and_(
                        Product.product_name == product_data.get('product_name'),
                        Product.company_name == product_data.get('company_name')
                    )
                )
            )
            product = result.scalar_one_or_none()

        if product:
            # Update existing product
            for key, value in product_data.items():
                setattr(product, key, value)
            product.updated_at = datetime.utcnow()
            logger.info(f"Updated product {product.id}: {product.product_name}")
        else:
            # Create new product
            product = Product(**product_data)
            session.add(product)
            await session.flush()
            logger.info(f"Created product {product.id}: {product.product_name}")

        return product

    @staticmethod
    async def get_product(
        session: AsyncSession,
        product_id: int
    ) -> Optional[Product]:
        """Get product with all related data"""
        result = await session.execute(
            select(Product).where(Product.id == product_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def search_products(
        session: AsyncSession,
        search_term: str,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[Product]:
        """Search products by name, brand, or category"""
        query = select(Product)

        # Search in multiple fields
        search_conditions = or_(
            Product.product_name.ilike(f"%{search_term}%"),
            Product.brand_name.ilike(f"%{search_term}%"),
            Product.company_name.ilike(f"%{search_term}%"),
            Product.description.ilike(f"%{search_term}%")
        )
        query = query.where(search_conditions)

        if category:
            query = query.where(Product.category == category)

        query = query.limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

    # =========================
    # PRICE DATA OPERATIONS
    # =========================

    @staticmethod
    async def add_price_data(
        session: AsyncSession,
        product_id: int,
        price_info: Dict[str, Any]
    ) -> PriceData:
        """Add price data for a product"""
        price_data = PriceData(product_id=product_id, **price_info)
        session.add(price_data)
        await session.flush()

        # Also add to price history
        history = PriceHistory(
            product_id=product_id,
            price=price_info['current_price'],
            currency=price_info.get('currency', 'USD'),
            seller_name=price_info.get('seller_name', 'Unknown')
        )
        session.add(history)

        logger.info(f"Added price data for product {product_id}")
        return price_data

    @staticmethod
    async def get_latest_price(
        session: AsyncSession,
        product_id: int
    ) -> Optional[PriceData]:
        """Get latest price for a product"""
        result = await session.execute(
            select(PriceData)
            .where(PriceData.product_id == product_id)
            .order_by(desc(PriceData.scraped_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_price_history(
        session: AsyncSession,
        product_id: int,
        days: int = 30
    ) -> List[PriceHistory]:
        """Get price history for a product"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = await session.execute(
            select(PriceHistory)
            .where(
                and_(
                    PriceHistory.product_id == product_id,
                    PriceHistory.recorded_at >= cutoff_date
                )
            )
            .order_by(PriceHistory.recorded_at)
        )
        return result.scalars().all()

    # =========================
    # REVIEW OPERATIONS
    # =========================

    @staticmethod
    async def bulk_add_reviews(
        session: AsyncSession,
        product_id: int,
        reviews: List[Dict[str, Any]]
    ):
        """Bulk add reviews for a product"""
        review_objects = [
            ProductReview(product_id=product_id, **review)
            for review in reviews
        ]
        session.add_all(review_objects)
        await session.flush()
        logger.info(f"Added {len(reviews)} reviews for product {product_id}")

    @staticmethod
    async def get_product_reviews(
        session: AsyncSession,
        product_id: int,
        limit: int = 50,
        min_rating: Optional[float] = None
    ) -> List[ProductReview]:
        """Get reviews for a product"""
        query = select(ProductReview).where(ProductReview.product_id == product_id)

        if min_rating:
            query = query.where(ProductReview.rating >= min_rating)

        query = query.order_by(desc(ProductReview.review_date)).limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def get_review_summary(
        session: AsyncSession,
        product_id: int
    ) -> Dict[str, Any]:
        """Get review statistics for a product"""
        result = await session.execute(
            select(
                func.count(ProductReview.id).label('total_reviews'),
                func.avg(ProductReview.rating).label('avg_rating'),
                func.sum(func.cast(ProductReview.verified_purchase, Integer)).label('verified_count')
            ).where(ProductReview.product_id == product_id)
        )
        row = result.one()

        # Get rating distribution
        rating_dist = await session.execute(
            select(
                ProductReview.rating,
                func.count(ProductReview.id).label('count')
            )
            .where(ProductReview.product_id == product_id)
            .group_by(ProductReview.rating)
        )

        return {
            'total_reviews': row.total_reviews or 0,
            'average_rating': float(row.avg_rating) if row.avg_rating else 0,
            'verified_purchases': row.verified_count or 0,
            'rating_distribution': {
                int(r.rating): r.count for r in rating_dist
            }
        }

    # =========================
    # COMPETITOR OPERATIONS
    # =========================

    @staticmethod
    async def create_or_update_competitor(
        session: AsyncSession,
        competitor_data: Dict[str, Any]
    ) -> Competitor:
        """Create or update competitor profile"""
        company_name = competitor_data.get('company_name')

        result = await session.execute(
            select(Competitor).where(Competitor.company_name == company_name)
        )
        competitor = result.scalar_one_or_none()

        if competitor:
            # Update existing
            for key, value in competitor_data.items():
                setattr(competitor, key, value)
            competitor.updated_at = datetime.utcnow()
            logger.info(f"Updated competitor {competitor.id}: {company_name}")
        else:
            # Create new
            competitor = Competitor(**competitor_data)
            session.add(competitor)
            await session.flush()
            logger.info(f"Created competitor {competitor.id}: {company_name}")

        return competitor

    @staticmethod
    async def track_competitor_change(
        session: AsyncSession,
        competitor_id: int,
        change_type: str,
        change_description: str,
        old_value: Any = None,
        new_value: Any = None
    ):
        """Track changes to competitor data"""
        tracking = CompetitorTracking(
            competitor_id=competitor_id,
            change_type=change_type,
            change_description=change_description,
            old_value=old_value if isinstance(old_value, (dict, list)) else {"value": old_value},
            new_value=new_value if isinstance(new_value, (dict, list)) else {"value": new_value}
        )
        session.add(tracking)
        await session.flush()
        logger.info(f"Tracked change for competitor {competitor_id}: {change_type}")

    @staticmethod
    async def get_competitors(
        session: AsyncSession,
        limit: int = 20
    ) -> List[Competitor]:
        """Get all competitors"""
        result = await session.execute(
            select(Competitor)
            .order_by(desc(Competitor.updated_at))
            .limit(limit)
        )
        return result.scalars().all()

    # =========================
    # TASK QUEUE OPERATIONS
    # =========================

    @staticmethod
    async def enqueue_task(
        session: AsyncSession,
        task_type: str,
        payload: Dict,
        priority: int = 5
    ) -> TaskQueue:
        """Add task to queue"""
        task = TaskQueue(
            task_type=task_type,
            payload=payload,
            priority=priority,
            status="pending"
        )
        session.add(task)
        await session.flush()
        logger.info(f"Enqueued task {task.id} of type {task_type}")
        return task

    @staticmethod
    async def dequeue_task(session: AsyncSession) -> Optional[TaskQueue]:
        """Get next pending task by priority"""
        result = await session.execute(
            select(TaskQueue)
            .where(TaskQueue.status == "pending")
            .order_by(TaskQueue.priority, TaskQueue.scheduled_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        task = result.scalar_one_or_none()

        if task:
            task.status = "processing"
            task.started_at = datetime.utcnow()
            await session.flush()
            logger.info(f"Dequeued task {task.id} for processing")

        return task

    @staticmethod
    async def complete_task(
        session: AsyncSession,
        task_id: int,
        status: str = "completed",
        error: Optional[str] = None
    ):
        """Mark task as completed or failed"""
        stmt = update(TaskQueue).where(TaskQueue.id == task_id).values(
            status=status,
            completed_at=datetime.utcnow(),
            error_message=error
        )
        await session.execute(stmt)
        logger.info(f"Task {task_id} marked as {status}")

    @staticmethod
    async def retry_task(
        session: AsyncSession,
        task_id: int
    ) -> bool:
        """Retry a failed task"""
        result = await session.execute(
            select(TaskQueue).where(TaskQueue.id == task_id)
        )
        task = result.scalar_one_or_none()

        if task and task.retry_count < task.max_retries:
            task.status = "pending"
            task.retry_count += 1
            task.error_message = None
            task.scheduled_at = datetime.utcnow() + timedelta(minutes=5 * task.retry_count)
            await session.flush()
            logger.info(f"Task {task_id} scheduled for retry #{task.retry_count}")
            return True

        return False

    # =========================
    # CACHE OPERATIONS
    # =========================

    @staticmethod
    async def get_cache(
        session: AsyncSession,
        key: str
    ) -> Optional[str]:
        """Get cached value if not expired"""
        result = await session.execute(
            select(Cache)
            .where(
                and_(
                    Cache.key == key,
                    Cache.expires_at > datetime.utcnow()
                )
            )
        )
        cache_entry = result.scalar_one_or_none()

        if cache_entry:
            logger.debug(f"Cache hit for key: {key}")
            return cache_entry.value

        logger.debug(f"Cache miss for key: {key}")
        return None

    @staticmethod
    async def set_cache(
        session: AsyncSession,
        key: str,
        value: str,
        ttl_seconds: int = 3600
    ):
        """Set cache value with expiration"""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        # Delete existing entry if present
        await session.execute(delete(Cache).where(Cache.key == key))

        cache_entry = Cache(
            key=key,
            value=value,
            expires_at=expires_at
        )
        session.add(cache_entry)
        await session.flush()
        logger.debug(f"Cached key: {key} with TTL: {ttl_seconds}s")

    @staticmethod
    async def cleanup_expired_cache(session: AsyncSession) -> int:
        """Remove expired cache entries"""
        result = await session.execute(
            delete(Cache).where(Cache.expires_at < datetime.utcnow())
        )
        count = result.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")
        return count

    # =========================
    # ANALYSIS OPERATIONS
    # =========================

    @staticmethod
    async def save_analysis(
        session: AsyncSession,
        search_id: int,
        analysis_type: str,
        content: str,
        insights: Dict,
        llm_info: Dict,
        scores: Optional[Dict] = None
    ) -> Analysis:
        """Save analysis results"""
        analysis = Analysis(
            search_id=search_id,
            analysis_type=analysis_type,
            content=content,
            insights=insights,
            llm_provider=llm_info.get('provider', 'ollama'),
            llm_model=llm_info.get('model', 'llama3.1:8b'),
            sentiment_score=scores.get('sentiment') if scores else None,
            confidence_score=scores.get('confidence') if scores else None
        )
        session.add(analysis)
        await session.flush()
        logger.info(f"Saved {analysis_type} analysis for search {search_id}")
        return analysis

    @staticmethod
    async def get_analyses(
        session: AsyncSession,
        search_id: int,
        analysis_type: Optional[str] = None
    ) -> List[Analysis]:
        """Get analyses for a search"""
        query = select(Analysis).where(Analysis.search_id == search_id)

        if analysis_type:
            query = query.where(Analysis.analysis_type == analysis_type)

        query = query.order_by(desc(Analysis.created_at))
        result = await session.execute(query)
        return result.scalars().all()

    # =========================
    # SERP DATA OPERATIONS
    # =========================

    @staticmethod
    async def bulk_insert_serp_data(
        session: AsyncSession,
        search_id: int,
        serp_results: List[Dict[str, Any]]
    ):
        """Bulk insert SERP data"""
        serp_objects = [
            SERPData(search_id=search_id, **result)
            for result in serp_results
        ]
        session.add_all(serp_objects)
        await session.flush()
        logger.info(f"Added {len(serp_results)} SERP results for search {search_id}")

    @staticmethod
    async def get_serp_data(
        session: AsyncSession,
        search_id: int,
        search_engine: Optional[str] = None
    ) -> List[SERPData]:
        """Get SERP data for a search"""
        query = select(SERPData).where(SERPData.search_id == search_id)

        if search_engine:
            query = query.where(SERPData.search_engine == search_engine)

        query = query.order_by(SERPData.result_position)
        result = await session.execute(query)
        return result.scalars().all()

    # =========================
    # RATE LIMITING
    # =========================

    @staticmethod
    async def check_rate_limit(
        session: AsyncSession,
        scraper_name: str,
        requests_per_minute: int = 10
    ) -> bool:
        """Check if scraper is within rate limit"""
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)

        result = await session.execute(
            select(RateLimit)
            .where(
                and_(
                    RateLimit.scraper_name == scraper_name,
                    RateLimit.last_request_at > one_minute_ago
                )
            )
        )
        rate_limit = result.scalar_one_or_none()

        if rate_limit and rate_limit.request_count >= requests_per_minute:
            logger.warning(f"Rate limit exceeded for {scraper_name}")
            return False

        return True

    @staticmethod
    async def update_rate_limit(
        session: AsyncSession,
        scraper_name: str
    ):
        """Update rate limit counter"""
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)

        result = await session.execute(
            select(RateLimit)
            .where(RateLimit.scraper_name == scraper_name)
        )
        rate_limit = result.scalar_one_or_none()

        if rate_limit:
            if rate_limit.last_request_at < one_minute_ago:
                # Reset counter
                rate_limit.request_count = 1
                rate_limit.last_request_at = datetime.utcnow()
                rate_limit.reset_at = datetime.utcnow() + timedelta(minutes=1)
            else:
                # Increment counter
                rate_limit.request_count += 1
                rate_limit.last_request_at = datetime.utcnow()
        else:
            # Create new rate limit entry
            rate_limit = RateLimit(
                scraper_name=scraper_name,
                last_request_at=datetime.utcnow(),
                request_count=1,
                reset_at=datetime.utcnow() + timedelta(minutes=1)
            )
            session.add(rate_limit)

        await session.flush()

    # =========================
    # STATISTICS
    # =========================

    @staticmethod
    async def get_database_stats(session: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {}

        # Get record counts for all tables
        table_models = [
            Search, SearchResult, Analysis, Product, PriceData, Offer,
            ProductReview, Competitor, MarketTrend, SERPData, SocialSentiment,
            NewsContent, FeatureComparison, Cache, TaskQueue, ScrapingJob, RateLimit
        ]

        for model in table_models:
            result = await session.execute(select(func.count()).select_from(model))
            stats[model.__tablename__] = result.scalar()

        # Get task queue breakdown
        result = await session.execute(
            select(
                TaskQueue.status,
                func.count(TaskQueue.id)
            ).group_by(TaskQueue.status)
        )
        stats['task_queue_by_status'] = {
            row[0]: row[1] for row in result
        }

        # Get active cache entries
        result = await session.execute(
            select(func.count())
            .select_from(Cache)
            .where(Cache.expires_at > datetime.utcnow())
        )
        stats['active_cache_entries'] = result.scalar()

        # Get recent search activity
        last_24h = datetime.utcnow() - timedelta(days=1)
        result = await session.execute(
            select(func.count())
            .select_from(Search)
            .where(Search.created_at > last_24h)
        )
        stats['searches_last_24h'] = result.scalar()

        return stats