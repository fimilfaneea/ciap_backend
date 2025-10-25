"""
Database Operations for CIAP
Provides CRUD operations and business logic for all models
"""

from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator, Callable, Type, TypeVar, Generic
from sqlalchemy import select, update, delete, and_, or_, func, desc, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import asyncio
from .models import (
    Search, SearchResult, Analysis, Product, PriceData, Offer,
    ProductReview, Competitor, MarketTrend, SERPData, SocialSentiment,
    NewsContent, FeatureComparison, Cache, TaskQueue, ScrapingJob,
    RateLimit, PriceHistory, CompetitorTracking, CompetitorProducts, Insights
)
from .manager import db_manager

logger = logging.getLogger(__name__)

# Type definitions for pagination
T = TypeVar('T')

@dataclass
class PaginatedResult(Generic[T]):
    """Pagination result wrapper with metadata"""
    items: List[T]
    total: int
    page: int
    per_page: int
    total_pages: int
    has_prev: bool
    has_next: bool


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
    # SEARCH RESULT OPERATIONS
    # =========================

    @staticmethod
    async def bulk_insert_results(
        session: AsyncSession,
        search_id: int,
        results: List[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Bulk insert search results"""
        result_objects = []
        for result in results:
            search_result = SearchResult(
                search_id=search_id,
                title=result.get('title', 'Untitled'),
                url=result.get('url'),
                snippet=result.get('snippet'),
                source=result.get('source', 'unknown'),
                position=result.get('rank', result.get('position'))
            )
            result_objects.append(search_result)

        session.add_all(result_objects)
        await session.flush()
        logger.info(f"Bulk inserted {len(result_objects)} search results for search {search_id}")
        return result_objects

    @staticmethod
    async def bulk_insert_results_chunked(
        session: AsyncSession,
        search_id: int,
        results: List[Dict[str, Any]],
        chunk_size: int = 100,
        progress_callback: Optional[Callable] = None
    ) -> List[SearchResult]:
        """
        Bulk insert search results with chunk processing for large datasets.

        Args:
            session: Database session
            search_id: ID of the parent search
            results: List of result dictionaries to insert
            chunk_size: Number of records to process at once
            progress_callback: Optional async callback for progress updates
                              Called with (current_chunk, total_chunks, total_processed)

        Returns:
            List of inserted SearchResult objects
        """
        if not results:
            return []

        inserted_results = []
        total_chunks = (len(results) + chunk_size - 1) // chunk_size

        for chunk_idx in range(0, len(results), chunk_size):
            chunk = results[chunk_idx:chunk_idx + chunk_size]
            chunk_objects = []

            for result in chunk:
                search_result = SearchResult(
                    search_id=search_id,
                    title=result.get('title', 'Untitled'),
                    url=result.get('url'),
                    snippet=result.get('snippet'),
                    source=result.get('source', 'unknown'),
                    position=result.get('rank', result.get('position'))
                )
                chunk_objects.append(search_result)

            session.add_all(chunk_objects)
            await session.flush()
            inserted_results.extend(chunk_objects)

            # Call progress callback if provided
            if progress_callback:
                current_chunk = (chunk_idx // chunk_size) + 1
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(current_chunk, total_chunks, len(inserted_results))
                else:
                    progress_callback(current_chunk, total_chunks, len(inserted_results))

            # Allow other tasks to run between chunks
            await asyncio.sleep(0)

        logger.info(f"Bulk inserted {len(inserted_results)} search results in {total_chunks} chunks")
        return inserted_results

    @staticmethod
    async def get_search_results(
        session: AsyncSession,
        search_id: int,
        source: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Get search results with optional filtering"""
        query = select(SearchResult).where(SearchResult.search_id == search_id)

        if source:
            query = query.where(SearchResult.source == source)

        query = query.order_by(SearchResult.position.nullslast(), SearchResult.scraped_at)

        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def update_result_analysis(
        session: AsyncSession,
        result_id: int,
        analysis: Dict[str, Any]
    ):
        """Update analysis data for a search result"""
        update_data = {}

        # Map analysis fields to SearchResult model fields
        if 'sentiment' in analysis or 'sentiment_score' in analysis:
            update_data['sentiment_score'] = analysis.get('sentiment_score', analysis.get('sentiment'))
        if 'competitors' in analysis:
            update_data['competitor_mentioned'] = analysis.get('competitors')
        if 'keywords' in analysis:
            update_data['keywords'] = analysis.get('keywords')

        # Store all analysis data in result_metadata
        update_data['result_metadata'] = analysis

        stmt = update(SearchResult).where(SearchResult.id == result_id).values(**update_data)
        await session.execute(stmt)
        logger.info(f"Updated analysis for search result {result_id}")

    @staticmethod
    async def count_results(
        session: AsyncSession,
        search_id: int,
        source: Optional[str] = None
    ) -> int:
        """Count search results for a search"""
        query = select(func.count(SearchResult.id)).where(SearchResult.search_id == search_id)

        if source:
            query = query.where(SearchResult.source == source)

        result = await session.execute(query)
        return result.scalar() or 0

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
            ProductReview(product_id=product_id, **{k: v for k, v in review.items() if k != 'product_id'})
            for review in reviews
        ]
        session.add_all(review_objects)
        await session.flush()
        logger.info(f"Added {len(reviews)} reviews for product {product_id}")

    @staticmethod
    async def bulk_add_reviews_chunked(
        session: AsyncSession,
        product_id: int,
        reviews: List[Dict[str, Any]],
        chunk_size: int = 100,
        progress_callback: Optional[Callable] = None
    ) -> int:
        """
        Bulk add product reviews with chunk processing for large datasets.

        Returns:
            Total number of reviews inserted
        """
        if not reviews:
            return 0

        total_inserted = 0
        total_chunks = (len(reviews) + chunk_size - 1) // chunk_size

        for chunk_idx in range(0, len(reviews), chunk_size):
            chunk = reviews[chunk_idx:chunk_idx + chunk_size]
            review_objects = [
                ProductReview(product_id=product_id, **{k: v for k, v in review.items() if k != 'product_id'})
                for review in chunk
            ]

            session.add_all(review_objects)
            await session.flush()
            total_inserted += len(review_objects)

            if progress_callback:
                current_chunk = (chunk_idx // chunk_size) + 1
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(current_chunk, total_chunks, total_inserted)
                else:
                    progress_callback(current_chunk, total_chunks, total_inserted)

            await asyncio.sleep(0)

        logger.info(f"Added {total_inserted} reviews for product {product_id} in {total_chunks} chunks")
        return total_inserted

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

    @staticmethod
    async def get_pending_tasks(
        session: AsyncSession,
        task_type: Optional[str] = None,
        limit: int = 100
    ) -> List[TaskQueue]:
        """Get list of pending tasks with optional filtering"""
        query = select(TaskQueue).where(TaskQueue.status == "pending")

        if task_type:
            query = query.where(TaskQueue.task_type == task_type)

        query = query.order_by(
            TaskQueue.priority.desc(),
            TaskQueue.scheduled_at
        ).limit(limit)

        result = await session.execute(query)
        tasks = result.scalars().all()
        logger.info(f"Retrieved {len(tasks)} pending tasks")
        return tasks

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
    async def delete_cache(
        session: AsyncSession,
        key: str
    ) -> bool:
        """Delete specific cache entry by key"""
        result = await session.execute(
            delete(Cache).where(Cache.key == key)
        )
        if result.rowcount > 0:
            logger.info(f"Deleted cache entry for key: {key}")
            return True
        logger.debug(f"No cache entry found for key: {key}")
        return False

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

    @staticmethod
    async def reset_rate_limits(
        session: AsyncSession,
        scraper_name: Optional[str] = None
    ) -> int:
        """Reset rate limits for all or specific scrapers"""
        if scraper_name:
            # Reset specific scraper
            stmt = update(RateLimit).where(
                RateLimit.scraper_name == scraper_name
            ).values(
                request_count=0,
                reset_at=datetime.utcnow()
            )
        else:
            # Reset all scrapers
            stmt = update(RateLimit).values(
                request_count=0,
                reset_at=datetime.utcnow()
            )

        result = await session.execute(stmt)
        count = result.rowcount

        if scraper_name:
            logger.info(f"Reset rate limit for scraper: {scraper_name}")
        else:
            logger.info(f"Reset rate limits for {count} scrapers")

        return count

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

    # =========================
    # BATCH UPDATE OPERATIONS
    # =========================

    @staticmethod
    async def batch_update_search_status(
        session: AsyncSession,
        updates: List[Dict[str, Any]]  # [{"id": 1, "status": "completed"}, ...]
    ) -> int:
        """
        Batch update multiple search statuses efficiently.

        Args:
            session: Database session
            updates: List of dicts with 'id', 'status', and optional 'error' keys

        Returns:
            Number of records updated
        """
        if not updates:
            return 0

        updated_count = 0
        for item in updates:
            stmt = update(Search).where(Search.id == item["id"]).values(
                status=item.get("status"),
                error_message=item.get("error"),
                updated_at=datetime.utcnow(),
                completed_at=datetime.utcnow() if item.get("status") in ["completed", "failed"] else None
            )
            result = await session.execute(stmt)
            updated_count += result.rowcount

        await session.flush()
        logger.info(f"Batch updated {updated_count} search statuses")
        return updated_count

    @staticmethod
    async def batch_update_search_results(
        session: AsyncSession,
        updates: List[Dict[str, Any]]
    ) -> int:
        """
        Batch update multiple search results efficiently.

        Args:
            session: Database session
            updates: List of dicts with 'id' and fields to update

        Returns:
            Number of records updated
        """
        if not updates:
            return 0

        updated_count = 0
        for item in updates:
            result_id = item.pop("id")
            if not result_id:
                continue

            stmt = update(SearchResult).where(SearchResult.id == result_id).values(**item)
            result = await session.execute(stmt)
            updated_count += result.rowcount

        await session.flush()
        logger.info(f"Batch updated {updated_count} search results")
        return updated_count

    @staticmethod
    async def batch_update_task_status(
        session: AsyncSession,
        task_ids: List[int],
        status: str,
        error_message: Optional[str] = None
    ) -> int:
        """
        Batch update task statuses for multiple tasks.

        Args:
            session: Database session
            task_ids: List of task IDs to update
            status: New status for all tasks
            error_message: Optional error message

        Returns:
            Number of tasks updated
        """
        if not task_ids:
            return 0

        update_values = {
            "status": status,
            "error_message": error_message
        }

        if status == "completed" or status == "failed":
            update_values["completed_at"] = datetime.utcnow()
        elif status == "processing":
            update_values["started_at"] = datetime.utcnow()

        stmt = update(TaskQueue).where(TaskQueue.id.in_(task_ids)).values(**update_values)
        result = await session.execute(stmt)

        await session.flush()
        logger.info(f"Batch updated {result.rowcount} tasks to status: {status}")
        return result.rowcount

    @staticmethod
    async def batch_update_product_prices(
        session: AsyncSession,
        price_updates: List[Dict[str, Any]]
    ) -> int:
        """
        Batch update product prices and add to price history.

        Args:
            session: Database session
            price_updates: List of dicts with 'product_id', 'current_price', etc.

        Returns:
            Number of price records created
        """
        if not price_updates:
            return 0

        price_objects = []
        history_objects = []

        for update in price_updates:
            # Create new price data entry
            price_data = PriceData(
                product_id=update["product_id"],
                current_price=update["current_price"],
                original_price=update.get("original_price"),
                currency=update.get("currency", "USD"),
                discount_percentage=update.get("discount_percentage"),
                availability_status=update.get("availability_status", "In Stock"),
                seller_name=update.get("seller_name", "Unknown"),
                shipping_cost=update.get("shipping_cost")
            )
            price_objects.append(price_data)

            # Also add to price history
            history = PriceHistory(
                product_id=update["product_id"],
                price=update["current_price"],
                currency=update.get("currency", "USD"),
                seller_name=update.get("seller_name", "Unknown")
            )
            history_objects.append(history)

        session.add_all(price_objects)
        session.add_all(history_objects)
        await session.flush()

        logger.info(f"Batch updated {len(price_objects)} product prices")
        return len(price_objects)

    # =========================
    # STREAMING OPERATIONS
    # =========================

    @staticmethod
    async def stream_search_results(
        session: AsyncSession,
        search_id: int,
        chunk_size: int = 100
    ) -> AsyncGenerator[List[SearchResult], None]:
        """
        Stream search results in chunks to handle large datasets efficiently.

        Args:
            session: Database session
            search_id: ID of the search
            chunk_size: Number of records per chunk

        Yields:
            List of SearchResult objects in chunks
        """
        offset = 0

        while True:
            query = (
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
                .order_by(SearchResult.position.nullslast(), SearchResult.scraped_at)
                .limit(chunk_size)
                .offset(offset)
            )

            result = await session.execute(query)
            chunk = result.scalars().all()

            if not chunk:
                break

            yield chunk
            offset += chunk_size

            # Allow other tasks to run
            await asyncio.sleep(0)

    @staticmethod
    async def stream_products(
        session: AsyncSession,
        category: Optional[str] = None,
        chunk_size: int = 50
    ) -> AsyncGenerator[List[Product], None]:
        """
        Stream products by category in chunks.

        Args:
            session: Database session
            category: Optional category filter
            chunk_size: Number of records per chunk

        Yields:
            List of Product objects in chunks
        """
        offset = 0

        while True:
            query = select(Product)

            if category:
                query = query.where(Product.category == category)

            query = query.order_by(Product.id).limit(chunk_size).offset(offset)
            result = await session.execute(query)
            chunk = result.scalars().all()

            if not chunk:
                break

            yield chunk
            offset += chunk_size
            await asyncio.sleep(0)

    @staticmethod
    async def stream_reviews(
        session: AsyncSession,
        product_id: int,
        min_rating: Optional[float] = None,
        chunk_size: int = 100
    ) -> AsyncGenerator[List[ProductReview], None]:
        """
        Stream product reviews in chunks.

        Args:
            session: Database session
            product_id: ID of the product
            min_rating: Optional minimum rating filter
            chunk_size: Number of records per chunk

        Yields:
            List of ProductReview objects in chunks
        """
        offset = 0

        while True:
            query = select(ProductReview).where(ProductReview.product_id == product_id)

            if min_rating:
                query = query.where(ProductReview.rating >= min_rating)

            query = query.order_by(desc(ProductReview.review_date)).limit(chunk_size).offset(offset)
            result = await session.execute(query)
            chunk = result.scalars().all()

            if not chunk:
                break

            yield chunk
            offset += chunk_size
            await asyncio.sleep(0)

    @staticmethod
    async def stream_task_queue(
        session: AsyncSession,
        status: str = "pending",
        chunk_size: int = 50
    ) -> AsyncGenerator[List[TaskQueue], None]:
        """
        Stream tasks from queue in priority order.

        Args:
            session: Database session
            status: Task status filter
            chunk_size: Number of records per chunk

        Yields:
            List of TaskQueue objects in chunks
        """
        offset = 0

        while True:
            query = (
                select(TaskQueue)
                .where(TaskQueue.status == status)
                .order_by(TaskQueue.priority, TaskQueue.scheduled_at)
                .limit(chunk_size)
                .offset(offset)
            )

            result = await session.execute(query)
            chunk = result.scalars().all()

            if not chunk:
                break

            yield chunk
            offset += chunk_size
            await asyncio.sleep(0)

    # =========================
    # PAGINATION OPERATIONS
    # =========================

    @staticmethod
    async def get_paginated_searches(
        session: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        search_type: Optional[str] = None
    ) -> PaginatedResult[Search]:
        """
        Get paginated search results with metadata.

        Args:
            session: Database session
            page: Page number (1-indexed)
            per_page: Items per page
            status: Optional status filter
            user_id: Optional user ID filter
            search_type: Optional search type filter

        Returns:
            PaginatedResult containing items and pagination metadata
        """
        # Build count query
        count_query = select(func.count(Search.id))
        if status:
            count_query = count_query.where(Search.status == status)
        if user_id:
            count_query = count_query.where(Search.user_id == user_id)
        if search_type:
            count_query = count_query.where(Search.search_type == search_type)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Calculate pagination metadata
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        offset = (page - 1) * per_page

        # Build data query
        query = select(Search)
        if status:
            query = query.where(Search.status == status)
        if user_id:
            query = query.where(Search.user_id == user_id)
        if search_type:
            query = query.where(Search.search_type == search_type)

        query = query.order_by(desc(Search.created_at)).limit(per_page).offset(offset)

        # Get items
        result = await session.execute(query)
        items = result.scalars().all()

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_prev=page > 1,
            has_next=page < total_pages
        )

    @staticmethod
    async def get_paginated_products(
        session: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        search_term: Optional[str] = None
    ) -> PaginatedResult[Product]:
        """
        Get paginated product results with metadata.

        Args:
            session: Database session
            page: Page number (1-indexed)
            per_page: Items per page
            category: Optional category filter
            brand: Optional brand filter
            search_term: Optional search term

        Returns:
            PaginatedResult containing products and pagination metadata
        """
        # Build count query
        count_query = select(func.count(Product.id))

        if category:
            count_query = count_query.where(Product.category == category)
        if brand:
            count_query = count_query.where(Product.brand_name == brand)
        if search_term:
            count_query = count_query.where(
                or_(
                    Product.product_name.ilike(f"%{search_term}%"),
                    Product.description.ilike(f"%{search_term}%")
                )
            )

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Calculate pagination
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        offset = (page - 1) * per_page

        # Build data query
        query = select(Product)
        if category:
            query = query.where(Product.category == category)
        if brand:
            query = query.where(Product.brand_name == brand)
        if search_term:
            query = query.where(
                or_(
                    Product.product_name.ilike(f"%{search_term}%"),
                    Product.description.ilike(f"%{search_term}%")
                )
            )

        query = query.order_by(Product.product_name).limit(per_page).offset(offset)

        # Get items
        result = await session.execute(query)
        items = result.scalars().all()

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_prev=page > 1,
            has_next=page < total_pages
        )

    @staticmethod
    async def get_paginated_reviews(
        session: AsyncSession,
        product_id: int,
        page: int = 1,
        per_page: int = 20,
        min_rating: Optional[float] = None,
        verified_only: bool = False
    ) -> PaginatedResult[ProductReview]:
        """
        Get paginated review results with metadata.

        Args:
            session: Database session
            product_id: Product ID filter
            page: Page number (1-indexed)
            per_page: Items per page
            min_rating: Optional minimum rating filter
            verified_only: Only show verified purchases

        Returns:
            PaginatedResult containing reviews and pagination metadata
        """
        # Build count query
        count_query = select(func.count(ProductReview.id)).where(
            ProductReview.product_id == product_id
        )

        if min_rating:
            count_query = count_query.where(ProductReview.rating >= min_rating)
        if verified_only:
            count_query = count_query.where(ProductReview.verified_purchase == True)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Calculate pagination
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        offset = (page - 1) * per_page

        # Build data query
        query = select(ProductReview).where(ProductReview.product_id == product_id)
        if min_rating:
            query = query.where(ProductReview.rating >= min_rating)
        if verified_only:
            query = query.where(ProductReview.verified_purchase == True)

        query = query.order_by(desc(ProductReview.review_date)).limit(per_page).offset(offset)

        # Get items
        result = await session.execute(query)
        items = result.scalars().all()

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_prev=page > 1,
            has_next=page < total_pages
        )

    @staticmethod
    async def get_paginated_competitors(
        session: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        market_share_min: Optional[float] = None
    ) -> PaginatedResult[Competitor]:
        """
        Get paginated competitor results with metadata.

        Args:
            session: Database session
            page: Page number (1-indexed)
            per_page: Items per page
            market_share_min: Optional minimum market share filter

        Returns:
            PaginatedResult containing competitors and pagination metadata
        """
        # Build count query
        count_query = select(func.count(Competitor.id))
        if market_share_min:
            count_query = count_query.where(Competitor.market_share >= market_share_min)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Calculate pagination
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        offset = (page - 1) * per_page

        # Build data query
        query = select(Competitor)
        if market_share_min:
            query = query.where(Competitor.market_share >= market_share_min)

        query = query.order_by(desc(Competitor.market_share.nullslast()),
                               Competitor.company_name).limit(per_page).offset(offset)

        # Get items
        result = await session.execute(query)
        items = result.scalars().all()

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_prev=page > 1,
            has_next=page < total_pages
        )

    @staticmethod
    async def get_paginated_task_queue(
        session: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        status: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> PaginatedResult[TaskQueue]:
        """
        Get paginated task queue with metadata.

        Args:
            session: Database session
            page: Page number (1-indexed)
            per_page: Items per page
            status: Optional status filter
            task_type: Optional task type filter

        Returns:
            PaginatedResult containing tasks and pagination metadata
        """
        # Build count query
        count_query = select(func.count(TaskQueue.id))
        if status:
            count_query = count_query.where(TaskQueue.status == status)
        if task_type:
            count_query = count_query.where(TaskQueue.task_type == task_type)

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Calculate pagination
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        offset = (page - 1) * per_page

        # Build data query
        query = select(TaskQueue)
        if status:
            query = query.where(TaskQueue.status == status)
        if task_type:
            query = query.where(TaskQueue.task_type == task_type)

        query = query.order_by(TaskQueue.priority, TaskQueue.scheduled_at).limit(per_page).offset(offset)

        # Get items
        result = await session.execute(query)
        items = result.scalars().all()

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_prev=page > 1,
            has_next=page < total_pages
        )

    # =========================
    # COMPETITOR-PRODUCT RELATIONSHIP OPERATIONS
    # =========================

    @staticmethod
    async def link_competitor_product(
        session: AsyncSession,
        competitor_id: int,
        product_id: int,
        relationship_type: str = "direct_competitor"
    ) -> CompetitorProducts:
        """Create relationship between competitor and product"""
        # Check if relationship already exists
        result = await session.execute(
            select(CompetitorProducts).where(
                and_(
                    CompetitorProducts.competitor_id == competitor_id,
                    CompetitorProducts.product_id == product_id
                )
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing relationship type if different
            if existing.relationship_type != relationship_type:
                existing.relationship_type = relationship_type
                await session.flush()
            return existing

        # Create new relationship
        relationship = CompetitorProducts(
            competitor_id=competitor_id,
            product_id=product_id,
            relationship_type=relationship_type
        )
        session.add(relationship)
        await session.flush()
        logger.info(f"Linked competitor {competitor_id} with product {product_id} as {relationship_type}")
        return relationship

    @staticmethod
    async def get_competitor_products(
        session: AsyncSession,
        competitor_id: int
    ) -> List[Product]:
        """Get all products linked to a competitor"""
        result = await session.execute(
            select(Product)
            .join(CompetitorProducts)
            .where(CompetitorProducts.competitor_id == competitor_id)
            .order_by(Product.product_name)
        )
        return result.scalars().all()

    @staticmethod
    async def get_product_competitors(
        session: AsyncSession,
        product_id: int
    ) -> List[Competitor]:
        """Get all competitors linked to a product"""
        result = await session.execute(
            select(Competitor)
            .join(CompetitorProducts)
            .where(CompetitorProducts.product_id == product_id)
            .order_by(Competitor.company_name)
        )
        return result.scalars().all()

    # =========================
    # INSIGHTS OPERATIONS
    # =========================

    @staticmethod
    async def create_insight(
        session: AsyncSession,
        insight_type: str,
        title: str,
        description: str,
        severity: str = "medium",
        confidence_score: float = 0.75,
        **kwargs
    ) -> Insights:
        """Create a new insight"""
        insight = Insights(
            insight_type=insight_type,
            title=title,
            description=description,
            severity=severity,
            confidence_score=confidence_score,
            product_id=kwargs.get('product_id'),
            competitor_id=kwargs.get('competitor_id'),
            insight_data=kwargs.get('insight_data'),
            action_items=kwargs.get('action_items')
        )
        session.add(insight)
        await session.flush()
        logger.info(f"Created {severity} {insight_type} insight: {title}")
        return insight

    @staticmethod
    async def get_unread_insights(
        session: AsyncSession,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Insights]:
        """Get unread insights, optionally filtered by severity"""
        query = select(Insights).where(Insights.is_read == False)

        if severity:
            query = query.where(Insights.severity == severity)

        query = query.order_by(
            desc(Insights.created_at)
        ).limit(limit)

        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def mark_insight_read(
        session: AsyncSession,
        insight_id: int
    ):
        """Mark an insight as read"""
        stmt = update(Insights).where(Insights.id == insight_id).values(
            is_read=True
        )
        await session.execute(stmt)
        logger.info(f"Marked insight {insight_id} as read")

    @staticmethod
    async def get_critical_insights(
        session: AsyncSession,
        days: int = 7
    ) -> List[Insights]:
        """Get critical insights from the last N days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = await session.execute(
            select(Insights).where(
                and_(
                    Insights.severity == "critical",
                    Insights.created_at >= cutoff_date
                )
            ).order_by(desc(Insights.created_at))
        )
        return result.scalars().all()

    # =========================
    # FULL-TEXT SEARCH OPERATIONS
    # =========================

    @staticmethod
    async def search_products_fts(
        session: AsyncSession,
        search_query: str,
        limit: int = 20
    ) -> List[Product]:
        """Full-text search on products using FTS5"""
        from sqlalchemy import text

        # First get matching product IDs from FTS
        fts_query = text("""
            SELECT rowid
            FROM products_fts
            WHERE products_fts MATCH :query
            ORDER BY rank
            LIMIT :limit
        """)

        result = await session.execute(
            fts_query,
            {"query": search_query, "limit": limit}
        )

        product_ids = [row[0] for row in result]

        if not product_ids:
            logger.info(f"FTS search for '{search_query}' returned 0 products")
            return []

        # Now get the actual Product objects using ORM
        products = await session.execute(
            select(Product).where(Product.id.in_(product_ids))
        )
        products = products.scalars().all()

        logger.info(f"FTS search for '{search_query}' returned {len(products)} products")
        return products

    @staticmethod
    async def search_competitors_fts(
        session: AsyncSession,
        search_query: str,
        limit: int = 20
    ) -> List[Competitor]:
        """Full-text search on competitors using FTS5"""
        from sqlalchemy import text

        # First get matching competitor IDs from FTS
        fts_query = text("""
            SELECT rowid
            FROM competitors_fts
            WHERE competitors_fts MATCH :query
            ORDER BY rank
            LIMIT :limit
        """)

        result = await session.execute(
            fts_query,
            {"query": search_query, "limit": limit}
        )

        competitor_ids = [row[0] for row in result]

        if not competitor_ids:
            logger.info(f"FTS search for '{search_query}' returned 0 competitors")
            return []

        # Now get the actual Competitor objects using ORM
        competitors = await session.execute(
            select(Competitor).where(Competitor.id.in_(competitor_ids))
        )
        competitors = competitors.scalars().all()

        logger.info(f"FTS search for '{search_query}' returned {len(competitors)} competitors")
        return competitors

    @staticmethod
    async def search_news_fts(
        session: AsyncSession,
        search_query: str,
        limit: int = 20
    ) -> List[NewsContent]:
        """Full-text search on news content using FTS5"""
        from sqlalchemy import text

        # First get matching news IDs from FTS
        fts_query = text("""
            SELECT rowid
            FROM news_content_fts
            WHERE news_content_fts MATCH :query
            ORDER BY rank
            LIMIT :limit
        """)

        result = await session.execute(
            fts_query,
            {"query": search_query, "limit": limit}
        )

        news_ids = [row[0] for row in result]

        if not news_ids:
            logger.info(f"FTS search for '{search_query}' returned 0 news items")
            return []

        # Now get the actual NewsContent objects using ORM
        news_items = await session.execute(
            select(NewsContent).where(NewsContent.id.in_(news_ids))
        )
        news_items = news_items.scalars().all()

        logger.info(f"FTS search for '{search_query}' returned {len(news_items)} news items")
        return news_items

    # =========================
    # PERFORMANCE UTILITY FUNCTIONS
    # =========================

    @staticmethod
    async def execute_in_chunks(
        session: AsyncSession,
        items: List[Any],
        operation: Callable,
        chunk_size: int = 100
    ) -> List[Any]:
        """
        Execute any operation in chunks for better performance.

        Args:
            session: Database session
            items: List of items to process
            operation: Async callable to execute on each chunk
            chunk_size: Size of each chunk

        Returns:
            Combined results from all chunks
        """
        if not items:
            return []

        results = []
        total_chunks = (len(items) + chunk_size - 1) // chunk_size

        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]

            # Execute operation on chunk
            if asyncio.iscoroutinefunction(operation):
                chunk_results = await operation(session, chunk)
            else:
                chunk_results = operation(session, chunk)

            if chunk_results:
                if isinstance(chunk_results, list):
                    results.extend(chunk_results)
                else:
                    results.append(chunk_results)

            # Allow other operations between chunks
            await asyncio.sleep(0)

        logger.info(f"Executed operation on {len(items)} items in {total_chunks} chunks")
        return results

    @staticmethod
    async def count_with_filters(
        session: AsyncSession,
        model: Type,
        filters: Dict[str, Any]
    ) -> int:
        """
        Generic count operation with dynamic filters.

        Args:
            session: Database session
            model: SQLAlchemy model class
            filters: Dictionary of field:value filters

        Returns:
            Count of matching records
        """
        query = select(func.count()).select_from(model)

        for key, value in filters.items():
            if hasattr(model, key):
                if value is None:
                    query = query.where(getattr(model, key).is_(None))
                elif isinstance(value, list):
                    query = query.where(getattr(model, key).in_(value))
                elif isinstance(value, str) and '%' in value:
                    query = query.where(getattr(model, key).ilike(value))
                else:
                    query = query.where(getattr(model, key) == value)

        result = await session.execute(query)
        count = result.scalar() or 0
        logger.debug(f"Count for {model.__tablename__} with filters {filters}: {count}")
        return count

    @staticmethod
    async def bulk_delete_with_filters(
        session: AsyncSession,
        model: Type,
        filters: Dict[str, Any],
        chunk_size: int = 100
    ) -> int:
        """
        Bulk delete records matching filters in chunks.

        Args:
            session: Database session
            model: SQLAlchemy model class
            filters: Dictionary of field:value filters
            chunk_size: Number of records to delete at once

        Returns:
            Total number of records deleted
        """
        # First, get IDs of records to delete
        query = select(model.id)

        for key, value in filters.items():
            if hasattr(model, key):
                query = query.where(getattr(model, key) == value)

        result = await session.execute(query)
        ids_to_delete = [row[0] for row in result]

        if not ids_to_delete:
            return 0

        total_deleted = 0

        # Delete in chunks
        for i in range(0, len(ids_to_delete), chunk_size):
            chunk_ids = ids_to_delete[i:i + chunk_size]
            stmt = delete(model).where(model.id.in_(chunk_ids))
            result = await session.execute(stmt)
            total_deleted += result.rowcount
            await session.flush()
            await asyncio.sleep(0)

        logger.info(f"Bulk deleted {total_deleted} records from {model.__tablename__}")
        return total_deleted

    @staticmethod
    async def optimize_query_with_joins(
        session: AsyncSession,
        base_model: Type,
        join_models: List[Type],
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List:
        """
        Optimize queries with eager loading of relationships.

        Args:
            session: Database session
            base_model: Primary model to query
            join_models: List of related models to join
            filters: Optional filters
            limit: Optional result limit

        Returns:
            Query results with relationships loaded
        """
        from sqlalchemy.orm import selectinload

        query = select(base_model)

        # Add joins for eager loading
        for join_model in join_models:
            # Assume relationship names match lowercase model names
            relationship_name = join_model.__tablename__
            if hasattr(base_model, relationship_name):
                query = query.options(selectinload(getattr(base_model, relationship_name)))

        # Apply filters
        if filters:
            for key, value in filters.items():
                if hasattr(base_model, key):
                    query = query.where(getattr(base_model, key) == value)

        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def get_table_statistics(
        session: AsyncSession,
        model: Type
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for a database table.

        Args:
            session: Database session
            model: SQLAlchemy model class

        Returns:
            Dictionary with table statistics
        """
        stats = {
            "table_name": model.__tablename__,
            "total_records": 0,
            "created_today": 0,
            "created_this_week": 0,
            "created_this_month": 0
        }

        # Total count
        total_result = await session.execute(
            select(func.count()).select_from(model)
        )
        stats["total_records"] = total_result.scalar() or 0

        # Date-based counts if model has created_at
        if hasattr(model, 'created_at'):
            now = datetime.utcnow()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)

            # Today's records
            today_result = await session.execute(
                select(func.count()).select_from(model)
                .where(model.created_at >= today)
            )
            stats["created_today"] = today_result.scalar() or 0

            # This week's records
            week_result = await session.execute(
                select(func.count()).select_from(model)
                .where(model.created_at >= week_ago)
            )
            stats["created_this_week"] = week_result.scalar() or 0

            # This month's records
            month_result = await session.execute(
                select(func.count()).select_from(model)
                .where(model.created_at >= month_ago)
            )
            stats["created_this_month"] = month_result.scalar() or 0

        logger.debug(f"Statistics for {model.__tablename__}: {stats}")
        return stats