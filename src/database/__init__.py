"""Database module - Models, manager, operations, and FTS"""

from .manager import DatabaseManager, db_manager, init_db, close_db, get_db
from .operations import DatabaseOperations, PaginatedResult
from .models import (
    Base, Search, SearchResult, Analysis, Product, PriceData, Offer,
    ProductReview, Competitor, MarketTrend, SERPData, SocialSentiment,
    NewsContent, FeatureComparison, Cache, TaskQueue, ScrapingJob,
    RateLimit, PriceHistory, CompetitorTracking, CompetitorProducts, Insights
)
from .fts import setup_fts5

__all__ = [
    # Manager
    "DatabaseManager", "db_manager", "init_db", "close_db", "get_db",

    # Operations
    "DatabaseOperations", "PaginatedResult",

    # Models (23 total)
    "Base", "Search", "SearchResult", "Analysis", "Product", "PriceData",
    "Offer", "ProductReview", "Competitor", "MarketTrend", "SERPData",
    "SocialSentiment", "NewsContent", "FeatureComparison", "Cache",
    "TaskQueue", "ScrapingJob", "RateLimit", "PriceHistory",
    "CompetitorTracking", "CompetitorProducts", "Insights",

    # FTS
    "setup_fts5"
]
