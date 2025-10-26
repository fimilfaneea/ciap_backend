"""
Google Search Results Scraper
Uses Scrapy GoogleSpider with Playwright for robust scraping
"""

import logging
from typing import Dict, List, Any, Optional, Type
from scrapy import Spider

from .base import BaseScraper
from ..config import settings
from ..cache import cache

logger = logging.getLogger(__name__)


class GoogleScraper(BaseScraper):
    """
    Google search results scraper using Scrapy

    Features:
    - Playwright integration for JavaScript rendering
    - Robust selector strategies with fallbacks
    - CAPTCHA detection
    - Pagination support
    - Cache integration (1 hour TTL)
    - Date range filtering
    """

    def __init__(self):
        """Initialize Google scraper"""
        super().__init__()
        self.base_url = settings.GOOGLE_SEARCH_URL

    def get_spider_class(self) -> Type[Spider]:
        """Return GoogleSpider class"""
        from .spiders import GoogleSpider
        return GoogleSpider

    async def scrape(
        self,
        query: str,
        max_results: int = 100,
        lang: str = "en",
        region: str = "us",
        date_range: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape Google search results using Scrapy

        Args:
            query: Search query
            max_results: Maximum results to return (default: 100)
            lang: Language code (default: "en")
            region: Region code (default: "us")
            date_range: Date filter - "d" (day), "w" (week), "m" (month), "y" (year)
            **kwargs: Additional parameters

        Returns:
            List of search results with title, snippet, url, position, metadata
        """
        # Check cache first
        cache_key = cache.make_key(
            "google_search",
            query=query,
            max_results=max_results,
            lang=lang,
            region=region,
            date_range=date_range or "none"
        )
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Using cached Google results for '{query}'")
            return cached

        # Run Scrapy spider
        results = await self._run_scrapy_spider(
            query=query,
            max_results=max_results,
            lang=lang,
            region=region,
            date_range=date_range
        )

        # Validate and clean results (already done by pipelines, but ensure consistency)
        results = await self.validate_results(results)

        # Cache results (1 hour TTL)
        await cache.set(cache_key, results, ttl=3600)

        logger.info(f"Google scraping completed: {len(results)} total results for '{query}'")
        return results

