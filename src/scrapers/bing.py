"""
Bing Search Results Scraper
Uses Scrapy BingSpider with Playwright for robust scraping
"""

import logging
from typing import Dict, List, Any, Type
from scrapy import Spider

from .base import BaseScraper
from ..config import settings
from ..cache import cache

logger = logging.getLogger(__name__)


class BingScraper(BaseScraper):
    """
    Bing search results scraper using Scrapy

    Features:
    - Playwright integration for JavaScript rendering
    - Robust selector strategies with fallbacks
    - Pagination support
    - Cache integration (1 hour TTL)
    - Metadata extraction (dates, deep links, display URLs)
    """

    def __init__(self):
        """Initialize Bing scraper"""
        super().__init__()
        self.base_url = settings.BING_SEARCH_URL

    def get_spider_class(self) -> Type[Spider]:
        """Return BingSpider class"""
        from .spiders import BingSpider
        return BingSpider

    async def scrape(
        self,
        query: str,
        max_results: int = 50,
        lang: str = "en",
        region: str = "us",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape Bing search results using Scrapy

        Args:
            query: Search query
            max_results: Maximum results to return (default: 50)
            lang: Language code (default: "en")
            region: Region code (default: "us")
            **kwargs: Additional parameters

        Returns:
            List of search results with title, snippet, url, position, metadata
        """
        # Check cache first
        cache_key = cache.make_key(
            "bing_search",
            query=query,
            max_results=max_results,
            lang=lang,
            region=region
        )
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Using cached Bing results for '{query}'")
            return cached

        # Run Scrapy spider
        results = await self._run_scrapy_spider(
            query=query,
            max_results=max_results,
            lang=lang,
            region=region
        )

        # Validate and clean results (already done by pipelines, but ensure consistency)
        results = await self.validate_results(results)

        # Cache results (1 hour TTL)
        await cache.set(cache_key, results, ttl=3600)

        logger.info(f"Bing scraping completed: {len(results)} total results for '{query}'")
        return results
