"""
Base Scraper for CIAP Web Scraping System
Provides abstract base class and common utilities for all scrapers

Updated to use Scrapy framework instead of httpx/BeautifulSoup
"""

import asyncio
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
import logging
from urllib.parse import urljoin, urlparse

from sqlalchemy import select
from scrapy import Spider

from ..config import settings
from ..database import db_manager, RateLimit

logger = logging.getLogger(__name__)


class ScraperException(Exception):
    """Base exception for scrapers"""
    pass


class RateLimitException(ScraperException):
    """Raised when rate limit is exceeded"""
    pass


class BlockedException(ScraperException):
    """Raised when scraper is blocked"""
    pass


class BaseScraper(ABC):
    """
    Abstract base class for all scrapers

    Updated to use Scrapy backend with Playwright for JavaScript rendering.

    Provides common functionality:
    - Scrapy spider execution via async wrapper
    - Rate limiting enforcement
    - Result validation and cleaning
    - Statistics tracking
    - Caching integration
    """

    def __init__(self):
        """Initialize base scraper"""
        self.name = self.__class__.__name__

        # Scraper configuration
        self.timeout = settings.SCRAPER_TIMEOUT
        self.rate_limit_delay = settings.SCRAPER_RATE_LIMIT_DELAY

        # Statistics
        self.stats = {
            "spiders_run": 0,
            "spiders_failed": 0,
            "results_scraped": 0
        }

    @abstractmethod
    def get_spider_class(self) -> Type[Spider]:
        """
        Get Scrapy spider class for this scraper

        Subclasses must implement this to return their Spider class.

        Returns:
            Scrapy Spider class

        Example:
            def get_spider_class(self):
                from .spiders import GoogleSpider
                return GoogleSpider
        """
        pass

    async def check_rate_limit(self) -> bool:
        """
        Check and enforce rate limiting

        Uses database RateLimit model to track requests and enforce delays.

        Returns:
            True if request can proceed, False otherwise
        """
        async with db_manager.get_session() as session:
            # Get last request time
            result = await session.execute(
                select(RateLimit)
                .where(RateLimit.scraper_name == self.name)
                .order_by(RateLimit.last_request_at.desc())
                .limit(1)
            )
            last_limit = result.scalar_one_or_none()

            if last_limit:
                time_since_last = (
                    datetime.utcnow() - last_limit.last_request_at
                ).total_seconds()

                if time_since_last < self.rate_limit_delay:
                    # Need to wait
                    wait_time = self.rate_limit_delay - time_since_last
                    logger.debug(
                        f"{self.name}: Rate limit wait {wait_time:.2f}s"
                    )
                    await asyncio.sleep(wait_time)

            # Record new request
            new_limit = RateLimit(
                scraper_name=self.name,
                last_request_at=datetime.utcnow(),
                request_count=1,
                reset_at=datetime.utcnow() + timedelta(minutes=1)
            )
            session.add(new_limit)
            await session.commit()

            return True

    async def _run_scrapy_spider(self, **spider_kwargs) -> List[Dict[str, Any]]:
        """
        Run Scrapy spider and return results

        Args:
            **spider_kwargs: Arguments to pass to spider

        Returns:
            List of scraped results

        Raises:
            ScraperException: If spider fails
        """
        # Check rate limit before running spider
        await self.check_rate_limit()

        try:
            # Import here to avoid circular imports
            from .scrapy_runner import run_spider

            # Get spider class
            spider_class = self.get_spider_class()

            # Run spider with timeout
            results = await run_spider(
                spider_class,
                timeout=self.timeout * 3,  # Triple timeout for full scraping
                **spider_kwargs
            )

            self.stats["spiders_run"] += 1
            self.stats["results_scraped"] += len(results)

            logger.info(
                f"{self.name}: Spider completed with {len(results)} results"
            )

            return results

        except Exception as e:
            self.stats["spiders_failed"] += 1
            logger.error(f"{self.name}: Spider failed - {e}")
            raise ScraperException(f"Scrapy spider failed: {e}")

    @abstractmethod
    async def scrape(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape search results using Scrapy spider

        Subclasses should implement this by calling _run_scrapy_spider()
        with appropriate spider arguments.

        Args:
            query: Search query
            max_results: Maximum results to return
            **kwargs: Additional parameters (lang, region, date_range, etc.)

        Returns:
            List of search results

        Example:
            async def scrape(self, query, max_results=100, **kwargs):
                results = await self._run_scrapy_spider(
                    query=query,
                    max_results=max_results,
                    **kwargs
                )
                return await self.validate_results(results)
        """
        pass

    def normalize_url(self, url: str, base_url: str = None) -> str:
        """
        Normalize and validate URL

        Handles relative URLs and removes tracking parameters.

        Args:
            url: URL to normalize
            base_url: Base URL for relative URLs

        Returns:
            Normalized URL
        """
        if not url:
            return ""

        # Handle relative URLs
        if base_url and not url.startswith(("http://", "https://")):
            url = urljoin(base_url, url)

        # Remove tracking parameters
        if "?" in url:
            parsed = urlparse(url)
            # Keep only essential parameters
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        return url

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Removes extra whitespace, special characters, and limits length.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove special characters
        text = text.replace("\n", " ").replace("\r", "").replace("\t", " ")

        # Limit length
        if len(text) > 500:
            text = text[:497] + "..."

        return text.strip()

    async def validate_results(
        self,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Validate and filter results

        Ensures all results have required fields and cleans data.

        Args:
            results: Raw results

        Returns:
            Validated results
        """
        validated = []

        for result in results:
            # Required fields
            if not result.get("url"):
                continue

            # Set defaults
            result.setdefault("title", "No title")
            result.setdefault("snippet", "")
            result.setdefault("position", len(validated) + 1)
            result.setdefault("source", self.name.lower())
            result.setdefault("scraped_at", datetime.utcnow())

            # Clean fields
            result["title"] = self.clean_text(result["title"])
            result["snippet"] = self.clean_text(result["snippet"])
            result["url"] = self.normalize_url(result["url"])

            validated.append(result)

        return validated

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scraper statistics

        Returns:
            Dictionary with scraper statistics
        """
        return {
            "scraper": self.name,
            **self.stats,
            "success_rate": (
                (self.stats["spiders_run"] - self.stats["spiders_failed"])
                / self.stats["spiders_run"] * 100
                if self.stats["spiders_run"] > 0 else 0
            )
        }
