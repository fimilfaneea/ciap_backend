"""
Base Scraper for CIAP Web Scraping System
Provides abstract base class and common utilities for all scrapers
"""

import asyncio
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from urllib.parse import urljoin, urlparse, quote_plus

import httpx
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from sqlalchemy import select

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

    Provides common functionality:
    - HTTP request handling with retry logic
    - Rate limiting enforcement
    - User agent rotation
    - HTML parsing
    - Result validation and cleaning
    - Statistics tracking
    """

    def __init__(self):
        """Initialize base scraper"""
        self.name = self.__class__.__name__
        self.user_agent = UserAgent()

        # HTTP client configuration
        self.timeout = settings.SCRAPER_TIMEOUT
        self.retry_count = settings.SCRAPER_RETRY_COUNT
        self.rate_limit_delay = settings.SCRAPER_RATE_LIMIT_DELAY

        # Headers pool
        self.headers_pool = self._create_headers_pool()

        # Statistics
        self.stats = {
            "requests_made": 0,
            "requests_failed": 0,
            "results_scraped": 0
        }

    def _create_headers_pool(self) -> List[Dict[str, str]]:
        """
        Create pool of headers for rotation

        Returns:
            List of header dictionaries
        """
        user_agents = settings.SCRAPER_USER_AGENTS or [
            self.user_agent.chrome,
            self.user_agent.firefox,
            self.user_agent.safari
        ]

        headers_list = []
        for ua in user_agents:
            headers_list.append({
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "max-age=0"
            })

        return headers_list

    def get_random_headers(self) -> Dict[str, str]:
        """
        Get random headers from pool

        Returns:
            Random header dictionary
        """
        return random.choice(self.headers_pool).copy()

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

    async def make_request(
        self,
        url: str,
        method: str = "GET",
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic

        Implements exponential backoff retry strategy and rate limiting.

        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional request parameters

        Returns:
            HTTP response

        Raises:
            ScraperException: On request failure after all retries
            BlockedException: If blocked by target (403 status)
            RateLimitException: If rate limited by target (429 status)
        """
        # Check rate limit
        await self.check_rate_limit()

        # Get random headers
        headers = kwargs.pop("headers", {})
        default_headers = self.get_random_headers()
        default_headers.update(headers)

        last_exception = None

        for attempt in range(self.retry_count):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True,
                    verify=False  # Skip SSL verification
                ) as client:
                    self.stats["requests_made"] += 1

                    response = await client.request(
                        method,
                        url,
                        headers=default_headers,
                        **kwargs
                    )

                    # Check for blocking
                    if response.status_code == 403:
                        raise BlockedException(
                            f"Blocked by {urlparse(url).netloc}"
                        )

                    if response.status_code == 429:
                        raise RateLimitException(
                            f"Rate limited by {urlparse(url).netloc}"
                        )

                    response.raise_for_status()

                    return response

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"{self.name}: Attempt {attempt + 1} failed, "
                    f"waiting {wait_time}s - {e}"
                )
                await asyncio.sleep(wait_time)

            except (BlockedException, RateLimitException):
                # Don't retry on blocking or rate limiting
                self.stats["requests_failed"] += 1
                raise

            except Exception as e:
                self.stats["requests_failed"] += 1
                raise ScraperException(f"Request failed: {e}")

        self.stats["requests_failed"] += 1
        raise ScraperException(
            f"Failed after {self.retry_count} attempts: {last_exception}"
        )

    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML content

        Args:
            html: HTML string

        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html, "lxml")

    @abstractmethod
    async def scrape(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape search results

        This method must be implemented by subclasses.

        Args:
            query: Search query
            max_results: Maximum results to return
            **kwargs: Additional parameters

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def parse_results(
        self,
        soup: BeautifulSoup,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Parse search results from HTML

        This method must be implemented by subclasses.

        Args:
            soup: BeautifulSoup object
            max_results: Maximum results to extract

        Returns:
            List of parsed results
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
                (self.stats["requests_made"] - self.stats["requests_failed"])
                / self.stats["requests_made"] * 100
                if self.stats["requests_made"] > 0 else 0
            )
        }
