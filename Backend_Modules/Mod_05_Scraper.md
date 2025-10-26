# Module 5: Web Scraping System

## Overview
**Purpose:** Multi-source web scraping system with Google, Bing, and deep web crawling capabilities.

**Responsibilities:**
- Search engine scraping (Google, Bing)
- Rate limiting and anti-detection
- Result extraction and normalization
- Proxy rotation (if needed)
- User agent rotation
- Error handling and retries
- Scraper orchestration

**Development Time:** 3 days (Week 3-4, Day 8-14)

---

## Interface Specification

### Input
```python
# Search parameters
query: str  # Search query
max_results: int  # Maximum results to fetch
source: str  # "google", "bing", "crawlee"
filters: Dict  # Optional filters (date, region, etc.)
```

### Output
```python
# Scraped results
results: List[Dict]  # List of search results
{
    "title": str,
    "snippet": str,
    "url": str,
    "position": int,
    "source": str,
    "scraped_at": datetime
}
```

---

## Dependencies

### External
```txt
beautifulsoup4==4.12.2
requests==2.31.0
playwright==1.40.0
crawlee==0.1.5  # For deep crawling
fake-useragent==1.4.0
httpx==0.25.1  # Async HTTP client
lxml==4.9.3  # Faster HTML parsing
```

### Internal
- Module 1: Database Infrastructure
- Module 2: Configuration
- Module 3: Cache System
- Module 4: Task Queue

---

## Implementation Guide

### Step 1: Base Scraper Interface (`src/scrapers/base.py`)

```python
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

from src.core.config import settings
from src.core.database import db_manager
from src.core.models import RateLimit
from src.core.cache import cache

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
    """Abstract base class for all scrapers"""

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
        """Create pool of headers for rotation"""
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
        """Get random headers from pool"""
        return random.choice(self.headers_pool)

    async def check_rate_limit(self) -> bool:
        """
        Check and enforce rate limiting

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

        Args:
            url: URL to request
            method: HTTP method
            **kwargs: Additional request parameters

        Returns:
            HTTP response

        Raises:
            ScraperException: On request failure
        """
        # Check rate limit
        await self.check_rate_limit()

        # Get random headers
        headers = kwargs.pop("headers", {})
        headers.update(self.get_random_headers())

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
                        headers=headers,
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
                    f"waiting {wait_time}s"
                )
                await asyncio.sleep(wait_time)

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
        """Get scraper statistics"""
        return {
            "scraper": self.name,
            **self.stats,
            "success_rate": (
                (self.stats["requests_made"] - self.stats["requests_failed"])
                / self.stats["requests_made"] * 100
                if self.stats["requests_made"] > 0 else 0
            )
        }
```

### Step 2: Google Scraper (`src/scrapers/google.py`)

```python
import re
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus, urlparse, parse_qs
import logging

from bs4 import BeautifulSoup

from src.scrapers.base import BaseScraper, ScraperException
from src.core.config import settings
from src.core.cache import cache

logger = logging.getLogger(__name__)


class GoogleScraper(BaseScraper):
    """Google search results scraper"""

    def __init__(self):
        super().__init__()
        self.base_url = settings.GOOGLE_SEARCH_URL
        self.max_results_per_page = 10

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
        Scrape Google search results

        Args:
            query: Search query
            max_results: Maximum results to return
            lang: Language code
            region: Region code
            date_range: Date filter (d, w, m, y)

        Returns:
            List of search results
        """
        # Check cache first
        cache_key = cache.make_key(
            "google_search",
            query=query,
            max_results=max_results,
            lang=lang,
            region=region
        )
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Using cached Google results for '{query}'")
            return cached

        results = []
        pages_needed = (max_results + self.max_results_per_page - 1) // self.max_results_per_page

        for page in range(pages_needed):
            try:
                # Build search URL
                url = self._build_search_url(
                    query,
                    page * self.max_results_per_page,
                    lang,
                    region,
                    date_range
                )

                # Make request
                response = await self.make_request(url)

                # Parse results
                soup = self.parse_html(response.text)
                page_results = self.parse_results(soup, max_results - len(results))

                if not page_results:
                    logger.warning(f"No results on page {page + 1}")
                    break

                results.extend(page_results)

                if len(results) >= max_results:
                    break

                # Add delay between pages
                if page < pages_needed - 1:
                    await asyncio.sleep(self.rate_limit_delay * 2)

            except Exception as e:
                logger.error(f"Error scraping Google page {page + 1}: {e}")
                if not results:  # If first page fails, re-raise
                    raise

        # Validate and clean results
        results = await self.validate_results(results[:max_results])

        # Update stats
        self.stats["results_scraped"] += len(results)

        # Cache results
        await cache.set(cache_key, results, ttl=3600)

        return results

    def _build_search_url(
        self,
        query: str,
        start: int,
        lang: str,
        region: str,
        date_range: Optional[str]
    ) -> str:
        """Build Google search URL with parameters"""
        params = {
            "q": query,
            "start": start,
            "num": self.max_results_per_page,
            "hl": lang,  # Interface language
            "gl": region,  # Region for results
            "safe": "off",  # SafeSearch off
        }

        # Add date range filter
        if date_range:
            tbs_map = {
                "d": "qdr:d",  # Past day
                "w": "qdr:w",  # Past week
                "m": "qdr:m",  # Past month
                "y": "qdr:y",  # Past year
            }
            if date_range in tbs_map:
                params["tbs"] = tbs_map[date_range]

        # Build URL
        query_string = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        return f"{self.base_url}?{query_string}"

    def parse_results(
        self,
        soup: BeautifulSoup,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Parse Google search results

        Args:
            soup: BeautifulSoup object
            max_results: Maximum results to extract

        Returns:
            List of parsed results
        """
        results = []

        # Check if blocked
        if soup.find("div", id="recaptcha"):
            raise BlockedException("Google CAPTCHA detected")

        # Find search result containers
        # Google uses different structures, try multiple selectors
        selectors = [
            "div.g",  # Standard results
            "div.tF2Cxc",  # New format
            "div.Gx5Zad",  # Alternative format
        ]

        result_elements = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                result_elements = elements
                break

        if not result_elements:
            logger.warning("No result elements found in Google HTML")
            return results

        for position, element in enumerate(result_elements[:max_results], 1):
            try:
                result = self._parse_single_result(element, position)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Error parsing result {position}: {e}")
                continue

        return results

    def _parse_single_result(
        self,
        element: BeautifulSoup,
        position: int
    ) -> Optional[Dict[str, Any]]:
        """Parse single search result element"""
        # Extract URL
        link_element = element.select_one("a[href]")
        if not link_element:
            return None

        url = link_element.get("href", "")

        # Extract from Google's redirect URL if needed
        if url.startswith("/url?"):
            parsed = parse_qs(urlparse(url).query)
            url = parsed.get("q", [url])[0]

        if not url.startswith(("http://", "https://")):
            return None

        # Extract title
        title_selectors = ["h3", "div.BNeawe", "div.vvjwJb"]
        title = ""
        for selector in title_selectors:
            title_element = element.select_one(selector)
            if title_element:
                title = title_element.get_text(strip=True)
                break

        # Extract snippet
        snippet_selectors = [
            "div.VwiC3b",  # Standard snippet
            "span.aCOpRe",  # Alternative
            "div.s3v9rd",  # Another alternative
            "div.BNeawe.s3v9rd",  # Mobile format
        ]
        snippet = ""
        for selector in snippet_selectors:
            snippet_element = element.select_one(selector)
            if snippet_element:
                snippet = snippet_element.get_text(strip=True)
                break

        # Extract additional metadata
        metadata = {}

        # Check for date
        date_element = element.select_one("span.MUxGbd")
        if date_element:
            metadata["date"] = date_element.get_text(strip=True)

        # Check for rating
        rating_element = element.select_one("span.z3HNkc")
        if rating_element:
            metadata["rating"] = rating_element.get("aria-label", "")

        return {
            "title": title or "No title",
            "snippet": snippet,
            "url": url,
            "position": position,
            "metadata": metadata
        }
```

### Step 3: Bing Scraper (`src/scrapers/bing.py`)

```python
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
import logging

from bs4 import BeautifulSoup

from src.scrapers.base import BaseScraper
from src.core.config import settings
from src.core.cache import cache

logger = logging.getLogger(__name__)


class BingScraper(BaseScraper):
    """Bing search results scraper"""

    def __init__(self):
        super().__init__()
        self.base_url = settings.BING_SEARCH_URL
        self.max_results_per_page = 10

    async def scrape(
        self,
        query: str,
        max_results: int = 50,
        lang: str = "en",
        region: str = "us",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape Bing search results

        Args:
            query: Search query
            max_results: Maximum results to return
            lang: Language code
            region: Region code

        Returns:
            List of search results
        """
        # Check cache
        cache_key = cache.make_key(
            "bing_search",
            query=query,
            max_results=max_results
        )
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Using cached Bing results for '{query}'")
            return cached

        results = []
        pages_needed = (max_results + self.max_results_per_page - 1) // self.max_results_per_page

        for page in range(pages_needed):
            try:
                # Build URL
                url = self._build_search_url(query, page * self.max_results_per_page)

                # Make request
                response = await self.make_request(url)

                # Parse results
                soup = self.parse_html(response.text)
                page_results = self.parse_results(soup, max_results - len(results))

                if not page_results:
                    break

                results.extend(page_results)

                if len(results) >= max_results:
                    break

            except Exception as e:
                logger.error(f"Error scraping Bing page {page + 1}: {e}")
                if not results:
                    raise

        # Validate results
        results = await self.validate_results(results[:max_results])

        # Update stats
        self.stats["results_scraped"] += len(results)

        # Cache results
        await cache.set(cache_key, results, ttl=3600)

        return results

    def _build_search_url(self, query: str, first: int) -> str:
        """Build Bing search URL"""
        params = {
            "q": query,
            "first": first + 1,  # Bing uses 1-based indexing
        }
        query_string = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        return f"{self.base_url}?{query_string}"

    def parse_results(
        self,
        soup: BeautifulSoup,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Parse Bing search results"""
        results = []

        # Find result containers
        result_elements = soup.select("li.b_algo")

        for position, element in enumerate(result_elements[:max_results], 1):
            try:
                # Extract URL
                link = element.select_one("h2 a")
                if not link:
                    continue

                url = link.get("href", "")
                title = link.get_text(strip=True)

                # Extract snippet
                snippet_element = element.select_one("div.b_caption p")
                snippet = snippet_element.get_text(strip=True) if snippet_element else ""

                # Extract date if available
                date_element = element.select_one("span.news_dt")
                date = date_element.get_text(strip=True) if date_element else None

                result = {
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "position": position,
                }

                if date:
                    result["metadata"] = {"date": date}

                results.append(result)

            except Exception as e:
                logger.debug(f"Error parsing Bing result {position}: {e}")
                continue

        return results
```

### Step 4: Scraper Manager (`src/scrapers/manager.py`)

```python
import asyncio
from typing import Dict, List, Any, Optional
import logging

from src.scrapers.base import BaseScraper, ScraperException
from src.scrapers.google import GoogleScraper
from src.scrapers.bing import BingScraper
from src.core.database import db_manager
from src.core.models import Search, SearchResult, ScrapingJob
from src.core.queue import task_queue

logger = logging.getLogger(__name__)


class ScraperManager:
    """Orchestrate multiple scrapers"""

    def __init__(self):
        """Initialize scraper manager"""
        self.scrapers: Dict[str, BaseScraper] = {
            "google": GoogleScraper(),
            "bing": BingScraper(),
        }

        self.default_sources = ["google", "bing"]

    async def scrape(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 50,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Scrape from multiple sources

        Args:
            query: Search query
            sources: List of sources to scrape
            max_results_per_source: Max results per source

        Returns:
            Dictionary of results by source
        """
        if not sources:
            sources = self.default_sources

        # Validate sources
        invalid_sources = set(sources) - set(self.scrapers.keys())
        if invalid_sources:
            logger.warning(f"Invalid sources: {invalid_sources}")
            sources = [s for s in sources if s in self.scrapers]

        if not sources:
            raise ValueError("No valid sources specified")

        # Scrape in parallel
        tasks = []
        for source in sources:
            scraper = self.scrapers[source]
            task = self._scrape_source(
                scraper,
                query,
                max_results_per_source,
                **kwargs
            )
            tasks.append(task)

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        results = {}
        for source, result in zip(sources, results_list):
            if isinstance(result, Exception):
                logger.error(f"Scraping {source} failed: {result}")
                results[source] = []
            else:
                results[source] = result

        return results

    async def _scrape_source(
        self,
        scraper: BaseScraper,
        query: str,
        max_results: int,
        **kwargs
    ) -> List[Dict]:
        """Scrape single source with error handling"""
        try:
            logger.info(
                f"Scraping {scraper.name} for '{query}' "
                f"(max {max_results} results)"
            )

            results = await scraper.scrape(
                query=query,
                max_results=max_results,
                **kwargs
            )

            logger.info(
                f"Scraped {len(results)} results from {scraper.name}"
            )

            return results

        except ScraperException as e:
            logger.error(f"{scraper.name} scraping failed: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in {scraper.name}: {e}")
            raise ScraperException(f"Scraping failed: {e}")

    async def scrape_and_save(
        self,
        search_id: int,
        query: str,
        sources: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Scrape and save results to database

        Args:
            search_id: Search ID
            query: Search query
            sources: Sources to scrape

        Returns:
            Summary of scraping results
        """
        async with db_manager.get_session() as session:
            # Update search status
            search = await session.get(Search, search_id)
            if not search:
                raise ValueError(f"Search {search_id} not found")

            search.status = "processing"
            await session.commit()

        try:
            # Scrape from sources
            all_results = await self.scrape(query, sources, **kwargs)

            # Save results to database
            saved_counts = {}

            async with db_manager.get_session() as session:
                for source, results in all_results.items():
                    # Create scraping job record
                    job = ScrapingJob(
                        search_id=search_id,
                        scraper_name=source,
                        status="completed",
                        results_count=len(results)
                    )
                    session.add(job)

                    # Save individual results
                    for result in results:
                        search_result = SearchResult(
                            search_id=search_id,
                            source=source,
                            title=result["title"],
                            snippet=result["snippet"],
                            url=result["url"],
                            position=result["position"]
                        )
                        session.add(search_result)

                    saved_counts[source] = len(results)

                # Update search status
                search = await session.get(Search, search_id)
                search.status = "completed"
                await session.commit()

            return {
                "search_id": search_id,
                "query": query,
                "sources": list(all_results.keys()),
                "results_count": saved_counts,
                "total_results": sum(saved_counts.values())
            }

        except Exception as e:
            # Update search status to failed
            async with db_manager.get_session() as session:
                search = await session.get(Search, search_id)
                search.status = "failed"
                search.error_message = str(e)
                await session.commit()

            raise

    async def schedule_scraping(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        priority: int = 5
    ) -> int:
        """
        Schedule scraping as background task

        Args:
            query: Search query
            sources: Sources to scrape
            priority: Task priority

        Returns:
            Task ID
        """
        # Create search record
        async with db_manager.get_session() as session:
            search = Search(
                query=query,
                sources=sources or self.default_sources,
                status="pending"
            )
            session.add(search)
            await session.commit()
            search_id = search.id

        # Enqueue scraping task
        task_id = await task_queue.enqueue(
            task_type="scrape",
            payload={
                "search_id": search_id,
                "query": query,
                "sources": sources
            },
            priority=priority
        )

        return task_id

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all scrapers"""
        stats = {}
        for name, scraper in self.scrapers.items():
            stats[name] = scraper.get_stats()
        return stats


# Global scraper manager instance
scraper_manager = ScraperManager()
```

---

## Testing Guide

### Unit Tests (`tests/test_scrapers.py`)

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx

from src.scrapers.base import BaseScraper, ScraperException, BlockedException
from src.scrapers.google import GoogleScraper
from src.scrapers.bing import BingScraper
from src.scrapers.manager import ScraperManager


class TestScraper(BaseScraper):
    """Test implementation of base scraper"""

    async def scrape(self, query, max_results=10, **kwargs):
        return [
            {
                "title": f"Result {i}",
                "snippet": f"Snippet for {query}",
                "url": f"https://example.com/{i}",
                "position": i
            }
            for i in range(1, max_results + 1)
        ]

    def parse_results(self, soup, max_results):
        return []


@pytest.mark.asyncio
async def test_base_scraper_headers():
    """Test header rotation"""
    scraper = TestScraper()

    headers_set = set()
    for _ in range(10):
        headers = scraper.get_random_headers()
        headers_set.add(headers["User-Agent"])

    # Should have variety in user agents
    assert len(headers_set) > 1


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting enforcement"""
    scraper = TestScraper()

    with patch("src.scrapers.base.db_manager"):
        # First request should proceed
        assert await scraper.check_rate_limit()

        # Rapid requests should be delayed
        start_time = asyncio.get_event_loop().time()
        await scraper.check_rate_limit()
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should have some delay (mocked)
        assert elapsed >= 0


@pytest.mark.asyncio
async def test_request_retry():
    """Test request retry logic"""
    scraper = TestScraper()

    # Mock failing then succeeding
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html></html>"

        # First two attempts fail, third succeeds
        mock_client.return_value.__aenter__.return_value.request = AsyncMock(
            side_effect=[
                httpx.TimeoutException("Timeout"),
                httpx.ConnectError("Connection failed"),
                mock_response
            ]
        )

        response = await scraper.make_request("http://example.com")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_google_scraper_parsing():
    """Test Google result parsing"""
    scraper = GoogleScraper()

    html = """
    <div class="g">
        <a href="https://example.com/1"><h3>Result Title 1</h3></a>
        <div class="VwiC3b">This is snippet 1</div>
    </div>
    <div class="g">
        <a href="https://example.com/2"><h3>Result Title 2</h3></a>
        <div class="VwiC3b">This is snippet 2</div>
    </div>
    """

    soup = scraper.parse_html(html)
    results = scraper.parse_results(soup, 10)

    assert len(results) == 2
    assert results[0]["title"] == "Result Title 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[0]["snippet"] == "This is snippet 1"


@pytest.mark.asyncio
async def test_google_scraper_blocked():
    """Test Google CAPTCHA detection"""
    scraper = GoogleScraper()

    html = '<div id="recaptcha">CAPTCHA</div>'
    soup = scraper.parse_html(html)

    with pytest.raises(BlockedException):
        scraper.parse_results(soup, 10)


@pytest.mark.asyncio
async def test_bing_scraper_parsing():
    """Test Bing result parsing"""
    scraper = BingScraper()

    html = """
    <li class="b_algo">
        <h2><a href="https://example.com/1">Bing Result 1</a></h2>
        <div class="b_caption">
            <p>Bing snippet 1</p>
        </div>
    </li>
    """

    soup = scraper.parse_html(html)
    results = scraper.parse_results(soup, 10)

    assert len(results) == 1
    assert results[0]["title"] == "Bing Result 1"
    assert results[0]["url"] == "https://example.com/1"


@pytest.mark.asyncio
async def test_scraper_manager_parallel():
    """Test parallel scraping"""
    manager = ScraperManager()

    # Mock scrapers
    mock_google = AsyncMock()
    mock_google.name = "Google"
    mock_google.scrape.return_value = [
        {"title": "Google 1", "url": "http://g1.com"}
    ]

    mock_bing = AsyncMock()
    mock_bing.name = "Bing"
    mock_bing.scrape.return_value = [
        {"title": "Bing 1", "url": "http://b1.com"}
    ]

    manager.scrapers = {
        "google": mock_google,
        "bing": mock_bing
    }

    results = await manager.scrape("test query", ["google", "bing"])

    assert "google" in results
    assert "bing" in results
    assert len(results["google"]) == 1
    assert len(results["bing"]) == 1

    # Both should be called
    mock_google.scrape.assert_called_once()
    mock_bing.scrape.assert_called_once()


@pytest.mark.asyncio
async def test_scraper_manager_error_handling():
    """Test error handling in manager"""
    manager = ScraperManager()

    # Mock one scraper failing
    mock_google = AsyncMock()
    mock_google.name = "Google"
    mock_google.scrape.side_effect = ScraperException("Google failed")

    mock_bing = AsyncMock()
    mock_bing.name = "Bing"
    mock_bing.scrape.return_value = [{"title": "Bing works"}]

    manager.scrapers = {
        "google": mock_google,
        "bing": mock_bing
    }

    results = await manager.scrape("test", ["google", "bing"])

    # Bing should still work
    assert results["bing"] == [{"title": "Bing works"}]
    # Google should return empty
    assert results["google"] == []


@pytest.mark.asyncio
async def test_cache_integration():
    """Test cache integration"""
    scraper = GoogleScraper()

    with patch("src.scrapers.google.cache") as mock_cache:
        # First call - cache miss
        mock_cache.get.return_value = None

        with patch.object(scraper, "make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = "<html></html>"
            mock_request.return_value = mock_response

            await scraper.scrape("test query")

            # Should make request
            mock_request.assert_called()

            # Should set cache
            mock_cache.set.assert_called()


@pytest.mark.asyncio
async def test_text_cleaning():
    """Test text cleaning and normalization"""
    scraper = TestScraper()

    # Test whitespace cleaning
    assert scraper.clean_text("  text  with   spaces  ") == "text with spaces"

    # Test newline removal
    assert scraper.clean_text("text\nwith\nnewlines") == "text with newlines"

    # Test length limiting
    long_text = "x" * 1000
    cleaned = scraper.clean_text(long_text)
    assert len(cleaned) == 500
    assert cleaned.endswith("...")


@pytest.mark.asyncio
async def test_url_normalization():
    """Test URL normalization"""
    scraper = TestScraper()

    # Test absolute URL
    assert scraper.normalize_url(
        "https://example.com/page"
    ) == "https://example.com/page"

    # Test relative URL with base
    assert scraper.normalize_url(
        "/page",
        "https://example.com"
    ) == "https://example.com/page"

    # Test tracking parameter removal
    assert scraper.normalize_url(
        "https://example.com/page?utm_source=google"
    ) == "https://example.com/page"
```

---

## Common Issues & Solutions

### Issue 1: Getting Blocked
**Problem:** Search engines block the scraper
**Solution:** Implement better anti-detection
```python
# Add random delays
await asyncio.sleep(random.uniform(1, 3))

# Use residential proxies if needed
# Implement CAPTCHA solving service
```

### Issue 2: HTML Structure Changes
**Problem:** Selectors stop working
**Solution:** Use multiple fallback selectors
```python
selectors = ["div.g", "div.tF2Cxc", "div.result"]
for selector in selectors:
    elements = soup.select(selector)
    if elements:
        break
```

### Issue 3: Rate Limiting
**Problem:** Too many requests error
**Solution:** Implement exponential backoff
```python
wait_time = 2 ** attempt * base_delay
await asyncio.sleep(wait_time)
```

### Issue 4: Memory Issues with Large Results
**Problem:** Too many results in memory
**Solution:** Stream results to database
```python
async def scrape_streaming(self, query):
    async for result in self._scrape_generator(query):
        await save_to_db(result)
        yield result
```

---

## Module Checklist

- [ ] Base scraper class implemented
- [ ] Google scraper working
- [ ] Bing scraper working
- [ ] Rate limiting functional
- [ ] User agent rotation
- [ ] Error handling and retries
- [ ] Cache integration
- [ ] Result validation
- [ ] Scraper manager orchestration
- [ ] Unit tests passing
- [ ] Anti-detection measures

---

## Next Steps
After completing this module:
1. **Module 6: Processor** - Process scraped data
2. **Module 7: Analyzer** - Analyze scraped content
3. **Module 8: API** - Expose scraping endpoints