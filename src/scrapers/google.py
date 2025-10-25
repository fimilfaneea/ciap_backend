"""
Google Search Results Scraper
Implements Google SERP scraping with pagination, filtering, and caching
"""

import asyncio
import re
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus, urlparse, parse_qs
import logging

from bs4 import BeautifulSoup

from .base import BaseScraper, BlockedException
from ..config import settings
from ..cache import cache

logger = logging.getLogger(__name__)


class GoogleScraper(BaseScraper):
    """
    Google search results scraper

    Features:
    - Pagination support (10 results per page)
    - Date range filtering (day, week, month, year)
    - Language and region filtering
    - CAPTCHA detection
    - Cache integration (1 hour TTL)
    - Multiple selector fallbacks for robustness
    """

    def __init__(self):
        """Initialize Google scraper"""
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
            max_results: Maximum results to return (default: 100)
            lang: Language code (default: "en")
            region: Region code (default: "us")
            date_range: Date filter - "d" (day), "w" (week), "m" (month), "y" (year)
            **kwargs: Additional parameters

        Returns:
            List of search results with title, snippet, url, position, metadata

        Raises:
            BlockedException: If CAPTCHA or blocking detected
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

                logger.debug(f"Scraping Google page {page + 1}/{pages_needed}: {url[:100]}...")

                # Make request
                response = await self.make_request(url)

                # Parse results
                soup = self.parse_html(response.text)
                page_results = self.parse_results(soup, max_results - len(results))

                if not page_results:
                    logger.warning(f"No results on page {page + 1}, stopping pagination")
                    break

                results.extend(page_results)

                logger.info(
                    f"Scraped {len(page_results)} results from Google page {page + 1} "
                    f"(total: {len(results)}/{max_results})"
                )

                if len(results) >= max_results:
                    break

                # Add delay between pages to avoid rate limiting
                if page < pages_needed - 1:
                    delay = self.rate_limit_delay * 2
                    logger.debug(f"Waiting {delay}s before next page...")
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Error scraping Google page {page + 1}: {e}")
                if not results:  # If first page fails, re-raise
                    raise
                break  # Continue with partial results

        # Validate and clean results
        results = await self.validate_results(results[:max_results])

        # Update stats
        self.stats["results_scraped"] += len(results)

        # Cache results (1 hour TTL)
        await cache.set(cache_key, results, ttl=3600)

        logger.info(f"Google scraping completed: {len(results)} total results for '{query}'")
        return results

    def _build_search_url(
        self,
        query: str,
        start: int,
        lang: str,
        region: str,
        date_range: Optional[str]
    ) -> str:
        """
        Build Google search URL with parameters

        Args:
            query: Search query
            start: Starting result index
            lang: Language code
            region: Region code
            date_range: Date filter (d/w/m/y)

        Returns:
            Complete search URL
        """
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

        Uses multiple selector fallbacks for robustness as Google
        frequently changes their HTML structure.

        Args:
            soup: BeautifulSoup object
            max_results: Maximum results to extract

        Returns:
            List of parsed results

        Raises:
            BlockedException: If CAPTCHA detected
        """
        results = []

        # Check if blocked by CAPTCHA
        if soup.find("div", id="recaptcha"):
            raise BlockedException("Google CAPTCHA detected")

        # Find search result containers
        # Google uses different structures, try multiple selectors
        selectors = [
            "div.g",  # Standard results
            "div.tF2Cxc",  # New format
            "div.Gx5Zad",  # Alternative format
            "div[data-sokoban-container]",  # Another variant
        ]

        result_elements = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                logger.debug(f"Found {len(elements)} results using selector: {selector}")
                result_elements = elements
                break

        if not result_elements:
            logger.warning("No result elements found in Google HTML")
            # Check if there's a "did you mean" suggestion
            did_you_mean = soup.find("a", class_="gL9Hy")
            if did_you_mean:
                logger.info(f"Google suggested: {did_you_mean.get_text()}")
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
        """
        Parse single search result element

        Args:
            element: BeautifulSoup element
            position: Result position

        Returns:
            Parsed result dictionary or None if parsing fails
        """
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
        title_selectors = [
            "h3",  # Standard title
            "div.BNeawe.vvjwJb",  # Mobile format
            "div.vvjwJb",  # Alternative
        ]
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
            "div.lEBKkf",  # Newer format
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
            rating_text = rating_element.get("aria-label", "")
            metadata["rating"] = rating_text

        # Check for sitelinks
        sitelinks = element.select("div.oJeuJe a")
        if sitelinks:
            metadata["sitelinks"] = [
                {
                    "title": link.get_text(strip=True),
                    "url": link.get("href", "")
                }
                for link in sitelinks[:3]  # Limit to 3 sitelinks
            ]

        return {
            "title": title or "No title",
            "snippet": snippet,
            "url": url,
            "position": position,
            "metadata": metadata if metadata else None
        }
