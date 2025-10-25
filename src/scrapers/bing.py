"""
Bing Search Results Scraper
Implements Bing SERP scraping with pagination and caching
"""

import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
import logging

from bs4 import BeautifulSoup

from .base import BaseScraper
from ..config import settings
from ..cache import cache

logger = logging.getLogger(__name__)


class BingScraper(BaseScraper):
    """
    Bing search results scraper

    Features:
    - Pagination support (10 results per page)
    - Language and region filtering
    - Cache integration (1 hour TTL)
    - Date extraction from search results
    """

    def __init__(self):
        """Initialize Bing scraper"""
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
            max_results: Maximum results to return (default: 50)
            lang: Language code (default: "en")
            region: Region code (default: "us")
            **kwargs: Additional parameters

        Returns:
            List of search results with title, snippet, url, position, metadata
        """
        # Check cache
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

        results = []
        pages_needed = (max_results + self.max_results_per_page - 1) // self.max_results_per_page

        for page in range(pages_needed):
            try:
                # Build URL
                url = self._build_search_url(
                    query,
                    page * self.max_results_per_page,
                    lang,
                    region
                )

                logger.debug(f"Scraping Bing page {page + 1}/{pages_needed}: {url[:100]}...")

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
                    f"Scraped {len(page_results)} results from Bing page {page + 1} "
                    f"(total: {len(results)}/{max_results})"
                )

                if len(results) >= max_results:
                    break

                # Add delay between pages
                if page < pages_needed - 1:
                    delay = self.rate_limit_delay * 2
                    logger.debug(f"Waiting {delay}s before next page...")
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Error scraping Bing page {page + 1}: {e}")
                if not results:
                    raise
                break

        # Validate results
        results = await self.validate_results(results[:max_results])

        # Update stats
        self.stats["results_scraped"] += len(results)

        # Cache results (1 hour TTL)
        await cache.set(cache_key, results, ttl=3600)

        logger.info(f"Bing scraping completed: {len(results)} total results for '{query}'")
        return results

    def _build_search_url(
        self,
        query: str,
        first: int,
        lang: str,
        region: str
    ) -> str:
        """
        Build Bing search URL

        Args:
            query: Search query
            first: Starting result index (1-based)
            lang: Language code
            region: Region code

        Returns:
            Complete search URL
        """
        params = {
            "q": query,
            "first": first + 1,  # Bing uses 1-based indexing
            "count": self.max_results_per_page,
            "setlang": lang,
            "cc": region,
        }
        query_string = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        return f"{self.base_url}?{query_string}"

    def parse_results(
        self,
        soup: BeautifulSoup,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Parse Bing search results

        Args:
            soup: BeautifulSoup object
            max_results: Maximum results to extract

        Returns:
            List of parsed results
        """
        results = []

        # Find result containers
        # Bing uses li.b_algo for organic search results
        result_elements = soup.select("li.b_algo")

        if not result_elements:
            logger.warning("No result elements found in Bing HTML")
            return results

        logger.debug(f"Found {len(result_elements)} Bing results")

        for position, element in enumerate(result_elements[:max_results], 1):
            try:
                result = self._parse_single_result(element, position)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Error parsing Bing result {position}: {e}")
                continue

        return results

    def _parse_single_result(
        self,
        element: BeautifulSoup,
        position: int
    ) -> Optional[Dict[str, Any]]:
        """
        Parse single Bing search result element

        Args:
            element: BeautifulSoup element
            position: Result position

        Returns:
            Parsed result dictionary or None if parsing fails
        """
        # Extract URL and title from h2 > a
        link = element.select_one("h2 a")
        if not link:
            return None

        url = link.get("href", "")
        title = link.get_text(strip=True)

        if not url.startswith(("http://", "https://")):
            return None

        # Extract snippet
        snippet_element = element.select_one("div.b_caption p")
        snippet = snippet_element.get_text(strip=True) if snippet_element else ""

        # If no snippet in p, try alternative selectors
        if not snippet:
            alt_snippet = element.select_one("div.b_caption div.b_algoSlug")
            snippet = alt_snippet.get_text(strip=True) if alt_snippet else ""

        # Extract additional metadata
        metadata = {}

        # Extract date if available
        date_element = element.select_one("span.news_dt")
        if date_element:
            metadata["date"] = date_element.get_text(strip=True)

        # Extract citation (display URL)
        cite_element = element.select_one("div.b_attribution cite")
        if cite_element:
            metadata["display_url"] = cite_element.get_text(strip=True)

        # Extract deep links (sitelinks)
        deep_links = element.select("ul.b_vList li a")
        if deep_links:
            metadata["deep_links"] = [
                {
                    "title": link.get_text(strip=True),
                    "url": link.get("href", "")
                }
                for link in deep_links[:3]  # Limit to 3 deep links
            ]

        result = {
            "title": title,
            "snippet": snippet,
            "url": url,
            "position": position,
        }

        if metadata:
            result["metadata"] = metadata

        return result
