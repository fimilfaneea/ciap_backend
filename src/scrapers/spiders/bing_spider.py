"""
Bing Search Spider for CIAP
Scrapes Bing search results using Playwright for JavaScript rendering
"""

import scrapy
import logging
from typing import Dict, Any, Optional
from urllib.parse import quote_plus
from datetime import datetime

from ..items import SearchResultItem

logger = logging.getLogger(__name__)


class BingSpider(scrapy.Spider):
    """
    Scrapy spider for Bing search results

    Features:
    - Playwright integration for JavaScript rendering
    - Multiple selector fallbacks
    - Pagination support
    - Metadata extraction (dates, deep links, display URLs)

    Spider arguments:
        query: Search query (required)
        max_results: Maximum results to scrape (default: 50)
        lang: Language code (default: "en")
        region: Region code (default: "us")
    """

    name = "bing"
    allowed_domains = ["bing.com"]

    # Custom settings for this spider
    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 2,
    }

    def __init__(
        self,
        query: str,
        max_results: int = 50,
        lang: str = "en",
        region: str = "us",
        *args,
        **kwargs
    ):
        """
        Initialize Bing spider

        Args:
            query: Search query
            max_results: Maximum results to return
            lang: Language code
            region: Region code
        """
        super().__init__(*args, **kwargs)

        self.query = query
        self.max_results = max_results
        self.lang = lang
        self.region = region

        self.results_per_page = 10
        self.pages_needed = (max_results + self.results_per_page - 1) // self.results_per_page

        # Initialize collection list for pipelines
        self.collected_items = []

        # Statistics
        self.results_count = 0
        self.pages_scraped = 0

        logger.info(
            f"BingSpider initialized: query='{query}', max_results={max_results}, "
            f"lang={lang}, region={region}, pages_needed={self.pages_needed}"
        )

    def start_requests(self):
        """Generate initial requests for each page"""
        for page in range(self.pages_needed):
            first_index = page * self.results_per_page
            url = self._build_search_url(first_index)

            # Use Playwright for JavaScript rendering
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_goto_kwargs": {
                        "wait_until": "networkidle",
                        "timeout": 30000,
                    },
                    "page": page,
                    "first_index": first_index,
                },
                errback=self.errback,
            )

    def _build_search_url(self, first: int) -> str:
        """
        Build Bing search URL

        Args:
            first: Starting result index (0-based, Bing uses 1-based)

        Returns:
            Complete search URL
        """
        base_url = "https://www.bing.com/search"

        params = {
            "q": self.query,
            "first": first + 1,  # Bing uses 1-based indexing
            "count": self.results_per_page,
            "setlang": self.lang,
            "cc": self.region,
        }

        # Build query string
        query_string = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        url = f"{base_url}?{query_string}"

        logger.debug(f"Built Bing URL: {url[:150]}...")
        return url

    async def parse(self, response):
        """
        Parse Bing search results page

        Args:
            response: Scrapy response with Playwright page

        Yields:
            SearchResultItem for each result found
        """
        page_num = response.meta.get("page", 0) + 1
        first_index = response.meta.get("first_index", 0)

        logger.info(f"Parsing Bing page {page_num} (first_index={first_index})")

        # Find result containers (Bing uses li.b_algo)
        results = response.css("li.b_algo")

        if not results:
            logger.warning(f"No results found on Bing page {page_num}")

            # Close Playwright page
            page = response.meta.get("playwright_page")
            if page:
                await page.close()

            return

        logger.info(f"Found {len(results)} result containers on page {page_num}")

        # Parse each result
        position_offset = first_index
        for idx, result in enumerate(results, 1):
            try:
                item = self._parse_result(result, position_offset + idx)
                if item:
                    self.results_count += 1
                    yield item

                    # Stop if we've reached max_results
                    if self.results_count >= self.max_results:
                        logger.info(f"Reached max_results ({self.max_results}), stopping")
                        break

            except Exception as e:
                logger.warning(f"Error parsing result {idx} on page {page_num}: {e}")
                continue

        self.pages_scraped += 1

        # Close Playwright page
        page = response.meta.get("playwright_page")
        if page:
            await page.close()

        logger.info(
            f"Page {page_num} complete: {self.results_count}/{self.max_results} results"
        )

    def _parse_result(self, result_element, position: int) -> Optional[SearchResultItem]:
        """
        Parse single Bing search result

        Args:
            result_element: Selector for result container
            position: Result position

        Returns:
            SearchResultItem or None if parsing fails
        """
        # Extract URL and title from h2 > a
        link = result_element.css("h2 a")
        if not link:
            return None

        url = link.attrib.get("href", "")
        title = link.css("::text").get() or "No title"

        if not url.startswith(("http://", "https://")):
            return None

        # Extract snippet (try multiple selectors)
        snippet = (
            result_element.css("div.b_caption p::text").get()
            or result_element.css("div.b_caption div.b_algoSlug::text").get()
            or result_element.xpath(".//div[@class='b_caption']//p//text()").get()
            or ""
        )

        # Extract metadata
        metadata = {}

        # Extract date
        date_elem = result_element.css("span.news_dt::text").get()
        if date_elem:
            metadata["date"] = date_elem

        # Extract citation (display URL)
        cite_elem = result_element.css("div.b_attribution cite::text").get()
        if cite_elem:
            metadata["display_url"] = cite_elem

        # Extract deep links (sitelinks)
        deep_links = result_element.css("ul.b_vList li a")
        if deep_links:
            metadata["deep_links"] = [
                {
                    "title": link.css("::text").get() or "",
                    "url": link.attrib.get("href", ""),
                }
                for link in deep_links[:3]  # Limit to 3
            ]

        # Create item
        item = SearchResultItem()
        item["title"] = title
        item["snippet"] = snippet
        item["url"] = url
        item["position"] = position
        item["source"] = "bing"
        item["scraped_at"] = datetime.utcnow()

        if metadata:
            item["metadata"] = metadata

        return item

    def errback(self, failure):
        """
        Handle request errors

        Args:
            failure: Twisted failure object
        """
        logger.error(f"Request failed: {failure.value}")

    def closed(self, reason):
        """
        Called when spider closes

        Args:
            reason: Reason for closure
        """
        logger.info(
            f"BingSpider closed: reason={reason}, "
            f"results={self.results_count}/{self.max_results}, "
            f"pages={self.pages_scraped}/{self.pages_needed}"
        )
