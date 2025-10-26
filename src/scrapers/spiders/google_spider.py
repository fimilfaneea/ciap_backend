"""
Google Search Spider for CIAP
Scrapes Google search results using Playwright for JavaScript rendering
"""

import scrapy
import logging
from typing import Dict, Any, Optional
from urllib.parse import quote_plus
from datetime import datetime

from ..items import SearchResultItem

logger = logging.getLogger(__name__)


class GoogleSpider(scrapy.Spider):
    """
    Scrapy spider for Google search results

    Features:
    - Playwright integration for JavaScript rendering
    - Multiple selector fallbacks for robustness
    - CAPTCHA detection
    - Pagination support
    - Metadata extraction (ratings, dates, sitelinks)

    Spider arguments:
        query: Search query (required)
        max_results: Maximum results to scrape (default: 100)
        lang: Language code (default: "en")
        region: Region code (default: "us")
        date_range: Date filter - d/w/m/y (optional)
    """

    name = "google"
    allowed_domains = ["google.com"]

    # Custom settings for this spider
    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 1,  # Be conservative with Google
        "DOWNLOAD_DELAY": 3,  # Longer delay for Google
    }

    def __init__(
        self,
        query: str,
        max_results: int = 100,
        lang: str = "en",
        region: str = "us",
        date_range: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        Initialize Google spider

        Args:
            query: Search query
            max_results: Maximum results to return
            lang: Language code
            region: Region code
            date_range: Date filter (d/w/m/y)
        """
        super().__init__(*args, **kwargs)

        self.query = query
        self.max_results = max_results
        self.lang = lang
        self.region = region
        self.date_range = date_range

        self.results_per_page = 10
        self.pages_needed = (max_results + self.results_per_page - 1) // self.results_per_page

        # Initialize collection list for pipelines
        self.collected_items = []

        # Statistics
        self.results_count = 0
        self.pages_scraped = 0

        logger.info(
            f"GoogleSpider initialized: query='{query}', max_results={max_results}, "
            f"lang={lang}, region={region}, pages_needed={self.pages_needed}"
        )

    def start_requests(self):
        """Generate initial requests for each page"""
        for page in range(self.pages_needed):
            start_index = page * self.results_per_page
            url = self._build_search_url(start_index)

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
                    "start_index": start_index,
                },
                errback=self.errback,
            )

    def _build_search_url(self, start: int) -> str:
        """
        Build Google search URL

        Args:
            start: Starting result index

        Returns:
            Complete search URL
        """
        base_url = "https://www.google.com/search"

        params = {
            "q": self.query,
            "start": start,
            "num": self.results_per_page,
            "hl": self.lang,
            "gl": self.region,
            "safe": "off",
        }

        # Add date range filter
        if self.date_range:
            tbs_map = {
                "d": "qdr:d",  # Past day
                "w": "qdr:w",  # Past week
                "m": "qdr:m",  # Past month
                "y": "qdr:y",  # Past year
            }
            if self.date_range in tbs_map:
                params["tbs"] = tbs_map[self.date_range]

        # Build query string
        query_string = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        url = f"{base_url}?{query_string}"

        logger.debug(f"Built Google URL: {url[:150]}...")
        return url

    async def parse(self, response):
        """
        Parse Google search results page

        Args:
            response: Scrapy response with Playwright page

        Yields:
            SearchResultItem for each result found
        """
        page_num = response.meta.get("page", 0) + 1
        start_index = response.meta.get("start_index", 0)

        logger.info(f"Parsing Google page {page_num} (start_index={start_index})")

        # Check for CAPTCHA
        if self._is_captcha(response):
            logger.error(f"CAPTCHA detected on page {page_num} - stopping spider")
            return

        # Try multiple selector strategies
        results = []

        # Strategy 1: Modern Google layout (div.g containers)
        results.extend(response.css("div.g"))

        # Strategy 2: Alternative layout (div.tF2Cxc)
        if not results:
            results.extend(response.css("div.tF2Cxc"))

        # Strategy 3: Older layout (div.Gx5Zad)
        if not results:
            results.extend(response.css("div.Gx5Zad"))

        # Strategy 4: Data attribute selector
        if not results:
            results.extend(response.css("div[data-sokoban-container]"))

        if not results:
            logger.warning(f"No results found on Google page {page_num}")

            # Check for "did you mean" suggestion
            suggestion = response.css("a.gL9Hy::text").get()
            if suggestion:
                logger.info(f"Google suggested: {suggestion}")

            # Close Playwright page
            page = response.meta.get("playwright_page")
            if page:
                await page.close()

            return

        logger.info(f"Found {len(results)} result containers on page {page_num}")

        # Parse each result
        position_offset = start_index
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
        Parse single Google search result

        Args:
            result_element: Selector for result container
            position: Result position

        Returns:
            SearchResultItem or None if parsing fails
        """
        # Extract URL (try multiple selectors)
        url = (
            result_element.css("a[href]::attr(href)").get()
            or result_element.xpath(".//a/@href").get()
        )

        if not url or not url.startswith(("http://", "https://")):
            return None

        # Extract title (try multiple selectors)
        title = (
            result_element.css("h3::text").get()
            or result_element.css("div.BNeawe.vvjwJb::text").get()
            or result_element.css("div.vvjwJb::text").get()
            or result_element.xpath(".//h3//text()").get()
            or "No title"
        )

        # Extract snippet (try multiple selectors)
        snippet = (
            result_element.css("div.VwiC3b::text").get()
            or result_element.css("span.aCOpRe::text").get()
            or result_element.css("div.s3v9rd::text").get()
            or result_element.css("div.BNeawe.s3v9rd::text").get()
            or result_element.css("div.lEBKkf::text").get()
            or result_element.xpath(".//div[contains(@class, 'VwiC3b')]//text()").get()
            or ""
        )

        # Extract metadata
        metadata = {}

        # Extract date
        date_text = result_element.css("span.MUxGbd::text").get()
        if date_text:
            metadata["date"] = date_text

        # Extract rating
        rating_elem = result_element.css("span.z3HNkc")
        if rating_elem:
            rating_text = rating_elem.attrib.get("aria-label", "")
            if rating_text:
                metadata["rating"] = rating_text

        # Extract sitelinks
        sitelinks = result_element.css("div.oJeuJe a")
        if sitelinks:
            metadata["sitelinks"] = [
                {
                    "title": link.css("::text").get() or "",
                    "url": link.attrib.get("href", ""),
                }
                for link in sitelinks[:3]  # Limit to 3
            ]

        # Create item
        item = SearchResultItem()
        item["title"] = title
        item["snippet"] = snippet
        item["url"] = url
        item["position"] = position
        item["source"] = "google"
        item["scraped_at"] = datetime.utcnow()

        if metadata:
            item["metadata"] = metadata

        return item

    def _is_captcha(self, response) -> bool:
        """
        Check if response contains CAPTCHA

        Args:
            response: Scrapy response

        Returns:
            True if CAPTCHA detected
        """
        # Check for reCAPTCHA div
        if response.css("div#recaptcha").get():
            return True

        # Check for CAPTCHA in page text
        page_text = response.text.lower()
        if "captcha" in page_text or "unusual traffic" in page_text:
            return True

        return False

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
            f"GoogleSpider closed: reason={reason}, "
            f"results={self.results_count}/{self.max_results}, "
            f"pages={self.pages_scraped}/{self.pages_needed}"
        )
