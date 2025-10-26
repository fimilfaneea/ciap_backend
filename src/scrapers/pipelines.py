"""
Scrapy Pipelines for CIAP Web Scraping
Processes scraped items through validation, cleaning, deduplication, and collection
"""

import re
import logging
from typing import Set, List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse, urljoin
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

from .items import SearchResultItem

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Validate scraped items have required fields

    Ensures all items have at minimum:
    - url: Valid HTTP/HTTPS URL
    - title: Non-empty string
    - position: Positive integer

    Items missing required fields are dropped.
    """

    def process_item(self, item: SearchResultItem, spider):
        """Validate item has required fields"""
        adapter = ItemAdapter(item)

        # Check required fields
        if not adapter.get("url"):
            raise DropItem(f"Missing URL in item: {item}")

        if not adapter.get("title"):
            raise DropItem(f"Missing title in item: {item}")

        url = adapter["url"]
        if not url.startswith(("http://", "https://")):
            raise DropItem(f"Invalid URL in item: {url}")

        # Set defaults for optional fields
        if not adapter.get("snippet"):
            adapter["snippet"] = ""

        if not adapter.get("position"):
            adapter["position"] = 0

        if not adapter.get("source"):
            adapter["source"] = spider.name.lower()

        if not adapter.get("scraped_at"):
            adapter["scraped_at"] = datetime.utcnow()

        return item


class CleaningPipeline:
    """
    Clean and normalize item data

    Operations:
    - Clean and truncate text fields
    - Normalize URLs (remove tracking parameters)
    - Remove extra whitespace
    - Sanitize special characters
    """

    def __init__(self):
        self.max_title_length = 500
        self.max_snippet_length = 1000

    def process_item(self, item: SearchResultItem, spider):
        """Clean and normalize item data"""
        adapter = ItemAdapter(item)

        # Clean title
        title = self._clean_text(adapter.get("title", ""))
        adapter["title"] = self._truncate(title, self.max_title_length)

        # Clean snippet
        snippet = self._clean_text(adapter.get("snippet", ""))
        adapter["snippet"] = self._truncate(snippet, self.max_snippet_length)

        # Normalize URL
        url = adapter.get("url", "")
        adapter["url"] = self._normalize_url(url)

        return item

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters

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

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _truncate(self, text: str, max_length: int) -> str:
        """
        Truncate text to maximum length

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text

        return text[: max_length - 3] + "..."

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing tracking parameters

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        if not url:
            return ""

        try:
            parsed = urlparse(url)

            # Remove common tracking parameters
            # Keep only essential parameters
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            # Remove trailing slash
            if url.endswith("/") and parsed.path != "/":
                url = url.rstrip("/")

            return url

        except Exception as e:
            logger.warning(f"Failed to normalize URL {url}: {e}")
            return url


class DeduplicationPipeline:
    """
    Remove duplicate items based on URL

    Maintains a set of seen URLs per spider instance.
    Drops items with duplicate URLs.
    """

    def __init__(self):
        self.seen_urls: Set[str] = set()

    def open_spider(self, spider):
        """Reset seen URLs when spider opens"""
        self.seen_urls = set()
        logger.debug(f"DeduplicationPipeline initialized for {spider.name}")

    def close_spider(self, spider):
        """Log deduplication stats when spider closes"""
        logger.info(
            f"DeduplicationPipeline: {len(self.seen_urls)} unique URLs for {spider.name}"
        )

    def process_item(self, item: SearchResultItem, spider):
        """Check for duplicate URLs"""
        adapter = ItemAdapter(item)
        url = adapter.get("url", "")

        if url in self.seen_urls:
            raise DropItem(f"Duplicate URL: {url}")

        self.seen_urls.add(url)
        return item


class CollectorPipeline:
    """
    Collect processed items for async retrieval

    Stores items in memory during spider execution.
    Allows async wrapper to retrieve results after spider completes.

    Note: Items are stored in spider.collected_items attribute
    """

    def open_spider(self, spider):
        """Initialize collection list"""
        if not hasattr(spider, "collected_items"):
            spider.collected_items = []
        logger.debug(f"CollectorPipeline initialized for {spider.name}")

    def close_spider(self, spider):
        """Log collection stats"""
        items_count = len(getattr(spider, "collected_items", []))
        logger.info(f"CollectorPipeline: Collected {items_count} items for {spider.name}")

    def process_item(self, item: SearchResultItem, spider):
        """Add item to collection"""
        if not hasattr(spider, "collected_items"):
            spider.collected_items = []

        # Convert item to dict for easier handling
        adapter = ItemAdapter(item)
        item_dict = {
            "title": adapter.get("title", ""),
            "snippet": adapter.get("snippet", ""),
            "url": adapter.get("url", ""),
            "position": adapter.get("position", 0),
            "source": adapter.get("source", ""),
            "scraped_at": adapter.get("scraped_at", datetime.utcnow()),
            "metadata": adapter.get("metadata"),
        }

        spider.collected_items.append(item_dict)

        return item
