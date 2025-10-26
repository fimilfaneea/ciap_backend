"""
Scrapy Items for CIAP Web Scraping
Defines structured data containers for scraped results
"""

import scrapy
from datetime import datetime
from typing import Optional, Dict, Any


class SearchResultItem(scrapy.Item):
    """
    Scrapy item for search engine results

    Represents a single search result with all relevant metadata.
    Compatible with existing SearchResult database model.

    Fields:
        title: Result title/headline
        snippet: Result description/preview text
        url: Result URL
        position: Result position in search results (1-indexed)
        source: Search engine source (google, bing, etc.)
        scraped_at: Timestamp when result was scraped
        metadata: Additional metadata (ratings, dates, sitelinks, etc.)
    """

    # Core fields
    title = scrapy.Field()
    snippet = scrapy.Field()
    url = scrapy.Field()
    position = scrapy.Field()

    # Source metadata
    source = scrapy.Field()
    scraped_at = scrapy.Field()

    # Additional metadata (optional)
    metadata = scrapy.Field()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert item to dictionary for database storage

        Returns:
            Dictionary compatible with SearchResult model
        """
        return {
            "title": self.get("title", ""),
            "snippet": self.get("snippet", ""),
            "url": self.get("url", ""),
            "position": self.get("position", 0),
            "source": self.get("source", "unknown"),
            "scraped_at": self.get("scraped_at", datetime.utcnow()),
            "metadata": self.get("metadata"),
        }

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"SearchResultItem(title='{self.get('title', '')[:50]}...', "
            f"url='{self.get('url', '')}', position={self.get('position', 0)})"
        )
