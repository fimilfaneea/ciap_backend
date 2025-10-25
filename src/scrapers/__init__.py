"""Scrapers module - Web scraping system for CIAP"""

from .base import BaseScraper, ScraperException, RateLimitException, BlockedException
from .google import GoogleScraper
from .bing import BingScraper
from .manager import ScraperManager, scraper_manager

__all__ = [
    # Base scraper
    "BaseScraper",

    # Exceptions
    "ScraperException",
    "RateLimitException",
    "BlockedException",

    # Scraper implementations
    "GoogleScraper",
    "BingScraper",

    # Manager
    "ScraperManager",
    "scraper_manager",  # Global singleton instance
]
