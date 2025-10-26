"""
Scrapy Spiders for CIAP Web Scraping
Collection of specialized spiders for different search engines
"""

from .google_spider import GoogleSpider
from .bing_spider import BingSpider

__all__ = ["GoogleSpider", "BingSpider"]
