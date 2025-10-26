"""
Scrapy Runner for CIAP - Async Wrapper
Bridges Scrapy (Twisted) with FastAPI (asyncio) using crochet
"""

import asyncio
import logging
from typing import List, Dict, Any, Type, Optional
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy import Spider
import crochet

from ..config import settings
from . import scrapy_settings

logger = logging.getLogger(__name__)

# Flag to track if crochet is initialized
_crochet_initialized = False
_crawler_runner = None


def initialize_scrapy():
    """
    Initialize Scrapy with crochet for async/Twisted integration

    Must be called once during application startup before running any spiders.
    Sets up the Twisted reactor in a separate thread using crochet.

    Important: AsyncioSelectorReactor should be installed BEFORE this function
    is called (via run.py or run_dev.py startup script). This function verifies
    the correct reactor is installed and initializes Scrapy with it.

    Note: If reactor is not AsyncioSelectorReactor, Scrapy-Playwright will fail.
    Always start the server using run.py or run_dev.py to ensure correct reactor.
    """
    global _crochet_initialized, _crawler_runner

    if _crochet_initialized:
        logger.warning("Scrapy already initialized, skipping")
        return

    try:
        # STEP 1: Verify which reactor is installed
        from twisted.internet import reactor
        reactor_name = reactor.__class__.__name__
        reactor_module = reactor.__class__.__module__
        reactor_type = f"{reactor_module}.{reactor_name}"

        logger.info(f"Using Twisted reactor: {reactor_type}")

        # Warn if not AsyncioSelectorReactor (Playwright won't work)
        if 'asyncioreactor' not in reactor_type.lower():
            logger.warning(
                f"Reactor type is {reactor_name}, not AsyncioSelectorReactor. "
                f"Scrapy-Playwright may not work correctly. "
                f"Ensure run.py or run_dev.py is used to start the server."
            )

        # STEP 2: Configure Scrapy logging
        configure_logging(
            settings={
                "LOG_LEVEL": settings.SCRAPY_LOG_LEVEL,
                "LOG_FORMAT": "%(levelname)s: %(message)s",
            }
        )

        # STEP 3: Setup crochet - uses existing reactor
        crochet.setup()

        # STEP 4: Create CrawlerRunner with our custom settings
        _crawler_runner = CrawlerRunner(settings=scrapy_settings.__dict__)

        _crochet_initialized = True
        logger.info("Scrapy initialized successfully with crochet")

    except Exception as e:
        logger.error(f"Failed to initialize Scrapy: {e}")
        raise


async def run_spider(
    spider_class: Type[Spider],
    timeout: int = 180,
    **spider_kwargs
) -> List[Dict[str, Any]]:
    """
    Run a Scrapy spider asynchronously and collect results

    This function bridges Scrapy's Twisted event loop with asyncio.
    Uses crochet to run the spider in a Twisted thread and returns
    results to the asyncio caller.

    Args:
        spider_class: Scrapy spider class to run
        timeout: Maximum time to wait for spider (seconds)
        **spider_kwargs: Arguments to pass to spider constructor

    Returns:
        List of scraped result dictionaries

    Raises:
        RuntimeError: If Scrapy not initialized
        TimeoutError: If spider exceeds timeout
        Exception: If spider fails

    Example:
        from .spiders import GoogleSpider

        results = await run_spider(
            GoogleSpider,
            query="AI news",
            max_results=50
        )
    """
    global _crawler_runner

    if not _crochet_initialized or not _crawler_runner:
        raise RuntimeError(
            "Scrapy not initialized. Call initialize_scrapy() first."
        )

    spider_name = spider_class.name
    logger.info(
        f"Running spider '{spider_name}' with kwargs: {spider_kwargs}"
    )

    # Use crochet's run_in_reactor decorator to run Twisted code
    @crochet.run_in_reactor
    def _run_spider_in_reactor():
        """Run spider in Twisted reactor thread"""
        deferred = _crawler_runner.crawl(spider_class, **spider_kwargs)
        return deferred

    try:
        # Start the spider in Twisted thread
        deferred = _run_spider_in_reactor()

        # Wait for spider to complete (with timeout)
        try:
            # crochet returns an EventualResult, we need to wait for it
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: deferred.wait(timeout=timeout)
                ),
                timeout=timeout
            )

        except asyncio.TimeoutError:
            logger.error(f"Spider '{spider_name}' exceeded timeout of {timeout}s")
            raise TimeoutError(f"Spider '{spider_name}' timed out after {timeout}s")

        # Retrieve results from spider's collected_items
        # We need to access the spider instance from the crawler
        results = []

        # Get the spider instance from the crawler
        # The spider stores results in self.collected_items (set by CollectorPipeline)
        for crawler in _crawler_runner.crawlers:
            if crawler.spider and hasattr(crawler.spider, 'collected_items'):
                results.extend(crawler.spider.collected_items)
                logger.info(
                    f"Retrieved {len(crawler.spider.collected_items)} "
                    f"results from spider '{spider_name}'"
                )

        logger.info(
            f"Spider '{spider_name}' completed successfully with {len(results)} results"
        )

        return results

    except Exception as e:
        logger.error(f"Spider '{spider_name}' failed: {e}")
        raise


async def run_google_spider(
    query: str,
    max_results: int = 100,
    lang: str = "en",
    region: str = "us",
    date_range: Optional[str] = None,
    timeout: int = 180
) -> List[Dict[str, Any]]:
    """
    Convenience function to run Google spider

    Args:
        query: Search query
        max_results: Maximum results to scrape
        lang: Language code
        region: Region code
        date_range: Date filter (d/w/m/y)
        timeout: Spider timeout in seconds

    Returns:
        List of search results
    """
    from .spiders import GoogleSpider

    return await run_spider(
        GoogleSpider,
        timeout=timeout,
        query=query,
        max_results=max_results,
        lang=lang,
        region=region,
        date_range=date_range
    )


async def run_bing_spider(
    query: str,
    max_results: int = 50,
    lang: str = "en",
    region: str = "us",
    timeout: int = 180
) -> List[Dict[str, Any]]:
    """
    Convenience function to run Bing spider

    Args:
        query: Search query
        max_results: Maximum results to scrape
        lang: Language code
        region: Region code
        timeout: Spider timeout in seconds

    Returns:
        List of search results
    """
    from .spiders import BingSpider

    return await run_spider(
        BingSpider,
        timeout=timeout,
        query=query,
        max_results=max_results,
        lang=lang,
        region=region
    )


def shutdown_scrapy():
    """
    Shutdown Scrapy and crochet reactor

    Call during application shutdown to cleanly stop Twisted reactor.
    """
    global _crochet_initialized, _crawler_runner

    if not _crochet_initialized:
        return

    try:
        # Note: crochet doesn't provide a clean shutdown method
        # The reactor will be stopped when the process exits
        logger.info("Scrapy shutdown initiated")

        _crochet_initialized = False
        _crawler_runner = None

    except Exception as e:
        logger.error(f"Error during Scrapy shutdown: {e}")
