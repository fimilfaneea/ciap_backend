"""
Scraper Manager for CIAP Web Scraping System
Orchestrates multiple scrapers and handles database integration
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .base import BaseScraper, ScraperException
from .google import GoogleScraper
from .bing import BingScraper
from ..database import db_manager, Search, SearchResult, ScrapingJob

logger = logging.getLogger(__name__)


class ScraperManager:
    """
    Orchestrate multiple scrapers

    Manages parallel scraping from multiple sources, database integration,
    and task queue scheduling.

    Features:
    - Multi-source parallel scraping
    - Database integration (Search, SearchResult, ScrapingJob)
    - Graceful error handling (continue if one source fails)
    - Statistics aggregation
    - Task queue integration
    """

    def __init__(self):
        """Initialize scraper manager"""
        self.scrapers: Dict[str, BaseScraper] = {
            "google": GoogleScraper(),
            "bing": BingScraper(),
        }

        self.default_sources = ["google", "bing"]

        logger.info(
            f"ScraperManager initialized with {len(self.scrapers)} scrapers: "
            f"{', '.join(self.scrapers.keys())}"
        )

    async def scrape(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 50,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Scrape from multiple sources in parallel

        Args:
            query: Search query
            sources: List of sources to scrape (default: all available)
            max_results_per_source: Max results per source (default: 50)
            **kwargs: Additional scraper parameters

        Returns:
            Dictionary of results by source: {"google": [...], "bing": [...]}

        Example:
            results = await manager.scrape(
                "AI news",
                sources=["google", "bing"],
                max_results_per_source=100
            )
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

        logger.info(
            f"Starting parallel scraping for '{query}' from {len(sources)} sources: {sources}"
        )

        # Scrape in parallel using asyncio.gather
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

        # Execute all tasks in parallel with exception handling
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        results = {}
        for source, result in zip(sources, results_list):
            if isinstance(result, Exception):
                logger.error(f"Scraping {source} failed: {result}")
                results[source] = []
            else:
                results[source] = result

        total_results = sum(len(r) for r in results.values())
        logger.info(
            f"Parallel scraping completed: {total_results} total results from "
            f"{len([r for r in results.values() if r])} sources"
        )

        return results

    async def _scrape_source(
        self,
        scraper: BaseScraper,
        query: str,
        max_results: int,
        **kwargs
    ) -> List[Dict]:
        """
        Scrape single source with error handling

        Args:
            scraper: Scraper instance
            query: Search query
            max_results: Maximum results
            **kwargs: Additional parameters

        Returns:
            List of results

        Raises:
            ScraperException: On scraping failure
        """
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

        Creates Search record, scrapes from sources, saves SearchResult
        and ScrapingJob records.

        Args:
            search_id: Search ID
            query: Search query
            sources: Sources to scrape
            **kwargs: Additional scraper parameters

        Returns:
            Summary of scraping results

        Raises:
            ValueError: If search_id not found
            ScraperException: On scraping failure
        """
        # Update search status to processing
        async with db_manager.get_session() as session:
            search = await session.get(Search, search_id)
            if not search:
                raise ValueError(f"Search {search_id} not found")

            search.status = "processing"
            search.updated_at = datetime.utcnow()
            await session.commit()

        logger.info(f"Starting scrape_and_save for search_id={search_id}, query='{query}'")

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
                        results_count=len(results),
                        started_at=datetime.utcnow(),
                        completed_at=datetime.utcnow()
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
                            position=result["position"],
                            scraped_at=result.get("scraped_at", datetime.utcnow()),
                            result_metadata=result.get("metadata")
                        )
                        session.add(search_result)

                    saved_counts[source] = len(results)

                # Update search status to completed
                search = await session.get(Search, search_id)
                search.status = "completed"
                search.completed_at = datetime.utcnow()
                search.updated_at = datetime.utcnow()
                await session.commit()

            logger.info(
                f"Scrape_and_save completed for search_id={search_id}: "
                f"{saved_counts}"
            )

            return {
                "search_id": search_id,
                "query": query,
                "sources": list(all_results.keys()),
                "results_count": saved_counts,
                "total_results": sum(saved_counts.values()),
                "status": "completed"
            }

        except Exception as e:
            # Update search status to failed
            async with db_manager.get_session() as session:
                search = await session.get(Search, search_id)
                search.status = "failed"
                search.error_message = str(e)
                search.updated_at = datetime.utcnow()
                await session.commit()

            logger.error(f"Scrape_and_save failed for search_id={search_id}: {e}")
            raise

    async def schedule_scraping(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        priority: int = 5,
        **kwargs
    ) -> int:
        """
        Schedule scraping as background task

        Creates a Search record and enqueues a scraping task to the task queue.

        Args:
            query: Search query
            sources: Sources to scrape
            priority: Task priority (1=high, 10=low)
            **kwargs: Additional scraper parameters

        Returns:
            Task ID

        Example:
            task_id = await manager.schedule_scraping(
                "AI news",
                sources=["google"],
                priority=3
            )
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
            await session.refresh(search)
            search_id = search.id

        logger.info(
            f"Created search record: search_id={search_id}, query='{query}', "
            f"sources={sources or self.default_sources}"
        )

        # Import task_queue here to avoid circular imports
        from ..task_queue import task_queue

        # Enqueue scraping task
        task_id = await task_queue.enqueue(
            task_type="scrape",
            payload={
                "search_id": search_id,
                "query": query,
                "sources": sources or self.default_sources,
                **kwargs
            },
            priority=priority
        )

        logger.info(
            f"Enqueued scraping task: task_id={task_id}, search_id={search_id}, "
            f"priority={priority}"
        )

        return task_id

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all scrapers

        Returns:
            Dictionary with statistics for each scraper

        Example:
            stats = manager.get_stats()
            # {
            #     "google": {"scraper": "GoogleScraper", "requests_made": 10, ...},
            #     "bing": {"scraper": "BingScraper", "requests_made": 5, ...}
            # }
        """
        stats = {}
        for name, scraper in self.scrapers.items():
            stats[name] = scraper.get_stats()

        # Add aggregate stats
        total_requests = sum(s["requests_made"] for s in stats.values())
        total_failed = sum(s["requests_failed"] for s in stats.values())
        total_results = sum(s["results_scraped"] for s in stats.values())

        stats["aggregate"] = {
            "total_requests": total_requests,
            "total_failed": total_failed,
            "total_results": total_results,
            "overall_success_rate": (
                (total_requests - total_failed) / total_requests * 100
                if total_requests > 0 else 0
            )
        }

        return stats


# Global scraper manager instance
scraper_manager = ScraperManager()
