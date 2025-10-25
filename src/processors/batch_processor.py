"""
Batch Processing Module for CIAP

Provides efficient batch processing of scraped data with cleaning,
normalization, deduplication, and database persistence.
"""

from typing import List, Dict, Any, Tuple
import asyncio
import logging

from .cleaner import DataCleaner, DataNormalizer, Deduplicator
from ..database import db_manager, SearchResult

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process data in batches for efficiency

    Handles cleaning, normalization, deduplication, and database persistence
    for large datasets with memory-efficient batching.

    Features:
    - Configurable batch size
    - Automatic cleaning and normalization
    - Content-based deduplication
    - Database integration
    - Statistics tracking
    """

    def __init__(self, batch_size: int = 100):
        """
        Initialize batch processor

        Args:
            batch_size: Number of results to process in each batch
        """
        self.batch_size = batch_size
        self.cleaner = DataCleaner()
        self.normalizer = DataNormalizer()
        self.deduplicator = Deduplicator()

        logger.info(f"BatchProcessor initialized with batch_size={batch_size}")

    async def process_search_results(
        self,
        results: List[Dict],
        source: str,
        search_id: int
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Process search results in batches

        Pipeline:
        1. Clean and normalize each result
        2. Deduplicate across all results
        3. Save to database in batches
        4. Track statistics

        Args:
            results: List of raw search result dicts
            source: Source name (google, bing, etc.)
            search_id: Database search ID for foreign key

        Returns:
            Tuple of (processed_results, statistics_dict)
        """
        stats = {
            "total": len(results),
            "cleaned": 0,
            "duplicates": 0,
            "saved": 0,
            "errors": 0
        }

        logger.info(
            f"Starting batch processing: {len(results)} results from {source} "
            f"for search_id={search_id}"
        )

        # Process in batches
        all_normalized = []

        for i in range(0, len(results), self.batch_size):
            batch = results[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(results) + self.batch_size - 1) // self.batch_size

            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            # Clean and normalize
            normalized_batch = []
            for result in batch:
                try:
                    normalized = self.normalizer.normalize_search_result(
                        result, source
                    )
                    normalized_batch.append(normalized)
                    stats["cleaned"] += 1
                except Exception as e:
                    logger.error(f"Error normalizing result: {e}")
                    stats["errors"] += 1

            all_normalized.extend(normalized_batch)

        # Deduplicate across all results
        logger.debug(f"Deduplicating {len(all_normalized)} normalized results")
        unique_results = self.deduplicator.deduplicate(all_normalized)
        stats["duplicates"] = len(all_normalized) - len(unique_results)

        # Save to database in batches
        logger.debug(f"Saving {len(unique_results)} unique results to database")
        for i in range(0, len(unique_results), self.batch_size):
            batch = unique_results[i:i + self.batch_size]

            try:
                await self._save_batch(batch, search_id)
                stats["saved"] += len(batch)
            except Exception as e:
                logger.error(f"Error saving batch: {e}")
                stats["errors"] += 1

        logger.info(
            f"Batch processing complete: {stats['total']} total, "
            f"{stats['cleaned']} cleaned, {stats['duplicates']} duplicates, "
            f"{stats['saved']} saved, {stats['errors']} errors"
        )

        return unique_results, stats

    async def _save_batch(
        self,
        results: List[Dict],
        search_id: int
    ):
        """
        Save batch to database

        Args:
            results: List of normalized result dicts
            search_id: Database search ID for foreign key
        """
        async with db_manager.get_session() as session:
            for result in results:
                db_result = SearchResult(
                    search_id=search_id,
                    source=result["source"],
                    title=result["title"],
                    snippet=result["snippet"],
                    url=result["url"],
                    position=result["position"],
                    scraped_at=result["scraped_at"]
                )
                session.add(db_result)

            await session.commit()

            logger.debug(f"Saved batch of {len(results)} results for search_id={search_id}")

    async def process_and_update_search(
        self,
        results: List[Dict],
        source: str,
        search_id: int,
        update_status: bool = True
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Process results and optionally update search status

        Convenience method that processes results and updates the Search
        record status to 'completed'.

        Args:
            results: List of raw search result dicts
            source: Source name
            search_id: Database search ID
            update_status: Whether to update search status to 'completed'

        Returns:
            Tuple of (processed_results, statistics_dict)
        """
        from ..database import Search

        # Process results
        processed_results, stats = await self.process_search_results(
            results, source, search_id
        )

        # Update search status if requested
        if update_status:
            async with db_manager.get_session() as session:
                search = await session.get(Search, search_id)
                if search:
                    search.status = "completed"
                    search.results_count = stats["saved"]
                    await session.commit()
                    logger.info(f"Updated search_id={search_id} status to 'completed'")

        return processed_results, stats

    def reset_deduplicator(self):
        """
        Reset deduplicator state

        Useful when starting a new processing session and you want
        to allow duplicates across different searches.
        """
        self.deduplicator.reset()
        logger.debug("Deduplicator state reset")


# Global batch processor instance
batch_processor = BatchProcessor(batch_size=100)
