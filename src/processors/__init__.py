"""
Processors Module - Data Processing System for CIAP

Provides data cleaning, normalization, deduplication, and batch processing
for scraped search results.

Features:
- Text cleaning and HTML sanitization
- URL normalization with tracking parameter removal
- Data standardization across sources
- Content-based duplicate detection
- Efficient batch processing with database integration
- Quality scoring for results

Usage:
    from src.processors import DataCleaner, DataNormalizer, Deduplicator
    from src.processors import BatchProcessor, batch_processor

    # Clean text
    clean_text = DataCleaner.clean_text("<p>Hello World</p>")

    # Normalize result
    normalized = DataNormalizer.normalize_search_result(raw_result, "google")

    # Deduplicate results
    dedup = Deduplicator(similarity_threshold=0.85)
    unique_results = dedup.deduplicate(results)

    # Batch process and save
    processor = BatchProcessor(batch_size=100)
    processed, stats = await processor.process_search_results(
        results, "google", search_id=1
    )
"""

from .cleaner import DataCleaner, DataNormalizer, Deduplicator
from .batch_processor import BatchProcessor, batch_processor

__all__ = [
    # Cleaner classes
    "DataCleaner",
    "DataNormalizer",
    "Deduplicator",
    # Batch processor
    "BatchProcessor",
    "batch_processor",
]

__version__ = "0.6.0"
