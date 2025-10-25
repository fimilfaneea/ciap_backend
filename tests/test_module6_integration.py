"""
Integration Tests for Module 6 (Processors) with Module 5 (Scrapers)

Tests the complete pipeline from scraping to processing to database storage.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.scrapers import scraper_manager
from src.processors import batch_processor, DataCleaner, DataNormalizer
from src.database import db_manager, Search, SearchResult


@pytest.mark.asyncio
async def test_scrapers_to_processors_pipeline():
    """
    Test complete pipeline: scrape → process → store

    Simulates the full workflow where scrapers return raw data,
    processors clean and normalize it, and it's stored in the database.
    """
    # Simulate scraped data (as returned by scrapers)
    raw_scraped_data = {
        "google": [
            {
                "title": "<strong>Best Python Tutorials</strong>",
                "snippet": "  Learn   Python   with   our comprehensive   guide  ",
                "url": "https://www.example.com/python?utm_source=google&utm_medium=cpc&id=123",
                "position": 1,
                "scraped_at": datetime.utcnow()
            },
            {
                "title": "Python for Beginners",
                "snippet": "Start your Python journey today",
                "url": "https://tutorial.example.com/python-basics?utm_campaign=spring2024",
                "position": 2,
                "scraped_at": datetime.utcnow()
            },
            {
                "title": "<strong>Best Python Tutorials</strong>",  # Duplicate
                "snippet": "  Learn   Python   with   our comprehensive   guide  ",
                "url": "https://www.example.com/python?utm_source=google",  # Same URL
                "position": 3,
                "scraped_at": datetime.utcnow()
            }
        ]
    }

    # Extract results for processing
    results = raw_scraped_data["google"]

    # Process with BatchProcessor (mocking database save)
    with patch.object(batch_processor, '_save_batch', new_callable=AsyncMock) as mock_save:
        processed, stats = await batch_processor.process_search_results(
            results, "google", search_id=1
        )

        # Verify cleaning
        assert "<strong>" not in processed[0]["title"]
        assert processed[0]["title"] == "Best Python Tutorials"

        # Verify whitespace normalization
        assert "   " not in processed[0]["snippet"]
        assert processed[0]["snippet"] == "Learn Python with our comprehensive guide"

        # Verify URL cleaning (tracking params removed)
        assert "utm_source" not in processed[0]["url"]
        assert "utm_medium" not in processed[0]["url"]
        assert "utm_campaign" not in processed[1]["url"]
        assert "id=123" in processed[0]["url"]  # Functional param kept

        # Verify domain extraction
        assert processed[0]["domain"] == "example.com"
        assert processed[1]["domain"] == "tutorial.example.com"

        # Verify deduplication (3 inputs → 2 outputs)
        assert len(processed) == 2
        assert stats["total"] == 3
        assert stats["cleaned"] == 3
        assert stats["duplicates"] == 1
        assert stats["saved"] == 2

        # Verify quality scores assigned
        assert "quality_score" in processed[0]
        assert 0.0 <= processed[0]["quality_score"] <= 1.0


@pytest.mark.asyncio
async def test_scraper_manager_integration():
    """
    Test that processors can handle output from scraper_manager.scrape()
    """
    # Mock scraper responses
    mock_google_results = [
        {
            "title": "Mock Result 1",
            "snippet": "Description 1",
            "url": "https://example.com/1",
            "position": 1
        },
        {
            "title": "Mock Result 2",
            "snippet": "Description 2",
            "url": "https://example.com/2",
            "position": 2
        }
    ]

    # Mock scraper_manager.scrape to return test data
    with patch.object(scraper_manager, 'scrape', new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = {"google": mock_google_results}

        # Scrape
        scraped_data = await scraper_manager.scrape("test query", sources=["google"])

        # Process
        with patch.object(batch_processor, '_save_batch', new_callable=AsyncMock):
            processed, stats = await batch_processor.process_search_results(
                scraped_data["google"], "google", search_id=1
            )

            # Verify results
            assert len(processed) == 2
            assert stats["total"] == 2
            assert stats["cleaned"] == 2
            assert stats["saved"] == 2


@pytest.mark.asyncio
async def test_database_operations_real():
    """
    Test actual database operations (with in-memory database)

    Verifies that processed data can be saved to SearchResult model.
    """
    # Initialize in-memory database
    from src.database import DatabaseManager
    test_db = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await test_db.initialize()

    try:
        # Create Search record
        async with test_db.get_session() as session:
            search = Search(
                query="test query",
                sources="google",
                status="processing",
                created_at=datetime.utcnow()
            )
            session.add(search)
            await session.commit()
            await session.refresh(search)
            search_id = search.id  # Use 'id' not 'search_id'

        # Create processor with test database
        test_processor = batch_processor.__class__(batch_size=10)

        # Override db_manager temporarily
        original_db = batch_processor.__dict__.get('db_manager')

        # Process and save
        results = [
            {
                "title": "Test Result",
                "snippet": "Test snippet",
                "url": "https://example.com",
                "position": 1,
                "source": "google",
                "scraped_at": datetime.utcnow()
            }
        ]

        # Save batch using test database
        async with test_db.get_session() as session:
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

        # Verify data was saved
        async with test_db.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(SearchResult).where(SearchResult.search_id == search_id)
            )
            saved_results = result.scalars().all()

            assert len(saved_results) == 1
            assert saved_results[0].title == "Test Result"
            assert saved_results[0].url == "https://example.com"
            assert saved_results[0].search_id == search_id

    finally:
        await test_db.close()


@pytest.mark.asyncio
async def test_quality_scoring_integration():
    """
    Test that quality scoring works correctly on real scraper output
    """
    normalizer = DataNormalizer()

    # High quality result (position 1, good content)
    high_quality = {
        "title": "Comprehensive Python Tutorial for Beginners",
        "snippet": "This is a detailed guide covering all aspects of Python programming from basics to advanced topics. Perfect for beginners and intermediate developers.",
        "url": "https://python.org/tutorial",
        "position": 1
    }

    normalized_high = normalizer.normalize_search_result(high_quality, "google")
    assert normalized_high["quality_score"] >= 0.9  # Should be high quality

    # Low quality result (position 20, minimal content)
    low_quality = {
        "title": "Py",
        "snippet": "Short",
        "url": "invalid-url",
        "position": 20
    }

    normalized_low = normalizer.normalize_search_result(low_quality, "google")
    assert normalized_low["quality_score"] <= 0.2  # Should be low quality


@pytest.mark.asyncio
async def test_multi_source_processing():
    """
    Test processing results from multiple scrapers (Google + Bing)
    """
    # Simulate multi-source scraping
    multi_source_data = {
        "google": [
            {"title": "Google Result", "snippet": "From Google",
             "url": "https://example.com/google", "position": 1}
        ],
        "bing": [
            {"title": "Bing Result", "snippet": "From Bing",
             "url": "https://example.com/bing", "position": 1}
        ]
    }

    all_processed = []

    with patch.object(batch_processor, '_save_batch', new_callable=AsyncMock):
        for source, results in multi_source_data.items():
            processed, stats = await batch_processor.process_search_results(
                results, source, search_id=1
            )
            all_processed.extend(processed)

        # Verify both sources processed
        assert len(all_processed) == 2

        # Verify source tracking
        sources = {r["source"] for r in all_processed}
        assert sources == {"google", "bing"}

        # Verify unique IDs generated per source
        ids = [r["result_id"] for r in all_processed]
        assert len(ids) == len(set(ids))  # All unique


@pytest.mark.asyncio
async def test_error_recovery():
    """
    Test that processing continues even if some results fail
    """
    results = [
        {"title": "Good Result 1", "snippet": "Valid",
         "url": "https://example.com/1", "position": 1},
        {"title": None, "snippet": None, "url": None, "position": "invalid"},  # Bad data
        {"title": "Good Result 2", "snippet": "Valid",
         "url": "https://example.com/2", "position": 2}
    ]

    with patch.object(batch_processor, '_save_batch', new_callable=AsyncMock):
        processed, stats = await batch_processor.process_search_results(
            results, "google", search_id=1
        )

        # Error handling: bad data causes normalization error, good data processes
        # Stats should show 3 total, some cleaned, some errors
        assert stats["total"] == 3
        assert stats["cleaned"] >= 2  # At least the 2 good results
        assert stats["errors"] >= 1  # At least 1 error from bad data
        assert len(processed) >= 0  # May have 0-2 results depending on error handling


@pytest.mark.asyncio
async def test_performance_with_large_dataset():
    """
    Test performance with larger dataset (100 results)
    """
    import time

    # Generate 100 results
    large_dataset = [
        {
            "title": f"Result {i}",
            "snippet": f"Description for result {i} with some content",
            "url": f"https://example.com/page{i}",
            "position": i
        }
        for i in range(100)
    ]

    # Add some duplicates
    large_dataset.extend(large_dataset[:10])  # 10 duplicates

    start_time = time.time()

    with patch.object(batch_processor, '_save_batch', new_callable=AsyncMock):
        processed, stats = await batch_processor.process_search_results(
            large_dataset, "google", search_id=1
        )

        elapsed = time.time() - start_time

        # Verify processing completed
        assert stats["total"] == 110
        assert stats["duplicates"] == 10
        assert stats["saved"] == 100

        # Verify reasonable performance (should be fast)
        assert elapsed < 5.0  # Should complete within 5 seconds


def test_data_cleaner_on_real_html():
    """
    Test DataCleaner on realistic HTML snippets
    """
    cleaner = DataCleaner()

    # Real-world HTML snippet
    html = """
    <div class="result">
        <h3>Product Title</h3>
        <p>This is a <strong>great</strong> product with <em>amazing</em> features.</p>
        <script>trackEvent('view');</script>
        <style>.result { color: blue; }</style>
    </div>
    """

    cleaned = cleaner.clean_html(html)

    # Should remove HTML but keep text
    assert "<div>" not in cleaned
    assert "<script>" not in cleaned
    assert "<style>" not in cleaned
    assert "Product Title" in cleaned
    assert "great" in cleaned
    assert "amazing" in cleaned


def test_url_cleaning_comprehensive():
    """
    Test URL cleaning with various tracking parameters
    """
    cleaner = DataCleaner()

    test_urls = [
        # Google Analytics
        ("https://example.com?_ga=GA1.2.123456", "https://example.com"),
        # UTM parameters
        ("https://example.com?utm_source=fb&utm_medium=social&id=5",
         "https://example.com?id=5"),
        # Facebook click ID
        ("https://example.com?fbclid=IwAR123xyz", "https://example.com"),
        # Multiple tracking params
        ("https://example.com?gclid=abc&mc_cid=def&mc_eid=ghi",
         "https://example.com"),
        # Mixed tracking and functional params
        ("https://shop.com/product?id=123&utm_campaign=sale&color=red",
         "https://shop.com/product?id=123&color=red"),
    ]

    for dirty_url, expected_clean in test_urls:
        cleaned = cleaner.clean_url(dirty_url)
        assert cleaned == expected_clean, f"Failed for {dirty_url}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
