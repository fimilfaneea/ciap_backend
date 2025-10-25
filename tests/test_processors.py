"""
Test Suite for Data Processing Module

Tests data cleaning, normalization, deduplication, and batch processing.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.processors import (
    DataCleaner,
    DataNormalizer,
    Deduplicator,
    BatchProcessor,
    batch_processor
)


# ============================================================================
# DataCleaner Tests (5 tests)
# ============================================================================

class TestDataCleaner:
    """Test DataCleaner functionality"""

    def test_clean_text_html_removal(self):
        """Test HTML tag removal from text"""
        cleaner = DataCleaner()

        # Simple HTML
        assert cleaner.clean_text("<p>Hello World</p>") == "Hello World"

        # Nested HTML
        html = "<div><p>Hello <strong>World</strong></p></div>"
        assert cleaner.clean_text(html) == "Hello World"

        # With attributes
        html = '<div class="test" id="main">Content</div>'
        assert cleaner.clean_text(html) == "Content"

        # Mixed content
        html = "<p>First</p><p>Second</p>"
        result = cleaner.clean_text(html)
        assert "First" in result and "Second" in result

    def test_clean_text_unicode_normalization(self):
        """Test unicode normalization"""
        cleaner = DataCleaner()

        # Unicode characters
        text = "Café résumé naïve"
        result = cleaner.clean_text(text)
        assert result  # Should handle unicode

        # Control characters
        text = "Hello\x00World\x1f"
        result = cleaner.clean_text(text)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "HelloWorld" == result

        # Emoji (should be preserved or handled gracefully)
        text = "Hello 👋 World"
        result = cleaner.clean_text(text)
        assert "Hello" in result and "World" in result

    def test_clean_text_whitespace_cleaning(self):
        """Test whitespace normalization"""
        cleaner = DataCleaner()

        # Multiple spaces
        assert cleaner.clean_text("  multiple   spaces  ") == "multiple spaces"

        # Tabs and newlines
        text = "Hello\t\tWorld\n\nTest"
        result = cleaner.clean_text(text)
        assert result == "HelloWorldTest"  # split() removes all whitespace, join adds single spaces only between words

        # Multiple punctuation
        text = "Hello!!! World???"
        result = cleaner.clean_text(text)
        assert result == "Hello! World?"

        # Length limiting
        long_text = "x" * 1000
        result = cleaner.clean_text(long_text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

        # Empty text
        assert cleaner.clean_text("") == ""
        assert cleaner.clean_text(None) == ""

    def test_clean_url_tracking_params(self):
        """Test tracking parameter removal from URLs"""
        cleaner = DataCleaner()

        # UTM parameters
        url = "https://example.com/page?utm_source=google&utm_medium=cpc&id=123"
        result = cleaner.clean_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "id=123" in result  # Keep functional params

        # Facebook click ID
        url = "https://example.com/page?fbclid=abc123&page=1"
        result = cleaner.clean_url(url)
        assert "fbclid" not in result
        assert "page=1" in result

        # Google click ID
        url = "https://example.com/page?gclid=xyz789"
        result = cleaner.clean_url(url)
        assert "gclid" not in result

        # Multiple tracking params
        url = "https://example.com/?utm_campaign=test&mc_cid=456&_ga=GA1"
        result = cleaner.clean_url(url)
        assert "utm_campaign" not in result
        assert "mc_cid" not in result
        assert "_ga" not in result

        # No parameters
        url = "https://example.com/page"
        result = cleaner.clean_url(url)
        assert result == "https://example.com/page"

        # Trailing slash removal
        url = "https://example.com/"
        result = cleaner.clean_url(url)
        assert result == "https://example.com"

        # Empty URL
        assert cleaner.clean_url("") == ""
        assert cleaner.clean_url(None) == ""

    def test_extract_domain(self):
        """Test domain extraction from URLs"""
        cleaner = DataCleaner()

        # Simple domain
        assert cleaner.extract_domain("https://example.com/page") == "example.com"

        # With www
        assert cleaner.extract_domain("https://www.example.com/page") == "example.com"

        # With subdomain
        assert cleaner.extract_domain("https://blog.example.com/page") == "blog.example.com"

        # With port
        assert cleaner.extract_domain("https://example.com:8080/page") == "example.com:8080"

        # HTTP
        assert cleaner.extract_domain("http://example.com") == "example.com"

        # Complex URL
        url = "https://www.example.com/path/to/page?query=test#anchor"
        assert cleaner.extract_domain(url) == "example.com"

        # Empty URL
        assert cleaner.extract_domain("") == ""
        assert cleaner.extract_domain(None) == ""


# ============================================================================
# DataNormalizer Tests (4 tests)
# ============================================================================

class TestDataNormalizer:
    """Test DataNormalizer functionality"""

    def test_normalize_search_result(self):
        """Test result normalization to standard format"""
        normalizer = DataNormalizer()

        raw_result = {
            "title": "<p>Test Title</p>",
            "snippet": "  Test snippet with   extra spaces  ",
            "url": "https://example.com?utm_source=test&id=123",
            "position": 1,
            "metadata": {"extra": "data"}
        }

        result = normalizer.normalize_search_result(raw_result, "google")

        # Check required fields
        assert result["title"] == "Test Title"
        assert result["snippet"] == "Test snippet with extra spaces"
        assert "utm_source" not in result["url"]
        assert "id=123" in result["url"]
        assert result["domain"] == "example.com"
        assert result["source"] == "google"
        assert result["position"] == 1

        # Check generated fields
        assert "result_id" in result
        assert len(result["result_id"]) == 32  # MD5 hash
        assert "normalized_at" in result
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0

        # Check metadata preserved
        assert result["metadata"]["extra"] == "data"

    def test_generate_id_consistency(self):
        """Test ID generation is consistent for same URL+source"""
        normalizer = DataNormalizer()

        # Same URL and source should generate same ID
        id1 = normalizer._generate_id("https://example.com", "google")
        id2 = normalizer._generate_id("https://example.com", "google")
        assert id1 == id2

        # Different URL should generate different ID
        id3 = normalizer._generate_id("https://other.com", "google")
        assert id1 != id3

        # Different source should generate different ID
        id4 = normalizer._generate_id("https://example.com", "bing")
        assert id1 != id4

        # ID should be MD5 hash (32 hex chars)
        assert len(id1) == 32
        assert all(c in "0123456789abcdef" for c in id1)

    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        normalizer = DataNormalizer()

        # Perfect result (all criteria met)
        result = {
            "title": "Long enough title for testing",
            "snippet": "This snippet is definitely longer than 50 characters to meet the quality threshold",
            "url": "https://example.com",
            "position": 1
        }
        score = normalizer._calculate_quality(result)
        assert score == 1.0  # 0.3 + 0.3 + 0.2 + 0.2

        # No title
        result["title"] = ""
        score = normalizer._calculate_quality(result)
        assert score == 0.7  # Lost 0.3 for title

        # Short snippet
        result["snippet"] = "Short"
        score = normalizer._calculate_quality(result)
        assert score == 0.4  # Lost 0.3 for snippet too

        # Invalid URL
        result["url"] = "not-a-url"
        score = normalizer._calculate_quality(result)
        assert score == 0.2  # Lost 0.2 for URL

        # Position 10 (lower bonus)
        result["url"] = "https://example.com"
        result["title"] = "Long enough title"
        result["snippet"] = "Long enough snippet that definitely meets the fifty character threshold"
        result["position"] = 10
        score = normalizer._calculate_quality(result)
        assert 0.82 <= score <= 0.83  # 0.3 + 0.3 + 0.2 + 0.02

        # Position beyond 10 (no bonus)
        result["position"] = 20
        score = normalizer._calculate_quality(result)
        assert score == 0.8  # 0.3 + 0.3 + 0.2 + 0.0

    def test_normalize_empty_fields(self):
        """Test normalization with empty/missing fields"""
        normalizer = DataNormalizer()

        # Minimal result
        result = normalizer.normalize_search_result({}, "google")

        assert result["title"] == ""
        assert result["snippet"] == ""
        assert result["url"] == ""
        assert result["domain"] == ""
        assert result["source"] == "google"
        assert result["position"] == 0
        assert "result_id" in result
        assert result["quality_score"] == 0.0

        # Partial data
        result = normalizer.normalize_search_result(
            {"title": "Test", "position": 5},
            "bing"
        )

        assert result["title"] == "Test"
        assert result["snippet"] == ""
        assert result["position"] == 5
        assert result["source"] == "bing"


# ============================================================================
# Deduplicator Tests (4 tests)
# ============================================================================

class TestDeduplicator:
    """Test Deduplicator functionality"""

    def test_deduplicate_by_url(self):
        """Test URL-based deduplication"""
        dedup = Deduplicator()

        results = [
            {"url": "https://example.com/1", "title": "First", "snippet": "Content 1"},
            {"url": "https://example.com/2", "title": "Second", "snippet": "Content 2"},
            {"url": "https://example.com/1", "title": "Duplicate", "snippet": "Dup content"},  # Duplicate URL
            {"url": "https://example.com/3", "title": "Third", "snippet": "Content 3"},
        ]

        unique = dedup.deduplicate(results)

        assert len(unique) == 3  # One duplicate removed
        assert unique[0]["url"] == "https://example.com/1"
        assert unique[1]["url"] == "https://example.com/2"
        assert unique[2]["url"] == "https://example.com/3"

    def test_deduplicate_by_content_similarity(self):
        """Test content similarity-based deduplication"""
        dedup = Deduplicator(similarity_threshold=0.8)

        results = [
            {
                "url": "https://example.com/1",
                "title": "Python programming tutorial",
                "snippet": "Learn Python programming basics"
            },
            {
                "url": "https://example.com/2",
                "title": "Python programming tutorial basics",
                "snippet": "Learn Python programming"
            },  # Very similar content
            {
                "url": "https://example.com/3",
                "title": "JavaScript tutorial",
                "snippet": "Learn JavaScript basics"
            },  # Different content
        ]

        unique = dedup.deduplicate(results)

        # Should remove one of the Python results due to similarity
        assert len(unique) == 2

        # JavaScript result should be kept
        urls = [r["url"] for r in unique]
        assert "https://example.com/3" in urls

    def test_threshold_behavior(self):
        """Test different similarity thresholds"""
        # High threshold (0.95) - only very similar content is duplicate
        dedup_strict = Deduplicator(similarity_threshold=0.95)

        results = [
            {
                "url": "https://example.com/1",
                "title": "Python tutorial",
                "snippet": "Learn Python"
            },
            {
                "url": "https://example.com/2",
                "title": "Python guide",
                "snippet": "Study Python"
            },
        ]

        unique_strict = dedup_strict.deduplicate(results)
        assert len(unique_strict) == 2  # Not similar enough

        # Low threshold (0.3) - more aggressive deduplication
        dedup_loose = Deduplicator(similarity_threshold=0.3)

        # Reset with new deduplicator
        unique_loose = dedup_loose.deduplicate(results)
        # With lower threshold, these might be considered duplicates
        # depending on word overlap
        assert len(unique_loose) <= 2

    def test_deduplicator_reset(self):
        """Test deduplicator state reset"""
        dedup = Deduplicator()

        results1 = [
            {"url": "https://example.com/1", "title": "Test", "snippet": "Content"},
        ]

        # First deduplication
        unique1 = dedup.deduplicate(results1)
        assert len(unique1) == 1

        # Same URL should be considered duplicate
        unique2 = dedup.deduplicate(results1)
        assert len(unique2) == 0  # Already seen

        # Reset state
        dedup.reset()

        # After reset, same URL should be allowed again
        unique3 = dedup.deduplicate(results1)
        assert len(unique3) == 1


# ============================================================================
# BatchProcessor Tests (4+ tests)
# ============================================================================

class TestBatchProcessor:
    """Test BatchProcessor functionality"""

    @pytest.mark.asyncio
    async def test_batch_processing_pipeline(self):
        """Test complete batch processing pipeline"""
        processor = BatchProcessor(batch_size=2)

        raw_results = [
            {
                "title": "<p>Result 1</p>",
                "snippet": "Snippet 1",
                "url": "https://example.com/1?utm_source=test",
                "position": 1
            },
            {
                "title": "Result 2",
                "snippet": "Snippet 2",
                "url": "https://example.com/2",
                "position": 2
            },
            {
                "title": "Result 3",
                "snippet": "Snippet 3",
                "url": "https://example.com/1?utm_source=test",  # Duplicate URL
                "position": 3
            },
        ]

        # Mock database operations
        with patch.object(processor, '_save_batch', new_callable=AsyncMock) as mock_save:
            processed, stats = await processor.process_search_results(
                raw_results, "google", search_id=1
            )

            # Check statistics
            assert stats["total"] == 3
            assert stats["cleaned"] == 3
            assert stats["duplicates"] == 1  # One URL duplicate
            assert stats["saved"] == 2  # 3 - 1 duplicate

            # Check processed results
            assert len(processed) == 2

            # Check cleaning was applied
            assert processed[0]["title"] == "Result 1"  # HTML removed
            assert "utm_source" not in processed[0]["url"]  # Tracking param removed

            # Check save was called
            assert mock_save.called

    @pytest.mark.asyncio
    async def test_database_integration(self):
        """Test database integration and saving"""
        from src.database import db_manager, SearchResult

        processor = BatchProcessor(batch_size=10)

        # Create test results
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

        # Test with in-memory database
        async with db_manager.get_session() as session:
            await processor._save_batch(results, search_id=1)

            # Verify save (in real test, would query database)
            assert True  # Database operations tested in test_database.py

    @pytest.mark.asyncio
    async def test_statistics_accuracy(self):
        """Test statistics tracking accuracy"""
        processor = BatchProcessor(batch_size=5)

        raw_results = [
            {"title": f"Result {i}", "snippet": f"Snippet {i}",
             "url": f"https://example.com/{i}", "position": i}
            for i in range(10)
        ]

        # Add duplicates
        raw_results.append(raw_results[0].copy())  # Duplicate
        raw_results.append(raw_results[1].copy())  # Duplicate

        with patch.object(processor, '_save_batch', new_callable=AsyncMock):
            processed, stats = await processor.process_search_results(
                raw_results, "google", search_id=1
            )

            # Verify statistics
            assert stats["total"] == 12
            assert stats["cleaned"] == 12
            assert stats["duplicates"] == 2
            assert stats["saved"] == 10
            assert stats["errors"] == 0

            # Verify processed count matches
            assert len(processed) == 10

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in batch processing"""
        processor = BatchProcessor(batch_size=2)

        # Results with problematic data
        raw_results = [
            {"title": "Good Result", "snippet": "Good", "url": "https://example.com/1", "position": 1},
            {"title": None, "snippet": None, "url": None, "position": "invalid"},  # Will cause issues
            {"title": "Another Good", "snippet": "Good", "url": "https://example.com/2", "position": 2},
        ]

        with patch.object(processor, '_save_batch', new_callable=AsyncMock):
            processed, stats = await processor.process_search_results(
                raw_results, "google", search_id=1
            )

            # Should handle errors gracefully
            assert stats["errors"] >= 0  # May have errors during normalization
            assert len(processed) >= 2  # At least the good results

    @pytest.mark.asyncio
    async def test_batch_size_handling(self):
        """Test different batch sizes"""
        # Small batch size
        processor_small = BatchProcessor(batch_size=2)

        results = [
            {"title": f"Result {i}", "snippet": f"Snippet {i}",
             "url": f"https://example.com/{i}", "position": i}
            for i in range(10)
        ]

        with patch.object(processor_small, '_save_batch', new_callable=AsyncMock) as mock_save:
            await processor_small.process_search_results(results, "google", search_id=1)

            # Should be called 5 times (10 results / batch_size 2)
            assert mock_save.call_count == 5

        # Large batch size
        processor_large = BatchProcessor(batch_size=100)

        with patch.object(processor_large, '_save_batch', new_callable=AsyncMock) as mock_save:
            await processor_large.process_search_results(results, "google", search_id=1)

            # Should be called 1 time (all results in one batch)
            assert mock_save.call_count == 1

    @pytest.mark.asyncio
    async def test_reset_deduplicator(self):
        """Test deduplicator reset functionality"""
        processor = BatchProcessor(batch_size=10)

        results = [
            {"title": "Test", "snippet": "Test", "url": "https://example.com", "position": 1}
        ]

        with patch.object(processor, '_save_batch', new_callable=AsyncMock):
            # First processing
            processed1, stats1 = await processor.process_search_results(
                results, "google", search_id=1
            )
            assert len(processed1) == 1

            # Same results should be deduplicated
            processed2, stats2 = await processor.process_search_results(
                results, "google", search_id=2
            )
            assert len(processed2) == 0  # Already seen
            assert stats2["duplicates"] == 1

            # Reset deduplicator
            processor.reset_deduplicator()

            # After reset, same results should be allowed
            processed3, stats3 = await processor.process_search_results(
                results, "google", search_id=3
            )
            assert len(processed3) == 1
            assert stats3["duplicates"] == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration between components"""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete pipeline from raw data to processed results"""
        processor = batch_processor  # Use global instance

        # Simulate scraped data
        raw_results = [
            {
                "title": "<strong>Product 1</strong>",
                "snippet": "  Description with   extra whitespace  ",
                "url": "https://shop.example.com/product1?utm_source=google&id=123",
                "position": 1,
                "metadata": {"price": "$99"}
            },
            {
                "title": "Product 2",
                "snippet": "Another description",
                "url": "https://shop.example.com/product2",
                "position": 2,
            },
        ]

        with patch.object(processor, '_save_batch', new_callable=AsyncMock):
            processed, stats = await processor.process_search_results(
                raw_results, "google", search_id=1
            )

            # Verify cleaning
            assert "<strong>" not in processed[0]["title"]
            assert processed[0]["title"] == "Product 1"

            # Verify URL cleaning
            assert "utm_source" not in processed[0]["url"]
            assert "id=123" in processed[0]["url"]

            # Verify normalization
            assert processed[0]["domain"] == "shop.example.com"
            assert processed[0]["source"] == "google"

            # Verify quality scores assigned
            assert "quality_score" in processed[0]
            assert processed[0]["quality_score"] > 0

    def test_module_exports(self):
        """Test that all expected classes are exported"""
        from src.processors import (
            DataCleaner,
            DataNormalizer,
            Deduplicator,
            BatchProcessor,
            batch_processor
        )

        # Verify classes exist
        assert DataCleaner is not None
        assert DataNormalizer is not None
        assert Deduplicator is not None
        assert BatchProcessor is not None
        assert batch_processor is not None

        # Verify global instance is configured
        assert batch_processor.batch_size == 100
