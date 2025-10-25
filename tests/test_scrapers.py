"""
Comprehensive tests for CIAP Web Scraping System

Tests all scrapers, manager, and integrations with extensive mocking.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import httpx

from src.scrapers.base import BaseScraper, ScraperException, BlockedException, RateLimitException
from src.scrapers.google import GoogleScraper
from src.scrapers.bing import BingScraper
from src.scrapers.manager import ScraperManager


class TestScraper(BaseScraper):
    """Test implementation of base scraper"""

    async def scrape(self, query, max_results=10, **kwargs):
        """Implement abstract scrape method"""
        return [
            {
                "title": f"Result {i}",
                "snippet": f"Snippet for {query}",
                "url": f"https://example.com/{i}",
                "position": i
            }
            for i in range(1, min(max_results + 1, 11))
        ]

    def parse_results(self, soup, max_results):
        """Implement abstract parse_results method"""
        return []


# Test 1: Header rotation
@pytest.mark.asyncio
async def test_base_scraper_headers():
    """Test header rotation functionality"""
    scraper = TestScraper()

    headers_set = set()
    for _ in range(10):
        headers = scraper.get_random_headers()
        headers_set.add(headers["User-Agent"])
        # Verify required headers
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "Accept-Language" in headers

    # Should have variety in user agents (at least 2 different)
    assert len(headers_set) >= 2


# Test 2: Rate limiting
@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting enforcement"""
    scraper = TestScraper()

    with patch("src.scrapers.base.db_manager") as mock_db:
        # Mock session
        mock_session = AsyncMock()
        mock_db.get_session.return_value.__aenter__.return_value = mock_session

        # Mock no previous rate limit record
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # First request should proceed immediately
        start_time = asyncio.get_event_loop().time()
        await scraper.check_rate_limit()
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should be nearly instant (< 0.1s)
        assert elapsed < 0.1


# Test 3: Request retry with exponential backoff
@pytest.mark.asyncio
async def test_request_retry():
    """Test request retry logic with exponential backoff"""
    scraper = TestScraper()

    # Mock rate limiting to skip delays
    with patch.object(scraper, 'check_rate_limit', return_value=True):
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html></html>"
            mock_response.raise_for_status = Mock()

            # First two attempts fail, third succeeds
            mock_request = AsyncMock(
                side_effect=[
                    httpx.TimeoutException("Timeout"),
                    httpx.ConnectError("Connection failed"),
                    mock_response
                ]
            )
            mock_client.return_value.__aenter__.return_value.request = mock_request

            # Should succeed after retries
            response = await scraper.make_request("http://example.com")
            assert response.status_code == 200

            # Should have attempted 3 times
            assert mock_request.call_count == 3


# Test 4: Google scraper parsing
@pytest.mark.asyncio
async def test_google_scraper_parsing():
    """Test Google result parsing"""
    scraper = GoogleScraper()

    html = """
    <div class="g">
        <a href="https://example.com/1"><h3>Result Title 1</h3></a>
        <div class="VwiC3b">This is snippet 1</div>
    </div>
    <div class="g">
        <a href="https://example.com/2"><h3>Result Title 2</h3></a>
        <div class="VwiC3b">This is snippet 2</div>
    </div>
    <div class="g">
        <a href="/url?q=https://example.com/3"><h3>Result Title 3</h3></a>
        <div class="VwiC3b">This is snippet 3</div>
    </div>
    """

    soup = scraper.parse_html(html)
    results = scraper.parse_results(soup, 10)

    assert len(results) == 3
    assert results[0]["title"] == "Result Title 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[0]["snippet"] == "This is snippet 1"
    assert results[0]["position"] == 1

    # Test Google redirect URL parsing
    assert results[2]["url"] == "https://example.com/3"


# Test 5: Google CAPTCHA detection
@pytest.mark.asyncio
async def test_google_scraper_blocked():
    """Test Google CAPTCHA detection"""
    scraper = GoogleScraper()

    html = '<div id="recaptcha">CAPTCHA</div>'
    soup = scraper.parse_html(html)

    with pytest.raises(BlockedException):
        scraper.parse_results(soup, 10)


# Test 6: Bing scraper parsing
@pytest.mark.asyncio
async def test_bing_scraper_parsing():
    """Test Bing result parsing"""
    scraper = BingScraper()

    html = """
    <li class="b_algo">
        <h2><a href="https://example.com/1">Bing Result 1</a></h2>
        <div class="b_caption">
            <p>Bing snippet 1</p>
        </div>
    </li>
    <li class="b_algo">
        <h2><a href="https://example.com/2">Bing Result 2</a></h2>
        <div class="b_caption">
            <p>Bing snippet 2</p>
        </div>
        <span class="news_dt">2 hours ago</span>
    </li>
    """

    soup = scraper.parse_html(html)
    results = scraper.parse_results(soup, 10)

    assert len(results) == 2
    assert results[0]["title"] == "Bing Result 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[0]["snippet"] == "Bing snippet 1"

    # Check metadata extraction
    assert results[1]["metadata"]["date"] == "2 hours ago"


# Test 7: Scraper manager parallel scraping
@pytest.mark.asyncio
async def test_scraper_manager_parallel():
    """Test parallel scraping from multiple sources"""
    manager = ScraperManager()

    # Mock scrapers
    mock_google = AsyncMock()
    mock_google.name = "GoogleScraper"
    mock_google.scrape.return_value = [
        {"title": "Google 1", "url": "http://g1.com", "snippet": "G snippet", "position": 1}
    ]

    mock_bing = AsyncMock()
    mock_bing.name = "BingScraper"
    mock_bing.scrape.return_value = [
        {"title": "Bing 1", "url": "http://b1.com", "snippet": "B snippet", "position": 1}
    ]

    manager.scrapers = {
        "google": mock_google,
        "bing": mock_bing
    }

    results = await manager.scrape("test query", ["google", "bing"])

    assert "google" in results
    assert "bing" in results
    assert len(results["google"]) == 1
    assert len(results["bing"]) == 1

    # Both should be called
    mock_google.scrape.assert_called_once()
    mock_bing.scrape.assert_called_once()


# Test 8: Scraper manager error handling
@pytest.mark.asyncio
async def test_scraper_manager_error_handling():
    """Test error handling in manager (graceful degradation)"""
    manager = ScraperManager()

    # Mock one scraper failing
    mock_google = AsyncMock()
    mock_google.name = "GoogleScraper"
    mock_google.scrape.side_effect = ScraperException("Google failed")

    mock_bing = AsyncMock()
    mock_bing.name = "BingScraper"
    mock_bing.scrape.return_value = [{"title": "Bing works", "url": "http://b.com", "snippet": "B", "position": 1}]

    manager.scrapers = {
        "google": mock_google,
        "bing": mock_bing
    }

    results = await manager.scrape("test", ["google", "bing"])

    # Bing should still work
    assert results["bing"] == [{"title": "Bing works", "url": "http://b.com", "snippet": "B", "position": 1}]
    # Google should return empty
    assert results["google"] == []


# Test 9: Cache integration
@pytest.mark.asyncio
async def test_cache_integration():
    """Test cache integration with scrapers"""
    scraper = GoogleScraper()

    with patch("src.scrapers.google.cache") as mock_cache:
        # First call - cache miss
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        with patch.object(scraper, "make_request") as mock_request:
            mock_response = Mock()
            mock_response.text = "<html></html>"
            mock_request.return_value = mock_response

            await scraper.scrape("test query", max_results=10)

            # Should make request
            mock_request.assert_called()

            # Should set cache
            mock_cache.set.assert_called()

        # Second call - cache hit
        mock_cache.get = AsyncMock(return_value=[{"title": "Cached", "url": "http://c.com"}])

        with patch.object(scraper, "make_request") as mock_request:
            results = await scraper.scrape("test query", max_results=10)

            # Should NOT make request (cache hit)
            mock_request.assert_not_called()

            # Should return cached results
            assert results == [{"title": "Cached", "url": "http://c.com"}]


# Test 10: Text cleaning
@pytest.mark.asyncio
async def test_text_cleaning():
    """Test text cleaning and normalization"""
    scraper = TestScraper()

    # Test whitespace cleaning
    assert scraper.clean_text("  text  with   spaces  ") == "text with spaces"

    # Test newline removal
    assert scraper.clean_text("text\nwith\nnewlines") == "text with newlines"

    # Test tab removal
    assert scraper.clean_text("text\twith\ttabs") == "text with tabs"

    # Test length limiting
    long_text = "x" * 1000
    cleaned = scraper.clean_text(long_text)
    assert len(cleaned) == 500
    assert cleaned.endswith("...")

    # Test empty string
    assert scraper.clean_text("") == ""
    assert scraper.clean_text(None) == ""


# Test 11: URL normalization
@pytest.mark.asyncio
async def test_url_normalization():
    """Test URL normalization and cleaning"""
    scraper = TestScraper()

    # Test absolute URL
    assert scraper.normalize_url(
        "https://example.com/page"
    ) == "https://example.com/page"

    # Test relative URL with base
    assert scraper.normalize_url(
        "/page",
        "https://example.com"
    ) == "https://example.com/page"

    # Test tracking parameter removal
    assert scraper.normalize_url(
        "https://example.com/page?utm_source=google&utm_medium=cpc"
    ) == "https://example.com/page"

    # Test empty URL
    assert scraper.normalize_url("") == ""
    assert scraper.normalize_url(None) == ""


# Test 12: Result validation
@pytest.mark.asyncio
async def test_result_validation():
    """Test result validation and filtering"""
    scraper = TestScraper()

    raw_results = [
        {"url": "https://example.com/1", "title": "  Title 1  "},
        {"url": "", "title": "No URL"},  # Should be filtered out
        {"url": "https://example.com/2"},  # No title - should get default
        {"url": "https://example.com/3", "title": "Title 3", "snippet": "  Extra  spaces  "},
    ]

    validated = await scraper.validate_results(raw_results)

    # Should have 3 results (one filtered out)
    assert len(validated) == 3

    # Check cleaning
    assert validated[0]["title"] == "Title 1"
    assert validated[1]["title"] == "No title"
    assert validated[2]["snippet"] == "Extra spaces"

    # Check defaults
    assert validated[0]["position"] == 1
    assert validated[1]["position"] == 2
    assert validated[0]["source"] == "testscraper"


# Test 13: Database integration
@pytest.mark.asyncio
async def test_database_integration():
    """Test saving to Search and SearchResult tables"""
    manager = ScraperManager()

    # Mock scrape method
    async def mock_scrape(*args, **kwargs):
        return {
            "google": [
                {"title": "Result 1", "url": "http://r1.com", "snippet": "S1", "position": 1},
                {"title": "Result 2", "url": "http://r2.com", "snippet": "S2", "position": 2}
            ]
        }

    with patch.object(manager, 'scrape', side_effect=mock_scrape):
        with patch("src.scrapers.manager.db_manager") as mock_db:
            # Mock session
            mock_session = AsyncMock()
            mock_db.get_session.return_value.__aenter__.return_value = mock_session

            # Mock Search object
            mock_search = Mock()
            mock_search.id = 123
            mock_session.get.return_value = mock_search

            # Run scrape_and_save
            result = await manager.scrape_and_save(
                search_id=123,
                query="test query",
                sources=["google"]
            )

            # Check result
            assert result["search_id"] == 123
            assert result["total_results"] == 2
            assert result["status"] == "completed"

            # Check search status was updated
            assert mock_search.status == "completed"
            assert mock_search.completed_at is not None


# Test 14: Task queue integration
@pytest.mark.asyncio
async def test_task_queue_integration():
    """Test task queue handler integration"""
    from src.task_queue.handlers import scrape_handler

    with patch("src.scrapers.manager.scraper_manager") as mock_manager:
        # Mock scrape_and_save as async
        mock_manager.scrape_and_save = AsyncMock(return_value={
            "search_id": 456,
            "query": "test",
            "total_results": 5,
            "status": "completed"
        })

        # Test with search_id
        result = await scrape_handler({
            "query": "test query",
            "sources": ["google"],
            "search_id": 456
        })

        assert result["status"] == "completed"
        assert result["total_results"] == 5
        mock_manager.scrape_and_save.assert_called_once()

        # Test without search_id
        mock_manager.scrape = AsyncMock(return_value={
            "google": [{"title": "R1", "url": "http://r1.com"}]
        })

        result = await scrape_handler({
            "query": "test query",
            "sources": ["google"]
        })

        assert result["status"] == "success"
        assert result["total_results"] == 1


# Test 15: Scraper statistics
@pytest.mark.asyncio
async def test_scraper_stats():
    """Test statistics tracking"""
    scraper = TestScraper()

    # Initialize stats
    assert scraper.stats["requests_made"] == 0
    assert scraper.stats["requests_failed"] == 0
    assert scraper.stats["results_scraped"] == 0

    # Mock successful scrape
    with patch.object(scraper, 'check_rate_limit', return_value=True):
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html></html>"
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )

            await scraper.make_request("http://example.com")

            # Check stats updated
            assert scraper.stats["requests_made"] == 1

    # Get stats
    stats = scraper.get_stats()
    assert stats["scraper"] == "TestScraper"
    assert stats["requests_made"] == 1
    assert stats["success_rate"] == 100.0


# Test 16: Manager statistics aggregation
@pytest.mark.asyncio
async def test_manager_stats_aggregation():
    """Test scraper manager statistics aggregation"""
    manager = ScraperManager()

    # Mock scrapers with stats
    mock_google = Mock()
    mock_google.get_stats.return_value = {
        "scraper": "GoogleScraper",
        "requests_made": 10,
        "requests_failed": 1,
        "results_scraped": 50
    }

    mock_bing = Mock()
    mock_bing.get_stats.return_value = {
        "scraper": "BingScraper",
        "requests_made": 5,
        "requests_failed": 0,
        "results_scraped": 25
    }

    manager.scrapers = {
        "google": mock_google,
        "bing": mock_bing
    }

    stats = manager.get_stats()

    # Check individual stats
    assert stats["google"]["requests_made"] == 10
    assert stats["bing"]["requests_made"] == 5

    # Check aggregate stats
    assert stats["aggregate"]["total_requests"] == 15
    assert stats["aggregate"]["total_failed"] == 1
    assert stats["aggregate"]["total_results"] == 75
    assert stats["aggregate"]["overall_success_rate"] > 90


# Test 17: Google date filtering
@pytest.mark.asyncio
async def test_google_date_filtering():
    """Test Google date range filtering"""
    scraper = GoogleScraper()

    # Test date filter URL building (URLs are percent-encoded)
    url = scraper._build_search_url("test", 0, "en", "us", "d")
    assert "tbs=qdr" in url and ("qdr:d" in url or "qdr%3Ad" in url)

    url = scraper._build_search_url("test", 0, "en", "us", "w")
    assert "tbs=qdr" in url and ("qdr:w" in url or "qdr%3Aw" in url)

    url = scraper._build_search_url("test", 0, "en", "us", "m")
    assert "tbs=qdr" in url and ("qdr:m" in url or "qdr%3Am" in url)

    url = scraper._build_search_url("test", 0, "en", "us", "y")
    assert "tbs=qdr" in url and ("qdr:y" in url or "qdr%3Ay" in url)


# Test 18: Blocking exception handling
@pytest.mark.asyncio
async def test_blocking_exception_handling():
    """Test handling of blocking (403) responses"""
    scraper = TestScraper()

    with patch.object(scraper, 'check_rate_limit', return_value=True):
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 403

            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(BlockedException):
                await scraper.make_request("http://example.com")


# Test 19: Rate limit exception handling
@pytest.mark.asyncio
async def test_rate_limit_exception_handling():
    """Test handling of rate limit (429) responses"""
    scraper = TestScraper()

    with patch.object(scraper, 'check_rate_limit', return_value=True):
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 429

            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RateLimitException):
                await scraper.make_request("http://example.com")


# Test 20: Schedule scraping task
@pytest.mark.asyncio
async def test_schedule_scraping():
    """Test scheduling scraping as background task"""
    manager = ScraperManager()

    with patch("src.scrapers.manager.db_manager") as mock_db:
        with patch("src.task_queue.task_queue") as mock_queue:
            # Mock session
            mock_session = AsyncMock()
            mock_db.get_session.return_value.__aenter__.return_value = mock_session

            # Mock Search object
            mock_search = Mock()
            mock_search.id = 789
            mock_session.add = Mock()
            mock_session.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, 'id', 789))

            # Mock task queue
            mock_queue.enqueue = AsyncMock(return_value=12345)

            # Schedule scraping
            task_id = await manager.schedule_scraping(
                "test query",
                sources=["google"],
                priority=3
            )

            assert task_id == 12345
            mock_queue.enqueue.assert_called_once()
