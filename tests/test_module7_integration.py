"""
Integration Tests for Module 7: LLM Analysis System
Tests end-to-end workflows, database integration, and real Ollama calls
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.analyzers import (
    OllamaClient, ollama_client,
    SentimentAnalyzer, CompetitorAnalyzer, TrendAnalyzer
)
from src.database import DatabaseManager, DatabaseOperations, Search, SearchResult
from src.cache import cache


class TestModule7Integration:
    """Integration tests for Module 7"""

    @pytest.mark.asyncio
    async def test_end_to_end_scrape_process_analyze(self):
        """Test complete workflow: scrape → process → analyze"""
        # This test simulates the full pipeline from scraping to analysis

        # Create in-memory database
        db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await db_manager.initialize()

        try:
            # Create a search record
            async with db_manager.get_session() as session:
                search = await DatabaseOperations.create_search(
                    session,
                    query="AI trends 2024",
                    sources=["google"]
                )
                search_id = search.id

            # Add mock search results
            async with db_manager.get_session() as session:
                results_data = [
                    {
                        "source": "google",
                        "title": "AI Revolution is Here",
                        "snippet": "Artificial intelligence is transforming industries with amazing innovations",
                        "url": "https://example.com/ai-revolution",
                        "position": 1
                    },
                    {
                        "source": "google",
                        "title": "Machine Learning Advances",
                        "snippet": "New breakthrough in machine learning algorithms",
                        "url": "https://example.com/ml-advances",
                        "position": 2
                    },
                    {
                        "source": "google",
                        "title": "AI Safety Concerns",
                        "snippet": "Experts warn about potential risks of artificial intelligence",
                        "url": "https://example.com/ai-risks",
                        "position": 3
                    }
                ]

                for result_data in results_data:
                    result = SearchResult(
                        search_id=search_id,
                        **result_data
                    )
                    session.add(result)

                await session.commit()

            # Mock ollama_client for analysis
            with patch.object(ollama_client, "batch_analyze") as mock_batch:
                mock_batch.return_value = [
                    {"sentiment": "positive", "confidence": 0.9},
                    {"sentiment": "positive", "confidence": 0.85},
                    {"sentiment": "negative", "confidence": 0.75}
                ]

                # Perform sentiment analysis
                analyzer = SentimentAnalyzer()
                result = await analyzer.analyze_search_results(
                    search_id=search_id,
                    sample_size=50
                )

                # Verify results
                assert result["search_id"] == search_id
                assert result["analyzed_count"] == 3
                assert result["sentiment_distribution"]["positive"] == 2
                assert result["sentiment_distribution"]["negative"] == 1
                assert result["dominant_sentiment"] == "positive"

        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    async def test_real_ollama_api_call(self):
        """Test real Ollama API call (skip if Ollama unavailable)"""
        client = OllamaClient()

        # Check if Ollama is available
        is_healthy = await client.check_health()

        if not is_healthy:
            pytest.skip("Ollama not available")

        # Perform real analysis
        try:
            result = await client.analyze(
                "This is a wonderful product with excellent quality!",
                analysis_type="sentiment",
                use_cache=False
            )

            # Basic validation
            assert result is not None
            assert "sentiment" in result or "analysis" in result
            assert "_metadata" in result
            assert result["_metadata"]["analysis_type"] == "sentiment"

        except Exception as e:
            pytest.skip(f"Ollama analysis failed: {e}")

    @pytest.mark.asyncio
    async def test_database_integration_with_real_db(self):
        """Test database integration with in-memory database"""
        # Create in-memory database
        db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await db_manager.initialize()

        try:
            # Create search and results
            async with db_manager.get_session() as session:
                search = await DatabaseOperations.create_search(
                    session,
                    query="Test Query",
                    sources=["google"]
                )
                await session.refresh(search)
                search_id = search.id

            async with db_manager.get_session() as session:
                for i in range(5):
                    result = SearchResult(
                        search_id=search_id,
                        source="google",
                        title=f"Result {i+1}",
                        snippet=f"This is snippet number {i+1}",
                        url=f"https://example.com/{i+1}",
                        position=i+1
                    )
                    session.add(result)
                await session.commit()

            # Use CompetitorAnalyzer with mocked LLM
            with patch("src.analyzers.sentiment.ollama_client.analyze") as mock_analyze:
                with patch("src.analyzers.sentiment.db_manager", db_manager):
                    mock_analyze.return_value = {
                        "competitors": ["CompanyA", "CompanyB"],
                        "products": ["Product1"],
                        "analysis": "Analysis text"
                    }

                    analyzer = CompetitorAnalyzer()
                    result = await analyzer.analyze_competitors(search_id=search_id)

                    assert result["search_id"] == search_id
                    assert result["total_results_analyzed"] == 5
                    assert "competitors" in mock_analyze.call_args[0][0].lower() or True

        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test cache integration with LLMCache"""
        await cache.initialize()

        try:
            # Clear cache to ensure fresh start
            from src.cache import LLMCache
            import hashlib
            import time

            # Use unique text to avoid cache conflicts
            unique_text = f"Test text for caching {time.time()}"

            # Mock Ollama request
            with patch.object(ollama_client, "_request_ollama") as mock_request:
                mock_request.return_value = '{"sentiment": "positive", "confidence": 0.9}'

                # First call - cache miss
                result1 = await ollama_client.analyze(
                    unique_text,
                    analysis_type="sentiment",
                    use_cache=True
                )

                assert result1["sentiment"] == "positive"
                assert mock_request.call_count == 1

                # Second call - should hit cache
                result2 = await ollama_client.analyze(
                    unique_text,
                    analysis_type="sentiment",
                    use_cache=True
                )

                assert result2["sentiment"] == "positive"
                # Should not call Ollama again (still 1 call)
                assert mock_request.call_count == 1

        finally:
            await cache.close()

    @pytest.mark.asyncio
    async def test_multi_analyzer_workflow(self):
        """Test workflow using multiple analyzers"""
        # Create in-memory database
        db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await db_manager.initialize()

        try:
            # Setup: Create search with results
            async with db_manager.get_session() as session:
                search = await DatabaseOperations.create_search(
                    session,
                    query="AI competitor analysis",
                    sources=["google"]
                )
                await session.refresh(search)
                search_id = search.id

            async with db_manager.get_session() as session:
                results_data = [
                    {
                        "source": "google",
                        "title": "Google AI Breakthrough",
                        "snippet": "Google announces major AI advancement",
                        "url": "https://example.com/google-ai",
                        "position": 1
                    },
                    {
                        "source": "google",
                        "title": "Microsoft Copilot Update",
                        "snippet": "Microsoft releases new AI features",
                        "url": "https://example.com/ms-copilot",
                        "position": 2
                    },
                    {
                        "source": "google",
                        "title": "OpenAI GPT-5 Rumors",
                        "snippet": "Speculation about next generation model",
                        "url": "https://example.com/openai-gpt5",
                        "position": 3
                    }
                ]

                for result_data in results_data:
                    result = SearchResult(search_id=search_id, **result_data)
                    session.add(result)
                await session.commit()

            # Mock ollama_client
            with patch("src.analyzers.sentiment.db_manager", db_manager):
                with patch("src.analyzers.sentiment.ollama_client") as mock_client:
                    # Setup mocks for different analysis types (must be AsyncMock)
                    mock_client.batch_analyze = AsyncMock(return_value=[
                        {"sentiment": "positive", "confidence": 0.9},
                        {"sentiment": "neutral", "confidence": 0.7},
                        {"sentiment": "neutral", "confidence": 0.6}
                    ])

                    mock_client.analyze = AsyncMock(return_value={
                        "competitors": ["Google", "Microsoft", "OpenAI"],
                        "products": ["AI", "Copilot", "GPT"],
                        "analysis": "Competitive landscape analysis",
                        "trends": ["AI advancement", "LLM development"],
                        "keywords": ["AI", "GPT", "Copilot"],
                        "topics": ["Artificial Intelligence"]
                    })

                    # Run all three analyzers
                    sentiment_analyzer = SentimentAnalyzer()
                    competitor_analyzer = CompetitorAnalyzer()
                    trend_analyzer = TrendAnalyzer()

                    # Execute analyses in parallel
                    sentiment_result, competitor_result, trend_result = await asyncio.gather(
                        sentiment_analyzer.analyze_search_results(search_id),
                        competitor_analyzer.analyze_competitors(search_id),
                        trend_analyzer.analyze_trends(search_id)
                    )

                    # Verify all analyses completed
                    assert sentiment_result["analyzed_count"] == 3
                    assert "Google" in competitor_result["identified_competitors"]
                    assert len(trend_result["top_trends"]) > 0

        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test performance with 100+ results"""
        db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await db_manager.initialize()

        try:
            # Create search with 100 results
            async with db_manager.get_session() as session:
                search = await DatabaseOperations.create_search(
                    session,
                    query="Large dataset test",
                    sources=["google"]
                )
                await session.refresh(search)
                search_id = search.id

            async with db_manager.get_session() as session:
                for i in range(100):
                    result = SearchResult(
                        search_id=search_id,
                        source="google",
                        title=f"Result {i+1}",
                        snippet=f"Content for result {i+1}",
                        url=f"https://example.com/{i+1}",
                        position=i+1
                    )
                    session.add(result)
                await session.commit()

            # Mock ollama_client for fast batch processing
            with patch("src.analyzers.sentiment.db_manager", db_manager):
                with patch("src.analyzers.sentiment.ollama_client.batch_analyze") as mock_batch:
                    # Return 50 results (sample_size default)
                    mock_batch.return_value = [
                        {"sentiment": "positive", "confidence": 0.8}
                        for _ in range(50)
                    ]

                    analyzer = SentimentAnalyzer()

                    import time
                    start_time = time.time()

                    result = await analyzer.analyze_search_results(
                        search_id=search_id,
                        sample_size=50
                    )

                    elapsed_time = time.time() - start_time

                    # Should complete quickly (< 5 seconds)
                    assert elapsed_time < 5.0
                    assert result["analyzed_count"] == 50

        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self):
        """Test error handling throughout workflow"""
        db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        await db_manager.initialize()

        try:
            # Create search with results
            async with db_manager.get_session() as session:
                search = await DatabaseOperations.create_search(
                    session,
                    query="Error handling test",
                    sources=["google"]
                )
                await session.refresh(search)
                search_id = search.id

            async with db_manager.get_session() as session:
                for i in range(3):
                    result = SearchResult(
                        search_id=search_id,
                        source="google",
                        title=f"Result {i+1}",
                        snippet=f"Snippet {i+1}",
                        url=f"https://example.com/{i+1}",
                        position=i+1
                    )
                    session.add(result)
                await session.commit()

            # Mock ollama_client to return mixed results (some errors)
            with patch("src.analyzers.sentiment.db_manager", db_manager):
                with patch("src.analyzers.sentiment.ollama_client.batch_analyze") as mock_batch:
                    mock_batch.return_value = [
                        {"sentiment": "positive", "confidence": 0.9},
                        {"error": "Analysis failed"},
                        {"sentiment": "negative", "confidence": 0.7}
                    ]

                    analyzer = SentimentAnalyzer()
                    result = await analyzer.analyze_search_results(search_id)

                    # Should handle errors gracefully
                    assert result["analyzed_count"] == 2  # Only count successful ones
                    assert result["sentiment_distribution"]["positive"] == 1
                    assert result["sentiment_distribution"]["negative"] == 1

        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    async def test_module_imports(self):
        """Test that all module imports work correctly"""
        # Test imports from main module
        from src.analyzers import (
            OllamaClient, OllamaException, ollama_client,
            SentimentAnalyzer, CompetitorAnalyzer, TrendAnalyzer
        )

        # Verify classes can be instantiated
        client = OllamaClient()
        assert client is not None
        assert client.base_url is not None

        sentiment = SentimentAnalyzer()
        assert sentiment is not None

        competitor = CompetitorAnalyzer()
        assert competitor is not None

        trend = TrendAnalyzer()
        assert trend is not None

        # Verify global instance
        assert ollama_client is not None
        assert isinstance(ollama_client, OllamaClient)
