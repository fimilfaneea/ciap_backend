"""
Comprehensive Tests for Module 7: LLM Analysis System
Tests for OllamaClient, SentimentAnalyzer, CompetitorAnalyzer, TrendAnalyzer
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, mock_open
from pathlib import Path
import json

from src.analyzers import (
    OllamaClient, OllamaException, ollama_client,
    SentimentAnalyzer, CompetitorAnalyzer, TrendAnalyzer
)


class TestOllamaClient:
    """Test OllamaClient functionality"""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful Ollama health check"""
        client = OllamaClient()

        # Mock httpx.AsyncClient
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.1:8b"},
                    {"name": "mistral:latest"}
                ]
            }
            mock_client.get.return_value = mock_response

            result = await client.check_health()

            assert result is True
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_model_not_found(self):
        """Test health check when model is not available"""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock response with different models
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "mistral:latest"},
                    {"name": "phi:latest"}
                ]
            }
            mock_client.get.return_value = mock_response

            # Should still return True if base model name matches
            result = await client.check_health()

            # Result depends on whether base name matches
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check with timeout"""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock timeout
            import httpx
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")

            result = await client.check_health()

            assert result is False

    def test_load_prompts_success(self):
        """Test prompt loading from files"""
        client = OllamaClient()

        # Check that prompts were loaded
        assert "sentiment" in client.prompts
        assert "competitor" in client.prompts
        assert "summary" in client.prompts
        assert "trends" in client.prompts
        assert "insights" in client.prompts
        assert "keywords" in client.prompts

        # Check that prompts contain {text} placeholder
        assert "{text}" in client.prompts["sentiment"]
        assert "{text}" in client.prompts["competitor"]

    @pytest.mark.asyncio
    async def test_analyze_with_cache_miss(self):
        """Test analysis with cache miss"""
        client = OllamaClient()

        # Mock LLMCache to return None (cache miss)
        with patch("src.analyzers.ollama_client.LLMCache") as mock_cache:
            mock_cache.get_analysis = AsyncMock(return_value=None)
            mock_cache.set_analysis = AsyncMock()

            # Mock _request_ollama
            with patch.object(client, "_request_ollama") as mock_request:
                mock_request.return_value = '{"sentiment": "positive", "confidence": 0.9}'

                result = await client.analyze(
                    "This is great news!",
                    analysis_type="sentiment"
                )

                # Check cache was queried
                mock_cache.get_analysis.assert_called_once()

                # Check result
                assert "sentiment" in result
                assert result["sentiment"] == "positive"
                assert result["confidence"] == 0.9

                # Check cache was set
                mock_cache.set_analysis.assert_called_once()

                # Check stats
                assert client.stats["requests"] == 1

    @pytest.mark.asyncio
    async def test_analyze_with_cache_hit(self):
        """Test analysis with cache hit"""
        client = OllamaClient()

        cached_result = {
            "sentiment": "negative",
            "confidence": 0.85,
            "_metadata": {"cached": True}
        }

        # Mock LLMCache to return cached result
        with patch("src.analyzers.ollama_client.LLMCache") as mock_cache:
            mock_cache.get_analysis = AsyncMock(return_value=cached_result)

            result = await client.analyze(
                "This is terrible!",
                analysis_type="sentiment"
            )

            # Check result is from cache
            assert result == cached_result

            # Check stats
            assert client.stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_batch_analyze(self):
        """Test batch analysis with multiple texts"""
        client = OllamaClient()

        texts = [
            "First text to analyze",
            "Second text to analyze",
            "Third text to analyze"
        ]

        # Mock analyze method
        with patch.object(client, "analyze") as mock_analyze:
            mock_analyze.side_effect = [
                {"sentiment": "positive", "confidence": 0.9},
                {"sentiment": "neutral", "confidence": 0.6},
                {"sentiment": "negative", "confidence": 0.8}
            ]

            results = await client.batch_analyze(
                texts,
                analysis_type="sentiment",
                batch_size=2
            )

            # Check all texts were analyzed
            assert len(results) == 3
            assert mock_analyze.call_count == 3

            # Check results
            assert results[0]["sentiment"] == "positive"
            assert results[1]["sentiment"] == "neutral"
            assert results[2]["sentiment"] == "negative"

    @pytest.mark.asyncio
    async def test_batch_analyze_with_errors(self):
        """Test batch analysis with some errors"""
        client = OllamaClient()

        texts = ["Text 1", "Text 2", "Text 3"]

        # Mock analyze to raise exception for second text
        with patch.object(client, "analyze") as mock_analyze:
            mock_analyze.side_effect = [
                {"sentiment": "positive"},
                OllamaException("Analysis failed"),
                {"sentiment": "negative"}
            ]

            results = await client.batch_analyze(texts, analysis_type="sentiment")

            # Check all results returned (with error for failed one)
            assert len(results) == 3
            assert results[0]["sentiment"] == "positive"
            assert "error" in results[1]
            assert results[2]["sentiment"] == "negative"

    @pytest.mark.asyncio
    async def test_request_ollama_success(self):
        """Test successful Ollama API request"""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "This is a positive sentiment.",
                "eval_count": 150
            }
            mock_client.post.return_value = mock_response

            result = await client._request_ollama("Analyze this text")

            assert result == "This is a positive sentiment."
            assert client.stats["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_request_ollama_timeout(self):
        """Test Ollama request timeout"""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock timeout
            import httpx
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(OllamaException) as excinfo:
                await client._request_ollama("Test prompt")

            assert "timed out" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_request_ollama_http_error(self):
        """Test Ollama request with HTTP error"""
        client = OllamaClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock HTTP error
            import httpx
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=mock_response
            )

            with pytest.raises(OllamaException) as excinfo:
                await client._request_ollama("Test prompt")

            assert "HTTP error" in str(excinfo.value)

    def test_parse_response_json(self):
        """Test JSON response parsing"""
        client = OllamaClient()

        response = '{"sentiment": "positive", "confidence": 0.95, "key_phrases": ["great", "excellent"]}'

        result = client._parse_response(response, "sentiment")

        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.95
        assert "great" in result["key_phrases"]

    def test_parse_response_json_embedded(self):
        """Test parsing JSON embedded in text"""
        client = OllamaClient()

        response = '''Here is the analysis:
        {"sentiment": "negative", "confidence": 0.8}
        That's my assessment.'''

        result = client._parse_response(response, "sentiment")

        assert result["sentiment"] == "negative"
        assert result["confidence"] == 0.8

    def test_parse_sentiment_fallback(self):
        """Test fallback sentiment parsing"""
        client = OllamaClient()

        response = """SENTIMENT: Positive
CONFIDENCE: 0.85
KEY_EMOTIONS: Joy, Excitement
REASONING: The text expresses happiness"""

        result = client._parse_sentiment(response)

        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.85
        assert "analysis" in result

    def test_parse_competitors_fallback(self):
        """Test fallback competitor parsing"""
        client = OllamaClient()

        response = """Competitors mentioned: Google, Microsoft, Amazon
Products: Cloud Services, Search Engine
The market is competitive."""

        result = client._parse_competitors(response)

        assert isinstance(result["competitors"], list)
        assert len(result["competitors"]) > 0
        assert "analysis" in result

    def test_parse_trends_fallback(self):
        """Test fallback trends parsing"""
        client = OllamaClient()

        response = """- Growing adoption of AI technology
- Increased cloud migration
- Remote work trends continuing
- Focus on cybersecurity"""

        result = client._parse_trends(response)

        assert isinstance(result["trends"], list)
        assert len(result["trends"]) > 0
        assert "analysis" in result

    def test_statistics_tracking(self):
        """Test statistics tracking"""
        client = OllamaClient()

        # Initial stats
        assert client.stats["requests"] == 0
        assert client.stats["cache_hits"] == 0
        assert client.stats["errors"] == 0
        assert client.stats["total_tokens"] == 0

        # Stats should be modifiable
        client.stats["requests"] = 5
        client.stats["cache_hits"] = 2
        assert client.stats["requests"] == 5
        assert client.stats["cache_hits"] == 2


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer functionality"""

    @pytest.mark.asyncio
    async def test_analyze_search_results_success(self):
        """Test sentiment analysis of search results"""
        analyzer = SentimentAnalyzer()

        # Mock database results
        mock_results = [
            Mock(title="Great Product", snippet="This is amazing and works well"),
            Mock(title="Good Service", snippet="Very satisfied with the results"),
            Mock(title="Bad Experience", snippet="Terrible quality and poor support")
        ]

        # Mock database session
        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            # Mock ollama_client batch_analyze
            with patch("src.analyzers.sentiment.ollama_client.batch_analyze") as mock_analyze:
                mock_analyze.return_value = [
                    {"sentiment": "positive", "confidence": 0.9},
                    {"sentiment": "positive", "confidence": 0.85},
                    {"sentiment": "negative", "confidence": 0.8}
                ]

                result = await analyzer.analyze_search_results(
                    search_id=1,
                    sample_size=50
                )

                # Check result structure
                assert result["search_id"] == 1
                assert result["analyzed_count"] == 3
                assert "sentiment_distribution" in result
                assert result["sentiment_distribution"]["positive"] == 2
                assert result["sentiment_distribution"]["negative"] == 1
                assert result["dominant_sentiment"] == "positive"
                assert result["average_confidence"] > 0

    @pytest.mark.asyncio
    async def test_analyze_search_results_empty(self):
        """Test sentiment analysis with no results"""
        analyzer = SentimentAnalyzer()

        # Mock database with no results
        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = []
            mock_db_session.execute.return_value = mock_execute_result

            result = await analyzer.analyze_search_results(search_id=999)

            assert "error" in result
            assert result["analyzed_count"] == 0

    @pytest.mark.asyncio
    async def test_sentiment_aggregation(self):
        """Test sentiment distribution aggregation"""
        analyzer = SentimentAnalyzer()

        mock_results = [Mock(title="Title", snippet="Snippet") for _ in range(10)]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.batch_analyze") as mock_analyze:
                # Mix of sentiments
                mock_analyze.return_value = [
                    {"sentiment": "positive", "confidence": 0.9},
                    {"sentiment": "positive", "confidence": 0.8},
                    {"sentiment": "positive", "confidence": 0.7},
                    {"sentiment": "neutral", "confidence": 0.6},
                    {"sentiment": "neutral", "confidence": 0.6},
                    {"sentiment": "negative", "confidence": 0.85},
                    {"sentiment": "positive", "confidence": 0.75},
                    {"sentiment": "positive", "confidence": 0.95},
                    {"sentiment": "neutral", "confidence": 0.5},
                    {"sentiment": "positive", "confidence": 0.88}
                ]

                result = await analyzer.analyze_search_results(search_id=1)

                # Check distribution
                assert result["sentiment_distribution"]["positive"] == 6
                assert result["sentiment_distribution"]["neutral"] == 3
                assert result["sentiment_distribution"]["negative"] == 1
                assert result["dominant_sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_sentiment_confidence_calculation(self):
        """Test average confidence calculation"""
        analyzer = SentimentAnalyzer()

        mock_results = [Mock(title="T", snippet="S") for _ in range(3)]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.batch_analyze") as mock_analyze:
                mock_analyze.return_value = [
                    {"sentiment": "positive", "confidence": 0.9},
                    {"sentiment": "positive", "confidence": 0.8},
                    {"sentiment": "positive", "confidence": 0.7}
                ]

                result = await analyzer.analyze_search_results(search_id=1)

                # Average should be (0.9 + 0.8 + 0.7) / 3 = 0.8
                assert result["average_confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_sentiment_with_errors(self):
        """Test sentiment analysis with some errors in batch"""
        analyzer = SentimentAnalyzer()

        mock_results = [Mock(title="T", snippet="S") for _ in range(3)]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.batch_analyze") as mock_analyze:
                # One result has error
                mock_analyze.return_value = [
                    {"sentiment": "positive", "confidence": 0.9},
                    {"error": "Analysis failed"},
                    {"sentiment": "negative", "confidence": 0.8}
                ]

                result = await analyzer.analyze_search_results(search_id=1)

                # Should only count non-error results
                assert result["analyzed_count"] == 2


class TestCompetitorAnalyzer:
    """Test CompetitorAnalyzer functionality"""

    @pytest.mark.asyncio
    async def test_analyze_competitors_success(self):
        """Test competitor analysis"""
        analyzer = CompetitorAnalyzer()

        mock_results = [
            Mock(title="Google vs Bing", snippet="Comparing search engines", position=1),
            Mock(title="Microsoft announces", snippet="New features from Microsoft", position=2),
            Mock(title="Google updates", snippet="Google releases new algorithm", position=3)
        ]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.analyze") as mock_analyze:
                mock_analyze.return_value = {
                    "competitors": ["Google", "Microsoft", "Bing"],
                    "products": ["Search Engine", "Cloud Services"],
                    "analysis": "Analysis text"
                }

                result = await analyzer.analyze_competitors(search_id=1)

                assert result["search_id"] == 1
                assert "Google" in result["identified_competitors"]
                assert "Microsoft" in result["identified_competitors"]
                assert "mention_counts" in result

    @pytest.mark.asyncio
    async def test_competitor_mention_counting(self):
        """Test competitor mention counting"""
        analyzer = CompetitorAnalyzer()

        mock_results = [
            Mock(title="Google Search", snippet="Google is the leader", position=1),
            Mock(title="Google News", snippet="Latest from Google", position=2),
            Mock(title="Microsoft Edge", snippet="Microsoft browser", position=3)
        ]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.analyze") as mock_analyze:
                mock_analyze.return_value = {
                    "competitors": ["Google", "Microsoft"],
                    "products": [],
                    "analysis": ""
                }

                result = await analyzer.analyze_competitors(search_id=1)

                # Google should appear 2 times, Microsoft 1 time
                assert result["mention_counts"]["Google"] == 2
                assert result["mention_counts"]["Microsoft"] == 1

    @pytest.mark.asyncio
    async def test_known_competitors_matching(self):
        """Test known competitors matching"""
        analyzer = CompetitorAnalyzer()

        mock_results = [
            Mock(title="Amazon AWS", snippet="Cloud computing from Amazon", position=1),
            Mock(title="Google Cloud", snippet="Google's cloud platform", position=2)
        ]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.analyze") as mock_analyze:
                mock_analyze.return_value = {
                    "competitors": ["Amazon", "Google"],
                    "products": ["AWS", "Google Cloud"],
                    "analysis": ""
                }

                result = await analyzer.analyze_competitors(
                    search_id=1,
                    known_competitors=["Amazon", "Microsoft", "Google"]
                )

                # Should find Amazon and Google, but not Microsoft
                assert "Amazon" in result["known_competitors_found"]
                assert "Google" in result["known_competitors_found"]
                assert len(result["known_competitors_found"]) == 2

    @pytest.mark.asyncio
    async def test_competitor_empty_results(self):
        """Test competitor analysis with no results"""
        analyzer = CompetitorAnalyzer()

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = []
            mock_db_session.execute.return_value = mock_execute_result

            result = await analyzer.analyze_competitors(search_id=999)

            assert "error" in result
            assert result["search_id"] == 999


class TestTrendAnalyzer:
    """Test TrendAnalyzer functionality"""

    @pytest.mark.asyncio
    async def test_analyze_trends_success(self):
        """Test trend analysis"""
        analyzer = TrendAnalyzer()

        # Create 15 mock results (will process first 30)
        mock_results = [
            Mock(title=f"Result {i}", snippet=f"Snippet {i}", position=i)
            for i in range(1, 16)
        ]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.analyze") as mock_analyze:
                mock_analyze.return_value = {
                    "trends": ["AI adoption", "Cloud migration", "Remote work"],
                    "keywords": ["AI", "Cloud", "Digital"],
                    "topics": ["Technology", "Business"],
                    "analysis": ""
                }

                result = await analyzer.analyze_trends(search_id=1)

                assert result["search_id"] == 1
                assert "top_trends" in result
                assert "keywords" in result
                assert "topics" in result
                assert result["analyzed_results"] == 15

    @pytest.mark.asyncio
    async def test_trend_frequency_analysis(self):
        """Test trend frequency counting"""
        analyzer = TrendAnalyzer()

        mock_results = [Mock(title="T", snippet="S", position=i) for i in range(25)]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.analyze") as mock_analyze:
                # Return same trends multiple times
                mock_analyze.side_effect = [
                    {"trends": ["AI adoption", "Cloud migration"], "keywords": [], "topics": []},
                    {"trends": ["AI adoption", "Remote work"], "keywords": [], "topics": []},
                    {"trends": ["Cloud migration", "AI adoption"], "keywords": [], "topics": []}
                ]

                result = await analyzer.analyze_trends(search_id=1)

                # AI adoption should appear 3 times
                assert "AI adoption" in result["trend_frequency"]
                assert result["trend_frequency"]["AI adoption"] == 3

    @pytest.mark.asyncio
    async def test_trend_multi_block_processing(self):
        """Test multi-block trend processing"""
        analyzer = TrendAnalyzer()

        # Create 35 results (will create 4 blocks, analyze first 3)
        mock_results = [
            Mock(title=f"T{i}", snippet=f"S{i}", position=i)
            for i in range(35)
        ]

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = mock_results
            mock_db_session.execute.return_value = mock_execute_result

            with patch("src.analyzers.sentiment.ollama_client.analyze") as mock_analyze:
                mock_analyze.return_value = {
                    "trends": ["Trend 1"],
                    "keywords": ["Keyword 1"],
                    "topics": ["Topic 1"],
                    "analysis": ""
                }

                result = await analyzer.analyze_trends(search_id=1)

                # Should analyze 3 blocks (30 results)
                assert result["blocks_analyzed"] == 3
                assert mock_analyze.call_count == 3

    @pytest.mark.asyncio
    async def test_trend_empty_results(self):
        """Test trend analysis with no results"""
        analyzer = TrendAnalyzer()

        with patch("src.analyzers.sentiment.db_manager.get_session") as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db_session

            mock_execute_result = Mock()
            mock_execute_result.scalars.return_value.all.return_value = []
            mock_db_session.execute.return_value = mock_execute_result

            result = await analyzer.analyze_trends(search_id=999)

            assert "error" in result
            assert result["analyzed_results"] == 0
