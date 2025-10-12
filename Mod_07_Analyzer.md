# Module 7: LLM Analysis System (Ollama Integration)

## Overview
**Purpose:** Integrate Ollama LLM for text analysis, sentiment detection, competitor intelligence, and trend identification.

**Responsibilities:**
- Ollama API integration
- Sentiment analysis
- Competitor mention detection
- Key insights extraction
- Trend analysis
- Prompt management
- Result caching

**Development Time:** 3 days (Week 6-7, Day 20-28)

---

## Implementation Guide

### Ollama Client (`src/analyzers/ollama_client.py`)

```python
import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

import httpx

from src.core.config import settings
from src.core.cache import cache

logger = logging.getLogger(__name__)


class OllamaException(Exception):
    """Ollama-specific exceptions"""
    pass


class OllamaClient:
    """Client for Ollama LLM API"""

    def __init__(self):
        self.base_url = settings.OLLAMA_URL
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT

        # Prompt templates
        self.prompts = self._load_prompts()

        # Statistics
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_tokens": 0
        }

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates"""
        return {
            "sentiment": """Analyze the sentiment of the following text.
Return a JSON object with:
- sentiment: positive, negative, or neutral
- confidence: 0.0 to 1.0
- key_phrases: list of important phrases

Text: {text}""",

            "competitor": """Identify competitors mentioned in this text.
Return a JSON object with:
- competitors: list of competitor names
- products: list of products mentioned
- comparisons: list of comparison points

Text: {text}""",

            "summary": """Provide a concise summary of this text.
Focus on key business insights.
Maximum 3 sentences.

Text: {text}""",

            "trends": """Identify trends and patterns in this text.
Return a JSON object with:
- trends: list of identified trends
- keywords: most frequent important terms
- topics: main topics discussed

Text: {text}""",

            "insights": """Extract actionable business insights from this text.
Return a JSON object with:
- opportunities: list of opportunities
- threats: list of threats
- recommendations: list of recommendations

Text: {text}"""
        }

    async def check_health(self) -> bool:
        """Check if Ollama is accessible and model is available"""
        try:
            async with httpx.AsyncClient() as client:
                # Check API endpoint
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    timeout=5
                )

                if response.status_code != 200:
                    return False

                # Check if our model is available
                models = response.json().get("models", [])
                return any(
                    model.get("name") == self.model
                    for model in models
                )

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def analyze(
        self,
        text: str,
        analysis_type: str = "summary",
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze text using Ollama

        Args:
            text: Text to analyze
            analysis_type: Type of analysis
            use_cache: Whether to use cached results
            **kwargs: Additional parameters for prompt

        Returns:
            Analysis results
        """
        # Generate cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"ollama:{analysis_type}:{text_hash}"

        # Check cache
        if use_cache:
            cached = await cache.get(cache_key)
            if cached:
                self.stats["cache_hits"] += 1
                logger.debug(f"Using cached {analysis_type} analysis")
                return cached

        # Get prompt template
        prompt_template = self.prompts.get(
            analysis_type,
            self.prompts["summary"]
        )

        # Format prompt
        prompt = prompt_template.format(text=text[:2000], **kwargs)

        try:
            # Make request to Ollama
            result = await self._request_ollama(prompt)

            # Parse result
            parsed = self._parse_response(result, analysis_type)

            # Cache result
            if use_cache:
                await cache.set(cache_key, parsed, ttl=7200)  # 2 hours

            self.stats["requests"] += 1

            return parsed

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ollama analysis failed: {e}")
            raise OllamaException(f"Analysis failed: {e}")

    async def _request_ollama(self, prompt: str) -> str:
        """Make request to Ollama API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 500
                        }
                    }
                )

                response.raise_for_status()
                result = response.json()

                # Update token count
                if "total_tokens" in result:
                    self.stats["total_tokens"] += result["total_tokens"]

                return result.get("response", "")

            except httpx.TimeoutException:
                raise OllamaException("Ollama request timed out")
            except httpx.HTTPError as e:
                raise OllamaException(f"HTTP error: {e}")

    def _parse_response(
        self,
        response: str,
        analysis_type: str
    ) -> Dict[str, Any]:
        """Parse Ollama response based on analysis type"""
        # Try to parse as JSON first
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback parsing based on type
        if analysis_type == "sentiment":
            return self._parse_sentiment(response)
        elif analysis_type == "competitor":
            return self._parse_competitors(response)
        elif analysis_type == "trends":
            return self._parse_trends(response)
        else:
            return {"text": response}

    def _parse_sentiment(self, text: str) -> Dict[str, Any]:
        """Parse sentiment from text response"""
        sentiment = "neutral"
        confidence = 0.5

        text_lower = text.lower()
        if "positive" in text_lower:
            sentiment = "positive"
            confidence = 0.8
        elif "negative" in text_lower:
            sentiment = "negative"
            confidence = 0.8

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "analysis": text
        }

    def _parse_competitors(self, text: str) -> Dict[str, Any]:
        """Parse competitors from text response"""
        # Simple extraction of capitalized words as potential competitors
        import re
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        competitors = list(set(words))[:5]  # Top 5 unique

        return {
            "competitors": competitors,
            "analysis": text
        }

    def _parse_trends(self, text: str) -> Dict[str, Any]:
        """Parse trends from text response"""
        # Extract lines that might be trends
        lines = text.split('\n')
        trends = [
            line.strip('- •*').strip()
            for line in lines
            if line.strip() and len(line) < 100
        ][:5]

        return {
            "trends": trends,
            "analysis": text
        }

    async def batch_analyze(
        self,
        texts: List[str],
        analysis_type: str = "sentiment",
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batches

        Args:
            texts: List of texts to analyze
            analysis_type: Type of analysis
            batch_size: Number of concurrent requests

        Returns:
            List of analysis results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Process batch concurrently
            tasks = [
                self.analyze(text, analysis_type)
                for text in batch
            ]

            batch_results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )

            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis error: {result}")
                    results.append({"error": str(result)})
                else:
                    results.append(result)

        return results


# Global Ollama client instance
ollama_client = OllamaClient()
```

### Specialized Analyzers (`src/analyzers/sentiment.py`)

```python
from typing import List, Dict, Any
import logging

from src.analyzers.ollama_client import ollama_client
from src.core.database import db_manager
from src.core.models import SearchResult

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment in search results"""

    async def analyze_search_results(
        self,
        search_id: int,
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of search results

        Args:
            search_id: Search ID
            sample_size: Number of results to analyze

        Returns:
            Sentiment analysis summary
        """
        # Get search results
        async with db_manager.get_session() as session:
            results = await session.execute(
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
                .limit(sample_size)
            )
            search_results = results.scalars().all()

        if not search_results:
            return {"error": "No results found"}

        # Prepare texts for analysis
        texts = [
            f"{r.title} {r.snippet}"
            for r in search_results
        ]

        # Batch analyze
        sentiments = await ollama_client.batch_analyze(
            texts,
            analysis_type="sentiment"
        )

        # Aggregate results
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        total_confidence = 0
        analyzed_count = 0

        for sentiment in sentiments:
            if "error" not in sentiment:
                sentiment_counts[sentiment.get("sentiment", "neutral")] += 1
                total_confidence += sentiment.get("confidence", 0)
                analyzed_count += 1

        # Calculate summary
        if analyzed_count > 0:
            dominant_sentiment = max(
                sentiment_counts,
                key=sentiment_counts.get
            )
            average_confidence = total_confidence / analyzed_count
        else:
            dominant_sentiment = "unknown"
            average_confidence = 0

        return {
            "search_id": search_id,
            "analyzed_count": analyzed_count,
            "sentiment_distribution": sentiment_counts,
            "dominant_sentiment": dominant_sentiment,
            "average_confidence": average_confidence,
            "details": sentiments[:10]  # First 10 for review
        }


class CompetitorAnalyzer:
    """Analyze competitor mentions"""

    async def analyze_competitors(
        self,
        search_id: int,
        known_competitors: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze competitor mentions in search results

        Args:
            search_id: Search ID
            known_competitors: List of known competitor names

        Returns:
            Competitor analysis
        """
        # Get search results
        async with db_manager.get_session() as session:
            results = await session.execute(
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
            )
            search_results = results.scalars().all()

        # Combine text for analysis
        combined_text = "\n".join([
            f"{r.title}: {r.snippet}"
            for r in search_results[:30]  # Analyze top 30
        ])

        # Analyze with Ollama
        analysis = await ollama_client.analyze(
            combined_text,
            analysis_type="competitor"
        )

        # Extract competitor mentions
        competitors = analysis.get("competitors", [])

        # Count mentions
        competitor_counts = {}
        for result in search_results:
            text = f"{result.title} {result.snippet}".lower()
            for competitor in competitors:
                if competitor.lower() in text:
                    competitor_counts[competitor] = \
                        competitor_counts.get(competitor, 0) + 1

        return {
            "search_id": search_id,
            "identified_competitors": competitors,
            "mention_counts": competitor_counts,
            "analysis": analysis.get("analysis", "")
        }


class TrendAnalyzer:
    """Analyze trends in search results"""

    async def analyze_trends(
        self,
        search_id: int
    ) -> Dict[str, Any]:
        """
        Identify trends in search results

        Args:
            search_id: Search ID

        Returns:
            Trend analysis
        """
        # Get results
        async with db_manager.get_session() as session:
            results = await session.execute(
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
                .order_by(SearchResult.position)
            )
            search_results = results.scalars().all()

        # Prepare text
        text_blocks = []
        for i in range(0, len(search_results), 10):
            block = search_results[i:i+10]
            combined = "\n".join([
                f"{r.title}: {r.snippet}"
                for r in block
            ])
            text_blocks.append(combined)

        # Analyze each block for trends
        trend_analyses = []
        for block_text in text_blocks[:3]:  # Top 3 blocks
            analysis = await ollama_client.analyze(
                block_text,
                analysis_type="trends"
            )
            trend_analyses.append(analysis)

        # Combine trends
        all_trends = []
        for analysis in trend_analyses:
            trends = analysis.get("trends", [])
            all_trends.extend(trends)

        # Deduplicate and rank
        trend_counts = {}
        for trend in all_trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        top_trends = sorted(
            trend_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "search_id": search_id,
            "top_trends": [t[0] for t in top_trends],
            "trend_frequency": dict(top_trends),
            "analyzed_results": len(search_results)
        }
```

---

## Testing

```python
@pytest.mark.asyncio
async def test_ollama_health_check():
    """Test Ollama connectivity"""
    client = OllamaClient()

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.1:8b"}]
        }
        mock_get.return_value = mock_response

        assert await client.check_health()


@pytest.mark.asyncio
async def test_sentiment_analysis():
    """Test sentiment analysis"""
    client = OllamaClient()

    # Mock Ollama response
    with patch.object(client, "_request_ollama") as mock_request:
        mock_request.return_value = '{"sentiment": "positive", "confidence": 0.9}'

        result = await client.analyze(
            "This is great news!",
            analysis_type="sentiment"
        )

        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.9
```

---

## Module Checklist

- [ ] Ollama client implemented
- [ ] Health check working
- [ ] Prompt templates loaded
- [ ] Sentiment analysis functional
- [ ] Competitor detection working
- [ ] Trend analysis implemented
- [ ] Batch processing optimized
- [ ] Caching integrated
- [ ] Error handling robust
- [ ] Unit tests passing

---

## Next Steps
- Module 8: API - Expose analysis endpoints
- Module 9: Export - Export analysis results