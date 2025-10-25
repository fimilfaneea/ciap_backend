"""
Specialized Analyzers for CIAP LLM Analysis System
Provides sentiment, competitor, and trend analysis for search results
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import select

from .ollama_client import ollama_client
from ..database import db_manager, SearchResult

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

        Fetches search results from database, performs sentiment analysis
        on title + snippet text, and aggregates results.

        Args:
            search_id: Search ID to analyze
            sample_size: Maximum number of results to analyze (default: 50)

        Returns:
            Dictionary containing:
            - search_id: The search ID analyzed
            - analyzed_count: Number of results analyzed
            - sentiment_distribution: Count of positive/negative/neutral
            - dominant_sentiment: Most common sentiment
            - average_confidence: Average confidence score
            - details: First 10 individual results for review
        """
        logger.info(
            f"Starting sentiment analysis for search_id={search_id}, "
            f"sample_size={sample_size}"
        )

        # Get search results from database
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
                .limit(sample_size)
            )
            search_results = result.scalars().all()

        if not search_results:
            logger.warning(f"No results found for search_id={search_id}")
            return {
                "error": "No results found",
                "search_id": search_id,
                "analyzed_count": 0
            }

        # Prepare texts for analysis (title + snippet)
        texts = [
            f"{r.title} {r.snippet}"
            for r in search_results
        ]

        logger.debug(f"Analyzing {len(texts)} texts for sentiment")

        # Batch analyze sentiment
        sentiments = await ollama_client.batch_analyze(
            texts,
            analysis_type="sentiment",
            batch_size=10
        )

        # Aggregate results
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        total_confidence = 0.0
        analyzed_count = 0

        for sentiment in sentiments:
            if "error" not in sentiment:
                sentiment_value = sentiment.get("sentiment", "neutral")
                sentiment_counts[sentiment_value] = sentiment_counts.get(sentiment_value, 0) + 1
                total_confidence += sentiment.get("confidence", 0.5)
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
            average_confidence = 0.0

        result = {
            "search_id": search_id,
            "analyzed_count": analyzed_count,
            "sentiment_distribution": sentiment_counts,
            "dominant_sentiment": dominant_sentiment,
            "average_confidence": round(average_confidence, 3),
            "details": sentiments[:10]  # First 10 for review
        }

        logger.info(
            f"Sentiment analysis complete: search_id={search_id}, "
            f"dominant={dominant_sentiment}, confidence={average_confidence:.3f}"
        )

        return result


class CompetitorAnalyzer:
    """Analyze competitor mentions in search results"""

    async def analyze_competitors(
        self,
        search_id: int,
        known_competitors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze competitor mentions in search results

        Fetches search results, combines text, and uses LLM to identify
        competitor names and count their mentions.

        Args:
            search_id: Search ID to analyze
            known_competitors: Optional list of known competitor names to track

        Returns:
            Dictionary containing:
            - search_id: The search ID analyzed
            - identified_competitors: List of competitor names found
            - mention_counts: Dictionary mapping competitor to mention count
            - known_competitors_found: Subset of known_competitors that were found
            - analysis: Raw LLM analysis text
        """
        logger.info(
            f"Starting competitor analysis for search_id={search_id}, "
            f"known_competitors={known_competitors}"
        )

        # Get search results from database
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
                .order_by(SearchResult.position)
            )
            search_results = result.scalars().all()

        if not search_results:
            logger.warning(f"No results found for search_id={search_id}")
            return {
                "error": "No results found",
                "search_id": search_id
            }

        # Combine text for analysis (top 30 results for context)
        combined_text = "\n".join([
            f"{r.title}: {r.snippet}"
            for r in search_results[:30]
        ])

        logger.debug(
            f"Analyzing {len(search_results[:30])} results for competitors "
            f"(combined length: {len(combined_text)} chars)"
        )

        # Analyze with Ollama
        analysis = await ollama_client.analyze(
            combined_text,
            analysis_type="competitor"
        )

        # Extract competitor mentions
        competitors = analysis.get("competitors", [])

        logger.debug(f"Identified competitors: {competitors}")

        # Count mentions across all results
        competitor_counts = {}
        for result in search_results:
            text = f"{result.title} {result.snippet}".lower()
            for competitor in competitors:
                if competitor.lower() in text:
                    competitor_counts[competitor] = \
                        competitor_counts.get(competitor, 0) + 1

        # Track known competitors if provided
        known_found = []
        if known_competitors:
            for known in known_competitors:
                # Check if known competitor appears in identified list
                if any(known.lower() in comp.lower() or comp.lower() in known.lower()
                       for comp in competitors):
                    known_found.append(known)
                else:
                    # Manual count in case LLM missed it
                    count = 0
                    for result in search_results:
                        text = f"{result.title} {result.snippet}".lower()
                        if known.lower() in text:
                            count += 1
                    if count > 0:
                        known_found.append(known)
                        competitor_counts[known] = count

        result_dict = {
            "search_id": search_id,
            "identified_competitors": competitors,
            "mention_counts": competitor_counts,
            "known_competitors_found": known_found,
            "total_results_analyzed": len(search_results),
            "products": analysis.get("products", []),
            "analysis": analysis.get("analysis", "")
        }

        logger.info(
            f"Competitor analysis complete: search_id={search_id}, "
            f"found={len(competitors)} competitors, "
            f"known_found={len(known_found)}/{len(known_competitors or [])}"
        )

        return result_dict


class TrendAnalyzer:
    """Analyze trends in search results"""

    async def analyze_trends(
        self,
        search_id: int
    ) -> Dict[str, Any]:
        """
        Identify trends in search results

        Fetches search results, processes them in blocks, and identifies
        recurring trends, keywords, and topics.

        Args:
            search_id: Search ID to analyze

        Returns:
            Dictionary containing:
            - search_id: The search ID analyzed
            - top_trends: List of identified trends (sorted by frequency)
            - trend_frequency: Dictionary mapping trend to occurrence count
            - keywords: List of important keywords
            - topics: List of main topics
            - analyzed_results: Number of results analyzed
        """
        logger.info(f"Starting trend analysis for search_id={search_id}")

        # Get search results from database
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(SearchResult)
                .where(SearchResult.search_id == search_id)
                .order_by(SearchResult.position)
            )
            search_results = result.scalars().all()

        if not search_results:
            logger.warning(f"No results found for search_id={search_id}")
            return {
                "error": "No results found",
                "search_id": search_id,
                "analyzed_results": 0
            }

        # Prepare text blocks (analyze in chunks of 10 results)
        text_blocks = []
        for i in range(0, len(search_results), 10):
            block = search_results[i:i+10]
            combined = "\n".join([
                f"{r.title}: {r.snippet}"
                for r in block
            ])
            text_blocks.append(combined)

        logger.debug(
            f"Prepared {len(text_blocks)} text blocks for trend analysis "
            f"({len(search_results)} total results)"
        )

        # Analyze first 3 blocks for trends (top 30 results)
        trend_analyses = []
        for idx, block_text in enumerate(text_blocks[:3]):
            logger.debug(f"Analyzing block {idx+1}/3 for trends")
            analysis = await ollama_client.analyze(
                block_text,
                analysis_type="trends"
            )
            trend_analyses.append(analysis)

        # Combine and deduplicate trends
        all_trends = []
        all_keywords = []
        all_topics = []

        for analysis in trend_analyses:
            trends = analysis.get("trends", [])
            keywords = analysis.get("keywords", [])
            topics = analysis.get("topics", [])

            all_trends.extend(trends)
            all_keywords.extend(keywords)
            all_topics.extend(topics)

        # Count trend frequency (case-insensitive)
        trend_counts = {}
        for trend in all_trends:
            # Normalize trend (lowercase for comparison)
            trend_lower = trend.lower()

            # Find existing similar trend
            found = False
            for existing_trend in trend_counts.keys():
                if existing_trend.lower() == trend_lower:
                    trend_counts[existing_trend] += 1
                    found = True
                    break

            if not found:
                trend_counts[trend] = 1

        # Sort by frequency
        top_trends_sorted = sorted(
            trend_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Deduplicate keywords and topics (keep original case)
        keywords_unique = list(dict.fromkeys(all_keywords))[:15]
        topics_unique = list(dict.fromkeys(all_topics))[:10]

        result_dict = {
            "search_id": search_id,
            "top_trends": [trend for trend, _ in top_trends_sorted],
            "trend_frequency": dict(top_trends_sorted),
            "keywords": keywords_unique,
            "topics": topics_unique,
            "analyzed_results": len(search_results),
            "blocks_analyzed": len(trend_analyses)
        }

        logger.info(
            f"Trend analysis complete: search_id={search_id}, "
            f"trends={len(top_trends_sorted)}, "
            f"keywords={len(keywords_unique)}, "
            f"topics={len(topics_unique)}"
        )

        return result_dict
