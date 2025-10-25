"""
Analysis Routes for CIAP API
LLM analysis endpoints using Ollama
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

from ...analyzers.ollama_client import ollama_client
from ...analyzers.sentiment import SentimentAnalyzer, CompetitorAnalyzer, TrendAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class AnalyzeTextRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze",
        examples=["This is a great product with innovative features"]
    )
    analysis_type: str = Field(
        default="summary",
        description="Type of analysis (sentiment, competitor, summary, trends, insights, keywords)",
        examples=["sentiment"]
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached results if available"
    )


class AnalyzeTextResponse(BaseModel):
    """Response model for text analysis"""
    text_length: int
    analysis_type: str
    result: Dict[str, Any]
    cached: bool = False


class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis"""
    search_id: int
    sample_size: int
    sentiment_distribution: Dict[str, int]
    dominant_sentiment: str
    average_confidence: float
    details: Optional[List[Dict[str, Any]]] = None


class CompetitorAnalysisResponse(BaseModel):
    """Response model for competitor analysis"""
    search_id: int
    competitors: List[str]
    products: List[str]
    mentions: Dict[str, int]
    analysis: str


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis"""
    search_id: int
    trends: List[str]
    keywords: List[str]
    topics: List[str]


# ============================================================
# Analysis Endpoints
# ============================================================

@router.post("/text", response_model=AnalyzeTextResponse)
async def analyze_text(request: AnalyzeTextRequest):
    """
    Analyze arbitrary text using Ollama LLM

    Supports multiple analysis types:
    - sentiment: Positive/negative/neutral classification
    - competitor: Competitor and product identification
    - summary: Text summarization
    - trends: Trend identification
    - insights: Business insights extraction
    - keywords: Keyword extraction

    Args:
        request: Text and analysis parameters

    Returns:
        Analysis results

    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Perform analysis
        result = await ollama_client.analyze(
            text=request.text,
            analysis_type=request.analysis_type,
            use_cache=request.use_cache
        )

        # Check if result came from cache
        cached = False
        if ollama_client.stats["cache_hits"] > 0:
            cached = True

        logger.info(
            f"Analyzed text ({len(request.text)} chars) "
            f"with type '{request.analysis_type}' "
            f"(cached={cached})"
        )

        return AnalyzeTextResponse(
            text_length=len(request.text),
            analysis_type=request.analysis_type,
            result=result,
            cached=cached
        )

    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/sentiment/{search_id}", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    search_id: int,
    sample_size: int = 50
):
    """
    Perform sentiment analysis on search results

    Analyzes the sentiment (positive/negative/neutral) of search results
    for a given search.

    Args:
        search_id: Search ID
        sample_size: Number of results to analyze (default: 50)

    Returns:
        Sentiment analysis results

    Raises:
        HTTPException: If search not found or analysis fails
    """
    try:
        analyzer = SentimentAnalyzer()

        result = await analyzer.analyze_search_results(
            search_id=search_id,
            sample_size=sample_size
        )

        logger.info(f"Sentiment analysis completed for search {search_id}")

        return SentimentAnalysisResponse(
            search_id=search_id,
            sample_size=result["sample_size"],
            sentiment_distribution=result["sentiment_distribution"],
            dominant_sentiment=result["dominant_sentiment"],
            average_confidence=result["average_confidence"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Sentiment analysis failed for search {search_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@router.get("/competitors/{search_id}", response_model=CompetitorAnalysisResponse)
async def analyze_competitors(
    search_id: int,
    known_competitors: Optional[str] = None
):
    """
    Identify competitors mentioned in search results

    Analyzes search results to identify competitor names, products,
    and mention frequency.

    Args:
        search_id: Search ID
        known_competitors: Optional comma-separated list of known competitors

    Returns:
        Competitor analysis results

    Raises:
        HTTPException: If search not found or analysis fails
    """
    try:
        analyzer = CompetitorAnalyzer()

        # Parse known competitors if provided
        known_list = None
        if known_competitors:
            known_list = [c.strip() for c in known_competitors.split(",")]

        result = await analyzer.analyze_competitors(
            search_id=search_id,
            known_competitors=known_list
        )

        logger.info(f"Competitor analysis completed for search {search_id}")

        return CompetitorAnalysisResponse(
            search_id=search_id,
            competitors=result["competitors"],
            products=result["products"],
            mentions=result["mentions"],
            analysis=result["analysis"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Competitor analysis failed for search {search_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Competitor analysis failed: {str(e)}"
        )


@router.get("/trends/{search_id}", response_model=TrendAnalysisResponse)
async def analyze_trends(search_id: int):
    """
    Identify trends in search results

    Analyzes search results to identify emerging trends, patterns,
    and key topics.

    Args:
        search_id: Search ID

    Returns:
        Trend analysis results

    Raises:
        HTTPException: If search not found or analysis fails
    """
    try:
        analyzer = TrendAnalyzer()

        result = await analyzer.analyze_trends(search_id=search_id)

        logger.info(f"Trend analysis completed for search {search_id}")

        return TrendAnalysisResponse(
            search_id=search_id,
            trends=result["trends"],
            keywords=result["keywords"],
            topics=result["topics"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Trend analysis failed for search {search_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Trend analysis failed: {str(e)}"
        )


@router.get("/insights/{search_id}", response_model=Dict[str, Any])
async def analyze_insights(search_id: int):
    """
    Generate business insights from search results

    Bonus endpoint: Analyzes search results to extract actionable
    business insights, opportunities, and recommendations.

    Args:
        search_id: Search ID

    Returns:
        Business insights

    Raises:
        HTTPException: If search not found or analysis fails
    """
    try:
        # Get search results
        from ...database import db_manager, DatabaseOperations

        async with db_manager.get_session() as session:
            results = await DatabaseOperations.get_search_results(session, search_id)

            if not results:
                raise ValueError(f"No results found for search {search_id}")

            # Combine top results for analysis
            combined_text = "\n\n".join([
                f"{r.title}\n{r.snippet}"
                for r in results[:20]  # Analyze top 20 results
            ])

            # Analyze with insights type
            insights = await ollama_client.analyze(
                text=combined_text,
                analysis_type="insights",
                use_cache=True
            )

            logger.info(f"Insights analysis completed for search {search_id}")

            return {
                "search_id": search_id,
                "sample_size": min(len(results), 20),
                "insights": insights
            }

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Insights analysis failed for search {search_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Insights analysis failed: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_analysis_stats():
    """
    Get analysis statistics from Ollama client

    Returns:
        Statistics including request counts, cache hits, errors, tokens

    """
    try:
        stats = ollama_client.stats.copy()

        # Add cache hit rate calculation
        total_requests = stats["requests"]
        cache_hits = stats["cache_hits"]

        if total_requests > 0:
            stats["cache_hit_rate"] = f"{(cache_hits / total_requests * 100):.1f}%"
        else:
            stats["cache_hit_rate"] = "0%"

        return stats

    except Exception as e:
        logger.error(f"Failed to get analysis stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )
