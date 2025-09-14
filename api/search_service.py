"""Search service to coordinate scraping and analysis"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from database import Search, SearchResult, Analysis, CompetitorProfile
from scrapers.google_scraper import GoogleScraper
from analysis.llm_analyzer import LLMAnalyzer
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchService:
    """Service to handle search, scraping, and analysis workflow"""

    def __init__(self, db: Session):
        self.db = db
        self.scraper = GoogleScraper()

        # Initialize LLM analyzer if API keys are available
        self.analyzer = None
        if OPENAI_API_KEY or ANTHROPIC_API_KEY:
            try:
                self.analyzer = LLMAnalyzer()
            except Exception as e:
                logger.warning(f"Could not initialize LLM analyzer: {e}")

    def perform_search(self, search_id: int) -> Dict:
        """Perform complete search workflow: scrape -> analyze -> save"""

        try:
            # Get search record
            search = self.db.query(Search).filter(Search.id == search_id).first()
            if not search:
                return {"error": "Search not found"}

            # Update status to scraping
            search.status = "scraping"
            self.db.commit()

            # Perform scraping based on search type
            results = self._scrape_data(search)

            # Save search results
            self._save_search_results(search_id, results)

            # Update status to analyzing
            search.status = "analyzing"
            self.db.commit()

            # Perform analysis if LLM is available
            if self.analyzer and results:
                analyses = self._analyze_results(search, results)
                self._save_analyses(search_id, analyses)

            # Update search as completed
            search.status = "completed"
            search.completed_at = datetime.utcnow()
            self.db.commit()

            return {
                "status": "success",
                "search_id": search_id,
                "results_count": len(results),
                "analyses_count": len(analyses) if self.analyzer else 0
            }

        except Exception as e:
            logger.error(f"Search failed for ID {search_id}: {e}")

            # Update search as failed
            if search:
                search.status = "failed"
                self.db.commit()

            return {"error": str(e)}

    def _scrape_data(self, search: Search) -> List[Dict]:
        """Perform scraping based on search type"""

        query = search.query
        search_type = search.search_type
        max_results = search.metadata.get("max_results", 10) if search.metadata else 10

        if search_type == "competitor":
            # For competitor search, modify query
            results = self.scraper.search_competitor(query)
        elif search_type == "market":
            results = self.scraper.search_market_trends(query)
        elif search_type == "sentiment":
            results = self.scraper.search_customer_sentiment(query)
        else:
            # General search
            results = self.scraper.search(query, max_results)

        return results

    def _save_search_results(self, search_id: int, results: List[Dict]):
        """Save search results to database"""

        for result in results:
            db_result = SearchResult(
                search_id=search_id,
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("snippet", ""),
                source=result.get("source", "unknown"),
                position=result.get("position", 0),
                metadata=result
            )
            self.db.add(db_result)

        self.db.commit()

    def _analyze_results(self, search: Search, results: List[Dict]) -> List[Dict]:
        """Perform LLM analysis on search results"""

        analyses = []

        try:
            # Generate different types of analysis based on search type
            if search.search_type == "competitor":
                analysis = self.analyzer.analyze_competitor(results)
                analyses.append(analysis)

                # Extract and save competitor profiles
                self._extract_competitor_profiles(analysis)

            elif search.search_type == "market":
                analysis = self.analyzer.analyze_market_trends(results)
                analyses.append(analysis)

            elif search.search_type == "sentiment":
                # Analyze sentiment for each result
                for result in results[:5]:  # Limit to avoid excessive API calls
                    text = f"{result.get('title', '')} {result.get('snippet', '')}"
                    sentiment = self.analyzer.analyze_sentiment(text)
                    analyses.append(sentiment)

            # Always generate a summary
            summary = self.analyzer.generate_summary(results)
            analyses.append(summary)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return analyses

    def _save_analyses(self, search_id: int, analyses: List[Dict]):
        """Save analysis results to database"""

        for analysis_data in analyses:
            analysis = Analysis(
                search_id=search_id,
                analysis_type=analysis_data.get("analysis_type", "unknown"),
                content=analysis_data.get("content", ""),
                insights=analysis_data.get("insights", {}),
                sentiment_score=analysis_data.get("sentiment_score"),
                confidence_score=analysis_data.get("confidence_score"),
                llm_provider=analysis_data.get("llm_provider", "unknown"),
                llm_model=analysis_data.get("llm_model", "unknown"),
                metadata=analysis_data
            )
            self.db.add(analysis)

        self.db.commit()

    def _extract_competitor_profiles(self, analysis: Dict):
        """Extract and save competitor profiles from analysis"""

        try:
            insights = analysis.get("insights", {})
            competitors = insights.get("competitors", [])

            for comp_name in competitors:
                # Check if profile already exists
                existing = self.db.query(CompetitorProfile).filter(
                    CompetitorProfile.name == comp_name
                ).first()

                if not existing:
                    profile = CompetitorProfile(
                        name=comp_name,
                        strengths=insights.get("strengths", []),
                        weaknesses=insights.get("weaknesses", []),
                        metadata={"source": "analysis"}
                    )
                    self.db.add(profile)

            self.db.commit()

        except Exception as e:
            logger.error(f"Failed to extract competitor profiles: {e}")


def process_search_background(search_id: int, db: Session):
    """Background task to process a search"""
    service = SearchService(db)
    result = service.perform_search(search_id)
    logger.info(f"Search {search_id} completed: {result}")
    return result