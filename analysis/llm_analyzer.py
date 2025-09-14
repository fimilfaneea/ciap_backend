"""LLM integration for competitive intelligence analysis"""
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    DEFAULT_LLM_PROVIDER,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conditional imports based on available API keys
try:
    if OPENAI_API_KEY:
        import openai
        openai.api_key = OPENAI_API_KEY
except ImportError:
    logger.warning("OpenAI library not installed")

try:
    if ANTHROPIC_API_KEY:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
except ImportError:
    logger.warning("Anthropic library not installed")


class LLMAnalyzer:
    """Unified LLM analyzer for competitive intelligence"""

    def __init__(self, provider: str = None):
        self.provider = provider or DEFAULT_LLM_PROVIDER
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE

        # Validate provider configuration
        if self.provider == "openai" and not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        elif self.provider == "anthropic" and not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not configured")

    def analyze_competitor(self, search_results: List[Dict]) -> Dict:
        """Analyze search results to identify competitor insights"""

        prompt = self._build_competitor_prompt(search_results)
        response = self._call_llm(prompt)

        try:
            # Parse JSON response
            insights = json.loads(response)
        except json.JSONDecodeError:
            # If response is not valid JSON, create a basic structure
            insights = {
                "competitors": [],
                "summary": response,
                "strengths": [],
                "weaknesses": [],
                "opportunities": []
            }

        return {
            "analysis_type": "competitor",
            "content": response,
            "insights": insights,
            "llm_provider": self.provider,
            "llm_model": self.model,
            "created_at": datetime.utcnow().isoformat()
        }

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text content"""

        prompt = f"""Analyze the sentiment of the following text and provide:
1. Overall sentiment (positive, negative, neutral)
2. Sentiment score (-1 to 1)
3. Key emotional indicators
4. Confidence level (0 to 1)

Text: {text[:2000]}

Provide response as JSON with keys: sentiment, score, indicators, confidence"""

        response = self._call_llm(prompt)

        try:
            result = json.loads(response)
            sentiment_score = float(result.get("score", 0))
            confidence_score = float(result.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError):
            # Fallback parsing
            sentiment_score = 0
            confidence_score = 0.5
            result = {"sentiment": "neutral", "analysis": response}

        return {
            "analysis_type": "sentiment",
            "content": response,
            "insights": result,
            "sentiment_score": sentiment_score,
            "confidence_score": confidence_score,
            "llm_provider": self.provider,
            "llm_model": self.model
        }

    def analyze_market_trends(self, search_results: List[Dict]) -> Dict:
        """Analyze market trends from search results"""

        prompt = self._build_market_trends_prompt(search_results)
        response = self._call_llm(prompt)

        try:
            insights = json.loads(response)
        except json.JSONDecodeError:
            insights = {
                "trends": [],
                "summary": response,
                "opportunities": [],
                "threats": []
            }

        return {
            "analysis_type": "market_trends",
            "content": response,
            "insights": insights,
            "llm_provider": self.provider,
            "llm_model": self.model
        }

    def generate_summary(self, search_results: List[Dict]) -> Dict:
        """Generate executive summary from search results"""

        prompt = f"""Based on these search results, provide an executive summary with:
1. Key findings
2. Important patterns
3. Actionable insights
4. Recommended next steps

Search Results:
{self._format_search_results(search_results[:10])}

Provide a concise summary focusing on business intelligence value."""

        response = self._call_llm(prompt)

        return {
            "analysis_type": "summary",
            "content": response,
            "insights": {
                "summary": response,
                "result_count": len(search_results)
            },
            "llm_provider": self.provider,
            "llm_model": self.model
        }

    def _build_competitor_prompt(self, search_results: List[Dict]) -> str:
        """Build prompt for competitor analysis"""

        results_text = self._format_search_results(search_results[:10])

        return f"""Analyze these search results to identify competitive intelligence:

{results_text}

Provide analysis as JSON with the following structure:
{{
    "competitors": ["list of identified competitors"],
    "summary": "brief overview of competitive landscape",
    "strengths": ["competitor strengths"],
    "weaknesses": ["competitor weaknesses"],
    "opportunities": ["market opportunities"],
    "threats": ["competitive threats"],
    "key_insights": ["main takeaways"]
}}"""

    def _build_market_trends_prompt(self, search_results: List[Dict]) -> str:
        """Build prompt for market trends analysis"""

        results_text = self._format_search_results(search_results[:10])

        return f"""Analyze these search results to identify market trends and insights:

{results_text}

Provide analysis as JSON with the following structure:
{{
    "trends": ["list of identified market trends"],
    "summary": "market overview",
    "growth_areas": ["areas of growth"],
    "declining_areas": ["areas of decline"],
    "opportunities": ["business opportunities"],
    "threats": ["market threats"],
    "predictions": ["future predictions"]
}}"""

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results for LLM prompt"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. Title: {result.get('title', '')}\n"
                f"   URL: {result.get('url', '')}\n"
                f"   Snippet: {result.get('snippet', '')}\n"
            )
        return "\n".join(formatted)

    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM provider"""

        try:
            if self.provider == "openai":
                return self._call_openai(prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Analysis failed: {str(e)}"

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a competitive intelligence analyst. Provide structured, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        try:
            message = anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a competitive intelligence analyst. Provide structured, actionable insights.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


# Test function
def test_analyzer():
    """Test the LLM analyzer with sample data"""

    # Sample search results
    sample_results = [
        {
            "title": "Top 10 Slack Competitors in 2024",
            "url": "https://example.com/slack-competitors",
            "snippet": "Microsoft Teams, Discord, and Zoom are leading alternatives to Slack..."
        },
        {
            "title": "Slack vs Microsoft Teams: Feature Comparison",
            "url": "https://example.com/slack-vs-teams",
            "snippet": "While Slack excels in integrations, Teams offers better video conferencing..."
        }
    ]

    try:
        analyzer = LLMAnalyzer()
        print("\n=== Testing Competitor Analysis ===")
        result = analyzer.analyze_competitor(sample_results)
        print(f"Analysis Type: {result['analysis_type']}")
        print(f"Provider: {result['llm_provider']}")
        print(f"Insights: {json.dumps(result['insights'], indent=2)[:500]}...")

        print("\n=== Testing Sentiment Analysis ===")
        sentiment = analyzer.analyze_sentiment("This product is amazing and has transformed our workflow!")
        print(f"Sentiment Score: {sentiment.get('sentiment_score')}")
        print(f"Confidence: {sentiment.get('confidence_score')}")

    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure to set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")


if __name__ == "__main__":
    test_analyzer()