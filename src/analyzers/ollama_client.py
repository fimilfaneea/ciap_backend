"""
Ollama Client for CIAP LLM Analysis System
Provides integration with Ollama LLM for text analysis
"""

import asyncio
import hashlib
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx

from ..config import settings
from ..cache import LLMCache

logger = logging.getLogger(__name__)


class OllamaException(Exception):
    """Ollama-specific exceptions"""
    pass


class OllamaClient:
    """Client for Ollama LLM API"""

    def __init__(self):
        """
        Initialize Ollama client

        Loads configuration from settings, loads prompt templates,
        and initializes statistics tracking.
        """
        self.base_url = settings.OLLAMA_URL
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT

        # Load prompt templates from files
        self.prompts = self._load_prompts()

        # Statistics tracking
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_tokens": 0
        }

        logger.info(
            f"OllamaClient initialized: model={self.model}, "
            f"url={self.base_url}, timeout={self.timeout}s"
        )

    def _load_prompts(self) -> Dict[str, str]:
        """
        Load prompt templates from config/prompts/ directory

        Returns:
            Dictionary mapping prompt names to their content
        """
        prompts = {}
        prompts_dir = Path("config/prompts")

        # Define the 6 expected prompt files
        prompt_files = {
            "sentiment": "sentiment.txt",
            "competitor": "competitor.txt",
            "summary": "summary.txt",
            "trends": "trends.txt",
            "insights": "insights.txt",
            "keywords": "keywords.txt"
        }

        for prompt_name, filename in prompt_files.items():
            filepath = prompts_dir / filename
            try:
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        prompts[prompt_name] = f.read().strip()
                    logger.debug(f"Loaded prompt template: {prompt_name}")
                else:
                    logger.warning(f"Prompt template not found: {filepath}")
                    # Provide a minimal fallback
                    prompts[prompt_name] = f"Analyze the following text for {prompt_name}:\n\n{{text}}"
            except Exception as e:
                logger.error(f"Error loading prompt {prompt_name}: {e}")
                prompts[prompt_name] = f"Analyze the following text for {prompt_name}:\n\n{{text}}"

        logger.info(f"Loaded {len(prompts)} prompt templates")
        return prompts

    async def check_health(self) -> bool:
        """
        Check if Ollama is accessible and model is available

        Returns:
            True if Ollama is healthy and model exists, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                # Check API endpoint
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    timeout=5.0
                )

                if response.status_code != 200:
                    logger.warning(
                        f"Ollama health check failed: HTTP {response.status_code}"
                    )
                    return False

                # Check if our model is available
                data = response.json()
                models = data.get("models", [])

                model_available = any(
                    model.get("name") == self.model or
                    model.get("name", "").startswith(self.model.split(":")[0])
                    for model in models
                )

                if model_available:
                    logger.info(f"Ollama health check passed: model {self.model} available")
                else:
                    logger.warning(
                        f"Ollama model {self.model} not found. "
                        f"Available models: {[m.get('name') for m in models]}"
                    )

                return model_available

        except httpx.TimeoutException:
            logger.error("Ollama health check timed out")
            return False
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
            text: Text to analyze (will be truncated to 2000 chars)
            analysis_type: Type of analysis (sentiment, competitor, summary, trends, insights, keywords)
            use_cache: Whether to use cached results (default: True)
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Analysis results as dictionary

        Raises:
            OllamaException: If analysis fails
        """
        # Truncate text to prevent token overflow
        text_truncated = text[:2000] if len(text) > 2000 else text

        # Generate cache key using MD5 hash
        text_hash = hashlib.md5(text_truncated.encode()).hexdigest()

        # Check cache first
        if use_cache:
            cached = await LLMCache.get_analysis(text_hash, analysis_type)
            if cached:
                self.stats["cache_hits"] += 1
                logger.debug(
                    f"Cache hit for {analysis_type} analysis (hash: {text_hash[:8]}...)"
                )
                return cached

        # Get prompt template
        prompt_template = self.prompts.get(
            analysis_type,
            self.prompts.get("summary", "Analyze the following text:\n\n{text}")
        )

        # Format prompt with text and any additional kwargs
        try:
            prompt = prompt_template.format(text=text_truncated, **kwargs)
        except KeyError as e:
            logger.warning(f"Missing prompt variable {e}, using text only")
            prompt = prompt_template.replace("{text}", text_truncated)

        try:
            # Make request to Ollama
            response_text = await self._request_ollama(prompt)

            # Parse response
            parsed = self._parse_response(response_text, analysis_type)

            # Add metadata
            parsed["_metadata"] = {
                "analysis_type": analysis_type,
                "model": self.model,
                "timestamp": datetime.utcnow().isoformat(),
                "text_length": len(text_truncated),
                "cached": False
            }

            # Cache result (2 hour TTL)
            if use_cache:
                await LLMCache.set_analysis(text_hash, analysis_type, parsed, ttl=7200)

            self.stats["requests"] += 1
            logger.info(f"Completed {analysis_type} analysis (hash: {text_hash[:8]}...)")

            return parsed

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ollama analysis failed for {analysis_type}: {e}")
            raise OllamaException(f"Analysis failed: {e}")

    async def _request_ollama(self, prompt: str) -> str:
        """
        Make HTTP request to Ollama API

        Args:
            prompt: The prompt to send to Ollama

        Returns:
            Response text from Ollama

        Raises:
            OllamaException: If request fails
        """
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

                # Update token count if available
                if "eval_count" in result:
                    self.stats["total_tokens"] += result.get("eval_count", 0)

                response_text = result.get("response", "")

                if not response_text:
                    raise OllamaException("Empty response from Ollama")

                logger.debug(f"Ollama response length: {len(response_text)} chars")
                return response_text

            except httpx.TimeoutException:
                raise OllamaException(
                    f"Ollama request timed out after {self.timeout}s"
                )
            except httpx.HTTPStatusError as e:
                raise OllamaException(
                    f"HTTP error {e.response.status_code}: {e.response.text}"
                )
            except httpx.HTTPError as e:
                raise OllamaException(f"HTTP error: {e}")
            except json.JSONDecodeError as e:
                raise OllamaException(f"Invalid JSON response: {e}")

    def _parse_response(
        self,
        response: str,
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Parse Ollama response based on analysis type

        Attempts to parse as JSON first, then falls back to type-specific parsing.

        Args:
            response: Response text from Ollama
            analysis_type: Type of analysis performed

        Returns:
            Parsed response as dictionary
        """
        # Try to parse as JSON first
        try:
            # Look for JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                logger.debug(f"Successfully parsed JSON response for {analysis_type}")
                return parsed
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON, using fallback parser for {analysis_type}")

        # Fallback parsing based on type
        if analysis_type == "sentiment":
            return self._parse_sentiment(response)
        elif analysis_type == "competitor":
            return self._parse_competitors(response)
        elif analysis_type == "trends":
            return self._parse_trends(response)
        elif analysis_type == "insights":
            return self._parse_insights(response)
        elif analysis_type == "keywords":
            return self._parse_keywords(response)
        else:
            # Generic fallback
            return {
                "analysis": response,
                "type": analysis_type
            }

    def _parse_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Parse sentiment from text response

        Args:
            text: Response text to parse

        Returns:
            Dictionary with sentiment, confidence, and analysis
        """
        sentiment = "neutral"
        confidence = 0.5

        text_lower = text.lower()

        # Check for sentiment indicators
        if "positive" in text_lower:
            sentiment = "positive"
            confidence = 0.8
        elif "negative" in text_lower:
            sentiment = "negative"
            confidence = 0.8

        # Try to extract confidence value
        confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', text_lower)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # Normalize to 0-1 range if needed
                if confidence > 1:
                    confidence = confidence / 100
            except ValueError:
                pass

        # Extract key phrases
        key_phrases = []
        phrases_match = re.search(
            r'key[_\s](?:phrases|emotions)[:\s]+(.+?)(?:\n\n|\n[A-Z]|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if phrases_match:
            phrases_text = phrases_match.group(1)
            # Split by common delimiters
            key_phrases = [
                p.strip(' -•*[]')
                for p in re.split(r'[,\n]', phrases_text)
                if p.strip(' -•*[]')
            ][:5]  # Limit to 5

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "key_phrases": key_phrases,
            "analysis": text
        }

    def _parse_competitors(self, text: str) -> Dict[str, Any]:
        """
        Parse competitors from text response

        Args:
            text: Response text to parse

        Returns:
            Dictionary with competitors, products, and analysis
        """
        # Extract capitalized words as potential competitors
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Remove common words
        common_words = {
            "The", "This", "That", "Analysis", "Text", "Company",
            "Product", "Service", "Market", "Customer", "Business"
        }
        competitors = [w for w in words if w not in common_words]

        # Deduplicate and limit
        competitors = list(dict.fromkeys(competitors))[:10]

        # Try to extract products
        products = []
        products_match = re.search(
            r'products?[:\s]+(.+?)(?:\n\n|\n[A-Z]|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if products_match:
            products_text = products_match.group(1)
            products = [
                p.strip(' -•*[]')
                for p in re.split(r'[,\n]', products_text)
                if p.strip(' -•*[]')
            ][:5]

        return {
            "competitors": competitors,
            "products": products,
            "analysis": text
        }

    def _parse_trends(self, text: str) -> Dict[str, Any]:
        """
        Parse trends from text response

        Args:
            text: Response text to parse

        Returns:
            Dictionary with trends, keywords, topics, and analysis
        """
        # Extract bullet points and numbered lists as trends
        lines = text.split('\n')
        trends = []
        keywords = []
        topics = []

        for line in lines:
            line_clean = line.strip(' -•*0123456789.')
            if line_clean and len(line_clean) < 150 and len(line_clean) > 10:
                # Categorize based on context
                if any(word in line_clean.lower() for word in ['trend', 'pattern', 'movement']):
                    trends.append(line_clean)
                elif any(word in line_clean.lower() for word in ['keyword', 'term', 'phrase']):
                    keywords.append(line_clean)
                elif any(word in line_clean.lower() for word in ['topic', 'theme', 'subject']):
                    topics.append(line_clean)
                else:
                    # Default to trends
                    if len(trends) < 10:
                        trends.append(line_clean)

        # Limit results
        trends = trends[:10]
        keywords = keywords[:10]
        topics = topics[:5]

        return {
            "trends": trends,
            "keywords": keywords,
            "topics": topics,
            "analysis": text
        }

    def _parse_insights(self, text: str) -> Dict[str, Any]:
        """
        Parse business insights from text response

        Args:
            text: Response text to parse

        Returns:
            Dictionary with opportunities, threats, recommendations, and analysis
        """
        opportunities = []
        threats = []
        recommendations = []

        # Extract sections
        sections = {
            "opportunities": r'opportunit(?:y|ies)[:\s]+(.+?)(?:\n\n|\n[A-Z]+:|\Z)',
            "threats": r'threats?[:\s]+(.+?)(?:\n\n|\n[A-Z]+:|\Z)',
            "recommendations": r'recommendations?[:\s]+(.+?)(?:\n\n|\n[A-Z]+:|\Z)'
        }

        for section_name, pattern in sections.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1)
                items = [
                    item.strip(' -•*[]0123456789.')
                    for item in re.split(r'\n', section_text)
                    if item.strip(' -•*[]0123456789.')
                ][:5]

                if section_name == "opportunities":
                    opportunities = items
                elif section_name == "threats":
                    threats = items
                elif section_name == "recommendations":
                    recommendations = items

        return {
            "opportunities": opportunities,
            "threats": threats,
            "recommendations": recommendations,
            "analysis": text
        }

    def _parse_keywords(self, text: str) -> Dict[str, Any]:
        """
        Parse keywords from text response

        Args:
            text: Response text to parse

        Returns:
            Dictionary with keywords and analysis
        """
        # Extract keywords from bullet points
        lines = text.split('\n')
        keywords = []

        for line in lines:
            line_clean = line.strip(' -•*0123456789.')
            if line_clean and len(line_clean) < 100:
                keywords.append(line_clean)

        keywords = keywords[:15]  # Top 15 keywords

        return {
            "keywords": keywords,
            "analysis": text
        }

    async def batch_analyze(
        self,
        texts: List[str],
        analysis_type: str = "sentiment",
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batches using concurrent requests

        Args:
            texts: List of texts to analyze
            analysis_type: Type of analysis to perform
            batch_size: Number of concurrent requests (default: 10)

        Returns:
            List of analysis results (same order as input texts)
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
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Batch analysis error for text {i+idx}: {result}"
                    )
                    results.append({
                        "error": str(result),
                        "analysis_type": analysis_type,
                        "_metadata": {
                            "analysis_type": analysis_type,
                            "timestamp": datetime.utcnow().isoformat(),
                            "cached": False,
                            "error": True
                        }
                    })
                else:
                    results.append(result)

        logger.info(
            f"Batch analysis completed: {len(texts)} texts, "
            f"type={analysis_type}, errors={sum(1 for r in results if 'error' in r)}"
        )

        return results


# Global Ollama client instance
ollama_client = OllamaClient()
