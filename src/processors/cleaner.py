"""
Data Cleaning and Normalization Module for CIAP

Provides text cleaning, URL normalization, data standardization,
and duplicate detection for scraped search results.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import unicodedata

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and sanitize scraped data"""

    @staticmethod
    def clean_text(text: str, max_length: int = 500) -> str:
        """
        Clean and normalize text content

        Args:
            text: Raw text to clean
            max_length: Maximum length (will truncate with ...)

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Remove HTML tags if present
        text = BeautifulSoup(text, "html.parser").get_text()

        # Normalize unicode (NFKD: compatibility decomposition)
        text = unicodedata.normalize("NFKD", text)

        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
        text = ' '.join(text.split())

        # Remove extra punctuation (collapse multiple punctuation marks)
        text = re.sub(r'([.!?,;])\1+', r'\1', text)

        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length-3] + "..."

        return text.strip()

    @staticmethod
    def clean_url(url: str) -> str:
        """
        Clean and normalize URLs

        Removes tracking parameters while preserving functional parameters.

        Args:
            url: URL to clean

        Returns:
            Cleaned URL without tracking parameters
        """
        if not url:
            return ""

        # Remove tracking parameters
        tracking_params = [
            'utm_source', 'utm_medium', 'utm_campaign',
            'utm_term', 'utm_content', 'fbclid', 'gclid',
            'msclkid', '_ga', 'mc_cid', 'mc_eid'
        ]

        try:
            parsed = urlparse(url)

            # Keep only non-tracking params
            query_pairs = []
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key = param.split('=')[0]
                        if key not in tracking_params:
                            query_pairs.append(param)

            clean_query = '&'.join(query_pairs) if query_pairs else ''
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            if clean_query:
                clean_url += f"?{clean_query}"

            # Remove trailing slash if no path
            if parsed.path == '/':
                clean_url = clean_url.rstrip('/')

            return clean_url

        except Exception as e:
            logger.warning(f"Error cleaning URL {url}: {e}")
            return url

    @staticmethod
    def extract_domain(url: str) -> str:
        """
        Extract domain from URL

        Args:
            url: Full URL

        Returns:
            Domain name (without www.)
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception as e:
            logger.warning(f"Error extracting domain from {url}: {e}")
            return ""

    @staticmethod
    def clean_html(html: str) -> str:
        """
        Remove unnecessary HTML elements and extract text

        Args:
            html: HTML content

        Returns:
            Clean text without HTML tags
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link", "noscript"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines
                     for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            logger.warning(f"Error cleaning HTML: {e}")
            return html


class DataNormalizer:
    """Normalize data to consistent format"""

    @staticmethod
    def normalize_search_result(
        result: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """
        Normalize search result to standard format

        Args:
            result: Raw search result dict
            source: Source name (google, bing, etc.)

        Returns:
            Normalized result dict with standard fields
        """
        cleaner = DataCleaner()

        normalized = {
            # Required fields
            "title": cleaner.clean_text(result.get("title", ""), 200),
            "snippet": cleaner.clean_text(result.get("snippet", ""), 500),
            "url": cleaner.clean_url(result.get("url", "")),
            "domain": cleaner.extract_domain(result.get("url", "")),
            "source": source.lower(),
            "position": int(result.get("position", 0)),

            # Timestamps
            "scraped_at": result.get("scraped_at", datetime.utcnow()),
            "normalized_at": datetime.utcnow(),

            # Optional metadata
            "metadata": result.get("metadata", {}),

            # Generate unique ID
            "result_id": DataNormalizer._generate_id(
                result.get("url", ""),
                source
            )
        }

        # Add quality score
        normalized["quality_score"] = DataNormalizer._calculate_quality(
            normalized
        )

        return normalized

    @staticmethod
    def _generate_id(url: str, source: str) -> str:
        """
        Generate unique ID for result

        Args:
            url: Result URL
            source: Source name

        Returns:
            MD5 hash of url:source
        """
        content = f"{url}:{source}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _calculate_quality(result: Dict) -> float:
        """
        Calculate quality score for result

        Scoring criteria:
        - Has meaningful title (30%)
        - Has meaningful snippet (30%)
        - Valid URL (20%)
        - Position bonus (20% - higher position = better)

        Args:
            result: Normalized result dict

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0

        # Has title (min 10 chars)
        if result["title"] and len(result["title"]) > 10:
            score += 0.3

        # Has snippet (min 50 chars)
        if result["snippet"] and len(result["snippet"]) > 50:
            score += 0.3

        # Valid URL
        if result["url"].startswith(("http://", "https://")):
            score += 0.2

        # Position bonus (higher position = better)
        # Top 10 results get bonus, position 1 gets full 0.2, position 10 gets 0.02
        if 1 <= result["position"] <= 10:
            score += 0.2 * (11 - result["position"]) / 10

        return min(score, 1.0)


class Deduplicator:
    """Remove duplicate results using URL and content similarity"""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize deduplicator

        Args:
            similarity_threshold: Minimum similarity (0.0-1.0) to consider duplicate
        """
        self.similarity_threshold = similarity_threshold
        self.seen_urls = set()
        self.seen_content = {}

        logger.debug(f"Deduplicator initialized with threshold={similarity_threshold}")

    def deduplicate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicates from results

        Uses two strategies:
        1. Exact URL matching
        2. Content similarity (Jaccard similarity on word sets)

        Args:
            results: List of normalized result dicts

        Returns:
            List of unique results (duplicates removed)
        """
        unique_results = []
        duplicates_removed = 0

        for result in results:
            # Check URL duplicate (exact match)
            url = result.get("url", "")
            if url in self.seen_urls:
                duplicates_removed += 1
                continue

            # Check content similarity
            content_hash = self._content_hash(result)
            is_duplicate = False

            for seen_hash, seen_result in self.seen_content.items():
                similarity = self._calculate_similarity(content_hash, seen_hash)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    logger.debug(
                        f"Content duplicate detected (similarity={similarity:.2f}): "
                        f"{result.get('title', 'N/A')[:50]}"
                    )
                    break

            if not is_duplicate:
                self.seen_urls.add(url)
                self.seen_content[content_hash] = result
                unique_results.append(result)

        logger.info(
            f"Deduplication complete: {len(results)} input, "
            f"{len(unique_results)} unique, {duplicates_removed} duplicates removed"
        )

        return unique_results

    def _content_hash(self, result: Dict) -> str:
        """
        Generate content hash for similarity check

        Creates a word-based hash from title and snippet.

        Args:
            result: Result dict

        Returns:
            Space-separated sorted word list (normalized)
        """
        content = f"{result.get('title', '')} {result.get('snippet', '')}"
        # Normalize: lowercase, split into words, sort, remove duplicates
        words = set(content.lower().split())
        return ' '.join(sorted(words))

    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate Jaccard similarity between content hashes

        Jaccard similarity = |intersection| / |union|

        Args:
            hash1: First content hash
            hash2: Second content hash

        Returns:
            Similarity score between 0.0 and 1.0
        """
        words1 = set(hash1.split())
        words2 = set(hash2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def reset(self):
        """Reset deduplicator state (clear seen URLs and content)"""
        self.seen_urls.clear()
        self.seen_content.clear()
        logger.debug("Deduplicator state reset")
