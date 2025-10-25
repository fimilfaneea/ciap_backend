    # Module 6: Data Processing System

    ## Overview
    **Purpose:** Process, clean, normalize, and deduplicate scraped data for analysis.

    **Responsibilities:**
    - Text cleaning and normalization
    - Data standardization across sources
    - Duplicate detection and removal
    - Data quality scoring
    - Metadata extraction

    **Development Time:** 2 days (Week 5, Day 15-17)

    ---

    ## Implementation Guide

    ### Core Data Processor (`src/processors/cleaner.py`)

    ```python
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
            """Clean and normalize text content"""
            if not text:
                return ""

            # Remove HTML tags if present
            text = BeautifulSoup(text, "html.parser").get_text()

            # Normalize unicode
            text = unicodedata.normalize("NFKD", text)

            # Remove control characters
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

            # Normalize whitespace
            text = ' '.join(text.split())

            # Remove extra punctuation
            text = re.sub(r'([.!?,;])\1+', r'\1', text)

            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length-3] + "..."

            return text.strip()

        @staticmethod
        def clean_url(url: str) -> str:
            """Clean and normalize URLs"""
            if not url:
                return ""

            # Remove tracking parameters
            tracking_params = [
                'utm_source', 'utm_medium', 'utm_campaign',
                'utm_term', 'utm_content', 'fbclid', 'gclid'
            ]

            parsed = urlparse(url)
            # Keep only non-tracking params
            query_pairs = []
            if parsed.query:
                for param in parsed.query.split('&'):
                    key = param.split('=')[0]
                    if key not in tracking_params:
                        query_pairs.append(param)

            clean_query = '&'.join(query_pairs) if query_pairs else ''
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            if clean_query:
                clean_url += f"?{clean_query}"

            return clean_url

        @staticmethod
        def extract_domain(url: str) -> str:
            """Extract domain from URL"""
            try:
                parsed = urlparse(url)
                return parsed.netloc.lower().replace('www.', '')
            except:
                return ""

        @staticmethod
        def clean_html(html: str) -> str:
            """Remove unnecessary HTML elements"""
            if not html:
                return ""

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines
                    for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text


    class DataNormalizer:
        """Normalize data to consistent format"""

        @staticmethod
        def normalize_search_result(
            result: Dict[str, Any],
            source: str
        ) -> Dict[str, Any]:
            """Normalize search result to standard format"""
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

                # Generate ID
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
            """Generate unique ID for result"""
            content = f"{url}:{source}"
            return hashlib.md5(content.encode()).hexdigest()

        @staticmethod
        def _calculate_quality(result: Dict) -> float:
            """Calculate quality score for result"""
            score = 0.0

            # Has title
            if result["title"] and len(result["title"]) > 10:
                score += 0.3

            # Has snippet
            if result["snippet"] and len(result["snippet"]) > 50:
                score += 0.3

            # Valid URL
            if result["url"].startswith(("http://", "https://")):
                score += 0.2

            # Position bonus (higher position = better)
            if result["position"] <= 10:
                score += 0.2 * (11 - result["position"]) / 10

            return min(score, 1.0)


    class Deduplicator:
        """Remove duplicate results"""

        def __init__(self, similarity_threshold: float = 0.85):
            self.similarity_threshold = similarity_threshold
            self.seen_urls = set()
            self.seen_content = {}

        def deduplicate(
            self,
            results: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Remove duplicates from results"""
            unique_results = []

            for result in results:
                # Check URL duplicate
                url = result.get("url", "")
                if url in self.seen_urls:
                    continue

                # Check content similarity
                content_hash = self._content_hash(result)
                is_duplicate = False

                for seen_hash, seen_result in self.seen_content.items():
                    if self._calculate_similarity(
                        content_hash,
                        seen_hash
                    ) > self.similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    self.seen_urls.add(url)
                    self.seen_content[content_hash] = result
                    unique_results.append(result)

            return unique_results

        def _content_hash(self, result: Dict) -> str:
            """Generate content hash for similarity check"""
            content = f"{result.get('title', '')} {result.get('snippet', '')}"
            # Simple word-based hash
            words = set(content.lower().split())
            return ' '.join(sorted(words))

        def _calculate_similarity(self, hash1: str, hash2: str) -> float:
            """Calculate similarity between content hashes"""
            words1 = set(hash1.split())
            words2 = set(hash2.split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0
    ```

    ### Batch Processor (`src/processors/batch_processor.py`)

    ```python
    from typing import List, Dict, Any, Tuple
    import asyncio
    import logging

    from src.processors.cleaner import DataCleaner, DataNormalizer, Deduplicator
    from src.core.database import db_manager
    from src.core.models import SearchResult

    logger = logging.getLogger(__name__)


    class BatchProcessor:
        """Process data in batches for efficiency"""

        def __init__(self, batch_size: int = 100):
            self.batch_size = batch_size
            self.cleaner = DataCleaner()
            self.normalizer = DataNormalizer()
            self.deduplicator = Deduplicator()

        async def process_search_results(
            self,
            results: List[Dict],
            source: str,
            search_id: int
        ) -> Tuple[List[Dict], Dict[str, int]]:
            """
            Process search results in batches

            Returns:
                Processed results and statistics
            """
            stats = {
                "total": len(results),
                "cleaned": 0,
                "duplicates": 0,
                "saved": 0
            }

            # Process in batches
            processed_results = []

            for i in range(0, len(results), self.batch_size):
                batch = results[i:i + self.batch_size]

                # Clean and normalize
                normalized_batch = []
                for result in batch:
                    try:
                        normalized = self.normalizer.normalize_search_result(
                            result, source
                        )
                        normalized_batch.append(normalized)
                        stats["cleaned"] += 1
                    except Exception as e:
                        logger.error(f"Error normalizing result: {e}")

                # Deduplicate
                unique_batch = self.deduplicator.deduplicate(normalized_batch)
                stats["duplicates"] += len(normalized_batch) - len(unique_batch)

                # Save to database
                if unique_batch:
                    await self._save_batch(unique_batch, search_id)
                    stats["saved"] += len(unique_batch)

                processed_results.extend(unique_batch)

            return processed_results, stats

        async def _save_batch(
            self,
            results: List[Dict],
            search_id: int
        ):
            """Save batch to database"""
            async with db_manager.get_session() as session:
                for result in results:
                    db_result = SearchResult(
                        search_id=search_id,
                        source=result["source"],
                        title=result["title"],
                        snippet=result["snippet"],
                        url=result["url"],
                        position=result["position"],
                        scraped_at=result["scraped_at"]
                    )
                    session.add(db_result)

                await session.commit()
    ```

    ---

    ## Testing

    ```python
    def test_text_cleaning():
        cleaner = DataCleaner()

        # Test HTML removal
        assert cleaner.clean_text("<p>Hello</p>") == "Hello"

        # Test whitespace normalization
        assert cleaner.clean_text("  multiple   spaces  ") == "multiple spaces"

        # Test length limiting
        long_text = "x" * 1000
        cleaned = cleaner.clean_text(long_text, max_length=100)
        assert len(cleaned) == 100

    def test_deduplication():
        dedup = Deduplicator()

        results = [
            {"url": "http://example.com/1", "title": "Same Title"},
            {"url": "http://example.com/2", "title": "Same Title"},  # Duplicate
            {"url": "http://example.com/3", "title": "Different"},
        ]

        unique = dedup.deduplicate(results)
        assert len(unique) == 2  # One duplicate removed
    ```

    ---

    ## Module Checklist

    - [ ] Text cleaning implemented
    - [ ] URL normalization working
    - [ ] Data standardization complete
    - [ ] Deduplication functional
    - [ ] Quality scoring added
    - [ ] Batch processing working
    - [ ] Database integration
    - [ ] Unit tests passing

    ---

    ## Next Steps
    - Module 7: Analyzer - Analyze processed data
    - Module 8: API - Expose processing endpoints