"""
Google Custom Search JSON API Integration

Official Google API for programmatic search - no CAPTCHA, no bot detection.
- 100 free queries per day
- $5 per 1000 queries after that (max 10k/day)
- Structured JSON results

Setup:
1. Create Custom Search Engine: https://programmablesearchengine.google.com/
2. Get API Key: https://console.cloud.google.com/apis/credentials
3. Set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


class GoogleCustomSearchAPI:
    """
    Google Custom Search JSON API client

    Official Google API for programmatic web search.
    Returns structured JSON data without CAPTCHA challenges.
    """

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        timeout: int = 30
    ):
        """
        Initialize Google Custom Search API client

        Args:
            api_key: Google API key from Cloud Console
            search_engine_id: Custom Search Engine ID (cx parameter)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.timeout = timeout

        logger.info(
            f"GoogleCustomSearchAPI initialized: "
            f"engine_id={search_engine_id[:8]}..., timeout={timeout}s"
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        lang: str = "en",
        region: Optional[str] = None,
        date_range: Optional[str] = None,
        safe_search: str = "off"
    ) -> List[Dict[str, Any]]:
        """
        Search Google using Custom Search API

        Args:
            query: Search query
            max_results: Maximum results to return (max 100, API returns 10 per page)
            lang: Language code (e.g., 'en', 'es')
            region: Region code (e.g., 'us', 'uk') - optional
            date_range: Date filter - not supported by Custom Search API
            safe_search: Safe search setting ('off', 'medium', 'high')

        Returns:
            List of search results with title, url, snippet, position
        """
        logger.info(
            f"Searching Google Custom Search API: query='{query}', "
            f"max_results={max_results}, lang={lang}"
        )

        results = []
        num_pages = min((max_results + 9) // 10, 10)  # Max 10 pages (100 results)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for page in range(num_pages):
                start_index = page * 10 + 1

                try:
                    page_results = await self._fetch_page(
                        client=client,
                        query=query,
                        start=start_index,
                        lang=lang,
                        region=region,
                        safe_search=safe_search
                    )

                    results.extend(page_results)

                    logger.debug(
                        f"Page {page + 1}: Retrieved {len(page_results)} results "
                        f"(total: {len(results)})"
                    )

                    # Stop if we have enough results
                    if len(results) >= max_results:
                        break

                    # Stop if no more results available
                    if len(page_results) < 10:
                        break

                    # Rate limiting: small delay between requests
                    if page < num_pages - 1:
                        await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error fetching page {page + 1}: {e}")
                    break

        # Trim to exact max_results
        results = results[:max_results]

        logger.info(f"Search completed: {len(results)} results retrieved")
        return results

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        query: str,
        start: int,
        lang: str,
        region: Optional[str],
        safe_search: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch a single page of search results

        Args:
            client: HTTP client
            query: Search query
            start: Start index (1-91, increments of 10)
            lang: Language code
            region: Region code (optional)
            safe_search: Safe search setting

        Returns:
            List of results from this page
        """
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "start": start,
            "num": 10,  # Results per page (max 10)
        }

        # Add optional parameters
        if lang:
            params["lr"] = f"lang_{lang}"

        if region:
            params["gl"] = region

        if safe_search and safe_search != "off":
            params["safe"] = safe_search

        try:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()

            data = response.json()

            # Parse results
            results = []
            items = data.get("items", [])

            for item in items:
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": len(results) + start,
                    "source": "google_api",
                    "scraped_at": datetime.utcnow().isoformat(),
                }

                # Optional fields
                if "displayLink" in item:
                    result["display_url"] = item["displayLink"]

                if "formattedUrl" in item:
                    result["formatted_url"] = item["formattedUrl"]

                results.append(result)

            return results

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error("API quota exceeded (429 Too Many Requests)")
                raise Exception("Google Custom Search API quota exceeded")
            elif e.response.status_code == 403:
                logger.error("API key invalid or API not enabled (403 Forbidden)")
                raise Exception("Invalid API key or API not enabled")
            else:
                logger.error(f"HTTP error {e.response.status_code}: {e}")
                raise

        except Exception as e:
            logger.error(f"Error fetching results: {e}")
            raise


async def search_google_api(
    query: str,
    api_key: str,
    search_engine_id: str,
    max_results: int = 10,
    lang: str = "en",
    region: Optional[str] = None,
    timeout: int = 30
) -> List[Dict[str, Any]]:
    """
    Convenience function to search Google using Custom Search API

    Args:
        query: Search query
        api_key: Google API key
        search_engine_id: Custom Search Engine ID
        max_results: Maximum results to return
        lang: Language code
        region: Region code (optional)
        timeout: Request timeout

    Returns:
        List of search results
    """
    api = GoogleCustomSearchAPI(
        api_key=api_key,
        search_engine_id=search_engine_id,
        timeout=timeout
    )

    return await api.search(
        query=query,
        max_results=max_results,
        lang=lang,
        region=region
    )
