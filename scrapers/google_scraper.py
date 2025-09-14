"""Simple Google search scraper using requests and BeautifulSoup"""
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urlparse
import logging

from config import USER_AGENT, SCRAPE_DELAY, MAX_RESULTS_PER_SEARCH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleScraper:
    """Simple Google search scraper"""

    def __init__(self):
        self.base_url = "https://www.google.com/search"
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Perform a Google search and return results

        Args:
            query: Search query string
            num_results: Number of results to fetch

        Returns:
            List of search results with title, url, snippet
        """
        results = []
        num_results = min(num_results, MAX_RESULTS_PER_SEARCH)

        try:
            # Prepare search parameters
            params = {
                "q": query,
                "num": num_results,
                "hl": "en"
            }

            logger.info(f"Searching Google for: {query}")

            # Make request
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "lxml")

            # Find search result divs
            search_divs = soup.find_all("div", class_="g")

            for position, div in enumerate(search_divs[:num_results], 1):
                result = self._parse_result(div, position)
                if result:
                    results.append(result)

            logger.info(f"Found {len(results)} results for query: {query}")

            # Respect rate limiting
            time.sleep(SCRAPE_DELAY)

        except requests.RequestException as e:
            logger.error(f"Request error during Google search: {e}")
        except Exception as e:
            logger.error(f"Error during Google search: {e}")

        return results

    def _parse_result(self, div, position: int) -> Optional[Dict]:
        """Parse a single search result div"""
        try:
            # Find title and URL
            link_elem = div.find("a")
            if not link_elem:
                return None

            url = link_elem.get("href", "")
            if not url or url.startswith("/search"):
                return None

            # Get title
            title_elem = div.find("h3")
            title = title_elem.text if title_elem else ""

            # Get snippet
            snippet_elem = div.find("span", class_="aCOpRe") or div.find("span", class_="hgKElc")
            if not snippet_elem:
                # Try alternative snippet locations
                snippet_elem = div.find("div", {"data-sncf": "1"}) or div.find("div", class_="VwiC3b")

            snippet = snippet_elem.text if snippet_elem else ""

            # Extract domain
            domain = urlparse(url).netloc

            return {
                "position": position,
                "title": title.strip(),
                "url": url,
                "snippet": snippet.strip(),
                "domain": domain,
                "source": "google"
            }

        except Exception as e:
            logger.debug(f"Error parsing result: {e}")
            return None

    def search_competitor(self, company_name: str) -> List[Dict]:
        """Search for competitor information"""
        queries = [
            f"{company_name} competitors",
            f"{company_name} vs",
            f"alternatives to {company_name}",
            f"{company_name} market share"
        ]

        all_results = []
        for query in queries:
            results = self.search(query, num_results=5)
            for result in results:
                result["search_query"] = query
                all_results.append(result)

        return all_results

    def search_market_trends(self, industry: str) -> List[Dict]:
        """Search for market trends in an industry"""
        queries = [
            f"{industry} market trends 2024",
            f"{industry} industry analysis",
            f"{industry} market growth forecast",
            f"{industry} emerging technologies"
        ]

        all_results = []
        for query in queries:
            results = self.search(query, num_results=5)
            for result in results:
                result["search_query"] = query
                all_results.append(result)

        return all_results

    def search_customer_sentiment(self, product_or_company: str) -> List[Dict]:
        """Search for customer sentiment and reviews"""
        queries = [
            f"{product_or_company} reviews",
            f"{product_or_company} customer feedback",
            f"{product_or_company} user experience",
            f"{product_or_company} complaints"
        ]

        all_results = []
        for query in queries:
            results = self.search(query, num_results=5)
            for result in results:
                result["search_query"] = query
                all_results.append(result)

        return all_results


# Standalone function for easy testing
def test_scraper():
    """Test the Google scraper with a sample query"""
    scraper = GoogleScraper()

    # Test basic search
    print("\n=== Testing basic search ===")
    results = scraper.search("Python web scraping", num_results=3)
    for r in results:
        print(f"\nPosition {r['position']}:")
        print(f"  Title: {r['title'][:60]}...")
        print(f"  URL: {r['url'][:60]}...")
        print(f"  Snippet: {r['snippet'][:100]}...")

    # Test competitor search
    print("\n=== Testing competitor search ===")
    comp_results = scraper.search_competitor("Slack")
    print(f"Found {len(comp_results)} competitor-related results")
    if comp_results:
        print(f"Sample: {comp_results[0]['title'][:60]}...")


if __name__ == "__main__":
    test_scraper()