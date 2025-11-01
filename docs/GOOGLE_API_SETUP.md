# Google Custom Search API Setup Guide

This guide explains how to set up and use the Google Custom Search JSON API for programmatic web searching without CAPTCHA challenges.

## Why Use Google Custom Search API?

### Problems with Web Scraping
- ❌ **CAPTCHA challenges** - Google blocks automated browsers
- ❌ **Bot detection** - Sophisticated fingerprinting
- ❌ **IP bans** - Automated requests get blocked
- ❌ **Unreliable** - Requires proxies, CAPTCHA solvers, anti-detection
- ❌ **Legal risks** - May violate Terms of Service

### Benefits of Official API
- ✅ **No CAPTCHA** - Official Google service
- ✅ **No bot detection** - Authorized API access
- ✅ **Reliable** - 99.9% uptime SLA
- ✅ **Legal** - Fully authorized by Google
- ✅ **Structured data** - Clean JSON responses
- ✅ **100 free queries/day** - Perfect for development

## Setup Instructions

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name (e.g., "CIAP Search")
4. Click "Create"

### Step 2: Enable Custom Search API

1. Go to [API Library](https://console.cloud.google.com/apis/library)
2. Search for "Custom Search API"
3. Click "Custom Search API"
4. Click "Enable"

### Step 3: Create API Key

1. Go to [Credentials](https://console.cloud.google.com/apis/credentials)
2. Click "Create Credentials" → "API Key"
3. Copy the API key (e.g., `AIzaSyCBGgcUyjcurfrJGcCeofeGcXSgN2Ilwa4`)
4. (Optional) Click "Restrict Key" to add security:
   - Select "Restrict key" under API restrictions
   - Check "Custom Search API"
   - Save

### Step 4: Create Programmable Search Engine

1. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
2. Click "Get started" or "Add"
3. Configure your search engine:
   - **Sites to search**: "Search the entire web"
   - **Name**: "CIAP Search"
   - **Language**: English
4. Click "Create"
5. Copy the **Search Engine ID** (cx parameter)
   - Example: `e389ee2f38773444d`

### Step 5: Configure CIAP

Edit your `.env` file:

```bash
# Google Custom Search API
GOOGLE_API_KEY=AIzaSyCBGgcUyjcurfrJGcCeofeGcXSgN2Ilwa4
GOOGLE_SEARCH_ENGINE_ID=e389ee2f38773444d
GOOGLE_API_ENABLED=true
```

## Usage

### Using the API

Once configured, CIAP will automatically use the Google Custom Search API instead of web scraping when:
- `GOOGLE_API_ENABLED=true` in `.env`
- `GOOGLE_API_KEY` is set
- `GOOGLE_SEARCH_ENGINE_ID` is set

### Example: Direct API Usage

```python
from src.scrapers.google_api import search_google_api

results = await search_google_api(
    query="mango pickle",
    api_key="YOUR_API_KEY",
    search_engine_id="YOUR_SEARCH_ENGINE_ID",
    max_results=10,
    lang="en",
    region="us"
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}")
```

### Example: Using ScraperManager

```python
from src.scrapers.manager import ScraperManager

manager = ScraperManager()

# Automatically uses Google API if enabled
results = await manager.scrape(
    query="mango pickle",
    sources=["google"],  # Will use API instead of scraping
    max_results_per_source=10
)

google_results = results["google"]
```

### Example: Using the API Endpoint

```bash
# Start the CIAP server
python run.py

# Make a search request
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mango pickle",
    "sources": ["google"],
    "max_results_per_source": 10
  }'
```

## Pricing

### Free Tier
- **100 queries per day**: FREE
- **Reset**: Daily at midnight Pacific Time
- **Perfect for**: Development, testing, small projects

### Paid Tier
- **Beyond 100 queries**: $5 per 1000 queries
- **Maximum**: 10,000 queries per day
- **Billing**: Requires Google Cloud billing account
- **Cost example**:
  - 500 queries/day = $60/month (400 paid × $0.005)
  - 1,000 queries/day = $135/month (900 paid × $0.005)
  - 5,000 queries/day = $735/month (4,900 paid × $0.005)

### How to Enable Billing (for >100 queries/day)

1. Go to [Google Cloud Billing](https://console.cloud.google.com/billing)
2. Click "Link a billing account" or "Create billing account"
3. Enter payment information
4. Link to your project

## Monitoring Usage

### Check API Usage

1. Go to [API Dashboard](https://console.cloud.google.com/apis/dashboard)
2. Click "Custom Search API"
3. View metrics:
   - Requests per day
   - Errors
   - Latency

### Set Up Quotas/Alerts

1. Go to [Quotas](https://console.cloud.google.com/apis/api/customsearch.googleapis.com/quotas)
2. Set daily limit (e.g., 100 to stay in free tier)
3. Set up billing alerts to avoid unexpected charges

## API Features

### Supported Parameters

- **query** (required): Search query string
- **max_results**: Number of results (1-100, returns 10 per page)
- **lang**: Language filter (e.g., 'en', 'es', 'fr')
- **region**: Region/country code (e.g., 'us', 'uk', 'in')
- **safe_search**: Safe search level ('off', 'medium', 'high')

### Response Format

```json
{
  "title": "Page title",
  "url": "https://example.com/page",
  "snippet": "Brief description of the page...",
  "position": 1,
  "source": "google_api",
  "scraped_at": "2025-01-15T10:30:00",
  "display_url": "example.com",
  "formatted_url": "https://example.com › page"
}
```

## Troubleshooting

### Error: "API key not valid"

**Solution**:
1. Verify API key in Google Cloud Console
2. Check that Custom Search API is enabled
3. Ensure no IP restrictions blocking your server

### Error: "Billing not enabled"

**Solution**: You've exceeded 100 free queries. Either:
1. Wait for daily reset (midnight Pacific)
2. Enable billing in Google Cloud Console

### Error: "Search engine ID not found"

**Solution**:
1. Verify Search Engine ID in [Programmable Search Console](https://programmablesearchengine.google.com/)
2. Ensure search engine is configured to "Search the entire web"

### No results returned

**Possible causes**:
1. Search engine restricted to specific sites (should be "entire web")
2. Query too specific or has no results
3. API quota exceeded

## Comparison: API vs Scraping vs Alternatives

### Google Custom Search API
- ✅ Official, legal, reliable
- ✅ 100 free queries/day
- ✅ No CAPTCHA, no bot detection
- ❌ $5 per 1000 queries after free tier
- ❌ 10k queries/day maximum

### Web Scraping (Scrapy + Playwright)
- ✅ Unlimited queries (if you can avoid detection)
- ❌ CAPTCHA challenges
- ❌ Bot detection and IP bans
- ❌ Requires proxies, CAPTCHA solvers (~$50-200/month)
- ❌ Legal gray area

### Alternative APIs

**SerpAPI** ($50/month for 5k searches)
- More expensive than Google API
- Supports multiple search engines
- No daily limits
- Good for high volume

**ScraperAPI** ($149/month for 40k searches)
- Best for very high volume
- Includes proxy rotation
- More features than basic search

**Scrapingdog** ($0.00029 per request at scale)
- Cheapest at very high scale (100k+ queries/month)
- Fast response times
- Limited to basic search

## Recommendation

### For CIAP Development
Use **Google Custom Search API** with free tier (100 queries/day)

### For CIAP Production (Low Volume)
Use **Google Custom Search API** with billing enabled
- Cost-effective up to ~1,000 queries/day
- Most reliable option

### For CIAP Production (High Volume)
Consider **SerpAPI** or **ScraperAPI**
- Better pricing at 10k+ queries/day
- More features and flexibility

## Security Best Practices

1. **Never commit API keys to Git**
   - Keep in `.env` file (already in `.gitignore`)
   - Use environment variables in production

2. **Restrict API key**
   - Limit to Custom Search API only
   - Add IP restrictions if server has static IP
   - Rotate keys periodically

3. **Monitor usage**
   - Set up billing alerts
   - Track API quotas
   - Log all API requests

4. **Handle errors gracefully**
   - Implement retry logic
   - Fall back to scraping if API fails
   - Cache results to reduce API calls

## Next Steps

1. ✅ API is configured and working
2. Test with various queries
3. Monitor daily usage
4. Consider enabling billing for production
5. Implement caching to reduce API calls

## Support

- **Google Cloud Support**: https://cloud.google.com/support
- **API Documentation**: https://developers.google.com/custom-search/v1/overview
- **Programmable Search Help**: https://support.google.com/programmable-search
- **CIAP Issues**: https://github.com/fimilfaneea/ciap_backend/issues
