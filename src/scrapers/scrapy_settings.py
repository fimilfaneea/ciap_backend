"""
Scrapy Settings for CIAP Web Scraping
Dynamic configuration loaded from src.config.settings
"""

from ..config import settings as app_settings

# Bot identification
BOT_NAME = "ciap_scraper"
USER_AGENT = "CIAP-Bot/1.0 (+https://github.com/yourusername/ciap)"

# Spider modules
SPIDER_MODULES = ["src.scrapers.spiders"]
NEWSPIDER_MODULE = "src.scrapers.spiders"

# Obey robots.txt (set to False for search engines as they don't allow scraping)
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = app_settings.SCRAPY_CONCURRENT_REQUESTS
CONCURRENT_REQUESTS_PER_DOMAIN = 8
CONCURRENT_REQUESTS_PER_IP = 8

# Configure download delay
DOWNLOAD_DELAY = app_settings.SCRAPY_DOWNLOAD_DELAY
# The download delay setting will honor only one of:
# CONCURRENT_REQUESTS_PER_DOMAIN = 16
# CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (search engines can track via cookies)
COOKIES_ENABLED = False

# Disable Telnet Console
TELNETCONSOLE_ENABLED = False

# Override the default request headers
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
    "scrapy.spidermiddlewares.httperror.HttpErrorMiddleware": 50,
    "scrapy.spidermiddlewares.offsite.OffsiteMiddleware": 500,
    "scrapy.spidermiddlewares.referer.RefererMiddleware": 700,
    "scrapy.spidermiddlewares.urllength.UrlLengthMiddleware": 800,
    "scrapy.spidermiddlewares.depth.DepthMiddleware": 900,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
    "scrapy.downloadermiddlewares.retry.RetryMiddleware": 550,
}

# Configure Playwright if enabled
if app_settings.SCRAPY_PLAYWRIGHT_ENABLED:
    DOWNLOAD_HANDLERS = {
        "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    }

    DOWNLOADER_MIDDLEWARES.update({
        "scrapy_playwright.middleware.ScrapyPlaywrightDownloadMiddleware": 585,
    })

    PLAYWRIGHT_BROWSER_TYPE = app_settings.SCRAPY_PLAYWRIGHT_BROWSER
    PLAYWRIGHT_LAUNCH_OPTIONS = {
        "headless": app_settings.SCRAPY_PLAYWRIGHT_HEADLESS,
        "timeout": 30000,  # 30 seconds
    }

    # Playwright contexts (for browser sessions)
    PLAYWRIGHT_CONTEXTS = {
        "default": {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": app_settings.get_user_agent(),
        },
    }

    # Abort requests for these resource types (save bandwidth)
    PLAYWRIGHT_ABORT_REQUEST = lambda req: req.resource_type in ["image", "stylesheet", "font", "media"]

# Enable and configure the AutoThrottle extension
if app_settings.SCRAPY_AUTOTHROTTLE_ENABLED:
    AUTOTHROTTLE_ENABLED = True
    AUTOTHROTTLE_START_DELAY = 1
    AUTOTHROTTLE_MAX_DELAY = 10
    AUTOTHROTTLE_TARGET_CONCURRENCY = app_settings.SCRAPY_AUTOTHROTTLE_TARGET_CONCURRENCY
    AUTOTHROTTLE_DEBUG = False

# Configure item pipelines
ITEM_PIPELINES = {
    "src.scrapers.pipelines.ValidationPipeline": 100,
    "src.scrapers.pipelines.CleaningPipeline": 200,
    "src.scrapers.pipelines.DeduplicationPipeline": 300,
    "src.scrapers.pipelines.CollectorPipeline": 900,
}

# Configure retry settings
RETRY_ENABLED = True
RETRY_TIMES = app_settings.SCRAPER_RETRY_COUNT
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# Redirect settings
REDIRECT_ENABLED = True
REDIRECT_MAX_TIMES = 3

# HTTP cache (disabled by default, enable for development)
HTTPCACHE_ENABLED = False
HTTPCACHE_EXPIRATION_SECS = 0
HTTPCACHE_DIR = "data/scrapy_httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = [403, 404, 429, 500, 502, 503]
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Logging
LOG_LEVEL = app_settings.SCRAPY_LOG_LEVEL
LOG_FORMAT = "%(levelname)s: %(message)s"
LOG_DATEFORMAT = "%Y-%m-%d %H:%M:%S"

# Request settings
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
# TWISTED_REACTOR - Managed by crochet, do not set here
FEED_EXPORT_ENCODING = "utf-8"

# Closespider settings (safety limits)
CLOSESPIDER_TIMEOUT = 300  # 5 minutes max per spider
CLOSESPIDER_ITEMCOUNT = 500  # Max 500 items per spider run
CLOSESPIDER_PAGECOUNT = 100  # Max 100 pages per spider run
CLOSESPIDER_ERRORCOUNT = 50  # Stop after 50 errors

# DNS settings
DNSCACHE_ENABLED = True
DNSCACHE_SIZE = 10000

# Memory usage settings
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 1024  # 1GB limit
MEMUSAGE_WARNING_MB = 512  # Warn at 512MB

# Depth limit (for pagination)
DEPTH_LIMIT = 20  # Max 20 pages deep
DEPTH_STATS_VERBOSE = True
