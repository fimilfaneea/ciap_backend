"""
Database Models for CIAP - Competitive Intelligence Automation Platform
Includes all data models for competitive intelligence, pricing, reviews, and analysis
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Float, Boolean, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# =========================
# EXISTING MODELS (Migrated)
# =========================

class Search(Base):
    """Main search queries and metadata"""
    __tablename__ = "searches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String(500), nullable=False)
    search_type = Column(String(50), default="competitor")  # competitor, market, product
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    sources = Column(JSON)  # ["google", "bing"]
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    user_id = Column(String(100), nullable=True)  # For multi-user support later
    search_metadata = Column(JSON, nullable=True)  # Store additional search parameters

    # Relationships
    search_results = relationship("SearchResult", back_populates="search", cascade="all, delete-orphan")
    serp_data = relationship("SERPData", back_populates="search", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="search", cascade="all, delete-orphan")


class SearchResult(Base):
    """Scraped search results"""
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("searches.id"))
    source = Column(String(50))  # google, bing, etc.
    title = Column(String(500))
    snippet = Column(Text)
    url = Column(String(1000))
    position = Column(Integer)  # Ranking position
    scraped_at = Column(DateTime, default=func.now())
    raw_content = Column(Text, nullable=True)  # Full scraped content if available

    # Analysis results
    sentiment_score = Column(Float, nullable=True)
    competitor_mentioned = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)
    result_metadata = Column(JSON, nullable=True)  # Additional scraped data

    # Relationships
    search = relationship("Search", back_populates="search_results")


class Analysis(Base):
    """LLM analysis results"""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    search_id = Column(Integer, ForeignKey("searches.id"))
    analysis_type = Column(String(50))  # sentiment, competitor, trend, summary
    content = Column(Text)  # The actual analysis text
    insights = Column(JSON)  # Structured insights as JSON
    sentiment_score = Column(Float, nullable=True)  # -1 to 1 sentiment scale
    confidence_score = Column(Float, nullable=True)  # 0 to 1 confidence scale
    llm_provider = Column(String(20))  # openai, anthropic, ollama
    llm_model = Column(String(50))  # specific model used
    created_at = Column(DateTime, default=func.now())
    analysis_metadata = Column(JSON, nullable=True)  # Additional analysis data

    # Relationships
    search = relationship("Search", back_populates="analyses")


# =========================
# NEW COMPETITIVE INTELLIGENCE MODELS
# =========================

class Product(Base):
    """Product/Service Entity Model - Core product information"""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_name = Column(String(500), nullable=False)
    brand_name = Column(String(200))
    company_name = Column(String(200))
    category = Column(String(100))
    industry = Column(String(100))
    sku = Column(String(100), unique=True, nullable=True)
    product_url = Column(String(1000))
    description = Column(Text)
    images = Column(JSON)  # List of image URLs
    media_urls = Column(JSON)  # List of media URLs
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    price_data = relationship("PriceData", back_populates="product", cascade="all, delete-orphan")
    offers = relationship("Offer", back_populates="product", cascade="all, delete-orphan")
    reviews = relationship("ProductReview", back_populates="product", cascade="all, delete-orphan")
    features = relationship("FeatureComparison", back_populates="product", cascade="all, delete-orphan")
    competitor_relationships = relationship("CompetitorProducts", back_populates="product", cascade="all, delete-orphan")
    insights = relationship("Insights", back_populates="product", cascade="all, delete-orphan")


class PriceData(Base):
    """Pricing Intelligence Model - Track price changes and availability"""
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    current_price = Column(Float)
    original_price = Column(Float)
    currency = Column(String(10))
    discount_percentage = Column(Float)
    availability_status = Column(String(50))  # In Stock, Out of Stock, Pre-order
    seller_name = Column(String(200))
    shipping_cost = Column(Float, nullable=True)
    geographic_location = Column(String(100))
    price_history = Column(JSON, nullable=True)  # Time series data
    scraped_at = Column(DateTime, default=func.now())

    # Relationships
    product = relationship("Product", back_populates="price_data")


class Offer(Base):
    """Offers & Promotions Model - Track special offers and deals"""
    __tablename__ = "offers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    offer_type = Column(String(50))  # Discount, BOGO, Bundle, Cashback
    offer_description = Column(Text)
    discount_code = Column(String(50), nullable=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    terms_conditions = Column(Text)
    minimum_purchase = Column(Float, nullable=True)
    offer_source = Column(String(100))
    exclusions = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    product = relationship("Product", back_populates="offers")


class ProductReview(Base):
    """Product Reviews & Ratings Model - Customer feedback and sentiment"""
    __tablename__ = "product_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    review_title = Column(String(500))
    review_text = Column(Text)
    rating = Column(Float)  # 1-5 stars or percentage
    reviewer_name = Column(String(200))
    review_date = Column(DateTime)
    verified_purchase = Column(Boolean, default=False)
    helpful_votes = Column(Integer, default=0)
    review_source = Column(String(100))  # Amazon, Google, Trustpilot, etc.
    pros = Column(JSON, nullable=True)  # List of pros
    cons = Column(JSON, nullable=True)  # List of cons
    sentiment_score = Column(Float, nullable=True)  # To be analyzed by LLM
    scraped_at = Column(DateTime, default=func.now())

    # Relationships
    product = relationship("Product", back_populates="reviews")


class Competitor(Base):
    """Enhanced Competitor Analysis Model - Detailed competitor intelligence"""
    __tablename__ = "competitors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(200), unique=True)
    domain = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    products_services = Column(JSON)  # List of offerings
    market_share = Column(Float, nullable=True)
    social_media_presence = Column(JSON)  # {platform: followers, engagement}
    website_traffic = Column(JSON)  # Traffic estimates and metrics
    key_features = Column(JSON)  # USPs and differentiators
    strengths = Column(JSON, nullable=True)  # List of identified strengths
    weaknesses = Column(JSON, nullable=True)  # List of identified weaknesses
    target_audience = Column(Text)
    geographic_presence = Column(JSON)  # List of regions/countries
    recent_news = Column(JSON)  # Recent announcements and updates
    funding_info = Column(JSON, nullable=True)  # Financial information
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    product_relationships = relationship("CompetitorProducts", back_populates="competitor", cascade="all, delete-orphan")
    insights = relationship("Insights", back_populates="competitor", cascade="all, delete-orphan")


class MarketTrend(Base):
    """Market Trends Model - Track industry trends and patterns"""
    __tablename__ = "market_trends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String(200))
    search_volume = Column(Integer)
    trend_direction = Column(String(20))  # Rising, Falling, Stable
    related_keywords = Column(JSON)  # List of related terms
    geographic_trends = Column(JSON)  # Trends by region
    time_period = Column(String(50))
    industry_category = Column(String(100))
    seasonal_patterns = Column(JSON)  # Seasonal variations
    consumer_interest_score = Column(Float)
    driving_events = Column(JSON)  # News/events driving the trend
    captured_at = Column(DateTime, default=func.now())


class SERPData(Base):
    """Search Engine Results Model - SERP tracking and analysis"""
    __tablename__ = "serp_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("searches.id"))
    search_query = Column(String(500))
    search_engine = Column(String(50))  # Google, Bing, DuckDuckGo
    result_position = Column(Integer)
    result_title = Column(String(500))
    result_url = Column(String(1000))
    snippet = Column(Text)
    featured_snippet = Column(Text, nullable=True)
    related_questions = Column(JSON)  # People also ask
    related_searches = Column(JSON)  # Related search terms
    geographic_location = Column(String(100))
    scraped_at = Column(DateTime, default=func.now())

    # Relationships
    search = relationship("Search", back_populates="serp_data")


class SocialSentiment(Base):
    """Social Media Sentiment Model - Social media monitoring and analysis"""
    __tablename__ = "social_sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(50))  # Twitter, Reddit, Facebook, etc.
    post_content = Column(Text)
    author_account = Column(String(200))
    post_date = Column(DateTime)
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    sentiment = Column(String(20))  # Positive, Negative, Neutral
    sentiment_score = Column(Float)  # Numeric sentiment value
    products_mentioned = Column(JSON)  # List of mentioned products
    companies_mentioned = Column(JSON)  # List of mentioned companies
    hashtags = Column(JSON)  # List of hashtags
    geographic_location = Column(String(100), nullable=True)
    captured_at = Column(DateTime, default=func.now())


class NewsContent(Base):
    """News & Content Model - News articles and content analysis"""
    __tablename__ = "news_content"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_title = Column(String(500))
    article_url = Column(String(1000))
    publication_source = Column(String(200))
    author = Column(String(200), nullable=True)
    publication_date = Column(DateTime)
    article_summary = Column(Text)
    full_content = Column(Text, nullable=True)
    category_topic = Column(String(100))
    companies_mentioned = Column(JSON)  # List of companies
    products_mentioned = Column(JSON)  # List of products
    sentiment = Column(String(20))  # Positive, Negative, Neutral
    keywords_tags = Column(JSON)  # List of keywords/tags
    scraped_at = Column(DateTime, default=func.now())


class FeatureComparison(Base):
    """Feature Comparison Model - Product feature analysis and comparison"""
    __tablename__ = "feature_comparisons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    feature_name = Column(String(200))
    feature_description = Column(Text)
    feature_availability = Column(String(20))  # Yes/No/Partial
    feature_specifications = Column(JSON)  # Detailed specs
    competitor_feature_mapping = Column(JSON)  # How competitors implement this
    feature_importance_score = Column(Float)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    product = relationship("Product", back_populates="features")


# =========================
# INFRASTRUCTURE MODELS
# =========================

class Cache(Base):
    """SQLite-based cache with TTL"""
    __tablename__ = "cache"

    key = Column(String(255), primary_key=True)
    value = Column(Text)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())


class TaskQueue(Base):
    """Background task queue"""
    __tablename__ = "task_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(100))  # scrape, analyze, export
    payload = Column(JSON)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    priority = Column(Integer, default=5)  # 1=highest, 10=lowest
    scheduled_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)


class ScrapingJob(Base):
    """Track individual scraping jobs"""
    __tablename__ = "scraping_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("searches.id"))
    scraper_name = Column(String(50))
    status = Column(String(50), default="pending")
    results_count = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_log = Column(Text, nullable=True)


class RateLimit(Base):
    """Track rate limits for scrapers"""
    __tablename__ = "rate_limits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scraper_name = Column(String(50))
    last_request_at = Column(DateTime)
    request_count = Column(Integer, default=0)
    reset_at = Column(DateTime)


# =========================
# ADDITIONAL TRACKING MODELS
# =========================

class PriceHistory(Base):
    """Historical price tracking for trend analysis"""
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    price = Column(Float)
    currency = Column(String(10))
    seller_name = Column(String(200))
    recorded_at = Column(DateTime, default=func.now())


class CompetitorTracking(Base):
    """Track competitor changes over time"""
    __tablename__ = "competitor_tracking"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competitor_id = Column(Integer, ForeignKey("competitors.id"))
    change_type = Column(String(100))  # price_change, new_product, feature_update
    change_description = Column(Text)
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    detected_at = Column(DateTime, default=func.now())


# =========================
# JUNCTION TABLE & INSIGHTS MODELS
# =========================

class CompetitorProducts(Base):
    """Junction table for many-to-many relationship between competitors and products"""
    __tablename__ = "competitor_products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    competitor_id = Column(Integer, ForeignKey("competitors.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    relationship_type = Column(String(50))  # 'direct_competitor', 'alternative', 'substitute'
    created_at = Column(DateTime, default=func.now())

    # Unique constraint to prevent duplicates
    __table_args__ = (
        UniqueConstraint('competitor_id', 'product_id', name='unique_competitor_product'),
    )

    # Relationships
    competitor = relationship("Competitor", back_populates="product_relationships")
    product = relationship("Product", back_populates="competitor_relationships")


class Insights(Base):
    """LLM-generated actionable insights"""
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    insight_type = Column(String(100), nullable=False)  # 'price_trend', 'sentiment_summary', 'competitive_gap'
    title = Column(String(500))
    description = Column(Text)
    insight_data = Column(JSON)  # Structured data supporting the insight
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    competitor_id = Column(Integer, ForeignKey("competitors.id"), nullable=True)
    confidence_score = Column(Float)  # 0.00 to 1.00
    severity = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    action_items = Column(JSON)  # Recommended actions
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    product = relationship("Product", back_populates="insights")
    competitor = relationship("Competitor", back_populates="insights")