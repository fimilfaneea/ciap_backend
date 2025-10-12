"""Database models and setup for CIAP using SQLAlchemy with SQLite"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from config import DATABASE_URL

# Create base class for models
Base = declarative_base()

# Database models
class Search(Base):
    """Represents a search query and its metadata"""
    __tablename__ = "searches"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String(500), nullable=False)
    search_type = Column(String(50), default="competitor")  # competitor, market, product
    status = Column(String(20), default="pending")  # pending, scraping, analyzing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    search_metadata = Column(JSON, nullable=True)  # Store additional search parameters


class SearchResult(Base):
    """Represents individual search results from scraping"""
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, index=True)
    search_id = Column(Integer, index=True)
    title = Column(String(500))
    url = Column(Text)
    snippet = Column(Text)
    source = Column(String(50))  # google, bing, etc.
    position = Column(Integer)  # Ranking position in search results
    scraped_at = Column(DateTime, default=datetime.utcnow)
    raw_content = Column(Text, nullable=True)  # Full scraped content if available
    result_metadata = Column(JSON, nullable=True)  # Additional scraped data


class Analysis(Base):
    """Represents LLM analysis results"""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    search_id = Column(Integer, index=True)
    analysis_type = Column(String(50))  # sentiment, competitor, trend, summary
    content = Column(Text)  # The actual analysis text
    insights = Column(JSON)  # Structured insights as JSON
    sentiment_score = Column(Float, nullable=True)  # -1 to 1 sentiment scale
    confidence_score = Column(Float, nullable=True)  # 0 to 1 confidence scale
    llm_provider = Column(String(20))  # openai, anthropic
    llm_model = Column(String(50))  # specific model used
    created_at = Column(DateTime, default=datetime.utcnow)
    analysis_metadata = Column(JSON, nullable=True)  # Additional analysis data


class CompetitorProfile(Base):
    """Stores identified competitor information"""
    __tablename__ = "competitor_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), unique=True)
    domain = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    strengths = Column(JSON, nullable=True)  # List of identified strengths
    weaknesses = Column(JSON, nullable=True)  # List of identified weaknesses
    products = Column(JSON, nullable=True)  # List of products/services
    last_updated = Column(DateTime, default=datetime.utcnow)
    profile_metadata = Column(JSON, nullable=True)


# Database initialization
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at {DATABASE_URL}")


def get_db() -> Session:
    """Get database session for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("Database setup complete!")