"""Main FastAPI application for CIAP"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
import uvicorn
from pathlib import Path

from config import API_HOST, API_PORT, API_RELOAD, STATIC_DIR
from database import init_database, get_db, Search, SearchResult, Analysis, CompetitorProfile
from api.search_service import SearchService


# Initialize FastAPI app
app = FastAPI(
    title="CIAP - Competitive Intelligence Automation Platform",
    description="Open-source competitive intelligence solution for SMEs",
    version="0.1.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()
    print("CIAP API Started Successfully!")


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    search_type: str = "competitor"
    max_results: int = 10


class SearchResponse(BaseModel):
    id: int
    query: str
    search_type: str
    status: str
    created_at: datetime
    message: str


class AnalysisResponse(BaseModel):
    id: int
    search_id: int
    analysis_type: str
    content: str
    insights: dict
    sentiment_score: Optional[float]
    confidence_score: Optional[float]


class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None


# Root endpoint - serve the HTML interface
@app.get("/")
async def root():
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("""
        <html>
            <head>
                <title>CIAP - Competitive Intelligence Platform</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    h1 { color: #333; }
                    .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .link { display: inline-block; margin: 10px 0; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
                    .link:hover { background: #0056b3; }
                    .status { color: green; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎯 CIAP - Competitive Intelligence Platform</h1>
                    <p class="status">✅ System Online</p>
                    <p>Open-source competitive intelligence automation for SMEs</p>
                    <a href="/docs" class="link">📚 API Documentation</a>
                    <a href="/redoc" class="link">📖 Alternative Docs</a>
                    <a href="/health" class="link">🏥 Health Check</a>
                </div>
            </body>
        </html>
        """)


# Health check endpoint
@app.get("/health", response_model=StatusResponse)
async def health_check(db: Session = Depends(get_db)):
    try:
        # Test database connection
        search_count = db.query(Search).count()
        return StatusResponse(
            status="healthy",
            message="CIAP is running",
            data={
                "database": "connected",
                "total_searches": search_count
            }
        )
    except Exception as e:
        return StatusResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}"
        )


# Create new search
@app.post("/api/search", response_model=SearchResponse)
async def create_search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Initiate a new competitive intelligence search"""

    # Create search record
    search = Search(
        query=request.query,
        search_type=request.search_type,
        status="pending",
        metadata={"max_results": request.max_results}
    )
    db.add(search)
    db.commit()
    db.refresh(search)

    # Add background task for scraping and analysis
    background_tasks.add_task(
        process_search_task,
        search.id,
        db
    )

    return SearchResponse(
        id=search.id,
        query=search.query,
        search_type=search.search_type,
        status=search.status,
        created_at=search.created_at,
        message="Search initiated successfully. Check status for updates."
    )


# Get search status
@app.get("/api/search/{search_id}/status")
async def get_search_status(search_id: int, db: Session = Depends(get_db)):
    """Get the status of a search"""
    search = db.query(Search).filter(Search.id == search_id).first()
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")

    result_count = db.query(SearchResult).filter(SearchResult.search_id == search_id).count()
    analysis_count = db.query(Analysis).filter(Analysis.search_id == search_id).count()

    return {
        "id": search.id,
        "query": search.query,
        "status": search.status,
        "created_at": search.created_at,
        "completed_at": search.completed_at,
        "results_found": result_count,
        "analyses_completed": analysis_count
    }


# Get search results
@app.get("/api/search/{search_id}/results")
async def get_search_results(search_id: int, db: Session = Depends(get_db)):
    """Get results for a specific search"""
    results = db.query(SearchResult).filter(SearchResult.search_id == search_id).all()
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this search")

    return [
        {
            "id": r.id,
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "source": r.source,
            "position": r.position,
            "scraped_at": r.scraped_at
        }
        for r in results
    ]


# Get analysis for a search
@app.get("/api/search/{search_id}/analysis")
async def get_search_analysis(search_id: int, db: Session = Depends(get_db)):
    """Get analysis results for a specific search"""
    analyses = db.query(Analysis).filter(Analysis.search_id == search_id).all()
    if not analyses:
        raise HTTPException(status_code=404, detail="No analysis found for this search")

    return [
        AnalysisResponse(
            id=a.id,
            search_id=a.search_id,
            analysis_type=a.analysis_type,
            content=a.content,
            insights=a.insights or {},
            sentiment_score=a.sentiment_score,
            confidence_score=a.confidence_score
        )
        for a in analyses
    ]


# Get all searches
@app.get("/api/searches")
async def list_searches(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """List all searches with pagination"""
    searches = db.query(Search).offset(skip).limit(limit).all()
    total = db.query(Search).count()

    return {
        "total": total,
        "searches": [
            {
                "id": s.id,
                "query": s.query,
                "search_type": s.search_type,
                "status": s.status,
                "created_at": s.created_at
            }
            for s in searches
        ]
    }


# Get competitor profiles
@app.get("/api/competitors")
async def list_competitors(db: Session = Depends(get_db)):
    """List all identified competitor profiles"""
    competitors = db.query(CompetitorProfile).all()

    return [
        {
            "id": c.id,
            "name": c.name,
            "domain": c.domain,
            "description": c.description,
            "strengths": c.strengths,
            "weaknesses": c.weaknesses,
            "products": c.products,
            "last_updated": c.last_updated
        }
        for c in competitors
    ]


# Background task function
def process_search_task(search_id: int, db: Session):
    """Background task to process search"""
    service = SearchService(db)
    result = service.perform_search(search_id)
    return result


# Mount static files (for future web UI)
# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )