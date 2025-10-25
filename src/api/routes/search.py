"""
Search Routes for CIAP API
CRUD operations for search functionality
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging

from ...database import get_db, DatabaseOperations, Search
from ...scrapers.manager import scraper_manager
from ...task_queue.manager import task_queue, TaskPriority

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class SearchRequest(BaseModel):
    """Request model for creating a new search"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query",
        examples=["artificial intelligence trends"]
    )
    sources: List[str] = Field(
        default=["google", "bing"],
        description="Search sources to use",
        examples=[["google", "bing"]]
    )
    max_results: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum results per source"
    )
    analyze: bool = Field(
        default=True,
        description="Run analysis after scraping"
    )


class SearchResponse(BaseModel):
    """Response model for search creation"""
    search_id: int = Field(..., description="Unique search ID")
    query: str = Field(..., description="Search query")
    status: str = Field(..., description="Search status")
    created_at: datetime = Field(..., description="Creation timestamp")
    task_id: Optional[int] = Field(None, description="Scraping task ID")


class SearchDetailResponse(BaseModel):
    """Response model for search details"""
    id: int
    query: str
    status: str
    sources: List[str]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_count: int


class SearchResultItem(BaseModel):
    """Individual search result item"""
    title: str
    snippet: str
    url: str
    source: str
    position: int


class SearchListResponse(BaseModel):
    """Response model for search list"""
    searches: List[SearchDetailResponse]
    total: int
    page: int
    per_page: int


# ============================================================
# Search Endpoints
# ============================================================

@router.post("/", response_model=SearchResponse, status_code=201)
async def create_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new search and schedule scraping task

    Args:
        request: Search parameters
        db: Database session

    Returns:
        Search details with task ID

    Raises:
        HTTPException: If search creation fails
    """
    try:
        # Create search record in database
        search = await DatabaseOperations.create_search(
            session=db,
            query=request.query,
            sources=request.sources
        )

        # Commit to get search ID
        await db.commit()
        await db.refresh(search)

        logger.info(f"Created search {search.id} for query: {request.query}")

        # Schedule scraping task
        task_id = None
        try:
            task_payload = {
                "query": request.query,
                "sources": request.sources,
                "search_id": search.id,
                "max_results_per_source": request.max_results
            }

            task_id = await task_queue.enqueue(
                task_type="scrape",
                payload=task_payload,
                priority=TaskPriority.HIGH
            )

            logger.info(f"Scheduled scraping task {task_id} for search {search.id}")

            # If analyze is enabled, schedule analysis task after scraping
            if request.analyze:
                # Analysis will be triggered by scraping completion
                # (This can be enhanced with task chaining in future)
                logger.info(f"Analysis will be scheduled after scraping completes for search {search.id}")

        except Exception as e:
            logger.error(f"Failed to schedule scraping task: {e}")
            # Update search status to reflect error
            await DatabaseOperations.update_search_status(
                session=db,
                search_id=search.id,
                status="failed",
                error=f"Failed to schedule task: {str(e)}"
            )
            await db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Search created but failed to schedule scraping: {str(e)}"
            )

        return SearchResponse(
            search_id=search.id,
            query=search.query,
            status=search.status,
            created_at=search.created_at,
            task_id=task_id
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Search creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create search: {str(e)}"
        )


@router.get("/{search_id}", response_model=dict)
async def get_search(
    search_id: int,
    include_results: bool = Query(True, description="Include search results in response"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get search details and optionally results

    Args:
        search_id: Search ID
        include_results: Whether to include search results
        db: Database session

    Returns:
        Search details with results

    Raises:
        HTTPException: If search not found
    """
    try:
        # Get search
        search = await DatabaseOperations.get_search(db, search_id)

        if not search:
            raise HTTPException(
                status_code=404,
                detail=f"Search {search_id} not found"
            )

        # Build response
        response = {
            "search": {
                "id": search.id,
                "query": search.query,
                "status": search.status,
                "sources": search.sources,
                "created_at": search.created_at,
                "completed_at": search.completed_at,
                "error_message": search.error_message
            }
        }

        # Get results if requested
        if include_results:
            results = await DatabaseOperations.get_search_results(db, search_id)

            response["results"] = [
                {
                    "title": r.title,
                    "snippet": r.snippet,
                    "url": r.url,
                    "source": r.source,
                    "position": r.position
                }
                for r in results
            ]
            response["result_count"] = len(results)
        else:
            # Just get count
            result = await db.execute(
                select(func.count())
                .select_from(Search)
                .where(Search.id == search_id)
            )
            response["result_count"] = result.scalar() or 0

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to get search {search_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve search: {str(e)}"
        )


@router.get("/", response_model=SearchListResponse)
async def list_searches(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db)
):
    """
    List all searches with pagination

    Args:
        skip: Offset for pagination
        limit: Number of results per page
        status: Optional status filter
        db: Database session

    Returns:
        Paginated list of searches
    """
    try:
        # Build query
        query = select(Search).order_by(Search.created_at.desc())

        # Apply status filter if provided
        if status:
            query = query.where(Search.status == status)

        # Get total count
        count_query = select(func.count()).select_from(Search)
        if status:
            count_query = count_query.where(Search.status == status)

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        searches = result.scalars().all()

        # Get result counts for each search
        search_details = []
        for search in searches:
            # Get result count
            count_result = await db.execute(
                select(func.count())
                .select_from(Search)
                .where(Search.id == search.id)
            )
            result_count = count_result.scalar() or 0

            search_details.append(
                SearchDetailResponse(
                    id=search.id,
                    query=search.query,
                    status=search.status,
                    sources=search.sources,
                    created_at=search.created_at,
                    completed_at=search.completed_at,
                    error_message=search.error_message,
                    result_count=result_count
                )
            )

        # Calculate page number
        page = (skip // limit) + 1 if limit > 0 else 1

        return SearchListResponse(
            searches=search_details,
            total=total,
            page=page,
            per_page=limit
        )

    except Exception as e:
        logger.error(f"Failed to list searches: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list searches: {str(e)}"
        )


@router.delete("/{search_id}", status_code=204)
async def delete_search(
    search_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a search and its results

    Args:
        search_id: Search ID to delete
        db: Database session

    Raises:
        HTTPException: If search not found
    """
    try:
        # Check if search exists
        search = await DatabaseOperations.get_search(db, search_id)

        if not search:
            raise HTTPException(
                status_code=404,
                detail=f"Search {search_id} not found"
            )

        # Delete search (cascade will delete results)
        await db.delete(search)
        await db.commit()

        logger.info(f"Deleted search {search_id}")

        return None

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to delete search {search_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete search: {str(e)}"
        )
