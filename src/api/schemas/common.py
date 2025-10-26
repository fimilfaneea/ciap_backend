"""
Common API Schemas for CIAP
Shared Pydantic models used across multiple endpoints
"""

from typing import Generic, TypeVar, List
from pydantic import BaseModel, Field

# Type variable for generic pagination
T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response model

    Matches frontend PaginatedResponse interface and database PaginatedResult dataclass.
    All list/search endpoints should use this model for consistency.

    Example:
        @router.get("/items", response_model=PaginatedResponse[ItemResponse])
        async def list_items():
            # Database returns PaginatedResult
            result = await DatabaseOperations.get_paginated_items(session)
            # Convert to API response
            return to_paginated_response(result, ItemResponse)
    """
    items: List[T] = Field(
        default_factory=list,
        description="List of items in current page"
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total number of items across all pages"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Current page number (1-indexed)"
    )
    per_page: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page"
    )
    total_pages: int = Field(
        default=0,
        ge=0,
        description="Total number of pages"
    )
    has_next: bool = Field(
        default=False,
        description="Whether there is a next page"
    )
    has_prev: bool = Field(
        default=False,
        description="Whether there is a previous page"
    )

    class Config:
        """Pydantic configuration"""
        from_attributes = True


def to_paginated_response(db_result, item_model=None):
    """
    Convert database PaginatedResult to API PaginatedResponse

    Args:
        db_result: PaginatedResult from DatabaseOperations
        item_model: Optional Pydantic model to convert items

    Returns:
        PaginatedResponse with converted items

    Example:
        result = await DatabaseOperations.get_paginated_searches(session)
        return to_paginated_response(result, SearchDetailResponse)
    """
    # Convert items if model provided
    items = db_result.items
    if item_model and items:
        # Check if items need conversion
        if not isinstance(items[0], item_model):
            items = [item_model.model_validate(item) for item in items]

    return PaginatedResponse(
        items=items,
        total=db_result.total,
        page=db_result.page,
        per_page=db_result.per_page,
        total_pages=db_result.total_pages,
        has_next=db_result.has_next,
        has_prev=db_result.has_prev
    )


def create_paginated_response(
    items: List[T],
    total: int,
    page: int,
    per_page: int
) -> PaginatedResponse[T]:
    """
    Create PaginatedResponse from raw data

    Utility function for endpoints that don't use DatabaseOperations pagination.
    Calculates total_pages, has_next, has_prev automatically.

    Args:
        items: List of items in current page
        total: Total count across all pages
        page: Current page number (1-indexed)
        per_page: Items per page

    Returns:
        PaginatedResponse with calculated metadata
    """
    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0

    # Calculate navigation flags
    has_next = page < total_pages
    has_prev = page > 1

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        has_next=has_next,
        has_prev=has_prev
    )
