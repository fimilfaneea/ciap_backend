"""
Task Management Routes for CIAP API
Endpoints for task queue operations
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ...task_queue.manager import task_queue, TaskPriority, TaskStatus
from ..schemas.common import PaginatedResponse, to_paginated_response

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class TaskRequest(BaseModel):
    """Request model for creating a new task"""
    type: str = Field(
        ...,
        description="Task type (scrape, analyze, export, batch)",
        examples=["scrape"]
    )
    payload: Dict[str, Any] = Field(
        ...,
        description="Task payload data",
        examples=[{"query": "AI trends", "sources": ["google"]}]
    )
    priority: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Task priority (1=highest, 10=lowest)"
    )


class BatchTaskRequest(BaseModel):
    """Request model for batch task creation"""
    tasks: List[TaskRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of tasks to enqueue"
    )


class TaskResponse(BaseModel):
    """Response model for task operations"""
    task_id: int = Field(..., description="Unique task ID")
    type: str = Field(..., description="Task type")
    status: str = Field(..., description="Task status")
    priority: int = Field(..., description="Task priority")
    created_at: datetime = Field(..., description="Creation timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")


# TaskListResponse replaced by PaginatedResponse[TaskResponse]
# This ensures consistent pagination contract across all API endpoints


# ============================================================
# Task Management Endpoints
# ============================================================

@router.post("/", response_model=TaskResponse, status_code=201)
async def enqueue_task(request: TaskRequest):
    """
    Enqueue a new task

    Args:
        request: Task parameters

    Returns:
        Task details with ID

    Raises:
        HTTPException: If task creation fails
    """
    try:
        # Enqueue task
        task_id = await task_queue.enqueue(
            task_type=request.type,
            payload=request.payload,
            priority=request.priority
        )

        logger.info(f"Enqueued task {task_id} of type {request.type}")

        # Get task status
        task_status = await task_queue.get_task_status(task_id)

        if not task_status:
            raise HTTPException(
                status_code=500,
                detail="Task enqueued but status not available"
            )

        return TaskResponse(
            task_id=task_id,
            type=task_status["type"],
            status=task_status["status"],
            priority=task_status["priority"],
            created_at=task_status["created_at"],
            result=task_status.get("result"),
            error=task_status.get("error")
        )

    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue task: {str(e)}"
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int):
    """
    Get task status and result

    Args:
        task_id: Task ID

    Returns:
        Task details

    Raises:
        HTTPException: If task not found
    """
    try:
        task_status = await task_queue.get_task_status(task_id)

        if not task_status:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )

        return TaskResponse(
            task_id=task_id,
            type=task_status["type"],
            status=task_status["status"],
            priority=task_status["priority"],
            created_at=task_status["created_at"],
            result=task_status.get("result"),
            error=task_status.get("error")
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve task: {str(e)}"
        )


@router.get("/", response_model=PaginatedResponse[TaskResponse])
async def list_tasks(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(20, ge=1, le=100, description="Number of records per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    task_type: Optional[str] = Query(None, description="Filter by task type")
):
    """
    List tasks with pagination and filters

    Args:
        page: Page number (1-indexed)
        per_page: Number of results per page
        status: Optional status filter
        task_type: Optional task type filter

    Returns:
        Paginated list of tasks with standard pagination metadata
    """
    try:
        from ...database import db_manager, DatabaseOperations

        async with db_manager.get_session() as session:
            # Get paginated tasks using DatabaseOperations
            result = await DatabaseOperations.get_paginated_task_queue(
                session=session,
                page=page,
                per_page=per_page,
                status=status,
                task_type=task_type
            )

            # Convert to response format
            tasks = []
            for item in result.items:
                tasks.append(
                    TaskResponse(
                        task_id=item.id,
                        type=item.task_type,  # Model uses task_type field
                        status=item.status,
                        priority=item.priority,
                        created_at=item.scheduled_at,  # Model uses scheduled_at for created_at
                        result=None,  # TaskQueue model doesn't have result field
                        error=item.error_message
                    )
                )

            # Convert PaginatedResult to PaginatedResponse
            return PaginatedResponse(
                items=tasks,
                total=result.total,
                page=result.page,
                per_page=result.per_page,
                total_pages=result.total_pages,
                has_next=result.has_next,
                has_prev=result.has_prev
            )

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tasks: {str(e)}"
        )


@router.post("/batch", response_model=Dict[str, Any], status_code=201)
async def enqueue_batch_tasks(request: BatchTaskRequest):
    """
    Enqueue multiple tasks at once

    Args:
        request: List of tasks to enqueue

    Returns:
        List of task IDs and summary

    Raises:
        HTTPException: If batch creation fails
    """
    try:
        # Convert to task queue format
        tasks = [
            {
                "type": task.type,
                "payload": task.payload,
                "priority": task.priority or 5
            }
            for task in request.tasks
        ]

        # Enqueue batch
        task_ids = await task_queue.enqueue_batch(tasks)

        logger.info(f"Enqueued batch of {len(task_ids)} tasks")

        return {
            "task_ids": task_ids,
            "count": len(task_ids),
            "message": f"Successfully enqueued {len(task_ids)} tasks"
        }

    except Exception as e:
        logger.error(f"Failed to enqueue batch tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue batch: {str(e)}"
        )


@router.delete("/{task_id}", status_code=200)
async def cancel_task(task_id: int):
    """
    Cancel a pending task

    Args:
        task_id: Task ID to cancel

    Returns:
        Cancellation confirmation

    Raises:
        HTTPException: If task not found or cannot be cancelled
    """
    try:
        # Check if task exists
        task_status = await task_queue.get_task_status(task_id)

        if not task_status:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )

        # Check if task can be cancelled
        if task_status["status"] not in [TaskStatus.PENDING.value, TaskStatus.PROCESSING.value]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task with status: {task_status['status']}"
            )

        # Cancel task
        cancelled = await task_queue.cancel_task(task_id)

        if not cancelled:
            raise HTTPException(
                status_code=400,
                detail="Task could not be cancelled (may have already started)"
            )

        logger.info(f"Cancelled task {task_id}")

        return {
            "task_id": task_id,
            "message": "Task cancelled successfully",
            "previous_status": task_status["status"]
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )
