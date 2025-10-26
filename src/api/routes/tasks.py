"""
Task Management Routes for CIAP API
Endpoints for task queue operations
"""

from fastapi import APIRouter, HTTPException, Query, Response
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

class TaskStatsResponse(BaseModel):
    """Response model for task queue statistics"""
    status_counts: Dict[str, int] = Field(
        ...,
        description="Count of tasks by status (pending, processing, completed, failed, etc.)"
    )
    type_counts: Dict[str, int] = Field(
        ...,
        description="Count of tasks by type (scrape, analyze, export, batch)"
    )
    avg_wait_time_seconds: float = Field(
        ...,
        description="Average wait time for pending tasks in seconds"
    )
    worker_count: int = Field(
        ...,
        description="Total number of worker threads"
    )
    active_workers: int = Field(
        ...,
        description="Number of currently active workers"
    )
    tasks_processed: int = Field(
        0,
        description="Total tasks processed since startup"
    )
    tasks_failed: int = Field(
        0,
        description="Total tasks failed since startup"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status_counts": {
                    "pending": 5,
                    "processing": 2,
                    "completed": 120,
                    "failed": 3,
                    "dead": 0,
                    "cancelled": 1
                },
                "type_counts": {
                    "scrape": 80,
                    "analyze": 40,
                    "export": 10,
                    "batch": 1
                },
                "avg_wait_time_seconds": 2.5,
                "worker_count": 3,
                "active_workers": 2,
                "tasks_processed": 123,
                "tasks_failed": 3
            }
        }


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


@router.get("/stats", response_model=TaskStatsResponse)
async def get_task_stats():
    """
    Get task queue statistics

    Returns comprehensive statistics about the task queue including:
    - Status counts (pending, processing, completed, failed, dead, cancelled)
    - Type counts (scrape, analyze, export, batch)
    - Average wait time for pending tasks
    - Worker information (total workers, active workers)
    - Performance metrics (tasks processed, tasks failed)

    This endpoint provides task-specific statistics. For system-wide statistics
    including database and cache metrics, use GET /api/v1/stats (deprecated,
    use this endpoint for task-specific data).

    Returns:
        TaskStatsResponse: Comprehensive task queue statistics

    Raises:
        HTTPException: If statistics cannot be retrieved
    """
    try:
        # Get stats from task queue manager
        stats = await task_queue.get_queue_stats()

        # Extract and structure the response
        return TaskStatsResponse(
            status_counts=stats.get("status_counts", {}),
            type_counts=stats.get("type_counts", {}),
            avg_wait_time_seconds=stats.get("avg_wait_time_seconds", 0.0),
            worker_count=stats.get("worker_count", 0),
            active_workers=stats.get("active_workers", 0),
            tasks_processed=stats.get("tasks_processed", 0),
            tasks_failed=stats.get("tasks_failed", 0)
        )

    except Exception as e:
        logger.error(f"Failed to get task queue statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve task statistics: {str(e)}"
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
