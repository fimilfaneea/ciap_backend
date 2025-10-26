"""
Scheduler Routes for CIAP API
Job scheduling and management functionality (Module 10)
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ...services.scheduler import scheduler

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class ScheduleSearchRequest(BaseModel):
    """Request model for scheduling recurring searches"""
    query: str = Field(..., description="Search query to schedule")
    sources: List[str] = Field(
        default=["google", "bing"],
        description="Data sources for search"
    )
    cron_expression: Optional[str] = Field(
        default=None,
        description="Cron expression (e.g., '0 9 * * *' for 9am daily)"
    )
    interval_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=168,
        description="Interval in hours (alternative to cron)"
    )


class ScheduleAnalysisRequest(BaseModel):
    """Request model for scheduling analysis"""
    search_id: int = Field(..., description="Search ID to analyze")
    analysis_types: List[str] = Field(
        default=["sentiment", "trends"],
        description="Types of analysis to run"
    )
    delay_minutes: int = Field(
        default=5,
        ge=0,
        le=1440,
        description="Delay in minutes before running analysis"
    )


class ScheduleExportRequest(BaseModel):
    """Request model for scheduling recurring exports"""
    search_id: int = Field(..., description="Search ID to export")
    format: str = Field(
        default="csv",
        description="Export format",
        pattern="^(csv|excel|xlsx|json|powerbi|html)$"
    )
    cron_expression: str = Field(
        ...,
        description="Cron expression (e.g., '0 8 * * 1' for Monday 8am)"
    )


class JobResponse(BaseModel):
    """Response model for job details"""
    id: str = Field(..., description="Job ID")
    name: str = Field(..., description="Job name")
    next_run: Optional[str] = Field(None, description="Next run time (ISO format)")
    trigger: str = Field(..., description="Trigger description")


class JobListResponse(BaseModel):
    """Response model for job list"""
    jobs: List[JobResponse] = Field(..., description="List of scheduled jobs")
    total: int = Field(..., description="Total number of jobs")


class JobActionResponse(BaseModel):
    """Response model for job actions (pause/resume/remove)"""
    job_id: str = Field(..., description="Job ID")
    success: bool = Field(..., description="Whether action was successful")
    message: str = Field(..., description="Status message")


class ScheduleResponse(BaseModel):
    """Response model for scheduling operations"""
    job_id: str = Field(..., description="Job ID of scheduled job")
    message: str = Field(..., description="Status message")


# ============================================================
# Scheduler Endpoints
# ============================================================

@router.post("/search", response_model=ScheduleResponse, status_code=201)
async def schedule_recurring_search(request: ScheduleSearchRequest):
    """
    Schedule recurring search job

    Schedule a search to run on a recurring basis using either:
    - Cron expression (e.g., "0 9 * * *" for 9am daily)
    - Interval in hours (e.g., 24 for daily)

    Args:
        request: ScheduleSearchRequest with query, sources, and schedule

    Returns:
        Job ID and confirmation message

    Raises:
        HTTPException 400: If neither cron_expression nor interval_hours provided
        HTTPException 400: If cron expression is invalid
    """
    try:
        job_id = scheduler.schedule_recurring_search(
            query=request.query,
            sources=request.sources,
            cron_expression=request.cron_expression,
            interval_hours=request.interval_hours
        )

        return ScheduleResponse(
            job_id=job_id,
            message=f"Search '{request.query}' scheduled successfully"
        )

    except ValueError as e:
        logger.error(f"Invalid schedule parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to schedule search: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule search: {e}")


@router.post("/analysis", response_model=ScheduleResponse, status_code=201)
async def schedule_analysis(request: ScheduleAnalysisRequest):
    """
    Schedule delayed analysis job

    Schedule analysis to run after a specified delay (useful for running
    analysis after a search completes).

    Args:
        request: ScheduleAnalysisRequest with search_id, types, and delay

    Returns:
        Job ID and confirmation message

    Raises:
        HTTPException 500: If scheduling fails
    """
    try:
        job_id = scheduler.schedule_analysis(
            search_id=request.search_id,
            analysis_types=request.analysis_types,
            delay_minutes=request.delay_minutes
        )

        return ScheduleResponse(
            job_id=job_id,
            message=f"Analysis for search {request.search_id} scheduled in {request.delay_minutes} minutes"
        )

    except Exception as e:
        logger.error(f"Failed to schedule analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule analysis: {e}")


@router.post("/export", response_model=ScheduleResponse, status_code=201)
async def schedule_recurring_export(request: ScheduleExportRequest):
    """
    Schedule recurring export job

    Schedule an export to run on a recurring basis using a cron expression.

    Common cron patterns:
    - "0 8 * * *" - Daily at 8am
    - "0 8 * * 1" - Monday at 8am
    - "0 0 1 * *" - First day of month at midnight

    Args:
        request: ScheduleExportRequest with search_id, format, and cron

    Returns:
        Job ID and confirmation message

    Raises:
        HTTPException 400: If cron expression is invalid
        HTTPException 500: If scheduling fails
    """
    try:
        job_id = scheduler.schedule_export(
            search_id=request.search_id,
            format=request.format,
            cron_expression=request.cron_expression
        )

        return ScheduleResponse(
            job_id=job_id,
            message=f"Export for search {request.search_id} scheduled (format: {request.format})"
        )

    except ValueError as e:
        logger.error(f"Invalid cron expression: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to schedule export: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule export: {e}")


@router.get("/jobs", response_model=JobListResponse, status_code=200)
async def list_scheduled_jobs():
    """
    List all scheduled jobs

    Get a list of all currently scheduled jobs with their details.

    Returns:
        List of jobs with total count

    Raises:
        HTTPException 500: If listing fails
    """
    try:
        jobs = scheduler.list_jobs()

        job_responses = [
            JobResponse(
                id=job["id"],
                name=job["name"],
                next_run=job["next_run"],
                trigger=job["trigger"]
            )
            for job in jobs
        ]

        return JobListResponse(
            jobs=job_responses,
            total=len(job_responses)
        )

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}")


@router.get("/jobs/{job_id}", response_model=JobResponse, status_code=200)
async def get_job_details(job_id: str):
    """
    Get job details

    Retrieve detailed information about a specific scheduled job.

    Args:
        job_id: Job identifier

    Returns:
        Job details

    Raises:
        HTTPException 404: If job not found
        HTTPException 500: If retrieval fails
    """
    try:
        job = scheduler.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JobResponse(
            id=job["id"],
            name=job["name"],
            next_run=job["next_run"],
            trigger=job["trigger"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {e}")


@router.delete("/jobs/{job_id}", response_model=JobActionResponse, status_code=200)
async def remove_scheduled_job(job_id: str):
    """
    Remove scheduled job

    Delete a scheduled job by ID. The job will no longer run.

    Args:
        job_id: Job identifier

    Returns:
        Success status and message

    Raises:
        HTTPException 404: If job not found
        HTTPException 500: If removal fails
    """
    try:
        success = scheduler.remove_job(job_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JobActionResponse(
            job_id=job_id,
            success=True,
            message="Job removed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove job: {e}")


@router.post("/jobs/{job_id}/pause", response_model=JobActionResponse, status_code=200)
async def pause_scheduled_job(job_id: str):
    """
    Pause scheduled job

    Pause a scheduled job. It will not run until resumed.

    Args:
        job_id: Job identifier

    Returns:
        Success status and message

    Raises:
        HTTPException 404: If job not found
        HTTPException 500: If pause fails
    """
    try:
        success = scheduler.pause_job(job_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JobActionResponse(
            job_id=job_id,
            success=True,
            message="Job paused successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause job: {e}")


@router.post("/jobs/{job_id}/resume", response_model=JobActionResponse, status_code=200)
async def resume_scheduled_job(job_id: str):
    """
    Resume paused job

    Resume a previously paused job. It will start running according to its schedule.

    Args:
        job_id: Job identifier

    Returns:
        Success status and message

    Raises:
        HTTPException 404: If job not found
        HTTPException 500: If resume fails
    """
    try:
        success = scheduler.resume_job(job_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JobActionResponse(
            job_id=job_id,
            success=True,
            message="Job resumed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume job: {e}")
