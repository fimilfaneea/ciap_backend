# Module 10: Job Scheduling System

## Overview
**Purpose:** Schedule and manage recurring tasks using APScheduler with SQLite backend.

**Responsibilities:**
- Schedule recurring searches
- Schedule analysis tasks
- Schedule exports
- Cron-based scheduling
- Job persistence
- Job monitoring

**Development Time:** 2 days (Week 8, Day 29-31)

---

## Implementation Guide

### Scheduler Service (`src/services/scheduler.py`)

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging

from src.core.config import settings
from src.core.queue import task_queue

logger = logging.getLogger(__name__)


class SchedulerService:
    """Job scheduling service using APScheduler"""

    def __init__(self):
        # Configure job store (SQLite)
        jobstores = {
            "default": SQLAlchemyJobStore(
                url=f"sqlite:///{settings.DATA_DIR}/scheduler.db"
            )
        }

        # Configure scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            job_defaults={
                "coalesce": True,  # Coalesce missed jobs
                "max_instances": 3,  # Max instances per job
                "misfire_grace_time": 30  # Grace time for misfired jobs
            },
            timezone="UTC"
        )

        # Job handlers registry
        self.handlers: Dict[str, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default job handlers"""
        self.register_handler("search", self._handle_search_job)
        self.register_handler("analysis", self._handle_analysis_job)
        self.register_handler("export", self._handle_export_job)
        self.register_handler("cleanup", self._handle_cleanup_job)

    def register_handler(self, job_type: str, handler: Callable):
        """Register job handler"""
        self.handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    async def _handle_search_job(self, **kwargs):
        """Handle scheduled search job"""
        query = kwargs.get("query")
        sources = kwargs.get("sources", ["google", "bing"])

        logger.info(f"Executing scheduled search: {query}")

        # Enqueue search task
        await task_queue.enqueue(
            task_type="scrape",
            payload={
                "query": query,
                "sources": sources,
                "scheduled": True
            }
        )

    async def _handle_analysis_job(self, **kwargs):
        """Handle scheduled analysis job"""
        search_id = kwargs.get("search_id")
        analysis_types = kwargs.get("types", ["sentiment", "trends"])

        logger.info(f"Executing scheduled analysis for search {search_id}")

        for analysis_type in analysis_types:
            await task_queue.enqueue(
                task_type="analyze",
                payload={
                    "search_id": search_id,
                    "type": analysis_type
                }
            )

    async def _handle_export_job(self, **kwargs):
        """Handle scheduled export job"""
        search_id = kwargs.get("search_id")
        format = kwargs.get("format", "csv")

        logger.info(f"Executing scheduled export for search {search_id}")

        await task_queue.enqueue(
            task_type="export",
            payload={
                "search_id": search_id,
                "format": format
            }
        )

    async def _handle_cleanup_job(self, **kwargs):
        """Handle cleanup job"""
        from src.core.cache import cache
        from src.core.database import db_manager
        from src.core.models import TaskQueue

        logger.info("Executing scheduled cleanup")

        # Clean expired cache
        deleted_cache = await cache.cleanup_expired()
        logger.info(f"Cleaned {deleted_cache} expired cache entries")

        # Clean old completed tasks
        cutoff = datetime.utcnow() - timedelta(days=7)

        async with db_manager.get_session() as session:
            result = await session.execute(
                delete(TaskQueue).where(
                    and_(
                        TaskQueue.status == "completed",
                        TaskQueue.completed_at < cutoff
                    )
                )
            )
            await session.commit()

            logger.info(f"Cleaned {result.rowcount} old tasks")

    def schedule_recurring_search(
        self,
        query: str,
        sources: List[str],
        cron_expression: str = None,
        interval_hours: int = None
    ) -> str:
        """
        Schedule recurring search

        Args:
            query: Search query
            sources: Data sources
            cron_expression: Cron schedule (e.g., "0 9 * * *" for 9am daily)
            interval_hours: Interval in hours (alternative to cron)

        Returns:
            Job ID
        """
        job_id = f"search_{query.replace(' ', '_')}_{datetime.now().timestamp()}"

        if cron_expression:
            trigger = CronTrigger.from_crontab(cron_expression)
        elif interval_hours:
            trigger = IntervalTrigger(hours=interval_hours)
        else:
            raise ValueError("Either cron_expression or interval_hours required")

        job = self.scheduler.add_job(
            func=self._handle_search_job,
            trigger=trigger,
            kwargs={
                "query": query,
                "sources": sources
            },
            id=job_id,
            name=f"Search: {query}",
            replace_existing=True
        )

        logger.info(f"Scheduled recurring search: {job_id}")
        return job_id

    def schedule_analysis(
        self,
        search_id: int,
        analysis_types: List[str],
        delay_minutes: int = 5
    ) -> str:
        """
        Schedule analysis after search completion

        Args:
            search_id: Search ID
            analysis_types: Types of analysis to run
            delay_minutes: Delay after search completion

        Returns:
            Job ID
        """
        job_id = f"analysis_{search_id}_{datetime.now().timestamp()}"

        run_time = datetime.now() + timedelta(minutes=delay_minutes)

        job = self.scheduler.add_job(
            func=self._handle_analysis_job,
            trigger=DateTrigger(run_date=run_time),
            kwargs={
                "search_id": search_id,
                "types": analysis_types
            },
            id=job_id,
            name=f"Analysis: Search {search_id}"
        )

        logger.info(f"Scheduled analysis: {job_id}")
        return job_id

    def schedule_export(
        self,
        search_id: int,
        format: str,
        cron_expression: str
    ) -> str:
        """
        Schedule recurring export

        Args:
            search_id: Search ID
            format: Export format
            cron_expression: Cron schedule

        Returns:
            Job ID
        """
        job_id = f"export_{search_id}_{datetime.now().timestamp()}"

        trigger = CronTrigger.from_crontab(cron_expression)

        job = self.scheduler.add_job(
            func=self._handle_export_job,
            trigger=trigger,
            kwargs={
                "search_id": search_id,
                "format": format
            },
            id=job_id,
            name=f"Export: Search {search_id}"
        )

        logger.info(f"Scheduled export: {job_id}")
        return job_id

    def schedule_cleanup(self):
        """Schedule daily cleanup job"""
        job = self.scheduler.add_job(
            func=self._handle_cleanup_job,
            trigger=CronTrigger(hour=2, minute=0),  # 2 AM daily
            id="cleanup_daily",
            name="Daily Cleanup",
            replace_existing=True
        )

        logger.info("Scheduled daily cleanup job")
        return job.id

    async def start(self):
        """Start scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")

            # Schedule default jobs
            self.schedule_cleanup()

    async def stop(self):
        """Stop scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details"""
        job = self.scheduler.get_job(job_id)

        if job:
            return {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat()
                if job.next_run_time else None,
                "trigger": str(job.trigger),
                "pending": job.pending,
                "kwargs": job.kwargs
            }

        return None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs"""
        jobs = []

        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat()
                if job.next_run_time else None,
                "trigger": str(job.trigger)
            })

        return jobs

    def remove_job(self, job_id: str) -> bool:
        """Remove scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        """Pause scheduled job"""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception:
            return False

    def resume_job(self, job_id: str) -> bool:
        """Resume paused job"""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception:
            return False

    def modify_job(
        self,
        job_id: str,
        trigger: Optional[Any] = None,
        **kwargs
    ) -> bool:
        """Modify existing job"""
        try:
            if trigger:
                self.scheduler.reschedule_job(job_id, trigger=trigger)

            if kwargs:
                job = self.scheduler.get_job(job_id)
                if job:
                    self.scheduler.modify_job(job_id, kwargs=kwargs)

            logger.info(f"Modified job: {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to modify job {job_id}: {e}")
            return False


# Global scheduler instance
scheduler = SchedulerService()
```

### Scheduler API Routes (`src/api/routes/scheduler.py`)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class ScheduleSearchRequest(BaseModel):
    query: str
    sources: List[str] = ["google", "bing"]
    cron_expression: Optional[str] = None
    interval_hours: Optional[int] = None


class ScheduleExportRequest(BaseModel):
    search_id: int
    format: str = "csv"
    cron_expression: str  # e.g., "0 8 * * 1" for Monday 8am


@router.post("/search")
async def schedule_search(request: ScheduleSearchRequest):
    """Schedule recurring search"""
    try:
        job_id = scheduler.schedule_recurring_search(
            query=request.query,
            sources=request.sources,
            cron_expression=request.cron_expression,
            interval_hours=request.interval_hours
        )

        return {
            "job_id": job_id,
            "message": "Search scheduled successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/export")
async def schedule_export(request: ScheduleExportRequest):
    """Schedule recurring export"""
    try:
        job_id = scheduler.schedule_export(
            search_id=request.search_id,
            format=request.format,
            cron_expression=request.cron_expression
        )

        return {
            "job_id": job_id,
            "message": "Export scheduled successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs")
async def list_jobs():
    """List all scheduled jobs"""
    return {
        "jobs": scheduler.list_jobs(),
        "total": len(scheduler.list_jobs())
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details"""
    job = scheduler.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.delete("/jobs/{job_id}")
async def remove_job(job_id: str):
    """Remove scheduled job"""
    if scheduler.remove_job(job_id):
        return {"message": "Job removed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause scheduled job"""
    if scheduler.pause_job(job_id):
        return {"message": "Job paused successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume paused job"""
    if scheduler.resume_job(job_id):
        return {"message": "Job resumed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")
```

---

## Testing

```python
@pytest.mark.asyncio
async def test_schedule_search():
    scheduler = SchedulerService()
    await scheduler.start()

    job_id = scheduler.schedule_recurring_search(
        query="test query",
        sources=["google"],
        interval_hours=1
    )

    assert job_id is not None

    # Check job exists
    job = scheduler.get_job(job_id)
    assert job is not None
    assert job["name"] == "Search: test query"

    # Remove job
    assert scheduler.remove_job(job_id)

    await scheduler.stop()


@pytest.mark.asyncio
async def test_cron_scheduling():
    scheduler = SchedulerService()

    job_id = scheduler.schedule_recurring_search(
        query="daily search",
        sources=["google"],
        cron_expression="0 9 * * *"  # 9 AM daily
    )

    job = scheduler.get_job(job_id)
    assert "cron" in str(job["trigger"]).lower()
```

---

## Module Checklist

- [ ] APScheduler configured
- [ ] SQLite job store setup
- [ ] Search scheduling working
- [ ] Analysis scheduling functional
- [ ] Export scheduling implemented
- [ ] Cleanup jobs scheduled
- [ ] Job management (pause/resume/remove)
- [ ] API endpoints complete
- [ ] Cron expression support
- [ ] Unit tests passing

---

## Summary

All 10 modules are now documented with:
- Clear interfaces and dependencies
- Complete implementation code
- Testing guidelines
- Integration points
- Common issues and solutions

This modular architecture allows you to:
1. Develop each module independently
2. Test in isolation
3. Swap implementations if needed
4. Track progress module by module
5. Debug issues easily

Start with Modules 1-4 (foundation), then build up through data collection (5-6), analysis (7), and finally API/Export/Scheduling (8-10).