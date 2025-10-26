"""
Scheduler Service for CIAP
Handles job scheduling using APScheduler with SQLite backend
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ..config.settings import settings
from ..task_queue.manager import task_queue
from ..cache.manager import cache
from ..database.manager import db_manager
from ..database.operations import DatabaseOperations

logger = logging.getLogger(__name__)


class SchedulerService:
    """Job scheduling service using APScheduler"""

    def __init__(self):
        """Initialize scheduler with SQLite jobstore and handler registry"""

        # Create scheduler database path
        scheduler_db_path = Path(settings.DATA_DIR) / "scheduler.db"
        scheduler_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure job store (SQLite)
        jobstores = {
            "default": SQLAlchemyJobStore(
                url=f"sqlite:///{scheduler_db_path}"
            )
        }

        # Configure scheduler with job defaults
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            job_defaults={
                "coalesce": True,  # Coalesce missed jobs
                "max_instances": settings.SCHEDULER_MAX_INSTANCES,
                "misfire_grace_time": settings.SCHEDULER_MISFIRE_GRACE_TIME
            },
            timezone=settings.SCHEDULER_TIMEZONE
        )

        # Job handlers registry
        self.handlers: Dict[str, Callable] = {}

        logger.info(f"SchedulerService initialized. Database: {scheduler_db_path}")

    async def _handle_search_job(self, **kwargs):
        """
        Handle scheduled search job

        Args:
            **kwargs: Job parameters including query and sources
        """
        query = kwargs.get("query")
        sources = kwargs.get("sources", ["google", "bing"])

        logger.info(f"Executing scheduled search: {query}")

        try:
            # Enqueue search task
            await task_queue.enqueue(
                task_type="scrape",
                payload={
                    "query": query,
                    "sources": sources,
                    "scheduled": True
                }
            )
            logger.info(f"Successfully enqueued search task for: {query}")

        except Exception as e:
            logger.error(f"Failed to enqueue search task for '{query}': {e}")

    async def _handle_analysis_job(self, **kwargs):
        """
        Handle scheduled analysis job

        Args:
            **kwargs: Job parameters including search_id and types
        """
        search_id = kwargs.get("search_id")
        analysis_types = kwargs.get("types", ["sentiment", "trends"])

        logger.info(f"Executing scheduled analysis for search {search_id}")

        try:
            # Enqueue analysis tasks for each type
            for analysis_type in analysis_types:
                await task_queue.enqueue(
                    task_type="analyze",
                    payload={
                        "search_id": search_id,
                        "type": analysis_type
                    }
                )

            logger.info(f"Successfully enqueued {len(analysis_types)} analysis tasks for search {search_id}")

        except Exception as e:
            logger.error(f"Failed to enqueue analysis tasks for search {search_id}: {e}")

    async def _handle_export_job(self, **kwargs):
        """
        Handle scheduled export job

        Args:
            **kwargs: Job parameters including search_id and format
        """
        search_id = kwargs.get("search_id")
        format = kwargs.get("format", "csv")

        logger.info(f"Executing scheduled export for search {search_id} (format: {format})")

        try:
            # Enqueue export task
            await task_queue.enqueue(
                task_type="export",
                payload={
                    "search_id": search_id,
                    "format": format
                }
            )
            logger.info(f"Successfully enqueued export task for search {search_id}")

        except Exception as e:
            logger.error(f"Failed to enqueue export task for search {search_id}: {e}")

    async def _handle_cleanup_job(self, **kwargs):
        """
        Handle cleanup job - clean expired cache and old tasks

        Args:
            **kwargs: Job parameters (optional)
        """
        logger.info("Executing scheduled cleanup")

        try:
            # Clean expired cache entries
            deleted_cache = await cache.cleanup_expired()
            logger.info(f"Cleaned {deleted_cache} expired cache entries")

        except Exception as e:
            logger.error(f"Failed to clean cache: {e}")

        try:
            # Clean old completed tasks (>7 days)
            from sqlalchemy import delete, and_
            from ..database.models import TaskQueue

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

                deleted_tasks = result.rowcount
                logger.info(f"Cleaned {deleted_tasks} old completed tasks")

        except Exception as e:
            logger.error(f"Failed to clean old tasks: {e}")

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
            sources: Data sources (e.g., ["google", "bing"])
            cron_expression: Cron schedule (e.g., "0 9 * * *" for 9am daily)
            interval_hours: Interval in hours (alternative to cron)

        Returns:
            Job ID

        Raises:
            ValueError: If neither cron_expression nor interval_hours provided
        """
        if not cron_expression and not interval_hours:
            raise ValueError("Either cron_expression or interval_hours required")

        # Generate unique job ID
        job_id = f"search_{query.replace(' ', '_')}_{int(datetime.now().timestamp())}"

        # Create appropriate trigger
        if cron_expression:
            trigger = CronTrigger.from_crontab(cron_expression)
            trigger_desc = f"cron: {cron_expression}"
        else:
            trigger = IntervalTrigger(hours=interval_hours)
            trigger_desc = f"interval: {interval_hours}h"

        # Add job to scheduler
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

        logger.info(f"Scheduled recurring search '{query}' ({trigger_desc}): {job_id}")
        return job_id

    def schedule_cleanup(self) -> str:
        """
        Schedule daily cleanup job at 2 AM UTC

        Returns:
            Job ID
        """
        job_id = "cleanup_daily"

        job = self.scheduler.add_job(
            func=self._handle_cleanup_job,
            trigger=CronTrigger(hour=2, minute=0),  # 2 AM UTC daily
            id=job_id,
            name="Daily Cleanup",
            replace_existing=True
        )

        logger.info(f"Scheduled daily cleanup job at 2 AM UTC: {job_id}")
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
            search_id: Search ID to analyze
            analysis_types: Types of analysis to run (e.g., ["sentiment", "trends"])
            delay_minutes: Delay after search completion (default: 5)

        Returns:
            Job ID
        """
        # Generate unique job ID
        job_id = f"analysis_{search_id}_{int(datetime.now().timestamp())}"

        # Calculate run time
        run_time = datetime.now() + timedelta(minutes=delay_minutes)

        # Add job to scheduler with DateTrigger
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

        logger.info(f"Scheduled analysis for search {search_id} in {delay_minutes} minutes: {job_id}")
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
            search_id: Search ID to export
            format: Export format (csv, excel, json, etc.)
            cron_expression: Cron schedule (e.g., "0 8 * * 1" for Monday 8am)

        Returns:
            Job ID

        Raises:
            ValueError: If cron_expression is invalid
        """
        # Generate unique job ID
        job_id = f"export_{search_id}_{int(datetime.now().timestamp())}"

        try:
            # Create cron trigger (this will validate the expression)
            trigger = CronTrigger.from_crontab(cron_expression)
        except Exception as e:
            logger.error(f"Invalid cron expression '{cron_expression}': {e}")
            raise ValueError(f"Invalid cron expression: {e}")

        # Add job to scheduler
        job = self.scheduler.add_job(
            func=self._handle_export_job,
            trigger=trigger,
            kwargs={
                "search_id": search_id,
                "format": format
            },
            id=job_id,
            name=f"Export: Search {search_id} ({format})",
            replace_existing=True
        )

        logger.info(f"Scheduled recurring export for search {search_id} (format: {format}, cron: {cron_expression}): {job_id}")
        return job_id

    def register_handler(self, job_type: str, handler: Callable):
        """
        Register job handler

        Args:
            job_type: Type of job (e.g., "search", "analysis", "export")
            handler: Async function to handle the job
        """
        self.handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    def _register_default_handlers(self):
        """Register default job handlers"""
        self.register_handler("search", self._handle_search_job)
        self.register_handler("analysis", self._handle_analysis_job)
        self.register_handler("export", self._handle_export_job)
        self.register_handler("cleanup", self._handle_cleanup_job)
        logger.info("Registered 4 default job handlers")

    async def start(self):
        """Start scheduler and schedule default cleanup job"""
        if not self.scheduler.running:
            # Register default handlers
            self._register_default_handlers()

            # Start the scheduler
            self.scheduler.start()
            logger.info("Scheduler started")

            # Schedule default cleanup job
            self.schedule_cleanup()
            logger.info("Default cleanup job scheduled")

    async def stop(self):
        """Stop scheduler gracefully"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job details

        Args:
            job_id: Job identifier

        Returns:
            Job details dict or None if not found
        """
        job = self.scheduler.get_job(job_id)

        if job:
            return {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
                "pending": job.pending,
                "kwargs": job.kwargs
            }

        return None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs

        Returns:
            List of job details dicts
        """
        jobs = []

        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })

        return jobs

    def remove_job(self, job_id: str) -> bool:
        """
        Remove scheduled job

        Args:
            job_id: Job identifier

        Returns:
            True if removed, False if not found
        """
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        """
        Pause scheduled job

        Args:
            job_id: Job identifier

        Returns:
            True if paused, False if not found
        """
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception:
            return False

    def resume_job(self, job_id: str) -> bool:
        """
        Resume paused job

        Args:
            job_id: Job identifier

        Returns:
            True if resumed, False if not found
        """
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
        """
        Modify existing job

        Args:
            job_id: Job identifier
            trigger: New trigger (optional)
            **kwargs: New job kwargs (optional)

        Returns:
            True if modified, False if failed
        """
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
