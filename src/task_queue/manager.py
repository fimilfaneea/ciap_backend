"""
Task Queue Manager for CIAP
SQLite-based asynchronous task queue with priority scheduling and retry logic
"""

import asyncio
import json
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import logging

from sqlalchemy import select, update, delete, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import db_manager, TaskQueue as TaskQueueModel
from ..cache import cache

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD = "dead"  # Failed after max retries


class TaskPriority(int, Enum):
    """Task priority levels (1=highest, 10=lowest)"""
    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 10


class TaskQueueManager:
    """SQLite-based task queue manager with worker pool"""

    def __init__(self):
        """Initialize task queue manager"""
        self.workers: List[asyncio.Task] = []
        self.worker_count = settings.TASK_QUEUE_MAX_WORKERS
        self.poll_interval = settings.TASK_QUEUE_POLL_INTERVAL
        self.max_retries = settings.TASK_MAX_RETRIES

        # Task handlers registry
        self.handlers: Dict[str, Callable] = {}

        # Statistics
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "avg_processing_time": 0.0
        }

        self.running = False
        self._shutdown_event = asyncio.Event()
        self.cleanup_task: Optional[asyncio.Task] = None

    def register_handler(self, task_type: str, handler: Callable):
        """
        Register task handler function

        Args:
            task_type: Type of task (e.g., "scrape", "analyze", "export")
            handler: Async function to handle task execution

        Example:
            queue.register_handler("scrape", scrape_handler)
        """
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = TaskPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Add task to queue

        Args:
            task_type: Type of task
            payload: Task data dictionary
            priority: Task priority (1-10, lower is higher priority)
            scheduled_at: When to execute (None = immediate)
            user_id: Optional user identifier

        Returns:
            Task ID

        Example:
            task_id = await queue.enqueue(
                "scrape",
                {"query": "AI news", "sources": ["google"]},
                priority=TaskPriority.HIGH
            )
        """
        try:
            async with db_manager.get_session() as session:
                task = TaskQueueModel(
                    task_type=task_type,
                    payload=payload,
                    priority=priority,
                    status=TaskStatus.PENDING,
                    scheduled_at=scheduled_at or datetime.utcnow(),
                    max_retries=self.max_retries
                )

                session.add(task)
                await session.commit()
                await session.refresh(task)

                logger.info(
                    f"Enqueued task {task.id}: type={task_type}, "
                    f"priority={priority}"
                )

                # Notify workers if immediate execution
                if not scheduled_at or scheduled_at <= datetime.utcnow():
                    self._notify_workers()

                return task.id

        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise

    async def enqueue_batch(
        self,
        tasks: List[Dict[str, Any]],
        priority: int = TaskPriority.NORMAL
    ) -> List[int]:
        """
        Enqueue multiple tasks efficiently

        Args:
            tasks: List of task definitions with 'type' and 'payload' keys
            priority: Default priority for all tasks

        Returns:
            List of task IDs

        Example:
            task_ids = await queue.enqueue_batch([
                {"type": "scrape", "payload": {"query": "AI"}},
                {"type": "scrape", "payload": {"query": "ML"}},
            ])
        """
        task_ids = []

        async with db_manager.get_session() as session:
            task_models = []
            for task_def in tasks:
                task = TaskQueueModel(
                    task_type=task_def["type"],
                    payload=task_def.get("payload", {}),
                    priority=task_def.get("priority", priority),
                    status=TaskStatus.PENDING,
                    scheduled_at=datetime.utcnow(),
                    max_retries=self.max_retries
                )
                session.add(task)
                task_models.append(task)

            await session.flush()  # Get IDs before commit
            task_ids = [t.id for t in task_models]
            await session.commit()

        logger.info(f"Enqueued {len(task_ids)} tasks in batch")
        self._notify_workers()

        return task_ids

    async def dequeue(self) -> Optional[TaskQueueModel]:
        """
        Get next task from queue (priority-based)

        Uses skip_locked=True for concurrent worker safety.

        Returns:
            Next task or None if queue is empty
        """
        async with db_manager.get_session() as session:
            # Find next pending task by priority
            result = await session.execute(
                select(TaskQueueModel)
                .where(
                    and_(
                        TaskQueueModel.status == TaskStatus.PENDING,
                        TaskQueueModel.scheduled_at <= datetime.utcnow()
                    )
                )
                .order_by(
                    TaskQueueModel.priority.asc(),
                    TaskQueueModel.scheduled_at.asc()
                )
                .limit(1)
                .with_for_update(skip_locked=True)  # Skip locked rows for concurrency
            )

            task = result.scalar_one_or_none()

            if task:
                # Mark as processing
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.utcnow()
                await session.commit()

                logger.debug(f"Dequeued task {task.id}: {task.task_type}")

            return task

    async def process_task(self, task: TaskQueueModel):
        """
        Process a single task by calling its handler

        Args:
            task: Task to process

        Raises:
            ValueError: If no handler registered for task type
        """
        try:
            # Get handler for task type
            handler = self.handlers.get(task.task_type)

            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")

            # Execute handler
            logger.info(f"Processing task {task.id}: {task.task_type}")
            result = await handler(task.payload)

            # Mark as completed
            await self.complete_task(task.id, result)

        except Exception as e:
            # Log full traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Task {task.id} failed: {error_msg}")

            # Mark as failed (will retry if applicable)
            await self.fail_task(task.id, str(e))

    async def complete_task(
        self,
        task_id: int,
        result: Optional[Any] = None
    ):
        """
        Mark task as completed successfully

        Args:
            task_id: Task ID
            result: Task execution result (will be cached)
        """
        async with db_manager.get_session() as session:
            task = await session.get(TaskQueueModel, task_id)

            if task:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()

                # Calculate processing time
                if task.started_at:
                    processing_time = (
                        task.completed_at - task.started_at
                    ).total_seconds()
                    self._update_stats("processing_time", processing_time)

                await session.commit()

                # Store result in cache
                if result is not None:
                    await cache.set(
                        f"task_result:{task_id}",
                        result,
                        ttl=86400  # Keep for 24 hours
                    )

                self.stats["tasks_processed"] += 1
                logger.info(f"Task {task_id} completed successfully")

    async def fail_task(
        self,
        task_id: int,
        error: str,
        retry: bool = True
    ):
        """
        Mark task as failed with retry logic

        Args:
            task_id: Task ID
            error: Error message
            retry: Whether to retry the task (default: True)
        """
        async with db_manager.get_session() as session:
            task = await session.get(TaskQueueModel, task_id)

            if task:
                task.error_message = error

                if retry and task.retry_count < task.max_retries:
                    # Schedule retry with exponential backoff
                    task.status = TaskStatus.PENDING
                    task.retry_count += 1
                    backoff_seconds = 2 ** task.retry_count  # 2, 4, 8, 16...
                    task.scheduled_at = datetime.utcnow() + timedelta(
                        seconds=backoff_seconds
                    )
                    self.stats["tasks_retried"] += 1

                    logger.warning(
                        f"Task {task_id} failed, retrying in {backoff_seconds}s "
                        f"({task.retry_count}/{task.max_retries})"
                    )
                else:
                    # Max retries exceeded or retry disabled
                    task.status = TaskStatus.DEAD if retry else TaskStatus.FAILED
                    task.completed_at = datetime.utcnow()
                    self.stats["tasks_failed"] += 1

                    logger.error(
                        f"Task {task_id} failed permanently: {error}"
                    )

                await session.commit()

    async def cancel_task(self, task_id: int) -> bool:
        """
        Cancel a pending task

        Args:
            task_id: Task ID

        Returns:
            True if task was cancelled, False if not found or already started
        """
        async with db_manager.get_session() as session:
            result = await session.execute(
                update(TaskQueueModel)
                .where(
                    and_(
                        TaskQueueModel.id == task_id,
                        TaskQueueModel.status == TaskStatus.PENDING
                    )
                )
                .values(
                    status=TaskStatus.CANCELLED,
                    completed_at=datetime.utcnow()
                )
            )
            await session.commit()

            cancelled = result.rowcount > 0
            if cancelled:
                logger.info(f"Task {task_id} cancelled")

            return cancelled

    async def get_task_status(self, task_id: int) -> Optional[Dict]:
        """
        Get task status and details

        Args:
            task_id: Task ID

        Returns:
            Task details dictionary or None if not found
        """
        async with db_manager.get_session() as session:
            task = await session.get(TaskQueueModel, task_id)

            if task:
                # Get result from cache if completed
                result = None
                if task.status == TaskStatus.COMPLETED:
                    result = await cache.get(f"task_result:{task_id}")

                return {
                    "id": task.id,
                    "type": task.task_type,
                    "status": task.status,
                    "priority": task.priority,
                    "created_at": task.scheduled_at.isoformat(),
                    "started_at": task.started_at.isoformat()
                    if task.started_at else None,
                    "completed_at": task.completed_at.isoformat()
                    if task.completed_at else None,
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries,
                    "error": task.error_message,
                    "result": result
                }

            return None

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive queue statistics

        Returns:
            Dictionary with queue statistics including counts by status and type
        """
        async with db_manager.get_session() as session:
            # Count tasks by status
            status_counts = {}
            for status in TaskStatus:
                result = await session.execute(
                    select(func.count())
                    .select_from(TaskQueueModel)
                    .where(TaskQueueModel.status == status.value)
                )
                status_counts[status.value] = result.scalar()

            # Count tasks by type
            type_counts_result = await session.execute(
                select(
                    TaskQueueModel.task_type,
                    func.count().label("count")
                )
                .group_by(TaskQueueModel.task_type)
            )
            type_counts = {
                row.task_type: row.count
                for row in type_counts_result
            }

            # Average wait time for pending tasks (in seconds)
            avg_wait_result = await session.execute(
                select(
                    func.avg(
                        func.julianday("now") -
                        func.julianday(TaskQueueModel.scheduled_at)
                    )
                )
                .where(TaskQueueModel.status == TaskStatus.PENDING)
            )
            avg_wait_days = avg_wait_result.scalar() or 0
            avg_wait_seconds = avg_wait_days * 86400  # Convert days to seconds

        return {
            "status_counts": status_counts,
            "type_counts": type_counts,
            "avg_wait_time_seconds": avg_wait_seconds,
            "worker_count": len(self.workers),
            "active_workers": sum(
                1 for w in self.workers if not w.done()
            ),
            **self.stats
        }

    async def worker(self, worker_id: int):
        """
        Background worker to process tasks

        Args:
            worker_id: Worker identifier for logging
        """
        logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get next task
                task = await self.dequeue()

                if task:
                    # Process task
                    await self.process_task(task)
                else:
                    # No tasks available, wait before polling again
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(self.poll_interval)

        logger.info(f"Worker {worker_id} stopped")

    async def start(self):
        """Start task queue workers"""
        if self.running:
            logger.warning("Task queue already running")
            return

        self.running = True
        self._shutdown_event.clear()

        # Start worker pool
        for i in range(self.worker_count):
            worker_task = asyncio.create_task(self.worker(i))
            self.workers.append(worker_task)

        logger.info(
            f"Started task queue with {self.worker_count} workers"
        )

    async def stop(self):
        """Stop task queue workers gracefully"""
        if not self.running:
            return

        logger.info("Stopping task queue...")
        self.running = False
        self._shutdown_event.set()

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        # Stop cleanup task if running
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Task queue stopped")

    async def clear_queue(self, status: Optional[TaskStatus] = None):
        """
        Clear tasks from queue

        Args:
            status: Clear only tasks with specific status (None = all tasks)

        Returns:
            Number of tasks cleared
        """
        async with db_manager.get_session() as session:
            query = delete(TaskQueueModel)

            if status:
                query = query.where(TaskQueueModel.status == status.value)

            result = await session.execute(query)
            await session.commit()

            deleted_count = result.rowcount
            logger.info(f"Cleared {deleted_count} tasks from queue")
            return deleted_count

    async def requeue_failed(self) -> int:
        """
        Requeue failed tasks for retry

        Returns:
            Number of requeued tasks
        """
        async with db_manager.get_session() as session:
            result = await session.execute(
                update(TaskQueueModel)
                .where(
                    and_(
                        TaskQueueModel.status.in_([
                            TaskStatus.FAILED,
                            TaskStatus.PROCESSING  # Stuck tasks
                        ]),
                        TaskQueueModel.retry_count < self.max_retries
                    )
                )
                .values(
                    status=TaskStatus.PENDING,
                    scheduled_at=datetime.utcnow(),
                    error_message=None
                )
            )
            await session.commit()

            requeued = result.rowcount
            if requeued > 0:
                logger.info(f"Requeued {requeued} failed tasks")
                self._notify_workers()

            return requeued

    async def cleanup_old_tasks(self, days: int = 7) -> int:
        """
        Clean up old completed tasks

        Args:
            days: Delete completed tasks older than this many days

        Returns:
            Number of tasks deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        async with db_manager.get_session() as session:
            result = await session.execute(
                delete(TaskQueueModel).where(
                    and_(
                        TaskQueueModel.status.in_([
                            TaskStatus.COMPLETED,
                            TaskStatus.CANCELLED,
                            TaskStatus.DEAD
                        ]),
                        TaskQueueModel.completed_at < cutoff
                    )
                )
            )
            await session.commit()

            deleted = result.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old tasks")

            return deleted

    def _notify_workers(self):
        """
        Notify workers that new tasks are available

        Note: Currently a no-op as workers poll continuously.
        Could be enhanced with asyncio.Event for immediate notification.
        """
        pass

    def _update_stats(self, metric: str, value: float):
        """
        Update running statistics

        Args:
            metric: Metric name
            value: New value to incorporate
        """
        if metric == "processing_time":
            # Calculate running average
            current_avg = self.stats["avg_processing_time"]
            count = self.stats["tasks_processed"]
            if count > 0:
                new_avg = (current_avg * count + value) / (count + 1)
                self.stats["avg_processing_time"] = new_avg
            else:
                self.stats["avg_processing_time"] = value


# Global task queue instance
task_queue = TaskQueueManager()
