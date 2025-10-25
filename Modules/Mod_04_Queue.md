# Module 4: Task Queue System

## Overview
**Purpose:** SQLite-based asynchronous task queue for background job processing without external dependencies like Celery or Redis.

**Responsibilities:**
- Task enqueue/dequeue operations
- Priority-based task execution
- Task status tracking
- Retry logic for failed tasks
- Background worker management
- Task result storage
- Dead letter queue for failed tasks

**Development Time:** 2 days (Week 1, Day 5-6)

---

## Interface Specification

### Input
```python
# Task definition
task_type: str  # "scrape", "analyze", "export"
payload: Dict  # Task-specific data
priority: int  # 1-10 (1=highest)
```

### Output
```python
# Task result
task_id: int
status: str  # "pending", "processing", "completed", "failed"
result: Any  # Task execution result
```

---

## Dependencies

### External
```txt
asyncio  # Built-in
aiosqlite==0.19.0
typing-extensions==4.8.0
```

### Internal
- Module 1: Database Infrastructure
- Module 2: Configuration
- Module 3: Cache System

---

## Implementation Guide

### Step 1: Task Queue Manager (`src/core/queue.py`)

```python
import asyncio
import json
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import logging

from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import db_manager
from src.core.models import TaskQueue as TaskQueueModel
from src.core.cache import cache

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
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 10


class TaskQueueManager:
    """SQLite-based task queue manager"""

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
            "avg_processing_time": 0
        }

        self.running = False
        self._shutdown_event = asyncio.Event()

    def register_handler(self, task_type: str, handler: Callable):
        """
        Register task handler

        Args:
            task_type: Type of task
            handler: Async function to handle task

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
            payload: Task data
            priority: Task priority (1-10)
            scheduled_at: When to execute (None = immediate)
            user_id: Optional user identifier

        Returns:
            Task ID
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
            tasks: List of task definitions
            priority: Default priority for all tasks

        Returns:
            List of task IDs
        """
        task_ids = []

        async with db_manager.get_session() as session:
            for task_def in tasks:
                task = TaskQueueModel(
                    task_type=task_def["type"],
                    payload=task_def.get("payload", {}),
                    priority=task_def.get("priority", priority),
                    status=TaskStatus.PENDING,
                    scheduled_at=datetime.utcnow()
                )
                session.add(task)

            await session.flush()  # Get IDs before commit
            task_ids = [t.id for t in session.new]
            await session.commit()

        logger.info(f"Enqueued {len(task_ids)} tasks")
        self._notify_workers()

        return task_ids

    async def dequeue(self) -> Optional[TaskQueueModel]:
        """
        Get next task from queue

        Returns:
            Next task or None if queue is empty
        """
        async with db_manager.get_session() as session:
            # Find next pending task
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
                .with_for_update(skip_locked=True)  # Skip locked rows
            )

            task = result.scalar_one_or_none()

            if task:
                # Mark as processing
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.utcnow()
                await session.commit()

                logger.debug(f"Dequeued task {task.id}: {task.task_type}")

            return task

    async def complete_task(
        self,
        task_id: int,
        result: Optional[Any] = None
    ):
        """
        Mark task as completed

        Args:
            task_id: Task ID
            result: Task execution result
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

                # Store result in cache
                if result is not None:
                    await cache.set(
                        f"task_result:{task_id}",
                        result,
                        ttl=86400  # Keep for 24 hours
                    )

                await session.commit()
                self.stats["tasks_processed"] += 1

                logger.info(f"Task {task_id} completed successfully")

    async def fail_task(
        self,
        task_id: int,
        error: str,
        retry: bool = True
    ):
        """
        Mark task as failed

        Args:
            task_id: Task ID
            error: Error message
            retry: Whether to retry the task
        """
        async with db_manager.get_session() as session:
            task = await session.get(TaskQueueModel, task_id)

            if task:
                task.error_message = error

                if retry and task.retry_count < task.max_retries:
                    # Schedule retry
                    task.status = TaskStatus.PENDING
                    task.retry_count += 1
                    task.scheduled_at = datetime.utcnow() + timedelta(
                        seconds=2 ** task.retry_count  # Exponential backoff
                    )
                    self.stats["tasks_retried"] += 1

                    logger.warning(
                        f"Task {task_id} failed, retrying "
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
            Success status
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
            Task details or None
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
                    "error": task.error_message,
                    "result": result
                }

            return None

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics

        Returns:
            Queue statistics
        """
        async with db_manager.get_session() as session:
            # Count tasks by status
            status_counts = {}
            for status in TaskStatus:
                result = await session.execute(
                    select(func.count())
                    .select_from(TaskQueueModel)
                    .where(TaskQueueModel.status == status)
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

            # Average wait time for pending tasks
            avg_wait_result = await session.execute(
                select(
                    func.avg(
                        func.julianday("now") -
                        func.julianday(TaskQueueModel.scheduled_at)
                    )
                )
                .where(TaskQueueModel.status == TaskStatus.PENDING)
            )
            avg_wait_time = avg_wait_result.scalar() or 0

        return {
            "status_counts": status_counts,
            "type_counts": type_counts,
            "avg_wait_time_seconds": avg_wait_time * 86400,  # Convert days to seconds
            "worker_count": len(self.workers),
            "active_workers": sum(
                1 for w in self.workers if not w.done()
            ),
            **self.stats
        }

    async def process_task(self, task: TaskQueueModel):
        """
        Process a single task

        Args:
            task: Task to process
        """
        try:
            # Get handler for task type
            handler = self.handlers.get(task.task_type)

            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")

            # Execute handler
            logger.info(f"Processing task {task.id}: {task.task_type}")
            result = await handler(task.payload)

            # Mark as completed
            await self.complete_task(task.id, result)

        except Exception as e:
            # Log full traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Task {task.id} failed: {error_msg}")

            # Mark as failed
            await self.fail_task(task.id, str(e))

    async def worker(self, worker_id: int):
        """
        Background worker to process tasks

        Args:
            worker_id: Worker identifier
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
                    # No tasks, wait
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

        # Start workers
        for i in range(self.worker_count):
            worker_task = asyncio.create_task(self.worker(i))
            self.workers.append(worker_task)

        logger.info(
            f"Started task queue with {self.worker_count} workers"
        )

    async def stop(self):
        """Stop task queue workers"""
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

        logger.info("Task queue stopped")

    async def clear_queue(self, status: Optional[TaskStatus] = None):
        """
        Clear tasks from queue

        Args:
            status: Clear only tasks with specific status
        """
        async with db_manager.get_session() as session:
            query = delete(TaskQueueModel)

            if status:
                query = query.where(TaskQueueModel.status == status)

            result = await session.execute(query)
            await session.commit()

            logger.info(f"Cleared {result.rowcount} tasks from queue")

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
                            TaskStatus.PROCESSING
                        ]),
                        TaskQueueModel.retry_count < self.max_retries
                    )
                )
                .values(
                    status=TaskStatus.PENDING,
                    scheduled_at=datetime.utcnow()
                )
            )
            await session.commit()

            requeued = result.rowcount
            if requeued > 0:
                logger.info(f"Requeued {requeued} failed tasks")
                self._notify_workers()

            return requeued

    def _notify_workers(self):
        """Notify workers that new tasks are available"""
        # In a real implementation, could use asyncio.Event
        pass

    def _update_stats(self, metric: str, value: float):
        """Update running statistics"""
        if metric == "processing_time":
            # Calculate running average
            current_avg = self.stats["avg_processing_time"]
            count = self.stats["tasks_processed"]
            new_avg = (current_avg * count + value) / (count + 1)
            self.stats["avg_processing_time"] = new_avg


# Global queue instance
task_queue = TaskQueueManager()
```

### Step 2: Task Handlers (`src/core/task_handlers.py`)

```python
import asyncio
from typing import Dict, Any, List
import logging

from src.core.queue import task_queue
from src.core.cache import cache

logger = logging.getLogger(__name__)


async def scrape_handler(payload: Dict[str, Any]) -> Dict:
    """
    Handle scraping tasks

    Args:
        payload: Task data with query and sources

    Returns:
        Scraping results
    """
    query = payload.get("query")
    sources = payload.get("sources", ["google"])
    search_id = payload.get("search_id")

    logger.info(f"Scraping '{query}' from {sources}")

    results = {}

    # Import scraper manager (avoid circular imports)
    from src.scrapers.manager import scraper_manager

    for source in sources:
        try:
            # Scrape from source
            source_results = await scraper_manager.scrape(
                query=query,
                source=source
            )

            results[source] = {
                "status": "success",
                "count": len(source_results),
                "results": source_results
            }

            # Cache results
            await cache.set(
                f"scrape:{search_id}:{source}",
                source_results,
                ttl=3600
            )

        except Exception as e:
            logger.error(f"Failed to scrape {source}: {e}")
            results[source] = {
                "status": "failed",
                "error": str(e)
            }

    return results


async def analyze_handler(payload: Dict[str, Any]) -> Dict:
    """
    Handle analysis tasks

    Args:
        payload: Task data with text and analysis type

    Returns:
        Analysis results
    """
    text = payload.get("text")
    analysis_type = payload.get("type", "sentiment")
    search_id = payload.get("search_id")

    logger.info(f"Analyzing text (type={analysis_type})")

    # Import analyzer (avoid circular imports)
    from src.analyzers.ollama_client import ollama_client

    try:
        result = await ollama_client.analyze(
            text=text,
            analysis_type=analysis_type
        )

        # Cache analysis result
        await cache.set(
            f"analysis:{search_id}:{analysis_type}",
            result,
            ttl=7200
        )

        return {
            "status": "success",
            "type": analysis_type,
            "result": result
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


async def export_handler(payload: Dict[str, Any]) -> Dict:
    """
    Handle export tasks

    Args:
        payload: Task data with search_id and format

    Returns:
        Export file path
    """
    search_id = payload.get("search_id")
    export_format = payload.get("format", "csv")

    logger.info(f"Exporting search {search_id} as {export_format}")

    # Import export service (avoid circular imports)
    from src.services.export_service import export_service

    try:
        file_path = await export_service.export_search(
            search_id=search_id,
            format=export_format
        )

        return {
            "status": "success",
            "file_path": file_path,
            "format": export_format
        }

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


async def batch_handler(payload: Dict[str, Any]) -> List[Dict]:
    """
    Handle batch processing tasks

    Args:
        payload: Batch task configuration

    Returns:
        Results for all sub-tasks
    """
    sub_tasks = payload.get("tasks", [])
    results = []

    for sub_task in sub_tasks:
        task_type = sub_task.get("type")
        task_payload = sub_task.get("payload")

        # Get appropriate handler
        handler = task_queue.handlers.get(task_type)

        if handler:
            try:
                result = await handler(task_payload)
                results.append({
                    "task": task_type,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "task": task_type,
                    "status": "failed",
                    "error": str(e)
                })

    return results


def register_default_handlers():
    """Register default task handlers"""
    task_queue.register_handler("scrape", scrape_handler)
    task_queue.register_handler("analyze", analyze_handler)
    task_queue.register_handler("export", export_handler)
    task_queue.register_handler("batch", batch_handler)

    logger.info("Registered default task handlers")
```

### Step 3: Task Utilities (`src/core/task_utils.py`)

```python
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from src.core.queue import task_queue, TaskStatus, TaskPriority
from src.core.cache import cache


class TaskChain:
    """Chain multiple tasks with dependencies"""

    def __init__(self):
        self.tasks: List[Dict] = []

    def add(
        self,
        task_type: str,
        payload: Dict,
        priority: int = TaskPriority.NORMAL
    ) -> "TaskChain":
        """Add task to chain"""
        self.tasks.append({
            "type": task_type,
            "payload": payload,
            "priority": priority
        })
        return self

    async def execute(self) -> List[int]:
        """Execute tasks in sequence"""
        task_ids = []

        for task in self.tasks:
            task_id = await task_queue.enqueue(
                task_type=task["type"],
                payload=task["payload"],
                priority=task["priority"]
            )
            task_ids.append(task_id)

            # Wait for completion before next task
            await wait_for_task(task_id, timeout=300)

        return task_ids


class TaskGroup:
    """Group tasks for parallel execution"""

    def __init__(self):
        self.tasks: List[Dict] = []

    def add(
        self,
        task_type: str,
        payload: Dict,
        priority: int = TaskPriority.NORMAL
    ) -> "TaskGroup":
        """Add task to group"""
        self.tasks.append({
            "type": task_type,
            "payload": payload,
            "priority": priority
        })
        return self

    async def execute(self) -> List[int]:
        """Execute all tasks in parallel"""
        # Enqueue all tasks
        task_ids = []
        for task in self.tasks:
            task_id = await task_queue.enqueue(
                task_type=task["type"],
                payload=task["payload"],
                priority=task["priority"]
            )
            task_ids.append(task_id)

        return task_ids

    async def wait_all(
        self,
        task_ids: List[int],
        timeout: int = 300
    ) -> List[Dict]:
        """Wait for all tasks to complete"""
        results = await asyncio.gather(
            *[wait_for_task(tid, timeout) for tid in task_ids],
            return_exceptions=True
        )
        return results


async def wait_for_task(
    task_id: int,
    timeout: int = 300,
    poll_interval: float = 1.0
) -> Optional[Dict]:
    """
    Wait for task to complete

    Args:
        task_id: Task ID
        timeout: Maximum wait time in seconds
        poll_interval: Status check interval

    Returns:
        Task result or None if timeout
    """
    start_time = datetime.utcnow()
    deadline = start_time + timedelta(seconds=timeout)

    while datetime.utcnow() < deadline:
        status = await task_queue.get_task_status(task_id)

        if status:
            if status["status"] in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.DEAD
            ]:
                return status

        await asyncio.sleep(poll_interval)

    return None


async def schedule_recurring_task(
    task_type: str,
    payload: Dict,
    interval_seconds: int,
    max_runs: Optional[int] = None
) -> asyncio.Task:
    """
    Schedule recurring task execution

    Args:
        task_type: Type of task
        payload: Task data
        interval_seconds: Interval between executions
        max_runs: Maximum number of executions

    Returns:
        Background task
    """
    async def recurring_executor():
        runs = 0
        while max_runs is None or runs < max_runs:
            # Enqueue task
            await task_queue.enqueue(
                task_type=task_type,
                payload=payload
            )

            runs += 1
            await asyncio.sleep(interval_seconds)

    return asyncio.create_task(recurring_executor())


async def create_workflow(
    name: str,
    steps: List[Dict]
) -> int:
    """
    Create multi-step workflow

    Args:
        name: Workflow name
        steps: List of workflow steps

    Returns:
        Workflow ID
    """
    # Store workflow definition
    workflow_id = await cache.set(
        f"workflow:{name}",
        {
            "name": name,
            "steps": steps,
            "created_at": datetime.utcnow().isoformat()
        },
        ttl=86400
    )

    # Execute first step
    if steps:
        first_step = steps[0]
        await task_queue.enqueue(
            task_type="workflow_step",
            payload={
                "workflow_id": workflow_id,
                "step_index": 0,
                **first_step
            }
        )

    return workflow_id
```

---

## Testing Guide

### Unit Tests (`tests/test_queue.py`)

```python
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.core.queue import TaskQueueManager, TaskStatus, TaskPriority
from src.core.task_handlers import scrape_handler, register_default_handlers
from src.core.task_utils import TaskChain, TaskGroup, wait_for_task


@pytest.fixture
async def test_queue():
    """Create test queue instance"""
    queue = TaskQueueManager()

    # Register test handler
    async def test_handler(payload):
        await asyncio.sleep(0.1)  # Simulate work
        return {"result": f"processed_{payload.get('data')}"}

    queue.register_handler("test", test_handler)

    await queue.start()
    yield queue
    await queue.stop()


@pytest.mark.asyncio
async def test_enqueue_dequeue(test_queue):
    """Test basic enqueue and dequeue operations"""
    # Enqueue task
    task_id = await test_queue.enqueue(
        "test",
        {"data": "test_value"},
        priority=TaskPriority.HIGH
    )
    assert task_id is not None

    # Dequeue task
    task = await test_queue.dequeue()
    assert task is not None
    assert task.task_type == "test"
    assert task.payload["data"] == "test_value"
    assert task.status == TaskStatus.PROCESSING


@pytest.mark.asyncio
async def test_priority_ordering(test_queue):
    """Test priority-based task ordering"""
    # Stop workers to control dequeue
    await test_queue.stop()

    # Enqueue tasks with different priorities
    await test_queue.enqueue("test", {"n": 1}, priority=TaskPriority.LOW)
    await test_queue.enqueue("test", {"n": 2}, priority=TaskPriority.HIGH)
    await test_queue.enqueue("test", {"n": 3}, priority=TaskPriority.NORMAL)

    # Dequeue should return high priority first
    task1 = await test_queue.dequeue()
    assert task1.payload["n"] == 2  # High priority

    task2 = await test_queue.dequeue()
    assert task2.payload["n"] == 3  # Normal priority

    task3 = await test_queue.dequeue()
    assert task3.payload["n"] == 1  # Low priority


@pytest.mark.asyncio
async def test_task_processing(test_queue):
    """Test end-to-end task processing"""
    # Enqueue task
    task_id = await test_queue.enqueue(
        "test",
        {"data": "process_me"}
    )

    # Wait for processing
    await asyncio.sleep(0.5)

    # Check status
    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.COMPLETED
    assert status["result"]["result"] == "processed_process_me"


@pytest.mark.asyncio
async def test_task_failure_and_retry(test_queue):
    """Test task failure and retry mechanism"""
    # Register failing handler
    fail_count = 0

    async def failing_handler(payload):
        nonlocal fail_count
        fail_count += 1
        if fail_count < 3:
            raise Exception("Simulated failure")
        return {"success": True}

    test_queue.register_handler("fail_test", failing_handler)

    # Enqueue task
    task_id = await test_queue.enqueue("fail_test", {})

    # Wait for retries
    await asyncio.sleep(5)

    # Should succeed after retries
    status = await test_queue.get_task_status(task_id)
    assert fail_count == 3
    assert status["status"] == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_task_cancellation(test_queue):
    """Test task cancellation"""
    # Stop workers
    await test_queue.stop()

    # Enqueue task
    task_id = await test_queue.enqueue("test", {"data": "cancel_me"})

    # Cancel task
    cancelled = await test_queue.cancel_task(task_id)
    assert cancelled

    # Task should be cancelled
    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_batch_enqueue(test_queue):
    """Test batch task enqueuing"""
    tasks = [
        {"type": "test", "payload": {"n": i}}
        for i in range(5)
    ]

    task_ids = await test_queue.enqueue_batch(tasks)
    assert len(task_ids) == 5

    # All tasks should be enqueued
    for task_id in task_ids:
        status = await test_queue.get_task_status(task_id)
        assert status is not None


@pytest.mark.asyncio
async def test_queue_statistics(test_queue):
    """Test queue statistics"""
    # Enqueue some tasks
    for i in range(3):
        await test_queue.enqueue("test", {"n": i})

    # Wait for processing
    await asyncio.sleep(1)

    # Get statistics
    stats = await test_queue.get_queue_stats()

    assert stats["status_counts"][TaskStatus.COMPLETED] >= 3
    assert stats["type_counts"]["test"] >= 3
    assert stats["worker_count"] == test_queue.worker_count


@pytest.mark.asyncio
async def test_task_chain():
    """Test task chaining"""
    chain = TaskChain()
    chain.add("test", {"step": 1})
    chain.add("test", {"step": 2})
    chain.add("test", {"step": 3})

    # Mock task queue
    with patch("src.core.task_utils.task_queue") as mock_queue:
        mock_queue.enqueue = AsyncMock(side_effect=[1, 2, 3])

        with patch("src.core.task_utils.wait_for_task") as mock_wait:
            mock_wait.return_value = {"status": "completed"}

            task_ids = await chain.execute()
            assert task_ids == [1, 2, 3]
            assert mock_wait.call_count == 3


@pytest.mark.asyncio
async def test_task_group():
    """Test task grouping for parallel execution"""
    group = TaskGroup()
    group.add("test", {"task": 1})
    group.add("test", {"task": 2})
    group.add("test", {"task": 3})

    # Mock task queue
    with patch("src.core.task_utils.task_queue") as mock_queue:
        mock_queue.enqueue = AsyncMock(side_effect=[1, 2, 3])

        task_ids = await group.execute()
        assert task_ids == [1, 2, 3]
        assert mock_queue.enqueue.call_count == 3


@pytest.mark.asyncio
async def test_scheduled_task(test_queue):
    """Test scheduled task execution"""
    # Schedule task for future
    future_time = datetime.utcnow() + timedelta(seconds=2)

    task_id = await test_queue.enqueue(
        "test",
        {"scheduled": True},
        scheduled_at=future_time
    )

    # Should not be processed immediately
    await asyncio.sleep(0.5)
    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.PENDING

    # Should be processed after scheduled time
    await asyncio.sleep(2)
    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_requeue_failed_tasks(test_queue):
    """Test requeuing failed tasks"""
    # Create failed tasks manually
    from src.core.database import db_manager
    from src.core.models import TaskQueue as TaskQueueModel

    async with db_manager.get_session() as session:
        for i in range(3):
            task = TaskQueueModel(
                task_type="test",
                payload={"n": i},
                status=TaskStatus.FAILED,
                retry_count=1,
                max_retries=3
            )
            session.add(task)
        await session.commit()

    # Requeue failed tasks
    requeued = await test_queue.requeue_failed()
    assert requeued == 3

    # Check they're pending again
    stats = await test_queue.get_queue_stats()
    assert stats["status_counts"][TaskStatus.PENDING] >= 3
```

---

## Integration Points

### With Database Module
```python
from src.core.database import db_manager
from src.core.models import TaskQueue as TaskQueueModel

# Queue uses database for persistence
async with db_manager.get_session() as session:
    # Task operations
    pass
```

### With Cache Module
```python
from src.core.cache import cache

# Store task results in cache
await cache.set(f"task_result:{task_id}", result, ttl=86400)
```

### With Scraper Module
```python
from src.core.queue import task_queue

# Enqueue scraping task
await task_queue.enqueue(
    "scrape",
    {"query": "AI news", "sources": ["google", "bing"]}
)
```

### With API Module
```python
from fastapi import BackgroundTasks
from src.core.queue import task_queue

@app.post("/api/tasks")
async def create_task(task_type: str, payload: dict):
    task_id = await task_queue.enqueue(task_type, payload)
    return {"task_id": task_id}

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: int):
    return await task_queue.get_task_status(task_id)
```

---

## Common Issues & Solutions

### Issue 1: Task Processing Hangs
**Problem:** Worker gets stuck processing task
**Solution:** Add timeout to handlers
```python
async def handler_with_timeout(payload):
    try:
        return await asyncio.wait_for(
            actual_handler(payload),
            timeout=60
        )
    except asyncio.TimeoutError:
        raise Exception("Task timeout")
```

### Issue 2: Memory Leak with Long-Running Queue
**Problem:** Memory usage grows over time
**Solution:** Periodically clean completed tasks
```python
async def cleanup_old_tasks():
    # Delete completed tasks older than 7 days
    cutoff = datetime.utcnow() - timedelta(days=7)
    await db.execute(
        delete(TaskQueueModel).where(
            and_(
                TaskQueueModel.status == TaskStatus.COMPLETED,
                TaskQueueModel.completed_at < cutoff
            )
        )
    )
```

### Issue 3: Database Lock Contention
**Problem:** Multiple workers blocking each other
**Solution:** Use skip_locked in dequeue
```python
.with_for_update(skip_locked=True)  # Skip locked rows
```

### Issue 4: Task Result Too Large
**Problem:** Result doesn't fit in cache
**Solution:** Store in file system
```python
if len(json.dumps(result)) > 1_000_000:  # 1MB
    # Save to file
    file_path = f"data/results/{task_id}.json"
    with open(file_path, "w") as f:
        json.dump(result, f)
    result = {"file": file_path}
```

### Issue 5: Worker Crashes
**Problem:** Worker stops processing tasks
**Solution:** Auto-restart workers
```python
async def monitor_workers(self):
    while self.running:
        dead_workers = [w for w in self.workers if w.done()]
        for worker in dead_workers:
            self.workers.remove(worker)
            # Start replacement
            new_worker = asyncio.create_task(
                self.worker(len(self.workers))
            )
            self.workers.append(new_worker)
        await asyncio.sleep(10)
```

---

## Performance Optimization

### 1. Batch Database Operations
```python
async def dequeue_batch(self, count: int = 10):
    """Dequeue multiple tasks at once"""
    async with db_manager.get_session() as session:
        result = await session.execute(
            select(TaskQueueModel)
            .where(TaskQueueModel.status == TaskStatus.PENDING)
            .order_by(TaskQueueModel.priority)
            .limit(count)
            .with_for_update(skip_locked=True)
        )
        tasks = result.scalars().all()

        # Mark all as processing
        for task in tasks:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()

        await session.commit()
        return tasks
```

### 2. Priority Queue Optimization
```python
# Add index for faster priority queries
CREATE INDEX idx_queue_priority_status
ON task_queue(status, priority, scheduled_at)
WHERE status = 'pending';
```

### 3. Worker Pool Sizing
```python
# Dynamic worker scaling based on queue size
async def auto_scale_workers(self):
    stats = await self.get_queue_stats()
    pending = stats["status_counts"]["pending"]

    if pending > 100 and len(self.workers) < 10:
        # Add workers
        for _ in range(2):
            worker = asyncio.create_task(
                self.worker(len(self.workers))
            )
            self.workers.append(worker)
```

---

## Module Checklist

- [ ] Queue manager implemented
- [ ] Enqueue/dequeue working
- [ ] Priority ordering tested
- [ ] Task handlers registered
- [ ] Retry logic functional
- [ ] Worker pool running
- [ ] Statistics tracking
- [ ] Task chaining/grouping
- [ ] Scheduled tasks working
- [ ] Unit tests passing
- [ ] Integration documented

---

## Next Steps
After completing this module:
1. **Module 5: Scraper** - Creates scraping tasks
2. **Module 7: Analyzer** - Creates analysis tasks
3. **Module 10: Scheduler** - Schedule recurring tasks