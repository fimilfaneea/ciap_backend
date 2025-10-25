"""
Task Queue Utilities for CIAP
Provides utilities for task chaining, grouping, and workflow management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging

from .manager import TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class TaskChain:
    """
    Chain multiple tasks for sequential execution with dependency handling

    Tasks are executed one after another, waiting for each to complete
    before starting the next.

    Example:
        chain = TaskChain()
        chain.add("scrape", {"query": "AI"})
        chain.add("analyze", {"text": "...", "type": "sentiment"})
        chain.add("export", {"search_id": 123, "format": "csv"})

        task_ids = await chain.execute()
    """

    def __init__(self):
        """Initialize empty task chain"""
        self.tasks: List[Dict] = []

    def add(
        self,
        task_type: str,
        payload: Dict,
        priority: int = TaskPriority.NORMAL
    ) -> "TaskChain":
        """
        Add task to chain

        Args:
            task_type: Type of task
            payload: Task data
            priority: Task priority

        Returns:
            Self for method chaining
        """
        self.tasks.append({
            "type": task_type,
            "payload": payload,
            "priority": priority
        })
        return self

    async def execute(self, timeout_per_task: int = 300) -> List[int]:
        """
        Execute tasks in sequence

        Each task waits for the previous one to complete before starting.

        Args:
            timeout_per_task: Timeout in seconds for each task (default: 300)

        Returns:
            List of task IDs

        Raises:
            TimeoutError: If any task times out
            Exception: If any task fails
        """
        from .manager import task_queue

        task_ids = []

        logger.info(f"Executing task chain with {len(self.tasks)} tasks")

        for idx, task in enumerate(self.tasks):
            logger.debug(
                f"Chain task {idx + 1}/{len(self.tasks)}: {task['type']}"
            )

            # Enqueue task
            task_id = await task_queue.enqueue(
                task_type=task["type"],
                payload=task["payload"],
                priority=task["priority"]
            )
            task_ids.append(task_id)

            # Wait for completion before next task
            result = await wait_for_task(task_id, timeout=timeout_per_task)

            if result is None:
                raise TimeoutError(
                    f"Task {task_id} timed out after {timeout_per_task}s"
                )

            if result["status"] in [TaskStatus.FAILED, TaskStatus.DEAD]:
                raise Exception(
                    f"Task {task_id} failed: {result.get('error', 'Unknown error')}"
                )

            logger.debug(f"Chain task {task_id} completed successfully")

        logger.info(f"Task chain completed: {len(task_ids)} tasks executed")
        return task_ids


class TaskGroup:
    """
    Group tasks for parallel execution

    All tasks are enqueued simultaneously and can be executed in parallel
    by multiple workers.

    Example:
        group = TaskGroup()
        group.add("scrape", {"query": "AI", "sources": ["google"]})
        group.add("scrape", {"query": "ML", "sources": ["bing"]})
        group.add("scrape", {"query": "Data", "sources": ["google", "bing"]})

        task_ids = await group.execute()
        results = await group.wait_all(task_ids, timeout=300)
    """

    def __init__(self):
        """Initialize empty task group"""
        self.tasks: List[Dict] = []

    def add(
        self,
        task_type: str,
        payload: Dict,
        priority: int = TaskPriority.NORMAL
    ) -> "TaskGroup":
        """
        Add task to group

        Args:
            task_type: Type of task
            payload: Task data
            priority: Task priority

        Returns:
            Self for method chaining
        """
        self.tasks.append({
            "type": task_type,
            "payload": payload,
            "priority": priority
        })
        return self

    async def execute(self) -> List[int]:
        """
        Execute all tasks in parallel

        Enqueues all tasks simultaneously. They will be picked up by
        available workers.

        Returns:
            List of task IDs
        """
        from .manager import task_queue

        logger.info(f"Executing task group with {len(self.tasks)} tasks")

        # Enqueue all tasks at once
        task_ids = []
        for task in self.tasks:
            task_id = await task_queue.enqueue(
                task_type=task["type"],
                payload=task["payload"],
                priority=task["priority"]
            )
            task_ids.append(task_id)

        logger.info(f"Task group enqueued: {len(task_ids)} tasks")
        return task_ids

    async def wait_all(
        self,
        task_ids: List[int],
        timeout: int = 300
    ) -> List[Dict]:
        """
        Wait for all tasks to complete

        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum wait time per task in seconds

        Returns:
            List of task results
        """
        logger.info(f"Waiting for {len(task_ids)} tasks to complete")

        # Wait for all tasks in parallel
        results = await asyncio.gather(
            *[wait_for_task(tid, timeout) for tid in task_ids],
            return_exceptions=True
        )

        # Check for exceptions
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {task_ids[idx]} failed with exception: {result}")

        logger.info(f"Task group completed: {len(results)} results")
        return results


async def wait_for_task(
    task_id: int,
    timeout: int = 300,
    poll_interval: float = 1.0
) -> Optional[Dict]:
    """
    Wait for task to complete

    Polls task status until it reaches a terminal state or timeout.

    Args:
        task_id: Task ID to wait for
        timeout: Maximum wait time in seconds (default: 300)
        poll_interval: Status check interval in seconds (default: 1.0)

    Returns:
        Task status dictionary or None if timeout

    Example:
        task_id = await queue.enqueue("scrape", {"query": "AI"})
        result = await wait_for_task(task_id, timeout=60)
        if result and result["status"] == "completed":
            print(f"Task result: {result['result']}")
    """
    from .manager import task_queue

    start_time = datetime.utcnow()
    deadline = start_time + timedelta(seconds=timeout)

    logger.debug(f"Waiting for task {task_id} (timeout={timeout}s)")

    while datetime.utcnow() < deadline:
        status = await task_queue.get_task_status(task_id)

        if status:
            # Check if task reached terminal state
            if status["status"] in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.DEAD,
                TaskStatus.CANCELLED
            ]:
                logger.debug(
                    f"Task {task_id} finished with status: {status['status']}"
                )
                return status

        # Wait before next poll
        await asyncio.sleep(poll_interval)

    logger.warning(f"Task {task_id} timed out after {timeout}s")
    return None


async def schedule_recurring_task(
    task_type: str,
    payload: Dict,
    interval_seconds: int,
    max_runs: Optional[int] = None,
    priority: int = TaskPriority.BACKGROUND
) -> asyncio.Task:
    """
    Schedule recurring task execution

    Creates a background task that enqueues the specified task at regular intervals.

    Args:
        task_type: Type of task
        payload: Task data
        interval_seconds: Interval between executions
        max_runs: Maximum number of executions (None = infinite)
        priority: Task priority

    Returns:
        Background asyncio task (can be cancelled)

    Example:
        # Run scraping every hour
        recurring = await schedule_recurring_task(
            "scrape",
            {"query": "AI news", "sources": ["google"]},
            interval_seconds=3600,
            max_runs=24  # Run 24 times (24 hours)
        )

        # Later, to stop:
        recurring.cancel()
    """
    from .manager import task_queue

    async def recurring_executor():
        runs = 0
        logger.info(
            f"Starting recurring task: {task_type} every {interval_seconds}s "
            f"(max_runs={max_runs})"
        )

        while max_runs is None or runs < max_runs:
            # Enqueue task
            task_id = await task_queue.enqueue(
                task_type=task_type,
                payload=payload,
                priority=priority
            )

            runs += 1
            logger.debug(
                f"Recurring task {task_type} enqueued (run {runs}): task_id={task_id}"
            )

            # Wait for next execution
            await asyncio.sleep(interval_seconds)

        logger.info(f"Recurring task {task_type} completed after {runs} runs")

    return asyncio.create_task(recurring_executor())


async def create_workflow(
    name: str,
    steps: List[Dict]
) -> Dict:
    """
    Create multi-step workflow

    A workflow is a sequence of tasks with state management.
    Each step can depend on the results of previous steps.

    Args:
        name: Workflow name
        steps: List of workflow steps, each with:
            - type: str - Task type
            - payload: Dict - Task data
            - priority: int - Task priority (optional)

    Returns:
        Workflow metadata with first task ID

    Example:
        workflow = await create_workflow(
            "competitor_analysis",
            [
                {"type": "scrape", "payload": {"query": "competitor"}},
                {"type": "analyze", "payload": {"type": "competitor"}},
                {"type": "export", "payload": {"format": "csv"}}
            ]
        )
    """
    from ..cache import cache

    logger.info(f"Creating workflow '{name}' with {len(steps)} steps")

    # Store workflow definition in cache
    workflow_data = {
        "name": name,
        "steps": steps,
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending"
    }

    workflow_key = f"workflow:{name}:{datetime.utcnow().timestamp()}"
    await cache.set(
        workflow_key,
        workflow_data,
        ttl=86400  # Keep for 24 hours
    )

    # Execute first step
    first_task_id = None
    if steps:
        from .manager import task_queue

        first_step = steps[0]
        first_task_id = await task_queue.enqueue(
            task_type=first_step["type"],
            payload={
                **first_step.get("payload", {}),
                "workflow_key": workflow_key,
                "workflow_step": 0
            },
            priority=first_step.get("priority", TaskPriority.NORMAL)
        )

        logger.info(
            f"Workflow '{name}' started with task {first_task_id}"
        )

    return {
        "workflow_key": workflow_key,
        "name": name,
        "steps_count": len(steps),
        "first_task_id": first_task_id,
        "status": "running"
    }


async def get_workflow_status(workflow_key: str) -> Optional[Dict]:
    """
    Get workflow status

    Args:
        workflow_key: Workflow key from create_workflow()

    Returns:
        Workflow status or None if not found
    """
    from ..cache import cache

    workflow_data = await cache.get(workflow_key)
    return workflow_data
