"""
Comprehensive tests for Task Queue System
Tests TaskQueueManager, handlers, and task utilities
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.task_queue import (
    TaskQueueManager,
    task_queue,
    TaskStatus,
    TaskPriority,
    scrape_handler,
    analyze_handler,
    export_handler,
    batch_handler,
    register_default_handlers,
    TaskChain,
    TaskGroup,
    wait_for_task,
    schedule_recurring_task,
    create_workflow,
)
from src.database import db_manager, TaskQueue as TaskQueueModel
from src.cache import cache


@pytest.fixture
async def test_queue():
    """Create isolated test queue instance with test handler"""
    queue = TaskQueueManager()

    # Register test handler
    async def test_handler(payload):
        """Simple test handler that simulates work"""
        await asyncio.sleep(0.1)  # Simulate work
        data = payload.get("data", "default")
        return {"result": f"processed_{data}", "success": True}

    queue.register_handler("test", test_handler)

    # Start queue
    await queue.start()

    yield queue

    # Cleanup
    await queue.stop()
    await queue.clear_queue()


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
    assert isinstance(task_id, int)

    # Stop workers to control dequeue manually
    await test_queue.stop()

    # Dequeue task
    task = await test_queue.dequeue()
    assert task is not None
    assert task.task_type == "test"
    assert task.payload["data"] == "test_value"
    assert task.status == TaskStatus.PROCESSING
    assert task.priority == TaskPriority.HIGH


@pytest.mark.asyncio
async def test_priority_ordering(test_queue):
    """Test priority-based task ordering"""
    # Stop workers to control dequeue
    await test_queue.stop()

    # Enqueue tasks with different priorities
    await test_queue.enqueue("test", {"n": 1}, priority=TaskPriority.LOW)
    await test_queue.enqueue("test", {"n": 2}, priority=TaskPriority.HIGH)
    await test_queue.enqueue("test", {"n": 3}, priority=TaskPriority.NORMAL)
    await test_queue.enqueue("test", {"n": 4}, priority=TaskPriority.CRITICAL)

    # Dequeue should return highest priority first
    task1 = await test_queue.dequeue()
    assert task1.payload["n"] == 4  # CRITICAL (1)

    task2 = await test_queue.dequeue()
    assert task2.payload["n"] == 2  # HIGH (3)

    task3 = await test_queue.dequeue()
    assert task3.payload["n"] == 3  # NORMAL (5)

    task4 = await test_queue.dequeue()
    assert task4.payload["n"] == 1  # LOW (7)


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
    assert status is not None
    assert status["status"] == TaskStatus.COMPLETED
    assert status["result"]["result"] == "processed_process_me"
    assert status["result"]["success"] is True


@pytest.mark.asyncio
async def test_task_failure_and_retry(test_queue):
    """Test task failure and retry mechanism with exponential backoff"""
    fail_count = 0

    # Register failing handler
    async def failing_handler(payload):
        nonlocal fail_count
        fail_count += 1
        if fail_count < 3:
            raise Exception("Simulated failure")
        return {"success": True, "attempts": fail_count}

    test_queue.register_handler("fail_test", failing_handler)

    # Enqueue task
    task_id = await test_queue.enqueue("fail_test", {})

    # Wait for retries (exponential backoff: 2s, 4s)
    await asyncio.sleep(8)

    # Should succeed after retries
    status = await test_queue.get_task_status(task_id)
    assert fail_count == 3
    assert status["status"] == TaskStatus.COMPLETED
    assert status["result"]["success"] is True


@pytest.mark.asyncio
async def test_task_cancellation(test_queue):
    """Test task cancellation"""
    # Stop workers so task stays pending
    await test_queue.stop()

    # Enqueue task
    task_id = await test_queue.enqueue("test", {"data": "cancel_me"})

    # Verify task is pending
    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.PENDING

    # Cancel task
    cancelled = await test_queue.cancel_task(task_id)
    assert cancelled is True

    # Task should be cancelled
    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.CANCELLED

    # Try to cancel again (should fail)
    cancelled_again = await test_queue.cancel_task(task_id)
    assert cancelled_again is False


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
    """Test queue statistics tracking"""
    # Enqueue some tasks
    for i in range(3):
        await test_queue.enqueue("test", {"n": i})

    # Wait for processing
    await asyncio.sleep(1)

    # Get statistics
    stats = await test_queue.get_queue_stats()

    assert "status_counts" in stats
    assert "type_counts" in stats
    assert stats["status_counts"][TaskStatus.COMPLETED] >= 3
    assert stats["type_counts"]["test"] >= 3
    assert stats["worker_count"] == test_queue.worker_count
    assert "tasks_processed" in stats


@pytest.mark.asyncio
async def test_task_chain():
    """Test task chaining for sequential execution"""
    # Create chain
    chain = TaskChain()
    chain.add("test", {"step": 1})
    chain.add("test", {"step": 2})
    chain.add("test", {"step": 3})

    # Mock task queue
    with patch("src.task_queue.utils.task_queue") as mock_queue:
        mock_queue.enqueue = AsyncMock(side_effect=[1, 2, 3])

        with patch("src.task_queue.utils.wait_for_task") as mock_wait:
            mock_wait.return_value = {"status": TaskStatus.COMPLETED}

            task_ids = await chain.execute()
            assert task_ids == [1, 2, 3]
            assert mock_wait.call_count == 3  # Wait called for each task


@pytest.mark.asyncio
async def test_task_group():
    """Test task grouping for parallel execution"""
    # Create group
    group = TaskGroup()
    group.add("test", {"task": 1})
    group.add("test", {"task": 2})
    group.add("test", {"task": 3})

    # Mock task queue
    with patch("src.task_queue.utils.task_queue") as mock_queue:
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
    # Stop workers
    await test_queue.stop()

    # Create failed tasks manually
    async with db_manager.get_session() as session:
        for i in range(3):
            task = TaskQueueModel(
                task_type="test",
                payload={"n": i},
                status=TaskStatus.FAILED,
                retry_count=1,
                max_retries=3,
                scheduled_at=datetime.utcnow()
            )
            session.add(task)
        await session.commit()

    # Requeue failed tasks
    requeued = await test_queue.requeue_failed()
    assert requeued == 3

    # Check they're pending again
    stats = await test_queue.get_queue_stats()
    assert stats["status_counts"][TaskStatus.PENDING] >= 3


@pytest.mark.asyncio
async def test_worker_pool(test_queue):
    """Test worker pool lifecycle management"""
    # Workers should be running
    assert test_queue.running is True
    assert len(test_queue.workers) == test_queue.worker_count
    assert all(not w.done() for w in test_queue.workers)

    # Stop workers
    await test_queue.stop()
    assert test_queue.running is False
    assert len(test_queue.workers) == 0

    # Restart workers
    await test_queue.start()
    assert test_queue.running is True
    assert len(test_queue.workers) == test_queue.worker_count


@pytest.mark.asyncio
async def test_handler_registration(test_queue):
    """Test handler registry functionality"""
    # Register custom handler
    async def custom_handler(payload):
        return {"custom": True}

    test_queue.register_handler("custom", custom_handler)

    # Verify handler is registered
    assert "custom" in test_queue.handlers
    assert test_queue.handlers["custom"] == custom_handler

    # Test handler execution
    task_id = await test_queue.enqueue("custom", {})
    await asyncio.sleep(0.3)

    status = await test_queue.get_task_status(task_id)
    assert status["status"] == TaskStatus.COMPLETED
    assert status["result"]["custom"] is True


@pytest.mark.asyncio
async def test_skip_locked():
    """Test concurrent dequeue with skip_locked"""
    # Create two queue instances
    queue1 = TaskQueueManager()
    queue2 = TaskQueueManager()

    # Stop workers
    await queue1.stop()
    await queue2.stop()

    # Enqueue single task
    task_id = await queue1.enqueue("test", {"concurrent": True})

    # Dequeue from first queue (locks the row)
    async def dequeue_and_hold():
        async with db_manager.get_session() as session:
            # This simulates a long-running transaction
            task = await queue1.dequeue()
            await asyncio.sleep(0.5)
            return task

    # Start first dequeue
    task1_future = asyncio.create_task(dequeue_and_hold())
    await asyncio.sleep(0.1)  # Give it time to lock

    # Second dequeue should skip locked row and return None
    task2 = await queue2.dequeue()
    assert task2 is None  # Locked by queue1

    # First dequeue should succeed
    task1 = await task1_future
    assert task1 is not None


@pytest.mark.asyncio
async def test_task_result_caching(test_queue):
    """Test result storage in cache"""
    # Enqueue task
    task_id = await test_queue.enqueue(
        "test",
        {"data": "cache_test"}
    )

    # Wait for completion
    await asyncio.sleep(0.3)

    # Result should be in cache
    cached_result = await cache.get(f"task_result:{task_id}")
    assert cached_result is not None
    assert cached_result["result"] == "processed_cache_test"

    # Should also be in task status
    status = await test_queue.get_task_status(task_id)
    assert status["result"] == cached_result


@pytest.mark.asyncio
async def test_placeholder_handlers():
    """Test placeholder handlers return mock data"""
    # Test scrape handler
    scrape_result = await scrape_handler({
        "query": "test query",
        "sources": ["google", "bing"]
    })
    assert scrape_result["status"] == "success"
    assert scrape_result["mock"] is True
    assert "google" in scrape_result["results"]
    assert "bing" in scrape_result["results"]

    # Test analyze handler
    analyze_result = await analyze_handler({
        "text": "test text",
        "type": "sentiment"
    })
    assert analyze_result["status"] == "success"
    assert analyze_result["mock"] is True
    assert "sentiment" in analyze_result["result"]

    # Test export handler
    export_result = await export_handler({
        "search_id": 123,
        "format": "csv"
    })
    assert export_result["status"] == "success"
    assert export_result["mock"] is True
    assert "csv" in export_result["file_path"]


@pytest.mark.asyncio
async def test_batch_handler(test_queue):
    """Test batch handler processes multiple tasks"""
    # Register handlers for batch
    register_default_handlers()

    # Create batch payload
    batch_payload = {
        "tasks": [
            {"type": "scrape", "payload": {"query": "AI"}},
            {"type": "analyze", "payload": {"text": "test", "type": "sentiment"}},
            {"type": "export", "payload": {"search_id": 1, "format": "json"}}
        ]
    }

    # Execute batch
    results = await batch_handler(batch_payload)

    assert len(results) == 3
    assert all(r["status"] == "success" for r in results)
    assert results[0]["task"] == "scrape"
    assert results[1]["task"] == "analyze"
    assert results[2]["task"] == "export"


@pytest.mark.asyncio
async def test_recurring_task():
    """Test recurring task scheduling"""
    execution_count = 0

    async def count_handler(payload):
        nonlocal execution_count
        execution_count += 1
        return {"count": execution_count}

    # Create temporary queue
    temp_queue = TaskQueueManager()
    temp_queue.register_handler("count", count_handler)
    await temp_queue.start()

    # Schedule recurring task (every 0.5 seconds, max 3 runs)
    recurring_task = await schedule_recurring_task(
        "count",
        {},
        interval_seconds=0.5,
        max_runs=3
    )

    # Wait for completion
    await asyncio.sleep(2)

    # Should have run 3 times
    assert execution_count >= 3

    # Cleanup
    recurring_task.cancel()
    await temp_queue.stop()


@pytest.mark.asyncio
async def test_workflow_creation():
    """Test workflow creation and management"""
    workflow = await create_workflow(
        "test_workflow",
        [
            {"type": "scrape", "payload": {"query": "test"}},
            {"type": "analyze", "payload": {"type": "sentiment"}},
            {"type": "export", "payload": {"format": "csv"}}
        ]
    )

    assert workflow["name"] == "test_workflow"
    assert workflow["steps_count"] == 3
    assert workflow["status"] == "running"
    assert "workflow_key" in workflow

    # Verify workflow stored in cache
    from src.task_queue.utils import get_workflow_status
    workflow_status = await get_workflow_status(workflow["workflow_key"])
    assert workflow_status is not None
    assert workflow_status["name"] == "test_workflow"


@pytest.mark.asyncio
async def test_cleanup_old_tasks(test_queue):
    """Test cleanup of old completed tasks"""
    # Stop workers
    await test_queue.stop()

    # Create old completed tasks
    async with db_manager.get_session() as session:
        old_date = datetime.utcnow() - timedelta(days=10)
        for i in range(3):
            task = TaskQueueModel(
                task_type="test",
                payload={"n": i},
                status=TaskStatus.COMPLETED,
                scheduled_at=old_date,
                completed_at=old_date
            )
            session.add(task)
        await session.commit()

    # Cleanup tasks older than 7 days
    deleted = await test_queue.cleanup_old_tasks(days=7)
    assert deleted == 3


@pytest.mark.asyncio
async def test_task_with_no_handler(test_queue):
    """Test task execution with missing handler"""
    # Enqueue task with unregistered type
    task_id = await test_queue.enqueue("nonexistent", {})

    # Wait for processing
    await asyncio.sleep(0.3)

    # Should fail with handler error
    status = await test_queue.get_task_status(task_id)
    assert status["status"] in [TaskStatus.FAILED, TaskStatus.DEAD]
    assert "No handler" in status["error"]


@pytest.mark.asyncio
async def test_wait_for_task_timeout():
    """Test wait_for_task timeout behavior"""
    # Create queue and enqueue task
    temp_queue = TaskQueueManager()
    await temp_queue.stop()  # Don't process

    task_id = await temp_queue.enqueue("test", {})

    # Wait should timeout
    result = await wait_for_task(task_id, timeout=1, poll_interval=0.2)
    assert result is None  # Timeout


@pytest.mark.asyncio
async def test_global_task_queue_instance():
    """Test global task_queue instance"""
    from src.task_queue import task_queue

    assert isinstance(task_queue, TaskQueueManager)
    assert task_queue.handlers is not None
    assert task_queue.workers is not None
